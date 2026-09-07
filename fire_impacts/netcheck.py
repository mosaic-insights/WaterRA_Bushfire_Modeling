"""
Diagnose TLS and connectivity problems against the remote data services.

Corporate networks routinely terminate outbound TLS and re-sign it with
an internal certificate authority.  Browsers accept this because the
internal root is installed in the operating system's certificate store;
Python does not, because it ships its own bundle and never looks at the
OS store.  The result is a wall of
``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate``
from whichever download happens to run first.

Diagnosing that from a traceback is hard because this package reaches
the network through three unrelated trust stores:

* ``requests`` (owslib's WCS client, pystac_client, direct calls) -
  trusts certifi's bundle, configurable via ``REQUESTS_CA_BUNDLE``.
* ``urllib`` (pystac's ``StacIO``) - trusts Python's default SSL
  context, configurable via ``SSL_CERT_FILE`` *only*.
* GDAL/curl (every ``/vsicurl`` raster read) - trusts neither of the
  above.  On Linux and macOS conda builds it uses OpenSSL and a PEM
  bundle, configurable via ``GDAL_HTTP_CAINFO``; on Windows conda
  builds libcurl is Schannel-backed and already reads the Windows
  certificate store, so it usually keeps working while Python fails.

``run_checks`` probes one endpoint per host in each stack and reports
which stacks are broken, so the advice given matches the failure.
"""

from dataclasses import dataclass
from enum import Enum


class Stack(str, Enum):
    """Which certificate trust store a probe exercises."""

    REQUESTS = "requests"
    URLLIB = "urllib"
    GDAL = "gdal"


class Outcome(str, Enum):
    """What happened to a probe."""

    OK = "ok"
    TLS = "tls"
    CONNECT = "connect"
    TIMEOUT = "timeout"
    HTTP = "http"
    OTHER = "other"


class HttpStatusError(Exception):
    """A probe reached the service, but it answered with an error status."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# Substrings that identify curl's certificate complaints.  GDAL surfaces
# these as free text inside RasterioIOError, so there is nothing to match
# on but the message.  Both TLS backends are covered.
_GDAL_TLS_MARKERS = (
    "ssl certificate problem",
    "certificate verify failed",
    "unable to get local issuer",
    "self signed certificate",
    "self-signed certificate",
    "ssl peer certificate",
    "cert_e_untrustedroot",
    "sec_e_untrusted_root",
    "schannel",
)

_GDAL_CONNECT_MARKERS = (
    "could not resolve host",
    "couldn't connect",
    "could not connect",
    "connection refused",
    "failed to connect",
)

_GDAL_TIMEOUT_MARKERS = (
    "timed out",
    "timeout was reached",
)


def _classify_message(message):
    """Classify a curl/GDAL error from its text, or None if unrecognised."""
    lowered = message.lower()
    if any(marker in lowered for marker in _GDAL_TLS_MARKERS):
        return Outcome.TLS
    if any(marker in lowered for marker in _GDAL_CONNECT_MARKERS):
        return Outcome.CONNECT
    if any(marker in lowered for marker in _GDAL_TIMEOUT_MARKERS):
        return Outcome.TIMEOUT
    return None


def classify(exc):
    """
    Work out what kind of failure an exception represents.

    Parameters:
    - exc: the exception raised by a probe.

    Returns:
    - (Outcome, detail) tuple.  detail is a human-readable string.
    """
    import ssl
    import socket
    import urllib.error

    import requests

    detail = str(exc)

    if isinstance(exc, HttpStatusError):
        return Outcome.HTTP, detail

    # SSLError subclasses ConnectionError in requests, so it has to be
    # tested first or every certificate problem reads as an unreachable
    # host.
    if isinstance(exc, (requests.exceptions.SSLError, ssl.SSLError)):
        return Outcome.TLS, detail

    if isinstance(exc, requests.Timeout):
        return Outcome.TIMEOUT, detail

    if isinstance(exc, requests.ConnectionError):
        return Outcome.CONNECT, detail

    if isinstance(exc, urllib.error.URLError):
        return classify(exc.reason) if isinstance(
            exc.reason, BaseException
        ) else (Outcome.CONNECT, detail)

    if isinstance(exc, socket.timeout):
        return Outcome.TIMEOUT, detail

    # Only genuine socket errors default to CONNECT.  Matching on OSError
    # at large would sweep up RasterioIOError, which subclasses it and
    # carries perfectly ordinary problems like an unreadable file format.
    if isinstance(exc, (socket.gaierror, ConnectionError)):
        return Outcome.CONNECT, detail

    from_message = _classify_message(detail)
    if from_message is not None:
        return from_message, detail

    return Outcome.OTHER, detail


# ---------------------------------------------------------------------------
# Naming the certificate authority that a proxy is signing with
# ---------------------------------------------------------------------------

def issuer_label(info):
    """
    Human-readable name of the authority that signed a certificate.

    Parameters:
    - info: certificate dictionary as returned by
      ``SSLContext.get_ca_certs()``.

    Returns:
    - The issuer's common name, falling back to its organisation name,
      or None if neither is present.
    """
    fields = {}
    for rdn in info.get("issuer", ()):
        for key, value in rdn:
            fields.setdefault(key, value)
    return fields.get("commonName") or fields.get("organizationName")


def decode_chain(der_certs):
    """
    Decode DER certificates into dictionaries, skipping unreadable ones.

    There is no public API for decoding a certificate in isolation, so
    each one is written out as PEM and loaded into a throwaway SSL
    context, which will describe it.  Certificates that fail to decode
    are dropped rather than raised: this runs while diagnosing a
    certificate problem, and replacing it with a decoding error would
    hide the thing being diagnosed.

    Parameters:
    - der_certs: iterable of DER-encoded certificates (as returned by
      ``SSLSocket.get_unverified_chain()``).

    Returns:
    - List of certificate dictionaries, in the order given.
    """
    import os
    import ssl
    import tempfile

    decoded = []
    for der in der_certs:
        path = None
        try:
            pem = ssl.DER_cert_to_PEM_cert(der)
            fd, path = tempfile.mkstemp(suffix=".pem")
            with os.fdopen(fd, "w") as handle:
                handle.write(pem)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(cafile=path)
            decoded.extend(context.get_ca_certs())
        except (ssl.SSLError, ValueError, OSError):
            continue
        finally:
            if path is not None and os.path.exists(path):
                os.unlink(path)
    return decoded


def top_issuer(der_certs):
    """
    Name the authority that signed the highest certificate presented.

    On an intercepted connection this is the internal root the user
    needs to obtain from their IT department.

    Parameters:
    - der_certs: DER-encoded certificate chain, leaf first.

    Returns:
    - Issuer name string, or None if the chain is empty or unreadable.
    """
    decoded = decode_chain(der_certs)
    if not decoded:
        return None
    return issuer_label(decoded[-1])


# ---------------------------------------------------------------------------
# Probes and their results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    """One endpoint, reached through one trust store."""

    host: str
    stack: Stack
    url: str
    description: str
    run: object = None


@dataclass
class ProbeResult:
    """What a probe found."""

    probe: Probe
    outcome: Outcome
    detail: str = ""
    issuer: str = None

    @property
    def ok(self):
        return self.outcome is Outcome.OK


# ---------------------------------------------------------------------------
# Turning results into advice
# ---------------------------------------------------------------------------

_TRUSTSTORE_ADVICE = """\
Fix (recommended): make Python use the operating system's certificate
store, where your organisation's root is already installed.

    conda install -c conda-forge truststore     # or: pip install truststore

then, once at the top of your notebook or script, before any download:

    import truststore
    truststore.inject_into_ssl()

That covers every Python library this package uses, including the ones
that ignore REQUESTS_CA_BUNDLE."""

# Raw strings: the Windows paths below must survive verbatim, and '\t'
# in a plain literal would be parsed as a tab.  A raw string cannot end
# a line with a backslash, so the leading newline is stripped instead.
_ENV_VAR_ADVICE = r"""
Fix (alternative): ask IT for the root certificate as a PEM file and
point Python at it.  Both variables are needed - requests reads the
first, everything built on urllib reads only the second:

    REQUESTS_CA_BUNDLE=C:\path\to\corporate-root.pem
    SSL_CERT_FILE=C:\path\to\corporate-root.pem""".lstrip("\n")

_GDAL_ADVICE = r"""
GDAL reads rasters over its own HTTP stack and consults neither of the
settings above - truststore has no effect on it.  Point it at the same
PEM file:

    GDAL_HTTP_CAINFO=C:\path\to\corporate-root.pem
    CURL_CA_BUNDLE=C:\path\to\corporate-root.pem""".lstrip("\n")

_GDAL_ALREADY_FINE = """\
The GDAL raster reads succeeded, so leave GDAL's HTTP settings alone.
GDAL is already trusting your organisation's root through the operating
system, and pointing it at a partial bundle would break what currently
works."""


def advice(results):
    """
    Build the fix recipe implied by a set of probe results.

    The recipe is keyed to which trust stores failed, so a user is never
    sent to configure GDAL when GDAL is working, nor told that
    truststore will fix a GDAL problem, which it cannot.

    Parameters:
    - results: iterable of ProbeResult.

    Returns:
    - List of advice paragraphs; empty if nothing failed.
    """
    results = list(results)
    tls = [r for r in results if r.outcome is Outcome.TLS]
    unreachable = [
        r for r in results
        if r.outcome in (Outcome.CONNECT, Outcome.TIMEOUT)
    ]
    other = [
        r for r in results
        if r.outcome in (Outcome.HTTP, Outcome.OTHER)
    ]

    paragraphs = []

    if tls:
        stacks = {r.probe.stack for r in tls}
        hosts = _unique(r.probe.host for r in tls)
        paragraphs.append(
            "TLS interception: the certificates presented by "
            + ", ".join(hosts)
            + " were signed by an authority this environment does not "
            "trust.  This is normal on a corporate network."
        )

        issuers = _unique(r.issuer for r in tls if r.issuer)
        if issuers:
            # A real interception signs everything with one root; several
            # distinct names usually mean a partial bundle is in use.
            if len(issuers) == 1:
                ask = (
                    "Ask IT for that root certificate as a PEM file - "
                    "naming it exactly makes the request unambiguous."
                )
            else:
                ask = (
                    "Ask IT for those root certificates as a PEM file - "
                    "they can be concatenated into one - and naming them "
                    "exactly makes the request unambiguous."
                )
            paragraphs.append("Signed by: " + ", ".join(issuers) + ".  " + ask)

        python_broken = bool(stacks & {Stack.REQUESTS, Stack.URLLIB})
        gdal_broken = Stack.GDAL in stacks

        if python_broken:
            paragraphs.append(_TRUSTSTORE_ADVICE)
            paragraphs.append(_ENV_VAR_ADVICE)
        if gdal_broken:
            paragraphs.append(_GDAL_ADVICE)
        elif python_broken and _attempted(results, Stack.GDAL):
            paragraphs.append(_GDAL_ALREADY_FINE)

    if unreachable:
        hosts = _unique(r.probe.host for r in unreachable)
        paragraphs.append(
            "Unreachable: "
            + ", ".join(hosts)
            + ".  Nothing answered at all, so this is a firewall or "
            "proxy blocking the host rather than a trust problem.  Ask "
            "for these hosts to be allowed, and check HTTPS_PROXY if "
            "your network requires a proxy."
        )

    if other:
        for result in other:
            paragraphs.append(
                f"{result.probe.host}: reached, but the request failed "
                f"({result.detail}).  This is unlikely to be a local "
                f"configuration problem; the service may be down."
            )

    return paragraphs


def _unique(values):
    """Preserve order while dropping duplicates."""
    seen = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _attempted(results, stack):
    """Whether any probe for a stack actually ran."""
    return any(r.probe.stack is stack for r in results)


# ---------------------------------------------------------------------------
# The probes themselves
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 15


def _probe_requests(url, timeout):
    """Reach a service through requests' trust store (certifi)."""
    import requests

    response = requests.head(url, timeout=timeout, allow_redirects=True)
    # Any answer at all means TLS succeeded, which is what is being
    # measured.  A 404 or 403 from a HEAD on a service endpoint is
    # normal and says nothing about trust.  A 5xx is worth reporting.
    if response.status_code >= 500:
        raise HttpStatusError(response.status_code)


def _probe_urllib(url, timeout):
    """Reach a service through Python's default SSL context, as pystac does."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read(1)
    except urllib.error.HTTPError as exc:
        if exc.code >= 500:
            raise HttpStatusError(exc.code) from exc
        # Reached the service; its answer is not our concern here.


def _probe_gdal(url, timeout):
    """Read a raster header through GDAL's own HTTP stack."""
    import rasterio

    options = {
        "GDAL_HTTP_CONNECTTIMEOUT": str(int(timeout)),
        "GDAL_HTTP_TIMEOUT": str(int(timeout)),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    with rasterio.Env(**options):
        with rasterio.open(url) as src:
            src.profile


def _default_probes():
    """Build the probe table from the URLs the pipeline actually uses."""
    from string import Template

    from .pre import data_sources as ds

    asris = Template(ds.ASRIS_WCS).substitute(LAYER="CLY")

    return (
        Probe(
            host="explorer.dea.ga.gov.au",
            stack=Stack.REQUESTS,
            url=ds.DEA_STAC,
            description="DEA STAC (fire severity imagery search)",
            run=_probe_requests,
        ),
        Probe(
            host="thredds.nci.org.au",
            stack=Stack.REQUESTS,
            url=ds.DEA_LANDCOVER,
            description="NCI THREDDS (DEA Land Cover mosaics)",
            run=_probe_requests,
        ),
        Probe(
            host="www.asris.csiro.au",
            stack=Stack.REQUESTS,
            url=asris,
            description="ASRIS WCS (soil grids, owslib)",
            run=_probe_requests,
        ),
        Probe(
            host="stochastic-rain.apps.hydrograph.au",
            stack=Stack.REQUESTS,
            url=ds.STOCHASTIC_RAINFALL_API,
            description="Stochastic rainfall API",
            run=_probe_requests,
        ),
        Probe(
            host="data.tern.org.au",
            stack=Stack.URLLIB,
            url=ds.TERN_SLGA_STAC,
            description="TERN SLGA STAC catalogue (pystac, urllib)",
            run=_probe_urllib,
        ),
        Probe(
            host="dea-public-data.s3-ap-southeast-2.amazonaws.com",
            stack=Stack.GDAL,
            url=ds.DEMH,
            description="SRTM DEM-H mosaic (GDAL /vsicurl)",
            run=_probe_gdal,
        ),
        Probe(
            host="bushfire.blob.core.windows.net",
            stack=Stack.GDAL,
            url=ds.ARIDITY_GRID_COARSE,
            description="Aridity grid (GDAL /vsicurl)",
            run=_probe_gdal,
        ),
    )


PROBES = _default_probes()


# ---------------------------------------------------------------------------
# Running the probes
# ---------------------------------------------------------------------------

def presented_issuer(host, port=443, timeout=DEFAULT_TIMEOUT):
    """
    Name the authority signing the certificates a host presents.

    Opens one connection with verification switched off, reads the
    chain and closes it.  Nothing is sent and no data is fetched: this
    exists only to turn "some certificate you don't trust" into a name
    the user can quote to their IT department.  It is called only after
    verification has already failed for that host.

    Parameters:
    - host: hostname to connect to.
    - port: TCP port (default 443).
    - timeout: connect timeout in seconds.

    Returns:
    - Issuer name string, or None if the chain could not be read.
    """
    import socket
    import ssl

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with socket.create_connection((host, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=host) as tls:
            chain = None
            if hasattr(tls, "get_unverified_chain"):
                chain = tls.get_unverified_chain()
            if not chain:
                leaf = tls.getpeercert(binary_form=True)
                chain = [leaf] if leaf else []
    return top_issuer(chain)


def run_checks(probes=None, timeout=DEFAULT_TIMEOUT, inspect=None):
    """
    Probe every endpoint and classify what happened.

    Parameters:
    - probes: iterable of Probe; defaults to PROBES.
    - timeout: per-probe timeout in seconds.
    - inspect: callable(host, timeout=...) naming the issuer of a
      host's certificate chain; defaults to presented_issuer.  Called
      at most once per host, and only after a verification failure.

    Returns:
    - List of ProbeResult, in probe order.
    """
    if probes is None:
        probes = PROBES
    if inspect is None:
        inspect = presented_issuer

    issuers = {}
    results = []

    for probe in probes:
        try:
            probe.run(probe.url, timeout)
        except Exception as exc:  # noqa: BLE001 - every failure is a result
            outcome, detail = classify(exc)
        else:
            outcome, detail = Outcome.OK, ""

        issuer = None
        if outcome is Outcome.TLS:
            if probe.host not in issuers:
                try:
                    issuers[probe.host] = inspect(probe.host, timeout=timeout)
                except Exception:  # noqa: BLE001 - diagnosis is best effort
                    issuers[probe.host] = None
            issuer = issuers[probe.host]

        results.append(
            ProbeResult(
                probe=probe, outcome=outcome, detail=detail, issuer=issuer,
            )
        )

    return results
