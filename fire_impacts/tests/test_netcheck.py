"""
Diagnosis of TLS/connectivity failures against the remote data services.

No network: every probe is injected. What is worth testing is the
judgement `netcheck` makes, not the sockets it opens - specifically
that it separates "your corporate proxy is re-signing certificates"
from "this host is unreachable", because the two need completely
different responses from the user, and the exceptions that carry them
come from three unrelated libraries (requests, urllib and GDAL).
"""

import socket
import ssl
import urllib.error

import pytest
import requests

from fire_impacts import netcheck
from fire_impacts.netcheck import Outcome, Stack


def _verify_error():
    """The exception OpenSSL raises when the issuer isn't trusted."""
    return ssl.SSLCertVerificationError(
        1,
        '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: '
        'unable to get local issuer certificate (_ssl.c:1016)',
    )


class TestClassifyRequestsFailures:
    """requests wraps everything in its own exception hierarchy."""

    def test_ssl_error_is_a_tls_failure(self):
        exc = requests.exceptions.SSLError(_verify_error())

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TLS

    def test_connection_error_is_not_mistaken_for_tls(self):
        exc = requests.ConnectionError('no route to host')

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.CONNECT

    def test_timeout_is_its_own_outcome(self):
        exc = requests.Timeout('too slow')

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TIMEOUT

    def test_ssl_error_subclasses_connection_error_but_wins(self):
        # requests.exceptions.SSLError inherits from ConnectionError, so
        # a naive isinstance chain in the wrong order reports every
        # certificate problem as an unreachable host.
        assert issubclass(
            requests.exceptions.SSLError, requests.ConnectionError
        )


class TestClassifyUrllibFailures:
    """pystac reads the STAC catalogue through urllib, not requests."""

    def test_urlerror_wrapping_a_verify_failure_is_tls(self):
        exc = urllib.error.URLError(_verify_error())

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TLS

    def test_urlerror_wrapping_a_socket_error_is_a_connect_failure(self):
        exc = urllib.error.URLError(socket.gaierror('name resolution'))

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.CONNECT

    def test_bare_verify_error_is_tls(self):
        outcome, _ = netcheck.classify(_verify_error())

        assert outcome is Outcome.TLS


class TestClassifyGdalFailures:
    """
    GDAL reports curl's problems as a string inside RasterioIOError, so
    classification here is necessarily message matching.  Both TLS
    backends are covered: OpenSSL (Linux/macOS conda) and Schannel
    (Windows conda).
    """

    def _raster_error(self, message):
        from rasterio.errors import RasterioIOError

        return RasterioIOError(message)

    def test_openssl_untrusted_issuer_is_tls(self):
        exc = self._raster_error(
            "'/vsicurl/https://example.org/x.tif' not recognized as being "
            "in a supported file format. curl error: SSL certificate "
            "problem: unable to get local issuer certificate"
        )

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TLS

    def test_schannel_untrusted_root_is_tls(self):
        exc = self._raster_error(
            'CURL error: schannel: next InitializeSecurityContext failed: '
            'CRYPT_E_NO_REVOCATION_CHECK / CERT_E_UNTRUSTEDROOT'
        )

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TLS

    def test_unresolvable_host_is_a_connect_failure(self):
        exc = self._raster_error(
            'CURL error: Could not resolve host: example.org'
        )

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.CONNECT

    def test_curl_timeout_is_a_timeout(self):
        exc = self._raster_error(
            'CURL error: Operation timed out after 30000 milliseconds'
        )

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.TIMEOUT

    def test_a_genuine_format_error_is_not_reported_as_a_network_problem(
        self,
    ):
        exc = self._raster_error(
            "'/vsicurl/https://example.org/x.tif' not recognized as being "
            'in a supported file format.'
        )

        outcome, _ = netcheck.classify(exc)

        assert outcome is Outcome.OTHER

    def test_the_detail_carries_the_original_message(self):
        exc = self._raster_error('CURL error: Could not resolve host: x')

        _, detail = netcheck.classify(exc)

        assert 'Could not resolve host' in detail


class TestClassifyHttpStatus:

    def test_an_http_error_status_is_not_a_tls_problem(self):
        exc = netcheck.HttpStatusError(503)

        outcome, detail = netcheck.classify(exc)

        assert outcome is Outcome.HTTP
        assert '503' in detail


class TestStacksAreDeclared:
    """The recipe depends on knowing which trust store each probe uses."""

    def test_the_three_trust_stacks_exist(self):
        assert {Stack.REQUESTS, Stack.URLLIB, Stack.GDAL}


# A real, public intermediate certificate.  Embedded rather than fetched
# so the DER -> dict decode path is exercised without a network call;
# the stdlib route it uses is roundabout enough to be worth locking in.
RAPIDSSL_INTERMEDIATE_PEM = """\
-----BEGIN CERTIFICATE-----
MIIEszCCA5ugAwIBAgIQCyWUIs7ZgSoVoE6ZUooO+jANBgkqhkiG9w0BAQsFADBh
MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3
d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH
MjAeFw0xNzExMDIxMjI0MzNaFw0yNzExMDIxMjI0MzNaMGAxCzAJBgNVBAYTAlVT
MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j
b20xHzAdBgNVBAMTFlJhcGlkU1NMIFRMUyBSU0EgQ0EgRzEwggEiMA0GCSqGSIb3
DQEBAQUAA4IBDwAwggEKAoIBAQC/uVklRBI1FuJdUEkFCuDL/I3aJQiaZ6aibRHj
ap/ap9zy1aYNrphe7YcaNwMoPsZvXDR+hNJOo9gbgOYVTPq8gXc84I75YKOHiVA4
NrJJQZ6p2sJQyqx60HkEIjzIN+1LQLfXTlpuznToOa1hyTD0yyitFyOYwURM+/CI
8FNFMpBhw22hpeAQkOOLmsqT5QZJYeik7qlvn8gfD+XdDnk3kkuuu0eG+vuyrSGr
5uX5LRhFWlv1zFQDch/EKmd163m6z/ycx/qLa9zyvILc7cQpb+k7TLra9WE17YPS
n9ANjG+ECo9PDW3N9lwhKQCNvw1gGoguyCQu7HE7BnW8eSSFAgMBAAGjggFmMIIB
YjAdBgNVHQ4EFgQUDNtsgkkPSmcKuBTuesRIUojrVjgwHwYDVR0jBBgwFoAUTiJU
IBiV5uNu5g/6+rkS7QYXjzkwDgYDVR0PAQH/BAQDAgGGMB0GA1UdJQQWMBQGCCsG
AQUFBwMBBggrBgEFBQcDAjASBgNVHRMBAf8ECDAGAQH/AgEAMDQGCCsGAQUFBwEB
BCgwJjAkBggrBgEFBQcwAYYYaHR0cDovL29jc3AuZGlnaWNlcnQuY29tMEIGA1Ud
HwQ7MDkwN6A1oDOGMWh0dHA6Ly9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEds
b2JhbFJvb3RHMi5jcmwwYwYDVR0gBFwwWjA3BglghkgBhv1sAQEwKjAoBggrBgEF
BQcCARYcaHR0cHM6Ly93d3cuZGlnaWNlcnQuY29tL0NQUzALBglghkgBhv1sAQIw
CAYGZ4EMAQIBMAgGBmeBDAECAjANBgkqhkiG9w0BAQsFAAOCAQEAGUSlOb4K3Wtm
SlbmE50UYBHXM0SKXPqHMzk6XQUpCheF/4qU8aOhajsyRQFDV1ih/uPIg7YHRtFi
CTq4G+zb43X1T77nJgSOI9pq/TqCwtukZ7u9VLL3JAq3Wdy2moKLvvC8tVmRzkAe
0xQCkRKIjbBG80MSyDX/R4uYgj6ZiNT/Zg6GI6RofgqgpDdssLc0XIRQEotxIZcK
zP3pGJ9FCbMHmMLLyuBd+uCWvVcF2ogYAawufChS/PT61D9rqzPRS5I2uqa3tmIT
44JhJgWhBnFMb7AGQkvNq9KNS9dd3GWc17H/dXa1enoxzWjE0hBdFjxPhUb0W3wi
8o34/m8Fxw==
-----END CERTIFICATE-----
"""


class TestIssuerLabel:
    """
    The point of naming the issuer is to give the user something exact to
    ask their IT department for, so the common name matters most.
    """

    def test_prefers_the_common_name(self):
        info = {'issuer': (
            (('countryName', 'US'),),
            (('organizationName', 'Zscaler Inc.'),),
            (('commonName', 'Zscaler Root CA'),),
        )}

        assert netcheck.issuer_label(info) == 'Zscaler Root CA'

    def test_falls_back_to_the_organisation_when_there_is_no_common_name(
        self,
    ):
        info = {'issuer': ((('organizationName', 'Acme Corp IT'),),)}

        assert netcheck.issuer_label(info) == 'Acme Corp IT'

    def test_returns_none_when_the_issuer_is_missing(self):
        assert netcheck.issuer_label({}) is None


class TestDecodeChain:

    def test_reads_the_issuer_out_of_a_der_certificate(self):
        der = ssl.PEM_cert_to_DER_cert(RAPIDSSL_INTERMEDIATE_PEM)

        infos = netcheck.decode_chain([der])

        assert netcheck.issuer_label(infos[0]) == 'DigiCert Global Root G2'

    def test_top_issuer_names_the_signer_of_the_highest_certificate(self):
        der = ssl.PEM_cert_to_DER_cert(RAPIDSSL_INTERMEDIATE_PEM)

        assert netcheck.top_issuer([der]) == 'DigiCert Global Root G2'

    def test_top_issuer_is_none_for_an_empty_chain(self):
        assert netcheck.top_issuer([]) is None

    def test_top_issuer_survives_an_undecodable_certificate(self):
        # A truncated or unexpected chain must degrade to "no name" rather
        # than replacing a certificate error with a decoding error.
        assert netcheck.top_issuer([b'not a certificate']) is None


def _result(stack, outcome, issuer=None, host='example.org'):
    """A ProbeResult without running anything."""
    probe = netcheck.Probe(
        host=host, stack=stack, url=f'https://{host}/', description='x',
    )
    return netcheck.ProbeResult(
        probe=probe, outcome=outcome, detail='', issuer=issuer,
    )


def _joined(results):
    return '\n'.join(netcheck.advice(results)).lower()


class TestAdvice:
    """
    The advice has to name the fix that matches the stacks that actually
    failed.  Sending a Windows user after GDAL_HTTP_CAINFO when GDAL is
    working, or recommending truststore when GDAL is the broken half, is
    the specific failure this whole command exists to prevent.
    """

    def test_nothing_to_say_when_everything_passes(self):
        results = [
            _result(Stack.REQUESTS, Outcome.OK),
            _result(Stack.GDAL, Outcome.OK),
        ]

        assert netcheck.advice(results) == []

    def test_recommends_truststore_when_a_python_stack_fails(self):
        results = [_result(Stack.REQUESTS, Outcome.TLS)]

        assert 'truststore' in _joined(results)

    def test_covers_the_urllib_stack_too(self):
        # pystac reads through urllib, which ignores REQUESTS_CA_BUNDLE -
        # so the manual alternative has to name SSL_CERT_FILE.
        results = [_result(Stack.URLLIB, Outcome.TLS)]
        text = _joined(results)

        assert 'ssl_cert_file' in text

    def test_does_not_send_the_user_after_gdal_when_gdal_is_fine(self):
        results = [
            _result(Stack.REQUESTS, Outcome.TLS),
            _result(Stack.GDAL, Outcome.OK),
        ]

        assert 'gdal_http_cainfo' not in _joined(results)

    def test_names_the_gdal_variables_when_the_raster_reads_fail(self):
        results = [_result(Stack.GDAL, Outcome.TLS)]

        assert 'gdal_http_cainfo' in _joined(results)

    def test_warns_that_truststore_does_not_cover_gdal(self):
        results = [
            _result(Stack.REQUESTS, Outcome.TLS),
            _result(Stack.GDAL, Outcome.TLS),
        ]
        text = _joined(results)

        assert 'truststore' in text
        assert 'gdal_http_cainfo' in text

    def test_quotes_the_issuer_so_it_can_be_asked_for_by_name(self):
        results = [
            _result(Stack.REQUESTS, Outcome.TLS, issuer='Zscaler Root CA'),
        ]

        assert 'Zscaler Root CA' in '\n'.join(netcheck.advice(results))

    def test_asks_for_several_roots_in_the_plural(self):
        # A real interception presents one issuer everywhere, but a
        # partial bundle can leave several distinct public roots
        # untrusted, and "that root certificate" then reads wrongly.
        results = [
            _result(Stack.REQUESTS, Outcome.TLS, issuer='Root A',
                    host='a.example'),
            _result(Stack.URLLIB, Outcome.TLS, issuer='Root B',
                    host='b.example'),
        ]
        text = '\n'.join(netcheck.advice(results))

        assert 'Root A, Root B' in text
        assert 'those root certificates' in text

    def test_asks_for_a_single_root_in_the_singular(self):
        results = [_result(Stack.REQUESTS, Outcome.TLS, issuer='Root A')]

        assert 'that root certificate' in '\n'.join(
            netcheck.advice(results)
        )

    def test_an_unreachable_host_does_not_produce_certificate_advice(self):
        results = [_result(Stack.REQUESTS, Outcome.CONNECT)]
        text = _joined(results)

        assert 'truststore' not in text
        assert 'certificate' not in text

    def test_an_unreachable_host_is_still_reported(self):
        results = [
            _result(Stack.REQUESTS, Outcome.CONNECT, host='blocked.example'),
        ]

        assert 'blocked.example' in '\n'.join(netcheck.advice(results))


class Spy:
    """Records the hosts a chain inspection was attempted for."""

    def __init__(self, issuer='Corporate Root CA', fail=False):
        self.hosts = []
        self.issuer = issuer
        self.fail = fail

    def __call__(self, host, port=443, timeout=None):
        self.hosts.append(host)
        if self.fail:
            raise OSError('socket refused')
        return self.issuer


def _probe(host, stack=Stack.REQUESTS, raises=None):
    def run(url, timeout):
        if raises is not None:
            raise raises

    return netcheck.Probe(
        host=host,
        stack=stack,
        url=f'https://{host}/',
        description='probe',
        run=run,
    )


class TestRunChecks:

    def test_a_probe_that_returns_cleanly_passes(self):
        results = netcheck.run_checks([_probe('ok.example')], inspect=Spy())

        assert [r.outcome for r in results] == [Outcome.OK]

    def test_every_probe_runs_even_after_one_fails(self):
        probes = [
            _probe('first.example', raises=requests.ConnectionError('x')),
            _probe('second.example'),
        ]

        results = netcheck.run_checks(probes, inspect=Spy())

        assert [r.probe.host for r in results] == [
            'first.example', 'second.example',
        ]
        assert [r.outcome for r in results] == [
            Outcome.CONNECT, Outcome.OK,
        ]

    def test_a_tls_failure_is_followed_up_by_naming_the_issuer(self):
        spy = Spy(issuer='Zscaler Root CA')
        probe = _probe(
            'mitm.example',
            raises=requests.exceptions.SSLError(_verify_error()),
        )

        results = netcheck.run_checks([probe], inspect=spy)

        assert spy.hosts == ['mitm.example']
        assert results[0].issuer == 'Zscaler Root CA'

    def test_an_unreachable_host_is_not_probed_again_without_verification(
        self,
    ):
        # Opening an unverified socket is only defensible as a follow-up
        # to a certificate error.  A blocked host must not trigger one.
        spy = Spy()
        probe = _probe(
            'blocked.example', raises=requests.ConnectionError('refused'),
        )

        results = netcheck.run_checks([probe], inspect=spy)

        assert spy.hosts == []
        assert results[0].issuer is None

    def test_a_failed_inspection_does_not_break_the_run(self):
        spy = Spy(fail=True)
        probe = _probe(
            'mitm.example',
            raises=requests.exceptions.SSLError(_verify_error()),
        )

        results = netcheck.run_checks([probe], inspect=spy)

        assert results[0].outcome is Outcome.TLS
        assert results[0].issuer is None

    def test_the_same_host_is_only_inspected_once(self):
        spy = Spy()
        probes = [
            _probe(
                'mitm.example', stack,
                raises=requests.exceptions.SSLError(_verify_error()),
            )
            for stack in (Stack.REQUESTS, Stack.URLLIB)
        ]

        netcheck.run_checks(probes, inspect=spy)

        assert spy.hosts == ['mitm.example']


class TestDefaultProbes:
    """
    The probes have to stay tied to the URLs the pipeline really uses,
    or the command will cheerfully pass while downloads fail.
    """

    def test_all_three_trust_stacks_are_covered(self):
        stacks = {p.stack for p in netcheck.PROBES}

        assert stacks == {Stack.REQUESTS, Stack.URLLIB, Stack.GDAL}

    def test_every_probe_url_comes_from_data_sources(self):
        from fire_impacts.pre import data_sources

        declared = [
            value for name, value in vars(data_sources).items()
            if isinstance(value, str) and value.startswith('https://')
        ]

        for probe in netcheck.PROBES:
            assert any(
                probe.host in url for url in declared
            ), f'{probe.host} is not a host this package downloads from'

    def test_every_probe_is_runnable(self):
        for probe in netcheck.PROBES:
            assert callable(probe.run)


class TestCheckNetworkCommand:
    """
    The command exists so a client on a locked-down network can send back
    one piece of output that says which half of the stack is broken.
    """

    @pytest.fixture()
    def invoke(self, monkeypatch):
        from typer.testing import CliRunner

        from fire_impacts import cli

        runner = CliRunner()

        def run(results):
            monkeypatch.setattr(
                cli.netcheck, 'run_checks', lambda **kwargs: results,
            )
            return runner.invoke(cli.app, ['check-network'])

        return run

    def test_reports_success_for_every_host(self, invoke):
        result = invoke([
            _result(Stack.REQUESTS, Outcome.OK, host='a.example'),
            _result(Stack.GDAL, Outcome.OK, host='b.example'),
        ])

        assert result.exit_code == 0
        assert 'a.example' in result.output
        assert 'b.example' in result.output

    def test_exits_non_zero_when_something_failed(self, invoke):
        result = invoke([_result(Stack.REQUESTS, Outcome.TLS)])

        assert result.exit_code != 0

    def test_prints_the_advice_for_the_stacks_that_failed(self, invoke):
        result = invoke([
            _result(Stack.URLLIB, Outcome.TLS, issuer='Acme Root CA'),
        ])

        assert 'truststore' in result.output
        assert 'Acme Root CA' in result.output

    def test_names_the_stack_alongside_each_host(self, invoke):
        result = invoke([
            _result(Stack.GDAL, Outcome.TLS, host='raster.example'),
        ])

        assert 'gdal' in result.output.lower()


@pytest.mark.network
class TestAgainstTheRealServices:
    """
    Opt in with `pytest -m network`.  This is the check that the probe
    URLs still point at something live - the rest of the suite would
    happily pass with every endpoint retired.
    """

    def test_every_service_is_reachable_from_an_open_network(self):
        results = netcheck.run_checks()

        failed = [
            f'{r.probe.host} ({r.outcome.value}): {r.detail}'
            for r in results if not r.ok
        ]

        assert failed == []
