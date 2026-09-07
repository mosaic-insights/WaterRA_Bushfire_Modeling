# Running on a corporate network

Most organisations inspect outbound HTTPS traffic. A proxy terminates
each connection, re-signs it with an internal certificate authority, and
passes it on. Web browsers accept this because the internal root
certificate is installed in the operating system's certificate store,
where browsers look. Python does not look there — it ships its own list
of trusted authorities — so downloads fail with:

```
SSLError: HTTPSConnectionPool(host='...', port=443): Max retries exceeded
(Caused by SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate'))
```

Nothing is wrong with your installation, and nothing is wrong with the
remote service. Python simply has not been told to trust your
organisation's certificate authority.

## Start here

```
fire-impacts check-network
```

This contacts one endpoint per remote data service and reports which of
them work. Run it before doing anything else: it prints the fix that
matches your situation, and its output is the single most useful thing
to send to whoever is helping you.

It exits with a non-zero status if anything failed, so it can also be
used as an installation check in a script.

## Why one fix is not enough

This package reaches the network through three separate libraries, and
**each has its own idea of which certificate authorities to trust**.
Configuring one does not configure the others, which is why a partial
fix produces the frustrating pattern of one download starting to work
while the next still fails.

| Trust store | Used for | Configured by |
| --- | --- | --- |
| `requests` | DEA STAC imagery search, ASRIS soil WCS, DEA Land Cover, the rainfall API | `REQUESTS_CA_BUNDLE` |
| `urllib` | the TERN soil catalogue (read through `pystac`) | `SSL_CERT_FILE` **only** — it ignores `REQUESTS_CA_BUNDLE` |
| GDAL / curl | every raster read over HTTPS: the DEM, the C and K factor grids, the aridity grid, TERN soil COGs | `GDAL_HTTP_CAINFO`, `CURL_CA_BUNDLE` — it ignores both of the above |

`check-network` probes all three, so its advice covers whichever are
actually broken.

## Fix 1 (recommended): use the operating system's certificate store

Your organisation's root certificate is already installed at the
operating-system level — it has to be, or browsers would not work.
[`truststore`](https://truststore.readthedocs.io/) makes Python read it
from there, so there is nothing to configure and no file to keep
up to date.

```
conda install -c conda-forge truststore
```

Then, once at the top of a notebook or script, **before** any download:

```python
import truststore
truststore.inject_into_ssl()
```

This covers `requests` and `urllib` together, including the libraries
that ignore `REQUESTS_CA_BUNDLE`.

`truststore` is a recommendation, not a dependency of this package. It
changes process-wide TLS behaviour, so the decision to install it and
where to activate it belongs with whoever runs the analysis, not with a
library that would make it for them silently.

### On Windows, this is usually the whole fix

The conda-forge build of GDAL on Windows uses Schannel, the operating
system's own TLS implementation, so raster reads already trust your
organisation's root and keep working while Python fails. If
`check-network` shows the `gdal` rows passing and only the Python rows
failing, `truststore` is all you need — **leave GDAL's HTTP settings
alone**. Pointing GDAL at a certificate bundle that contains only the
corporate root would break the reads that currently work.

On Linux and macOS, conda's GDAL is built against OpenSSL and reads a
bundle file instead, so it will fail alongside Python and needs Fix 3
as well.

## Fix 2: point Python at a certificate bundle

If `truststore` cannot be installed, ask IT for the internal root
certificate as a PEM file. When `check-network` reports a certificate
failure it names the authority that signed the connection, so you can
ask for that root by name rather than describing the problem.

Set **both** variables — `requests` reads the first, everything built on
`urllib` reads only the second:

```
REQUESTS_CA_BUNDLE=C:\path\to\corporate-root.pem
SSL_CERT_FILE=C:\path\to\corporate-root.pem
```

The file must contain the public roots as well as the corporate one, or
connections to services that are *not* intercepted will start failing.
The simplest way to get that is to concatenate the corporate root onto a
copy of certifi's bundle:

```
python -c "import certifi, shutil; shutil.copy(certifi.where(), 'combined.pem')"
```

then append the corporate root to `combined.pem` and point both
variables at it. `truststore` avoids this bookkeeping entirely, which is
why it is the first recommendation.

## Fix 3: point GDAL at the same bundle

Only if `check-network` reports the `gdal` rows failing:

```
GDAL_HTTP_CAINFO=C:\path\to\corporate-root.pem
CURL_CA_BUNDLE=C:\path\to\corporate-root.pem
```

GDAL never consults `REQUESTS_CA_BUNDLE` or `SSL_CERT_FILE`, and
`truststore` has no effect on it — it is a separate HTTP stack written
in C. This is the usual reason for "we fixed it, but it still fails on
the rasters".

## Setting the variables permanently

Set them in the conda environment rather than system-wide, so they
travel with the environment and do not affect unrelated software:

```powershell
conda activate bushfire-py313
conda env config vars set SSL_CERT_FILE=C:\path\to\corporate-root.pem
conda env config vars set REQUESTS_CA_BUNDLE=C:\path\to\corporate-root.pem
conda deactivate
conda activate bushfire-py313
```

## What not to do

**Do not disable certificate verification.** Suggestions to pass
`verify=False`, set `CURL_CA_BUNDLE=""`, or use
`GDAL_HTTP_UNSAFESSL=YES` circulate widely and will appear to work. They
disable the check that a connection is going where it claims, for every
download, permanently — and they tend to survive into production long
after the person who added them has moved on. This package deliberately
offers no such switch.

## Hosts to have allowed

If `check-network` reports a host as unreachable rather than untrusted,
a firewall is blocking it outright and no certificate configuration will
help. These are the hosts the package downloads from:

| Host | Data |
| --- | --- |
| `explorer.dea.ga.gov.au` | DEA STAC — Sentinel-2 and Landsat imagery search |
| `thredds.nci.org.au` | DEA Land Cover mosaics |
| `www.asris.csiro.au` | ASRIS soil grids (WCS) |
| `data.tern.org.au` | TERN Soil and Landscape Grid catalogue and rasters |
| `dea-public-data.s3-ap-southeast-2.amazonaws.com` | SRTM DEM-H elevation mosaic |
| `bushfire.blob.core.windows.net` | RUSLE C and K factor grids, aridity grid, reference dNBR rasters |
| `stochastic-rain.apps.hydrograph.au` | Stochastic rainfall API |

If your network requires an explicit proxy, set `HTTPS_PROXY` as well;
`requests` and GDAL both honour it.
