"""
Download soil-related data (Silt, Clay, Sand, Bulk Density) and aridity.

Supports two download paths: a WCS path using the ASRIS WCS service and
a STAC path using TERN's SLGA Cloud-Optimised GeoTIFF catalogue.
"""

from string import Template
import rasterio
import rasterio.mask
import os
import logging
from .project import FireImpactsProject
from ..context import RunContext
from .util import (
    clip_and_reproject_raster,
    reproject_raster,
    retrieve_grid_from_wcs_for_bounds,
)
from .data_sources import ASRIS_WCS, TERN_SLGA_STAC, ARIDITY_GRID_COARSE
from contextlib import contextmanager
import socket
import tempfile

logger = logging.getLogger(__name__)

# Defaults for the remote TERN STAC downloads.  Tuned to abort a run
# that has stalled (API slow / not cleanly failing) rather than hang
# indefinitely.  Override per-call via download_soil_data_stac kwargs.
DEFAULT_STAC_CONNECT_TIMEOUT = 30    # seconds, TCP connect
DEFAULT_STAC_REQUEST_TIMEOUT = 600   # seconds, single HTTP request cap
DEFAULT_STAC_LOW_SPEED_LIMIT = 100   # bytes/sec — below this counts as stalled
DEFAULT_STAC_LOW_SPEED_TIME = 60     # seconds stalled before aborting


@contextmanager
def _socket_default_timeout(seconds):
    """
    Temporarily set the process-wide default socket timeout.

    Covers network libraries (pystac/urllib3) that don't expose a
    timeout parameter directly.  The original timeout is restored
    on exit.

    Parameters:
    - seconds: timeout in seconds, or None to skip (no-op).

    Returns:
    - Context manager that yields with the socket timeout set.
    """
    if seconds is None:
        yield
        return
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


LAYER_NAMES = {
    "SILT": "SLT",
    "CLAY": "CLY",
    "SAND": "SND",
    "BULK_DENSITY": "BDW",
}

DEFAULT_WCS = ASRIS_WCS
DEFAULT_RESOLUTION = 0.00024955  # 1 arc second (SRTM)
SOIL_DEPTHS = ["000_005", "005_015"]


# ---------------------------------------------------------------------------
# WCS download path
# ---------------------------------------------------------------------------

def download_soil_data_wcs(
    ctx: RunContext,
    wcs_urls=None,
    resx=DEFAULT_RESOLUTION,
    resy=DEFAULT_RESOLUTION,
):
    """
    Download soil data (Silt, Clay, Sand, Bulk Density) from WCS URLs.

    Retrieves each soil variable within the catchment bounding box and
    saves the reprojected GeoTIFFs to the catchment's Soils subfolder.
    Intended for use with the ASRIS WCS, which may not always be
    available; prefer download_soil_data_stac where possible.

    Parameters:
    - ctx: catchment-only RunContext.
    - wcs_urls: dict mapping layer names (SILT, CLAY, SAND,
      BULK_DENSITY) to WCS URLs; defaults to the ASRIS WCS server.
    - resx: x resolution for the coverage request.
    - resy: y resolution for the coverage request.

    Returns:
    - None.  Writes GeoTIFFs to the catchment's Soils/<layer> folder.
    """
    if wcs_urls is None:
        wcs_urls = {
            key: Template(DEFAULT_WCS).substitute(LAYER=value)
            for key, value in LAYER_NAMES.items()
        }

    catchment = ctx.catchment
    # Only download soil data for the top two depth intervals
    filter_layers = SOIL_DEPTHS
    bbox = ctx.project.catchment_bounds(catchment, 10.0)
    bbox = [float(f) for f in list(bbox)]
    crs = ctx.project.catchment_crs(catchment)
    logger.info(
        "Processing catchment: %s with bounding box %s", catchment, bbox
    )

    # Iterate through each dataset type (SILT, CLAY, SAND, BULK DENSITY)
    for data_type, wcs_url in wcs_urls.items():
        dataset_folder = ctx.catchment_path("Soils", data_type)

        try:
            retrieve_grid_from_wcs_for_bounds(
                data_type, wcs_url, bbox, resx, resy,
                crs, dataset_folder, filter_layers,
            )
        except Exception as e:
            logger.info(
                "Error processing %s for %s", data_type, catchment,
                exc_info=True,
            )
            raise

    logger.info("Process Done!")


# ---------------------------------------------------------------------------
# Aridity extraction
# ---------------------------------------------------------------------------

def extract_aridity_data(
    ctx: RunContext,
    aridity_raster: str = None,
):
    """
    Extract aridity data for the context's catchment.

    Parameters:
    - ctx: catchment-only RunContext.
    - aridity_raster: path or URL to the aridity raster; defaults to
      ARIDITY_GRID_COARSE.

    Returns:
    - None.  Writes Aridity.tif to the catchment's Soils folder.
    """
    shapefile = ctx.project.boundary_files[ctx.catchment]

    if aridity_raster is None:
        aridity_raster = ARIDITY_GRID_COARSE
    output_path = ctx.catchment_path("Soils", "Aridity.tif")

    clip_and_reproject_raster(aridity_raster, shapefile, output_path)

    logger.info("Aridity extraction completed.")


# ---------------------------------------------------------------------------
# TERN STAC helpers
# ---------------------------------------------------------------------------

def get_stac(base_uri, api_key=None):
    """
    Open a TERN STAC Catalog or Collection with retry logic.

    Parameters:
    - base_uri: URI to the STAC root JSON (catalog.json or
      collection JSON).
    - api_key: optional TERN API key for authenticated access.

    Returns:
    - pystac Catalog or Collection object.
    """
    from pystac import Catalog, Collection
    from pystac.stac_io import RetryStacIO
    from urllib3.util import Retry

    retries = Retry(total=5, backoff_factor=1)

    headers = {"X-Api-Key": api_key} if api_key else None
    stac_io = RetryStacIO(headers=headers, retry=retries)

    if base_uri.endswith("catalog.json"):
        return Catalog.from_file(base_uri, stac_io=stac_io)
    return Collection.from_file(base_uri, stac_io=stac_io)


def find_slga_grids(
    base_catalog=TERN_SLGA_STAC,
    variables=None,
    depths=None,
    version="v2",
    api_key=None,
):
    """
    Traverse the TERN SLGA STAC catalogue and return asset download URLs.

    Parameters:
    - base_catalog: URI to the SLGA STAC root catalog JSON.
    - variables: list of soil variable codes to retrieve (default
      ['SLT', 'CLY', 'SND', 'BDW']).
    - depths: list of depth-range strings to filter by (default
      SOIL_DEPTHS = ['000_005', '005_015']).
    - version: STAC version string to look up under each variable
      (default 'v2').
    - api_key: optional TERN API key for authenticated access.

    Returns:
    - List of (variable, item_id, href) tuples for each matched asset.
    """
    if variables is None:
        variables = ["SLT", "CLY", "SND", "BDW"]
    if depths is None:
        depths = SOIL_DEPTHS

    logger.info(
        "Finding SLGA grids in STAC catalog %s for variables %s "
        "and depths %s",
        base_catalog, variables, depths,
    )
    catalog = get_stac(base_catalog, api_key=api_key)
    entries = list(catalog.get_children())
    relevant = [e for e in entries if e.id in variables]
    assets = []
    for cat in relevant:
        logger.info(
            "Processing variable %s with %d items",
            cat.id, len(list(cat.get_children())),
        )
        variable = cat.id
        # Get the items for the specified version
        version_catalog = cat.get_child(version)
        these_assets = []
        for item in version_catalog.get_items():
            # Filter to relevant dataset types (EV) and depths (0-5, 5-15)
            if any(d in item.id for d in depths) and "_EV_" in item.id:
                data_key = item.id + ".tif"
                this_id = item.id
                asset_dict = item.assets
                this_href = asset_dict[data_key]
                these_assets.append((
                    variable,
                    this_id,
                    this_href.get_absolute_href(),
                ))
            else:
                continue

            assets += these_assets

    return assets


@contextmanager
def gdal_api_key(key):
    """
    Set GDAL to read an API key from a temporary header file.

    Writes the key to a temporary file, sets GDAL_HTTP_HEADER_FILE
    to point at it, and restores the previous environment variable
    on exit.  The temporary file is deleted when the context exits.

    Parameters:
    - key: API key string to include in GDAL HTTP headers.

    Returns:
    - Context manager; yields with GDAL configured to use the key.
    """
    fn = tempfile.mktemp("gdal_api_key.txt")
    env_var = "GDAL_HTTP_HEADER_FILE"
    old_env_var = os.environ.get(env_var)
    os.environ[env_var] = fn
    with open(fn, "w") as f:
        f.write(f"X-Api-Key: {key}\n")
    logger.info("Using GDAL API key from temporary file %s", fn)
    with rasterio.Env(GDAL_HTTP_HEADER_FILE=fn):
        try:
            yield
        finally:
            if old_env_var is None:
                del os.environ[env_var]
            else:
                os.environ[env_var] = old_env_var
            logger.info("Removing temporary GDAL API key file %s", fn)
            os.remove(fn)


# ---------------------------------------------------------------------------
# STAC download path
# ---------------------------------------------------------------------------

def download_soil_data_stac(
    ctx: RunContext,
    api_key: str = None,
    base_stac_catalog=TERN_SLGA_STAC,
    version="v2",
    connect_timeout: float = DEFAULT_STAC_CONNECT_TIMEOUT,
    request_timeout: float = DEFAULT_STAC_REQUEST_TIMEOUT,
    low_speed_limit: int = DEFAULT_STAC_LOW_SPEED_LIMIT,
    low_speed_time: int = DEFAULT_STAC_LOW_SPEED_TIME,
):
    """
    Download soil data (Silt, Clay, Sand, Bulk Density) from TERN STAC.

    Traverses the TERN SLGA STAC catalogue, clips each COG to the
    catchment boundary, reprojects to the catchment CRS, and saves the
    result to the catchment's Soils folder.  Requires a TERN API key
    (obtain from https://account.tern.org.au/).

    Parameters:
    - ctx: catchment-only RunContext.
    - api_key: TERN API key (required).
    - base_stac_catalog: URI to the SLGA STAC root catalog.
    - version: STAC version string to use (default 'v2').
    - connect_timeout: TCP connect timeout in seconds for STAC
      traversal and each GDAL raster read (default 30).
    - request_timeout: hard cap in seconds on a single HTTP request
      (default 600; increase if large COG reads are being cut off).
    - low_speed_limit: bytes/sec below which a GDAL transfer is
      treated as stalled (default 100).
    - low_speed_time: seconds a transfer must stay below
      low_speed_limit before GDAL aborts it (default 60).

    Returns:
    - None.  Writes reprojected GeoTIFFs to the catchment's
      Soils/<variable> folder.
    """
    catchment = ctx.catchment
    if api_key is None:
        logger.error("API key is required for STAC access.")
        raise ValueError("API key is required for STAC access.")

    # STAC catalog reads go through pystac/urllib3 and don't accept a
    # timeout kwarg directly; a socket-default timeout is a coarse but
    # effective guard against the catalog hanging.
    with _socket_default_timeout(request_timeout):
        grids = find_slga_grids(
            base_stac_catalog, version=version, api_key=api_key,
        )
    logger.info(
        "Processing catchment: %s with %d grids found",
        catchment, len(grids),
    )
    gdal_timeouts = {
        "GDAL_HTTP_CONNECTTIMEOUT": str(int(connect_timeout)),
        "GDAL_HTTP_TIMEOUT": str(int(request_timeout)),
        "GDAL_HTTP_LOW_SPEED_LIMIT": str(int(low_speed_limit)),
        "GDAL_HTTP_LOW_SPEED_TIME": str(int(low_speed_time)),
    }
    with gdal_api_key(api_key), rasterio.Env(**gdal_timeouts):
        for var, fn, url in grids:
            logger.info("Downloading %s", fn)
            dest_dir = ctx.catchment_path("Soils", var)
            os.makedirs(dest_dir, exist_ok=True)
            dest_fn = os.path.join(dest_dir, fn + ".tif")
            clip_and_reproject_raster(
                url,
                ctx.project.boundary_files[catchment],
                dest_fn,
            )
