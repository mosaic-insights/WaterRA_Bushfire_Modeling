'''
This module contains functions to download soil-related data (Silt, Clay, Sand, Bulk Density) and aridity data
'''

from string import Template
import rasterio
import rasterio.mask
import os
import logging
from .project import FireImpactsProject
from .util import clip_and_reproject_raster, reproject_raster, retrieve_grid_from_wcs_for_bounds
from .data_sources import ASRIS_WCS, TERN_SLGA_STAC, ARIDITY_GRID_COARSE
from contextlib import contextmanager
import socket
import tempfile
logger = logging.getLogger(__name__)

# Defaults for the remote TERN STAC downloads. Tuned to abort a run
# that has stalled (API slow / not cleanly failing) rather than hang
# indefinitely. Override per-call via download_soil_data_stac kwargs.
DEFAULT_STAC_CONNECT_TIMEOUT = 30     # seconds, TCP connect
DEFAULT_STAC_REQUEST_TIMEOUT = 600    # seconds, single HTTP request cap
DEFAULT_STAC_LOW_SPEED_LIMIT = 100    # bytes/sec — below this counts as stalled
DEFAULT_STAC_LOW_SPEED_TIME = 60      # seconds stalled before aborting


@contextmanager
def _socket_default_timeout(seconds):
    """Temporarily set the process-wide default socket timeout.

    Covers network libraries (pystac / urllib3) that don't expose a
    timeout knob directly. Restored on exit.
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

LAYER_NAMES={
    'SILT':'SLT',
    'CLAY':'CLY',
    'SAND':'SND',
    'BULK_DENSITY':'BDW'
}

DEFAULT_WCS=ASRIS_WCS
DEFAULT_RESOLUTION=0.00024955 # 1 arc second (SRTM)
SOIL_DEPTHS=['000_005', '005_015']

def download_soil_data_wcs(project:FireImpactsProject, catchment:str=None, wcs_urls=None, resx=DEFAULT_RESOLUTION, resy=DEFAULT_RESOLUTION):
    """
    Download soil-related data (Silt, Clay, Sand, Bulk Density) for each catchment
    from WCS URLs using bounding boxes, and save the data in the appropriate folder.

    Intended for use with original ASRIS WCS, which may not be available.

    Parameters:
    - project (fire_impacts.FireImpactsProject): A dictionary of project folders created for catchments.
    - catchment (str): OPTIONAL: Name of the catchment to process. If None, process all catchments.
    - wcs_urls (dict): A dictionary of WCS URLs for Silt, Clay, Sand, and Bulk Density data. Defaults to CSIRO WCS server.
    - resx (float): The x resolution for the data download.
    - resy (float): The y resolution for the data download.
    """
    if catchment is None:
        project.for_each_catchment(lambda c: download_soil_data_wcs(project,c, wcs_urls, resx, resy))
        return

    if wcs_urls is None:
        wcs_urls = {key: Template(DEFAULT_WCS).substitute(LAYER=value) for key, value in LAYER_NAMES.items()}
    
    # Just downlaod soil data for depth 5 and 15 cm
    filter_layers = SOIL_DEPTHS
    bbox = project.catchment_bounds(catchment,10.0)
    bbox = [float(f) for f in list(bbox)]
    crs = project.catchment_crs(catchment)
    logger.info(f"Processing catchment: {catchment} with bounding box {bbox}")

    # Iterate through each dataset type (SILT, CLAY, SAND, BULK DENSITY)
    for data_type, wcs_url in wcs_urls.items():
        # Get the correct subfolder for the dataset type (Silt, Sand, Clay, Bulk Density)
        dataset_folder = project.catchment_path(catchment,'Soils',data_type)

        try:
            retrieve_grid_from_wcs_for_bounds(data_type, wcs_url, bbox, resx, resy, crs, dataset_folder, filter_layers)
        except Exception as e:
            logger.info(f"Error processing {data_type} for {catchment}", exc_info=True)
            raise

    logger.info("Process Done!.")

###############################################################################
def extract_aridity_data(
    project:FireImpactsProject,
    aridity_raster:str=None,
    catchment=None
    ):
    """
    Extract aridity data for each catchment and save the clipped raster 
    in the Aridity folder.

    Parameters:
    - project (fire_impacts.FireImpactsProject): A dictionary of 
    project folders created for catchments.
    - aridity_raster_path (str): Path to the aridity raster layer.
    - catchment (str): OPTIONAL: Name of the catchment to process. If 
    None, process all catchments.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    if catchment is None:
        project.for_each_catchment(
            lambda c: extract_aridity_data(project,aridity_raster, c)
            )
        return

    # Extract the catchment boundary from the project:
    shapefile = project.boundary_files[catchment]

    # Save the clipped aridity data to the appropriate folder
    if aridity_raster is None:
        aridity_raster = ARIDITY_GRID_COARSE
    output_path = project.catchment_path(catchment,'Soils','Aridity.tif')

    clip_and_reproject_raster(
        aridity_raster,
        shapefile,
        output_path
        )

    logger.info("Aridity extraction completed.")


def get_stac(base_uri, api_key=None):
    from pystac import Catalog, Collection
    from pystac.stac_io import RetryStacIO
    from urllib3.util import Retry

    
    retries = Retry(total=5, backoff_factor=1)

    headers = {'X-Api-Key': api_key} if api_key else None
    stac_io = RetryStacIO(headers=headers, retry=retries)

    if base_uri.endswith("catalog.json"):
        return Catalog.from_file(base_uri, stac_io=stac_io)
    return Collection.from_file(base_uri, stac_io=stac_io)

def find_slga_grids(
    base_catalog=TERN_SLGA_STAC,
    variables=None,
    depths=None,
    version='v2',
    api_key=None
    ):
    if variables is None:
        variables = ['SLT','CLY','SND','BDW']
    if depths is None:
        depths = SOIL_DEPTHS

    logger.info(f"Finding SLGA grids in STAC catalog {base_catalog} for variables {variables} and depths {depths}")
    catalog = get_stac(base_catalog, api_key=api_key)
    entries = list(catalog.get_children())
    relevant = [e for e in entries if e.id in variables]
    assets = []
    for cat in relevant:
        logger.info(f"Processing variable {cat.id} with {len(list(cat.get_children()))} items")
        # Name of the current soil variable e.g. CLY for clay
        variable = cat.id
        # Get the items for the specified version:
        version_catalog = cat.get_child(version)
        # Containter for assets for the current variable:
        these_assets = []
        for item in version_catalog.get_items():
            # Filter so we only get relevant dataset types (EV) and
            #depths (0-5, 5-15)
            if (
                any(d in item.id for d in depths) 
                and '_EV_' in item.id
                ):
                data_key = item.id + '.tif'
                this_id = item.id
                asset_dict = item.assets
                this_href = asset_dict[data_key]
                yoosfle_toople = (
                    variable,
                    this_id,
                    this_href.get_absolute_href()
                    )
                these_assets.append(yoosfle_toople)
            else:
                continue
            
            assets += these_assets
        
    return assets

@contextmanager
def gdal_api_key(key):
    # Create an environment with temporary file containing api key
    # Remove the temporary file after use
    fn = tempfile.mktemp('gdal_api_key.txt')
    env_var = 'GDAL_HTTP_HEADER_FILE'
    old_env_var = os.environ.get(env_var)
    os.environ[env_var] = fn
    with open(fn,'w') as f:
        f.write(f"X-Api-Key: {key}\n")
    logger.info(f"Using GDAL API key from temporary file {fn}")
    with rasterio.Env(GDAL_HTTP_HEADER_FILE=fn):
        try:
            yield
        finally:
            if old_env_var is None:
                del os.environ[env_var]
            else:
                os.environ[env_var] = old_env_var
            logger.info(f"Removing temporary GDAL API key file {fn}")
            os.remove(fn)

def download_soil_data_stac(
    project:FireImpactsProject,
    catchment:str=None,
    api_key:str=None,
    base_stac_catalog=TERN_SLGA_STAC,version='v2',
    connect_timeout:float=DEFAULT_STAC_CONNECT_TIMEOUT,
    request_timeout:float=DEFAULT_STAC_REQUEST_TIMEOUT,
    low_speed_limit:int=DEFAULT_STAC_LOW_SPEED_LIMIT,
    low_speed_time:int=DEFAULT_STAC_LOW_SPEED_TIME,
    ):
    """
    Download soil-related data (Silt, Clay, Sand, Bulk Density) for
    each catchment from STAC URLs using catchment boundary, and save
    the data in the appropriate folder.

    Intended for use with original TERN's STAC, which requires an API
    key. Create a TERN account and obtain an API key from the TERN
    website (https://account.tern.org.au/)

    Parameters:
    - project (fire_impacts.FireImpactsProject): A dictionary of
    project folders created for catchments.
    - catchment (str): OPTIONAL: Name of the catchment to process. If
    None, process all catchments.
    - tern_api_key (str): API key for accessing the TERN STAC API.
    - base_stac_catalog (str): Base URL for the TERN STAC catalog.
    - connect_timeout (float): TCP connect timeout in seconds for both
    the STAC catalog traversal and each GDAL raster read. Default 30.
    - request_timeout (float): Hard cap in seconds on a single HTTP
    request. Default 600. Large COG reads can take a while; set higher
    if you see legitimate transfers being cut off.
    - low_speed_limit (int): Bytes/sec below which a GDAL transfer is
    treated as stalled. Default 100.
    - low_speed_time (int): Seconds a transfer must stay below
    *low_speed_limit* before GDAL aborts it. Default 60. This is what
    catches "API is very slow / not failing cleanly" — the download
    errors out instead of hanging forever.
    """
    if catchment is None:
        project.for_each_catchment(
            lambda c: download_soil_data_stac(
                project, c, api_key, base_stac_catalog, version,
                connect_timeout=connect_timeout,
                request_timeout=request_timeout,
                low_speed_limit=low_speed_limit,
                low_speed_time=low_speed_time,
                )
            )
        return

    if api_key is None:
        logger.error("API key is required for STAC access.")
        raise ValueError("API key is required for STAC access.")

    # STAC catalog reads go through pystac / urllib3 and don't accept a
    # timeout kwarg here; a socket-default timeout is a coarse but
    # effective guard against the catalog hanging.
    with _socket_default_timeout(request_timeout):
        grids = find_slga_grids(
            base_stac_catalog, version=version, api_key=api_key,
        )
    logger.info(
        f'Processing catchment: {catchment} with {len(grids)} grids '
        'found'
        )
    gdal_timeouts = {
        'GDAL_HTTP_CONNECTTIMEOUT': str(int(connect_timeout)),
        'GDAL_HTTP_TIMEOUT': str(int(request_timeout)),
        'GDAL_HTTP_LOW_SPEED_LIMIT': str(int(low_speed_limit)),
        'GDAL_HTTP_LOW_SPEED_TIME': str(int(low_speed_time)),
    }
    with gdal_api_key(api_key), rasterio.Env(**gdal_timeouts):
        for var,fn,url in grids:
            logger.info('Downloading %s',fn)
            dest_dir = project.catchment_path(catchment,'Soils',var)
            os.makedirs(dest_dir, exist_ok=True)
            dest_fn = os.path.join(dest_dir, fn + '.tif')
            clip_and_reproject_raster(
                url,
                project.boundary_files[catchment],
                dest_fn
                )
