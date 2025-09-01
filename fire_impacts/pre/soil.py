'''
This module contains functions to download soil-related data (Silt, Clay, Sand, Bulk Density) and aridity data
'''

from owslib.wcs import WebCoverageService
import rasterio
import rasterio.mask
import os
import logging
from .project import FireImpactsProject
from .util import clip_and_reproject_raster, reproject_raster
from ..util import retry
from .data_sources import ASRIS_WCS, TERN_SLGA_STAC
from contextlib import contextmanager
import tempfile
logger = logging.getLogger(__name__)

LAYER_NAMES={
    'SILT':'SLT',
    'CLAY':'CLY',
    'SAND':'SND',
    'BULK_DENSITY':'BDW'
}

DEFAULT_WCS=ASRIS_WCS
DEFAULT_RESOLUTION=0.00024955 # 1 arc second (SRTM)
SOIL_DEPTHS=['000_005', '005_015']

def get_wcs(url):
    return retry(lambda:WebCoverageService(url, version="1.0.0"))

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
        try:
            # Connect to the WCS server for each dataset
            wcs = get_wcs(wcs_url)

            # Iterate through all coverages available in the WCS contents
            for coverage_id in wcs.contents:
                # Get the metadata for the current coverage
                coverage_metadata = wcs.contents[coverage_id]

                # Get the title of the coverage for file naming
                coverage_title = coverage_metadata.title

                # Filter coverages based on the presence of specific substrings
                if not any(layer in coverage_title for layer in filter_layers):
                    continue

                # Request the coverage data using the GetCoverage operation
                response = wcs.getCoverage(
                    identifier=coverage_id,
                    bbox=bbox,
                    format="GeoTIFF",  # Use GeoTIFF format
                    crs="EPSG:4326",  # Coordinate reference system
                    resx=resx,  # Set the resolution for x
                    resy=resy   # Set the resolution for y
                )

                # Get the correct subfolder for the dataset type (Silt, Sand, Clay, Bulk Density)
                dataset_folder = project.catchment_path(catchment,'Soils',data_type)
                os.makedirs(dataset_folder, exist_ok=True)

                # Define the output filename based on the coverage title and data type
                output_filename = os.path.join(dataset_folder, f"{coverage_title}.tif")
                tmp_filename = os.path.join(dataset_folder, f"{coverage_title}_tmp.tif")
                # Save the downloaded data as a GeoTIFF
                with open(tmp_filename, "wb") as file:
                    file.write(response.read())

                # Reproject the raster to the catchment CRS and resolution
                reproject_raster(tmp_filename, crs, output_filename)
                os.remove(tmp_filename)

                logger.info(f"Downloaded {data_type} data saved as {coverage_title}.tif")

        except Exception as e:
            logger.info(f"Error processing {data_type} for {catchment}", exc_info=True)
            raise

    logger.info("Process Done!.")


def extract_aridity_data(project:FireImpactsProject,aridity_raster:str, catchment=None):
    """
    Extract aridity data for each catchment bounding box and save the clipped raster in the Aridity folder.

    Parameters:
    - project (fire_impacts.FireImpactsProject): A dictionary of project folders created for catchments.
    - aridity_raster_path (str): Path to the aridity raster layer.
    - catchment (str): OPTIONAL: Name of the catchment to process. If None, process all catchments.
    """
    if catchment is None:
        project.for_each_catchment(lambda c: extract_aridity_data(project,aridity_raster, c))
        return

    bbox = project.catchment_bounds(catchment,10.0)
    crs = project.catchment_crs(catchment)

    with rasterio.open(aridity_raster) as src:
        logger.info(f"Processing Aridity data for catchment: {catchment}")
        try:
            # Define the bounding box geometry
            geometry = {
                'type': 'Polygon',
                'coordinates': [[
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]],
                    [bbox[0], bbox[1]]
                ]]
            }
            # Mask the raster data using the bounding box
            out_image, out_transform = rasterio.mask.mask(src, [geometry], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            # Save the clipped aridity data to the appropriate folder
            output_path = project.catchment_path(catchment,'Soils','Aridity.tif')
            tmp_path = project.catchment_path(catchment,'Soils','Aridity_tmp.tif')

            with rasterio.open(tmp_path, "w", **out_meta) as dest:
                dest.write(out_image)

            reproject_raster(tmp_path, crs, output_path)
            os.remove(tmp_path)
        except Exception as e:
            logger.error(f"Error processing {catchment}", exc_info=True)
            raise

    logger.info("Aridity extraction completed.")


def get_stac(base_uri):
    from pystac import Catalog, Collection
    from pystac.stac_io import RetryStacIO
    from urllib3.util import Retry

    retries = Retry(total=5, backoff_factor=1)
    stac_io = RetryStacIO()
    if base_uri.endswith("catalog.json"):
        return Catalog.from_file(base_uri, stac_io=stac_io)
    return Collection.from_file(base_uri, stac_io=stac_io)

def find_slga_grids(base_catalog=TERN_SLGA_STAC,variables=None,depths=None,version='v2'):
    if variables is None:
        variables = ['SLT','CLY','SND','BDW']
    if depths is None:
        depths = SOIL_DEPTHS

    catalog = get_stac(base_catalog)
    entries = list(catalog.get_children())
    relevant = [e for e in entries if e.id in variables]
    assets = []
    for cat in relevant:
        variable = cat.id
        version_catalog = cat.get_child(version)
        assets += [(variable,k,v.get_absolute_href()) \
                   for k,v in version_catalog.get_assets().items() \
                    if any(d for d in depths if d in k) and '_EV_' in k]
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
        f.write(f"x-api-key: {key}\n")
    logger.info(f"Using GDAL API key from temporary file {fn}")
    try:
        yield
    finally:
        if old_env_var is None:
            del os.environ[env_var]
        else:
            os.environ[env_var] = old_env_var
        logger.info(f"Removing temporary GDAL API key file {fn}")
        os.remove(fn)

def download_soil_data_stac(project:FireImpactsProject, catchment:str=None, api_key:str=None,
                            base_stac_catalog=TERN_SLGA_STAC,version='v2'):
    """
    Download soil-related data (Silt, Clay, Sand, Bulk Density) for each catchment
    from STAC URLs using catchment boundary, and save the data in the appropriate folder.

    Intended for use with original TERN's STAC, which requires an API key.
    Create a TERN account and obtain an API key from the TERN website (https://account.tern.org.au/)

    Parameters:
    - project (fire_impacts.FireImpactsProject): A dictionary of project folders created for catchments.
    - catchment (str): OPTIONAL: Name of the catchment to process. If None, process all catchments.
    - tern_api_key (str): API key for accessing the TERN STAC API.
    - base_stac_catalog (str): Base URL for the TERN STAC catalog.
    """
    if catchment is None:
        project.for_each_catchment(lambda c: download_soil_data_stac(project,c, api_key,base_stac_catalog,version))
        return

    if api_key is None:
        logger.error("API key is required for STAC access.")
        raise ValueError("API key is required for STAC access.")

    grids = find_slga_grids(base_stac_catalog, version=version)
    logger.info(f"Processing catchment: {catchment} with {len(grids)} grids found")
    with gdal_api_key(api_key):
        for var,fn,url in grids:
            logger.info('Downloading %s',fn)
            dest_dir = project.catchment_path(catchment,'Soils',var)
            os.makedirs(dest_dir, exist_ok=True)
            dest_fn = os.path.join(dest_dir, fn)
            clip_and_reproject_raster(url,project.boundary_files[catchment],dest_fn)
