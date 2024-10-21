from owslib.wcs import WebCoverageService
import rasterio
import rasterio.mask
import os
from string import Template
import logging
from .project import FireImpactsProject
from .util import reproject_raster
from ..util import retry
logger = logging.getLogger(__name__)

LAYER_NAMES={
    'SILT':'SLT',
    'CLAY':'CLY',
    'SAND':'SND',
    'BULK_DENSITY':'BDW'
}

DEFAULT_WCS='https://www.asris.csiro.au/arcgis/services/TERN/${LAYER}_ACLEP_AU_NAT_C/MapServer/WCSServer?SERVICE=WCS&REQUEST=GetCapabilities'
DEFAULT_RESOLUTION=0.00024955 # 1 arc second (SRTM)
SOIL_DEPTHS=['_000_005_', '_005_015_']

def get_wcs(url):
    return retry(lambda:WebCoverageService(url, version="1.0.0"))

def download_soil_data(project:FireImpactsProject, catchment:str=None, wcs_urls=None, resx=DEFAULT_RESOLUTION, resy=DEFAULT_RESOLUTION):
    """
    Download soil-related data (Silt, Clay, Sand, Bulk Density) for each catchment
    from WCS URLs using bounding boxes, and save the data in the appropriate folder.

    Parameters:
    - shapefile_bboxes (dict): A dictionary of catchment names and their bounding boxes.
    - project (dict): A dictionary of project folders created for catchments.
    - wcs_urls (dict): A dictionary of WCS URLs for Silt, Clay, Sand, and Bulk Density data.
    - resx (float): The x resolution for the data download.
    - resy (float): The y resolution for the data download.
    """
    if catchment is None:
        catchment_names = project.catchments
        for catchment in catchment_names:
            download_soil_data(project, catchment, wcs_urls, resx, resy)
        return

    if wcs_urls is None:
        wcs_urls = {key: Template(DEFAULT_WCS).substitute(LAYER=value) for key, value in LAYER_NAMES.items()}

    # Just downlaod soil data for depth 5 and 15 cm
    filter_layers = SOIL_DEPTHS
    bbox = project.catchment_bounds(catchment,10.0)
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
    - aridity_raster_path (str): Path to the aridity raster layer.
    - shapefile_bboxes (dict): A dictionary of catchment names and their bounding boxes.
    - project_folders (dict): A dictionary of project folders created for catchments.
    """
    if catchment is None:
        catchment_names = project.catchments
        for catchment in catchment_names:
            extract_aridity_data(project,aridity_raster, catchment)
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