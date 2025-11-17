import geopandas as gpd
from .project import APPROX_KM_PER_DEGREE
from owslib.wcs import WebCoverageService
from .. import const as c
from ..util import retry
import rasterio as rio
import numpy as np
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

def clip_raster(
    raster_file:str,
    shapefile:str
    ):
    """
    Clip a raster to a shapefile, keeping the raster CRS.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Read the raster file to get its CRS and resolution
    with rio.open(raster_file) as src:
        raster_crs = src.crs
        raster_res = src.res  

    # Read the shapefile
    catchment = gpd.read_file(shapefile)
    # Get the CRS of the shapefile
    shapefile_crs = catchment.crs.to_string()
    # Ensure the shapefile is in the same CRS as the raster before
    #clipping, buffering the shapefile by 2 pixels to ensure it covers 
    #the raster:
    catchment = catchment.to_crs(raster_crs).buffer(raster_res[0]*2)

    # Read the raster file
    with rio.open(raster_file) as src:
        # Clip the raster with the shapefile
        out_image, out_transform = mask(
            src,
            catchment.geometry.apply(mapping),
            crop=True,
            all_touched=True,
            nodata=np.nan,
            pad=False
            )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            'nodata': np.nan
            })

    # Write the clipped raster to a temporary file
    temp_file = 'clipped_temp.tif'
    with rio.open(temp_file, 'w', **out_meta) as dest:
        dest.write(out_image)

    return temp_file, shapefile_crs


###############################################################################
def clip_and_reproject_raster(
    raster_file:str,
    shapefile:str,
    output_file:str,
    target_resolution:float=None
    ):
    """
    Clips a raster file using a shapefile and reprojects the clipped 
    raster to the CRS of the shapefile.

    Parameters:
    - raster_file (str): Path to the input raster file.
    - shapefile (str): Path to the shapefile for clipping.
    - output_files (list): List of paths to the output reprojected 
    raster files.
    - target_resolution (tuple): OPTIONAL: Desired resolution for the 
    output rasters. Default to automatic selection of resolution.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Clip the raster to the slightly-buffered extent of the shapefile:
    temp_file, shapefile_crs = clip_raster(
        raster_file,
        shapefile
        )

    # Reproject the raster to the shapefile's CRS:
    reproject_raster(
        temp_file,
        shapefile_crs,
        output_file,
        target_resolution
        )
    # Clean up temporary file
    os.remove(temp_file)

###############################################################################
def reproject_raster(
    temp_file:str,
    target_crs:str,
    output_file:str,
    target_resolution:float=None
    ):
    """
    
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Reproject the clipped raster to the CRS of the shapefile with the 
    #target resolution
    with rio.open(temp_file) as src:
        logger.info(
           f'Reprojecting raster from %s to %s', src.crs, target_crs
           )
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution
            )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'res': target_resolution  # Set the target resolution explicitly
        })

        with rio.open(output_file, 'w', **kwargs) as dst:
            logger.info(
               f'Reprojecting clipped raster to: {output_file}'
               )
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                    )

###############################################################################
def read_raster(fn:str):
  with rio.open(fn) as src:
    return src.read(1), src.transform, src.crs

###############################################################################
def read_aligned(raster_fn:str, transform, crs,shape,resampling=Resampling.nearest):
    '''
    Read a raster and reproject it to a given crs and window (transform)
    '''
    logger.info(f'Reading raster {raster_fn} and reprojecting to {crs}')
    with rio.open(raster_fn) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': shape[1],
            'height': shape[0],
        })

        with rio.MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
              reproject(
                  source=rio.band(src, 1),
                  destination=rio.band(dst, 1),
                  src_transform=src.transform,
                  src_crs=src.crs,
                  dst_transform=transform,
                  dst_crs=crs,
                  resampling=resampling
              )

            with memfile.open() as src:
              data = src.read(1,masked=True)
              data[data.mask] = np.nan
              return data

###############################################################################
def metres_to_approx_degrees(m:float):
   return m * c.M_TO_KM / APPROX_KM_PER_DEGREE


def retrieve_grid_from_wcs_for_bounds(label, wcs_url, bbox, resx, resy, crs, output_folder, filter_layers=None):
    # Connect to the WCS server for each dataset
    wcs = get_wcs(wcs_url)

    # Iterate through all coverages available in the WCS contents
    for coverage_id in wcs.contents:
        # Get the metadata for the current coverage
        coverage_metadata = wcs.contents[coverage_id]

        # Get the title of the coverage for file naming
        coverage_title = coverage_metadata.title

        # Filter coverages based on the presence of specific substrings
        if (filter_layers is not None) and not any(layer in coverage_title for layer in filter_layers):
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

        os.makedirs(output_folder, exist_ok=True)

        # Define the output filename based on the coverage title and data type
        output_filename = os.path.join(output_folder, f"{coverage_title}.tif")
        tmp_filename = os.path.join(output_folder, f"{coverage_title}_tmp.tif")
        # Save the downloaded data as a GeoTIFF
        with open(tmp_filename, "wb") as file:
            file.write(response.read())

        # Reproject the raster to the catchment CRS and resolution
        reproject_raster(tmp_filename, crs, output_filename)
        os.remove(tmp_filename)

        logger.info(f"Downloaded {label} data saved as {coverage_title}.tif")

def get_wcs(url):
    return retry(lambda:WebCoverageService(url, version="1.0.0"))

