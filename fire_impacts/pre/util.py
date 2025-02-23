import geopandas as gpd
import rasterio as rio
import numpy as np
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def clip_and_reproject_raster(raster_file:str, shapefile:str, output_file:str, target_resolution:float=None):
    """
    Clips a raster file using a shapefile and reprojects the clipped raster to the CRS of the shapefile.

    Parameters:
    - raster_file (str): Path to the input raster file.
    - shapefile (str): Path to the shapefile for clipping.
    - output_files (list): List of paths to the output reprojected raster files.
    - target_resolution (tuple): OPTIONAL: Desired resolution for the output rasters. Default to automatic selection of resolution.
    """
    # Read the raster file to get its CRS and resolution
    with rio.open(raster_file) as src:
        raster_crs = src.crs
        raster_res = src.res  # Get the resolution of the input raster in original CRS

    # Read the shapefile
    catchment = gpd.read_file(shapefile)
    # Get the CRS of the shapefile
    shapefile_crs = catchment.crs.to_string()
    # Ensure the shapefile is in the same CRS as the raster before clipping
    catchment = catchment.to_crs(raster_crs).buffer(raster_res[0]*2)  # Buffer the shapefile by 2 pixels to ensure it covers the raster
    # Read the raster file
    with rio.open(raster_file) as src:
        # Clip the raster with the shapefile
        out_image, out_transform = mask(src, catchment.geometry.apply(mapping), crop=True,all_touched=True,pad=False)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "crs": src.crs})

    # Write the clipped raster to a temporary file
    temp_file = 'clipped_temp.tif'
    with rio.open(temp_file, 'w', **out_meta) as dest:
        dest.write(out_image)

    reproject_raster(temp_file, shapefile_crs, output_file, target_resolution)
    # Clean up temporary file
    os.remove(temp_file)

def reproject_raster(temp_file:str, target_crs:str, output_file:str, target_resolution:float=None):
    # Reproject the clipped raster to the CRS of the shapefile with the target resolution
    with rio.open(temp_file) as src:
        logger.info(f'Reprojecting raster from %s to %s', src.crs, target_crs)
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=target_resolution)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'res': target_resolution  # Set the target resolution explicitly
        })

        with rio.open(output_file, 'w', **kwargs) as dst:
            logger.info(f'Reprojecting clipped raster to: {output_file}')
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

def read_raster(fn:str):
  with rio.open(fn) as src:
    return src.read(1), src.transform, src.crs

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
