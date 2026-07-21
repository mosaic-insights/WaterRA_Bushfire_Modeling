"""
Low-level raster and WCS utility functions for pre-processing.

Functions here are mostly format-conversion helpers — clipping,
reprojecting, and reading rasters — used by the higher-level
pre-processing modules.
"""

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


# ---------------------------------------------------------------------------
# Raster clip and reproject helpers
# ---------------------------------------------------------------------------

def clip_raster(
    raster_file: str,
    shapefile: str,
):
    """
    Clip a raster to a shapefile (buffered by 2 pixels) in the raster
    CRS and write the result to a temporary file.

    Parameters:
    - raster_file: path to the input raster.
    - shapefile: path to the shapefile to clip against.

    Returns:
    - temp_file: path to the temporary clipped GeoTIFF on disk.
    - shapefile_crs: CRS string from the shapefile.
    """
    # Read the raster file to get its CRS and resolution
    with rio.open(raster_file) as src:
        raster_crs = src.crs
        raster_res = src.res

    # Read the shapefile
    catchment = gpd.read_file(shapefile)
    # Get the CRS of the shapefile
    shapefile_crs = catchment.crs.to_string()
    # Reproject the shapefile to the raster CRS and buffer by 2 pixels
    # to ensure it fully covers the raster at the clip boundary.
    catchment = catchment.to_crs(raster_crs).buffer(raster_res[0] * 2)

    # Read the raster file and clip to the buffered catchment boundary
    with rio.open(raster_file) as src:
        out_image, out_transform = mask(
            src,
            catchment.geometry.apply(mapping),
            crop=True,
            all_touched=True,
            nodata=np.nan,
            pad=False,
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs,
            "nodata": np.nan,
        })

    # Write the clipped raster to a temporary file
    temp_file = "clipped_temp.tif"
    with rio.open(temp_file, "w", **out_meta) as dest:
        dest.write(out_image)

    return temp_file, shapefile_crs


def clip_and_reproject_raster(
    raster_file: str,
    shapefile: str,
    output_file: str,
    target_resolution: float = None,
):
    """
    Clip a raster to a shapefile and reproject it to the shapefile's
    CRS.

    Parameters:
    - raster_file: path to the input raster file.
    - shapefile: path to the shapefile for clipping.
    - output_file: path to the output reprojected raster file.
    - target_resolution: target pixel resolution; defaults to automatic
      selection.

    Returns:
    - None.  Writes the reprojected raster to output_file.
    """
    # Clip the raster to the slightly-buffered extent of the shapefile
    temp_file, shapefile_crs = clip_raster(raster_file, shapefile)

    # Reproject the raster to the shapefile's CRS
    reproject_raster(temp_file, shapefile_crs, output_file, target_resolution)

    # Clean up temporary file
    os.remove(temp_file)


def reproject_raster(
    temp_file: str,
    target_crs: str,
    output_file: str,
    target_resolution: float = None,
):
    """
    Reproject a raster to a target CRS and write it to disk.

    Parameters:
    - temp_file: path to the input raster to reproject.
    - target_crs: CRS string for the output raster.
    - output_file: path for the reprojected output GeoTIFF.
    - target_resolution: target pixel resolution; defaults to automatic
      selection based on the input transform.

    Returns:
    - None.  Writes the reprojected raster to output_file.
    """
    with rio.open(temp_file) as src:
        logger.info(
            "Reprojecting raster from %s to %s", src.crs, target_crs
        )
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_resolution,
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "res": target_resolution,
        })

        with rio.open(output_file, "w", **kwargs) as dst:
            logger.info("Reprojecting clipped raster to: %s", output_file)
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )


def read_raster(path_to_file: str):
    """
    Read a single-band raster and return its data and metadata.

    Parameters:
    - path_to_file: path to the raster file.

    Returns:
    - data: 2-D numpy array of band 1 values.
    - meta: rasterio metadata dict (includes 'crs' and 'transform').
    """
    with rio.open(path_to_file) as src:
        return (src.read(1), src.meta)


def read_aligned(
    raster_fn: str,
    transform,
    crs,
    shape,
    resampling=Resampling.nearest,
):
    """
    Read a raster and reproject it onto a given grid in memory.

    Parameters:
    - raster_fn: path to the source raster file.
    - transform: affine transform defining the target grid.
    - crs: target CRS.
    - shape: (rows, cols) of the target grid.
    - resampling: rasterio Resampling method (default nearest).

    Returns:
    - 2-D numpy array reprojected onto the target grid; NoData pixels,
      and any part of the target grid the source does not cover, are
      NaN.
    ------------------------------------------------------------------------
    Notes:
    - The destination is always float with NaN as its nodata, rather
      than inheriting whatever the source declares. Inheriting means a
      source carrying no nodata tag gets reproject's default 0 fill,
      and the read-back mask is then empty, so uncovered area comes back
      as real zeros. That is silent: a 0 in a C or K factor is a valid
      value meaning "no erosion here", where a NaN propagates visibly
      through the KLSCP multiplication.
    - Integer sources are promoted to float32 for the same reason (NaN
      is not representable in an integer band) and a warning is logged.
      The function has always returned NaN-filled data, so a float
      return was already implied by the contract.
    - This function is for continuous data. Categorical and topological
      grids belong in read_raster, which reads at native dtype without
      regridding. In particular a D8 flow direction grid must never be
      reprojected: the codes name a neighbouring cell in the source
      grid's own geometry, and reprojection changes which cell is the
      neighbour. Nearest-neighbour resampling carries the code values
      across intact, so the result looks like a valid dirmap while
      pointing at the wrong cells. Reproject the DEM and recompute flow
      direction from the reprojected DEM instead.
    ------------------------------------------------------------------------
    """
    logger.info("Reading raster %s and reprojecting to %s", raster_fn, crs)
    with rio.open(raster_fn) as src:
        kwargs = src.meta.copy()

        # Keep float64 sources at float64; promote anything else (int
        # categoricals, bytes) to float32 so NaN can be represented.
        dtype = kwargs.get("dtype")
        if not np.issubdtype(np.dtype(dtype), np.floating):
            logger.warning(
                "read_aligned promoting %s from %s to float32 so NoData "
                "can be NaN. read_aligned is for continuous data - if "
                "this is a categorical or topological grid, read it with "
                "read_raster instead. A D8 flow direction grid in "
                "particular must not be reprojected: the codes name a "
                "neighbour in the source grid's geometry, so the values "
                "survive resampling while no longer pointing at the "
                "right cells. Reproject the DEM and recompute flow "
                "direction from it.",
                raster_fn, dtype,
            )
            dtype = "float32"

        kwargs.update({
            "crs": crs,
            "transform": transform,
            "width": shape[1],
            "height": shape[0],
            "dtype": dtype,
            "nodata": np.nan,
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
                    resampling=resampling,
                )

            with memfile.open() as src:
                data = src.read(1, masked=True)
                data[data.mask] = np.nan
                return data


# ---------------------------------------------------------------------------
# Unit conversion helper
# ---------------------------------------------------------------------------

def metres_to_approx_degrees(m: float):
    """Convert a distance in metres to approximate decimal degrees."""
    return m * c.M_TO_KM / APPROX_KM_PER_DEGREE


# ---------------------------------------------------------------------------
# WCS download helpers
# ---------------------------------------------------------------------------

def retrieve_grid_from_wcs_for_bounds(
    label,
    wcs_url,
    bbox,
    resx,
    resy,
    crs,
    output_folder,
    filter_layers=None,
):
    """
    Download coverage layers from a WCS service for a bounding box.

    Iterates over all coverages in the WCS catalogue (optionally
    filtered by layer-title substrings), requests each one as a
    GeoTIFF, reprojects it to the target CRS and resolution, and
    saves the result to output_folder.

    Parameters:
    - label: human-readable label used in log messages.
    - wcs_url: URL of the WCS service.
    - bbox: bounding box tuple (minx, miny, maxx, maxy) in EPSG:4326.
    - resx: x resolution for the coverage request.
    - resy: y resolution for the coverage request.
    - crs: target CRS string for reprojection.
    - output_folder: directory to write the downloaded GeoTIFFs.
    - filter_layers: optional list of title substrings; only coverages
      whose title contains at least one substring are downloaded.

    Returns:
    - None.  Writes reprojected GeoTIFFs to output_folder.
    """
    # Connect to the WCS server for each dataset
    wcs = get_wcs(wcs_url)

    # Iterate through all coverages available in the WCS contents
    for coverage_id in wcs.contents:
        coverage_metadata = wcs.contents[coverage_id]
        coverage_title = coverage_metadata.title

        # Filter coverages based on the presence of specific substrings
        if filter_layers is not None:
            if not any(layer in coverage_title for layer in filter_layers):
                continue

        # Request the coverage data using the GetCoverage operation
        response = wcs.getCoverage(
            identifier=coverage_id,
            bbox=bbox,
            format="GeoTIFF",
            crs="EPSG:4326",
            resx=resx,
            resy=resy,
        )

        os.makedirs(output_folder, exist_ok=True)

        # Define output filenames based on the coverage title
        output_filename = os.path.join(
            output_folder, f"{coverage_title}.tif"
        )
        tmp_filename = os.path.join(
            output_folder, f"{coverage_title}_tmp.tif"
        )

        # Save the downloaded data as a GeoTIFF
        with open(tmp_filename, "wb") as file:
            file.write(response.read())

        # Reproject to the catchment CRS and resolution
        reproject_raster(tmp_filename, crs, output_filename)
        os.remove(tmp_filename)

        logger.info(
            "Downloaded %s data saved as %s.tif", label, coverage_title
        )


def get_wcs(url):
    """Connect to a WCS service with retry logic and return the client."""
    return retry(lambda: WebCoverageService(url, version="1.0.0"))
