"""
Low-level raster and WCS utility functions for pre-processing.

Functions here are mostly format-conversion helpers — clipping,
reprojecting, and reading rasters — used by the higher-level
pre-processing modules.
"""

from dataclasses import dataclass

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
# In-memory raster container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RasterGrid:
    """
    A single-band raster in memory: data plus the spatial metadata needed
    to interpret or re-save it.

    Produced by read_raster_masked(), where nodata cells have already been
    replaced with NaN in `data`. `nodata_mask` records which cells matched
    the source's declared nodata value (all-False when the source declared
    none, or declared NaN — NaN never compares equal to itself, matching
    the long-standing `data == nodata` behaviour of the call sites this
    consolidates).
    """
    data: np.ndarray
    transform: object   # affine.Affine
    crs: object         # rasterio CRS
    nodata_mask: np.ndarray

    @property
    def shape(self):
        return self.data.shape

    @property
    def xres(self):
        """East-west pixel width in CRS units."""
        return self.transform[0]

    @property
    def yres(self):
        """North-south pixel height in CRS units (positive)."""
        return abs(self.transform[4])

    @property
    def pixel_area(self):
        return self.xres * self.yres

    def meta(self, dtype='float32', nodata=np.nan, **updates):
        """
        Build a rasterio metadata dict for writing a single-band GeoTIFF
        on this grid. Defaults describe the library's standard output
        profile (float32, NaN nodata).
        """
        out = {
            'driver': 'GTiff',
            'height': self.shape[0],
            'width': self.shape[1],
            'count': 1,
            'dtype': dtype,
            'crs': self.crs,
            'transform': self.transform,
            'nodata': nodata,
        }
        out.update(updates)
        return out


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

    # Write the clipped raster to a unique temporary file so concurrent
    # runs (e.g. parallel catchments) don't clobber each other.
    tmp = tempfile.NamedTemporaryFile(
        prefix="clipped_", suffix=".tif", delete=False,
    )
    temp_file = tmp.name
    tmp.close()
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


def read_raster_masked(path_to_file: str) -> RasterGrid:
    """
    Read a single-band raster with nodata replaced by NaN.

    Consolidates the open/read/`np.where(data == nodata, nan, data)`
    preamble repeated across the preprocessing modules. Sources whose
    declared nodata is already NaN come back unchanged (their nodata
    cells are already NaN in the data).

    Parameters:
    - path_to_file: path to the raster file.

    Returns:
    - RasterGrid with `data` (nodata as NaN), `transform`, `crs`, and
      `nodata_mask` (True where the raw data equalled the declared
      nodata value).
    """
    data, meta = read_raster(path_to_file)
    nodata = meta.get('nodata')
    if nodata is None:
        nodata_mask = np.zeros(data.shape, dtype=bool)
    else:
        nodata_mask = data == nodata
        data = np.where(nodata_mask, np.nan, data)
    return RasterGrid(
        data=data,
        transform=meta['transform'],
        crs=meta['crs'],
        nodata_mask=nodata_mask,
    )


def read_aligned_like(
    raster_fn: str,
    like: RasterGrid,
    resampling=Resampling.nearest,
):
    """
    Read a raster reprojected onto the grid of an existing RasterGrid.

    Convenience wrapper for read_aligned() that takes the target grid
    from `like` instead of a (transform, crs, shape) triple.
    """
    return read_aligned(
        raster_fn, like.transform, like.crs, like.shape,
        resampling=resampling,
    )


def write_raster(
    path: str,
    data,
    meta: dict,
    *,
    dtype='float32',
    nodata=np.nan,
    compress='lzw',
    **meta_updates,
):
    """
    Write a single-band GeoTIFF with the library's standard profile.

    Consolidates the repeated `meta.update(...)` + `rio.open(path, 'w')`
    blocks. The base metadata supplies the georeferencing (transform,
    crs, height, width); dtype, nodata, and compression are standardised
    here, and the data is cast to the output dtype on write.

    Parameters:
    - path: output GeoTIFF path (parent directory is created if needed).
    - data: 2-D array to write.
    - meta: base rasterio metadata dict (e.g. from the template raster,
      or RasterGrid.meta()).
    - dtype: output dtype (default float32).
    - nodata: output nodata value (default NaN).
    - compress: compression (default 'lzw'; pass None for uncompressed).
    - meta_updates: any further metadata overrides.

    Returns:
    - None. Writes the raster to path.
    """
    out_meta = dict(meta)
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'dtype': dtype,
        'nodata': nodata,
    })
    if compress is not None:
        out_meta['compress'] = compress
    out_meta.update(meta_updates)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Masked arrays (e.g. from read_aligned) must be filled with the
    # output nodata value, matching what rasterio's write() does with a
    # masked array. np.asarray alone would silently drop the mask and
    # expose whatever values sit under it.
    if np.ma.isMaskedArray(data):
        fill = out_meta['nodata']
        if fill is None:
            fill = data.fill_value
        data = data.filled(fill)

    with rio.open(path, 'w', **out_meta) as dst:
        dst.write(np.asarray(data).astype(out_meta['dtype']), 1)
    logger.debug('Wrote raster to %s', path)


def clip_raster_in_memory(
    raster_path: str,
    geom_gdf: gpd.GeoDataFrame,
    fallback_nodata=None,
):
    """
    Clip a raster to a geometry and return the result as a rioxarray
    DataArray, without writing anything to disk.

    Uses rasterio.mask to crop the raster, builds an in-memory GeoTIFF
    so that rioxarray gets the correct CRS and transform, then returns
    the result.

    Parameters:
    - raster_path: path or GDAL-readable URL to the source raster.
    - geom_gdf: GeoDataFrame whose geometry defines the clip extent
      (reprojected to the raster CRS internally).
    - fallback_nodata: nodata value to use if the source declares none.

    Returns:
    - rioxarray DataArray clipped to geom_gdf (squeeze separately if
      needed to remove the band dimension).
    """
    import rioxarray as rxr

    with rio.open(raster_path) as src:
        # Reproject the clipping geometry to the raster CRS
        geom_in = geom_gdf.to_crs(src.crs)

        geoms = [
            g for g in geom_in.geometry
            if g is not None and not g.is_empty
        ]
        if not geoms:
            raise RuntimeError("No valid geometry to clip against.")

        nd = src.nodata if src.nodata is not None else fallback_nodata

        img, tr = mask(
            src,
            geoms,
            crop=True,
            nodata=nd,
            filled=True,
        )

        # Build an in-memory GeoTIFF so rioxarray can open it with a
        # valid CRS and transform (rioxarray needs an open rasterio src).
        out_meta = src.meta.copy()
        out_meta.update({
            "height": img.shape[1],
            "width": img.shape[2],
            "transform": tr,
            "nodata": nd,
        })

        with rio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dst:
                dst.write(img)
            # Re-open with rioxarray from the in-memory bytes
            with memfile.open() as memsrc:
                da = rxr.open_rasterio(memsrc, masked=True)

    return da


# ---------------------------------------------------------------------------
# Terrain and flow-routing helpers
# ---------------------------------------------------------------------------

def slope_from_dem(dem_data: np.ndarray, xres: float, yres: float):
    """
    Compute terrain slope from a DEM array via central differences.

    The single implementation of the gradient -> slope-ratio math that
    was previously repeated in topography.dem_to_slope, rusle.compute_lsi
    and rusle.compute_sediment_delivery_ratio.

    Parameters:
    - dem_data: 2-D DEM array (nodata as NaN), in metres.
    - xres: east-west pixel size in metres.
    - yres: north-south pixel size in metres.

    Returns:
    - slope_ratio: rise/run slope (dimensionless).
    - dz_dx, dz_dy: the two np.gradient components (for aspect
      calculations).
    """
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)
    slope_ratio = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    return slope_ratio, dz_dx, dz_dy


def condition_dem(dem_path: str):
    """
    Hydrologically condition a DEM: fill pits, fill depressions, and
    resolve flats.

    The single pysheds conditioning chain, used by
    topography.hydro_force_dem and dem_flow_layers.

    Parameters:
    - dem_path: path to the DEM raster.

    Returns:
    - inflated_dem: pysheds Raster with pits, depressions, and flats
      resolved.
    - grid: pysheds Grid object for subsequent routing operations.
    """
    from pysheds.grid import Grid as _PyshedsGrid

    grid = _PyshedsGrid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    logger.info("Filling pits")
    fill_dem = grid.fill_pits(dem)
    logger.info("Filling depressions")
    flooded_dem = grid.fill_depressions(fill_dem)
    logger.info("Resolving flats")
    inflated_dem = grid.resolve_flats(flooded_dem)
    return inflated_dem, grid


def dem_flow_layers(
    dem_path: str,
    dirmap: tuple = c.D8_FLOW_DIRECTIONS,
    routing: str = c.FLOW_ROUTING_TYPE,
):
    """
    Condition a DEM and compute D8 flow direction and accumulation.

    Parameters:
    - dem_path: path to the DEM raster.
    - dirmap: D8 flow direction mapping tuple.
    - routing: pysheds routing method (default 'd8').

    Returns:
    - grid: pysheds Grid initialised from the DEM.
    - fdir: flow direction raster.
    - acc: flow accumulation raster.
    """
    inflated_dem, grid = condition_dem(dem_path)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)
    return grid, fdir, acc


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
