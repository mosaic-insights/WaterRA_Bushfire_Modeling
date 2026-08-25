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
    elif np.isnan(nodata):
        # A NaN nodata never equals itself, so `data == nodata` would
        # match nothing and report an empty mask even though the nodata
        # cells are already NaN in the array. Detect them with isnan
        # instead. (This is the same trap that made pysheds treat the
        # catchment boundary as valid terrain — see condition_dem.)
        nodata_mask = np.isnan(data)
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


def read_dnbr_aligned_like(path, template, **kwargs):
    """
    Read a dNBR raster onto a template grid, on the conventional scale.

    dNBR is stored as the raw band-ratio difference (~[0, 1]) but every
    threshold in the package is quoted on the 0-1000 scale, so the factor
    is applied here rather than at each call site — see const.DNBR_SCALE.

    Parameters:
    - path: dNBR raster (dNBR.tif or masked_dNBR.tif).
    - template: RasterGrid to align to.

    Returns:
    - 2-D array on the conventional 0-1000 scale.
    """
    return read_aligned_like(path, template, **kwargs) * c.DNBR_SCALE


def read_dnbr_aligned(path, transform, crs, shape, **kwargs):
    """
    Read a dNBR raster onto an explicit grid, on the conventional scale.

    The transform/crs/shape counterpart of read_dnbr_aligned_like, for
    callers that hold a target grid rather than a RasterGrid.

    Returns:
    - 2-D array on the conventional 0-1000 scale.
    """
    return read_aligned(path, transform, crs, shape, **kwargs) \
        * c.DNBR_SCALE


def to_dnbr_scale(values):
    """
    Convert stored dNBR values to the conventional 0-1000 scale.

    For callers that already hold an array or Series (zonal statistics,
    for instance) rather than a path.
    """
    return values * c.DNBR_SCALE


def from_dnbr_scale(values):
    """
    Convert conventional-scale dNBR back to the stored representation.

    Used by producers that source values already on the 0-1000 scale
    (the synthetic-fire reference rasters), so that everything written to
    masked_dNBR.tif shares one convention.
    """
    return values / c.DNBR_SCALE


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


# pysheds' D8 flowdir marks a cell with no downhill neighbour as a pit
# with this sentinel; residual pits after conditioning carry this code.
PYSHEDS_PIT_CODE = -2
# Per-cell drop used to guarantee a strictly descending breach channel
# across otherwise-flat valley floors (metres).
BREACH_EPSILON = 1e-3


def _dem_with_real_nodata(dem, sentinel=None):
    """
    Return a pysheds Raster whose nodata is a real (matchable) value.

    pysheds identifies nodata cells with the test ``dem == nodata``. When
    the DEM's nodata is NaN this matches nothing (``NaN != NaN``), so
    pysheds treats every nodata cell — including the whole catchment
    boundary of a clipped DEM — as valid terrain. Flow then dead-ends
    against the boundary instead of draining out of it, leaving a rash of
    spurious pits hugging the perimeter.

    Replace NaN nodata with a real sentinel below the DEM range and return
    a Raster carrying that sentinel so pysheds recognises the boundary.
    A DEM that already has a finite, matchable nodata is returned
    unchanged.

    Crucially the returned viewfinder also carries a **mask** (True on the
    valid cells). The sentinel alone makes the boundary matchable, but a
    finite sentinel value is also routable: without a mask pysheds treats
    the sentinel-filled nodata region as one huge low flat and
    resolve_flats drains it, spraying spurious streams/headwaters across
    the missing-data area outside the catchment. The mask marks those
    cells out-of-domain so they are excluded from filling, routing and
    accumulation entirely.
    """
    from pysheds.view import Raster, ViewFinder

    arr = np.asarray(dem)
    vf = dem.viewfinder
    nodata = vf.nodata
    valid = np.isfinite(arr)
    nan_cells = ~valid
    existing_mask = np.asarray(vf.mask) if vf.mask is not None else None
    mask_covers_nodata = (
        existing_mask is not None and not existing_mask[nan_cells].any()
    )
    if (nodata is not None and np.isfinite(nodata) and not nan_cells.any()
            and (existing_mask is None or mask_covers_nodata)):
        return dem

    if sentinel is None:
        finite = arr[valid]
        low = float(finite.min()) if finite.size else 0.0
        sentinel = low - 1000.0
    sentinel = np.dtype(dem.dtype).type(sentinel)

    new_arr = np.where(nan_cells, sentinel, arr).astype(dem.dtype)
    new_vf = ViewFinder(
        affine=vf.affine, shape=new_arr.shape,
        nodata=sentinel, crs=vf.crs, mask=valid,
    )
    return Raster(new_arr, viewfinder=new_vf)


def breach_pits(grid, dem, max_search=None):
    """
    Prototype: least-cost breaching of residual pits.

    Depression *filling* raises a basin to its spill elevation. On a
    near-flat valley floor (e.g. a hydrologically-enforced DEM smoothed by
    bilinear reprojection) float32 precision leaves the spill cell a hair
    proud of its own filled flat, so pysheds still flags it as a pit and
    the trunk stream dead-ends there. Breaching instead *cuts* an outlet:
    for every cell D8 flow direction flags as a pit, carve a monotonically
    descending channel to the nearest lower cell (or nodata/edge),
    lowering only the intervening cells. This preserves the valley-floor
    elevations that feed the downslope slope/connectivity terms rather
    than flooding them flat.

    Parameters:
    - grid: pysheds Grid whose viewfinder matches ``dem``.
    - dem: conditioned pysheds Raster (real nodata, see
      _dem_with_real_nodata).
    - max_search: cap on cells expanded per pit before giving up
      (defaults to the whole grid).

    Returns:
    - A new pysheds Raster with the breach channels carved in.
    """
    import heapq
    from pysheds.view import Raster, ViewFinder

    vf = dem.viewfinder
    arr = np.asarray(dem, dtype=np.float64).copy()
    height, width = arr.shape
    nodata = float(vf.nodata)
    nod = ~np.isfinite(arr) if np.isnan(nodata) else (arr == nodata)
    if max_search is None:
        max_search = height * width

    fdir = np.asarray(grid.flowdir(dem))
    pit_cells = [tuple(rc) for rc in
                 np.argwhere((fdir == PYSHEDS_PIT_CODE) & ~nod)]

    def _on_border(r, col):
        return r == 0 or col == 0 or r == height - 1 or col == width - 1

    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1))
    breached = 0
    for (pit_r, pit_c) in pit_cells:
        z0 = arr[pit_r, pit_c]
        # Dijkstra minimising the highest barrier crossed to reach an
        # outlet. An outlet is a valid cell below the pit (every genuine
        # depression has an interior spill point) or the raster edge,
        # where pysheds routes flow off the grid. Interior nodata is the
        # catchment wall, not a drain: pysheds won't route flow into it,
        # so the search treats it as impassable rather than carving
        # through the catchment boundary.
        best = {(pit_r, pit_c): z0}
        parent = {}
        frontier = [(z0, pit_r, pit_c)]
        outlet = None
        expanded = 0
        while frontier and expanded < max_search:
            cost, r, col = heapq.heappop(frontier)
            if cost > best.get((r, col), np.inf):
                continue
            if (r, col) != (pit_r, pit_c) and (arr[r, col] < z0
                                               or _on_border(r, col)):
                outlet = (r, col)
                break
            expanded += 1
            for d_r, d_c in neighbours:
                n_r, n_c = r + d_r, col + d_c
                if not (0 <= n_r < height and 0 <= n_c < width):
                    continue
                if nod[n_r, n_c]:      # catchment wall — impassable
                    continue
                nb_cost = max(cost, arr[n_r, n_c])
                if nb_cost < best.get((n_r, n_c), np.inf):
                    best[(n_r, n_c)] = nb_cost
                    parent[(n_r, n_c)] = (r, col)
                    heapq.heappush(frontier, (nb_cost, n_r, n_c))
        if outlet is None:
            logger.warning(
                "breach_pits: no outlet found for pit at (%d, %d)",
                pit_r, pit_c)
            continue

        # Reconstruct the pit -> outlet path and carve a strictly
        # descending channel along it.
        path = [outlet]
        while path[-1] != (pit_r, pit_c):
            path.append(parent[path[-1]])
        path.reverse()
        # Descend from the pit to the outlet; never above the pit level so
        # the carved channel is monotonic (BREACH_EPSILON guarantees a
        # strict drop even when the outlet is not lower, e.g. a raster
        # edge).
        z_out = min(arr[outlet], z0)
        span = z0 - z_out
        n = len(path)
        for i, (r, col) in enumerate(path):
            if nod[r, col]:
                continue
            frac = i / (n - 1) if n > 1 else 1.0
            carve = z0 - span * frac - BREACH_EPSILON * i
            if carve < arr[r, col]:
                arr[r, col] = carve
        breached += 1

    logger.info(
        "breach_pits: carved %d of %d residual pit(s)",
        breached, len(pit_cells))
    new_vf = ViewFinder(
        affine=vf.affine, shape=arr.shape, nodata=vf.nodata, crs=vf.crs)
    return Raster(arr.astype(dem.dtype), viewfinder=new_vf)


def condition_dem(dem_path: str, breach: bool = True,
                  max_breach_iters: int = 5):
    """
    Hydrologically condition a DEM: fill pits, fill depressions, and
    resolve flats.

    The single pysheds conditioning chain, used by
    topography.hydro_force_dem and dem_flow_layers.

    Parameters:
    - dem_path: path to the DEM raster.
    - breach: if True, follow the fill/resolve chain with least-cost
      breaching of any residual pits (see breach_pits). Off by default;
      fixing the NaN-nodata boundary already removes the great majority
      of pits, and breaching only targets the few genuine flat-valley
      pits that survive.
    - max_breach_iters: cap on breach+resolve passes when breach is True
      (carving one pit can expose a tied neighbour).

    Returns:
    - inflated_dem: pysheds Raster with pits, depressions, and flats
      resolved.
    - grid: pysheds Grid object for subsequent routing operations.
    """
    from pysheds.grid import Grid as _PyshedsGrid

    grid = _PyshedsGrid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    # Give pysheds a nodata value it can actually match, then route on a
    # grid that shares that viewfinder (otherwise the boundary is treated
    # as terrain and flow dead-ends against it).
    dem = _dem_with_real_nodata(dem)
    grid = _PyshedsGrid(viewfinder=dem.viewfinder)
    logger.info("Filling pits")
    fill_dem = grid.fill_pits(dem)
    logger.info("Filling depressions")
    flooded_dem = grid.fill_depressions(fill_dem)
    logger.info("Resolving flats")
    inflated_dem = grid.resolve_flats(flooded_dem)

    if breach:
        for i in range(max_breach_iters):
            fdir = np.asarray(grid.flowdir(inflated_dem))
            n_pits = int((fdir == PYSHEDS_PIT_CODE).sum())
            if not n_pits:
                break
            logger.info(
                "Breaching residual pits (pass %d, %d pit(s))",
                i + 1, n_pits)
            inflated_dem = breach_pits(grid, inflated_dem)
            inflated_dem = grid.resolve_flats(inflated_dem)

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


def upslope_weighted_mean(grid, fdir, weight_raster):
    """
    Mean of a weight over each cell's upslope contributing area, excluding
    cells whose weight is NaN.

    pysheds flow accumulation propagates a NaN weight to every downstream
    cell, so a single undefined weight nulls the whole downstream path —
    worst on the largest streams, which have the most upslope cells and
    thus the highest chance of a NaN somewhere above them. The classic
    trigger is slope on the one-cell halo of valid DEM cells bordering
    nodata (np.gradient needs all eight neighbours), or a factor raster
    that doesn't cover the full DEM extent.

    Accumulate the weight with NaN cells zeroed AND a companion count of
    the cells that actually contributed, then divide. The result is the
    exact mean of the valid upslope weights, with the dropped cells left
    out of the denominator rather than diluting it. Cells with no valid
    contributor come back NaN.

    Parameters:
    - grid: pysheds Grid used for the accumulation.
    - fdir: D8 flow direction raster on that grid.
    - weight_raster: pysheds Raster of per-cell weights (NaN where the
      weight is undefined).

    Returns:
    - 2-D float32 array of the upslope mean weight at each cell.
    """
    from pysheds.view import Raster

    vals = np.asarray(weight_raster, dtype='float64')
    missing = np.isnan(vals)
    vf = weight_raster.viewfinder
    weighted = Raster(np.where(missing, 0.0, vals), viewfinder=vf)
    counted = Raster(np.where(missing, 0.0, 1.0), viewfinder=vf)
    acc_w = np.array(
        grid.accumulation(fdir=fdir, weights=weighted), dtype=np.float32)
    acc_n = np.array(
        grid.accumulation(fdir=fdir, weights=counted), dtype=np.float32)
    return acc_w / np.where(acc_n == 0, np.nan, acc_n)


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
