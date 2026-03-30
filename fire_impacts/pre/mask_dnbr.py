"""
mask_dnbr.py

Mask an existing dNBR raster so it only retains values where DEA Land Cover
is class 112: Natural Terrestrial Vegetation (NTV).

Outputs (per catchment) saved into FireSeverity:
- masked_dNBR.tif
- DEA_LC_<year>_<level>_match_dNBR.tif   (projected/aligned to dNBR grid)
"""

from __future__ import annotations

# ---------- stdlib ----------
import os
import logging
from datetime import datetime
from typing import Optional, Tuple

# ---------- third-party ----------
import requests
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import box

import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask

# ---------- local ----------
from .project import FireImpactsProject
from .data_sources import DEA_LANDCOVER

logger = logging.getLogger(__name__)

DEFAULT_DEA_LEVEL = "level3"
DEA_FALLBACK_NODATA = 255


# ======================================================================================
# Helpers
# ======================================================================================

def _build_bbox_polygon_gdf(bounds: Tuple[float, float, float, float], crs) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame polygon from raster bounds."""
    geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)


def _find_latest_dea_url(
    dea_level: str,
    start_year: Optional[int],
    lookback: int,
    timeout: int = 30,
) -> Tuple[int, str]:
    """
    Find the URL of the latest available DEA Land Cover mosaic.
    Tries start_year first, then falls back to previous years using HEAD requests.

    Returns (year, url)
    """
    if start_year is None:
        start_year = datetime.now().year - 1

    last_err: Exception | None = None

    for y in range(start_year, start_year - lookback - 1, -1):
        remote_fname = f"ga_ls_landcover_class_cyear_3_mosaic_{y}--P1Y_{dea_level}.tif"
        url = f"{DEA_LANDCOVER}/{y}--P1Y/{remote_fname}"

        try:
            resp = requests.head(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            logger.info("[found] DEA Land Cover year=%s: %s", y, url)
            return y, url
        except requests.HTTPError as e:
            last_err = e
            logger.warning("[HTTP] year=%s not available (%s). Trying older year...", y, e)
            continue
        except requests.RequestException as e:
            last_err = e
            logger.warning("[Network] year=%s failed (%s). Trying older year...", y, e)
            continue

    raise RuntimeError(
        f"Could not find any DEA mosaic for level={dea_level} "
        f"trying years {start_year} back to {start_year - lookback}. "
        f"Last error: {last_err}"
    )


def _clip_raster_to_geom_in_memory(
    raster_path: str,
    geom_gdf: gpd.GeoDataFrame,
    fallback_nodata: int = DEA_FALLBACK_NODATA,
) -> rxr.rioxarray.RasterArray:
    """
    Clip raster_path to geom_gdf using rasterio.mask, but keep the result in memory
    (no on-disk clipped GeoTIFF).

    Returns a rioxarray DataArray (squeezable later).
    """
    with rio.open(raster_path) as src:
        # Reproject the clipping geometry to the raster CRS
        geom_in = geom_gdf.to_crs(src.crs)

        geoms = [g for g in geom_in.geometry if g is not None and (not g.is_empty)]
        if not geoms:
            raise RuntimeError("No valid geometry to clip against.")

        nd = src.nodata if src.nodata is not None else fallback_nodata

        # Clip/crop
        img, tr = rio_mask(
            src,
            geoms,
            crop=True,
            nodata=nd,
            filled=True,
        )

        # Build an in-memory GeoTIFF so rioxarray can open it cleanly with CRS/transform
        meta = src.meta.copy()
        meta.update(
            {
                "height": img.shape[1],
                "width": img.shape[2],
                "transform": tr,
                "nodata": nd,
            }
        )

        with MemoryFile() as memfile:
            with memfile.open(**meta) as dst:
                dst.write(img)
            # Re-open with rioxarray from the in-memory bytes
            with memfile.open() as memsrc:
                da = rxr.open_rasterio(memsrc, masked=True)

    return da


# ======================================================================================
# Main
# ======================================================================================

def mask_dnbr(
    project: FireImpactsProject,
    catchment: Optional[str] = None,
    dea_level: str = DEFAULT_DEA_LEVEL,
    dea_start_year: Optional[int] = None,
    dea_lookback: int = 6,
    natural_code: int = 112,
    quiet: bool = False,
) -> None:
    """
    Mask an existing dNBR raster so only pixels overlapping DEA class natural_code remain.

    Saves to each catchment FireSeverity folder:
      - masked_dNBR.tif
      - DEA_LC_<year>.tif   (projected/aligned to dNBR grid)

    The DEA Land Cover mosaic is accessed remotely via HTTP range requests
    (GDAL vsicurl), so only the blocks covering the catchment are downloaded.
    """
    if quiet:
        logger.setLevel(logging.WARNING)

    # If a catchment is not specified, run for each catchment.
    if catchment is None:
        return project.for_each_catchment(
            lambda c: mask_dnbr(
                project=project,
                catchment=c,
                dea_level=dea_level,
                dea_start_year=dea_start_year,
                dea_lookback=dea_lookback,
                natural_code=natural_code,
                quiet=quiet,
            )
        )

    # -------------------------------
    # Locate inputs
    # -------------------------------
    sev_folder = project.catchment_path(catchment, "FireSeverity")
    os.makedirs(sev_folder, exist_ok=True)

    dnbr_path = os.path.join(sev_folder, "dNBR.tif")
    if not os.path.exists(dnbr_path):
        raise FileNotFoundError(
            f"dNBR raster not found for catchment='{catchment}'. Expected: {dnbr_path}\n"
            "Run severity.calculate_fire_severity(...) first."
        )

    # Read dNBR
    dnbr = rxr.open_rasterio(dnbr_path, masked=True).squeeze()
    if dnbr.rio.crs is None:
        raise RuntimeError(f"dNBR has no CRS. Please ensure {dnbr_path} has a valid CRS.")

    dnbr_crs = dnbr.rio.crs
    dnbr_bounds = dnbr.rio.bounds()
    dnbr_bbox_gdf = _build_bbox_polygon_gdf(dnbr_bounds, dnbr_crs)

    logger.info("mask_dnbr(): catchment=%s", catchment)
    logger.info("mask_dnbr(): dNBR CRS=%s", dnbr_crs)
    logger.info("mask_dnbr(): dNBR bounds=%s", dnbr_bounds)

    # -------------------------------
    # Find latest DEA URL and read only the relevant window via HTTP range requests
    # -------------------------------
    latest_year, dea_url = _find_latest_dea_url(
        dea_level=dea_level,
        start_year=dea_start_year,
        lookback=dea_lookback,
    )

    # Clip DEA to dNBR bbox (reads only needed blocks via GDAL vsicurl)
    dea_da = _clip_raster_to_geom_in_memory(
        raster_path=dea_url,
        geom_gdf=dnbr_bbox_gdf,
        fallback_nodata=DEA_FALLBACK_NODATA,
    ).squeeze()

    # Ensure nodata is set (important for categorical masking)
    try:
        if dea_da.rio.nodata is None:
            dea_da = dea_da.rio.write_nodata(DEA_FALLBACK_NODATA)
    except Exception:
        pass

    # Reproject to match dNBR grid (nearest for categorical classes)
    dea_match = dea_da.rio.reproject_match(
        dnbr,
        resampling=rio.warp.Resampling.nearest,
    )

    # Save ONLY the projected/aligned DEA
    dea_match_path = os.path.join(sev_folder, f"DEA_LC_{latest_year}.tif")
    logger.info("[write] %s", dea_match_path)
    dea_match.rio.to_raster(dea_match_path, compress="deflate")
    logger.info("[OK] Aligned DEA saved: %s", dea_match_path)

    # -------------------------------
    # Mask: keep only natural_code
    # -------------------------------
    nd = dea_match.rio.nodata
    keep = (dea_match == natural_code)
    if nd is not None:
        keep = keep & (dea_match != nd)

    # Set non-vegetation pixels to NaN so they are excluded from
    # downstream erosion calculations (e.g. water bodies inside the
    # catchment boundary).
    dnbr_masked = xr.where(keep, dnbr, float('nan')).astype('float32')
    dnbr_masked = dnbr_masked.rio.write_crs(dnbr_crs)
    dnbr_masked = dnbr_masked.rio.write_nodata(float('nan'))

    out_path = os.path.join(sev_folder, "masked_dNBR.tif")
    logger.info("[write] %s", out_path)
    dnbr_masked.rio.to_raster(out_path, compress="deflate")
    logger.info("[OK] masked_dNBR saved: %s", out_path)