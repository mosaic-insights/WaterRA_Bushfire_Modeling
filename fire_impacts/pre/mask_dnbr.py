"""
mask_dnbr.py

Mask an existing dNBR raster so it only retains values where DEA Land Cover
is class 112: Natural Terrestrial Vegetation (NTV).

Workflow (same simple logic as src/chm/dea_landuse.py):
1) Download DEA annual mosaic (full GeoTIFF) to a cache folder (only once).
2) Clip the DEA mosaic locally to the dNBR extent (bbox polygon) using rasterio.mask.
3) Reproject the clipped DEA to match the dNBR grid (nearest neighbour).
4) Mask dNBR: keep only where DEA == natural_code, else set to 0.
5) Save masked dNBR and the clipped DEA into the FireSeverity folder.

Outputs (per catchment) saved into FireSeverity:
- masked_dNBR.tif
- DEA_LC_<year>_<level>_clipped.tif
(and a cache copy of the raw national DEA tif in FireSeverity/_dea_cache/)
"""

from __future__ import annotations

# ---------- stdlib ----------
import os
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# ---------- third-party ----------
import requests
import numpy as np
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import box

import rasterio as rio
from rasterio.mask import mask as rio_mask

# ---------- local ----------
from .project import FireImpactsProject
from .data_sources import DEA_LANDCOVER

logger = logging.getLogger(__name__)

DEFAULT_DEA_LEVEL = "level3"
DEA_FALLBACK_NODATA = 255
DEA_CACHE_SUBDIR = "_dea_cache"


# ======================================================================================
# Helpers (same style as dea_landuse.py)
# ======================================================================================

def _url_exists(url: str, timeout: int = 30) -> bool:
    """Lightweight existence check (no full download)."""
    headers = {"User-Agent": "python-requests", "Range": "bytes=0-0"}
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=timeout)
        return r.status_code in (200, 206)
    except requests.RequestException:
        return False


def _find_latest_available_dea_year(
    dea_level: str,
    start_year: Optional[int],
    lookback: int,
) -> int:
    """
    Probe backwards to find the latest year that exists on the server.

    Example:
      start_year=2025, lookback=6 -> try 2025..2019
    """
    if start_year is None:
        start_year = datetime.now().year - 1

    for y in range(start_year, start_year - lookback - 1, -1):
        fname = f"ga_ls_landcover_class_cyear_3_mosaic_{y}--P1Y_{dea_level}.tif"
        url = f"{DEA_LANDCOVER}/{y}--P1Y/{fname}"
        logger.info("[probe] DEA LC year %s -> %s", y, url)
        if _url_exists(url):
            logger.info("[OK] Found available DEA Land Cover year: %s (level=%s)", y, dea_level)
            return y

    raise RuntimeError(
        f"Could not find an available DEA mosaic for level={dea_level} "
        f"trying years {start_year} back to {start_year - lookback}."
    )


def _stream_download(url: str, out_fp: str, timeout: int = 180, chunk: int = 1024 * 1024) -> None:
    """
    Streaming download (same pattern as dea_landuse.py).
    Uses a temporary .part file then renames (avoid partial files looking valid).
    """
    out_fp = str(out_fp)
    tmp_fp = out_fp + ".part"

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    headers = {"User-Agent": "python-requests"}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_fp, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)

    # Replace atomically-ish
    if os.path.exists(out_fp):
        os.remove(out_fp)
    os.rename(tmp_fp, out_fp)


def _valid_tif(fp: str) -> bool:
    """Quick sanity check: exists, non-empty, rasterio can open."""
    try:
        if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
            return False
        with rio.open(fp) as src:
            _ = src.count
        return True
    except Exception:
        return False


def _build_bbox_polygon_gdf(bounds: Tuple[float, float, float, float], crs) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame polygon from raster bounds."""
    geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)


def _clip_raster_to_geom(
    raster_path: str,
    geom_gdf: gpd.GeoDataFrame,
    out_path: str,
) -> str:
    """
    Clip raster_path to geom_gdf and write to out_path (crop=True).
    This is exactly the same style as dea_landuse.py.
    """
    with rio.open(raster_path) as src:
        geom_in = geom_gdf.to_crs(src.crs)
        geoms = [g for g in geom_in.geometry if g is not None and (not g.is_empty)]
        if not geoms:
            raise RuntimeError("No valid geometry to clip against.")

        nd = src.nodata if src.nodata is not None else DEA_FALLBACK_NODATA

        img, tr = rio_mask(
            src,
            geoms,
            crop=True,
            nodata=nd,
            filled=True,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "height": img.shape[1],
                "width": img.shape[2],
                "transform": tr,
                "nodata": nd,
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rio.open(out_path, "w", **meta) as dst:
        dst.write(img)

    return out_path


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
      - DEA_LC_<year>_<level>_clipped.tif
    """

    # Optional: silence our module logs (does NOT silence rasterio/gdal logs)
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
    # Find latest available DEA year
    # -------------------------------
    latest_year = _find_latest_available_dea_year(
        dea_level=dea_level,
        start_year=dea_start_year,
        lookback=dea_lookback,
    )

    remote_fname = f"ga_ls_landcover_class_cyear_3_mosaic_{latest_year}--P1Y_{dea_level}.tif"
    dea_url = f"{DEA_LANDCOVER}/{latest_year}--P1Y/{remote_fname}"

    # -------------------------------
    # Download raw DEA (cached)
    # -------------------------------
    cache_dir = os.path.join(sev_folder, DEA_CACHE_SUBDIR)
    raw_dea_path = os.path.join(cache_dir, remote_fname)

    if not _valid_tif(raw_dea_path):
        logger.info("[download] %s -> %s", dea_url, raw_dea_path)
        _stream_download(dea_url, raw_dea_path, timeout=180, chunk=1024 * 1024)
        logger.info("[done] Downloaded DEA file: %s", raw_dea_path)
    else:
        logger.info("[cache] Using existing DEA file: %s", raw_dea_path)

    # -------------------------------
    # Clip DEA to dNBR bbox (saved!)
    # -------------------------------
    dea_clipped_name = f"DEA_LC_{latest_year}.tif"
    dea_clipped_path = os.path.join(sev_folder, dea_clipped_name)

    if not _valid_tif(dea_clipped_path):
        logger.info("[clip] Writing clipped DEA: %s", dea_clipped_path)
        _clip_raster_to_geom(raw_dea_path, dnbr_bbox_gdf, dea_clipped_path)
        logger.info("[done] Clipped DEA saved: %s", dea_clipped_path)
    else:
        logger.info("[cache] Using existing clipped DEA: %s", dea_clipped_path)

    # -------------------------------
    # Read clipped DEA and match to dNBR grid
    # -------------------------------
    dea_da = rxr.open_rasterio(dea_clipped_path, masked=True).squeeze()

    # Ensure nodata is set
    try:
        if dea_da.rio.nodata is None:
            dea_da = dea_da.rio.write_nodata(DEA_FALLBACK_NODATA)
    except Exception:
        pass

    # Reproject to match dNBR grid (nearest for categorical classes)
    dea_match = dea_da.rio.reproject_match(dnbr, resampling=rio.warp.Resampling.nearest)

    # -------------------------------
    # Mask: keep only natural_code
    # -------------------------------
    nd = dea_match.rio.nodata
    keep = (dea_match == natural_code)
    if nd is not None:
        keep = keep & (dea_match != nd)

    dnbr_masked = xr.where(keep, dnbr, 0).astype(dnbr.dtype)
    dnbr_masked = dnbr_masked.rio.write_crs(dnbr_crs)

    # Keep original dNBR nodata if present
    try:
        dnbr_nd = dnbr.rio.nodata
        if dnbr_nd is not None:
            dnbr_masked = dnbr_masked.rio.write_nodata(dnbr_nd)
    except Exception:
        pass

    out_path = os.path.join(sev_folder, "masked_dNBR.tif")
    logger.info("[write] %s", out_path)
    dnbr_masked.rio.to_raster(out_path, compress="deflate")
    logger.info("[OK] masked_dNBR saved: %s", out_path)