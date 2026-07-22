"""
Mask a dNBR raster to retain only naturally-vegetated pixels.

Uses the DEA Land Cover mosaic (class 112: Natural Terrestrial
Vegetation) to exclude non-vegetated areas from downstream erosion
calculations.  Outputs per-catchment masked_dNBR.tif and an aligned
DEA_LC_<year>.tif into each catchment's FireSeverity folder.
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

from ..context import RunContext  # noqa: F401  (used in type-only annotation)

import rasterio as rio

from .util import clip_raster_in_memory

# ---------- local ----------
from .data_sources import DEA_LANDCOVER

logger = logging.getLogger(__name__)

DEFAULT_DEA_LEVEL = "level3"
DEA_FALLBACK_NODATA = 255


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_bbox_polygon_gdf(
    bounds: Tuple[float, float, float, float],
    crs,
) -> gpd.GeoDataFrame:
    """Create a single-row GeoDataFrame polygon from raster bounds."""
    geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=crs)


def _find_latest_dea_url(
    dea_level: str,
    start_year: Optional[int],
    lookback: int,
    timeout: int = 10,
) -> Tuple[int, str]:
    """
    Find the URL of the latest available DEA Land Cover mosaic.

    Tries start_year first, then falls back year-by-year on HTTP 404.
    Fails immediately with an informative error for connectivity
    problems or unexpected HTTP status codes — those are not
    year-specific issues and retrying other years would not help.

    Parameters:
    - dea_level: DEA Land Cover level string (e.g. 'level3').
    - start_year: most recent year to try; defaults to current year - 1.
    - lookback: how many years back to try before giving up.
    - timeout: per-request connect/read timeout in seconds.

    Returns:
    - (year, url) tuple for the first available mosaic found.
    """
    if start_year is None:
        start_year = datetime.now().year - 1

    years_tried = []

    for y in range(start_year, start_year - lookback - 1, -1):
        remote_fname = (
            f"ga_ls_landcover_class_cyear_3_mosaic_{y}"
            f"--P1Y_{dea_level}.tif"
        )
        url = f"{DEA_LANDCOVER}/{y}--P1Y/{remote_fname}"
        years_tried.append(y)

        try:
            resp = requests.head(url, timeout=timeout, allow_redirects=True)
        except requests.ConnectionError as e:
            # The server is unreachable (host down, DNS failure, network
            # error). No point trying other years — they all use the same
            # host. Fail immediately with a clear message.
            raise RuntimeError(
                f"Could not connect to the DEA Land Cover server. "
                f"Check your internet connection and try again later.\n"
                f"  Host: {DEA_LANDCOVER}\n"
                f"  Detail: {e}"
            ) from e
        except requests.Timeout as e:
            # The server did not respond within the timeout. Same logic as
            # ConnectionError — retrying other years won't help.
            raise RuntimeError(
                f"Connection to the DEA Land Cover server timed out "
                f"(timeout={timeout}s). The server may be temporarily "
                f"unavailable. Try again later.\n"
                f"  Host: {DEA_LANDCOVER}\n"
                f"  Detail: {e}"
            ) from e
        except requests.RequestException as e:
            # Any other request-level failure (e.g. SSL error). Fail fast.
            raise RuntimeError(
                f"Unexpected network error contacting DEA Land Cover "
                f"server: {e}"
            ) from e

        # Inspect the HTTP status code directly rather than relying on
        # raise_for_status(), so we can give a specific message per code.
        if resp.status_code == 200:
            logger.info("[found] DEA Land Cover year=%s: %s", y, url)
            return y, url
        elif resp.status_code == 404:
            # This year's file does not exist yet — try the next older year.
            logger.info(
                "DEA Land Cover mosaic not yet available for year=%s "
                "(HTTP 404). Trying older year...", y
            )
            continue
        elif resp.status_code in (401, 403):
            raise RuntimeError(
                f"Access denied to DEA Land Cover server "
                f"(HTTP {resp.status_code}) for year={y}.\n"
                f"  URL: {url}\n"
                f"This is a permissions issue, not a year-availability "
                f"issue. Check that the DEA data catalogue is publicly "
                f"accessible."
            )
        else:
            raise RuntimeError(
                f"Unexpected HTTP {resp.status_code} response from DEA "
                f"Land Cover server for year={y}.\n"
                f"  URL: {url}\n"
                f"This may indicate a server-side problem or a change in "
                f"the URL structure. Try again later or check:\n"
                f"  {DEA_LANDCOVER}"
            )

    # Reached here only if every year returned HTTP 404. The server is
    # reachable but no mosaic was found — the URL pattern may have changed.
    raise RuntimeError(
        f"DEA Land Cover mosaic not found for any year in "
        f"{years_tried[0]}–{years_tried[-1]} "
        f"(all returned HTTP 404).\n"
        f"The URL structure may have changed. Check the DEA catalogue:\n"
        f"  {DEA_LANDCOVER}"
    )


def _clip_raster_to_geom_in_memory(
    raster_path: str,
    geom_gdf: gpd.GeoDataFrame,
    fallback_nodata: int = DEA_FALLBACK_NODATA,
) -> rxr.rioxarray.RasterArray:
    """
    Clip a raster to a geometry and return the result as a DataArray.

    Thin wrapper around pre.util.clip_raster_in_memory that supplies the
    DEA Land Cover fallback nodata value.
    """
    return clip_raster_in_memory(
        raster_path, geom_gdf, fallback_nodata=fallback_nodata,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mask_dnbr(
    ctx: 'RunContext',
    dea_level: str = DEFAULT_DEA_LEVEL,
    dea_start_year: Optional[int] = None,
    dea_lookback: int = 6,
    natural_code: int = 112,
    quiet: bool = False,
) -> None:
    """
    Mask dNBR so only pixels over DEA natural vegetation are retained.

    Fetches the latest available DEA Land Cover mosaic via HTTP range
    requests (only the blocks covering the catchment are downloaded),
    reprojects it to match the dNBR grid, then sets non-vegetation
    pixels to NaN.

    Parameters:
    - ctx: event-level RunContext identifying the catchment + event.
    - dea_level: DEA Land Cover level string (default 'level3').
    - dea_start_year: most recent year to try for the DEA mosaic;
      defaults to current year minus 1.
    - dea_lookback: number of years back to search before giving up.
    - natural_code: DEA Land Cover class code for natural vegetation
      (default 112 = Natural Terrestrial Vegetation).
    - quiet: if True, suppress INFO-level log output.

    Returns:
    - None.  Writes masked_dNBR.tif and DEA_LC_<year>.tif into the
      event's FireSeverity folder under Events/<event>/FireSeverity/.
    """
    if quiet:
        logger.setLevel(logging.WARNING)

    ctx.validate(require_event_dir=False)
    catchment = ctx.catchment

    # -------------------------------
    # Locate inputs
    # -------------------------------
    sev_folder = ctx.event_path("FireSeverity")
    os.makedirs(sev_folder, exist_ok=True)

    dnbr_path = os.path.join(sev_folder, "dNBR.tif")
    if not os.path.exists(dnbr_path):
        raise FileNotFoundError(
            f"dNBR raster not found for catchment='{catchment}'. "
            f"Expected: {dnbr_path}\n"
            "Run severity.calculate_fire_severity(...) first."
        )

    # Read dNBR
    dnbr = rxr.open_rasterio(dnbr_path, masked=True).squeeze()
    if dnbr.rio.crs is None:
        raise RuntimeError(
            f"dNBR has no CRS. "
            f"Please ensure {dnbr_path} has a valid CRS."
        )

    dnbr_crs = dnbr.rio.crs
    dnbr_bounds = dnbr.rio.bounds()
    dnbr_bbox_gdf = _build_bbox_polygon_gdf(dnbr_bounds, dnbr_crs)

    logger.info("mask_dnbr(): catchment=%s", catchment)
    logger.info("mask_dnbr(): dNBR CRS=%s", dnbr_crs)
    logger.info("mask_dnbr(): dNBR bounds=%s", dnbr_bounds)

    # -------------------------------
    # Find latest DEA URL; only the blocks covering the catchment
    # are downloaded via GDAL vsicurl HTTP range requests.
    # -------------------------------
    latest_year, dea_url = _find_latest_dea_url(
        dea_level=dea_level,
        start_year=dea_start_year,
        lookback=dea_lookback,
    )

    # Clip DEA to dNBR bbox
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

    # Reproject to match dNBR grid (nearest-neighbour for categorical data)
    dea_match = dea_da.rio.reproject_match(
        dnbr,
        resampling=rio.warp.Resampling.nearest,
    )

    # Save the projected/aligned DEA raster
    dea_match_path = os.path.join(sev_folder, f"DEA_LC_{latest_year}.tif")
    logger.info("[write] %s", dea_match_path)
    dea_match.rio.to_raster(dea_match_path, compress="deflate")
    logger.info("[OK] Aligned DEA saved: %s", dea_match_path)

    # -------------------------------
    # Mask: keep only natural_code pixels
    # -------------------------------
    nd = dea_match.rio.nodata
    keep = (dea_match == natural_code)
    if nd is not None:
        keep = keep & (dea_match != nd)

    # Set non-vegetation pixels to NaN so they are excluded from
    # downstream erosion calculations (e.g. water bodies inside the
    # catchment boundary).
    dnbr_masked = (
        xr.where(keep, dnbr, float("nan")).astype("float32")
    )
    dnbr_masked = dnbr_masked.rio.write_crs(dnbr_crs)
    dnbr_masked = dnbr_masked.rio.write_nodata(float("nan"))

    out_path = os.path.join(sev_folder, "masked_dNBR.tif")
    logger.info("[write] %s", out_path)
    dnbr_masked.rio.to_raster(out_path, compress="deflate")
    logger.info("[OK] masked_dNBR saved: %s", out_path)
