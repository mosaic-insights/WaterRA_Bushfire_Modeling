"""
Generate synthetic dNBR maps by sampling from the empirical distribution
of a real fire of a given severity.

Low-level functions accept/return numpy arrays and GeoDataFrames.
High-level functions integrate with the FireImpactsProject directory
structure.
"""

import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterio.warp import reproject, calculate_default_transform, Resampling
from affine import Affine
import geopandas as gpd
import logging
import os

from . import data_sources
from .project import FireImpactsProject, save_catchment_raster
from ..context import RunContext  # noqa: F401
from .util import from_dnbr_scale, read_raster
from .. import const as c

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference fire URLs — pre-clipped dNBR rasters hosted via HTTPS.
# Values outside the burned area are NaN; no shapefile needed.
# ---------------------------------------------------------------------------
REFERENCE_FIRES = {
    'medium': data_sources.SYNTHETIC_FIRE_MEDIUM_DNBR,
    'high': data_sources.SYNTHETIC_FIRE_HIGH_DNBR,
}

_SEVERITY_ALIASES = {
    'high': 'high', 'hi': 'high', 'h': 'high',
    'medium': 'medium', 'med': 'medium', 'm': 'medium',
}


# ---------------------------------------------------------------------------
# Low-level functions — data in, data out
# ---------------------------------------------------------------------------

def extract_dnbr_distribution(dnbr_array):
    """
    Extract the empirical dNBR distribution from a raster array.

    Parameters:
    - dnbr_array: 2-D numpy array of dNBR values.  NaN marks pixels
      outside the burn area.

    Returns:
    - 1-D array of valid (non-NaN) dNBR values.
    """
    flat = dnbr_array.ravel()
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        raise ValueError("dNBR array contains no valid (non-NaN) values.")
    return valid


def generate_synthetic_dnbr(
    distribution,
    catchment_boundary,
    transform,
    shape,
    random_seed=None,
):
    """
    Sample from an empirical dNBR distribution onto a target grid.

    Parameters:
    - distribution: 1-D array of dNBR values to sample from (as
      returned by extract_dnbr_distribution).
    - catchment_boundary: GeoDataFrame or list of geometries defining
      the area to fill.  Must be in the same CRS as the target grid.
    - transform: affine.Affine transform for the target grid.
    - shape: (rows, cols) of the target grid.
    - random_seed: integer seed for reproducibility, or None.

    Returns:
    - 2-D float32 array with synthetic dNBR values inside the
      catchment and NaN outside.
    """
    rng = np.random.default_rng(random_seed)

    # Rasterize the catchment boundary to build a mask
    if isinstance(catchment_boundary, gpd.GeoDataFrame):
        geometries = catchment_boundary.geometry.values
    else:
        geometries = catchment_boundary

    mask = rasterize(
        geometries,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    inside = mask == 1

    # Sample from the known distribution and assign to valid pixels
    n_cells = int(inside.sum())
    sampled = rng.choice(distribution, size=n_cells)

    dnbr = np.full(shape, np.nan, dtype=np.float32)
    dnbr[inside] = sampled

    return dnbr


def load_reference_dnbr(url_or_path):
    """
    Load a reference dNBR raster and return its value distribution.

    The source can be a local file path or an HTTPS URL (read directly
    by rasterio/GDAL without downloading the full file).

    Parameters:
    - url_or_path: path or URL to a GeoTIFF with pre-clipped dNBR
      values (NaN outside the burned area).

    Returns:
    - distribution: 1-D array of valid dNBR values.
    - meta: rasterio metadata dict from the source raster (useful for
      cell size and CRS).
    """
    logger.info("Loading reference dNBR from %s", url_or_path)
    with rio.open(url_or_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
    distribution = extract_dnbr_distribution(data)
    logger.info(
        "Extracted %d valid pixels (dNBR range %.0f–%.0f)",
        distribution.size, np.nanmin(distribution), np.nanmax(distribution),
    )
    return distribution, meta


# ---------------------------------------------------------------------------
# High-level function — project-aware
# ---------------------------------------------------------------------------

def generate_synthetic_fire(
    ctx: 'RunContext',
    severity='medium',
    random_seed=None,
    reference_url=None,
):
    """
    Generate a synthetic dNBR map and save it under the context's event.

    Fetches a pre-clipped reference dNBR raster for the requested
    severity, extracts its empirical distribution, samples onto the
    catchment's DEM grid, and saves the result as masked_dNBR.tif in
    the event's FireSeverity folder.

    This is the synthetic-fire equivalent of the real-fire pipeline
    (severity.calculate_fire_severity + mask_dnbr.mask_dnbr). The
    output is consumed directly by the RUSLE preprocessing and
    simulation modules.

    Parameters:
    - ctx: event-level RunContext identifying the catchment + event.
    - severity: fire severity template to use: 'medium' or 'high'
      (also accepts 'med', 'm', 'hi', 'h').
    - random_seed: integer seed for reproducible output, or None.
    - reference_url: override the default reference dNBR URL for the
      given severity.  Useful for custom or locally-hosted fires.

    Returns:
    - The generated synthetic dNBR array.
    """
    ctx.validate(require_event_dir=False)
    project = ctx.project
    catchment = ctx.catchment

    # --- Resolve severity ---
    sev_key = _SEVERITY_ALIASES.get(severity.strip().lower())
    if sev_key is None:
        raise ValueError(
            f"Unknown severity '{severity}'. "
            f"Use one of: {list(_SEVERITY_ALIASES.keys())}"
        )

    # --- Load reference distribution ---
    url = reference_url or REFERENCE_FIRES[sev_key]
    distribution, _ = load_reference_dnbr(url)

    # --- Read DEM to get the target grid parameters ---
    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    dem_data, dem_meta = read_raster(dem_path)
    transform = dem_meta['transform']
    crs = dem_meta['crs']
    shape = dem_data.shape

    # --- Get catchment boundary in the DEM's CRS ---
    boundary = project.catchment_boundary(catchment).to_crs(crs)

    # --- Generate synthetic dNBR ---
    logger.info(
        "Generating synthetic %s-severity dNBR for %s (%d×%d grid)",
        sev_key, catchment, shape[0], shape[1],
    )
    dnbr = generate_synthetic_dnbr(
        distribution, boundary, transform, shape, random_seed,
    )
    # The reference rasters are published on the conventional 0-1000 scale,
    # but masked_dNBR.tif stores the raw band-ratio difference. Convert, so
    # the synthetic and real severity paths write the same convention —
    # they previously differed by 1000x, and no consumer could tell which
    # one it had been given.
    dnbr = from_dnbr_scale(dnbr)

    # --- Save as masked_dNBR.tif in the event's FireSeverity folder ---
    out_path = ctx.event_path(c.FIRE_SEVERITY_FOLDER_NAME, 'masked_dNBR.tif')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_catchment_raster(
        project=project,
        catchment=catchment,
        file_name='masked_dNBR',
        section=c.FIRE_SEVERITY_FOLDER_NAME,
        data=dnbr,
        meta=dem_meta,
        out_path=out_path,
    )
    logger.info("Saved synthetic masked_dNBR.tif for %s", catchment)

    return dnbr
