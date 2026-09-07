"""
Simulate RUSLE (Revised Universal Soil Loss Equation) erosion and
sediment delivery for fire-impacted catchments.

Key functions:
- run_usle_simulation: Run the full RUSLE simulation with recorders.
- generate_rusle: Generator yielding per-timestep RUSLE result dicts.
- default_rusle_recorders: Build a factory of standard output recorders.
- run_rusle_all_replicates: Run RUSLE over rainfall replicates with Dask.
"""

# warnings.deprecated was added in Python 3.13. Provide a compatible
# fallback for older environments that emits a DeprecationWarning.
try:
    from warnings import deprecated
except ImportError:
    import functools
    import warnings as _warnings

    def deprecated(msg):
        """Backport shim for warnings.deprecated (Python < 3.13)."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _warnings.warn(
                    f"{func.__name__} is deprecated: {msg}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator

from affine import Affine
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import os
import logging
import time
import warnings
from fire_impacts import const as c
from fire_impacts.const import M2_TO_HA, MILLIGRAMS_TO_KILOGRAMS
from fire_impacts.pre.util import (
    read_aligned, read_dnbr_aligned, read_raster)
from fire_impacts.const import UNSET
from fire_impacts.params import ErosionParams, deprecated_overrides
from fire_impacts.provenance import (
    check_layers_fresh, check_run_not_overwritten, run_signature)
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid
from fire_impacts.util import load_package_data, get_zonal_stats
logger = logging.getLogger(__name__)

from fire_impacts.pre import FireImpactsProject
from fire_impacts.pre.project import save_catchment_raster
from fire_impacts.context import RunContext

# Recorders moved to their own module; re-exported so existing imports
# (including the private helpers the tests reach for) keep working.
from fire_impacts.sim.recorders import (  # noqa: F401
    _calendar_floor,
    _compute_periods,
    _PERIOD_OFFSETS,
    _spatial_coords_from_transform,
    default_rusle_recorders,
    record_grid_transform,
    record_multi_period_grid,
    record_subcatchment_timeseries,
    record_timestep_grid,
)

DNBR_SEVERITY_THRESHOLD = c.DEFAULT_DNBR_SEVERITY_THRESHOLD
# Rate constant of the unit kinetic-energy relation — the RUSLE2 value.
# See const.py for the derivation, the alternative (RUSLE) value, and why
# 0.29/0.72 are not parameters.
EMPIRICAL_COEFFICIENT = c.DEFAULT_KE_RATE_RUSLE2

# The model runs on 30-minute rainfall. Intensity is depth per hour, so
# the conversion is depth / 0.5 — derived here rather than written as a
# literal, which previously appeared twice and silently encoded the
# timestep in two places.
_MODEL_TIMESTEP = c.MODEL_TIMESTEP
_MODEL_TIMESTEP_HOURS = _MODEL_TIMESTEP.total_seconds() / 3600.0


def unit_kinetic_energy(intensity, rate=None):
    """
    Unit kinetic energy of rainfall at a given intensity.

    Implements the exponential KE-intensity relation

        e_r = 0.29 * [1 - 0.72 * exp(-k * i_r)]

    where 0.29 is the asymptotic maximum (drops reach terminal velocity,
    so energy per mm saturates) and 0.72 fixes the drizzle floor at
    0.0812 MJ/ha/mm. Only the rate constant k differs between published
    versions of the model — see const.py.

    Parameters:
    - intensity: rainfall intensity in mm/h (scalar or array).
    - rate: rate constant k. Defaults to the RUSLE2 value.

    Returns:
    - Unit kinetic energy in MJ/ha/mm, same shape as intensity.
    """
    if rate is None:
        rate = EMPIRICAL_COEFFICIENT
    return c.KE_ASYMPTOTE * (
        1 - c.KE_FLOOR_FRACTION * np.exp(-rate * intensity)
    )


def rainfall_erosivity(depth, timestep_hours=None, rate=None):
    """
    Convert one timestep's rainfall depth into intensity and erosivity.

    Erosivity is the storm energy times the intensity (the EI form used
    throughout RUSLE): E = e_r * depth, R = E * intensity.

    Parameters:
    - depth: rainfall depth for the timestep, in mm.
    - timestep_hours: length of the timestep in hours. Defaults to the
      model timestep (30 minutes).
    - rate: kinetic-energy rate constant; see unit_kinetic_energy.

    Returns:
    - Tuple of (intensity in mm/h, erosivity).
    """
    if timestep_hours is None:
        timestep_hours = _MODEL_TIMESTEP_HOURS
    intensity = depth / timestep_hours
    energy = unit_kinetic_energy(intensity, rate) * depth
    return intensity, energy * intensity
LOG_INTERVAL_SECONDS = 15.0

# ---------------------------------------------------------------------------
# RUSLE parameter grid helpers
# ---------------------------------------------------------------------------

def compute_klscp_layer(
    ctx: RunContext,
    support_practice_factor=UNSET,
    use_fire_adjusted: bool = True,
    recovery_time: float = None,
    params=None,
):
    """
    Combine C, K, and LS factor rasters into a single KLSCP layer.

    When use_fire_adjusted is False the unadjusted C and K factors are
    used, producing a pre-fire baseline KLSCP for comparison against the
    fire-impacted result.

    Parameters:
    - ctx: event-level (or run-level) RunContext.
    - support_practice_factor: Deprecated. Use the erosion parameter
      group (erosion.support_practice_factor), which is reachable from
      run_usle_simulation; this argument was not. Supplying it here is
      honoured as a call-layer override.
    - params: Calibration parameters — a ParameterRecord (from
      ctx.parameters()) or a ModelParameters. When None the layers are
      resolved from the context.
    - use_fire_adjusted: If True, use the fire-adjusted C and K rasters
      from Events/<event>/Erodibility/; otherwise the base catchment-
      level rasters are used.
    - recovery_time: Years since the fire for the recovery window being
      modelled; selects the C/K_factor_adjusted_<suffix>.tif pair.
      Required when use_fire_adjusted is True.

    Returns:
    - Tuple of (klscp_array, metadata_dict) where klscp_array is a
      float32 numpy array and metadata_dict is a rasterio metadata dict
      matching the LS-factor raster resolution and extent.
    """
    p = ctx._resolved_params(
        params,
        **deprecated_overrides({
            'erosion.support_practice_factor': support_practice_factor,
        }),
    ).parameters.erosion

    if use_fire_adjusted:
        if recovery_time is None:
            raise ValueError(
                "recovery_time must be provided when use_fire_adjusted=True."
            )
        # The fire-adjusted layers are per-event and per-recovery-time.
        suffix = c.recovery_time_suffix(recovery_time)
        c_factor_path = ctx.event_path(
            'Erodibility', f'C_factor_adjusted_{suffix}.tif')
        k_factor_path = ctx.event_path(
            'Erodibility', f'K_factor_adjusted_{suffix}.tif')
    else:
        c_factor_path = ctx.catchment_path('Erodibility', 'C_factor.tif')
        k_factor_path = ctx.catchment_path('Erodibility', 'K_factor.tif')
    ls_factor_path = ctx.catchment_path('Erodibility', 'LS_factor.tif')

    # LS is always at DEM resolution. Align C and K to it — the
    # unadjusted rasters are stored at their native (coarse) source
    # resolution, so a direct read() would give mismatched shapes.
    with rasterio.open(ls_factor_path) as ls_factor:
        ls_array = ls_factor.read(1)
        meta = ls_factor.meta.copy()
        transform = meta['transform']
        crs = meta['crs']

    c_array = read_aligned(c_factor_path, transform, crs, ls_array.shape)
    k_array = read_aligned(k_factor_path, transform, crs, ls_array.shape)

    # Multiply the four RUSLE factors to get the base erosion layer
    base = (
        c_array * k_array * ls_array * p.support_practice_factor
    ).astype(np.float32)

    meta.update(dtype=rasterio.float32, count=1, compress='lzw')

    return base, meta


def _rusle_parameter_grids(
    ctx: RunContext,
    use_fire_adjusted: bool = True,
    recovery_time: float = None,
    params=None,
):
    """
    Load KLSCP, SDR, and dNBR rasters plus spatial metadata for a
    catchment, ready for use in RUSLE calculations.

    Parameters:
    - ctx: event-level (or run-level) RunContext.
    - use_fire_adjusted: If True, use fire-adjusted C and K factors.
    - recovery_time: Years since the fire for the recovery window being
      modelled. Required when use_fire_adjusted is True.
    - params: Calibration parameters, passed through to
      compute_klscp_layer. Threading this is what makes the RUSLE P
      factor reachable from run_usle_simulation: it previously stopped
      here, so P was fixed at 1.0 from every realistic entry point.

    Returns:
    - Tuple of (klscp, sdr, dnbr, cell_area_ha, transform) where all
      three rasters are float32 numpy arrays, cell_area_ha is the cell
      area in hectares, and transform is the rasterio Affine transform
      shared by all three arrays.
    """
    cell_area_m2 = ctx.project.cell_area(ctx.catchment)
    cell_area_ha = cell_area_m2 * M2_TO_HA  # Convert to hectares

    # Compute KLSCP layer in memory
    klscp, klscp_meta = compute_klscp_layer(
        ctx,
        use_fire_adjusted=use_fire_adjusted,
        recovery_time=recovery_time,
        params=params,
    )

    transform = klscp_meta['transform']
    crs = klscp_meta['crs']

    # The per-recovery SDRs are per-event; the baseline SDR is derived
    # from the base C factor and lives at catchment scope.
    if use_fire_adjusted:
        if recovery_time is None:
            raise ValueError(
                "recovery_time must be provided when reading T-specific SDR."
            )
        suffix = c.recovery_time_suffix(recovery_time)
        sdr_path = ctx.event_path('Delivery', f'SDR_{suffix}.tif')
    else:
        sdr_path = ctx.catchment_path('Delivery', 'SDR_baseline.tif')

    sdr, _ = read_raster(sdr_path)

    # Get the delta Normalised Burn Ratio (dNBR) raster generated by
    # severity.calculate_fire_severity(). Use read_aligned to ensure it
    # is in the same CRS and at the same resolution as the RUSLE layers.
    # Read through the dNBR helper so the array is on the same 0-1000
    # scale as DNBR_SEVERITY_THRESHOLD. Comparing the stored fraction
    # against 400 made the high-severity branch unreachable, so the
    # severity split (and the per-severity constituent loads derived from
    # it) silently reported everything as low severity.
    dnbr = read_dnbr_aligned(
        ctx.event_path('FireSeverity', 'masked_dNBR.tif'),
        transform, crs, klscp.shape,
    )
    return klscp, sdr, dnbr, cell_area_ha, transform


# ---------------------------------------------------------------------------
# Lumped daily RUSLE (high-level entry point)
# ---------------------------------------------------------------------------

def lumped_daily_rusle(
    ctx: RunContext,
    rainfall,
    recovery_time: float = None,
    params=None,
):
    """
    Run RUSLE and SDR calculations for sub-catchments using 30-min
    rainfall and return a per-sub-catchment daily summary DataFrame.

    Parameters:
    - ctx: event-level RunContext.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.
    - recovery_time: Years since the fire for the recovery window being
      modelled; selects the fire-adjusted layers to read.
    - params: Calibration parameters (ParameterRecord or ModelParameters).

    Returns:
    - DataFrame summarising RUSLE and sediment delivery results, one
      row per sub-catchment per day.
    """
    ctx.validate()
    record = ctx._resolved_params(params)

    if 'units' not in rainfall.attrs:
        logger.warning(
            "Rainfall data has no units attribute, "
            "assuming units are correct (mm)"
        )
    elif rainfall.attrs['units'] != 'mm':
        logger.error(
            "Rainfall data has units '%s', expected 'mm'",
            rainfall.attrs['units'],
        )
        raise ValueError(
            "Rainfall data has units '%s', expected 'mm'"
            % rainfall.attrs['units']
        )
    RUSLE_df = calculate_lumped_rusle(
        ctx.project.get_subcatchments(ctx.catchment),
        rainfall,
        *_rusle_parameter_grids(
            ctx, recovery_time=recovery_time, params=record),
        erosion=record.parameters.erosion,
    )

    logger.info('Done')
    return RUSLE_df


# ---------------------------------------------------------------------------
# Subcatchment aggregation
# ---------------------------------------------------------------------------

def aggregate_rusle_to_subcatchments(
    ctx: RunContext,
    results_section: str = c.RESULTS_FOLDER_NAME,
    raster_names=None,
) -> 'pd.DataFrame | None':
    """
    Compute zonal statistics for each saved RUSLE output raster and
    write a per-subcatchment summary CSV to the Results folder.

    Parameters:
    - ctx: run-level RunContext (event + ensemble both required).
      Rasters are read from
      Runs/<event>/<ensemble>/<results_section>/.
    - results_section: Sub-folder name within the run directory where
      output rasters are stored. Defaults to the standard results
      folder name.
    - raster_names: Base names of the rasters to aggregate. When None,
      the standard RUSLE output rasters are used.

    Returns:
    - DataFrame with one row per subcatchment containing aggregated
      raster values, or None if no subcatchments are defined.
    ------------------------------------------------------------------------
    Notes:
    - 'Total' rasters (erosion_y1, delivered_y1, etc.) are aggregated
      using SUM. Each cell value is total tonnes over the simulation
      period, so summing over a subcatchment gives the total tonnes for
      that subcatchment — physically sound.
    - 'Peak' rasters (peak_erosion_y1, etc.) are aggregated using MEAN.
      Each cell stores the highest 30-min erosion at that cell, but peaks
      at different cells occur at different times. Summing would imply all
      cells peaked simultaneously, overstating the worst-case event load.
      Mean gives the average peak intensity per cell, characterising how
      erosion-prone the subcatchment is on its worst day.
    ------------------------------------------------------------------------
    """
    # Skip gracefully if subcatchments have not been set up yet
    try:
        subcatch_gdf = ctx.project.get_subcatchments(ctx.catchment)
    except FileNotFoundError:
        logger.info(
            'No subcatchments defined for %s — skipping RUSLE '
            'subcatchment aggregation.',
            ctx.catchment,
        )
        return None

    sc_id_col = ctx.project.subcatchment_id
    # Start the summary table with just the subcatchment ID
    summary = subcatch_gdf[[sc_id_col]].copy().reset_index(drop=True)

    names = (
        raster_names if raster_names is not None
        else c.RUSLE_OUTPUT_RASTER_NAMES
    )
    for raster_name in names:
        raster_path = ctx.run_path(results_section, f'{raster_name}.tif')
        if not os.path.exists(raster_path):
            continue

        # Choose aggregation stat based on raster type (see Notes): peak /
        # max grids average across cells, totals sum.
        stat = (
            'mean' if ('peak' in raster_name or 'max' in raster_name)
            else 'sum'
        )

        zstats = get_zonal_stats(
            subcatch_gdf, raster_path, raster_name, stats=[stat]
        )
        # Column name makes the aggregation method explicit
        col_name = f'{raster_name}_{stat}'
        summary[col_name] = [s[stat] for s in zstats]

    out_path = ctx.run_path(
        results_section, c.RUSLE_SC_SUMMARY_NAME + '.csv',
    )
    summary.to_csv(out_path, index=False)
    logger.info('Saved RUSLE subcatchment summary to %s', out_path)

    return summary


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------
# A 3-D grid with more time slices than this is not written to disk (it
# would be one raster per slice) — kept in memory only.
_MAX_GRID_SLICES_TO_DISK = 500


def _save_grid_results(ctx: RunContext, section, results, template_meta):
    """
    Write grid-type recorder results to the run's results folder as
    GeoTIFFs and return the saved base names.

    Handles both the low-level 2-D numpy grids (e.g. 'erosion_total') and
    the factory's xarray.DataArray grids: a 2-D array saves as one raster
    keyed by its recorder name; a 3-D (time, …) array saves one raster per
    time slice, suffixed with the period label (e.g. 'RUSLE_sum_yearly_20190101').
    Non-grid results (timeseries DataFrames, the transform, 'params') are
    skipped.
    """
    saved = []
    for key, data in results.items():
        if key == 'params' or data is None:
            continue

        # xarray DataArray (2-D or 3-D)?
        if hasattr(data, 'dims') and hasattr(data, 'values'):
            if 'time' in tuple(data.dims):
                times = list(data['time'].values)
                if len(times) > _MAX_GRID_SLICES_TO_DISK:
                    logger.warning(
                        "Recorder '%s' has %d time slices; not writing "
                        "rasters (kept in memory). Use a coarser "
                        "grid_timestep to save it.",
                        key, len(times),
                    )
                    continue
                # Period grids are daily-or-coarser and label cleanly by
                # date. record_timestep_grid is sub-daily though, so a
                # date-only label would collide (48 slices/day at the
                # 30 min model timestep) and each write would silently
                # overwrite the last. Fall back to including the time
                # only when the dates aren't unique, so existing
                # period-grid file names are unchanged.
                fmt = '%Y%m%d'
                if len({pd.Timestamp(t).strftime(fmt) for t in times}) \
                        < len(times):
                    fmt = '%Y%m%d_%H%M'
                for t in times:
                    label = pd.Timestamp(t).strftime(fmt)
                    name = f'{key}_{label}'
                    save_catchment_raster(
                        project=ctx.project, catchment=ctx.catchment,
                        file_name=name, section=section,
                        data=data.sel(time=t).values, meta=template_meta,
                        out_path=ctx.run_path(section, f'{name}.tif'),
                    )
                    saved.append(name)
            else:
                save_catchment_raster(
                    project=ctx.project, catchment=ctx.catchment,
                    file_name=key, section=section,
                    data=data.values, meta=template_meta,
                    out_path=ctx.run_path(section, f'{key}.tif'),
                )
                saved.append(key)
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            save_catchment_raster(
                project=ctx.project, catchment=ctx.catchment,
                file_name=key, section=section,
                data=data, meta=template_meta,
                out_path=ctx.run_path(section, f'{key}.tif'),
            )
            saved.append(key)
        # else: DataFrame / transform / scalar — not a grid, skip.
    return saved


def _recovery_run_segments(ctx: RunContext, rainfall, use_fire_adjusted):
    """
    Split rainfall into the chronological (recovery_time, segment) pairs a
    continuous run should process.

    Fire-adjusted: one segment per recovery window in the event
    definition, each carrying its window-start recovery_time so the
    matching C/K/SDR layers are used; a missing layer raises. Baseline: a
    single (None, rainfall) segment using the baseline layers over the
    whole period.
    """
    if not use_fire_adjusted:
        return [(None, rainfall)]

    definition = ctx.event_definition()

    index = rainfall.index
    segments = []
    for recovery_time, _ in definition.windows():
        window_start, window_end = definition.absolute_window(recovery_time)
        segment = rainfall[(index >= window_start) & (index < window_end)]
        if segment.empty:
            logger.warning(
                'No rainfall for recovery window T=%s (%s to %s) in '
                'catchment %s event %s; skipping.',
                recovery_time, window_start.date(), window_end.date(),
                ctx.catchment, ctx.event,
            )
            continue
        suffix = c.recovery_time_suffix(recovery_time)
        layer = ctx.event_path(
            'Erodibility', f'C_factor_adjusted_{suffix}.tif')
        if not os.path.exists(layer):
            raise FileNotFoundError(
                f"Missing fire-adjusted layer for recovery T={recovery_time} "
                f"({layer}). Run compute_adjusted_k_c first."
            )
        segments.append((recovery_time, segment))

    if not segments:
        raise ValueError(
            f'No rainfall overlaps any recovery window for catchment '
            f'{ctx.catchment} event {ctx.event}. Check the rainfall '
            f'period (see RunContext.simulation_period).'
        )
    return segments


def _layers_read_by(ctx, segments, use_fire_adjusted):
    """Return (path, consumed_paths) for every layer a run will read.

    Mirrors the paths _rusle_parameter_grids resolves, so the freshness
    check covers exactly what is about to be opened — including the
    per-recovery layers, which a single shared record cannot describe.
    """
    from fire_impacts.pre.rusle import (
        ADJUSTED_CK_CONSUMES, LS_CONSUMES, SDR_CONSUMES)

    layers = [(ctx.catchment_path('Erodibility', 'LS_factor.tif'),
               LS_CONSUMES)]
    if not use_fire_adjusted:
        layers.append(
            (ctx.catchment_path('Delivery', 'SDR_baseline.tif'),
             SDR_CONSUMES))
        return layers
    for recovery_time, _ in segments:
        suffix = c.recovery_time_suffix(recovery_time)
        layers += [
            (ctx.event_path('Erodibility',
                            f'C_factor_adjusted_{suffix}.tif'),
             ADJUSTED_CK_CONSUMES),
            (ctx.event_path('Erodibility',
                            f'K_factor_adjusted_{suffix}.tif'),
             ADJUSTED_CK_CONSUMES),
            (ctx.event_path('Delivery', f'SDR_{suffix}.tif'),
             SDR_CONSUMES),
        ]
    return layers


def _warn_about_materialising(recorders, timestep, data, counter):
    """
    Log which recorders force a full grid, using one probe timestep.

    Parameters:
    - recorders: the run's recorder dict.
    - timestep: the timestep to probe with - the first wet one.
    - data: that timestep's data dict.
    - counter: the run's MaterialisationCounter.
    ------------------------------------------------------------------------
    Notes:
    - Called once per run. The recorders are invoked here and must not be
      invoked again for this timestep.
    ------------------------------------------------------------------------
    """
    greedy = []
    for name, recorder in recorders.items():
        before = counter.count
        recorder(timestep, **data)
        if counter.count > before:
            greedy.append(name)
    if greedy:
        logger.warning(
            'Recorder(s) %s materialise a full grid every timestep; this '
            'run will be substantially slower than a collapsed one.',
            ', '.join(sorted(greedy)),
        )


def run_usle_simulation(
    ctx: RunContext,
    rainfall,
    recorders=None,
    save_rasters: bool = True,
    save_timeseries: bool = True,
    use_fire_adjusted: bool = True,
    results_section: str = None,
    params=None,
    allow_stale: bool = False,
    overwrite: bool = False,
    materialise_grids: bool = False,
):
    """
    Run the USLE simulation for the context and record outputs.

    The whole rainfall period is run continuously. For fire-adjusted runs
    the recovery windows in the project run-context are applied internally:
    rainfall is processed in chronological segments, each using the C/K/SDR
    layers for its recovery window, while the recorders accumulate across
    the whole period (reset once, finalised once). The result is a single
    set of outputs — recovery time is not an output dimension. Baseline
    runs use the baseline layers over the whole period in one segment.

    Parameters:
    - ctx: run-level RunContext. Outputs are written under
      Runs/<event>/<ensemble>/<results_section>/.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.
    - recorders: Dict of recorder closures to call at each timestep.
      If None, an empty dict is used and no outputs are recorded.
    - save_rasters: Whether to save output rasters as GeoTIFF files
      to the catchment results folder.
    - save_timeseries: Whether to save the daily time series as a CSV
      file to the catchment results folder.
    - use_fire_adjusted: If True, use fire-adjusted C and K rasters.
    - results_section: Sub-folder name for outputs within the run
      directory. If None, defaults to the standard results folder
      (or the baseline folder when use_fire_adjusted is False).
    - params: Calibration parameters — a ParameterRecord (from
      ctx.parameters()) or a ModelParameters. When None the project /
      catchment / event layers are resolved from the context. Resolved
      once here and reused for every recovery segment, so a run cannot
      straddle two resolutions.
    - allow_stale: proceed when the fire-adjusted layers this run reads
      were built with different parameters than it resolves. False (the
      default) raises instead, because the alternative is a run that
      silently mixes two calibrations — changing max_sdr and re-running
      the simulation used to reuse the old SDR rasters with no signal at
      all. Set True when the mismatch is understood and deliberate.
    - materialise_grids: escape hatch that forces every ScaledGrid to
      compute a real array on every timestep, bypassing the collapsed
      (scale-only) code path. Off by default. Use it for debugging and
      for the differential test that compares collapsed against
      uncollapsed results; a run with any recorder that already forces
      materialisation gets a logged warning regardless of this flag.

    Returns:
    - Dict of finalised recorder outputs keyed by recorder name, with
      an additional 'params' key holding the RUSLE parameter tuple
      (klscp, sdr, dnbr, cell_area_ha, transform).
    ------------------------------------------------------------------------
    Notes:
    - Each recorder must accept (timestep, **data) and return its
      running result. Recorders must also expose .reset() and
      .finalize() methods to manage state across calls.
    - The data dict passed to each recorder contains keys such as
      'RUSLE', 'delivered', and related per-cell arrays, plus a 'dry'
      flag. Every recorder gets the same arrays, so none of them may
      write into one without copying it first; on a dry timestep those
      arrays are shared and read-only, and accumulating them is both
      unnecessary and an error.
    ------------------------------------------------------------------------
    """
    ctx.validate()
    record = ctx._resolved_params(params)
    erosion = record.parameters.erosion

    if results_section is None:
        results_section = (
            c.RESULTS_FOLDER_NAME if use_fire_adjusted
            else c.RESULTS_BASELINE_FOLDER_NAME
        )

    # If no recorders were passed, use an empty dict so the rest of
    # the code works consistently
    if recorders is None:
        recorders = dict()

    # Build the chronological run segments (one per recovery window for
    # fire-adjusted runs; a single whole-period segment for baseline).
    segments = _recovery_run_segments(ctx, rainfall, use_fire_adjusted)

    # Check the layers this run is about to read against the parameters
    # it resolves. The layers are produced by a separate preprocessing
    # step, so nothing otherwise connects a parameter change to the
    # rasters built before it.
    layers = _layers_read_by(ctx, segments, use_fire_adjusted)
    check_layers_fresh(layers, record, strict=not allow_stale)

    # Refuse to land on top of results produced under a different
    # configuration. The signature covers this run's own parameters and
    # the digests of the layers it read, so rebuilt inputs count as a
    # change even when the run's parameters are untouched.
    signature = run_signature(record, layers, 'erosion')
    if save_rasters or save_timeseries:
        check_run_not_overwritten(
            ctx.run_path(results_section, c.PROVENANCE_FILE_NAME),
            signature, strict=not overwrite,
        )

    # Reset each recorder once so they accumulate across every segment.
    for recorder in recorders.values():
        recorder.reset()

    results = dict()
    grids = None
    # Created fresh per call (never shared module state): replicates run
    # concurrently on threads by default, and a shared counter or probe
    # flag would mix deltas across runs, blaming the wrong recorder.
    counter = MaterialisationCounter()
    probed = materialise_grids
    for recovery_time, segment_rain in segments:
        # Load the RUSLE parameter grids for this segment's recovery window
        grids = _rusle_parameter_grids(
            ctx,
            use_fire_adjusted=use_fire_adjusted,
            recovery_time=recovery_time,
            params=record,
        )
        klscp, sdr, dnbr, cell_area_ha, transform = grids

        # Rasterise the catchment boundary: 1 inside, NaN outside
        geometry = ctx.project.catchment_boundary(
            ctx.catchment).geometry.values
        mask = rasterio.features.rasterize(
            geometry,
            transform=transform,
            fill=np.nan,
            dtype=np.float32,
            out_shape=klscp.shape,
        )
        klscp_masked = klscp * mask
        sdr_masked = sdr * mask
        dnbr_masked = dnbr * mask

        # Feed every timestep of this segment into the (shared) recorders
        for timestep, data in generate_rusle(
            segment_rain,
            klscp_masked,
            sdr_masked,
            dnbr_masked,
            cell_area_ha,
            erosion=erosion,
            materialise_grids=materialise_grids,
            counter=counter,
        ):
            if not probed and not data['dry']:
                _warn_about_materialising(
                    recorders, timestep,
                    {**data, 'catchment': ctx.catchment,
                     'transform': transform},
                    counter,
                )
                probed = True
                continue
            for recorder in recorders.values():
                recorder(
                    timestep,
                    **data,
                    catchment=ctx.catchment,
                    transform=transform,
                )

    # Finalise each recorder after all segments are processed
    for key, recorder in recorders.items():
        results[key] = recorder.finalize()

    run_results_dir = ctx.run_path(results_section)
    if save_rasters or save_timeseries:
        ctx.ensure_run_directory()
        os.makedirs(run_results_dir, exist_ok=True)

    if save_rasters:
        template_raster = ctx.catchment_path(
            'Erodibility', 'LS_factor.tif',
        )
        _, template_meta = read_raster(template_raster)

        # Write every grid recorder result (2-D numpy or xarray 2-D/3-D)
        # to the run's results folder and aggregate those to subcatchments.
        saved_names = _save_grid_results(
            ctx, results_section, results, template_meta)
        aggregate_rusle_to_subcatchments(
            ctx,
            results_section=results_section,
            raster_names=saved_names,
        )

    if save_timeseries and c.RUSLE_OP_TIMESERIES_NAME in results:
        out_name = os.path.join(
            run_results_dir,
            c.RUSLE_OP_TIMESERIES_NAME + '.csv',
        )
        output = pd.DataFrame(data=results[c.RUSLE_OP_TIMESERIES_NAME])
        output.index.name = 'Datetime'
        output.to_csv(out_name)

    # Record what this run actually used, beside its outputs. Written
    # per results section so the fire-adjusted and baseline runs each
    # describe themselves — they resolve the same parameters today, but
    # a caller can pass params= to only one of them.
    if save_rasters or save_timeseries:
        ctx.write_provenance(
            record, scope='run', section=results_section,
            extra={'run_signature': signature},
        )

    # Attach a pointer to all the RUSLE parameters used for these calcs
    results['params'] = grids

    return results


# ---------------------------------------------------------------------------
# Deprecated simulation wrappers
# ---------------------------------------------------------------------------

@deprecated(
    "This function is deprecated and will be removed in a future version."
    " Please use run_usle_simulation() with appropriate recorders instead."
)
def gridded_total_rusle(ctx: RunContext, rainfall, params=None):
    """
    Compute total RUSLE erosion and delivery grids over a simulation.

    Deprecated. Use run_usle_simulation() with appropriate recorders.

    Parameters:
    - ctx: event-level RunContext.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.

    Returns:
    - Tuple of (total_eroded, total_delivered, transform) where the
      first two are float32 numpy arrays and the last is the Affine
      transform object from the RUSLE parameter grids.
    """
    ctx.validate()
    record = ctx._resolved_params(params)
    result = None
    # Get the boundary geometry for the first subcatchment only
    subcatch_boundaries = (
        ctx.project.get_subcatchments(ctx.catchment).iloc[0].geometry
    )

    total_eroded = None
    total_delivered = None
    grids = _rusle_parameter_grids(ctx, params=record)
    for day_data in generate_rusle_for_feature(
        [subcatch_boundaries], rainfall, *grids,
        erosion=record.parameters.erosion,
    ):
        day, _, _, _, \
        daily_RUSLE, daily_SDR, \
        _, _, _, _ = day_data
        if total_eroded is None:
            total_eroded = daily_RUSLE
            total_delivered = daily_SDR
        else:
            total_eroded += daily_RUSLE
            total_delivered += daily_SDR

    logger.info('Done')
    return total_eroded, total_delivered, grids[-1]


@deprecated(
    "This function is deprecated and will be removed in a future version."
    " Please use run_usle_simulation() with appropriate recorders instead."
)
def calculate_lumped_rusle(
    subcatchments: gpd.GeoDataFrame,
    rainfall: pd.DataFrame,
    klscp: np.array,
    sdr: np.array,
    dnbr: np.array,
    cell_area_ha: float,
    transform: Affine,
    erosion: ErosionParams = None,
):
    """
    Compute lumped daily RUSLE totals for each subcatchment polygon.

    Deprecated. Use run_usle_simulation() with appropriate recorders.

    Parameters:
    - subcatchments: GeoDataFrame of subcatchment polygons.
    - rainfall: DataFrame of 30-minute rainfall depth values in mm.
    - klscp: KLSCP raster array.
    - sdr: Sediment Delivery Ratio raster array.
    - dnbr: dNBR raster array.
    - cell_area_ha: Area of each raster cell in hectares.
    - transform: Affine transform shared by all three raster arrays.
    - erosion: ErosionParams supplying the severity threshold and the
      kinetic-energy rate constant. Defaults to the package values.

    Returns:
    - DataFrame with one row per subcatchment per day containing RUSLE
      totals, constituent loads, and severity-split erosion values.
    """
    daily_erosion = []
    for ind, subcatchment in subcatchments.iterrows():
        logger.info('Processing subcatchment %d', ind + 1)
        geometry = [subcatchment['geometry']]

        for day_data in generate_rusle_for_feature(
            geometry, rainfall, klscp, sdr, dnbr,
            cell_area_ha, transform, erosion=erosion,
        ):
            day, daily_total_rain, max_intensity, max_erosivity, \
            daily_RUSLE, daily_SDR, \
            daily_RUSLE_below_threshold, daily_RUSLE_above_threshold, \
            daily_SDR_below_threshold, daily_SDR_above_threshold = day_data

            daily_erosion.append({
                'Sub-catchment': f"Sub_{ind + 1}",
                'Day': day,
                'Rainfall (total daily)': daily_total_rain,
                'Max Rain Intensity (30 mins)': max_intensity,
                'Max Erosivity (30 mins)': max_erosivity,
                'RUSLE': np.nansum(daily_RUSLE),
                'RUSLE_SDR': np.nansum(daily_SDR),
                'RUSLE (Low severity)': np.nansum(
                    daily_RUSLE_below_threshold
                ),
                'RUSLE (High severity)': np.nansum(
                    daily_RUSLE_above_threshold
                ),
                'RUSLE_SDR (Low severity)': np.nansum(
                    daily_SDR_below_threshold
                ),
                'RUSLE_SDR (High severity)': np.nansum(
                    daily_SDR_above_threshold
                ),
            })

    # Convert the results to a DataFrame and compute constituent loads
    RUSLE_df = pd.DataFrame(daily_erosion)
    RUSLE_df = compute_particulates(RUSLE_df, erosion=erosion)
    RUSLE_df = RUSLE_df.round(1)
    logger.info('Done')
    return RUSLE_df


# ---------------------------------------------------------------------------
# RUSLE generators
# ---------------------------------------------------------------------------

def _log_progress(iteration_count, total_timesteps, start_time,
                  current_time, timestep):
    """Log simulation progress at the configured interval."""
    progress_pct = (iteration_count / total_timesteps) * 100
    elapsed_time = current_time - start_time
    avg_time_per_iteration = elapsed_time / iteration_count
    remaining_iterations = total_timesteps - iteration_count
    estimated_time_remaining = (
        avg_time_per_iteration * remaining_iterations
    )

    # Format time remaining as HH:MM:SS or MM:SS
    hours, remainder = divmod(int(estimated_time_remaining), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        time_str = f"{minutes:02d}:{seconds:02d}"

    logger.info(
        f"Progress: {iteration_count}/{total_timesteps} "
        f"timesteps ({progress_pct:.1f}%) - "
        f"Current timestep: {timestep} - "
        f"Estimated time remaining: {time_str}"
    )


def generate_rusle(
    rainfall: pd.Series,
    klscp: np.array,
    sdr: np.array,
    dnbr: np.array,
    cell_area_ha: float,
    erosion: ErosionParams = None,
    materialise_grids: bool = False,
    counter: MaterialisationCounter = None,
):
    """
    Yield per-timestep RUSLE erosion and delivery results as a generator.

    Parameters:
    - rainfall: Series with 30-minute rainfall depth values in mm.
    - klscp: KLSCP raster array.
    - sdr: Sediment Delivery Ratio raster array.
    - dnbr: dNBR raster array, on the conventional 0-1000 scale (read it
      through pre.util.read_dnbr_aligned, not read_aligned).
    - cell_area_ha: Area of each raster cell in hectares.
    - erosion: ErosionParams supplying the severity threshold and the
      kinetic-energy rate constant. Defaults to the package values. A
      plain parameter group rather than a RunContext, so this stays a
      data-in/data-out generator.
    - materialise_grids: When True, yield plain float arrays instead of
      ScaledGrid — the eager path used to check the deferred one agrees.
      May be forced True internally regardless of the argument; see
      Notes.
    - counter: optional MaterialisationCounter passed through to every
      yielded ScaledGrid, so a caller can tally how many of them a
      recorder ends up forcing.

    Returns:
    - Generator yielding (timestep, data_dict) tuples. Each data_dict
      contains the following keys:
      - 'total_rain': total rainfall depth for the timestep (float).
      - 'intensity': 30-min rainfall intensity in mm/hr (float).
      - 'erosivity': kinetic energy × intensity erosivity (float).
      - 'dry': True when the timestep had no rainfall (bool).
      - 'RUSLE': per-cell erosion, ScaledGrid, or a plain float array
        when materialise_grids is set.
      - 'delivered': RUSLE × SDR delivered sediment, same type as above.
      - 'RUSLE_below_threshold': erosion at low-severity cells.
      - 'RUSLE_above_threshold': erosion at high-severity cells.
      - 'delivered_below_threshold': delivered at low-severity cells.
      - 'delivered_above_threshold': delivered at high-severity cells.
    ------------------------------------------------------------------------
    Notes:
    - klscp, sdr, and dnbr must share the same shape and transform.
    - This is a generator function; results are produced one timestep at
      a time rather than all at once to keep memory usage manageable.
    - A dry timestep's grids are `0.0 * unit`, not an exact-zero array:
      outside the catchment mask (where the unit layers are NaN) they
      materialise as NaN, the same as a wet timestep. This is consistent
      but is a change from a prior version that yielded exact zeros
      everywhere on dry timesteps via a shared zero buffer.
    - A negative static layer cell or a negative rainfall value forces
      materialise_grids to True for the whole segment, regardless of
      the argument, because deferring a maximum across timesteps is
      only valid when both are non-negative. A warning is logged when
      this happens.
    ------------------------------------------------------------------------
    """
    # Convert to Series if we've got a DataFrame, to ensure consistency
    if isinstance(rainfall, pd.DataFrame):
        rainfall = pd.Series(
            data=rainfall['rainfall'], index=rainfall.index
        )

    if erosion is None:
        erosion = ErosionParams()

    # The static half of the model. Erosion in a cell is this layer times
    # the timestep's erosivity, so it is built once per segment rather
    # than rebuilt every timestep.
    below = dnbr < erosion.dnbr_severity_threshold
    above = dnbr >= erosion.dnbr_severity_threshold
    # cell_area_ha must be a strong (numpy) float64 here: under NEP 50 a
    # plain Python float is a weak scalar, so klscp (float32) * a Python
    # float stays float32 — the old code's accumulation was float64
    # throughout because R (np.float64) was the strong operand.
    unit_rusle = klscp * np.float64(cell_area_ha)
    unit_delivered = unit_rusle * sdr
    units = {
        'RUSLE': unit_rusle,
        'delivered': unit_delivered,
        'RUSLE_below_threshold': np.where(below, unit_rusle, 0),
        'RUSLE_above_threshold': np.where(above, unit_rusle, 0),
        'delivered_below_threshold': np.where(below, unit_delivered, 0),
        'delivered_above_threshold': np.where(above, unit_delivered, 0),
    }

    # Deferring a maximum assumes a layer times a scale is monotone in
    # the scale, which holds only while both are non-negative. A noisy
    # DEM can produce a negative LS cell, and that must not silently
    # produce a wrong maximum - nor stop a run that works today.
    negative_layer = any(
        np.nanmin(unit) < 0 for unit in units.values() if unit.size
    )
    negative_rain = bool(np.nanmin(rainfall.values) < 0) \
        if len(rainfall.values) else False
    if negative_layer or negative_rain:
        logger.warning(
            'Negative %s in this segment, so its grids cannot be '
            'deferred; computing them in full instead. Results are '
            'unaffected, but this segment will be slower.',
            'erosion layer' if negative_layer else 'rainfall',
        )
        materialise_grids = True

    total_timesteps = len(rainfall.index)
    start_time = time.time()
    last_log_time = start_time
    iteration_count = 0

    for timestep, delta_v_r in zip(rainfall.index, rainfall.values):
        iteration_count += 1

        if delta_v_r == 0:
            intensity = 0.0
            R = 0.0
        else:
            intensity, R = rainfall_erosivity(
                delta_v_r, rate=erosion.kinetic_energy_coefficient)

        # One scale array shared by all six grids of this timestep. It is
        # an array, not a float, so that spatially varying rainfall widens
        # it without any accumulator needing a second code path.
        scale = np.array([R], dtype=np.float64)
        result = {
            'total_rain': delta_v_r,
            'intensity': intensity,
            'erosivity': R,
            'dry': bool(delta_v_r == 0),
        }
        for key, unit in units.items():
            if materialise_grids:
                result[key] = scale[0] * unit
            else:
                result[key] = ScaledGrid(scale, unit, counter=counter)

        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
            _log_progress(iteration_count, total_timesteps, start_time,
                          current_time, timestep)
            last_log_time = current_time

        yield (timestep, result)


@deprecated(
    "This function is deprecated and will be removed in a future version."
    " Please use run_usle_simulation() with appropriate recorders instead."
)
def generate_rusle_for_feature(
    geometry: list,
    rainfall: pd.DataFrame,
    klscp: np.array,
    sdr: np.array,
    dnbr: np.array,
    cell_area_ha: float,
    transform: Affine,
    erosion: ErosionParams = None,
):
    """
    Yield daily RUSLE results clipped to a single feature geometry.

    Deprecated. Use run_usle_simulation() with appropriate recorders.

    Parameters:
    - geometry: List of shapely geometries defining the sub-catchment.
    - rainfall: DataFrame with 30-minute rainfall depth values in mm.
    - klscp: KLSCP raster array.
    - sdr: Sediment Delivery Ratio raster array.
    - dnbr: dNBR raster array, on the conventional 0-1000 scale.
    - cell_area_ha: Area of each raster cell in hectares.
    - transform: Affine transform shared by all three raster arrays.
    - erosion: ErosionParams supplying the severity threshold and the
      kinetic-energy rate constant. Defaults to the package values.

    Returns:
    - Generator yielding one tuple per day:
      (day, daily_total_rain, max_intensity, max_erosivity,
       daily_RUSLE, daily_SDR,
       daily_RUSLE_below_threshold, daily_RUSLE_above_threshold,
       daily_SDR_below_threshold, daily_SDR_above_threshold)
    """
    if erosion is None:
        erosion = ErosionParams()

    mask = rasterio.features.rasterize(
        geometry,
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
        out_shape=klscp.shape,
    )
    klscp_masked = klscp * mask
    sdr_masked = sdr * mask
    dnbr_masked = dnbr * mask

    days = pd.Series(rainfall.index.date).drop_duplicates()
    for day in days:
        rainfall_data = rainfall[rainfall.index.date == day]
        # Initialise daily accumulators
        daily_RUSLE = np.zeros_like(klscp_masked, dtype=np.float32)
        daily_SDR = np.zeros_like(klscp_masked, dtype=np.float32)
        daily_total_rain = 0.0
        max_intensity = 0.0
        max_erosivity = 0.0

        # Loop over each 30-min interval within the day
        for subday in rainfall_data.values:
            delta_v_r = subday
            daily_total_rain += delta_v_r

            if delta_v_r == 0:
                continue

            intensity, R = rainfall_erosivity(
                delta_v_r, rate=erosion.kinetic_energy_coefficient)
            max_intensity = max(max_intensity, intensity)
            max_erosivity = max(max_erosivity, R)

            # Total erosion in tonnes per hectare
            # TODO: sediment eroded? kg? t?
            RUSLE = (R * klscp_masked) * cell_area_ha
            daily_RUSLE += RUSLE

            # Total delivered sediment
            # TODO: sediment delivered? kg? t?
            SDR_RUSLE = RUSLE * sdr_masked
            daily_SDR += SDR_RUSLE

        # Split daily totals by dNBR severity threshold
        dnbr_below_threshold = dnbr_masked < erosion.dnbr_severity_threshold
        dnbr_above_threshold = dnbr_masked >= erosion.dnbr_severity_threshold

        daily_RUSLE_below_threshold = np.where(
            dnbr_below_threshold, daily_RUSLE, 0
        )
        daily_RUSLE_above_threshold = np.where(
            dnbr_above_threshold, daily_RUSLE, 0
        )
        daily_SDR_below_threshold = np.where(
            dnbr_below_threshold, daily_SDR, 0
        )
        daily_SDR_above_threshold = np.where(
            dnbr_above_threshold, daily_SDR, 0
        )

        yield (
            day, daily_total_rain, max_intensity, max_erosivity,
            daily_RUSLE, daily_SDR,
            daily_RUSLE_below_threshold, daily_RUSLE_above_threshold,
            daily_SDR_below_threshold, daily_SDR_above_threshold,
        )


# ---------------------------------------------------------------------------
# Constituent calculations
# ---------------------------------------------------------------------------

def compute_particulates(rusle_df, constituents_df=None,
                         erosion: ErosionParams = None):
    """
    Add constituent load columns to a RUSLE results DataFrame.

    Multiplies the low- and high-severity RUSLE_SDR values by empirical
    constituent ratios from the ash_constituents lookup table to estimate
    particulate loads for each constituent.

    Parameters:
    - rusle_df: DataFrame of RUSLE results containing 'RUSLE_SDR
      (Low severity)' and 'RUSLE_SDR (High severity)' columns.
    - constituents_df: DataFrame of ash constituent ratios. Takes
      precedence over erosion.ash_constituents_table when given.
    - erosion: ErosionParams naming the table to load when
      constituents_df is None. Defaults to the package values.

    Returns:
    - The input DataFrame with additional columns for each constituent
      load in tonnes.
    """
    if constituents_df is None:
        if erosion is None:
            erosion = ErosionParams()
        constituents_df = load_package_data(erosion.ash_constituents_table)

    # Iterate through each constituent row and compute loads
    for _, row in constituents_df.iterrows():
        particulate = row['Particulate constituent (ash)']
        low_severity = (
            row['Low severity- mean amount (mgkg-1)']
            * MILLIGRAMS_TO_KILOGRAMS
        )
        high_severity = (
            row['High severity- mean amount (mgkg-1)']
            * MILLIGRAMS_TO_KILOGRAMS
        )

        column_name = f"{particulate} (Tonne)"
        rusle_df[column_name] = (
            rusle_df['RUSLE_SDR (Low severity)'] * low_severity
            + rusle_df['RUSLE_SDR (High severity)'] * high_severity
        )
    return rusle_df


# ---------------------------------------------------------------------------
# Ensemble runners (Dask-parallel)
# ---------------------------------------------------------------------------

def run_rusle_replicate(
    ctx: RunContext,
    rainfall_30min,
    replicate_idx,
    recorder_factory=None,
    use_fire_adjusted=True,
):
    """
    Run the RUSLE simulation for a single rainfall replicate.

    For each catchment the full replicate rainfall is run through
    run_usle_simulation, which applies the recovery windows internally, so
    the result is a single continuous set of outputs per catchment.

    Parameters:
    - ctx: run-level RunContext.
    - rainfall_30min: xarray.Dataset of rainfall replicates with a
      'replicate' dimension.
    - replicate_idx: Index of the replicate to run.
    - recorder_factory: Callable (ctx, start, end) → dict of recorders,
      as returned by default_rusle_recorders(). When None, a default
      factory is used.
    - use_fire_adjusted: If True, use the fire-adjusted layers; if False,
      the baseline layers.

    Returns:
    - Dict keyed by catchment name (single key for ctx.catchment);
      value is the recorder results dict for this replicate. The
      per-catchment wrapper matches the shape consumed by save_run /
      ensemble.py helpers.
    """
    rain_seq = rainfall_30min.rainfall[:, replicate_idx].to_pandas()
    start, end = rain_seq.index[0], rain_seq.index[-1]

    if recorder_factory is None:
        recorder_factory = default_rusle_recorders()

    start = rain_seq.index[0]
    end = rain_seq.index[-1]
    recorders = recorder_factory(ctx, start, end)
    results = run_usle_simulation(
        ctx,
        rain_seq,
        recorders=recorders,
        save_rasters=False,
        save_timeseries=False,
        use_fire_adjusted=use_fire_adjusted,
    )
    return {ctx.catchment: results}


def run_rusle_all_replicates(
    ctx: RunContext,
    rainfall_30min,
    n_workers=None,
    scheduler='threads',
    replicate_indices=None,
    recorder_factory=None,
    use_fire_adjusted=True,
):
    """
    Run RUSLE for all rainfall replicates in parallel.

    Returns {replicate: {catchment: recorder-results}} — the standard shape
    the ensemble aggregation/save helpers expect. Recovery windows are
    applied internally by run_usle_simulation, so each replicate yields a
    single continuous set of outputs (not split by recovery time).

    Parameters:
    - ctx: run-level RunContext.
    - rainfall_30min: xarray.Dataset of rainfall replicates with a
      'replicate' dimension.
    - n_workers: Number of Dask workers. If None, Dask chooses.
    - scheduler: Dask scheduler to use, e.g. 'threads' or 'processes'.
    - replicate_indices: Iterable of replicate indices to run. If None,
      all replicates are run.
    - recorder_factory: Callable (ctx, start, end) → dict of recorders,
      as returned by default_rusle_recorders(). When None, a default
      factory is used.
    - use_fire_adjusted: If True, use the fire-adjusted layers; if False,
      the baseline layers.

    Returns:
    - Dict mapping replicate index (int) to {catchment: results} dicts.
    """
    import dask

    if recorder_factory is None:
        recorder_factory = default_rusle_recorders()

    if replicate_indices is None:
        replicate_indices = list(range(rainfall_30min.sizes['replicate']))
    else:
        replicate_indices = list(replicate_indices)

    tasks = [
        dask.delayed(run_rusle_replicate)(
            ctx,
            rainfall_30min,
            i,
            recorder_factory=recorder_factory,
            use_fire_adjusted=use_fire_adjusted,
        )
        for i in replicate_indices
    ]

    computed = dask.compute(
        *tasks, scheduler=scheduler, num_workers=n_workers
    )
    return {i: result for i, result in zip(replicate_indices, computed)}
