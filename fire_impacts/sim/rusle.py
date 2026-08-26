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
from fire_impacts.provenance import check_layers_fresh
from fire_impacts.util import load_package_data, get_zonal_stats
logger = logging.getLogger(__name__)

from fire_impacts.pre import FireImpactsProject
from fire_impacts.pre.project import save_catchment_raster
from fire_impacts.context import RunContext

DNBR_SEVERITY_THRESHOLD = c.DEFAULT_DNBR_SEVERITY_THRESHOLD
# Rate constant of the unit kinetic-energy relation — the RUSLE2 value.
# See const.py for the derivation, the alternative (RUSLE) value, and why
# 0.29/0.72 are not parameters.
EMPIRICAL_COEFFICIENT = c.DEFAULT_KE_RATE_RUSLE2

# The model runs on 30-minute rainfall. Intensity is depth per hour, so
# the conversion is depth / 0.5 — derived here rather than written as a
# literal, which previously appeared twice and silently encoded the
# timestep in two places.
_MODEL_TIMESTEP = pd.Timedelta(minutes=30)
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
# Recorder closures
# ---------------------------------------------------------------------------

def record_subcatchment_timeseries(
    ctx: RunContext,
    variable_name: str,
    fn="sum",
    label_field=None,
    agg_count=1,
):
    """
    Build a RUSLE recorder that summarises a raster variable over
    subcatchments and accumulates a spatial time series.

    Parameters:
    - ctx: RunContext identifying the catchment whose subcatchments
      are aggregated.
    - variable_name: Name of the raster variable to summarise (key in
      the timestep data dict).
    - fn: Spatial aggregation function: 'sum', 'mean', or 'max'.
      Default is 'sum'.
    - label_field: Column in the subcatchment GeoDataFrame to use as
      zone labels. If None, the integer index is used.
    - agg_count: Number of model timesteps to accumulate before
      recording one output row. Default is 1.

    Returns:
    - A recorder closure compatible with run_usle_simulation, with
      .reset() and .finalize() methods attached.
    ------------------------------------------------------------------------
    Notes:
    - While the time series can always be resampled after the simulation,
      agg_count allows aggregation during the run, which is significantly
      faster for large ensembles.
    - agg_count of 1 is appropriate when each model row already
      represents the desired output interval.
    ------------------------------------------------------------------------
    """
    result = None
    index = None
    zones = None
    zone_names = None

    intermediate = None
    intermediate_count = 0

    # -----------------------------------------------------------------------
    def timeseries_recorder(timestep, catchment, transform, **kwargs):
        """
        Accumulate one timestep of raster data into the running result.

        Parameters:
        - timestep: Datetime label for the current model timestep.
        - catchment: Name of the catchment being processed.
        - transform: Rasterio Affine transform for converting polygon
          geometries to raster masks.

        Returns:
        - None until agg_count timesteps have been accumulated; then a
          dict mapping zone names to lists of aggregated values.
        """
        # Declare the variables from the outer scope so this closure
        # remembers their values between calls
        nonlocal result, index
        nonlocal zones, zone_names
        nonlocal intermediate, intermediate_count

        data = kwargs.get(variable_name)
        # Raise an error if the requested variable isn't in the data
        if data is None:
            raise ValueError(
                f"Variable {variable_name} not found in simulation data."
            )

        # On the first call, build zone masks from the subcatchment
        # boundaries. Fall back to the whole catchment boundary if no
        # subcatchments have been registered.
        if zones is None:
            try:
                boundaries_v = ctx.project.get_subcatchments(ctx.catchment)
            except FileNotFoundError:
                boundaries_v = ctx.project.catchment_boundary(ctx.catchment)
            resolved_label = label_field
            if resolved_label is None:
                resolved_label = ctx.project.subcatchment_label_field(
                    ctx.catchment,
                )
            # Rasterise each subcatchment polygon separately to produce
            # one binary mask per zone
            zones = [
                rasterio.features.rasterize(
                    [g],
                    transform=transform,
                    fill=np.nan,
                    dtype=np.float32,
                    out_shape=data.shape,
                ) for g in boundaries_v.geometry
            ]
            if resolved_label is None:
                zone_names = boundaries_v.index.values
            elif resolved_label not in boundaries_v.columns:
                logger.warning(
                    "Subcatchment label field '%s' is configured for "
                    "catchment '%s' but is not present in the saved "
                    "subcatchments shapefile (columns: %s). Falling "
                    "back to integer indices. Re-run "
                    "FireImpactsProject.add_subcatchments(..., "
                    "label_field='%s') to rewrite the shapefile with "
                    "the label column retained.",
                    resolved_label, ctx.catchment,
                    list(boundaries_v.columns), resolved_label,
                )
                zone_names = boundaries_v.index.values
            else:
                zone_names = boundaries_v[resolved_label].values

        # Accumulate data into the current aggregation cycle
        intermediate_count += 1
        if intermediate is None:
            intermediate = data
        else:
            intermediate += data

        # Return early if we haven't reached the requested agg_count yet
        if intermediate_count < agg_count:
            return result

        # Flush the accumulated data and reset the intermediate state
        data = intermediate
        intermediate = None
        intermediate_count = 0

        if index is None:
            index = []
        index.append(timestep)

        # Mask each zone so only cells inside it retain their values
        masked = [data * zone for zone in zones]

        def agg(d):
            """Apply the requested spatial aggregation to one zone."""
            if fn == "sum":
                return np.nansum(d)
            elif fn == "mean":
                return np.nanmean(d)
            elif fn == "max":
                return np.nanmax(d)
            else:
                raise ValueError(f"Function {fn} not recognized.")

        if result is None:
            result = {name: [] for name in zone_names}

        grouped = [agg(d) for d in masked]
        for ix, name in enumerate(zone_names):
            result[name].append(grouped[ix])

        return result

    # -----------------------------------------------------------------------
    def reset():
        """Reset all accumulated state back to initial values."""
        nonlocal result, index
        nonlocal zones, zone_names
        nonlocal intermediate, intermediate_count
        index = None
        zones = None
        zone_names = None
        result = None
        intermediate = None
        intermediate_count = 0

    # -----------------------------------------------------------------------
    def finalize():
        """Convert accumulated lists to arrays and return a DataFrame."""
        nonlocal result, index
        for key in result:
            result[key] = np.array(result[key])
        return pd.DataFrame(result, index=index)

    timeseries_recorder.reset = reset
    timeseries_recorder.finalize = finalize
    return timeseries_recorder


def record_grid_transform():
    """
    Build a recorder that captures the raster transform at each timestep.

    Returns:
    - A recorder closure with .reset() and .finalize() methods; finalize
      returns the most recently captured affine transform object.
    """
    t = None

    def get_transform(timestep, transform, **kwargs):
        nonlocal t
        t = transform
        return t

    def r():
        """Reset the captured transform."""
        pass

    def f():
        """Return the most recently captured transform."""
        return t

    get_transform.reset = r
    get_transform.finalize = f

    return get_transform


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
      'RUSLE', 'delivered', and related per-cell arrays.
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
    check_layers_fresh(
        _layers_read_by(ctx, segments, use_fire_adjusted),
        record, strict=not allow_stale,
    )

    # Reset each recorder once so they accumulate across every segment.
    for recorder in recorders.values():
        recorder.reset()

    results = dict()
    grids = None
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
        ):
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
        ctx.write_provenance(record, scope='run', section=results_section)

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
    RUSLE_df = compute_particulates(RUSLE_df)
    RUSLE_df = RUSLE_df.round(1)
    logger.info('Done')
    return RUSLE_df


# ---------------------------------------------------------------------------
# RUSLE generators
# ---------------------------------------------------------------------------

def generate_rusle(
    rainfall: pd.Series,
    klscp: np.array,
    sdr: np.array,
    dnbr: np.array,
    cell_area_ha: float,
    erosion: ErosionParams = None,
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

    Returns:
    - Generator yielding (timestep, data_dict) tuples. Each data_dict
      contains the following keys:
      - 'total_rain': total rainfall depth for the timestep (float).
      - 'intensity': 30-min rainfall intensity in mm/hr (float).
      - 'erosivity': kinetic energy × intensity erosivity (float).
      - 'RUSLE': per-cell erosion array (float32 numpy array).
      - 'delivered': RUSLE × SDR delivered sediment array.
      - 'RUSLE_below_threshold': erosion at low-severity cells.
      - 'RUSLE_above_threshold': erosion at high-severity cells.
      - 'delivered_below_threshold': delivered at low-severity cells.
      - 'delivered_above_threshold': delivered at high-severity cells.
    ------------------------------------------------------------------------
    Notes:
    - klscp, sdr, and dnbr must share the same shape and transform.
    - This is a generator function; results are produced one timestep at
      a time rather than all at once to keep memory usage manageable.
    ------------------------------------------------------------------------
    """
    # Convert to Series if we've got a DataFrame, to ensure consistency
    if isinstance(rainfall, pd.DataFrame):
        rainfall = pd.Series(
            data=rainfall['rainfall'], index=rainfall.index
        )

    if erosion is None:
        erosion = ErosionParams()

    # Pre-compute severity masks based on dNBR thresholds
    dnbr_below_threshold = dnbr < erosion.dnbr_severity_threshold
    dnbr_above_threshold = dnbr >= erosion.dnbr_severity_threshold

    # Initialise timing variables for progress logging
    total_timesteps = len(rainfall.index)
    start_time = time.time()
    last_log_time = start_time
    iteration_count = 0

    # Loop over each 30-min timestep
    for timestep in rainfall.index:
        iteration_count += 1
        # Get rainfall depth (∆V_r) during the 30-min period
        delta_v_r = rainfall[timestep]

        # Initialise the result dict with zeros/default values
        result = {
            'total_rain': delta_v_r,
            'intensity': 0.0,
            'erosivity': 0.0,
            'RUSLE': np.zeros_like(klscp, dtype=np.float32),
            'delivered': np.zeros_like(klscp, dtype=np.float32),
            'RUSLE_below_threshold': np.zeros_like(
                klscp, dtype=np.float32
            ),
            'RUSLE_above_threshold': np.zeros_like(
                klscp, dtype=np.float32
            ),
            'delivered_below_threshold': np.zeros_like(
                klscp, dtype=np.float32
            ),
            'delivered_above_threshold': np.zeros_like(
                klscp, dtype=np.float32
            ),
        }

        # Skip RUSLE calculations for dry timesteps
        if delta_v_r == 0:
            yield (timestep, result)
            continue

        # Rainfall intensity (∆V_r / ∆t_r) in mm/hr, and the erosivity
        # factor (R) derived from it.
        intensity, R = rainfall_erosivity(
            delta_v_r, rate=erosion.kinetic_energy_coefficient)
        result['intensity'] = intensity
        result['erosivity'] = R

        # Total erosion in tonnes per hectare
        # TODO: sediment eroded? kg? t?
        RUSLE = (R * klscp) * cell_area_ha
        result['RUSLE'] = RUSLE

        # Sediment delivered to streams: erosion × SDR ratio
        # TODO: confirm units (tonnes?)
        delivered = RUSLE * sdr
        result['delivered'] = delivered

        # Apply dNBR severity masks
        result['RUSLE_below_threshold'] = np.where(
            dnbr_below_threshold, RUSLE, 0
        )
        result['RUSLE_above_threshold'] = np.where(
            dnbr_above_threshold, RUSLE, 0
        )
        result['delivered_below_threshold'] = np.where(
            dnbr_below_threshold, delivered, 0
        )
        result['delivered_above_threshold'] = np.where(
            dnbr_above_threshold, delivered, 0
        )

        # Log progress at the configured interval
        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
            progress_pct = (iteration_count / total_timesteps) * 100
            elapsed_time = current_time - start_time
            avg_time_per_iteration = elapsed_time / iteration_count
            remaining_iterations = total_timesteps - iteration_count
            estimated_time_remaining = (
                avg_time_per_iteration * remaining_iterations
            )

            # Format time remaining as HH:MM:SS or MM:SS
            hours, remainder = divmod(
                int(estimated_time_remaining), 3600
            )
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                time_str = (
                    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                )
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"

            logger.info(
                f"Progress: {iteration_count}/{total_timesteps} "
                f"timesteps ({progress_pct:.1f}%) - "
                f"Current timestep: {timestep} - "
                f"Estimated time remaining: {time_str}"
            )
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

def compute_particulates(rusle_df, constituents_df=None):
    """
    Add constituent load columns to a RUSLE results DataFrame.

    Multiplies the low- and high-severity RUSLE_SDR values by empirical
    constituent ratios from the ash_constituents lookup table to estimate
    particulate loads for each constituent.

    Parameters:
    - rusle_df: DataFrame of RUSLE results containing 'RUSLE_SDR
      (Low severity)' and 'RUSLE_SDR (High severity)' columns.
    - constituents_df: DataFrame of ash constituent ratios. If None,
      the built-in ash_constituents.csv package data is used.

    Returns:
    - The input DataFrame with additional columns for each constituent
      load in tonnes.
    """
    if constituents_df is None:
        constituents_df = load_package_data('ash_constituents.csv')

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


# ---------------------------------------------------------------------------
# Period helpers and recorder factory
# ---------------------------------------------------------------------------

_PERIOD_OFFSETS = {
    'yearly': pd.DateOffset(years=1),
    'quarterly': pd.DateOffset(months=3),
    'monthly': pd.DateOffset(months=1),
    'weekly': pd.DateOffset(weeks=1),
    'daily': pd.DateOffset(days=1),
}


def _calendar_floor(ts, granularity):
    """
    Snap a timestamp down to the start of its calendar period.

    yearly -> Jan 1; quarterly -> quarter start; monthly -> 1st;
    weekly -> Monday; daily -> midnight.
    """
    ts = pd.Timestamp(ts)
    if granularity == 'yearly':
        return pd.Timestamp(year=ts.year, month=1, day=1)
    if granularity == 'quarterly':
        month = ((ts.month - 1) // 3) * 3 + 1
        return pd.Timestamp(year=ts.year, month=month, day=1)
    if granularity == 'monthly':
        return pd.Timestamp(year=ts.year, month=ts.month, day=1)
    if granularity == 'weekly':
        return ts.normalize() - pd.Timedelta(days=ts.weekday())
    if granularity == 'daily':
        return ts.normalize()
    raise ValueError(f"Cannot calendar-floor granularity '{granularity}'.")


def _compute_periods(start, end, timestep_type, origin='calendar'):
    """
    Compute non-overlapping time-period boundaries for a simulation span.

    Parameters:
    - start: Start of the simulation period (pd.Timestamp).
    - end: End of the simulation period (pd.Timestamp).
    - timestep_type: Period granularity: 'total', 'yearly', 'quarterly',
      'monthly', 'weekly', or 'daily'.
    - origin: 'calendar' (default) snaps the first period to the calendar
      boundary for the granularity (so bins are calendar-aligned; the
      first bin may be partial); 'fire' starts the first period at
      ``start`` and steps by the offset (e.g. year-since-fire).

    Returns:
    - List of (period_start, period_end) tuples. The period_start is used
      as the time coordinate label, so calendar bins are labelled by their
      calendar boundary even when the first bin is partial. For non-final
      periods period_end is offset by -1 second so boundary timesteps are
      not double-counted across adjacent periods.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if timestep_type == 'total':
        return [(start, end)]

    offset = _PERIOD_OFFSETS.get(timestep_type)
    if offset is None:
        raise ValueError(
            f"Unsupported grid_timestep '{timestep_type}'. Use one of: "
            f"'total', {', '.join(repr(k) for k in _PERIOD_OFFSETS)}."
        )
    if origin not in ('calendar', 'fire'):
        raise ValueError(
            f"origin must be 'calendar' or 'fire'; got {origin!r}."
        )

    period_start = (
        start if origin == 'fire'
        else _calendar_floor(start, timestep_type)
    )

    periods = []
    while period_start < end:
        period_end = period_start + offset
        periods.append((period_start, min(period_end, end)))
        period_start = period_end

    # Offset non-final period ends by 1 s to avoid double-counting
    return [
        (ps, pe - pd.Timedelta(seconds=1))
        if i < len(periods) - 1 else (ps, pe)
        for i, (ps, pe) in enumerate(periods)
    ]


def _spatial_coords_from_transform(transform, shape):
    """Build easting/northing coordinate arrays from an affine transform."""
    if transform is None:
        return {}
    rows, cols = shape
    easting = np.array(
        [transform.c + (col + 0.5) * transform.a for col in range(cols)]
    )
    northing = np.array(
        [transform.f + (row + 0.5) * transform.e for row in range(rows)]
    )
    return {'easting': easting, 'northing': northing}


def record_timestep_grid(variable):
    """
    Build a recorder that captures a grid variable at every model timestep.

    Finalises to a 3-D xarray.DataArray (time, northing, easting) whose time
    coordinate holds the actual model timesteps. This keeps one grid slice
    per 30-minute timestep, so it is memory-heavy — intended for short
    windows or diagnostics.

    Parameters:
    - variable: Key to extract from the per-timestep data dict.

    Returns:
    - A recorder closure with .reset() and .finalize() methods.
    """
    grids = []
    times = []
    captured_transform = [None]

    def recorder(timestep, **kwargs):
        if captured_transform[0] is None and 'transform' in kwargs:
            captured_transform[0] = kwargs['transform']
        grids.append(kwargs[variable].copy())
        times.append(pd.Timestamp(timestep))

    def reset():
        grids.clear()
        times.clear()
        captured_transform[0] = None

    def finalize():
        import xarray as xr
        if not grids:
            return None
        spatial = _spatial_coords_from_transform(
            captured_transform[0], grids[0].shape)
        stacked = np.stack(grids, axis=0)
        return xr.DataArray(
            stacked,
            dims=['time', 'northing', 'easting'],
            coords={'time': times, **spatial},
        )

    recorder.reset = reset
    recorder.finalize = finalize
    return recorder


def record_multi_period_grid(variable, fn, periods):
    """
    Build a recorder that accumulates a summary grid for each time period.

    The finalised result is an xarray.DataArray with georeferenced
    easting and northing coordinates derived from the affine transform
    passed at each timestep. Single-period results are 2-D
    (northing, easting); multi-period results add a time dimension.

    Parameters:
    - variable: Key to extract from the per-timestep data dict.
    - fn: Summary function: 'sum', 'max', or 'mean'.
    - periods: List of (start, end) pd.Timestamp pairs defining each
      non-overlapping accumulation window.

    Returns:
    - A recorder closure with .reset() and .finalize() methods; finalize
      returns an xarray.DataArray of accumulated grids.
    """
    # One accumulator array and count per period
    grids = [None] * len(periods)
    counts = [0] * len(periods)
    captured_transform = [None]  # mutable container for nonlocal capture

    def recorder(timestep, **kwargs):
        data = kwargs[variable]
        if captured_transform[0] is None and 'transform' in kwargs:
            captured_transform[0] = kwargs['transform']
        for i, (ps, pe) in enumerate(periods):
            if timestep < ps or timestep > pe:
                continue
            counts[i] += 1
            if grids[i] is None:
                grids[i] = data.copy()
            elif fn == 'max':
                np.maximum(grids[i], data, out=grids[i])
            else:
                grids[i] += data

    def reset():
        for i in range(len(periods)):
            grids[i] = None
            counts[i] = 0
        captured_transform[0] = None

    def finalize():
        import xarray as xr

        # Find the grid shape from the first populated accumulator
        shape = None
        for g in grids:
            if g is not None:
                shape = g.shape
                break
        if shape is None:
            return None

        arrays = []
        for i in range(len(periods)):
            g = grids[i]
            if g is None:
                g = np.zeros(shape, dtype=np.float32)
            elif fn == 'mean' and counts[i] > 0:
                g = g / counts[i]
            arrays.append(g)

        spatial = _spatial_coords_from_transform(captured_transform[0], shape)

        if len(arrays) == 1:
            return xr.DataArray(
                arrays[0],
                dims=['northing', 'easting'],
                coords=spatial,
            )

        time_coords = [ps for ps, _ in periods]
        stacked = np.stack(arrays, axis=0)
        coords = {'time': time_coords, **spatial}
        return xr.DataArray(
            stacked,
            dims=['time', 'northing', 'easting'],
            coords=coords,
        )

    recorder.reset = reset
    recorder.finalize = finalize
    return recorder


# Model timestep used to convert timeseries_timestep to an agg_count


def default_rusle_recorders(
    include_grids=True,
    grid_variables=('RUSLE',),
    grid_fns=('sum', 'max'),
    grid_timesteps=('yearly',),
    grid_period_origin='calendar',
    include_timeseries=True,
    timeseries_variables=('RUSLE',),
    timeseries_fn='sum',
    timeseries_timestep='24h',
    timeseries_label_field=None,
    timeseries_mode='full',
    include_transform=True,
):
    """
    Configure RUSLE output recorders and return a factory function.

    The returned factory creates a fresh set of recorder closures each
    time it is called, so every simulation run or Dask task gets
    independent state. Because the factory closure captures only plain
    Python values, it is trivially serialisable for Dask.

    Grid recorders are built from the Cartesian product of
    grid_variables × grid_fns × grid_timesteps. Each combination
    produces one entry keyed as '{variable}_{fn}_{timestep}'
    (e.g. 'RUSLE_sum_yearly'). All grid results are xarray.DataArray
    objects with georeferenced easting and northing coordinates.

    Parameters:
    - include_grids: Whether to include grid summary recorders.
      Default True.
    - grid_variables: Variables from the RUSLE generator to record in
      summary grids. Default ('RUSLE',).
    - grid_fns: Summary functions per period. Supported: 'sum', 'max',
      'mean'. Default ('sum', 'max').
    - grid_timesteps: Temporal aggregation levels. Supported: 'total',
      'yearly', 'monthly'. Default ('yearly',).
    - include_timeseries: Whether to include subcatchment timeseries
      recorders. Default True.
    - timeseries_variables: Variables to record as subcatchment
      timeseries. One recorder per variable. Default ('RUSLE',).
    - timeseries_fn: Spatial aggregation function for the timeseries.
      Default 'sum'.
    - timeseries_timestep: Output timestep for timeseries rows, e.g.
      '24h', '1h', '12h'. Converted to an aggregation count using the
      30-min model timestep. Default '24h' (daily).
    - timeseries_label_field: Column in subcatchment boundaries to use
      as zone labels. If None, the integer index is used.
    - timeseries_mode: 'full' returns the complete DataFrame;
      'percentiles' finalises to a DataFrame of 101 percentiles per
      subcatchment; 'none' skips timeseries entirely.
    - include_transform: Whether to include the transform recorder.
      Default True.

    Returns:
    - factory: Callable (ctx, start, end) → dict of recorder
      closures ready for use with run_usle_simulation().
    ------------------------------------------------------------------------
    Notes:
    - Typical usage: make_recorders = default_rusle_recorders(), then
      recorders = make_recorders(project, '2020-01-01', '2021-12-31').
    - For ensemble runs with Dask, use timeseries_mode='percentiles' to
      avoid storing full daily timeseries per replicate.
    - timeseries_mode='none' is equivalent to include_timeseries=False.
    ------------------------------------------------------------------------
    """
    if timeseries_mode == 'none':
        include_timeseries = False

    # Convert timeseries_timestep to an agg_count
    ts_delta = pd.Timedelta(timeseries_timestep)
    agg_count = max(1, int(ts_delta / _MODEL_TIMESTEP))

    def factory(ctx, start, end):
        """Build and return a fresh dict of recorders for one simulation."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        recorders = {}

        # Grid recorders: Cartesian product of variables × fns × timesteps.
        # A grid_timesteps entry is either a granularity string ('yearly')
        # using grid_period_origin, or a (granularity, origin) tuple. The
        # special granularity 'timestep' records every model timestep.
        if include_grids:
            for ts_entry in grid_timesteps:
                if isinstance(ts_entry, (tuple, list)):
                    ts_type, origin = ts_entry[0], ts_entry[1]
                else:
                    ts_type, origin = ts_entry, grid_period_origin

                # Origin only qualifies periodic grids; suffix the key only
                # when the origin differs from the default so common keys
                # stay clean.
                if ts_type in ('total', 'timestep') or origin == grid_period_origin:
                    origin_suffix = ''
                else:
                    origin_suffix = f'_{origin}'

                periods = (
                    None if ts_type == 'timestep'
                    else _compute_periods(start, end, ts_type, origin=origin)
                )
                for variable in grid_variables:
                    for fn in grid_fns:
                        key = f'{variable}_{fn}_{ts_type}{origin_suffix}'
                        if ts_type == 'timestep':
                            recorders[key] = record_timestep_grid(variable)
                        else:
                            recorders[key] = record_multi_period_grid(
                                variable, fn, periods,
                            )

        # Transform recorder
        if include_transform:
            recorders['the_transform'] = record_grid_transform()

        # Subcatchment timeseries recorders
        if include_timeseries:
            _add_timeseries_recorders(ctx, recorders)

        return recorders

    def _add_timeseries_recorders(ctx, recorders):
        """Add subcatchment timeseries recorders to the recorders dict."""
        for ts_var in timeseries_variables:
            base_ts = record_subcatchment_timeseries(
                ctx,
                ts_var,
                fn=timeseries_fn,
                label_field=timeseries_label_field,
                agg_count=agg_count,
            )
            # Use the standard constant key when there is only one
            # variable, for backward compatibility; otherwise qualify
            # the key with the variable name
            if len(timeseries_variables) == 1:
                ts_key = c.RUSLE_OP_TIMESERIES_NAME
            else:
                ts_key = f'{ts_var}_{c.RUSLE_OP_TIMESERIES_NAME}'

            if timeseries_mode == 'full':
                recorders[ts_key] = base_ts

            elif timeseries_mode == 'percentiles':
                def _make_percentiles(base):
                    def pct_recorder(timestep, **data):
                        return base(timestep, **data)

                    def _reset():
                        base.reset()

                    def _finalize():
                        df = base.finalize()
                        if df is None or df.empty:
                            return pd.DataFrame()
                        pctiles = np.arange(101)
                        result = df.apply(
                            lambda col: np.percentile(col, pctiles)
                        )
                        result.index = pctiles
                        result.index.name = 'percentile'
                        return result

                    pct_recorder.reset = _reset
                    pct_recorder.finalize = _finalize
                    return pct_recorder

                recorders[ts_key + '_percentiles'] = (
                    _make_percentiles(base_ts)
                )

            else:
                raise ValueError(
                    f"Unsupported timeseries_mode='{timeseries_mode}'. "
                    "Use 'full', 'percentiles', or 'none'."
                )

    return factory
