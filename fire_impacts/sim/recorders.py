"""
Recorders: the closures that turn per-timestep simulation output into the
grids and time series a run actually keeps.

Each recorder is a callable taking (timestep, **data) with .reset() and
.finalize() attached. They are built fresh per run, so a replicate running
in parallel gets its own state.
"""

import logging

import numpy as np
import pandas as pd
import rasterio.features

from fire_impacts import const as c
from fire_impacts.context import RunContext
from fire_impacts.sim.scaled_grid import ScaledGrid

logger = logging.getLogger(__name__)


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


def _zonal_weights(unit, rain_index, zone_positions):
    """
    Reduce a static layer to per-(rain cell, subcatchment) weights.

    Parameters:
    - unit: the static per-cell layer.
    - rain_index: None for uniform rainfall, else each cell's rain cell.
    - zone_positions: flat cell positions of each subcatchment.

    Returns:
    - Tuple of (sums, counts, maxima). `sums` has shape
      (n_rain_cells, n_zones); `counts` and `maxima` have one entry per
      zone, and `maxima` is only meaningful for uniform rainfall.
    ------------------------------------------------------------------------
    Notes:
    - A zonal sum and mean are linear in the scale vector, so they fold
      into a matrix product. A maximum does not, which is why the caller
      falls back to materialising when there is more than one rain cell.
    ------------------------------------------------------------------------
    """
    flat = unit.reshape(-1)
    n_rain = 1 if rain_index is None else int(rain_index.max()) + 1
    n_zones = len(zone_positions)
    sums = np.zeros((n_rain, n_zones))
    counts = np.zeros(n_zones)
    maxima = np.full(n_zones, np.nan)

    flat_rain = None if rain_index is None else rain_index.reshape(-1)
    for z, positions in enumerate(zone_positions):
        values = flat[positions]
        keep = ~np.isnan(values)
        counts[z] = keep.sum()
        if counts[z]:
            maxima[z] = values[keep].max()
        if rain_index is None:
            sums[0, z] = values[keep].sum()
        else:
            sums[:, z] = np.bincount(
                flat_rain[positions][keep], weights=values[keep],
                minlength=n_rain)
    return sums, counts, maxima


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
      Default is 'sum'. Raises ValueError immediately if given anything
      else.
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
    - Timesteps flagged dry are counted but not accumulated; a window of
      nothing but dry timesteps records a row of zeros.
    - Per-timestep data may be a ScaledGrid instead of an ndarray. When
      it is, the recorder accumulates only its (unit, rain_index, scale)
      triple across the window; a grid is materialised only when a
      window cannot be collapsed straight to per-zone values (a window
      spanning a recovery boundary, or fn='max' with spatially varying
      rainfall) - never on every timestep.
    - Distinct layers are matched by the OBJECT IDENTITY of `unit` (and
      of `rain_index`), never by value equality. A producer must reuse
      the same unit array for every timestep of one recovery segment
      and hand over a new object at a segment boundary.
    - fn='max' on the collapsed path assumes the window's accumulated
      scale is non-negative: it takes the max of `unit` within each
      zone once and multiplies by the (uniform) scale afterwards, which
      only equals the max of the per-timestep grids when the scale
      cannot flip the sign of the comparison. True in production, since
      RUSLE erosivity accumulates non-negative.
    - fn='max' with spatially varying rainfall (a non-None rain_index)
      never collapses, even for a single-layer window - a zonal max is
      not linear in the scale vector - so that combination always
      materialises the window's grid and reduces it directly.
    ------------------------------------------------------------------------
    """
    if fn not in ('sum', 'mean', 'max'):
        raise ValueError(f"Function {fn} not recognized.")

    result = None
    index = None
    zone_indices = None
    zone_names = None

    intermediate = None
    intermediate_count = 0

    # One [unit, rain_index, scale] entry per distinct layer the current
    # aggregation window has seen (ScaledGrid input only); one layer per
    # recovery segment, so this stays tiny.
    window = []
    # Zonal weights already built for a layer, cached across the whole
    # run (not just one window) and matched by object identity of both
    # unit and rain_index, never by value.
    weights = []

    def _weights_for(unit, rain_index):
        """Zonal weights for this layer, built once per segment."""
        for cached_unit, cached_rain_index, cached in weights:
            if cached_unit is unit and cached_rain_index is rain_index:
                return cached
        cached = _zonal_weights(unit, rain_index, zone_indices)
        weights.append((unit, rain_index, cached))
        return cached

    def _collapsed_row(unit, rain_index, scale):
        """
        Reduce one window's accumulated scale straight to a list of
        zone values, without materialising a grid.

        Returns:
        - A list of one aggregated value per zone, or None when fn is
          'max' and rainfall is spatially varying - a zonal max is not
          linear in the scale vector, so the caller must materialise.
        """
        sums, counts, maxima = _weights_for(unit, rain_index)
        if fn == 'max':
            if rain_index is None:
                return list(scale[0] * maxima)
            return None          # caller must materialise instead
        totals = scale @ sums
        if fn == 'mean':
            with np.errstate(invalid='ignore', divide='ignore'):
                return list(np.where(counts > 0, totals / counts,
                                      np.nan))
        return list(totals)

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
        nonlocal zone_indices, zone_names
        nonlocal intermediate, intermediate_count

        data = kwargs.get(variable_name)
        # Raise an error if the requested variable isn't in the data
        if data is None:
            raise ValueError(
                f"Variable {variable_name} not found in simulation data."
            )
        dry = kwargs.get('dry', False)

        # On the first call, build zone masks from the subcatchment
        # boundaries. Fall back to the whole catchment boundary if no
        # subcatchments have been registered.
        if zone_indices is None:
            try:
                boundaries_v = ctx.project.get_subcatchments(ctx.catchment)
            except FileNotFoundError:
                boundaries_v = ctx.project.catchment_boundary(ctx.catchment)
            resolved_label = label_field
            if resolved_label is None:
                resolved_label = ctx.project.subcatchment_label_field(
                    ctx.catchment,
                )
            # Rasterise each subcatchment polygon separately, then keep
            # only the flat positions of the cells it covers. Holding
            # indices rather than one full-grid mask per zone turns the
            # aggregation below from a pass over the whole grid per zone
            # into a single pass over the catchment.
            zone_indices = [
                np.flatnonzero(
                    ~np.isnan(
                        rasterio.features.rasterize(
                            [g],
                            transform=transform,
                            fill=np.nan,
                            dtype=np.float32,
                            out_shape=data.shape,
                        )
                    )
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

        # Accumulate data into the current aggregation cycle. A dry
        # timestep contributes nothing - and, for an eager grid, the
        # buffer it carries is shared and read-only. It still advances
        # the cycle, so the output cadence is unchanged. A ScaledGrid
        # is never materialised here: only its (unit, rain_index,
        # scale) triple is accumulated into the window, matched to an
        # existing entry by the object identity of both unit and
        # rain_index, never by value.
        intermediate_count += 1
        if not dry:
            if isinstance(data, ScaledGrid):
                for entry in window:
                    if (entry[0] is data.unit
                            and entry[1] is data.rain_index):
                        entry[2] += data.scale
                        break
                else:
                    window.append([data.unit, data.rain_index,
                                   data.scale.astype(np.float64)])
            elif intermediate is None:
                # Copy: the simulation reuses the buffer it handed us.
                intermediate = data.copy()
            else:
                intermediate += data

        # Return early if we haven't reached the requested agg_count yet
        if intermediate_count < agg_count:
            return result

        # Flush the accumulated data and reset the intermediate state
        flushed_intermediate = intermediate
        flushed_window = list(window)
        intermediate = None
        window.clear()
        intermediate_count = 0

        if index is None:
            index = []
        index.append(timestep)

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

        grouped = None
        if flushed_intermediate is None and not flushed_window:
            # Every timestep in this window was dry, so every zone
            # eroded nothing. Short-circuit rather than multiply a
            # zero scale through the weights: W is NaN for a zone
            # whose cells are all NaN, and 0 * NaN is NaN.
            grouped = [0.0] * len(zone_indices)
        elif flushed_intermediate is None and len(flushed_window) == 1:
            grouped = _collapsed_row(*flushed_window[0])

        if grouped is None:
            # A window spanning a recovery boundary, eager input, or a
            # maximum over more than one rain cell (_collapsed_row
            # declined). Materialising is exact for every aggregation
            # function, and at a boundary it can only happen once.
            total = flushed_intermediate
            for unit, rain_index, scale in flushed_window:
                part = _spread(scale, rain_index, unit)
                total = part if total is None else total + part
            flat = np.asarray(total).reshape(-1)
            grouped = [agg(flat[positions]) for positions in zone_indices]

        for ix, name in enumerate(zone_names):
            result[name].append(grouped[ix])

        return result

    # -----------------------------------------------------------------------
    def reset():
        """Reset all accumulated state back to initial values."""
        nonlocal result, index
        nonlocal zone_indices, zone_names
        nonlocal intermediate, intermediate_count
        index = None
        zone_indices = None
        zone_names = None
        result = None
        intermediate = None
        intermediate_count = 0
        window.clear()
        weights.clear()

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


def _spread(scale, rain_index, unit):
    """Turn an accumulated scale back into a grid."""
    if rain_index is None:
        return scale[0] * unit
    return scale[rain_index] * unit


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
    ------------------------------------------------------------------------
    Notes:
    - A timestep flagged dry contributes an all-zero grid, which changes
      neither a sum nor a max of non-negative erosion. It is counted but
      not accumulated, so it still divides a mean correctly.
    - Per-timestep data may be a ScaledGrid instead of an ndarray. When
      it is, the recorder accumulates only its (unit, rain_index, scale)
      triple, and the grid itself is built once, at finalize() - not
      materialised on every timestep.
    - Distinct layers are matched by the OBJECT IDENTITY of `unit` (and
      of `rain_index`), never by value equality. A producer must reuse
      the same unit array for every timestep of one recovery segment
      and hand over a new object at a segment boundary.
    - fn='max' assumes every unit cell is non-negative: it takes the max
      of the accumulated scale and multiplies by unit afterwards, and
      that only equals the max of the per-timestep grids when unit
      cannot flip the sign of the comparison. True in production, since
      unit is a product of non-negative RUSLE factors.
    ------------------------------------------------------------------------
    """
    # Per period: an eager grid accumulator, plus one (unit, rain_index,
    # scale) entry for each distinct layer the period has seen. There is
    # one layer per recovery segment, so the list stays tiny.
    grids = [None] * len(periods)
    scales = [[] for _ in periods]
    counts = [0] * len(periods)
    captured_transform = [None]  # mutable container for nonlocal capture
    captured_shape = [None]

    def _accumulate_scale(index, data):
        for entry in scales[index]:
            if entry[0] is data.unit and entry[1] is data.rain_index:
                if fn == 'max':
                    np.maximum(entry[2], data.scale, out=entry[2])
                else:
                    entry[2] += data.scale
                return
        scales[index].append(
            [data.unit, data.rain_index, data.scale.astype(np.float64)])

    def _accumulate_grid(index, data):
        if grids[index] is None:
            grids[index] = data.copy()
        elif fn == 'max':
            np.maximum(grids[index], data, out=grids[index])
        else:
            grids[index] += data

    def recorder(timestep, **kwargs):
        data = kwargs[variable]
        dry = kwargs.get('dry', False)
        if captured_transform[0] is None and 'transform' in kwargs:
            captured_transform[0] = kwargs['transform']
        if captured_shape[0] is None:
            captured_shape[0] = data.shape
        for i, (ps, pe) in enumerate(periods):
            if timestep < ps or timestep > pe:
                continue
            counts[i] += 1
            # A dry timestep erodes nothing. Registering its layer anyway
            # would widen the finalised NaN mask to the union of every
            # segment's, rather than only those that contributed.
            if dry:
                continue
            if isinstance(data, ScaledGrid):
                _accumulate_scale(i, data)
            else:
                _accumulate_grid(i, data)

    def reset():
        for i in range(len(periods)):
            grids[i] = None
            scales[i] = []
            counts[i] = 0
        captured_transform[0] = None
        captured_shape[0] = None

    def _combine(index):
        """Fold this period's accumulators into one grid, or None."""
        parts = []
        if grids[index] is not None:
            parts.append(grids[index])
        parts.extend(
            _spread(scale, rain_index, unit)
            for unit, rain_index, scale in scales[index]
        )
        if not parts:
            return None
        out = parts[0].copy()
        for part in parts[1:]:
            if fn == 'max':
                np.maximum(out, part, out=out)
            else:
                out += part
        if fn == 'mean' and counts[index] > 0:
            out = out / counts[index]
        return out

    def finalize():
        import xarray as xr

        arrays = [_combine(i) for i in range(len(periods))]

        # Find the grid shape from the first populated accumulator,
        # falling back to the shape seen at the first timestep - every
        # period can be dry, and that is still a result of zeros rather
        # than no result at all.
        shape = None
        for a in arrays:
            if a is not None:
                shape = a.shape
                break
        if shape is None:
            shape = captured_shape[0]
        if shape is None:
            return None

        arrays = [
            a if a is not None else np.zeros(shape, dtype=np.float32)
            for a in arrays
        ]

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
    agg_count = max(1, int(ts_delta / c.MODEL_TIMESTEP))

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
