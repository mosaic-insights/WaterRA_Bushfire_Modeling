"""
Post-processing and visualisation of ensemble simulation results.

Provides functions for computing exceedance probability grids from
replicate runs and producing publication-quality maps.
"""

import warnings

import numpy as np
import xarray as xr
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ===================================================================
# Grid extraction helpers
# ===================================================================

def _extract_grid(result_dict, result_key, time=None):
    """Extract a 2-D numpy array from a single replicate's result dict.

    Handles both 2-D numpy arrays and 3-D xarray DataArrays.  For 3-D
    arrays, *time* selects a slice:

    - ``None`` — use the sole time step (error if more than one).
    - ``int`` — positional index (``isel``).
    - other — coordinate label (``sel``).

    Returns
    -------
    numpy.ndarray (2-D)
    """
    data = result_dict[result_key]
    if data is None:
        raise ValueError(f"Result key '{result_key}' is None.")

    if isinstance(data, xr.DataArray):
        if data.ndim == 2:
            return data.values
        if data.ndim == 3:
            if time is None:
                if data.sizes['time'] == 1:
                    return data.isel(time=0).values
                raise ValueError(
                    f"Result '{result_key}' has {data.sizes['time']} time "
                    f"steps — pass 'time' to select one."
                )
            if isinstance(time, (int, np.integer)):
                return data.isel(time=time).values
            return data.sel(time=time).values
        raise ValueError(f"Unexpected DataArray ndim={data.ndim}")

    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"Result '{result_key}' has unexpected shape {arr.shape}. "
        f"Expected a 2-D array or 3-D xarray DataArray."
    )


def _resolve_catchment(sample_replicate, catchment):
    """Pick the catchment name, defaulting when there's only one."""
    catchment_names = [
        k for k in sample_replicate
        if isinstance(sample_replicate[k], dict)
    ]
    if catchment is not None:
        if catchment not in catchment_names:
            raise KeyError(
                f"Catchment '{catchment}' not found. "
                f"Available: {catchment_names}"
            )
        return catchment
    if len(catchment_names) == 1:
        return catchment_names[0]
    raise ValueError(
        f"Multiple catchments in results — specify one of: "
        f"{catchment_names}"
    )


def _iter_replicates(ensemble_results):
    """Yield per-replicate result dicts from either a dict or list."""
    if isinstance(ensemble_results, dict):
        yield from ensemble_results.values()
    else:
        yield from ensemble_results


# ===================================================================
# Exceedance probability computation
# ===================================================================

def exceedance_probability(
    ensemble_results,
    result_key,
    threshold,
    catchment=None,
    time=None,
):
    """Compute per-pixel probability of exceeding a threshold across
    ensemble members.

    Parameters
    ----------
    ensemble_results : dict or list
        Output of :func:`run_rusle_all_replicates` (dict keyed by
        replicate index) or a plain list of per-replicate result dicts.
        Each replicate is ``{catchment_name: {result_key: grid, ...}}``.
    result_key : str
        Key identifying the gridded result to analyse, e.g.
        ``'RUSLE_sum_yearly'``.
    threshold : float
        Value to test exceedance against, in the same units as the
        grid (e.g. tonnes/ha).
    catchment : str or None
        Catchment name.  If *None* and only one catchment exists in
        the results, it is used automatically.
    time : int, timestamp, or None
        For 3-D xarray results (e.g. yearly grids), selects which time
        slice to analyse.  Pass an ``int`` for positional indexing or a
        coordinate label (e.g. a ``pd.Timestamp``).  *None* works when
        the result has exactly one time step.

    Returns
    -------
    xarray.DataArray
        2-D grid of exceedance probabilities (0–1) with dims
        ``(y, x)``.

    Examples
    --------
    Probability that year-1 erosion exceeds 0.25 t/ha::

        prob = exceedance_probability(
            results, 'RUSLE_sum_yearly', 0.25, time=0
        )
        plot_exceedance(prob, project=proj, catchment='Thomson')
    """
    replicates = list(_iter_replicates(ensemble_results))
    if not replicates:
        raise ValueError("No replicates in ensemble_results.")

    catchment = _resolve_catchment(replicates[0], catchment)

    # Stack grids from all replicates
    grids = []
    for rep in replicates:
        grid = _extract_grid(rep[catchment], result_key, time=time)
        grids.append(grid)

    stacked = np.stack(grids, axis=0)  # (n_replicates, rows, cols)
    n = stacked.shape[0]
    logger.info(
        "Computing exceedance P(X > %.4g) from %d replicates, "
        "grid shape %s",
        threshold, n, stacked.shape[1:],
    )

    # Count exceedances, treating NaN pixels as never exceeding
    exceed_count = np.nansum(stacked > threshold, axis=0).astype(np.float32)

    # Count valid (non-NaN) replicates per pixel
    valid_count = np.sum(~np.isnan(stacked), axis=0).astype(np.float32)

    # Probability = exceedances / valid replicates
    with np.errstate(invalid='ignore'):
        prob = np.where(valid_count > 0, exceed_count / valid_count, np.nan)

    return xr.DataArray(
        prob.astype(np.float32),
        dims=['y', 'x'],
        attrs={
            'result_key': result_key,
            'threshold': threshold,
            'n_replicates': n,
            'description': f'P({result_key} > {threshold})',
        },
    )


def ensemble_statistic(
    ensemble_results,
    result_key,
    statistic='mean',
    catchment=None,
    time=None,
):
    """Compute a per-pixel summary statistic across ensemble members.

    Parameters
    ----------
    ensemble_results : dict or list
        Same format as :func:`exceedance_probability`.
    result_key : str
        Key identifying the gridded result.
    statistic : str
        One of ``'mean'``, ``'median'``, ``'std'``, ``'min'``,
        ``'max'``, or ``'cv'`` (coefficient of variation).
    catchment : str or None
        Catchment name.
    time : int, timestamp, or None
        Time selector for 3-D results.

    Returns
    -------
    xarray.DataArray
        2-D grid with the requested statistic.
    """
    replicates = list(_iter_replicates(ensemble_results))
    catchment = _resolve_catchment(replicates[0], catchment)

    grids = [
        _extract_grid(rep[catchment], result_key, time=time)
        for rep in replicates
    ]
    stacked = np.stack(grids, axis=0)

    stat_fns = {
        'mean': lambda s: np.nanmean(s, axis=0),
        'median': lambda s: np.nanmedian(s, axis=0),
        'std': lambda s: np.nanstd(s, axis=0),
        'min': lambda s: np.nanmin(s, axis=0),
        'max': lambda s: np.nanmax(s, axis=0),
        'cv': lambda s: np.nanstd(s, axis=0) / np.nanmean(s, axis=0),
    }
    if statistic not in stat_fns:
        raise ValueError(
            f"Unknown statistic '{statistic}'. "
            f"Use one of: {list(stat_fns.keys())}"
        )

    result = stat_fns[statistic](stacked).astype(np.float32)

    return xr.DataArray(
        result,
        dims=['y', 'x'],
        attrs={
            'result_key': result_key,
            'statistic': statistic,
            'n_replicates': len(replicates),
        },
    )


# ===================================================================
# Plotting
# ===================================================================

def _dem_meta(project, catchment):
    """Read the catchment DEM metadata (transform + CRS)."""
    from ..pre.util import read_raster
    _, meta = read_raster(
        project.catchment_path(catchment, 'Topography', 'DEM.tif')
    )
    return meta


def _resolve_transform(project, catchment, transform):
    """Get the affine transform from the project DEM if not provided."""
    if transform is not None:
        return transform
    if project is not None and catchment is not None:
        return _dem_meta(project, catchment)['transform']
    return None


def _extent_from_transform(transform, shape):
    """Compute imshow extent (left, right, bottom, top) from an affine."""
    if transform is None:
        return None
    rows, cols = shape
    left = transform.c
    top = transform.f
    right = left + cols * transform.a
    bottom = top + rows * transform.e
    return (left, right, bottom, top)


def plot_grid(
    grid,
    project=None,
    catchment=None,
    transform=None,
    ax=None,
    title=None,
    cmap='plasma',
    vmin=None,
    vmax=None,
    cbar_label='',
    cbar_ticks=None,
    cbar_ticklabels=None,
    boundary_color='#333333',
    boundary_linewidth=1.0,
    figsize=(8, 6),
):
    """Plot a 2-D grid as a georeferenced map with colorbar and optional
    catchment boundary overlay.

    This is the general-purpose plotting function used by the
    convenience wrappers :func:`plot_exceedance` and
    :func:`plot_ensemble_grid`.

    Parameters
    ----------
    grid : xarray.DataArray or numpy.ndarray
        2-D grid to plot.
    project : FireImpactsProject or None
        If provided (along with *catchment*), overlays the catchment
        boundary and infers the transform from the DEM.
    catchment : str or None
        Catchment name for boundary overlay and transform lookup.
    transform : affine.Affine or None
        Georeferencing transform.  If *None*, inferred from the
        project DEM when *project* and *catchment* are given.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If *None*, a new figure is created.
    title : str or None
        Plot title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Colorbar range.
    cbar_label : str
        Label for the colorbar.
    cbar_ticks : list or None
        Explicit colorbar tick positions.
    cbar_ticklabels : list or None
        Labels corresponding to *cbar_ticks*.
    boundary_color : str
        Color for the catchment boundary line.
    boundary_linewidth : float
        Line width for the catchment boundary.
    figsize : tuple
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if isinstance(grid, xr.DataArray):
        data = grid.values
    else:
        data = np.asarray(grid)

    # Resolve georeferencing (transform + CRS for boundary reproject)
    dem_meta = None
    if transform is None and project is not None and catchment is not None:
        dem_meta = _dem_meta(project, catchment)
        transform = dem_meta['transform']
    extent = _extent_from_transform(transform, data.shape)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Colormap with NaN rendered as neutral grey so outside-catchment
    # pixels don't dominate the figure.
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('0.88')
    masked = np.ma.masked_invalid(data)

    im = ax.imshow(
        masked,
        extent=extent,
        origin='upper',
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    # Catchment boundary overlay
    if project is not None and catchment is not None:
        boundary = project.catchment_boundary(catchment)
        if dem_meta is None:
            dem_meta = _dem_meta(project, catchment)
        boundary = boundary.to_crs(dem_meta['crs'])
        boundary.boundary.plot(
            ax=ax,
            color=boundary_color,
            linewidth=boundary_linewidth,
        )

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3.5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)

    if title:
        ax.set_title(title)

    if extent is not None:
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        # Plain-format tick labels — avoid scientific notation on
        # projected CRS coordinates which is hard to read on maps.
        fmt = mticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax


def plot_exceedance(prob_grid, title=None, cmap='RdYlGn_r', **kwargs):
    """Plot an exceedance probability grid as a map.

    Thin wrapper around :func:`plot_grid` with probability-appropriate
    defaults (0–1 range, percentage colorbar labels).

    Parameters
    ----------
    prob_grid : xarray.DataArray or numpy.ndarray
        2-D grid of probabilities (0–1), as returned by
        :func:`exceedance_probability`.
    title : str or None
        Plot title.  If *None*, auto-generated from ``prob_grid.attrs``.
    cmap : str
        Colormap.  Default ``'RdYlGn_r'`` (red = high probability).
    **kwargs :
        Passed to :func:`plot_grid` (``project``, ``catchment``,
        ``transform``, ``ax``, ``figsize``, ``boundary_color``, etc.).

    Returns
    -------
    matplotlib.axes.Axes
    """
    attrs = prob_grid.attrs if isinstance(prob_grid, xr.DataArray) else {}

    if title is None:
        desc = attrs.get('description', '')
        n = attrs.get('n_replicates', '?')
        title = f'{desc}  (n={n})'

    return plot_grid(
        prob_grid,
        title=title,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_label='Exceedance probability',
        cbar_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        cbar_ticklabels=['0%', '25%', '50%', '75%', '100%'],
        **kwargs,
    )


def plot_ensemble_grid(grid, units='', title=None, cmap='plasma', **kwargs):
    """Plot an ensemble summary grid (mean, median, std, etc.).

    Thin wrapper around :func:`plot_grid` with auto-generated title
    from xarray attrs.

    Parameters
    ----------
    grid : xarray.DataArray or numpy.ndarray
        2-D grid to plot, e.g. from :func:`ensemble_statistic`.
    units : str
        Colorbar label (e.g. ``'t/ha'``).
    title : str or None
        If *None*, auto-generated from ``grid.attrs``.
    cmap : str
        Colormap.  Default ``'plasma'``.
    **kwargs :
        Passed to :func:`plot_grid`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    attrs = grid.attrs if isinstance(grid, xr.DataArray) else {}

    if title is None:
        stat = attrs.get('statistic', '')
        key = attrs.get('result_key', '')
        if stat or key:
            title = f'{stat} {key}'.strip()

    return plot_grid(
        grid,
        title=title,
        cmap=cmap,
        cbar_label=units,
        **kwargs,
    )


# ===================================================================
# Higher-level ensemble views
# ===================================================================

def _stack_ensemble_grids(ensemble_results, result_key, catchment=None, time=None):
    replicates = list(_iter_replicates(ensemble_results))
    if not replicates:
        raise ValueError("No replicates in ensemble_results.")
    catchment = _resolve_catchment(replicates[0], catchment)
    grids = [
        _extract_grid(rep[catchment], result_key, time=time)
        for rep in replicates
    ]
    return np.stack(grids, axis=0), catchment


def plot_ensemble_statistics_panel(
    ensemble_results,
    result_key,
    catchment=None,
    time=None,
    project=None,
    cell_area_ha=None,
    units='',
    cmap='YlOrRd',
    vmax_percentile=99,
    suptitle=None,
    figsize=(16, 5.5),
):
    """Render a three-panel map of ensemble statistics (median, 90th
    percentile, IQR) for a gridded result key.

    Parameters
    ----------
    ensemble_results : dict or list
        Output of :func:`run_rusle_all_replicates`.
    result_key : str
        Gridded result key (e.g. ``'RUSLE_sum_yearly'``).
    catchment : str or None
    time : int, timestamp, or None
        Time slice for 3-D xarray results.
    project : FireImpactsProject or None
        When provided, adds georeferenced axes and boundary overlays.
    cell_area_ha : float or None
        If provided, grid values are divided by this area to convert
        per-cell totals into an area-normalised rate (e.g. t/cell → t/ha).
    units : str
        Colorbar label (units after any *cell_area_ha* conversion).
    cmap : str
    vmax_percentile : float
        Upper clip (percentile across replicates and pixels) for shared
        colour scale — prevents single extreme cells from dominating.
    suptitle : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    stacked, catchment = _stack_ensemble_grids(
        ensemble_results, result_key, catchment=catchment, time=time,
    )
    if cell_area_ha is not None:
        stacked = stacked / cell_area_ha

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(stacked, axis=0)
        p90 = np.nanpercentile(stacked, 90, axis=0)
        iqr = (
            np.nanpercentile(stacked, 75, axis=0)
            - np.nanpercentile(stacked, 25, axis=0)
        )
        vmax = float(np.nanpercentile(stacked, vmax_percentile))

    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    n = stacked.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    if suptitle is None:
        suptitle = f'{result_key} — ensemble statistics (n={n} replicates)'
    fig.suptitle(suptitle, fontsize=13)

    panels = [
        (median, 'Median'),
        (p90, '90th percentile'),
        (iqr, 'IQR (P75 − P25)'),
    ]
    for ax, (data, subtitle) in zip(axes, panels):
        plot_grid(
            data,
            project=project,
            catchment=catchment,
            ax=ax,
            title=subtitle,
            cmap=cmap,
            vmin=norm.vmin,
            vmax=norm.vmax,
            cbar_label=units,
        )
    return fig


def catchment_total_per_replicate(
    ensemble_results,
    result_key,
    catchment=None,
    time=None,
):
    """Sum a gridded result over the catchment for every replicate.

    Returns
    -------
    numpy.ndarray (1-D, length = n_replicates)
        Spatial total for each replicate, NaN-aware.
    """
    stacked, _ = _stack_ensemble_grids(
        ensemble_results, result_key, catchment=catchment, time=time,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nansum(stacked, axis=(1, 2))


def plot_catchment_exceedance_curve(
    ensemble_results,
    result_key,
    catchment=None,
    time=None,
    scale=1.0,
    value_units='',
    ax=None,
    title=None,
    color='steelblue',
    figsize=(7, 5),
):
    """Flood-frequency-style exceedance curve for a catchment-total
    result across replicates.

    Parameters
    ----------
    ensemble_results : dict or list
    result_key : str
    catchment : str or None
    time : int, timestamp, or None
    scale : float
        Multiplier applied to the catchment totals before plotting
        (e.g. ``1e-3`` to show kilotonnes).
    value_units : str
        Units label for the y-axis (after *scale*).
    ax : matplotlib.axes.Axes or None
    title : str or None
    color : str
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    totals = catchment_total_per_replicate(
        ensemble_results, result_key, catchment=catchment, time=time,
    )
    totals = np.asarray(totals) * scale
    n = len(totals)
    sorted_totals = np.sort(totals)[::-1]
    # Weibull plotting positions
    aep = np.arange(1, n + 1) / (n + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(aep * 100, sorted_totals, 'o-', color=color, lw=2, ms=7)
    ax.set_xlabel('Annual exceedance probability (%)')
    ax.set_ylabel(f'Catchment total ({value_units})' if value_units else 'Catchment total')
    if title is None:
        title = f'Exceedance curve — {result_key} (n={n})'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


def plot_ensemble_daily_ribbon(
    ensemble_results,
    catchment=None,
    timeseries_key='erosion_daily_time_series',
    resample='D',
    ax=None,
    title=None,
    ylabel='Daily sediment yield (t)',
    color='steelblue',
    figsize=(14, 5),
):
    """Spread plot of a spatially-summed daily timeseries across
    replicates — median line with IQR and P10–P90 ribbons.

    Parameters
    ----------
    ensemble_results : dict or list
    catchment : str or None
    timeseries_key : str
        Key under each catchment's results holding a timeseries
        DataFrame (rows = time, columns = subcatchments).
    resample : str
        Pandas resample rule applied to the spatial sum.  Default ``'D'``.
    ax : matplotlib.axes.Axes or None
    title : str or None
    ylabel : str
    color : str
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes or None
        ``None`` if no replicate exposes *timeseries_key*.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    replicates = list(_iter_replicates(ensemble_results))
    if not replicates:
        raise ValueError("No replicates in ensemble_results.")
    catchment = _resolve_catchment(replicates[0], catchment)

    daily_totals = {}
    for i, rep in enumerate(replicates):
        ts = rep[catchment].get(timeseries_key)
        if ts is None:
            continue
        series = ts.sum(axis=1)
        if resample is not None:
            series = series.resample(resample).sum()
        daily_totals[i] = series

    if not daily_totals:
        logger.warning(
            "No replicate exposes timeseries key '%s' — skipping ribbon plot.",
            timeseries_key,
        )
        return None

    daily_df = pd.DataFrame(daily_totals)
    p10 = daily_df.quantile(0.10, axis=1)
    p25 = daily_df.quantile(0.25, axis=1)
    p50 = daily_df.median(axis=1)
    p75 = daily_df.quantile(0.75, axis=1)
    p90 = daily_df.quantile(0.90, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    for col in daily_df.columns:
        ax.plot(daily_df.index, daily_df[col], color=color, alpha=0.15, lw=0.8)
    ax.fill_between(p10.index, p10, p90, alpha=0.20, color=color, label='P10–P90')
    ax.fill_between(p25.index, p25, p75, alpha=0.35, color=color, label='IQR (P25–P75)')
    ax.plot(p50.index, p50, color='darkblue', lw=2, label='Median')

    ax.set_ylabel(ylabel)
    ax.set_title(
        title
        or f'Ensemble daily timeseries — {timeseries_key} '
        f'(n={len(daily_totals)} replicates)'
    )
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.4)
    return ax


def _subcatchment_label_map(project, catchment, label_field):
    """Return a ``{sc_ID: label}`` dict for *catchment*, or ``None`` if
    unavailable.  Silently returns ``None`` when no subcatchment layer
    is registered, when *label_field* is falsy, or when the field is
    absent — callers fall back to the raw ``sc_ID`` columns."""
    if project is None or not label_field:
        return None
    try:
        subs = project.get_subcatchments(catchment)
    except Exception:
        return None
    id_col = getattr(project, 'subcatchment_id', 'sc_ID')
    if id_col not in subs.columns or label_field not in subs.columns:
        return None
    return dict(zip(subs[id_col], subs[label_field]))


def _apply_label_map(df, label_map):
    if label_map is None:
        return df
    renamed = df.rename(columns=label_map)
    # drop columns whose label we don't have a mapping for — keeps the
    # output clean when the subcatchment shapefile is a subset of the
    # RUSLE/debris zones.
    keep = [c for c in renamed.columns if c in set(label_map.values())]
    if not keep:
        return renamed
    return renamed[keep]


_SENTINEL = object()


def combine_rusle_and_debris_subcatchment(
    rusle_results,
    debris_subcatchment_ts,
    project=None,
    catchment=None,
    freq='YS',
    subcatchment_label_field=_SENTINEL,
    rusle_timeseries_key='erosion_daily_time_series',
    rusle_scale=1000.0,
):
    """Combine RUSLE and debris-flow subcatchment timeseries across the
    ensemble, aggregated to the requested temporal resolution and
    labelled by a string subcatchment attribute.

    Parameters
    ----------
    rusle_results : dict
        ``{replicate: {catchment: {<timeseries_key>: DataFrame, ...}}}``
        as produced by :func:`run_rusle_all_replicates`.  Column keys
        in the RUSLE timeseries are numeric subcatchment indices
        (``sc_ID``).
    debris_subcatchment_ts : dict
        ``{replicate: DataFrame}`` — per-replicate subcatchment debris
        timeseries (kg), e.g. from ``postprocess_debris_flow``'s
        ``'resampled'`` entry for the chosen catchment.  Column keys
        are also numeric ``sc_ID`` values.
    project : FireImpactsProject or None
        Required when *subcatchment_label_field* is used — the project
        is queried for the subcatchment attribute table so columns can
        be relabelled from ``sc_ID`` to the chosen string attribute.
    catchment : str or None
        Which catchment to pull from *rusle_results*.
    freq : str or None
        Pandas resample rule for the combined output:

        - ``'h'`` / ``'H'`` — hourly
        - ``'D'`` — daily
        - ``'MS'`` — monthly (month-start)
        - ``'YS'`` — annual (year-start, default)
        - ``None`` or ``'total'`` — single row summing the entire
          simulation.

        Note that RUSLE timeseries are recorded at whatever timestep
        was requested via ``default_rusle_recorders`` (daily by
        default), so requesting ``'h'`` without first re-running RUSLE
        with ``timeseries_timestep='1h'`` simply up-samples the daily
        RUSLE series via forward-fill divided by 24.  In practice,
        request the native RUSLE resolution up-front when you need
        hourly output.
    subcatchment_label_field : str or None
        Column in ``project.get_subcatchments(catchment)`` to use as
        output column labels.  When omitted, the project's configured
        label field is used — set via ``add_subcatchments(...,
        label_field=...)`` or
        :meth:`FireImpactsProject.set_subcatchment_label_field` and
        persisted per catchment in ``settings.json``.  Pass *None*
        explicitly to keep the raw ``sc_ID`` integer columns.
    rusle_timeseries_key : str
    rusle_scale : float
        Multiplier applied to the RUSLE timeseries to bring it into the
        same units as the debris timeseries.  Default ``1000.0``
        (tonnes → kilograms).

    Returns
    -------
    dict
        ``{replicate: DataFrame}``.  Each DataFrame has time on the
        index (one row per period; a single row labelled ``'total'``
        when ``freq`` is ``None``/``'total'``) and subcatchment labels
        as columns.  Values are in kilograms.  Only subcatchments
        present in both inputs are included.
    """
    if not rusle_results:
        raise ValueError('rusle_results is empty.')
    sample_rep = next(iter(rusle_results.values()))
    catchment = _resolve_catchment(sample_rep, catchment)

    rusle_ens = rusle_subcatchment_ensemble(
        rusle_results,
        project=project,
        catchment=catchment,
        freq=freq,
        subcatchment_label_field=subcatchment_label_field,
        rusle_timeseries_key=rusle_timeseries_key,
        scale=rusle_scale,
    )
    debris_ens = debris_subcatchment_ensemble(
        debris_subcatchment_ts,
        project=project,
        catchment=catchment,
        freq=freq,
        subcatchment_label_field=subcatchment_label_field,
    )

    combined = {}
    for key, rusle_at_freq in rusle_ens.items():
        if key not in debris_ens:
            continue
        debris_at_freq = debris_ens[key]
        summed = rusle_at_freq.add(debris_at_freq, fill_value=0)
        shared = [c for c in summed.columns
                  if c in rusle_at_freq.columns
                  and c in debris_at_freq.columns]
        combined[key] = summed[shared].dropna(how='all')
    return combined


# -------------------------------------------------------------------
# Per-modality helpers used by combine_rusle_and_debris_subcatchment
# -------------------------------------------------------------------

def _resolve_label_map(project, catchment, subcatchment_label_field):
    """Resolve the label field sentinel and build the ``sc_ID → label``
    map used to rename subcatchment columns."""
    if subcatchment_label_field is _SENTINEL:
        subcatchment_label_field = (
            project.subcatchment_label_field(catchment)
            if project is not None else None
        )
    return _subcatchment_label_map(
        project, catchment, subcatchment_label_field,
    )


def _make_resampler(freq):
    """Return a function that resamples a wide timeseries to *freq*.

    ``freq=None`` or ``'total'`` collapses the whole series to a single
    row indexed ``'total'``.
    """
    is_total = freq is None or (isinstance(freq, str) and freq.lower() == 'total')

    def _resample(df):
        if is_total:
            total = df.sum(axis=0).to_frame().T
            total.index = pd.Index(['total'], name='period')
            return total
        return df.resample(freq).sum()

    return _resample


def rusle_subcatchment_ensemble(
    rusle_results,
    *,
    project=None,
    catchment=None,
    freq='YS',
    subcatchment_label_field=_SENTINEL,
    rusle_timeseries_key='erosion_daily_time_series',
    scale=1000.0,
):
    """Extract per-replicate RUSLE subcatchment timeseries in the form
    ``{replicate: DataFrame}``, aggregated to *freq* and labelled by the
    configured subcatchment label field.

    This produces the same shape as
    :func:`combine_rusle_and_debris_subcatchment` so the same plotting
    and reduction helpers (e.g. :func:`reduce_ensemble_subcatchments`,
    :func:`plot_subcatchment_ensemble`) can map RUSLE outputs alone.

    Parameters
    ----------
    rusle_results : dict
        ``{replicate: {catchment: {<timeseries_key>: DataFrame, ...}}}``
        from :func:`run_rusle_all_replicates`.
    scale : float
        Multiplier applied to each replicate's timeseries.  Default
        ``1000.0`` (tonnes → kilograms) to match the combined output.
        Pass ``1.0`` to keep tonnes.

    Other parameters are as for
    :func:`combine_rusle_and_debris_subcatchment`.
    """
    if not rusle_results:
        raise ValueError('rusle_results is empty.')
    sample_rep = next(iter(rusle_results.values()))
    catchment = _resolve_catchment(sample_rep, catchment)
    label_map = _resolve_label_map(
        project, catchment, subcatchment_label_field,
    )
    resample = _make_resampler(freq)

    out = {}
    for key, reps in rusle_results.items():
        ts = reps[catchment].get(rusle_timeseries_key)
        if ts is None:
            raise KeyError(
                f"Replicate {key} has no '{rusle_timeseries_key}' entry."
            )
        df = resample(ts * scale)
        out[key] = _apply_label_map(df, label_map)
    return out


def debris_subcatchment_ensemble(
    debris_subcatchment_ts,
    *,
    project=None,
    catchment=None,
    freq='YS',
    subcatchment_label_field=_SENTINEL,
):
    """Resample and relabel per-replicate debris-flow subcatchment
    timeseries into the common ``{replicate: DataFrame}`` form.

    Parameters
    ----------
    debris_subcatchment_ts : dict
        ``{replicate: DataFrame}`` — native-resolution per-subcatchment
        debris-flow timeseries (kg), e.g. the ``'aggregated'`` entry
        from :func:`postprocess_debris_flow` for the chosen catchment.

    Other parameters are as for
    :func:`combine_rusle_and_debris_subcatchment`.
    """
    if not debris_subcatchment_ts:
        raise ValueError('debris_subcatchment_ts is empty.')
    label_map = _resolve_label_map(
        project, catchment, subcatchment_label_field,
    )
    resample = _make_resampler(freq)

    out = {}
    for key, df in debris_subcatchment_ts.items():
        out[key] = _apply_label_map(resample(df), label_map)
    return out


# ===================================================================
# Subcatchment choropleth helpers
# ===================================================================

_AREA_UNIT_FACTORS = {
    # Multiplier converting an area in m² to the named unit.
    'area_m2': 1.0,
    'area_ha': 1.0 / 10_000.0,
    'area_km2': 1.0 / 1_000_000.0,
}

_AREA_UNIT_SUFFIX = {
    'area_m2': 'per_m2',
    'area_ha': 'per_ha',
    'area_km2': 'per_km2',
}

_AREA_UNIT_LABEL = {
    'area_m2': '/m²',
    'area_ha': '/ha',
    'area_km2': '/km²',
}


def _subcatchment_areas(project, catchment, kind):
    """Series of subcatchment areas keyed by the registered label (or
    sc_ID when no label field is configured).  *kind* is one of
    ``'area_m2'``, ``'area_ha'``, ``'area_km2'``."""
    if kind not in _AREA_UNIT_FACTORS:
        raise ValueError(
            f"Unknown area unit '{kind}'. "
            f"Choose one of {sorted(_AREA_UNIT_FACTORS)} or pass a callable."
        )
    subs = project.get_subcatchments(catchment)
    id_col = getattr(project, 'subcatchment_id', 'sc_ID')
    label_field = project.subcatchment_label_field(catchment)
    # Use the catchment-CRS geometry's .area (m²) since subcatchments
    # are stored in the catchment's projected CRS.
    area_m2 = subs.geometry.area
    if label_field and label_field in subs.columns:
        index = subs[label_field].values
    else:
        index = subs[id_col].values
    return pd.Series(
        area_m2.values * _AREA_UNIT_FACTORS[kind],
        index=index,
        name=kind,
    )


def _resolve_normaliser(normalise_by, project, catchment):
    """Return ``(denominator_series, suffix, unit_label)`` for the chosen
    normalisation, or ``(None, '', '')`` when *normalise_by* is None."""
    if normalise_by is None:
        return None, '', ''
    if callable(normalise_by):
        denom = normalise_by(project.get_subcatchments(catchment))
        if not isinstance(denom, pd.Series):
            raise TypeError(
                "Callable normalise_by must return a pandas Series indexed "
                "by subcatchment label."
            )
        name = getattr(normalise_by, '__name__', 'custom')
        return denom, f'per_{name}', f'/{name}'
    if normalise_by in _AREA_UNIT_FACTORS:
        denom = _subcatchment_areas(project, catchment, normalise_by)
        return denom, _AREA_UNIT_SUFFIX[normalise_by], _AREA_UNIT_LABEL[normalise_by]
    raise ValueError(
        f"Unknown normalise_by '{normalise_by}'. "
        f"Pass one of {sorted(_AREA_UNIT_FACTORS)}, a callable, or None."
    )


def _select_time(df, time):
    """Reduce a wide per-subcatchment frame to a Series indexed by
    subcatchment label.

    - ``time=None``  — sum all rows.
    - ``time=int``   — positional (``iloc``) row.
    - otherwise      — label-based (``loc``) row.
    """
    if time is None:
        return df.sum(axis=0)
    if isinstance(time, (int, np.integer)) and not isinstance(time, bool):
        return df.iloc[int(time)]
    return df.loc[time]


def _long_frame(series, project, catchment, value_col):
    """Convert a Series indexed by subcatchment label into a long
    DataFrame with the subcatchment id column (``sc_ID``), the label
    column, and *value_col* — ready for ``plot_subcatchments``."""
    subs = project.get_subcatchments(catchment)
    id_col = getattr(project, 'subcatchment_id', 'sc_ID')
    label_field = project.subcatchment_label_field(catchment)
    if label_field and label_field in subs.columns:
        label_to_id = dict(zip(subs[label_field], subs[id_col]))
        key_name = label_field
    else:
        label_to_id = dict(zip(subs[id_col], subs[id_col]))
        key_name = id_col

    out = series.rename(value_col).to_frame()
    out.index.name = key_name
    out = out.reset_index()
    out[id_col] = out[key_name].map(label_to_id)
    # Drop rows whose label we can't match to a subcatchment (keeps the
    # output aligned with the geometry layer):
    out = out.dropna(subset=[id_col])
    return out[[id_col, key_name, value_col]]


def subcatchment_series_to_long(
    df,
    *,
    project,
    catchment,
    time=None,
    normalise_by=None,
    value_col=None,
):
    """Reshape a wide per-subcatchment timeseries into the long form
    expected by :meth:`FireImpactsProject.plot_subcatchments`.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-form frame with time on the index and one column per
        subcatchment (labelled by ``SiteID`` or similar).  Typical
        input: a single replicate's output from
        :func:`combine_rusle_and_debris_subcatchment`.
    project : FireImpactsProject
    catchment : str
    time : None, int, or label
        Row selection; see module docs.  ``None`` sums the full series.
    normalise_by : None, str, or callable
        ``None`` → native units; ``'area_ha'``/``'area_km2'``/``'area_m2'``
        → divide by subcatchment area; a callable receives the
        subcatchment ``GeoDataFrame`` and returns a Series of
        denominators indexed by subcatchment label.
    value_col : str or None
        Output column name for the value.  When *None*, an informative
        default is used (``'value'``, ``'value_per_ha'``, etc.).
    """
    series = _select_time(df, time)
    denom, suffix, _ = _resolve_normaliser(normalise_by, project, catchment)
    if denom is not None:
        series = series / denom.reindex(series.index)
    if value_col is None:
        value_col = 'value' + (f'_{suffix}' if suffix else '')
    return _long_frame(series, project, catchment, value_col)


_REDUCTION_SUFFIX = {
    'mean': 'mean',
    'median': 'median',
}


def _apply_reduction(stack, reduction):
    """Collapse a ``(replicate × subcatchment)`` frame along axis 0."""
    if callable(reduction):
        name = getattr(reduction, '__name__', 'custom')
        if name == '<lambda>':
            name = 'custom'
        return reduction(stack), name
    if isinstance(reduction, str):
        if reduction in _REDUCTION_SUFFIX:
            fn = getattr(stack, reduction)
            return fn(axis=0), _REDUCTION_SUFFIX[reduction]
        raise ValueError(
            f"Unknown reduction '{reduction}'. "
            "Use 'mean', 'median', ('quantile', q), ('exceedance', thresh), "
            "or a callable."
        )
    if isinstance(reduction, tuple) and len(reduction) == 2:
        kind, arg = reduction
        if kind == 'quantile':
            return stack.quantile(arg, axis=0), f'p{int(round(arg * 100))}'
        if kind == 'exceedance':
            return (stack > arg).mean(axis=0), f'exceed_{arg:g}'
        raise ValueError(
            f"Unknown tuple reduction '{kind}'. Use 'quantile' or 'exceedance'."
        )
    raise TypeError(f"Unsupported reduction spec: {reduction!r}")


def reduce_ensemble_subcatchments(
    ensemble,
    *,
    project,
    catchment,
    time=None,
    reduction='mean',
    normalise_by=None,
    value_col=None,
):
    """Collapse an ensemble of wide per-subcatchment timeseries into a
    single long-form frame ready for choropleth plotting.

    Normalisation is applied **per replicate before the reduction**, so
    that e.g. ``('exceedance', 0.5)`` with ``normalise_by='area_ha'``
    gives "probability that per-hectare load exceeds 0.5 (in whatever
    units the input was in)" — not "exceedance probability of the
    total load, divided by area" (which would be nonsensical).

    Parameters
    ----------
    ensemble : dict
        ``{replicate: wide DataFrame}``; e.g. output of
        :func:`combine_rusle_and_debris_subcatchment`.
    time : as for :func:`subcatchment_series_to_long`.
    reduction : str, tuple, or callable
        - ``'mean'`` / ``'median'``
        - ``('quantile', q)`` with ``0 <= q <= 1``
        - ``('exceedance', threshold)`` — probability of exceeding
          *threshold* (in post-normalisation units)
        - callable: ``f(stack_df) -> Series`` indexed by subcatchment
          label.  *stack_df* is a ``(replicate × subcatchment)`` frame
          with normalisation already applied.
    normalise_by, value_col : as for :func:`subcatchment_series_to_long`.
    """
    if not ensemble:
        raise ValueError("ensemble is empty.")
    denom, norm_suffix, _ = _resolve_normaliser(
        normalise_by, project, catchment,
    )

    rows = {}
    for rep, df in ensemble.items():
        s = _select_time(df, time)
        if denom is not None:
            s = s / denom.reindex(s.index)
        rows[rep] = s
    stack = pd.DataFrame(rows).T  # (replicate × subcatchment)
    reduced, red_suffix = _apply_reduction(stack, reduction)

    if value_col is None:
        parts = ['value', red_suffix]
        if norm_suffix:
            parts.append(norm_suffix)
        value_col = '_'.join(p for p in parts if p)

    return _long_frame(reduced, project, catchment, value_col)


# ===================================================================
# Subcatchment choropleth plot wrappers
# ===================================================================

def _choropleth_units_label(normalise_by, base_units, reduction=None):
    """Derive a human-readable units string for the colour bar."""
    if isinstance(reduction, tuple) and reduction and reduction[0] == 'exceedance':
        return 'probability'
    if normalise_by is None:
        return base_units
    if normalise_by in _AREA_UNIT_LABEL:
        return f'{base_units}{_AREA_UNIT_LABEL[normalise_by]}'
    name = getattr(normalise_by, '__name__', 'custom')
    return f'{base_units}/{name}'


def _plot_subcatchment_long(
    long_df, *, project, catchment, value_col, title, units, cmap,
    vmin, vmax, existing_figure, existing_axes,
):
    """Hand a long-form frame to ``plot_catchment_polygons`` with a
    vis_params dict built from *units*/*cmap*.  Avoids
    ``plot_subcatchments``' column-name auto-detection so that generic
    column names like ``value_mean_per_ha`` render cleanly."""
    subs = project.get_subcatchments(catchment)
    id_col = getattr(project, 'subcatchment_id', 'sc_ID')
    vis_params = {
        'cmap': cmap,
        'measure': value_col,
        'units': units or 'n/a',
        'norm': None,
        'cbar_extend': 'neither',
        'title_varname': value_col,
    }
    if vmin is not None:
        vis_params['vmin'] = vmin
    if vmax is not None:
        vis_params['vmax'] = vmax
    return project.plot_catchment_polygons(
        catchment=catchment,
        polygons=subs,
        colour_col=value_col,
        vis_params=vis_params,
        title=title or f'{catchment}: {value_col}',
        non_geo_data=long_df[[id_col, value_col]],
        id_col=id_col,
        existing_figure=existing_figure,
        existing_axes=existing_axes,
    )


def plot_subcatchment_simulation(
    df,
    *,
    project,
    catchment,
    time=None,
    normalise_by=None,
    units='kg',
    title=None,
    cmap='viridis',
    value_col=None,
    vmin=None,
    vmax=None,
    existing_figure=None,
    existing_axes=None,
):
    """Choropleth map of a single-simulation per-subcatchment timeseries.

    *df* is a wide frame (one subcatchment per column, time on the
    index), typically a single replicate from
    :func:`combine_rusle_and_debris_subcatchment`.

    *vmin* / *vmax* lock the colour-scale endpoints; when either is
    ``None`` the corresponding endpoint is taken from the data range.
    """
    long = subcatchment_series_to_long(
        df, project=project, catchment=catchment,
        time=time, normalise_by=normalise_by, value_col=value_col,
    )
    colour_col = long.columns[-1]
    unit_label = _choropleth_units_label(normalise_by, units)
    return _plot_subcatchment_long(
        long, project=project, catchment=catchment,
        value_col=colour_col, title=title, units=unit_label, cmap=cmap,
        vmin=vmin, vmax=vmax,
        existing_figure=existing_figure, existing_axes=existing_axes,
    )


def plot_subcatchment_ensemble(
    ensemble,
    *,
    project,
    catchment,
    time=None,
    reduction='mean',
    normalise_by=None,
    units='kg',
    title=None,
    cmap='viridis',
    value_col=None,
    vmin=None,
    vmax=None,
    existing_figure=None,
    existing_axes=None,
):
    """Choropleth map of an ensemble-reduced per-subcatchment timeseries.

    *ensemble* is a ``{replicate: wide DataFrame}`` dict; see
    :func:`reduce_ensemble_subcatchments` for the reduction options.

    *vmin* / *vmax* lock the colour-scale endpoints.  When either is
    ``None`` the corresponding endpoint is taken from the data.  For
    ``reduction=('exceedance', ...)`` the output is a probability in
    ``[0, 1]`` so a missing *vmin*/*vmax* defaults to ``0``/``1`` —
    pass explicit values to override.
    """
    long = reduce_ensemble_subcatchments(
        ensemble, project=project, catchment=catchment,
        time=time, reduction=reduction, normalise_by=normalise_by,
        value_col=value_col,
    )
    colour_col = long.columns[-1]
    unit_label = _choropleth_units_label(normalise_by, units, reduction)

    # Probabilities benefit from a fixed 0-1 colour scale so maps from
    # different thresholds / time slices are directly comparable.
    if (isinstance(reduction, tuple) and reduction
            and reduction[0] == 'exceedance'):
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0

    return _plot_subcatchment_long(
        long, project=project, catchment=catchment,
        value_col=colour_col, title=title, units=unit_label, cmap=cmap,
        vmin=vmin, vmax=vmax,
        existing_figure=existing_figure, existing_axes=existing_axes,
    )


def combine_rusle_and_debris_annual(
    rusle_results,
    debris_subcatchment_ts,
    catchment=None,
    rusle_timeseries_key='erosion_daily_time_series',
    rusle_scale=1000.0,
):
    """Annual-resolution wrapper around
    :func:`combine_rusle_and_debris_subcatchment`.

    Retained for backwards compatibility.  Keeps the original
    behaviour: numeric ``sc_ID`` column labels (no project lookup),
    annual totals in kilograms.  New code should call
    :func:`combine_rusle_and_debris_subcatchment` directly.
    """
    return combine_rusle_and_debris_subcatchment(
        rusle_results,
        debris_subcatchment_ts,
        project=None,
        catchment=catchment,
        freq='YS',
        subcatchment_label_field=None,
        rusle_timeseries_key=rusle_timeseries_key,
        rusle_scale=rusle_scale,
    )
