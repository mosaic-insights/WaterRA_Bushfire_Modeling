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
from fire_impacts import const as c
from fire_impacts.const import M2_TO_HA, MILLIGRAMS_TO_KILOGRAMS
from fire_impacts.pre.util import read_aligned, read_raster
from fire_impacts.util import load_package_data, get_zonal_stats
logger = logging.getLogger(__name__)

from fire_impacts.pre import FireImpactsProject
from fire_impacts.pre.project import save_catchment_raster

DNBR_SEVERITY_THRESHOLD = 400
EMPIRICAL_COEFFICIENT = 0.082
LOG_INTERVAL_SECONDS = 15.0


# ---------------------------------------------------------------------------
# RUSLE parameter grid helpers
# ---------------------------------------------------------------------------

def compute_klscp_layer(
    proj: FireImpactsProject,
    catchment: str,
    support_practice_factor: float = 1.0,
    use_fire_adjusted: bool = True,
):
    """
    Combine C, K, and LS factor rasters into a single KLSCP layer.

    When use_fire_adjusted is False the unadjusted C and K factors are
    used, producing a pre-fire baseline KLSCP for comparison against the
    fire-impacted result.

    Parameters:
    - proj: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to process.
    - support_practice_factor: RUSLE P factor (support practice).
      Default is 1.0 (no conservation practice applied).
    - use_fire_adjusted: If True, use the fire-adjusted C and K rasters.

    Returns:
    - Tuple of (klscp_array, metadata_dict) where klscp_array is a
      float32 numpy array and metadata_dict is a rasterio metadata dict
      matching the LS-factor raster resolution and extent.
    """
    c_name = (
        'C_factor_adjusted.tif' if use_fire_adjusted else 'C_factor.tif'
    )
    k_name = (
        'K_factor_adjusted.tif' if use_fire_adjusted else 'K_factor.tif'
    )
    c_factor_path = proj.catchment_path(catchment, 'Erodibility', c_name)
    k_factor_path = proj.catchment_path(catchment, 'Erodibility', k_name)
    ls_factor_path = proj.catchment_path(
        catchment, 'Erodibility', 'LS_factor.tif'
    )

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
        c_array * k_array * ls_array * support_practice_factor
    ).astype(np.float32)

    meta.update(dtype=rasterio.float32, count=1, compress='lzw')

    return base, meta


def _rusle_parameter_grids(
    project: FireImpactsProject,
    catchment: str,
    use_fire_adjusted: bool = True,
):
    """
    Load KLSCP, SDR, and dNBR rasters plus spatial metadata for a
    catchment, ready for use in RUSLE calculations.

    Parameters:
    - project: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to get the grids for.
    - use_fire_adjusted: If True, use fire-adjusted C and K factors.

    Returns:
    - Tuple of (klscp, sdr, dnbr, cell_area_ha, transform) where all
      three rasters are float32 numpy arrays, cell_area_ha is the cell
      area in hectares, and transform is the rasterio Affine transform
      shared by all three arrays.
    """
    cell_area_m2 = project.cell_area(catchment)
    cell_area_ha = cell_area_m2 * M2_TO_HA  # Convert to hectares

    # Compute KLSCP layer in memory
    klscp, klscp_meta = compute_klscp_layer(
        project, catchment, use_fire_adjusted=use_fire_adjusted
    )

    transform = klscp_meta['transform']
    crs = klscp_meta['crs']

    # Get the Sediment Delivery Ratio (SDR) raster, which is generated
    # ultimately by the calculate_lumped_rusle() function in this module
    sdr, _ = read_raster(
        project.catchment_path(catchment, 'Delivery', 'SDR.tif')
    )

    # Get the delta Normalised Burn Ratio (dNBR) raster generated by
    # severity.calculate_fire_severity(). Use read_aligned to ensure it
    # is in the same CRS and at the same resolution as the RUSLE layers.
    # Do we need to do this? Why don't we do it for SDR too?
    dnbr = read_aligned(
        project.catchment_path(
            catchment, 'FireSeverity', 'masked_dNBR.tif'
        ),
        transform,
        crs,
        klscp.shape,
    )
    return klscp, sdr, dnbr, cell_area_ha, transform


# ---------------------------------------------------------------------------
# Lumped daily RUSLE (high-level entry point)
# ---------------------------------------------------------------------------

def lumped_daily_rusle(
    project: FireImpactsProject,
    rainfall,
    catchment=None,
):
    """
    Run RUSLE and SDR calculations for sub-catchments using 30-min
    rainfall and return a per-sub-catchment daily summary DataFrame.

    Parameters:
    - project: FireImpactsProject instance for the current project.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.
    - catchment: Name of the catchment to process. If None, all
      catchments in the project are processed.

    Returns:
    - DataFrame summarising RUSLE and sediment delivery results, one
      row per sub-catchment per day.
    """
    if catchment is None:
        return project.for_each_catchment(
            lambda c: lumped_daily_rusle(project, rainfall, c)
        )

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
        project.get_subcatchments(catchment),
        rainfall,
        *_rusle_parameter_grids(project, catchment),
    )

    logger.info('Done')
    return RUSLE_df


# ---------------------------------------------------------------------------
# Recorder closures
# ---------------------------------------------------------------------------

def record_summary_grid(
    variable,
    fn='sum',
    start_time=None,
    end_time=None,
):
    """
    Build a RUSLE recorder that summarises a grid variable over time.

    Parameters:
    - variable: Name of the variable to summarise (key in the timestep
      data dict).
    - fn: Summary function to apply: 'sum', 'mean', or 'max'.
      Default is 'sum'.
    - start_time: Optional start of the summary window. Timesteps
      before this are ignored.
    - end_time: Optional end of the summary window. Timesteps after
      this are ignored.

    Returns:
    - A recorder closure compatible with run_usle_simulation, with
      .reset() and .finalize() methods attached.
    """
    result = None
    count = 0

    def grid_recorder(timestep, **kwargs):
        nonlocal result, count
        count += 1
        if start_time is not None and timestep < start_time:
            return result
        if end_time is not None and timestep > end_time:
            return result

        data = kwargs[variable]
        if result is None:
            result = data
        elif fn == 'max':
            result = np.maximum(result, data)
        else:  # sum or mean
            result += data
        if fn == 'mean':
            return result / count

        return result

    def reset():
        nonlocal result, count
        result = None
        count = 0

    def finalize():
        nonlocal result, count
        if fn == 'mean':
            return result / count
        return result

    grid_recorder.reset = reset
    grid_recorder.finalize = finalize

    return grid_recorder


def record_subcatchment_timeseries(
    proj: FireImpactsProject,
    variable_name: str,
    fn="sum",
    label_field=None,
    agg_count=1,
):
    """
    Build a RUSLE recorder that summarises a raster variable over
    subcatchments and accumulates a spatial time series.

    Parameters:
    - proj: FireImpactsProject instance for the current project.
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
                boundaries_v = proj.get_subcatchments(catchment)
            except FileNotFoundError:
                boundaries_v = proj.catchment_boundary(catchment)
            resolved_label = label_field
            if resolved_label is None:
                resolved_label = proj.subcatchment_label_field(catchment)
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
                    resolved_label, catchment,
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
    project: FireImpactsProject,
    catchment: str,
    results_section: str = c.RESULTS_FOLDER_NAME,
) -> 'pd.DataFrame | None':
    """
    Compute zonal statistics for each saved RUSLE output raster and
    write a per-subcatchment summary CSV to the Results folder.

    Parameters:
    - project: FireImpactsProject managing the directory structure.
    - catchment: Name of the catchment to process.
    - results_section: Project sub-folder where output rasters are
      stored. Defaults to the standard results folder name.

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
        subcatch_gdf = project.get_subcatchments(catchment)
    except FileNotFoundError:
        logger.info(
            'No subcatchments defined for %s — skipping RUSLE '
            'subcatchment aggregation.',
            catchment,
        )
        return None

    sc_id_col = project.subcatchment_id
    # Start the summary table with just the subcatchment ID
    summary = subcatch_gdf[[sc_id_col]].copy().reset_index(drop=True)

    for raster_name in c.RUSLE_OUTPUT_RASTER_NAMES:
        raster_path = project.catchment_path(
            catchment,
            results_section,
            f'{raster_name}.tif',
        )
        if not os.path.exists(raster_path):
            continue

        # Choose aggregation stat based on raster type (see Notes)
        stat = 'mean' if 'peak' in raster_name else 'sum'

        zstats = get_zonal_stats(
            subcatch_gdf, raster_path, raster_name, stats=[stat]
        )
        # Column name makes the aggregation method explicit
        col_name = f'{raster_name}_{stat}'
        summary[col_name] = [s[stat] for s in zstats]

    out_path = project.catchment_path(
        catchment,
        results_section,
        c.RUSLE_SC_SUMMARY_NAME + '.csv',
    )
    summary.to_csv(out_path, index=False)
    logger.info('Saved RUSLE subcatchment summary to %s', out_path)

    return summary


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def run_usle_simulation(
    project: FireImpactsProject,
    rainfall,
    catchment=None,
    recorders=None,
    save_rasters: bool = True,
    save_timeseries: bool = True,
    use_fire_adjusted: bool = True,
    results_section: str = None,
):
    """
    Run the USLE simulation for a catchment and record outputs.

    Parameters:
    - project: FireImpactsProject instance for the current project.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.
    - catchment: Name of the catchment to process. If None, all
      catchments in the project are processed.
    - recorders: Dict of recorder closures to call at each timestep.
      If None, an empty dict is used and no outputs are recorded.
    - save_rasters: Whether to save output rasters as GeoTIFF files
      to the catchment results folder.
    - save_timeseries: Whether to save the daily time series as a CSV
      file to the catchment results folder.
    - use_fire_adjusted: If True, use fire-adjusted C and K rasters.
    - results_section: Sub-folder name for outputs. If None, defaults
      to the standard results folder (or the baseline folder when
      use_fire_adjusted is False).

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
    if catchment is None:
        return project.for_each_catchment(
            lambda ctmt: run_usle_simulation(
                project, rainfall, ctmt, recorders,
                save_rasters, save_timeseries,
                use_fire_adjusted=use_fire_adjusted,
                results_section=results_section,
            )
        )

    # Resolve the output folder. Baseline runs use a sibling folder so
    # they don't overwrite fire-impacted results.
    if results_section is None:
        results_section = (
            c.RESULTS_FOLDER_NAME if use_fire_adjusted
            else c.RESULTS_BASELINE_FOLDER_NAME
        )

    # If no recorders were passed, use an empty dict so the rest of
    # the code works consistently
    if recorders is None:
        recorders = dict()

    # Reset each recorder so we're building fresh aggregations
    for recorder in recorders.values():
        recorder.reset()

    # Load the relevant RUSLE parameter grids for this catchment
    params = _rusle_parameter_grids(
        project, catchment, use_fire_adjusted=use_fire_adjusted
    )
    klscp, sdr, dnbr, cell_area_ha, transform = params

    # Get the catchment boundary geometry in a form rasterio can use
    catchment_boundary = project.catchment_boundary(catchment)
    geometry = catchment_boundary.geometry.values

    # Rasterise the catchment boundary: 1 inside, NaN outside
    mask = rasterio.features.rasterize(
        geometry,
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
        out_shape=klscp.shape,
    )

    # Apply the mask so all cells outside the catchment become NaN
    klscp_masked = klscp * mask
    sdr_masked = sdr * mask
    dnbr_masked = dnbr * mask

    # Iterate over per-timestep (timestep, data_dict) tuples from the
    # generate_rusle generator and pass each to all recorder closures
    results = dict()
    for ts_data in generate_rusle(
        rainfall,
        klscp_masked,
        sdr_masked,
        dnbr_masked,
        cell_area_ha,
    ):
        timestep, data = ts_data
        for key, recorder in recorders.items():
            results[key] = recorder(
                timestep,
                **data,
                catchment=catchment,
                transform=transform,
            )

    # Finalise each recorder after all timesteps are processed
    for key, recorder in recorders.items():
        results[key] = recorder.finalize()

    if save_rasters or save_timeseries:
        # Baseline runs land in a non-standard folder that the project
        # template doesn't pre-create. Make sure it exists.
        os.makedirs(
            project.catchment_path(catchment, results_section),
            exist_ok=True,
        )

    if save_rasters:
        template_raster = project.catchment_path(
            catchment, 'Erodibility', 'LS_factor.tif'
        )
        _, template_meta = read_raster(template_raster)
        for output_raster in c.RUSLE_OUTPUT_RASTER_NAMES:
            save_data = results.get(output_raster)
            if save_data is None:
                continue
            save_catchment_raster(
                project=project,
                catchment_name=catchment,
                file_name=output_raster,
                section=results_section,
                data=save_data,
                meta=template_meta,
            )

        # Aggregate the saved rasters to subcatchments and write a
        # summary CSV alongside them (skipped if no subcatchments exist)
        aggregate_rusle_to_subcatchments(
            project, catchment, results_section=results_section
        )

    if save_timeseries:
        out_name = project.catchment_path(
            catchment,
            results_section,
            c.RUSLE_OP_TIMESERIES_NAME + '.csv',
        )
        output = pd.DataFrame(data=results[c.RUSLE_OP_TIMESERIES_NAME])
        output.index.name = 'Datetime'
        output.to_csv(out_name)

    # Attach a pointer to all the RUSLE parameters used for these calcs
    results['params'] = params

    return results


# ---------------------------------------------------------------------------
# Deprecated simulation wrappers
# ---------------------------------------------------------------------------

@deprecated(
    "This function is deprecated and will be removed in a future version."
    " Please use run_usle_simulation() with appropriate recorders instead."
)
def gridded_total_rusle(
    project: FireImpactsProject,
    rainfall,
    catchment=None,
):
    """
    Compute total RUSLE erosion and delivery grids over a simulation.

    Deprecated. Use run_usle_simulation() with appropriate recorders.

    Parameters:
    - project: FireImpactsProject instance for the current project.
    - rainfall: Series-like with 30-minute rainfall depth values in mm.
    - catchment: Name of the catchment to process. If None, all
      catchments are processed.

    Returns:
    - Tuple of (total_eroded, total_delivered, transform) where the
      first two are float32 numpy arrays and the last is the Affine
      transform object from the RUSLE parameter grids.
    """
    if catchment is None:
        return project.for_each_catchment(
            lambda c: lumped_daily_rusle(project, rainfall, c)
        )
    result = None
    # Get the boundary geometry for the first subcatchment only
    subcatch_boundaries = (
        project.get_subcatchments(catchment).iloc[0].geometry
    )

    total_eroded = None
    total_delivered = None
    params = _rusle_parameter_grids(project, catchment)
    for day_data in generate_rusle_for_feature(
        [subcatch_boundaries], rainfall, *params
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
    return total_eroded, total_delivered, params[-1]


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
            cell_area_ha, transform,
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
):
    """
    Yield per-timestep RUSLE erosion and delivery results as a generator.

    Parameters:
    - rainfall: Series with 30-minute rainfall depth values in mm.
    - klscp: KLSCP raster array.
    - sdr: Sediment Delivery Ratio raster array.
    - dnbr: dNBR raster array.
    - cell_area_ha: Area of each raster cell in hectares.

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

    # Pre-compute severity masks based on dNBR thresholds
    dnbr_below_threshold = dnbr < DNBR_SEVERITY_THRESHOLD
    dnbr_above_threshold = dnbr >= DNBR_SEVERITY_THRESHOLD

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

        # Calculate rainfall intensity (∆V_r / ∆t_r) in mm/hr
        intensity = delta_v_r / 0.5
        result['intensity'] = intensity

        # Calculate unit kinetic energy (e_r)
        e_r = 0.29 * (
            1 - 0.72 * np.exp(-EMPIRICAL_COEFFICIENT * intensity)
        )

        # Calculate kinetic energy (E) and erosivity factor (R)
        E = e_r * delta_v_r
        R = E * intensity
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
):
    """
    Yield daily RUSLE results clipped to a single feature geometry.

    Deprecated. Use run_usle_simulation() with appropriate recorders.

    Parameters:
    - geometry: List of shapely geometries defining the sub-catchment.
    - rainfall: DataFrame with 30-minute rainfall depth values in mm.
    - klscp: KLSCP raster array.
    - sdr: Sediment Delivery Ratio raster array.
    - dnbr: dNBR raster array.
    - cell_area_ha: Area of each raster cell in hectares.
    - transform: Affine transform shared by all three raster arrays.

    Returns:
    - Generator yielding one tuple per day:
      (day, daily_total_rain, max_intensity, max_erosivity,
       daily_RUSLE, daily_SDR,
       daily_RUSLE_below_threshold, daily_RUSLE_above_threshold,
       daily_SDR_below_threshold, daily_SDR_above_threshold)
    """
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

            intensity = delta_v_r / 0.5  # mm/hr
            max_intensity = max(max_intensity, intensity)

            e_r = 0.29 * (
                1 - 0.72 * np.exp(-EMPIRICAL_COEFFICIENT * intensity)
            )
            E = e_r * delta_v_r
            R = E * intensity
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
        dnbr_below_threshold = dnbr_masked < DNBR_SEVERITY_THRESHOLD
        dnbr_above_threshold = dnbr_masked >= DNBR_SEVERITY_THRESHOLD

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
    proj,
    rainfall_30min,
    replicate_idx,
    recorder_factory=None,
    use_fire_adjusted=True,
):
    """
    Run the RUSLE simulation for a single rainfall replicate.

    Parameters:
    - proj: FireImpactsProject instance for the current project.
    - rainfall_30min: xarray.Dataset of rainfall replicates with a
      'replicate' dimension.
    - replicate_idx: Index of the replicate to run.
    - recorder_factory: Callable (project, start, end) → dict of
      recorders, as returned by default_rusle_recorders(). When None,
      a default factory is used.
    - use_fire_adjusted: If True, use fire-adjusted C and K rasters.

    Returns:
    - Dict keyed by catchment name; values are the recorder results
      dict for that catchment.
    """
    rain_seq = rainfall_30min.rainfall[:, replicate_idx].to_pandas()

    if recorder_factory is None:
        recorder_factory = default_rusle_recorders()

    start = rain_seq.index[0]
    end = rain_seq.index[-1]

    replicate_results = {}
    for c_name in proj.catchments:
        recorders = recorder_factory(proj, start, end)
        replicate_results[c_name] = run_usle_simulation(
            proj,
            rain_seq,
            catchment=c_name,
            recorders=recorders,
            save_rasters=False,
            save_timeseries=False,
            use_fire_adjusted=use_fire_adjusted,
        )

    return replicate_results


def run_rusle_all_replicates(
    proj,
    rainfall_30min,
    n_workers=None,
    scheduler='threads',
    replicate_indices=None,
    recorder_factory=None,
    use_fire_adjusted=True,
):
    """
    Run RUSLE simulations for selected rainfall replicates in parallel.

    Parameters:
    - proj: FireImpactsProject instance for the current project.
    - rainfall_30min: xarray.Dataset of rainfall replicates with a
      'replicate' dimension.
    - n_workers: Number of Dask workers. If None, Dask chooses.
    - scheduler: Dask scheduler to use, e.g. 'threads' or 'processes'.
    - replicate_indices: Iterable of replicate indices to run. If None,
      all replicates are run.
    - recorder_factory: Callable (project, start, end) → dict of
      recorders, as returned by default_rusle_recorders(). When None,
      a default factory is used.
    - use_fire_adjusted: If True, use fire-adjusted C and K rasters.

    Returns:
    - Dict mapping replicate index (int) to simulation output dicts.
    """
    import dask

    if recorder_factory is None:
        recorder_factory = default_rusle_recorders()

    if replicate_indices is None:
        replicate_indices = list(range(rainfall_30min.dims['replicate']))
    else:
        replicate_indices = list(replicate_indices)

    tasks = [
        dask.delayed(run_rusle_replicate)(
            proj,
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
    'monthly': pd.DateOffset(months=1),
}


def _compute_periods(start, end, timestep_type):
    """
    Compute non-overlapping time-period boundaries for a simulation span.

    Parameters:
    - start: Start of the simulation period (pd.Timestamp).
    - end: End of the simulation period (pd.Timestamp).
    - timestep_type: Granularity of the periods: 'total', 'yearly', or
      'monthly'.

    Returns:
    - List of (period_start, period_end) tuples. For non-final periods
      period_end is offset by -1 second so boundary timesteps are not
      double-counted across adjacent periods.
    """
    if timestep_type == 'total':
        return [(start, end)]

    offset = _PERIOD_OFFSETS.get(timestep_type)
    if offset is None:
        raise ValueError(
            f"Unsupported grid_timestep '{timestep_type}'. "
            "Use 'total', 'yearly', or 'monthly'."
        )

    periods = []
    period_start = start
    while period_start < end:
        period_end = period_start + offset
        if period_end > end:
            period_end = end
        periods.append((period_start, period_end))
        period_start = period_end

    # Offset non-final period ends by 1 s to avoid double-counting
    return [
        (ps, pe - pd.Timedelta(seconds=1))
        if i < len(periods) - 1 else (ps, pe)
        for i, (ps, pe) in enumerate(periods)
    ]


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

    def _build_spatial_coords(shape):
        """Build easting/northing coordinate arrays from the transform."""
        t = captured_transform[0]
        if t is None:
            return {}
        rows, cols = shape
        easting = np.array(
            [t.c + (col + 0.5) * t.a for col in range(cols)]
        )
        northing = np.array(
            [t.f + (row + 0.5) * t.e for row in range(rows)]
        )
        return {'easting': easting, 'northing': northing}

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

        spatial = _build_spatial_coords(shape)

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
_MODEL_TIMESTEP = pd.Timedelta(minutes=30)


def default_rusle_recorders(
    include_grids=True,
    grid_variables=('RUSLE',),
    grid_fns=('sum', 'max'),
    grid_timesteps=('yearly',),
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
    - factory: Callable (project, start, end) → dict of recorder
      closures ready for use with run_usle_simulation().
    ------------------------------------------------------------------------
    Notes:
    - Typical usage: make_recorders = default_rusle_recorders(), then
      recorders = make_recorders(proj, '2020-01-01', '2021-12-31').
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

    def factory(project, start, end):
        """Build and return a fresh dict of recorders for one simulation."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        recorders = {}

        # Grid recorders: Cartesian product of variables × fns × timesteps
        if include_grids:
            for ts_type in grid_timesteps:
                periods = _compute_periods(start, end, ts_type)
                for variable in grid_variables:
                    for fn in grid_fns:
                        key = f'{variable}_{fn}_{ts_type}'
                        recorders[key] = record_multi_period_grid(
                            variable, fn, periods,
                        )

        # Transform recorder
        if include_transform:
            recorders['the_transform'] = record_grid_transform()

        # Subcatchment timeseries recorders
        if include_timeseries:
            _add_timeseries_recorders(project, recorders)

        return recorders

    def _add_timeseries_recorders(project, recorders):
        """Add subcatchment timeseries recorders to the recorders dict."""
        for ts_var in timeseries_variables:
            base_ts = record_subcatchment_timeseries(
                project,
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
