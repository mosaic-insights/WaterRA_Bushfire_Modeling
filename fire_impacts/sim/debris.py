"""
Debris flow simulation for post-fire catchments.

Computes pixel-level net erosion from slope and clay fraction inputs,
accumulates erosion along the flow network, and applies rainfall-
intensity thresholds to determine debris flow events at each headwater.
Constants (HILLSLOPE_PARAMETERS, CHANNEL_PARAMETERS, etc.) are imported
from fire_impacts.const via wildcard import.
"""

import xarray as xr
import pandas as pd
from shapely.geometry import mapping
import numpy as np
from numpy.typing import ArrayLike
import rasterio
from rasterio.warp import Resampling
from rasterio.transform import from_origin, Affine, rowcol
from fire_impacts.const import *
from fire_impacts.pre import topography
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.context import RunContext
from fire_impacts.pre.util import read_aligned, read_raster, write_raster
from fire_impacts.util import load_package_data, unique_file_matching
from fire_impacts.params import deprecated_overrides
from pysheds.grid import Grid
import os
import tempfile
import logging
import geopandas as gpd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow layer helpers
# ---------------------------------------------------------------------------

def get_flow_layers(
    hydro_dem,
    dem_meta: dict,
    grid: Grid,
    dirmap: tuple,
    ctx: RunContext,
):
    """
    Return flow-direction and flow-accumulation rasters, loading from
    disk if already saved or computing them fresh if not.

    Parameters:
    - hydro_dem: Hydrologically enforced DEM array.
    - dem_meta: Rasterio metadata dictionary for hydro_dem.
    - grid: pysheds Grid object for hydrological operations.
    - dirmap: Tuple of D8 flow-direction cell values.
    - project: FireImpactsProject for directory management.
    - catchment: Name of the catchment to load layers for.

    Returns:
    - flow_dir_data: pysheds Raster of flow directions.
    - flow_dir_meta: Metadata dict for the flow-direction raster.
    - flow_acc_data: pysheds Raster of flow accumulation counts.
    - flow_acc_meta: Metadata dict for the flow-accumulation raster.
    """
    # Check whether a pre-computed flow direction raster is already saved
    try_flowdir_path = ctx.catchment_path(
        'Topography', f'{FLOW_DIRECTION_FN}.tif',
    )
    try:
        flow_dir_array, flow_dir_meta = read_raster(try_flowdir_path)
        logger.info(
            'Existing flow direction raster found at '
            f'{try_flowdir_path}. Reading this in instead of '
            'computing new raster.'
        )
        PYSHEDS_D8_NODATA_VAL = np.int32(0)
        flow_dir_meta['nodata'] = PYSHEDS_D8_NODATA_VAL
        flow_dir_array = np.where(
            flow_dir_array < NODATA_VAL_INT,
            PYSHEDS_D8_NODATA_VAL,
            flow_dir_array,
        )
        flow_dir_data = topography.rio_to_pysheds(
            flow_dir_array, flow_dir_meta, try_flowdir_path
        )

    # No saved raster — compute from the hydrologically enforced DEM.
    # (rasterio raises its own RasterioIOError, not FileNotFoundError,
    # for a missing path.)
    except (FileNotFoundError, rasterio.errors.RasterioIOError):
        flow_dir_data, flow_dir_meta, grid = topography.compute_flow_dir(
            hydro_dem, dem_meta, grid, dirmap, ctx,
        )

    # Check whether a pre-computed flow accumulation raster is saved
    try_flowacc_path = ctx.catchment_path(
        'Topography', f'{FLOW_ACCUMULATION_FN}.tif',
    )
    try:
        flow_acc_array, flow_acc_meta = read_raster(try_flowacc_path)
        logger.info(
            'Existing flow accumulation raster found at '
            f'{try_flowacc_path}. Reading this in instead of '
            'computing new raster.'
        )
        flow_acc_meta['nodata'] = np.int32(flow_acc_meta['nodata'])
        flow_acc_data = topography.rio_to_pysheds(
            flow_acc_array, flow_acc_meta, try_flowacc_path
        )

    # No saved raster — compute from the flow direction raster
    except (FileNotFoundError, rasterio.errors.RasterioIOError):
        flow_acc_data, flow_acc_meta, _ = topography.compute_flow_accum(
            flow_dir_data, flow_dir_meta, grid, dirmap, ctx,
        )

    return flow_dir_data, flow_dir_meta, flow_acc_data, flow_acc_meta


def get_clay_fraction(
    ctx: RunContext,
    depth: str,
    transform,
    crs,
    shape,
):
    """
    Read a clay-percentage raster for a given depth range and return it
    as a dimensionless fraction aligned to a target raster grid.

    Parameters:
    - project: FireImpactsProject for directory management.
    - catchment: Name of the catchment being processed.
    - depth: Depth range string in the form 'xxx_yyy' (cm), e.g.
      '000_005' for 0–5 cm.
    - transform: Affine transform of the target raster grid.
    - crs: CRS object of the target raster grid.
    - shape: Shape (rows, cols) of the target raster grid.

    Returns:
    - 2-D numpy array of clay fraction values (0–1) aligned to the
      target grid.
    ------------------------------------------------------------------------
    Notes:
    - unique_file_matching assumes specific naming conventions for clay
      files and raises an error if multiple matches are found.
    ------------------------------------------------------------------------
    """
    # Locate the clay raster directory and find the matching file
    clay_directory = ctx.catchment_path('Soils', 'CLY')
    file_name = unique_file_matching(
        clay_directory, 'CLY', depth, 'EV', extension='.tif'
    )
    file_path = os.path.join(clay_directory, file_name)
    # Read, reproject to the target grid, and convert % → fraction
    return (
        read_aligned(file_path, transform, crs, shape)
        * PERCENT_TO_FRACTION
    )


# ---------------------------------------------------------------------------
# Debris flow preparation
# ---------------------------------------------------------------------------

def prep_debris_flow_simulation(
    ctx: RunContext,
    dnbr_threshold=UNSET,
    params=None,
):
    """
    Assemble all spatial inputs required to run the debris flow simulation.

    Reads or computes the DEM, slope, flow layers, and clay fractions,
    loads lookup tables and condition data, then calls debris_flow_load
    to produce the per-headwater result table.

    Parameters:
    - ctx: event-level RunContext.
    - dnbr_threshold: Deprecated. Use the debris parameter group
      (debris.dnbr_threshold). Supplying it here is honoured as a
      call-layer override.
    - params: Calibration parameters — a ParameterRecord (from
      ctx.parameters()) or a ModelParameters. Supplies the dNBR cutoff
      and the I12 lookup table to use.

    Returns:
    - DataFrame of per-headwater debris flow inputs and results, as
      produced by debris_flow_load().
    ------------------------------------------------------------------------
    Notes:
    - Headwaters with excessive NaN coverage in masked_dNBR should be
      filtered upstream via summary_stats(masked_nan_threshold=...).
      The inner join against the condition CSV enforces that filtering.
    ------------------------------------------------------------------------
    """
    ctx.validate()
    p = ctx._resolved_params(
        params,
        **deprecated_overrides({'debris.dnbr_threshold': dnbr_threshold}),
    ).parameters.debris
    dnbr_threshold = p.dnbr_threshold
    project = ctx.project
    catchment = ctx.catchment
    id_field = project.headwater_id
    # Load the DEM and its metadata
    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    dem_data, dem_meta = read_raster(dem_path)

    # Build the hydrologically enforced DEM and the pysheds grid object
    hydro_dem, grid_obj = topography.hydro_force_dem(dem_path)

    # Compute slope from the hydro DEM as a rise/run ratio
    slope_h_ratio, slope_h_meta = topography.dem_to_slope(
        ctx,
        (dem_data, dem_meta),
        gradient=True,
        hydro=True,
        save=False,
    )

    # Load or compute flow direction and flow accumulation rasters
    flw_lyr_tuple = get_flow_layers(
        hydro_dem, dem_meta, grid_obj, D8_FLOW_DIRECTIONS, ctx,
    )
    flow_dir_data, flow_dir_meta, flow_acc_data, flow_acc_meta = (
        flw_lyr_tuple
    )
    transform = flow_acc_meta['transform']
    crs = flow_acc_meta['crs']
    shape = slope_h_ratio.shape

    # Read clay fractions at 0–5 cm and 5–15 cm depth, masking pixels
    # where the slope raster has no data
    clay0_5, clay5_15 = [
        np.where(
            np.isnan(slope_h_ratio),
            np.nan,
            get_clay_fraction(ctx, depth, transform, crs, shape),
        )
        for depth in ['000_005', '005_015']
    ]

    # Check for NaN in clay fractions beyond what the slope mask introduces
    slope_nan_count = int(np.isnan(slope_h_ratio).sum())
    for label, clay in [('clay_0_5', clay0_5), ('clay_5_15', clay5_15)]:
        clay_nan_count = int(np.isnan(clay).sum())
        extra_nans = clay_nan_count - slope_nan_count
        if extra_nans > 0:
            logger.warning(
                '%s has %d more NaN pixels than the topography mask '
                '(%d total NaN vs %d from slope). '
                'These likely come from gaps in soil data and where '
                'they overlap with headwaters, they will '
                'propagate through erosion calculations.',
                label, extra_nans, clay_nan_count, slope_nan_count,
            )

    # Load the HF lookup table (I12 critical rainfall thresholds) and
    # the debris constituent proportions table from package data
    hf_lookup = load_package_data(p.i12_lookup)
    hf_lookup['I12_crit_mean'] = hf_lookup['I12_crit_mean'].round(1)
    debris_lookup = load_package_data('debris-constituents.csv')

    # Create the per-event DebrisFlow prep directory if needed
    out_path = ctx.event_path('DebrisFlow')
    os.makedirs(out_path, exist_ok=True)

    # Load the pre-computed soil/slope/aridity/dNBR condition summary
    # (per-event because it depends on the fire's dNBR).
    condition_data = pd.read_csv(
        ctx.event_path('Soil_Slope_Aridity_dNBR_headwaters.csv'),
    )

    # Log any NaN columns in the condition data before filling
    nan_cols = (
        condition_data.columns[condition_data.isna().any()].tolist()
    )
    if nan_cols:
        logger.warning(
            'NaN values found in condition data before join for '
            'columns: %s', nan_cols,
        )
    condition_data = condition_data.fillna(0.0)
    # Remove headwaters where mean dNBR is below the debris-flow burn threshold.
    n_before = len(condition_data)

    condition_data = condition_data[
        condition_data[DNBR_MEAN] >= dnbr_threshold
    ].copy()

    n_removed = n_before - len(condition_data)

    if n_removed > 0:
        logger.info(
            "%d headwaters removed from debris-flow analysis because "
            "mean dNBR was below %s.",
            n_removed, dnbr_threshold,
        )

    # Load the headwaters topographic summary
    topo_data = pd.read_csv(
        ctx.catchment_path('Topography', 'Headwaters.csv'),
    )

    # Inner-join condition and topographic data.  Log the row accounting
    # so it is easy to see how many headwaters were excluded and why.
    n_condition = len(condition_data)
    n_topo = len(topo_data)
    fire_impact_data = pd.merge(
        condition_data, topo_data, on=id_field, how='inner'
    )
    n_joined = len(fire_impact_data)

    # Headwaters in topo but not condition were already excluded upstream
    pre_excluded = n_topo - n_joined
    if pre_excluded > 0:
        logger.info(
            '%d of %d headwaters already excluded upstream '
            '(not present in condition data).',
            pre_excluded, n_topo,
        )
    # Headwaters in condition but not topo is unexpected — both files
    # should derive from the same Headwaters.shp
    if n_joined < n_condition:
        condition_ids = set(condition_data[id_field])
        topo_ids = set(topo_data[id_field])
        missing_from_topo = sorted(condition_ids - topo_ids)
        raise ValueError(
            f'{len(missing_from_topo)} headwater(s) present in '
            f'condition data (Soil_Slope_Aridity_dNBR_headwaters.csv) '
            f'but missing from Headwaters.csv: {missing_from_topo}. '
            f'Both files should derive from the same Headwaters.shp. '
            f'Condition data has {id_field} range '
            f'{condition_data[id_field].min()}–'
            f'{condition_data[id_field].max()}, '
            f'Headwaters.csv has {id_field} range '
            f'{topo_data[id_field].min()}–'
            f'{topo_data[id_field].max()}.'
        )

    return debris_flow_load(
        dem_data, slope_h_ratio, transform, flow_acc_data,
        flow_dir_data, clay0_5, clay5_15, out_path,
        fire_impact_data, hf_lookup, debris_lookup, dem_meta, id_field,
    )


# ---------------------------------------------------------------------------
# Pixel-level erosion computation
# ---------------------------------------------------------------------------

def net_erosion(
    threshold_met: np.ndarray,
    ae: float,
    be: float,
    ad: float,
    bd: float,
    rock: float,
    clay_fraction: np.ndarray,
    flow_area: np.ndarray,
    gradient_arr: np.ndarray,
    pixel_area: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net erosion mass per pixel using empirical depth equations.

    Applies erosion and deposition depth formulae to produce net mass
    for total material, clay fraction, and non-clay sediment.

    Parameters:
    - threshold_met: Boolean array indicating which pixels have flow
      area exceeding the erosion threshold.
    - ae: Erosion depth coefficient.
    - be: Erosion depth exponent.
    - ad: Deposition depth coefficient.
    - bd: Deposition depth exponent.
    - rock: Rock fraction of the material (0–1).
    - clay_fraction: 2-D array of clay fractions (0–1) per pixel.
    - flow_area: 2-D array of upstream contributing area in m².
    - gradient_arr: 2-D array of slope gradient (rise/run ratio).
    - pixel_area: Area of each raster pixel in m².

    Returns:
    - e_net_mass: 2-D array of total net erosion mass (kg) per pixel.
    - e_clay_mass: 2-D array of clay fraction of erosion mass (kg).
    - e_sediment_mass: 2-D array of non-clay sediment mass (kg).
    """
    e0 = np.where(threshold_met, flow_area, 0)  # Erosion
    e = ae * (gradient_arr * e0) ** be  # erosion depth (m)
    d = ad * (gradient_arr * e0) ** bd  # deposition depth (m)
    e_net = e - d  # net erosion depth (m)
    e_net_vol = e_net * pixel_area  # erosion volume (m³)
    e_sediment_mass = (
        e_net_vol * (1 - rock) * SEDIMENT_BULK_DENSITY
    )  # sediment only (kg)
    e_rock_mass = (
        e_net_vol * rock * ROCK_BULK_DENSITY
    )  # rock only (kg)
    e_net_mass = e_sediment_mass + e_rock_mass  # net erosion mass (kg)
    e_clay_mass = e_sediment_mass * clay_fraction
    return e_net_mass, e_clay_mass, e_sediment_mass


def compute_net_erosion(
    flow_acc_area: np.ndarray,
    clay_frac_0_05: np.ndarray,
    clay_frac_05_15: np.ndarray,
    gradient_arr: np.ndarray,
    pixel_area: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net erosion layers for both hillslope and channelised flow.

    Applies net_erosion() twice — once with hillslope parameters for
    pixels below the hillslope area threshold, and once with channel
    parameters for pixels in the channelised flow zone — then sums.

    Parameters:
    - flow_acc_area: 2-D array of upstream contributing area in m².
    - clay_frac_0_05: Clay fraction array for the 0–5 cm depth layer.
    - clay_frac_05_15: Clay fraction array for the 5–15 cm depth layer.
    - gradient_arr: 2-D slope gradient array (rise/run).
    - pixel_area: Area of each raster pixel in m².

    Returns:
    - erosion_mass_all: Total net erosion mass (kg) per pixel.
    - erosion_mass_clay: Clay component of erosion mass (kg) per pixel.
    - Sediment_mass: Non-clay sediment mass (kg) per pixel.
    """
    # Hillslope erosion — pixels with small upstream area
    er_mass_hs_total, er_mass_hs_clay, er_mass_hs_sediment = net_erosion(
        threshold_met=flow_acc_area <= HILLSLOPE_AREA,
        **HILLSLOPE_PARAMETERS,
        clay_fraction=clay_frac_0_05,
        flow_area=flow_acc_area,
        gradient_arr=gradient_arr,
        pixel_area=pixel_area,
    )
    # Channelised flow erosion — pixels in the channelised zone
    er_mass_ch_total, er_mass_ch_clay, er_mass_ch_sediment = net_erosion(
        threshold_met=(
            (flow_acc_area > HILLSLOPE_AREA)
            & (flow_acc_area <= CHANNELISED_FLOW_THRESHOLD)
        ),
        **CHANNEL_PARAMETERS,
        clay_fraction=clay_frac_05_15,
        flow_area=flow_acc_area,
        gradient_arr=gradient_arr,
        pixel_area=pixel_area,
    )

    erosion_mass_all = er_mass_hs_total + er_mass_ch_total
    erosion_mass_clay = er_mass_hs_clay + er_mass_ch_clay
    Sediment_mass = er_mass_hs_sediment + er_mass_ch_sediment

    return erosion_mass_all, erosion_mass_clay, Sediment_mass


# ---------------------------------------------------------------------------
# Erosion accumulation along the flow network
# ---------------------------------------------------------------------------

def accumulate_erosion(
    erosion_values: np.ndarray,
    rio_meta: dict,
    flow_dir_raster: ArrayLike,
    catchment_mask: np.ndarray,
    save_path: str = None,
) -> ArrayLike:
    """
    Accumulate per-pixel erosion values along the flow network.

    Uses pysheds' weighted flow accumulation to route erosion mass
    downstream, in the same way a standard flow accumulation raster
    accumulates cell counts.  Sets any NaN inside the catchment to 0.

    Parameters:
    - erosion_values: 2-D array of per-pixel erosion mass (kg).
    - rio_meta: Rasterio metadata dict used to write the scratch raster.
    - flow_dir_raster: pysheds Raster of flow directions.
    - catchment_mask: Boolean array, True where pixels are inside the
      catchment.
    - save_path: If provided, save the accumulated raster to this path.
      Any pre-existing file at save_path is removed first to avoid GDAL
      errors from corrupt/truncated leftovers.

    Returns:
    - 2-D array of accumulated erosion mass (kg) at each pixel.
    """
    # Write to a unique temp file so concurrent replicate runs don't
    # overwrite each other's scratch raster.  delete=False because
    # pysheds reopens the file by name; cleaned up in the finally block.
    tmp = tempfile.NamedTemporaryFile(
        prefix='tmp_erosion_', suffix='.tif', delete=False,
    )
    fn = tmp.name
    tmp.close()
    try:
        write_raster(
            fn, erosion_values, rio_meta,
            nodata=rio_meta.get('nodata'),
        )
        grid = Grid.from_raster(fn)
        e_raster = grid.read_raster(fn)
        accum_raster = grid.accumulation(
            fdir=flow_dir_raster, weights=e_raster
        )
        # Set NaN values inside the catchment to 0
        accum_raster[np.isnan(accum_raster) & catchment_mask] = 0
        if save_path is not None:
            if os.path.exists(save_path):
                os.remove(save_path)
            write_raster(
                save_path, accum_raster, rio_meta,
                nodata=rio_meta.get('nodata'),
            )
            logger.info(
                f'Saved cumulative erosion raster to {save_path}'
            )
        return accum_raster
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def create_cum_erosion_layers(
    erosion_mass_all: np.ndarray,
    erosion_mass_clay: np.ndarray,
    erosion_mass_sediment: np.ndarray,
    flow_dir_raster: ArrayLike,
    out_path: str,
    rio_meta: dict,
    catchment_mask: np.ndarray,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Build cumulative erosion rasters for total, clay, and sediment mass.

    Calls accumulate_erosion() for each of the three input layers and
    optionally saves the results as GeoTIFFs.

    Parameters:
    - erosion_mass_all: Per-pixel total erosion mass (kg).
    - erosion_mass_clay: Per-pixel clay erosion mass (kg).
    - erosion_mass_sediment: Per-pixel non-clay sediment mass (kg).
    - flow_dir_raster: pysheds Raster of flow directions.
    - out_path: Directory for output GeoTIFFs.  Pass None to skip
      saving.
    - rio_meta: Rasterio metadata dict for output rasters.
    - catchment_mask: Boolean array of in-catchment pixels.

    Returns:
    - e_all_accum: Accumulated total erosion mass array.
    - e_clay_accum: Accumulated clay erosion mass array.
    - sediment_mass_accum: Accumulated sediment erosion mass array.
    """
    # Build output paths, or None if saving is disabled
    if out_path is None:
        E_all_cum_path = None
        E_clay_cum_path = None
        Sediment_mass_cum_path = None
    else:
        E_all_cum_path = os.path.join(
            out_path, f"{ERO_CUM_M_ALL_FN}.tif"
        )
        E_clay_cum_path = os.path.join(
            out_path, f"{ERO_CUM_M_CLY_FN}.tif"
        )
        Sediment_mass_cum_path = os.path.join(
            out_path, f"{ERO_CUM_M_SED_FN}.tif"
        )

    e_all_accum = accumulate_erosion(
        erosion_mass_all, rio_meta, flow_dir_raster,
        catchment_mask, E_all_cum_path,
    )
    e_clay_accum = accumulate_erosion(
        erosion_mass_clay, rio_meta, flow_dir_raster,
        catchment_mask, E_clay_cum_path,
    )
    sediment_mass_accum = accumulate_erosion(
        erosion_mass_sediment, rio_meta, flow_dir_raster,
        catchment_mask, Sediment_mass_cum_path,
    )

    return e_all_accum, e_clay_accum, sediment_mass_accum


def create_erosion_sense_check(
    accum_erosion: ArrayLike,
    flow_acc_area: np.ndarray,
    rio_meta: dict,
    save_loc: str,
    erosion_type: str = 'all',
) -> np.ndarray:
    """
    Save a GeoTIFF of accumulated erosion per accumulated flow area (ha).

    Divides the accumulated erosion mass by the accumulated flow area
    to produce a kg/ha layer useful for sanity-checking the magnitude
    of modelled erosion.

    Parameters:
    - accum_erosion: 2-D accumulated erosion mass array (kg).
    - flow_acc_area: 2-D accumulated flow area array (m²).
    - rio_meta: Rasterio metadata dict for the output raster.
    - save_loc: Directory in which to save the output GeoTIFF.
    - erosion_type: Label string used in the output filename.

    Returns:
    - 2-D numpy array of erosion mass per hectare (kg/ha).
    """
    flow_acc_area_ha = flow_acc_area * M2_TO_HA
    acc_ero_data = np.array(accum_erosion, dtype=np.float32)
    # Negative values are physically impossible — set to 0
    acc_ero_data = np.where(acc_ero_data < 0, 0, acc_ero_data)
    with np.errstate(divide='ignore', invalid='ignore'):
        E_all_mass_ha = np.divide(acc_ero_data, flow_acc_area_ha)

    E_all_mass_ha_path = os.path.join(
        save_loc, f"Erosion_{erosion_type}_mass_per_ha.tif"
    )
    write_raster(
        E_all_mass_ha_path, E_all_mass_ha, rio_meta,
        nodata=rio_meta.get('nodata'),
    )

    return E_all_mass_ha


# ---------------------------------------------------------------------------
# Debris load column helpers
# ---------------------------------------------------------------------------

def get_debris_volume(
    x: float,
    y: float,
    transform: Affine,
    debris_volume_array: ArrayLike,
):
    """
    Extract an array value at a given (x, y) coordinate.

    Parameters:
    - x: Easting (or longitude) coordinate.
    - y: Northing (or latitude) coordinate.
    - transform: Affine transform of the raster array.
    - debris_volume_array: 2-D raster array to sample.

    Returns:
    - The array value at the pixel containing (x, y), or numpy.nan
      if the coordinate falls outside the raster bounds.
    """
    row, col = rowcol(transform, x, y)
    rows, cols = debris_volume_array.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        logger.debug(
            'Headwater endpoint (%.1f, %.1f) maps to row=%d, col=%d '
            'which is outside raster bounds (%d, %d). Returning NaN.',
            x, y, row, col, rows, cols,
        )
        return np.nan
    return debris_volume_array[row, col]


def debris_column_values(
    fire_impact_data: pd.DataFrame,
    transform: Affine,
    e_all_accum: ArrayLike,
    e_clay_accum: ArrayLike,
    e_sed_accum: ArrayLike,
    E_all_mass_ha: np.ndarray,
) -> None:
    """
    Populate debris load columns in-place for each headwater endpoint.

    Samples four accumulated erosion rasters at each headwater's outlet
    coordinates and adds the values as new columns to fire_impact_data.

    Parameters:
    - fire_impact_data: DataFrame of per-headwater inputs (modified
      in-place).
    - transform: Affine transform shared by all erosion rasters.
    - e_all_accum: Accumulated total erosion mass raster (kg).
    - e_clay_accum: Accumulated clay erosion mass raster (kg).
    - e_sed_accum: Accumulated sediment erosion mass raster (kg).
    - E_all_mass_ha: Accumulated total erosion per hectare raster
      (kg/ha).

    Returns:
    - None
    """
    iter_this = [
        (e_clay_accum, CLY_M_ACC_KG),
        (e_all_accum, TOT_EM_ACC_KG),
        (E_all_mass_ha, TOT_EM_ACC_KG_HA),
        (e_sed_accum, SED_M_ACC_KG),
    ]

    # For each raster, look up the value at each headwater endpoint
    for array, field_name in iter_this:
        fire_impact_data[field_name] = fire_impact_data.apply(
            lambda row: get_debris_volume(
                x=row[HW_ENDP_X],
                y=row[HW_ENDP_Y],
                transform=transform,
                debris_volume_array=array,
            ),  # type: ignore[arg-type]
            axis=1,
        )
    return None


def calc_debris_constituent_cols(
    fire_impact_data: pd.DataFrame,
    debris_flow_constituents: pd.DataFrame,
) -> None:
    """
    Estimate elemental constituent masses from debris sediment loads.

    Uses average mg/kg concentrations from the debris-constituents
    lookup table to populate new columns in fire_impact_data in-place.

    Parameters:
    - fire_impact_data: DataFrame of per-headwater debris results
      (modified in-place).
    - debris_flow_constituents: Lookup DataFrame with constituent
      name and average mg/kg columns.

    Returns:
    - None
    ------------------------------------------------------------------------
    Notes:
    - Each element's mass is estimated as average_mg_per_kg converted
      to kg/kg, then multiplied by the total sediment mass in kg.
    ------------------------------------------------------------------------
    """
    for _, row in debris_flow_constituents.iterrows():
        particulate = row[PCLE_CTUENT_NAME]
        Average_Amount = row[AVG_CTUENT_MGPKG]
        column_name = f"{particulate} (Kg)"
        fire_impact_data[column_name] = (
            fire_impact_data[SED_M_ACC_KG]
            * (Average_Amount * MILLIGRAMS_TO_KILOGRAMS)
        )
    return None


def _clip_to_lookup_bins(values, bins, label):
    """
    Round values onto the lookup's discrete bins, clipping out-of-range.

    The join against the HF lookup is a left join on exact bin values, so
    anything outside the tabulated range simply fails to match, leaving
    I12_crit as NaN — and a headwater with a NaN threshold is skipped
    entirely by the event count, silently dropping it from the results.

    Clipping to the nearest tabulated bin makes that an explicit,
    reported saturation instead. It matters for slope in particular:
    gradients exceed the table's top bin of 1.0 above 45 degrees, which
    is steep but reachable for a headwater mean in alpine terrain.

    Parameters:
    - values: Series of continuous values.
    - bins: the lookup column holding the discrete bin values.
    - label: name used in the warning.

    Returns:
    - Series rounded to 1 dp and clipped to the bin range.
    """
    low, high = float(np.min(bins)), float(np.max(bins))
    rounded = values.round(1)
    outside = (rounded < low) | (rounded > high)
    n_outside = int(outside.sum())
    if n_outside:
        logger.warning(
            '%d of %d headwaters have a %s outside the lookup range '
            '[%.1f, %.1f] (min %.2f, max %.2f); clipping to the nearest '
            'tabulated bin. Without clipping these would not match the '
            'lookup and would be dropped from the debris-flow results '
            'without further warning.',
            n_outside, len(rounded), label, low, high,
            float(values.min()), float(values.max()),
        )
    return rounded.clip(lower=low, upper=high)


def calc_I12_crit_columns(
    fire_impact_data: pd.DataFrame,
    hf_lookup: pd.DataFrame,
    hw_id_field: str,
) -> pd.DataFrame:
    """
    Merge critical 12-minute rainfall intensity thresholds into the
    headwater table for Year 1 and Year 2 post-fire.

    Rounds aridity index, dNBR, and slope values to match the lookup
    table's discrete bins, then left-joins the HF lookup twice —
    once for years < 1 and once for years >= 1.

    Parameters:
    - fire_impact_data: DataFrame of per-headwater debris inputs.
    - hf_lookup: HFlookup DataFrame with I12_crit_mean values keyed
      by aridity index, dNBR, slope, and years-since-fire bins.
    - hw_id_field: Column name used as the headwater ID key.

    Returns:
    - Updated DataFrame with I12_crit_Year1 and I12_crit_Year2
      columns added and the join-key columns dropped.
    """
    # Round aridity index up to nearest 0.25 to match HFlookup bins
    fire_impact_data[ARID_MEAN_ADJ] = np.ceil(
        fire_impact_data[ARID_MEAN].round(2) / 0.25
    ) * 0.25
    # Round dNBR to nearest 100 to match lookup bins
    fire_impact_data[DNBR_MEAN_ADJ] = (
        fire_impact_data[DNBR_MEAN]
    ).round(-2).astype("int64")
    # The lookup's slope column holds dimensionless gradients (rise/run),
    # so degrees must be converted with tan, not divided by 100. The old
    # `/ 100` put every headwater in a flatter bin than it belonged to:
    # a 26 degree slope became 0.3 instead of 0.5, and since I12_crit
    # falls as slope rises, the critical intensity came out 1.2-1.4x too
    # high across the usual range — debris flows were under-triggered.
    gradient = np.tan(np.radians(fire_impact_data[SLOPE_DEG_MEAN]))
    fire_impact_data[SLOPE_DEG_MEAN_ADJ] = _clip_to_lookup_bins(
        gradient, hf_lookup[HF_GRADIENT_THRESH], 'slope gradient',
    )

    join_keys_in_lookup = [
        HF_ARID_IDX_THRESH, HF_DNBR_THRESH, HF_GRADIENT_THRESH
    ]
    join_keys_in_data = [
        ARID_MEAN_ADJ, DNBR_MEAN_ADJ, SLOPE_DEG_MEAN_ADJ
    ]

    # Merge Year 1 thresholds (years < 1 in the lookup)
    HFlookup_year_1 = hf_lookup[hf_lookup["years"] < 1]
    merged_year_1 = pd.merge(
        fire_impact_data,
        HFlookup_year_1,
        left_on=join_keys_in_data,
        right_on=join_keys_in_lookup,
        how="left",
    ).rename(columns={
        HF_YEARS_THRESH: "TSF_Year_1",
        HF_I12_CRIT: I12_CRIT_Y + '1',
    })

    # Merge Year 2 thresholds (years >= 1 in the lookup)
    HFlookup_year_2 = hf_lookup[hf_lookup["years"] >= 1]
    merged_year_2 = pd.merge(
        fire_impact_data,
        HFlookup_year_2,
        left_on=join_keys_in_data,
        right_on=join_keys_in_lookup,
        how="left",
    ).rename(columns={
        HF_YEARS_THRESH: "TSF_Year_2",
        HF_I12_CRIT: I12_CRIT_Y + '2',
    })

    # Combine Year 1 and Year 2 results into a single table
    fire_impact_data = pd.merge(
        merged_year_1,
        merged_year_2[[hw_id_field, "TSF_Year_2", I12_CRIT_Y + '2']],
        on=[hw_id_field],
        how="left",
    ).drop(columns=join_keys_in_lookup, errors="ignore")

    return fire_impact_data


def debris_flow_load(
    dem_data,
    slope_ratio,
    slope_transform,
    flow_accumulation,
    flowdir,
    clay0_5_fraction,
    clay5_15_fraction,
    out_path,
    fire_impact_data: pd.DataFrame,
    hf_lookup: pd.DataFrame,
    debris_flow_constituents: pd.DataFrame,
    raster_meta,
    id_field: str,
):
    """
    Calculate per-pixel debris flow erosion and populate headwater results.

    Runs the full debris flow load calculation: computes pixel erosion,
    accumulates it along the flow network, samples the accumulated values
    at each headwater outlet, and appends constituent masses and I12
    critical rainfall thresholds.

    Parameters:
    - dem_data: 2-D DEM array (used to define the catchment mask).
    - slope_ratio: 2-D slope gradient array (rise/run).
    - slope_transform: Affine transform for the slope/flow rasters.
    - flow_accumulation: pysheds Raster of flow accumulation counts.
    - flowdir: pysheds Raster of flow directions.
    - clay0_5_fraction: 2-D clay fraction array for 0–5 cm depth.
    - clay5_15_fraction: 2-D clay fraction array for 5–15 cm depth.
    - out_path: Directory for output rasters and CSVs.
    - fire_impact_data: Per-headwater input DataFrame; extended
      in-place with erosion and threshold columns.
    - hf_lookup: HFlookup DataFrame for I12 critical thresholds.
    - debris_flow_constituents: Lookup table of constituent mg/kg
      values.
    - raster_meta: Rasterio metadata dict for output rasters.
    - id_field: Column name used as the headwater ID key.

    Returns:
    - Updated fire_impact_data DataFrame with all debris load and
      I12 threshold columns populated.
    """
    # Derive pixel area and convert flow accumulation count to area
    xres = slope_transform[0]
    yres = abs(slope_transform[4])
    pixel_area = xres * yres
    flow_acc_area = flow_accumulation * pixel_area
    catchment_mask = ~np.isnan(dem_data)
    inter_meta = raster_meta.copy()
    inter_meta.update(dtype=rasterio.float32)

    # Compute per-pixel net erosion for hillslope and channel zones
    erosion_mass_all, erosion_mass_clay, Sediment_mass = (
        compute_net_erosion(
            flow_acc_area, clay0_5_fraction, clay5_15_fraction,
            slope_ratio, pixel_area,
        )
    )

    # Check for unexpected NaN inside the catchment
    in_catchment = catchment_mask.sum()
    for label, arr in [('erosion_mass_all', erosion_mass_all),
                       ('erosion_mass_clay', erosion_mass_clay),
                       ('Sediment_mass', Sediment_mass)]:
        nan_inside = int(np.isnan(arr[catchment_mask]).sum())
        if nan_inside > 0:
            logger.warning(
                '%s has %d NaN pixels inside the catchment '
                '(out of %d). Likely caused by NaN in clay '
                'fraction or slope inputs.',
                label, nan_inside, in_catchment,
            )

    # Accumulate pixel erosion along the flow network
    e_all_accum, e_clay_accum, e_sed_accum = create_cum_erosion_layers(
        erosion_mass_all, erosion_mass_clay, Sediment_mass,
        flowdir, out_path, inter_meta, catchment_mask,
    )

    # Check accumulated layers for NaN inside catchment
    for label, arr in [('e_all_accum', e_all_accum),
                       ('e_clay_accum', e_clay_accum),
                       ('e_sed_accum', e_sed_accum)]:
        nan_inside = int(
            np.isnan(np.asarray(arr)[catchment_mask]).sum()
        )
        if nan_inside > 0:
            logger.warning(
                '%s has %d NaN pixels inside the catchment after '
                'flow accumulation.',
                label, nan_inside,
            )

    # Save a kg/ha sense-check raster
    E_all_mass_ha = create_erosion_sense_check(
        e_all_accum, flow_acc_area, inter_meta, out_path
    )

    # Sample the accumulated rasters at each headwater outlet
    debris_column_values(
        fire_impact_data, slope_transform,
        e_all_accum=e_all_accum,
        e_clay_accum=e_clay_accum,
        e_sed_accum=e_sed_accum,
        E_all_mass_ha=E_all_mass_ha,
    )

    # Check for NaN in the extracted headwater values
    debris_cols = [
        CLY_M_ACC_KG, TOT_EM_ACC_KG, TOT_EM_ACC_KG_HA, SED_M_ACC_KG
    ]
    for col in debris_cols:
        if col in fire_impact_data.columns:
            n_nan = int(fire_impact_data[col].isna().sum())
            if n_nan > 0:
                nan_ids = fire_impact_data.loc[
                    fire_impact_data[col].isna(), id_field
                ].tolist()
                logger.warning(
                    '%s has NaN for %d headwaters: %s. '
                    'Likely caused by endpoint coordinates falling '
                    'outside the erosion accumulation raster bounds.',
                    col, n_nan, nan_ids,
                )

    # Append elemental constituent mass columns
    calc_debris_constituent_cols(
        fire_impact_data, debris_flow_constituents
    )

    # Append I12 critical rainfall thresholds for Year 1 and Year 2
    updated_data = calc_I12_crit_columns(
        fire_impact_data, hf_lookup, id_field
    )

    return updated_data


# ---------------------------------------------------------------------------
# Subcatchment aggregation of debris flow results
# ---------------------------------------------------------------------------

def allocate_headwaters_to_subcatchments(
    project: FireImpactsProject,
    catchment: str,
    area_fraction_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Allocate headwaters to subcatchments based on spatial overlay.

    Each headwater is assigned to the subcatchment that contains the
    largest fraction of its area.  Allocations below the area-fraction
    threshold are dropped to handle minor boundary misalignments.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment: Catchment identifier.
    - area_fraction_threshold: Minimum overlap fraction for a valid
      allocation.  Default is 0.1 (10%).

    Returns:
    - DataFrame with columns: SiteID, hw_ID, Area_m2,
      area_intersect, area_fraction.
    """
    headwaters = project.get_headwaters(catchment)
    subcatchments = project.get_subcatchments(catchment)

    # Spatial overlay to find headwater–subcatchment intersections
    hw_intersection = gpd.overlay(headwaters, subcatchments)
    hw_intersection['area_intersect'] = (
        hw_intersection.geometry.area
    )
    hw_intersection['area_fraction'] = (
        hw_intersection['area_intersect']
        / hw_intersection['Area_m2']
    )

    # Keep only the largest intersection per headwater
    hw_allocations = (
        hw_intersection
        .sort_values('area_fraction', ascending=False)
        .drop_duplicates(subset=['hw_ID'], keep='first')
    )

    # Filter out allocations below the area threshold
    hw_allocations = hw_allocations[
        hw_allocations['area_fraction'] > area_fraction_threshold
    ]

    result = hw_allocations[
        [SC_ID, HW_ID, 'Area_m2', 'area_intersect', 'area_fraction']
    ].copy()

    logger.info(
        f'Allocated {len(result)} headwaters to subcatchments '
        f'in {catchment}. '
        f'Filtered {len(hw_allocations) - len(result)} allocations '
        f'below {area_fraction_threshold} threshold.'
    )
    return result


def scale_debris_timeseries_by_allocation(
    debris_timeseries: pd.DataFrame,
    hw_allocations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale debris flow timeseries by headwater-to-subcatchment fractions.

    Multiplies each headwater's sediment load column by its area
    fraction to account for partial overlap with a subcatchment.

    Parameters:
    - debris_timeseries: DataFrame with headwater IDs as columns and
      a datetime index.  Values are sediment loads in kg.
    - hw_allocations: Allocation DataFrame from
      allocate_headwaters_to_subcatchments().

    Returns:
    - DataFrame of scaled timeseries, limited to headwaters present
      in hw_allocations.
    """
    # Cap fractions at 1.0 — minor rounding can push them slightly over
    fractions = np.minimum(
        1.0,
        hw_allocations.set_index(HW_ID)['area_fraction'],
    )

    # Keep only headwaters that have a spatial allocation
    scaled = debris_timeseries.copy()
    scaled = scaled[
        [col for col in scaled.columns if col in fractions.index]
    ]
    scaled = scaled * fractions

    logger.info(
        f'Scaled debris timeseries for {len(fractions)} headwaters. '
        f'Data shape: {scaled.shape}'
    )
    return scaled


def aggregate_debris_to_subcatchments(
    debris_timeseries: pd.DataFrame,
    hw_allocations: pd.DataFrame,
    time_resolution: str = '12min',
) -> pd.DataFrame:
    """
    Aggregate debris flow timeseries from headwaters to subcatchments.

    Scales each headwater's timeseries by its area fraction, maps
    headwater IDs to subcatchment IDs (SiteID), then sums across
    headwaters within each subcatchment.

    Parameters:
    - debris_timeseries: DataFrame with headwater IDs as columns, a
      datetime index, and sediment loads (kg) as values.
    - hw_allocations: Allocation DataFrame from
      allocate_headwaters_to_subcatchments().
    - time_resolution: Time resolution label used in log messages.

    Returns:
    - DataFrame with subcatchment SiteIDs as columns and the same
      datetime index, containing total sediment loads per subcatchment
      at the original 12-minute resolution.
    ------------------------------------------------------------------------
    Notes:
    - Use resample() on the returned DataFrame to aggregate to hourly
      ('H'), daily ('D'), or other resolutions.
    ------------------------------------------------------------------------
    """
    scaled = scale_debris_timeseries_by_allocation(
        debris_timeseries, hw_allocations
    )

    # Map headwater IDs to subcatchment IDs, then sum within each SC
    sc_map = hw_allocations.set_index(HW_ID)[SC_ID].to_dict()
    scaled = scaled.rename(columns=sc_map)
    aggregated = scaled.groupby(level=0, axis=1).sum()

    logger.info(
        f'Aggregated debris timeseries from '
        f'{len(scaled.columns)} headwaters '
        f'(at {time_resolution} resolution) to '
        f'{len(aggregated.columns)} subcatchments.'
    )
    return aggregated


def resample_debris_timeseries(
    debris_timeseries: pd.DataFrame,
    freq: str = 'H',
) -> pd.DataFrame:
    """
    Resample a debris flow timeseries to a coarser temporal resolution.

    Parameters:
    - debris_timeseries: DataFrame with a datetime index and
      subcatchment columns.  Values are sediment loads in kg.
    - freq: Resampling frequency string accepted by pandas resample()
      (e.g. 'H' for hourly, 'D' for daily).  Default is 'H'.

    Returns:
    - DataFrame of summed sediment loads at the requested resolution.
    """
    resampled = debris_timeseries.resample(freq).sum()

    logger.info(
        f'Resampled debris timeseries to {freq} resolution. '
        f'Shape: {debris_timeseries.shape} -> {resampled.shape}'
    )
    return resampled


def postprocess_debris_flow(
    ctx: RunContext,
    debris_timeseries,
    area_fraction_threshold: float = 0.1,
    resample_freq: str = None,
    save: bool = True,
) -> dict:
    """
    Complete post-processing workflow for debris flow simulation results.

    Performs the spatial headwater-to-subcatchment allocation once, then
    aggregates and optionally resamples each replicate's timeseries.

    Parameters:
    - ctx: run-level RunContext. Saved outputs land under
      Runs/<event>/<ensemble>/DebrisFlow/.
    - debris_timeseries: Either a single DataFrame (one replicate) or
      a dict keyed by replicate index where each value is a DataFrame.
      Each DataFrame has headwater IDs as columns, a datetime index,
      and sediment loads (kg) as values.
    - area_fraction_threshold: Minimum overlap fraction for headwater
      allocation.  Default 0.1.
    - resample_freq: If provided, resample aggregated outputs to this
      frequency (e.g. 'H', 'D').  None keeps original resolution.
    - save: If True, save results as CSV files in the DebrisFlow
      directory.

    Returns:
    - Dict with keys: aggregated, allocations, and (if resample_freq
      is set) resampled.  When debris_timeseries is a single
      DataFrame, aggregated and resampled are DataFrames.  When it is
      a dict of DataFrames, they are dicts keyed by replicate index.
    """
    ctx.validate()
    project = ctx.project
    catchment = ctx.catchment

    # Compute the spatial allocation once, reused across all replicates
    hw_allocations = allocate_headwaters_to_subcatchments(
        project, catchment, area_fraction_threshold
    )

    # Normalise input to a dict so single- and multi-replicate paths
    # share the same code
    if isinstance(debris_timeseries, pd.DataFrame):
        ts_dict = {0: debris_timeseries}
        single_replicate = True
    else:
        ts_dict = debris_timeseries
        single_replicate = False

    aggregated = {}
    resampled = {}
    for rep_key, ts in ts_dict.items():
        agg = aggregate_debris_to_subcatchments(ts, hw_allocations)
        aggregated[rep_key] = agg
        if resample_freq is not None:
            resampled[rep_key] = resample_debris_timeseries(
                agg, resample_freq
            )

    # Relabel sc_ID columns with the configured subcatchment label
    # field (typically 'SiteID') so outputs are human-readable
    label_field = project.subcatchment_label_field(catchment)
    if label_field:
        subs = project.get_subcatchments(catchment)
        if label_field in subs.columns and SC_ID in subs.columns:
            label_map = dict(zip(subs[SC_ID], subs[label_field]))
            for rep_key in list(aggregated.keys()):
                aggregated[rep_key] = aggregated[rep_key].rename(
                    columns=label_map,
                )
                if rep_key in resampled:
                    resampled[rep_key] = resampled[rep_key].rename(
                        columns=label_map,
                    )
        else:
            logger.warning(
                "Subcatchment label field '%s' is configured for "
                "catchment '%s' but is not present in the saved "
                "subcatchments shapefile (columns: %s). Debris-flow "
                "outputs will keep integer sc_ID column labels.",
                label_field, catchment, list(subs.columns),
            )

    if save:
        out_path = ctx.run_path('DebrisFlow')
        os.makedirs(out_path, exist_ok=True)

        alloc_file = os.path.join(
            out_path, 'headwater_to_subcatchment_allocations.csv'
        )
        hw_allocations.to_csv(alloc_file, index=False)
        logger.info(f'Saved headwater allocations to {alloc_file}')

        for rep_key, agg in aggregated.items():
            suffix = '' if single_replicate else f'_rep{rep_key}'
            agg_file = os.path.join(
                out_path,
                f'debris_flow_aggregated_by_subcatchment'
                f'{suffix}.csv',
            )
            agg.to_csv(agg_file)
            logger.info(
                f'Saved aggregated debris timeseries to {agg_file}'
            )

            if resample_freq is not None:
                res_file = os.path.join(
                    out_path,
                    f'debris_flow_aggregated_by_subcatchment'
                    f'{suffix}_{resample_freq}.csv',
                )
                resampled[rep_key].to_csv(res_file)
                logger.info(
                    f'Saved resampled ({resample_freq}) timeseries '
                    f'to {res_file}'
                )

    logger.info(
        f'Post-processing complete for catchment: {catchment}'
    )

    if single_replicate:
        output = {
            'aggregated': aggregated[0],
            'allocations': hw_allocations,
        }
        if resample_freq is not None:
            output['resampled'] = resampled[0]
    else:
        output = {
            'aggregated': aggregated,
            'allocations': hw_allocations,
        }
        if resample_freq is not None:
            output['resampled'] = resampled

    return output


def aggregate_debris_flow_summary_to_subcatchments(
    ctx: RunContext,
    debris_flow_data: pd.DataFrame,
) -> 'pd.DataFrame | None':
    """
    Aggregate headwater-level debris flow summary statistics to
    subcatchments and save a CSV to the DebrisFlow folder.

    Parameters:
    - ctx: run-level RunContext.
    - debris_flow_data: Per-headwater summary DataFrame produced by
      debris_flow().

    Returns:
    - DataFrame with one row per subcatchment, or None if no
      subcatchments are defined.
    ------------------------------------------------------------------------
    Notes:
    - Event counts (Year1_num_events, Year2_num_events) are summed
      directly across headwaters — a debris flow event is a discrete
      occurrence and area weighting is not applied.
    - Mass columns (clay, total erosion, sediment) are multiplied by
      each headwater's area_fraction before summing, so a headwater
      80% within subcatchment X contributes 80% of its mass there.
    - Headwaters without a spatial allocation are excluded.
    ------------------------------------------------------------------------
    """
    try:
        hw_alloc = allocate_headwaters_to_subcatchments(
            ctx.project, ctx.catchment,
        )
    except FileNotFoundError:
        logger.info(
            'No subcatchments defined for %s — skipping debris flow '
            'subcatchment aggregation.',
            ctx.catchment,
        )
        return None

    # Build column lists, checking each column actually exists
    count_cols = [
        f'Year{y}_num_events' for y in range(1, NUM_SIM_YEARS + 1)
    ]
    mass_cols = [CLY_M_ACC_KG, TOT_EM_ACC_KG, SED_M_ACC_KG]
    # I12 threshold columns — aggregate min (most vulnerable) and mean
    thresh_cols = [
        I12_CRIT_Y + str(y) for y in range(1, NUM_SIM_YEARS + 1)
    ]

    avail_count = [
        col for col in count_cols if col in debris_flow_data.columns
    ]
    avail_mass = [
        col for col in mass_cols if col in debris_flow_data.columns
    ]
    avail_thresh = [
        col for col in thresh_cols if col in debris_flow_data.columns
    ]

    keep = [HW_ID] + avail_count + avail_mass + avail_thresh
    working = debris_flow_data[keep].copy()

    # Merge in sc_ID and area_fraction from the spatial allocation
    working = pd.merge(
        working,
        hw_alloc[[HW_ID, SC_ID, 'area_fraction']],
        on=HW_ID,
        how='inner',  # headwaters without an allocation are dropped
    )

    # Weight mass columns by area_fraction before summing
    for col in avail_mass:
        working[col] = working[col] * working['area_fraction']

    # Sum event counts and weighted masses within each subcatchment
    agg_cols = avail_count + avail_mass
    result = (
        working.groupby(SC_ID)[agg_cols]
        .sum()
        .reset_index()
    )

    # Min and mean I12 threshold per subcatchment (suffixed _min/_mean)
    if avail_thresh:
        thresh_agg = (
            working.groupby(SC_ID)[avail_thresh]
            .agg(['min', 'mean'])
        )
        thresh_agg.columns = [
            f'{col}_{stat}' for col, stat in thresh_agg.columns
        ]
        result = pd.merge(
            result, thresh_agg.reset_index(), on=SC_ID, how='left'
        )

    out_path = ctx.run_path('DebrisFlow', DEBRIS_SC_SUMMARY_NAME + '.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info(
        'Saved debris flow subcatchment summary to %s', out_path
    )
    return result


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------

def debris_flow(
    ctx: RunContext,
    rainfall,
    save: bool = True,
    save_daily_catchment_timeseries: bool = True,
    prepared=None,
    dnbr_threshold=UNSET,
    params=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the debris flow simulation for the context.

    Iterates over rainfall timesteps to count events exceeding each
    headwater's I12 critical threshold for Year 1 and Year 2 post-fire.

    Parameters:
    - ctx: run-level RunContext. Outputs land under
      Runs/<event>/<ensemble>/DebrisFlow/.
    - rainfall: pd.Series of rainfall intensities (mm/h) with a
      datetime index at 12-minute resolution.
    - save: If True, save per-headwater results to CSV.
    - save_daily_catchment_timeseries: If True, save a daily aggregate
      of total debris mass for the whole catchment.
    - prepared: Pre-computed output of prep_debris_flow_simulation()
      for this context. Reusing prepared data is required when running
      multiple rainfall replicates concurrently to avoid scratch-raster
      write races.
    - dnbr_threshold: Deprecated. Use the debris parameter group. Only
      used when prepared is None (i.e. when this call runs
      prep_debris_flow_simulation itself).
    - params: Calibration parameters, forwarded to
      prep_debris_flow_simulation. Only used when prepared is None.

    Returns:
    - Tuple of (Debris_Flow_Data, event_ts) where Debris_Flow_Data is
      the per-headwater summary DataFrame and event_ts is a DataFrame
      of per-headwater event counts indexed by datetime.
    ------------------------------------------------------------------------
    Notes:
    - Headwaters with excessive NaN in masked_dNBR should be filtered
      upstream by summary_stats(masked_nan_threshold=...). The inner
      join in prep_debris_flow_simulation enforces that filtering.
    ------------------------------------------------------------------------
    """
    ctx.validate()
    out_path = ctx.run_path('DebrisFlow')
    os.makedirs(out_path, exist_ok=True)

    # Validate rainfall units
    if 'units' not in rainfall.attrs:
        logger.warning(
            "Rainfall data has no units attribute, assuming units "
            "are correct (mm/hr)"
        )
    elif rainfall.attrs['units'] != 'mm/h':
        raise ValueError(
            "Rainfall data has units '%s', expected 'mm/h'",
            rainfall.attrs['units'],
        )

    # Use pre-computed inputs if provided, otherwise compute now.
    # Deep-copy so per-year mutations don't leak back to the cached copy.
    if prepared is not None:
        working_deb_flow_data = prepared.copy(deep=True)
    else:
        working_deb_flow_data = prep_debris_flow_simulation(
            ctx, dnbr_threshold=dnbr_threshold, params=params
        )

    # --- Note: this section may be superseded by recorders ----------
    event_ts = pd.DataFrame(
        data=np.zeros(
            (len(rainfall.index), len(working_deb_flow_data[HW_ID])),
            dtype=np.uint8,
        ),
        index=rainfall.index,
        columns=working_deb_flow_data[HW_ID],
    )

    years = range(1, NUM_SIM_YEARS + 1)
    year_results = {
        year: {
            "event_counts": [],
            "rainfall_events": [],
            "event_dates": [],
        }
        for year in years
    }
    t0 = rainfall.index[0]

    # Iterate through each simulated year
    for year in years:
        t1 = t0 + pd.Timedelta(days=365)
        threshold_col = I12_CRIT_Y + str(year)
        rain_year = rainfall[
            (rainfall.index >= t0) & (rainfall.index < t1)
        ]
        t0 = t1

        for idx, row in working_deb_flow_data.iterrows():
            threshold = row[threshold_col]
            hw_id = row[HW_ID]
            if np.isnan(threshold):
                year_results[year]["event_counts"].append(0)
                year_results[year]["rainfall_events"].append([])
                year_results[year]["event_dates"].append([])
                continue

            rain_flat = rain_year.values
            indices = np.where(rain_flat >= threshold)[0]
            events = rain_flat[indices]
            event_dates_row = rain_year.index[indices]

            year_results[year]["event_counts"].append(len(events))
            year_results[year]["rainfall_events"].append(
                events.tolist()
            )
            year_results[year]["event_dates"].append(event_dates_row)
            for d in event_dates_row:
                event_ts.at[d, hw_id] += 1

        # Add event count column for this year
        working_deb_flow_data[
            f"Year{year}_num_events"
        ] = year_results[year]["event_counts"]

        max_events = max(
            len(ev)
            for ev in year_results[year]["rainfall_events"]
        )

        # Build per-event rainfall and date columns for this year
        sim_columns = {}
        for j in range(max_events):
            sim_columns[f"Year{year}_rainfall_event{j+1}"] = [
                ev[j] if j < len(ev) else np.nan
                for ev in year_results[year]["rainfall_events"]
            ]
            sim_columns[f"Year{year}_event{j+1}_date"] = [
                f"{date[j].date().isoformat()}"
                if j < len(date) else np.nan
                for date in year_results[year]["event_dates"]
            ]

        for col_name, col_values in sim_columns.items():
            working_deb_flow_data[col_name] = col_values

    Debris_Flow_Data = working_deb_flow_data.copy()

    if save:
        Debris_Flow_Data_path = os.path.join(
            out_path, 'DebrisFlowData.csv'
        )
        Debris_Flow_Data.to_csv(Debris_Flow_Data_path, index=False)
        logger.info(
            'Saved debris flow by headwater results table to '
            f'{Debris_Flow_Data_path}'
        )
        # Aggregate summary stats to subcatchments (skipped if none)
        aggregate_debris_flow_summary_to_subcatchments(
            ctx, Debris_Flow_Data,
        )

    if save_daily_catchment_timeseries:
        # Multiply per-timestamp event counts by headwater mass to get
        # debris mass delivered, then aggregate to daily totals
        mass_df = Debris_Flow_Data[
            [HW_ID, DEBRIS_MASS_FIELD]
        ].set_index(HW_ID, drop=True)
        flow_mass = event_ts.mul(
            mass_df[DEBRIS_MASS_FIELD], axis=1
        )
        flow_mass[CATCH_TOTAL_DEBRIS_TONNES] = (
            flow_mass.sum(axis=1) / 1e3
        )
        flow_mass = (
            flow_mass[[CATCH_TOTAL_DEBRIS_TONNES]].resample('D').sum()
        )
        out_name = ctx.run_path(
            RESULTS_FOLDER_NAME, DEBRIS_OP_TIMESERIES_NAME + '.csv',
        )
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        logger.info(
            'Saved daily catchment-level debris flow mass totals to '
            f'{out_path}'
        )
        flow_mass.to_csv(out_name)

    logger.info('Done!')
    return Debris_Flow_Data, event_ts


def event_ts_to_mass(
    summary_df: pd.DataFrame,
    event_ts: pd.DataFrame,
    mass_col: str = DEBRIS_MASS_FIELD,
) -> pd.DataFrame:
    """
    Convert a per-headwater event-count timeseries to a mass timeseries.

    Parameters:
    - summary_df: Headwater summary table (first element of the tuple
      returned by debris_flow()), containing hw_ID and a mass column.
    - event_ts: Event-count timeseries (second element of the tuple
      returned by debris_flow()) with headwater IDs as columns.
    - mass_col: Name of the mass-per-event column in summary_df.

    Returns:
    - DataFrame with the same shape as event_ts, with values in kg.
    """
    mass = summary_df[[HW_ID, mass_col]].set_index(HW_ID)[mass_col]
    return event_ts * mass


def _prepare_debris_flow_per_catchment(
    ctx: RunContext,
    dnbr_threshold: float = DEFAULT_DEBRIS_DNBR_THRESHOLD,
):
    """
    Run prep_debris_flow_simulation() once for the context.

    The preparation step writes scratch rasters under the event's
    DebrisFlow directory and is expensive but rainfall-independent, so
    it is run once up-front and the result is reused across every
    rainfall replicate.

    Parameters:
    - ctx: event-level RunContext.
    - dnbr_threshold: Mean-dNBR cutoff below which headwaters are
      excluded from the analysis.

    Returns:
    - The prepared DataFrame returned by prep_debris_flow_simulation();
      pass straight through to debris_flow(prepared=...).
    """
    return prep_debris_flow_simulation(
        ctx, dnbr_threshold=dnbr_threshold)


def run_debris_flow_replicate(
    ctx: RunContext,
    rainfall_12min,
    replicate_idx: int,
    save: bool = False,
    save_daily_catchment_timeseries: bool = False,
    prepared=None,
) -> dict:
    """
    Run the debris flow simulation for a single rainfall replicate.

    Parameters:
    - ctx: run-level RunContext.
    - rainfall_12min: xarray.Dataset of 12-minute rainfall intensities
      with a 'replicate' dimension.
    - replicate_idx: Index of the replicate to run.
    - save: Whether to save per-headwater results to disk.
    - save_daily_catchment_timeseries: Whether to save daily totals.
    - prepared: Optional pre-computed prep_debris_flow_simulation()
      DataFrame for this context. Required when running concurrently
      across replicates — see run_debris_flow_all_replicates.

    Returns:
    - Dict keyed by catchment name (single key for ctx.catchment) with
      a (summary_df, mass_ts) tuple. mass_ts is the event-count
      timeseries converted to kg via event_ts_to_mass(). The per-
      catchment wrapper matches the shape consumed by save_run /
      ensemble.py helpers.
    """
    rain_seq = rainfall_12min.rainfall[:, replicate_idx].to_pandas()
    # Preserve units attribute if present on the dataset
    units = rainfall_12min.rainfall.attrs.get('units')
    if units is not None:
        rain_seq.attrs['units'] = units

    summary_df, event_ts = debris_flow(
        ctx, rain_seq,
        save=save,
        save_daily_catchment_timeseries=save_daily_catchment_timeseries,
        prepared=prepared,
    )
    mass_ts = event_ts_to_mass(summary_df, event_ts)
    return {ctx.catchment: (summary_df, mass_ts)}


def run_debris_flow_all_replicates(
    ctx: RunContext,
    rainfall_12min,
    n_workers: int = None,
    scheduler: str = 'threads',
    replicate_indices=None,
    save: bool = False,
    save_daily_catchment_timeseries: bool = False,
    prepared=None,
    dnbr_threshold: float = DEFAULT_DEBRIS_DNBR_THRESHOLD,
) -> dict:
    """
    Run the debris flow simulation across all replicates in parallel.

    The expensive, rainfall-independent preparation step is computed
    once up-front and reused to avoid scratch-raster write races when
    replicates run concurrently.

    Parameters:
    - ctx: run-level RunContext.
    - rainfall_12min: xarray.Dataset of 12-minute rainfall intensities
      with a 'replicate' dimension.
    - n_workers: Number of Dask workers.  None lets Dask decide.
    - scheduler: Dask scheduler — 'threads' or 'processes'.
    - replicate_indices: Iterable of replicate indices to run.
      Defaults to all replicates.
    - save: Whether to save per-headwater results to disk per replicate.
    - save_daily_catchment_timeseries: Whether to save daily totals per
      replicate.
    - prepared: Optional pre-computed DataFrame from
      prep_debris_flow_simulation(). When None, the prep step is
      invoked once before dispatching replicates.
    - dnbr_threshold: Mean-dNBR cutoff below which headwaters are
      excluded from the analysis. Only used when prepared is None.

    Returns:
    - Dict of {replicate_idx: {catchment: (summary_df, mass_ts)}}.
    """
    import dask

    if replicate_indices is None:
        replicate_indices = list(
            range(rainfall_12min.dims['replicate'])
        )
    else:
        replicate_indices = list(replicate_indices)

    if prepared is None:
        logger.info(
            'Preparing debris-flow inputs for %s once before '
            'dispatching replicates.', ctx.catchment,
        )
        prepared = _prepare_debris_flow_per_catchment(
            ctx, dnbr_threshold=dnbr_threshold, params=params)

    tasks = [
        dask.delayed(run_debris_flow_replicate)(
            ctx, rainfall_12min, i,
            save=save,
            save_daily_catchment_timeseries=(
                save_daily_catchment_timeseries
            ),
            prepared=prepared,
        )
        for i in replicate_indices
    ]
    computed = dask.compute(
        *tasks, scheduler=scheduler, num_workers=n_workers
    )
    return {i: r for i, r in zip(replicate_indices, computed)}


def run_debris_flow_sim(
    ctx: RunContext,
    rainfall: pd.DataFrame,
    recorders=None,
):
    """
    Run the debris flow simulation and record results as specified.

    Parameters:
    - ctx: run-level RunContext.
    - rainfall: 12-minute rainfall intensity data (mm/h) as a Series
      or DataFrame with a datetime index.
    - recorders: Optional dict of recorder functions to accumulate
      results during the simulation.

    Returns:
    - None (results are accumulated via the recorder objects).
    ------------------------------------------------------------------------
    Notes:
    - The core simulation logic is in debris_flow().
    ------------------------------------------------------------------------
    """
    ctx.validate()
    # Trim rainfall to start from the fire end date
    fire_end_dt = ctx.fire_end_date
    rainfall_trimmed = rainfall.loc[fire_end_dt:]

    # Check if timeseries covers a full 2 years since fire
    # (TODO: implement validation)

    if recorders is None:
        recorders = dict()
    for recorder in recorders.values():
        recorder.reset()


# ---------------------------------------------------------------------------
# Legacy and generator functions
# ---------------------------------------------------------------------------

def post_debris_flow_mass_adjustment(
    debris_flow_data: pd.DataFrame,
    ids_with_events: list[str],
    event_year: int,
    mass_col: str = CLY_M_ACC_KG,
):
    """
    Placeholder for adjusting available mass after a debris flow event.

    After an event the mass available in a headwater for subsequent
    events in the same year should decrease.  This function is a stub
    and currently returns the input DataFrame unchanged.

    Parameters:
    - debris_flow_data: Per-headwater DataFrame to adjust.
    - ids_with_events: List of headwater IDs that had an event.
    - event_year: The simulation year (1 or 2) in which the event
      occurred.
    - mass_col: Name of the mass column to adjust.

    Returns:
    - debris_flow_data unchanged (adjustment not yet implemented).
    """
    return debris_flow_data


def generate_debris_flow(
    rainfall: pd.Series,
    debris_flow_data: pd.DataFrame,
    id_field: str,
    out_path: str,
):
    """
    Generate per-timestep debris flow results for a rainfall series.

    Yields a (timestep, result_dict) tuple for each 12-minute interval
    in the rainfall series.

    Parameters:
    - rainfall: Series of rainfall intensities in mm/h at 12-minute
      resolution.  The first timestamp is assumed to be immediately
      after the fire ends.
    - debris_flow_data: Per-headwater input DataFrame as produced by
      prep_debris_flow_simulation().
    - id_field: Column name used as the headwater ID.
    - out_path: Path for any output files (passed through, not
      currently used by the generator itself).

    Returns:
    - Generator yielding (timestep, result) tuples where result is a
      dict with keys: total_rain, debris_flow_event_y1,
      debris_flow_mass_t_y1, debris_flow_event_y2,
      debris_flow_mass_t_y2.
    ------------------------------------------------------------------------
    Notes:
    - Both Year 1 and Year 2 values are present in every yielded dict.
      The caller is responsible for selecting the appropriate year
      based on the current timestep's position in the post-fire window.
    - Mass reset after an event (preventing the same headwater from
      contributing repeatedly) is not yet implemented here.
    - Spatially varying rainfall is not yet supported; a single scalar
      intensity is applied to all headwaters at each timestep.
    ------------------------------------------------------------------------
    """
    # Ensure rainfall is a Series for consistent indexing
    if isinstance(rainfall, pd.DataFrame):
        rainfall = pd.Series(
            data=rainfall['rainfall'], index=rainfall.index
        )

    working_copy = debris_flow_data.copy()
    year_1_thresh_col = I12_CRIT_Y + '1'
    year_2_thresh_col = I12_CRIT_Y + '2'
    subset = working_copy[[
        id_field,
        CLY_M_ACC_KG,
        year_1_thresh_col,
        year_2_thresh_col,
    ]]

    for timestep in rainfall.index:
        rain_intensity_12min = rainfall[timestep]
        rain_depth_over_12_min = rain_intensity_12min / 5

        result = {
            'total_rain': rain_depth_over_12_min,
            'debris_flow_event_y1': 0,
            'debris_flow_mass_t_y1': 0.0,
            'debris_flow_event_y2': 0,
            'debris_flow_mass_t_y2': 0.0,
        }

        # Skip timesteps with no rain
        if rain_intensity_12min == 0:
            yield (timestep, result)
            continue

        # Check if any headwater threshold is exceeded for either year
        mask = (
            subset[year_1_thresh_col] < rain_intensity_12min
            | subset[year_2_thresh_col] < rain_intensity_12min
        )
        if not mask.any():
            yield (timestep, result)
            continue

        # Year 1 events
        mask_y1 = subset[year_1_thresh_col] < rain_intensity_12min
        if mask_y1.any():
            y1_event_count = mask_y1.sum()
            y1_event_deets = subset.loc[mask_y1]
            y1_mass_kg = np.nansum(y1_event_deets[CLY_M_ACC_KG])
            y1_mass_t = y1_mass_kg * KG_TO_TONNES
            result['debris_flow_event_y1'] = y1_event_count
            result['debris_flow_mass_t_y1'] = y1_mass_t

        # Year 2 events
        mask_y2 = subset[year_2_thresh_col] < rain_intensity_12min
        if mask_y2.any():
            y2_event_count = mask_y2.sum()
            y2_event_deets = subset.loc[mask_y2]
            y2_mass_kg = np.nansum(y2_event_deets[CLY_M_ACC_KG])
            y2_mass_t = y2_mass_kg * KG_TO_TONNES
            result['debris_flow_event_y2'] = y2_event_count
            result['debris_flow_mass_t_y2'] = y2_mass_t

        yield (timestep, result)


def record_headwaters_timeseries(
    project: FireImpactsProject,
    variable_name: str,
    agg_type: str = 'sum',
    label_field=None,
    agg_count: int = 1,
):
    """
    Build a debris flow recorder that summarises mass over time per
    headwater.

    Parameters:
    - project: FireImpactsProject instance.
    - variable_name: Name of the variable to record.
    - agg_type: Aggregation type to apply (e.g. 'sum', 'mean').
    - label_field: Optional field to use as the headwater label.
    - agg_count: Number of timesteps to aggregate per output row.

    Returns:
    - None (stub — not yet implemented).
    """
    pass


# NOTES/QUESTIONS
#
# Debris flow
# * Two years? (But assume starting 1 January, should be any day of
#   the year)
# * Two years? Parameter?
#
# RUSLE
# * Gridded output? (Generator?)
#
# Both
# * Independent of stochastic replicates
# * Possibility of spatial rainfall?
# * Results on different spatial / temporal aggregations
# * Not duplicating code we already have (ie pysheds)
