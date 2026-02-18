import xarray as xr
# import geopandas as gpd
# import rioxarray
import pandas as pd
from shapely.geometry import mapping
import numpy as np
from numpy.typing import ArrayLike
# import dask
import rasterio
from rasterio.warp import Resampling
from rasterio.transform import from_origin, Affine, rowcol
from fire_impacts.const import *
from fire_impacts.pre import topography
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre.util import read_aligned, read_raster
from fire_impacts.util import load_package_data, unique_file_matching
from pysheds.grid import Grid
import os
import logging
import geopandas as gpd

logger = logging.getLogger(__name__)


###############################################################################
def get_flow_layers(
    hydro_dem,
    dem_meta:dict,
    grid:Grid,
    dirmap:tuple,
    project:FireImpactsProject,
    catchment_name:str
    ):
    """
    Grab the flow direction and flow accumulation rasters if they're
    already saved, otherwise compute new ones of each

    Parameters:
    - hydro_dem: Hydrologically enforced DEM
    - dem_meta: Metadata dictionary for the hydro_dem
    - grid: pysheds Grid object which can be used for hydro DEM ops
    - dirmap: tuple of values for D8 flow direction cell assignment
    - project: FireImpactsProject object for managing directory
    structure
    - catchment_name: string name of the catchment the flow layers are
    required for

    Returns:
    - data for flow direction
    - meta for flow direction
    - data for flow accumulation
    - meta for flow accumulation
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # See if the flow direction raster is already saved. If so, use
    #that:
    try_flowdir_path = project.catchment_path(
        catchment_name, 'Topography', f'{FLOW_DIRECTION_FN}.tif'
        )
    try:
        flow_dir_array, flow_dir_meta = read_raster(try_flowdir_path)
        logger.info(
            'Existing flow direction raster found at '
            f'{try_flowdir_path}. Reading this in instead of computing '
            'new raster.'
            )
        flow_dir_data = topography.rio_to_pysheds(
            flow_dir_array,
            flow_dir_meta,
            try_flowdir_path
            )

    # If we don't already have a flow accumulation raster, compute one:
    except FileNotFoundError:
        flow_dir_data, flow_dir_meta, grid = topography.compute_flow_dir(
            hydro_dem,
            dem_meta,
            grid,
            dirmap,
            project,
            catchment_name
            )

    # See if the flow accumulation reaster is already saved. If so, use
    #that:
    try_flowacc_path = project.catchment_path(
        catchment_name, 'Topography', f'{FLOW_ACCUMULATION_FN}.tif'
        )
    try:
        flow_acc_array, flow_acc_meta = read_raster(try_flowacc_path)
        logger.info(
            'Existing flow accumulation raster found at '
            f'{try_flowacc_path}. Reading this in instead of computing '
            'new raster.'
            )
        flow_acc_data = topography.rio_to_pysheds(
            flow_acc_array,
            flow_acc_meta,
            try_flowacc_path
            )

    # Otherwise, make a new one:
    except FileNotFoundError:
        flow_acc_data, flow_acc_meta, _ = topography.compute_flow_accum(
            flow_dir_data,
            flow_dir_meta,
            grid,
            dirmap,
            project,
            catchment_name
            )

    return flow_dir_data, flow_dir_meta, flow_acc_data, flow_acc_meta

###############################################################################
def get_clay_fraction(
    proj: FireImpactsProject,
    catchment:str,
    depth:str,
    transform,
    crs,
    shape
    ):
    """
    Read clay percentage rasters for a certain depth range and convert
    them to a fraction

    Parameters:
    - proj: FireImpactsProject object to handle directories
    - catchment: string name of the catchment being worked with
    - depth: depth range the desired soil datset applies for, expressed
    as a string in the form 'xxx_yyy' where xxx is the start
    (top/shallow) of the range and yyy is the end (bottom/deep). In cm.
    - transform: affine transformation for the raster as expected by
    rasterio
    crs: crs object for the raster as expected by rasterio
    - shape: shape of the data expressed as width * height in terms of
    number of cells.
    --------------------------------------------------------------------
    Notes:
    - unique_file_matching assumes certain naming conventions for clay
    files and will throw an error if there's duplicates that match the
    criteria
    --------------------------------------------------------------------
    """
    # Get the location of clay soil datasets within the project:
    clay_directory = proj.catchment_path(catchment, 'Soils','CLY')
    # Get the files with the relevant soil measure, depth and 'EV' in
    #the file name:
    file_name = unique_file_matching(clay_directory,'CLY',depth,'EV')
    # Append to the directory
    file_path = os.path.join(clay_directory, file_name)
    # Get a version of the raster in the same CRS and convert it from
    #percentage to a fraction:
    return read_aligned(file_path,transform,crs,shape)*PERCENT_TO_FRACTION

###############################################################################
def prep_debris_flow_simulation(
    proj: FireImpactsProject,
    catchment:str
    ):
    """

    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    id_field = proj.headwater_id
    # Get the DEM and its metadata:
    dem_path = proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    dem_data, dem_meta = read_raster(dem_path)

    # Get a hydrologically-enforced DEM and pysheds grid object for
    #computing hydro layers:
    hydro_dem, grid_obj = topography.hydro_force_dem(dem_path)

    # Get a hydro slope layer from the hydro DEM:
    slope_h_ratio, slope_h_meta = topography.dem_to_slope(
        proj,
        (dem_data, dem_meta),
        catchment,
        gradient=True,
        hydro=True,
        save=False
        )

    # Get the relevant flow layers as a tuple:
    flw_lyr_tuple = get_flow_layers(
        hydro_dem,
        dem_meta,
        grid_obj,
        D8_FLOW_DIRECTIONS,
        proj,
        catchment
        )
    # Unpack the tuple:
    flow_dir_data, flow_dir_meta, flow_acc_data, flow_acc_meta = flw_lyr_tuple
    # Get important objects for matching rasters:
    transform = flow_acc_meta['transform']
    crs = flow_acc_meta['crs']
    shape = slope_h_ratio.shape
    # Get the 0-5cm and 5-16cm clay values as a fraction, with values
    #masked where there is no data in the hydrolgically-enforced
    #slope raster:
    clay0_5, clay5_15 = [
        np.where(
            np.isnan(slope_h_ratio),
            np.nan,
            get_clay_fraction(
                proj,
                catchment,
                depth,
                transform,
                crs,
                shape
                )
            ) for depth in ['000_005', '005_015']
        ]

    # From the data folder (in the same directory as fire_impacts
    #and test_data), get the HF lookup csv file which contains the
    #thresholds of 12-minute rainfall intensity at which debris flow
    #would occur, for a large range of combinations of Aridity Index,
    #dNBR, years since fire, and slope values:
    hf_lookup = load_package_data('HFlookup_b30pt27.csv')
    # Round the critical mean values to one decimal place:
    hf_lookup['I12_crit_mean'] = hf_lookup['I12_crit_mean'].round(1)

    debris_lookup = load_package_data('debris-constituents.csv')


    # Make a path for DebrisFlow outputs and create the directory if it
    #doesn't already exist:
    out_path = proj.catchment_path(catchment, 'DebrisFlow')
    os.makedirs(out_path, exist_ok=True)

    # Go get  the soil, slope, and aridity data required, assuming it's
    #already been saved:
    condition_data = pd.read_csv(
        proj.catchment_path(
            catchment,
            'Soil_Slope_Aridity_dNBR_headwaters.csv'
            )
        )
    # Replace any missing values with zero:
    condition_data = condition_data.fillna(0.0) # TODO Check - is this correct? Example is getting nulls in dNBR columns

    # Get the headwaters table, assuming it's already been generated:
    topo_data = pd.read_csv(
        proj.catchment_path(catchment,'Topography','Headwaters.csv')
        )
    # Join condition and topographic data together:
    fire_impact_data = pd.merge(
        condition_data, topo_data, on=id_field,how='outer'
        )

    return debris_flow_load(
        dem_data,
        slope_h_ratio,
        transform,
        flow_acc_data,
        flow_dir_data,
        clay0_5,clay5_15,
        out_path,
        fire_impact_data,
        hf_lookup,
        debris_lookup,
        dem_meta,
        id_field
        )

###############################################################################
def net_erosion(
    threshold_met:np.ndarray,
    ae:float,
    be:float,
    ad:float,
    bd:float,
    rock:float,
    clay_fraction:np.ndarray,
    flow_area:np.ndarray,
    gradient_arr:np.ndarray,
    pixel_area:float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """


    Parameters:
    -

    Returns:
    - Arrays of total mass eroded for each pixel:
        - Total
        - Clay
        - Sediment
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    e0 = np.where(threshold_met, flow_area, 0)  # Erosion
    e = ae * (gradient_arr * e0) ** be  # erosion depth (meter)
    d = ad * (gradient_arr * e0) ** bd  # deposition depth (meter)
    e_net = e - d  # net erosion depth (meter)
    e_net_vol = e_net * pixel_area  # erosion volume (m3)
    e_sediment_mass = e_net_vol * (1 - rock) * SEDIMENT_BULK_DENSITY  # sediment only (m3)
    e_rock_mass = e_net_vol * rock * ROCK_BULK_DENSITY  # rock only (m3)
    e_net_mass = e_sediment_mass + e_rock_mass  # net erosion mass (kg)
    e_clay_mass = e_sediment_mass * clay_fraction
    return e_net_mass, e_clay_mass, e_sediment_mass

###############################################################################
def accumulate_erosion(
    erosion_values:np.ndarray,
    rio_meta:dict,
    flow_dir_raster:ArrayLike,
    catchment_mask:np.ndarray,
    save_path:str,
    save:bool=True
    ) -> ArrayLike:
    """
    Create an erosion accumulation raster in the same way a flow
    accumulation raster is created, and set missing values that are
    still in the catchment to 0:
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    fn = 'tmp_erosion.tif'
    try:
        # Temporarily save a copy of the erosion raster so we can use
        #the pysheds Grid.from_raster and read_raster:
        with rasterio.open(fn, 'w', **rio_meta) as dest:
            dest.write(erosion_values.astype('float32'), 1)
        # Create the grid objcet, then the Pysheds Raster version of
        #the input erosion raster:
        grid = Grid.from_raster(fn)
        e_raster = grid.read_raster(fn)
        #Create and return a flow accumulation raster weighted by
        #erosion values:
        accum_raster = grid.accumulation(
            fdir=flow_dir_raster,
            weights=e_raster
            )
        # Set NaN values inside the catchment to 0:
        accum_raster[np.isnan(accum_raster) & catchment_mask] = 0
        # Save the raster to file and then return it:
        if save:
            with rasterio.open(save_path, 'w', **rio_meta) as dest:
                dest.write(accum_raster.astype(rasterio.float32), 1)
                logger.info(
                    f'Saved cumulative erosion raster to {save_path}'
                    )
        return accum_raster
    # Remove the current instance of the non-accumulated erosion raster:
    finally:
        if os.path.exists(fn):
            os.remove(fn)

###############################################################################
def create_cum_erosion_layers(
    erosion_mass_all:np.ndarray,
    erosion_mass_clay:np.ndarray,
    erosion_mass_sediment:np.ndarray,
    flow_dir_raster:ArrayLike,
    out_path:str,
    rio_meta:dict,
    catchment_mask:np.ndarray
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Take basic per pixel erosion layers and convert to cumulative
    erosion for the catchment
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Define output file paths for cumulative erosion and clay
    E_all_cum_path = os.path.join(out_path, f"{ERO_CUM_M_ALL_FN}.tif")
    E_clay_cum_path = os.path.join(out_path, f"{ERO_CUM_M_CLY_FN}.tif")
    Sediment_mass_cum_path = os.path.join(out_path, f"{ERO_CUM_M_SED_FN}.tif")


    # Build cumulative erosion rasters for total, clay, and sediment
    #erosion:
    e_all_accum = accumulate_erosion(
        erosion_mass_all,
        rio_meta,
        flow_dir_raster,
        catchment_mask,
        E_all_cum_path
        )
    e_clay_accum = accumulate_erosion(
        erosion_mass_clay,
        rio_meta,
        flow_dir_raster,
        catchment_mask,
        E_clay_cum_path
        )
    sediment_mass_accum = accumulate_erosion(
        erosion_mass_sediment,
        rio_meta,
        flow_dir_raster,
        catchment_mask,
        Sediment_mass_cum_path
        )

    return e_all_accum, e_clay_accum, sediment_mass_accum

###############################################################################
def create_erosion_sense_check(
    accum_erosion:ArrayLike,
    flow_acc_area:np.ndarray,
    rio_meta:dict,
    save_loc:str,
    erosion_type:str='all',
    ) -> np.ndarray:
    """
    Save a geotiff of total accumulated erosion per total accumulated
    flow area in hectares
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Mass per heactare as a check on accurqcy
    flow_acc_area_ha= flow_acc_area * M2_TO_HA
    # get the array data from raster data
    acc_ero_data=np.array(accum_erosion, dtype=np.float32)
    # Convert any negative values to 0:
    acc_ero_data=np.where(acc_ero_data < 0, 0, acc_ero_data)
    # Create a sense-checking layer of accumulated erosion vs.
    #accumulated flow area in hectares:
    with np.errstate(divide='ignore', invalid='ignore'):
        E_all_mass_ha = np.divide(acc_ero_data, flow_acc_area_ha)

    E_all_mass_ha_path = os.path.join(
        save_loc, f"Erosion_{erosion_type}_mass_per_ha.tif"
        )
    with rasterio.open(E_all_mass_ha_path, 'w', **rio_meta) as dest:
        dest.write(E_all_mass_ha.astype('float32'), 1)

    return E_all_mass_ha

###############################################################################
def compute_net_erosion(
    flow_acc_area:np.ndarray,
    clay_frac_0_05:np.ndarray,
    clay_frac_05_15:np.ndarray,
    gradient_arr:np.ndarray,
    pixel_area:float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net erosion layers for hillslope and channel erosion
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Get hillslope erosion mass per pixel, for normal hillslope
    #erosion:
    er_mass_hs_total, er_mass_hs_clay, er_mass_hs_sediment = net_erosion(
        threshold_met=flow_acc_area<=HILLSLOPE_AREA,
        **HILLSLOPE_PARAMETERS,
        clay_fraction=clay_frac_0_05,
        flow_area=flow_acc_area,
        gradient_arr=gradient_arr,
        pixel_area=pixel_area
        )
    # Get mass for channelised flow:
    er_mass_ch_total, er_mass_ch_clay, er_mass_ch_sediment = net_erosion(
        threshold_met = (
            (flow_acc_area > HILLSLOPE_AREA) &
            (flow_acc_area <= CHANNELISED_FLOW_THRESHOLD)
            ),
        **CHANNEL_PARAMETERS,
        clay_fraction=clay_frac_05_15,
        flow_area=flow_acc_area,
        gradient_arr=gradient_arr,
        pixel_area=pixel_area
        )

    # Array arithmetic to produce the base erosion layers:
    erosion_mass_all = er_mass_hs_total + er_mass_ch_total
    erosion_mass_clay = er_mass_hs_clay + er_mass_ch_clay
    Sediment_mass = er_mass_hs_sediment + er_mass_ch_sediment

    return erosion_mass_all, erosion_mass_clay, Sediment_mass

###############################################################################
def get_debris_volume(
    x:float,
    y:float,
    transform:Affine,
    debris_volume_array:ArrayLike
    ):
    """
    Get the value of an array based on x- and y-coordinates and the
    appropriate affine transform
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Get the row and column in the array corresponding to the X, Y coordinates
    row, col = rowcol(transform, x, y)
    try:
        # Return the debris volume at the calculated row, col position
        return debris_volume_array[row, col]
    except IndexError:
        return np.nan  # return NaN if out of bounds

###############################################################################
def debris_column_values(
    fire_impact_data:pd.DataFrame,
    transform:Affine,
    e_all_accum:ArrayLike,
    e_clay_accum:ArrayLike,
    e_sed_accum:ArrayLike,
    E_all_mass_ha: np.ndarray
    ) -> None:
    """
    Get values for each of the four main debris load accumulations at
    each headwater end point
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Create bespoke iterable so we don't have to retype next code:
    iter_this = [
        (e_clay_accum, CLY_M_ACC_KG),
        (e_all_accum, TOT_EM_ACC_KG),
        (E_all_mass_ha, TOT_EM_ACC_KG_HA),
        (e_sed_accum, SED_M_ACC_KG)
        ]

    # For each of the required erosion layers, go through each
    #headwater endpoint and get that layer's value at that point:
    for array, field_name in iter_this:
        fire_impact_data[field_name] = fire_impact_data.apply(
            lambda row: get_debris_volume(
                x=row[HW_ENDP_X],
                y=row[HW_ENDP_Y],
                transform=transform,
                debris_volume_array=array
                ), # type: ignore[arg-type]
            axis=1
            )
    return None

###############################################################################
def calc_debris_constituent_cols(
    fire_impact_data:pd.DataFrame,
    debris_flow_constituents:pd.DataFrame
    ) -> None:
    """
    Estimates mass of elemental constituents of debris load via the
    debris-constituents.csv lookup table, and populates new columns
    in-place with those estimates.
    --------------------------------------------------------------------
    Notes:
    - Proportion of each element is estimated by the average milligrams
    per kilogram in the lookup table. This is then converted to
    kilograms per kilogram, and multiplied by the total kg of sediment.
    --------------------------------------------------------------------
    """
    # Go through each consituent element in the file:
    for _, row in debris_flow_constituents.iterrows():
        particulate = row[PCLE_CTUENT_NAME]
        Average_Amount = row[AVG_CTUENT_MGPKG]

        # Define new column name
        column_name = f"{particulate} (Kg)"
        fire_impact_data[column_name] = (
            fire_impact_data[SED_M_ACC_KG]
            * (Average_Amount*MILLIGRAMS_TO_KILOGRAMS))

    return None

###############################################################################
def calc_I12_crit_columns(
    fire_impact_data:pd.DataFrame,
    hf_lookup:pd.DataFrame,
    hw_id_field:str
    ) -> pd.DataFrame:
    """
    Use the hf_lookup table to calculate 12-minute rainfall intensity
    thresholds for each headwater, for the first and second year
    post-fire
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Step 4: Read HFlookup_i12 data and merge it with fire_impact_data
    #and debris flow data:
    # adjust (round up mean aridity to nesrest 0.25 to match it with AI
    #in HFlookup_I12 data)
    fire_impact_data[ARID_MEAN_ADJ] = np.ceil(
        fire_impact_data[ARID_MEAN].round(2) / 0.25
        ) * 0.25
    # For dNBR we will ROUND it to the NEAREST 100:
    fire_impact_data[DNBR_MEAN_ADJ] = (
        fire_impact_data[DNBR_MEAN]
        ).round(-2).astype("int64")
    # The slope values in hf_lookup are in tenths of 100 degrees i.e.
    #a slope value of 26 degrees will be 0.3 in the lookup, and 90
    # degrees will be 0.9
    fire_impact_data[SLOPE_DEG_MEAN_ADJ] = (
        fire_impact_data[SLOPE_DEG_MEAN] / 100
        ).round(1)

    join_keys_in_lookup = [
        HF_ARID_IDX_THRESH, HF_DNBR_THRESH, HF_GRADIENT_THRESH
        ]
    join_keys_in_data = [
        ARID_MEAN_ADJ, DNBR_MEAN_ADJ, SLOPE_DEG_MEAN_ADJ
        ]
    
    # Split HFlookup_I12 into two subsets based on 'years' and merge 
    #fire_impact_data with each subset:
    HFlookup_year_1 = hf_lookup[hf_lookup["years"] < 1]
    merged_year_1 = pd.merge(
        fire_impact_data,
        HFlookup_year_1,
        left_on=join_keys_in_data,
        right_on=join_keys_in_lookup,
        how="left"
    ).rename(columns={
        HF_YEARS_THRESH: "TSF_Year_1",
        HF_I12_CRIT: I12_CRIT_Y + '1'
        })

    HFlookup_year_2 = hf_lookup[hf_lookup["years"] >= 1]
    merged_year_2 = pd.merge(
        fire_impact_data,
        HFlookup_year_2,
        left_on=join_keys_in_data,
        right_on=join_keys_in_lookup,
        how="left"
    ).rename(columns={
        HF_YEARS_THRESH: "TSF_Year_2",
        HF_I12_CRIT: I12_CRIT_Y + '2'
        })

    # Combine the results
    fire_impact_data = pd.merge(
        merged_year_1,
        merged_year_2[[hw_id_field,"TSF_Year_2", I12_CRIT_Y + '2']],
        on=[hw_id_field],
        how="left"
    ).drop(columns=join_keys_in_lookup, errors="ignore")

    return fire_impact_data


###############################################################################
def debris_flow_load(
    dem_data,
    slope_ratio,
    slope_transform,
    flow_accumulation,
    flowdir,
    clay0_5_fraction,
    clay5_15_fraction,
    out_path,
    fire_impact_data:pd.DataFrame,
    hf_lookup:pd.DataFrame,
    debris_flow_constituents:pd.DataFrame,
    raster_meta,
    id_field:str
    ):
    """
    Function to calculate debris flow load for each pixel and integrate
    fire impact analysis.

    Parameters:
    - slope_ratio (array): Slope ratio
    - clay0_5_path (array): Path to the 0-5cm clay fraction data
    (raster).
    - clay5_15_path (array): Path to the 5-15cm clay fraction data
    (raster).
    - out_path (str): Path to the folder where outputs will be saved.
    - fire_impact_data_path (str): Path to the fire impact CSV file.
    - hflookup_i12_path (pd.DataFrame): Path to the HF lookup file.

    Returns:
    tuple: Cumulative erosion and clay data arrays, erosion mass per
    hectare array, updated Fire Impact DataFrame.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """

    # Step 1: calculate debris flow

    # get pixel area
    xres = slope_transform[0]  # Pixel width (east-west resolution)
    yres = abs(slope_transform[4])   # Pixel height (north-south resolution)
    pixel_area = xres * yres
    # Multiply the flow accumulation by the resolution to get the area:
    flow_acc_area = flow_accumulation * pixel_area  # This is area in meter square
    # Create a catchment mask based on non-NaN values in the array
    #(assumes non-NaN is within catchment):
    catchment_mask = ~np.isnan(dem_data)
    # Make sure the dtype for output rasters is rasterio float 32:
    inter_meta = raster_meta.copy()
    inter_meta.update(dtype=rasterio.float32)

    # Get net erosion layers for clay only, sediment, and total:
    erosion_mass_all, erosion_mass_clay, Sediment_mass = compute_net_erosion(
        flow_acc_area,
        clay0_5_fraction,
        clay5_15_fraction,
        slope_ratio,
        pixel_area
        )

    # Get cumulative erosion layers for each type:
    e_all_accum, e_clay_accum, e_sed_accum = create_cum_erosion_layers(
        erosion_mass_all,
        erosion_mass_clay,
        Sediment_mass,
        flowdir,
        out_path,
        inter_meta,
        catchment_mask
        )

    # Save a GeoTIFF for sense-checking the erosion results:
    E_all_mass_ha = create_erosion_sense_check(
        e_all_accum, flow_acc_area, inter_meta, out_path
        )

    # Populate columns in the fire_impacts_data table with the relevant
    #erosion values for each headwater:
    debris_column_values(
        fire_impact_data,
        slope_transform,
        e_all_accum=e_all_accum,
        e_clay_accum=e_clay_accum,
        e_sed_accum=e_sed_accum,
        E_all_mass_ha=E_all_mass_ha
        )

    # Populate columns with estimates of the mass of each element
    #present in the debris:
    calc_debris_constituent_cols(fire_impact_data, debris_flow_constituents)

    # For each headwater, calculate the 12-minute rainfall intensity
    #threshold at which a debris flow would occur:
    updated_data = calc_I12_crit_columns(fire_impact_data, hf_lookup, id_field)

    # Return the populated dataframe:
    return updated_data


###############################################################################
def allocate_headwaters_to_subcatchments(
    proj: FireImpactsProject,
    catchment: str,
    area_fraction_threshold: float = 0.1
    ) -> pd.DataFrame:
    """
    Allocate headwaters to subcatchments based on spatial overlay.

    Each headwater is assigned to the subcatchment it is mostly contained within.
    Small misalignments in boundaries are handled by setting a minimum area
    fraction threshold.

    Parameters:
    -----------
    proj : FireImpactsProject
        The FireImpactsProject instance.
    catchment : str
        The catchment identifier.
    area_fraction_threshold : float, optional
        Minimum fraction of headwater area that must overlap with a subcatchment
        for allocation. Default is 0.1 (10%).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: SiteID, hw_ID, Area_m2, area_intersect,
        area_fraction. Each row represents a headwater-to-subcatchment allocation.

    Notes:
    ------
    This function performs a spatial overlay of headwaters and subcatchments,
    then selects the largest intersection for each headwater. It filters out
    allocations below the specified area_fraction_threshold to handle minor
    boundary misalignments.
    """
    headwaters = proj.get_headwaters(catchment)
    subcatchments = proj.get_subcatchments(catchment)

    # Perform spatial overlay
    hw_intersection = gpd.overlay(headwaters, subcatchments)
    hw_intersection['area_intersect'] = hw_intersection.geometry.area
    hw_intersection['area_fraction'] = hw_intersection['area_intersect'] / hw_intersection['Area_m2']

    # Keep only the largest intersection for each headwater (most contained within)
    hw_allocations = (
        hw_intersection
        .sort_values('area_fraction', ascending=False)
        .drop_duplicates(subset=['hw_ID'], keep='first')
    )

    # Filter by area fraction threshold
    hw_allocations = hw_allocations[hw_allocations['area_fraction'] > area_fraction_threshold]

    # Select and return relevant columns
    result = hw_allocations[[SC_ID, HW_ID, 'Area_m2', 'area_intersect', 'area_fraction']].copy()

    logger.info(
        f'Allocated {len(result)} headwaters to subcatchments in {catchment}. '
        f'Filtered {len(hw_allocations) - len(result)} allocations below {area_fraction_threshold} threshold.'
    )

    return result


###############################################################################
def scale_debris_timeseries_by_allocation(
    debris_timeseries: pd.DataFrame,
    hw_allocations: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Scale debris flow timeseries by headwater allocation fractions.

    Multiplies each headwater's sediment load timeseries by its area fraction
    to account for partial allocation to subcatchments.

    Parameters:
    -----------
    debris_timeseries : pd.DataFrame
        DataFrame with headwater IDs as columns and datetime index.
        Values are sediment loads (kg).
    hw_allocations : pd.DataFrame
        DataFrame from allocate_headwaters_to_subcatchments() with area fractions.

    Returns:
    --------
    pd.DataFrame
        Scaled timeseries with same shape as input, with values multiplied by
        allocation fractions. Includes only headwaters present in hw_allocations.
    """
    # Extract fractions, limiting to 1.0 (some may be slightly > 1.0 due to rounding)
    fractions = np.minimum(1.0, hw_allocations.set_index(HW_ID)['area_fraction'])

    # Keep only headwaters that have allocations
    scaled = debris_timeseries.copy()
    scaled = scaled[[col for col in scaled.columns if col in fractions.index]]

    # Multiply by fractions
    scaled = scaled * fractions

    logger.info(
        f'Scaled debris timeseries for {len(fractions)} headwaters. '
        f'Data shape: {scaled.shape}'
    )

    return scaled


###############################################################################
def aggregate_debris_to_subcatchments(
    debris_timeseries: pd.DataFrame,
    hw_allocations: pd.DataFrame,
    time_resolution: str = '12min'
    ) -> pd.DataFrame:
    """
    Aggregate debris flow results from headwaters to subcatchments.

    Post-processes debris flow simulation results by:
    1. Scaling headwater timeseries by allocation fractions
    2. Mapping headwater IDs to subcatchment IDs (SiteID)
    3. Summing across headwaters to get timeseries per subcatchment

    Parameters:
    -----------
    debris_timeseries : pd.DataFrame
        DataFrame with headwater IDs as columns and datetime index (12 minute resolution).
        Values are sediment loads (kg).
    hw_allocations : pd.DataFrame
        DataFrame from allocate_headwaters_to_subcatchments() with area fractions.
    time_resolution : str, optional
        Original time resolution of the debris timeseries. Used for logging.
        Default is '12min'.

    Returns:
    --------
    pd.DataFrame
        Aggregated timeseries with subcatchment SiteIDs as columns and datetime index.
        Values are total sediment loads per subcatchment (kg).

    Notes:
    ------
    The returned DataFrame maintains the original 12-minute resolution.
    Use resample() to aggregate to hourly, daily, or other resolutions.

    Example:
    --------
    >>> # Get 12-minute aggregated results
    >>> df_by_sc = aggregate_debris_to_subcatchments(debris_ts, hw_alloc)
    >>>
    >>> # Resample to hourly
    >>> df_hourly = df_by_sc.resample('H').sum()
    >>>
    >>> # Resample to daily
    >>> df_daily = df_by_sc.resample('D').sum()
    """
    # Scale timeseries by allocation fractions
    scaled = scale_debris_timeseries_by_allocation(
        debris_timeseries,
        hw_allocations
        )

    # Create mapping from hw_ID to SiteID
    sc_map = hw_allocations.set_index(HW_ID)[SC_ID].to_dict()

    # Rename columns from hw_ID to SiteID
    scaled = scaled.rename(columns=sc_map)

    # Group by subcatchment and sum
    aggregated = scaled.groupby(level=0, axis=1).sum()

    logger.info(
        f'Aggregated debris timeseries from {len(scaled.columns)} headwaters '
        f'(at {time_resolution} resolution) to {len(aggregated.columns)} subcatchments.'
    )

    return aggregated


###############################################################################
def resample_debris_timeseries(
    debris_timeseries: pd.DataFrame,
    freq: str = 'H'
    ) -> pd.DataFrame:
    """
    Resample debris timeseries to a coarser temporal resolution.

    Parameters:
    -----------
    debris_timeseries : pd.DataFrame
        DataFrame with datetime index and subcatchment columns.
        Values are sediment loads (kg).
    freq : str, optional
        Resampling frequency. Options: 'H' (hourly), 'D' (daily), 'W' (weekly),
        'MS' (month start), 'YS' (year start), etc. Default is 'H' (hourly).

    Returns:
    --------
    pd.DataFrame
        Resampled timeseries with aggregated (summed) values at the new resolution.

    Example:
    --------
    >>> df_hourly = resample_debris_timeseries(df_12min, 'H')
    >>> df_daily = resample_debris_timeseries(df_12min, 'D')
    """
    resampled = debris_timeseries.resample(freq).sum()

    logger.info(
        f'Resampled debris timeseries to {freq} resolution. '
        f'Shape: {debris_timeseries.shape} -> {resampled.shape}'
    )

    return resampled


###############################################################################
def postprocess_debris_flow(
    proj: FireImpactsProject,
    catchment: str,
    debris_timeseries: pd.DataFrame,
    area_fraction_threshold: float = 0.1,
    resample_freq: str = None,
    save: bool = True
    ) -> dict:
    """
    Complete post-processing workflow for debris flow simulation results.

    Aggregates headwater-scaled debris flow results to the user's subcatchment map
    for reporting. This involves:
    1. Spatial overlay of headwater and subcatchment layers
    2. Allocation of each headwater to its containing subcatchment
    3. Scaling debris timeseries by area fractions
    4. Aggregation to subcatchment timeseries
    5. Optional resampling to coarser temporal resolution

    Parameters:
    -----------
    proj : FireImpactsProject
        The FireImpactsProject instance.
    catchment : str
        The catchment identifier.
    debris_timeseries : pd.DataFrame
        DataFrame with headwater IDs as columns, datetime index, and sediment loads (kg).
        Expected to be at 12-minute resolution from debris_flow() function.
    area_fraction_threshold : float, optional
        Minimum fraction of headwater area for allocation. Default is 0.1.
    resample_freq : str, optional
        Resample to this frequency. None (default) keeps original 12-minute resolution.
        Options: 'H' (hourly), 'D' (daily), etc.
    save : bool, optional
        Whether to save results to CSV files. Default is True.

    Returns:
    --------
    dict
        Dictionary with keys:
        - 'aggregated': Aggregated timeseries by subcatchment at original resolution
        - 'resampled': Resampled timeseries (only if resample_freq is specified)
        - 'allocations': DataFrame showing hw-to-subcatchment allocations

    Notes:
    ------
    Small misalignments in boundary layers are handled by:
    - Performing spatial overlay to find all intersecting areas
    - Computing area fractions for each intersection
    - Keeping only the largest intersection per headwater
    - Filtering out allocations below area_fraction_threshold

    Example:
    --------
    >>> # Run debris flow simulation
    >>> df_results = debris_flow(proj, rainfall_intensity_seq)
    >>> debris_ts = df_results[1]  # Get timeseries component
    >>>
    >>> # Post-process and aggregate to subcatchments
    >>> results = postprocess_debris_flow(
    ...     proj, catchment_name, debris_ts,
    ...     resample_freq='H', save=True
    ... )
    >>>
    >>> # Access aggregated hourly results by subcatchment
    >>> hourly_by_sc = results['resampled']
    """
    # Allocate headwaters to subcatchments
    hw_allocations = allocate_headwaters_to_subcatchments(
        proj, catchment, area_fraction_threshold
    )

    # Aggregate to subcatchments at original resolution
    aggregated = aggregate_debris_to_subcatchments(debris_timeseries, hw_allocations)

    # Prepare output dictionary
    output = {
        'aggregated': aggregated,
        'allocations': hw_allocations
    }

    # Optionally resample
    if resample_freq is not None:
        resampled = resample_debris_timeseries(aggregated, resample_freq)
        output['resampled'] = resampled

    # Save results
    if save:
        out_path = proj.catchment_path(catchment, 'DebrisFlow')
        os.makedirs(out_path, exist_ok=True)

        # Save aggregated timeseries
        agg_file = os.path.join(out_path, 'debris_flow_aggregated_by_subcatchment.csv')
        aggregated.to_csv(agg_file)
        logger.info(f'Saved aggregated debris timeseries to {agg_file}')

        # Save allocations
        alloc_file = os.path.join(out_path, 'headwater_to_subcatchment_allocations.csv')
        hw_allocations.to_csv(alloc_file, index=False)
        logger.info(f'Saved headwater allocations to {alloc_file}')

        # Save resampled if applicable
        if resample_freq is not None:
            resampled_file = os.path.join(
                out_path,
                f'debris_flow_aggregated_by_subcatchment_{resample_freq}.csv'
            )
            output['resampled'].to_csv(resampled_file)
            logger.info(f'Saved resampled ({resample_freq}) debris timeseries to {resampled_file}')

    logger.info(f'Post-processing complete for catchment: {catchment}')

    return output

###############################################################################
def debris_flow(
    proj:FireImpactsProject,
    rainfall,
    catchment:str=None,
    save:bool=True
    ):
    """
    Run debris flow simulation for a given catchment or all catchments
    in the project.

    Parameters:
    - proj (FireImpactsProject): The FireImpactsProject instance.
    - rainfall (pd.Series): A pandas Series containing rainfall
    intensities (mm/hr) with a DateTime index.
    - catchment (str, optional): The catchment to run the simulation
    for. If None, run for all catchments.
    - save (bool, optional): Whether to save the results. Defaults to
    True.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Iterate through simulations and calculate the number of events,
    #rainfall values, and event dates for both Year 1 and Year 2
    if catchment is None:
        return proj.for_each_catchment(lambda c: debris_flow(proj,rainfall,c))

    out_path = proj.catchment_path(catchment, 'DebrisFlow')
    
    # Check to make sure units are as expected:
    if 'units' not in rainfall.attrs:
        logger.warning(
            "Rainfall data has no units attribute, assuming units are "
            "correct (mm/hr)")
    elif rainfall.attrs['units'] != 'mm/h':
        raise ValueError(
            "Rainfall data has units '%s', expected 'mm/h'",
            rainfall.attrs['units']
            )

    # Get a dataframe with all the values, for each headwater, needed 
    #to calculate debris flow:
    working_deb_flow_data = prep_debris_flow_simulation(proj, catchment)

    #------ This code may be superseded by recorders: ------------------
    event_ts = pd.DataFrame(
        data=0, #All debris flows start at 0
        index=rainfall.index, # One row for each timestamp
        columns=working_deb_flow_data[HW_ID] # One column for each HW
        )

    years = range(1, NUM_SIM_YEARS + 1)
    # Initialize a dictionary to store results for each year:
    year_results = {
        year: {
            "event_counts": [], "rainfall_events": [], "event_dates": []
            } for year in years
        }
    t0 = rainfall.index[0]

    # Iterate through each year
    for year in years:
        # Add a year to the date-time stamp for the start of the 
        #rainfall values; 
        t1 = t0 + pd.Timedelta(days=365)
        # Make the field name:
        threshold_col = I12_CRIT_Y + str(year)
        # Get all the rainfall values where the index falls between 
        #that of the start and end of the current year:
        rain_year = rainfall[(rainfall.index >= t0) & (rainfall.index < t1)]
        # Increment the timestamp for the next loop:
        t0 = t1
        # Iterate through each row in fire_impact_data
        for idx, row in working_deb_flow_data.iterrows():
            threshold = row[threshold_col]
            hw_id = row[HW_ID]
            if np.isnan(threshold):  # Skip rows with NaN thresholds
                year_results[year]["event_counts"].append(0)
                year_results[year]["rainfall_events"].append([])
                year_results[year]["event_dates"].append([])
                continue

            # Select rainfall and coordinates of time 
            #(day, subday_12mins) for the current simulation:
            rain_flat = rain_year.values

            # Find events where rainfall exceeds the threshold
            indices = np.where(rain_flat >= threshold)[0]
            events = rain_flat[indices]
            # Get only the rows where there's an event:
            event_dates_row = rain_year.index[indices]

            # Append results for the current year
            year_results[year]["event_counts"].append(len(events))
            year_results[year]["rainfall_events"].append(events.tolist())
            year_results[year]["event_dates"].append(event_dates_row)
            for d in event_dates_row:
                event_ts.at[d, hw_id] += 1

        # Add the number of events as a new column for the current year
        working_deb_flow_data[
            f"Year{year}_num_events"
            ] = year_results[year]["event_counts"]

        # Determine the maximum number of events for this year and simulation
        max_events = max(
            len(ev) for ev in year_results[year]["rainfall_events"]
            )

        # Organize columns for this year and simulation
        sim_columns = {}
        for j in range(max_events):
            # Add rainfall values for event[j]
            sim_columns[f"Year{year}_rainfall_event{j+1}"] = [
                ev[j] if j < len(ev) else np.nan for ev in year_results[year]["rainfall_events"]
            ]
            # Add event dates for event[j]
            sim_columns[f"Year{year}_event{j+1}_date"] = [
                f"{date[j].date().isoformat()}" if j < len(date) else np.nan for date in year_results[year]["event_dates"]
            ]

        # Add all columns for the current year and simulation to the DataFrame
        for col_name, col_values in sim_columns.items():
            working_deb_flow_data[col_name] = col_values
    # Write the outputs as a new dataframe (debris flow)
    Debris_Flow_Data = working_deb_flow_data.copy()

    res_file_name = 'DebrisFlowData.csv'
    if save:
        Debris_Flow_Data_path = os.path.join(out_path, res_file_name)
        Debris_Flow_Data.to_csv(Debris_Flow_Data_path, index=False)
        logger.info(
            'Saved debris flow by headweater results table to '
            f'{Debris_Flow_Data_path}'
            )

    logger.info('Done!')
    return Debris_Flow_Data, event_ts

###############################################################################
def run_debris_flow_sim(
    project:FireImpactsProject,
    rainfall:pd.DataFrame,
    catchment=None,
    recorders=None
    ):
    """
    Run the debris flow simulation for a given project and set of
    rainfall data, recording results as specified.

    Parameters:
    - project (fire_impacts.FireImpactsProject): Current project
    - rainfall (Series-like): 12 minute rainfall intensity data (mm/hr)
    - catchment (str): Name of the catchment to process. If None,
    process all catchments.
    - recorders (dict): OPTIONAL: Dictionary of recorder functions to
    use during the simulation.

    --------------------------------------------------------------------
    Notes:
    - This function handles the catchment and recorder side of things
    --------------------------------------------------------------------
    """
    # Run for all catchments if none was specified:
    if catchment is None:
        return project.for_each_catchment(
            lambda c: run_debris_flow_sim(project,rainfall,c,recorders)
            )
    # Only get rainfall values from the fire's end date onwards:
    fire_end_dt = project.get_fire_end_date(catchment)
    rainfall_trimmed = rainfall.loc[fire_end_dt:]

    # Check if timeseries covers a full 2 years since fire; raise error 
    #if not:
    

    # If no recorders were passed, use an empty dictionary so the rest
    #of the code works consistently:
    if recorders is None:
        recorders = dict()
    # Reset each recorder so we're building new arrays for aggregation:
    for recorder in recorders.values():
        recorder.reset()

###############################################################################
def post_debris_flow_mass_adjustment(
    debris_flow_data:pd.DataFrame,
    ids_with_events:list[str],
    event_year:int,
    mass_col:str=CLY_M_ACC_KG
    ):
    """
    Placeholder function for adjusting the mass available for 
    subsequent debris flows in the same headwater after an event
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    return debris_flow_data

###############################################################################
def generate_debris_flow(
    rainfall:pd.Series,
    debris_flow_data:pd.DataFrame,
    id_field:str,
    out_path:str
    ):
    """
    Generator function that produces an iterable of (timestep, dict) 
    tuples for each timestep entry the provided rainfall data. Dict is 
    a lookup of all relevant debris flow results for that timestep.

    Parameters:
    - rainfall: Series of rainfall values which MUST be intensity in 
    mm/hr recorded at 12-minute intervals
        - So I think it matters whether the first timestamp in the 
        rainfall data can be considered to be immediately after the 
        fire ends. Do we ask for a fire end date? That might be key.
    - debris_flow_data: Dataframe of debris flow input variables by 
    headwater, as created by prep_debris_flow_simulation()

    Yields:
    - tuple of (timestep, result) where:
        - timestep is the datetime stamp for the current 12-minute 
        interval
        - result is a dictionary with the following values:
            - total rain (converted from intensity to depth)
            - number of events year 1
            - mass of debris from events year 1 (tonnes)
            - number of events year 2
            - mass of debris from events year 2 (tonnes)
    --------------------------------------------------------------------
    Notes:
    - Currently this generator does not check which year the current 
    timestep is part of, so the results contain debris flow values for 
    both years. It's assumed that the function using this generator 
    will know which year to grab.
        - This also means that any reset of debris flow mass as a 
        result of an event should be handled by the function using 
        this generator? Or maybe it has to be inside here....
    - The function running this generator should subset the headwaters 
    by subcatchment if that's required, before running.
    TODO: 
    - We also need to engineer this to accept spatially varying 
    rainfall at some point, but this can probably be processed before 
    this function by headwater so a relatively minor adjustment.
    --------------------------------------------------------------------
    """
    # Convert to Series if we've got a dataframe, to ensure consistency:
    if isinstance(rainfall, pd.DataFrame):
        rainfall = pd.Series(data=rainfall['rainfall'], index=rainfall.index)

    working_copy = debris_flow_data.copy()

    year_1_thresh_col = I12_CRIT_Y + '1'
    year_2_thresh_col = I12_CRIT_Y + '2'
    # Get a smaller subset of the debris flow data - just what we need:
    subset = working_copy[[
        id_field,
        CLY_M_ACC_KG,
        year_1_thresh_col,
        year_2_thresh_col,
        ]]

    # Go through the timesteps
    for timestep in rainfall.index:
        rain_intensity_12min = rainfall[timestep]
        rain_depth_over_12_min = rain_intensity_12min / 5

        # Define the basic structure of the output of this generator:
        result = {
            'total_rain': rain_depth_over_12_min,
            'debris_flow_event_y1': 0,
            'debris_flow_mass_t_y1': 0.0,
            'debris_flow_event_y2': 0,
            'debris_flow_mass_t_y2': 0.0
            }

        # If there's no rain at all, skip and continue:
        if rain_intensity_12min == 0:
            yield (timestep, result)
            continue
        
        # Check if the rainfall intensity exceeds the threshold for 
        #any of the headwaters, for either year:
        mask = (
            subset[year_1_thresh_col] < rain_intensity_12min
            | subset[year_2_thresh_col] < rain_intensity_12min
            )
        if not mask.any():
            yield (timestep, result)
            continue

        # Now handle what happens if the rain intensity IS greater than 
        #the debris flow threshold for at least one of the years:
        mask_y1 = subset[year_1_thresh_col] < rain_intensity_12min
        if mask_y1.any():
            # The number of debris flow events is the number of rows 
            #where the condition is true i.e. raifnall is greater than 
            #threshold:
            y1_event_count = mask_y1.sum()
            # Get just the rows from the workind df where there's an 
            #event:
            y1_event_deets = subset.loc[mask_y1]
            # Get the sum of mass for all those rows in kg then tonnes:
            y1_mass_kg = np.nansum(y1_event_deets[CLY_M_ACC_KG])
            y1_mass_t = y1_mass_kg * KG_TO_TONNES
            # Update the values in the result dictionary:
            result['debris_flow_event_y1'] = y1_event_count
            result['debris_flow_mass_t_y1'] = y1_mass_t
        # Same for year 2:
        mask_y2 = subset[year_2_thresh_col] < rain_intensity_12min
        if mask_y2.any():
            y2_event_count = mask_y2.sum()
            y2_event_deets = subset.loc[mask_y2]
            y2_mass_kg = np.nansum(y2_event_deets[CLY_M_ACC_KG])
            y2_mass_t = y2_mass_kg * KG_TO_TONNES
            result['debris_flow_event_y2'] = y2_event_count
            result['debris_flow_mass_t_y2'] = y2_mass_t

        yield (timestep, result)

###############################################################################
def record_headwaters_timeseries(
    proj:FireImpactsProject,
    variable_name:str,
    agg_type:str='sum',
    label_field=None,
    agg_count:int=1
    ):
    """
    Build a debris flow recorder that summarises debris flow mass over 
    time for each headwater
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    result = None
    index = None

    intermediate=None
    intermediate_count = 0
    ###########################################################################
    def hw_timeseries_recorder(timestep, catchment, **kwargs):
        """
        Primary closure function for building a timeseries of 
        aggregated values all headwaters in a catchment
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        
        def agg(d):
            """
            Handle the requested aggregation
            """
            if agg_type == 'sum':
                return np.nansum(d)
            elif agg_type == 'mean':
                return np.nanmean(d)
            elif agg_type == 'max':
                return np.nanmax(d)
            else:
                raise ValueError(f'Aggregation: {agg_type} not known')
            
    ###########################################################################
    def reset():
        """
        Secondary closure function to revert all encapsulated variables 
        to starting values.
        """
        pass

    ###########################################################################
    def finalise():
        """
        Secondary closure function to give the final results
        """
        pass
    
    # Add the two secondary closure functions as method-like 
    #attachments to the main one
    hw_timeseries_recorder.reset = reset
    hw_timeseries_recorder.finalise = finalise
    return hw_timeseries_recorder
