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
        HF_I12_CRIT: "I12_crit_mean_Year_1"
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
        HF_I12_CRIT: "I12_crit_mean_Year_2"
        })

    # Combine the results
    fire_impact_data = pd.merge(
        merged_year_1,
        merged_year_2[[hw_id_field,"TSF_Year_2", "I12_crit_mean_Year_2"]],
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
    
    if 'units' not in rainfall.attrs:
        logger.warning("Rainfall data has no units attribute, assuming units are correct (mm/hr)")
    elif rainfall.attrs['units'] != 'mm/h':
        logger.error("Rainfall data has units '%s', expected 'mm/h'", rainfall.attrs['units'])
        raise ValueError("Rainfall data has units '%s', expected 'mm/h'"%rainfall.attrs['units'])

    NUM_YEARS=2
    result = prep_debris_flow_simulation(proj, catchment)
    event_ts = pd.DataFrame(0,index=rainfall.index, columns=result['hw_ID'])

    years = range(1, NUM_YEARS + 1)
    # for sim in ds_12min['simulation'].values:
        # Initialize a dictionary to store results for each year
    year_results = {year: {"event_counts": [], "rainfall_events": [], "event_dates": []} for year in years}
    t0 = rainfall.index[0]

    # Iterate through each year
    for year in years:
        t1 = t0 + pd.Timedelta(days=365)
        threshold_col = f"I12_crit_mean_Year_{year}"
        rain_year = rainfall[(rainfall.index >= t0) & (rainfall.index < t1)]
        t0 = t1

        # Iterate through each row in fire_impact_data
        for idx, row in result.iterrows():
            threshold = row[threshold_col]
            hw_id = row['hw_ID']
            if np.isnan(threshold):  # Skip rows with NaN thresholds
                year_results[year]["event_counts"].append(0)
                year_results[year]["rainfall_events"].append([])
                year_results[year]["event_dates"].append([])
                continue

            # Select rainfall and coordinates of time (day, subday_12mins) for the current simulation
            rain_flat = rain_year.values

            # Find events where rainfall exceeds the threshold
            indices = np.where(rain_flat >= threshold)[0]
            events = rain_flat[indices]
            # event_dates_row = [(days_flat[i], subdays_flat[i]) for i in indices]
            event_dates_row = rain_year.index[indices]

            # Append results for the current year
            year_results[year]["event_counts"].append(len(events))
            year_results[year]["rainfall_events"].append(events.tolist())
            year_results[year]["event_dates"].append(event_dates_row)
            for d in event_dates_row:
                event_ts.at[d, hw_id] += 1

        # Add the number of events as a new column for the current year
        result[f"Year{year}_num_events"] = year_results[year]["event_counts"]

        # Determine the maximum number of events for this year and simulation
        max_events = max(len(ev) for ev in year_results[year]["rainfall_events"])

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
            result[col_name] = col_values
    # Write the outputs as a new dataframe (debris flow)
    Debris_Flow_Data = result.copy()

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
    rainfall,
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
    # If no recorders were passed, use an empty dictionary so the rest 
    #of the code works consistently:
    if recorders is None:
        recorders = dict()
    # Reset each recorder so we're building new arrays for aggregation:
    for recorder in recorders.values():
        recorder.reset()




###############################################################################
def generate_debris_flow(
    rainfall:pd.Series,
    elevation_arr:ArrayLike,
    gradient_arr:ArrayLike,
    flow_direction_arr:ArrayLike,
    flow_accumulation_arr:ArrayLike,
    clay_frac_arr_0_5:ArrayLike,
    clay_frac_arr_5_15:ArrayLike,
    condition_by_hw:pd.DataFrame,
    debris_thresh_lookup:pd.DataFrame,
    debris_constit_lookup:pd.DataFrame,
    transform,
    rio_meta,
    id_field:str,
    out_path:str
    ):
    """

    Parameters:
    - rainfall: Series of rainfall intensity (mm/hr) values at 
    12-minute intervales
    - elevation_arr: Hydrologically-enforced DEM data
    - gradient_arr: Array of gradient (rise/run) values
    - flow_direction_arr: Array of flow direction integer values (d8)
    - flow_accumulation_arr: Array of flow accumulation values
    - clay_frac_arr_0_5: Array of values giving the fraction of clay in 
    the top 5cm of soil
    - clay_frac_arr_5_15: Array of clay fraction for 5-15cm soil depths
    - condition_by_hw: Dataframe with a row for each headwater 
    identifiable by a hw_ID field (must match id_field parameter), and 
    the following calculated values for each headwater:
        - X_EndP: The x-coordinate of the headwater end/outlet
        - Y_EndP: The y-coordinate of the headwater end/outlet
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
