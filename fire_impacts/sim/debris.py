import xarray as xr
# import geopandas as gpd
# import rioxarray
import pandas as pd
from shapely.geometry import mapping
import numpy as np
# import dask
import rasterio
from rasterio.warp import Resampling
from rasterio.transform import from_origin, Affine, rowcol
from ..const import M2_TO_HA, MILLIGRAMS_TO_KILOGRAMS, PERCENT_TO_FRACTION
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre.util import read_aligned, read_raster
from fire_impacts.util import load_package_data, unique_file_matching
from pysheds.grid import Grid
import os
import logging
logger = logging.getLogger(__name__)

###############################################################################
def get_slope(dem_path):
    """
    
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)  # Read the first band (DEM data)
        dem_meta = src.meta.copy()
        transform = src.transform  # Get the affine transform (coordinates)
        crs = src.crs  # Get the CRS (UTM in your case)
        nodata = src.nodata

    # Calculate the slope
    xres = transform[0]  # Pixel width (east-west resolution)
    yres = abs(transform[4])   # Pixel height (north-south resolution)
    dem_data = np.where(dem_data == nodata, np.nan, dem_data) # mask nodata values
    grid = Grid.from_raster(dem_path)
    dem_grid = grid.read_raster(dem_path)
    fill_dem = grid.fill_pits(dem_grid)
    flooded_dem = grid.fill_depressions(fill_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)  # Calculate flow accumulation
    acc_data=np.array(acc, dtype=np.float32) # get the array data from raster data
    # get slope
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres) # Use numpy.gradient to calculate the change in elevation (dz) in both directions
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2) # Calculate the slope as a ratio (dimensionless number, rise/run)

    # slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) # Calculate the slope as radian
    # slope_degrees = np.degrees(slope_radians) # convert slope to degrees
    # slope_path = os.path.join(out_path, 'slope.tif')
    # with rasterio.open(slope_path, 'w', **dem_meta) as dest:
    #     dest.write(slope_degrees.astype('float32'), 1)

    return slope_ratio, acc_data, fdir, dem_data, transform, crs, dem_meta

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
    
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    clay_directory = proj.catchment_path(catchment, 'Soils','CLY')
    fn = unique_file_matching(clay_directory,'CLY',depth,'EV')
    path_fn = os.path.join(clay_directory, fn)
    return read_aligned(path_fn,transform,crs,shape)*PERCENT_TO_FRACTION

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
    dem_path = proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    slope_ratio, acc_data, flowdir, dem_data, transform, crs, raster_meta = get_slope(dem_path)
    shape = slope_ratio.shape
    clay0_5, clay5_15 = [np.where(np.isnan(slope_ratio),np.nan,get_clay_fraction(proj, catchment, depth,transform,crs,shape)) for depth in ['000_005', '005_015']]

    hf_lookup = load_package_data('HFlookup_b30pt27.csv')
    hf_lookup['I12_crit_mean'] = hf_lookup['I12_crit_mean'].round(1)

    out_path = proj.catchment_path(catchment, 'DebrisFlow')
    os.makedirs(out_path, exist_ok=True)

    condition_data = pd.read_csv(
        proj.catchment_path(
            catchment,
            'Soil_Slope_Aridity_dNBR.csv'
            )
        ) #This seems to have the correct ID column
    
    condition_data = condition_data.fillna(0.0) # TODO Check - is this correct? Example is getting nulls in dNBR columns
    topo_data = pd.read_csv(proj.catchment_path(catchment,'Topography','Headwaters.csv'))
    fire_impact_data = pd.merge(condition_data, topo_data, on=id_field,how='outer')

    return debris_flow_load(dem_data,slope_ratio,transform,acc_data,flowdir,
                            clay0_5,clay5_15,
                            out_path,
                            fire_impact_data,
                            hf_lookup,
                            load_package_data('debris-constituents.csv'),
                            raster_meta,id_field)

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
    slope_ratio (array): Slope ratio
    clay0_5_path (array): Path to the 0-5cm clay fraction data (raster).
    clay5_15_path (array): Path to the 5-15cm clay fraction data (raster).
    out_path (str): Path to the folder where outputs will be saved.
    fire_impact_data_path (str): Path to the fire impact CSV file.
    hflookup_i12_path (pd.DataFrame): Path to the HF lookup file.

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
    # Multiply the flow accumulation by the resolution to get the area
    area = flow_accumulation * pixel_area  # This is area in meter square

    # Define hillslope and domain
    HILLSLOPE_AREA = 1.3e4
    CHANNELISED_FLOW_THRESHOLD = 1.4e7
    SEDIMENT_BULK_DENSITY=1270 # kg/m3
    ROCK_BULK_DENSITY=2220 # kg/m3
    HILLSLOPE_ROCK_FRACTION=0.12
    CHANNEL_ROCK_FRACTION=0.45
    HILLSLOPE_PARAMETERS=dict(
        ae=4.5e-4,be=0.36,
        ad=0.3*4.5e-4,bd=0.36,
        rock=HILLSLOPE_ROCK_FRACTION
    )

    CHANNEL_PARAMETERS=dict(
        ae=4.1e-4,be=0.52,
        ad=3.7e-7,bd=1.06,
        rock=CHANNEL_ROCK_FRACTION
    )

    def net_erosion(threshold_met,ae,be,ad,bd,rock,clay_fraction):
        e0 = np.where(threshold_met, area, 0)  # Erosion
        e = ae * (slope_ratio * e0) ** be  # erosion depth (meter)
        d = ad * (slope_ratio * e0) ** bd  # deposition depth (meter)
        e_net = e - d  # net erosion depth (meter)
        e_net_vol = e_net * pixel_area  # erosion volume (m3)
        e_sediment_mass = e_net_vol * (1 - rock) * SEDIMENT_BULK_DENSITY  # sediment only (m3)
        e_rock_mass = e_net_vol * rock * ROCK_BULK_DENSITY  # rock only (m3)
        e_net_mass = e_sediment_mass + e_rock_mass  # net erosion mass (kg)
        e_clay_mass = e_sediment_mass * clay_fraction
        return e_net_mass, e_clay_mass, e_sediment_mass

    eh_mass, eh_clay, eh_sediment = net_erosion(
        area<=HILLSLOPE_AREA,**HILLSLOPE_PARAMETERS,clay_fraction=clay0_5_fraction)
    ec_mass, ec_clay, ec_sediment = net_erosion(
        (area > HILLSLOPE_AREA) & (area <= CHANNELISED_FLOW_THRESHOLD),
        **CHANNEL_PARAMETERS,clay_fraction=clay5_15_fraction)

    E_all = eh_mass + ec_mass
    E_clay = eh_clay + ec_clay
    Sediment_mass = eh_sediment + ec_sediment

    # E_all_path = os.path.join(out_path, 'E_all.tif')
    # with rasterio.open(E_all_path, 'w', **dem_meta) as dest:
    #     dest.write(E_all.astype('float32'), 1)

    def accumulate_erosion(grid):
        fn = 'tmp_erosion.tif'
        try:
            with rasterio.open(fn, 'w', **raster_meta) as dest:
                dest.write(grid.astype('float32'), 1)
            grid = Grid.from_raster(fn)
            e_raster = grid.read_raster(fn)
            return grid.accumulation(fdir=flowdir, weights=e_raster)
        finally:
            if os.path.exists(fn):
              os.remove(fn)

    e_all_accum = accumulate_erosion(E_all)
    e_clay_accum = accumulate_erosion(E_clay)
    sediment_mass_accum = accumulate_erosion(Sediment_mass)

    # Temporary file paths
    # temp_E_all_path = os.path.join(out_path, 'temp_E_all.tif')
    # temp_E_clay_path = os.path.join(out_path, 'temp_E_clay.tif')
    # temp_sediment_mass_path = os.path.join(out_path, 'temp_sediment_mass.tif')

    # Write E_all and E_clay arrays to temporary raster files
    raster_meta.update(dtype='float32')
    acc_path = os.path.join(out_path, 'acc.tif')
    with rasterio.open(acc_path, 'w', **raster_meta) as dest:
        dest.write(flow_accumulation.astype('float32'), 1)

    # with rasterio.open(temp_E_all_path, 'w', **dem_meta) as dest:
    #     dest.write(E_all.astype('float32'), 1)

    # with rasterio.open(temp_E_clay_path, 'w', **dem_meta) as dest:
    #     dest.write(E_clay.astype('float32'), 1)

    # with rasterio.open(temp_sediment_mass_path, 'w', **dem_meta) as dest:
    #     dest.write(Sediment_mass.astype('float32'), 1)

    # Read the temporary rasters into pysheds as Raster objects (in kg)
    # E_all_raster = grid.read_raster(temp_E_all_path)
    # E_all_raster_path = os.path.join(out_path, 'E_all_raster.tif')
    # with rasterio.open(E_all_raster_path, 'w', **raster_meta) as dest:
    #     dest.write(E_all_raster.astype('float32'), 1)

    # E_clay_raster = grid.read_raster(temp_E_clay_path)
    # Sediment_mass_raster = grid.read_raster(temp_sediment_mass_path)

    # Erosion mass accumulation (in kg)
    # E_all_cum = grid.accumulation(fdir=fdir, weights=E_all_raster)
    # E_clay_cum = grid.accumulation(fdir=fdir, weights=E_clay_raster)
    # Sediment_mass_cum = grid.accumulation(fdir=fdir, weights=Sediment_mass_raster)

    # Define output file paths for cumulative erosion and clay
    E_all_cum_path = os.path.join(out_path, "E_all_cum.tif")
    E_clay_cum_path = os.path.join(out_path, "E_clay_cum.tif")
    Sediment_mass_cum_path = os.path.join(out_path, "Sediment_mass_cum.tif")
    # Create a catchment mask based on non-NaN values in the array (assumes non-NaN is within catchment)
    catchment_mask = ~np.isnan(dem_data)

    # Set NaN values inside the catchment to 0
    e_all_accum[np.isnan(e_all_accum) & catchment_mask] = 0
    # Save the output rasters in kg using rasterio
    with rasterio.open(E_all_cum_path, 'w', **raster_meta) as dest:
        dest.write(e_all_accum.astype('float32'), 1)
    # Set NaN values inside the catchment to 0
    e_clay_accum[np.isnan(e_clay_accum) & catchment_mask] = 0
    with rasterio.open(E_clay_cum_path, 'w', **raster_meta) as dest:
        dest.write(e_clay_accum.astype('float32'), 1)
    # Set NaN values inside the catchment to 0
    sediment_mass_accum[np.isnan(sediment_mass_accum) & catchment_mask] = 0
    # Save the output rasters in kg using rasterio
    with rasterio.open(Sediment_mass_cum_path, 'w', **raster_meta) as dest:
        dest.write(sediment_mass_accum.astype('float32'), 1)

    # Clean up temporary files
    # os.remove(temp_E_all_path)
    # os.remove(temp_E_clay_path)
    # os.remove(temp_sediment_mass_path)

    # Mass per heactare as a ceck on accurqcy
    Area_ha= area * M2_TO_HA  # Area (hectare)
    E_all_cum_data=np.array(e_all_accum, dtype=np.float32) # get the array data from raster data
    E_all_cum_data=np.where(E_all_cum_data < 0, 0, E_all_cum_data)
    with np.errstate(divide='ignore', invalid='ignore'):
        E_all_mass_ha = np.divide(E_all_cum_data, Area_ha)

    E_all_mass_ha_path = os.path.join(out_path, "E_all_mass_per_ha.tif")
    E_all_mass_ha[np.isnan(E_all_mass_ha) & catchment_mask] = 0
    with rasterio.open(E_all_mass_ha_path, 'w', **raster_meta) as dest:
        dest.write(E_all_mass_ha.astype('float32'), 1)
    # ----------------------------------------------------------------------------------------------------------------------------------

    # Define a function to get the debris volume for each point
    def get_debris_volume(x, y, transform, debris_volume_array):
        # Get the row and column in the array corresponding to the X, Y coordinates
        row, col = rowcol(transform, x, y)
        try:
            # Return the debris volume at the calculated row, col position
            return debris_volume_array[row, col]
        except IndexError:
            return np.nan  # return NaN if out of bounds

    # Apply the function to each row in the DataFrame to retrieve debris volume values
    fire_impact_data['Clay mass accumulation (kg)'] = fire_impact_data.apply(
        lambda row: get_debris_volume(row['X_EndP'], row['Y_EndP'], slope_transform, e_clay_accum), axis=1)
    fire_impact_data['Total E-mass accumulation (kg)'] = fire_impact_data.apply(
        lambda row: get_debris_volume(row['X_EndP'], row['Y_EndP'], slope_transform, e_all_accum), axis=1)
    fire_impact_data['Total E-mass accumulation (kg/ha)'] = fire_impact_data.apply(
        lambda row: get_debris_volume(row['X_EndP'], row['Y_EndP'], slope_transform, E_all_mass_ha), axis=1)
    fire_impact_data['Sediment mass accumulation (kg)'] = fire_impact_data.apply(
        lambda row: get_debris_volume(row['X_EndP'], row['Y_EndP'], slope_transform, sediment_mass_accum), axis=1)
    # ----------------------------------------------------------------------------------------------------------------
    # Step 3: Read constituent ratio data and calculate the constituents in tonnes
    # Iterate through each constituent
    for _, row in debris_flow_constituents.iterrows():
        particulate = row['Particulate constituent']
        Average_Amount = row['Average amount (mgkg-1)']

        # Define new column name
        column_name = f"{particulate} (Kg)"
        fire_impact_data[column_name] = (fire_impact_data['Sediment mass accumulation (kg)'] * (Average_Amount*MILLIGRAMS_TO_KILOGRAMS))
    # --------------------------------------------------------------------------------------------------------------------------------------
    # Step 4: Read HFlookup_i12 data and merge it with fire_impact_data and debris flow data
    fire_impact_data["Aridity_mean_adjusted"] = np.ceil(
        fire_impact_data["Aridity_mean"].round(2) / 0.25
        ) * 0.25 # adjust (round up mean aridity to nesrest 0.25 to match it with AI in HFlookup_I12 data)
    fire_impact_data["dNBR_mean_adjusted"] = (((
                fire_impact_data["dNBR_mean"] * 1000 + 50
                ) // 100
            ) * 100
        ).astype("int64")  # adjust (round up mean dNBR to nearest 100 to match it with dNBR in HFlookup_I12 data)
    fire_impact_data["Slope_mean_adjusted"] = (
        fire_impact_data["Slope_mean"] / 100
        ).round(1)  # # adjust (get ratio and round up slope to match it with slope in HFlookup_I12 data)

    # Split HFlookup_I12 into two subsets based on 'years'

    # Merge fire_impact_data with each subset
    HFlookup_year_1 = hf_lookup[hf_lookup["years"] < 1]
    merged_year_1 = pd.merge(
        fire_impact_data,
        HFlookup_year_1,
        left_on=["Aridity_mean_adjusted", "dNBR_mean_adjusted", "Slope_mean_adjusted"],
        right_on=["AI", "dNBR", "slope"],
        how="left"
    ).rename(columns={"years": "TSF_Year_1", "I12_crit_mean": "I12_crit_mean_Year_1"})

    HFlookup_year_2 = hf_lookup[hf_lookup["years"] >= 1]
    merged_year_2 = pd.merge(
        fire_impact_data,
        HFlookup_year_2,
        left_on=["Aridity_mean_adjusted", "dNBR_mean_adjusted", "Slope_mean_adjusted"],
        right_on=["AI", "dNBR", "slope"],
        how="left"
    ).rename(columns={"years": "TSF_Year_2","I12_crit_mean": "I12_crit_mean_Year_2"})

    # Combine the results
    fire_impact_data = pd.merge(
        merged_year_1,
        merged_year_2[[id_field,"TSF_Year_2", "I12_crit_mean_Year_2"]],
        on=[id_field],
        how="left"
    ).drop(columns=["AI", "dNBR", "slope"], errors="ignore")

    return fire_impact_data


# NOTES/QUESTIONS
#
# Debris flow
# * Two years? (But assume starting 1 January, should be any day of the year)
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

###############################################################################
def debris_flow(
    proj:FireImpactsProject,
    rainfall,
    catchment:str=None,
    save:bool=True
    ):
    """
    Run debris flow simulation for a given catchment or all catchments in the project.

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
    # Iterate through simulations and calculate the number of events, rainfall values, and event dates for both Year 1 and Year 2
    
    if catchment is None:
        return proj.for_each_catchment(lambda c: debris_flow(proj,rainfall,c))
    
    out_path = proj.catchment_path(catchment, 'DebrisFlow')
    
    if 'units' not in rainfall.attrs:
        logger.warning("Rainfall data has no units attribute, assuming units are correct (mm/hr)")
    elif rainfall.attrs['units'] != 'mm/h':
        logger.error("Rainfall data has units '%s', expected 'mm/h'", rainfall.attrs['units'])
        raise ValueError("Rainfall data has units '%s', expected 'mm/h'"%rainfall.attrs['units'])

    result = prep_debris_flow_simulation(proj, catchment)

    NUM_YEARS=2
    years = range(1, NUM_YEARS + 1)
    # for sim in ds_12min['simulation'].values:
        # Initialize a dictionary to store results for each year
    year_results = {year: {"event_counts": [], "rainfall_events": [], "event_dates": []} for year in years}

    # Iterate through each year
    for year in years:
        threshold_col = f"I12_crit_mean_Year_{year}"

        # Iterate through each row in fire_impact_data
        for idx, threshold in enumerate(result[threshold_col]):
            if np.isnan(threshold):  # Skip rows with NaN thresholds
                year_results[year]["event_counts"].append(0)
                year_results[year]["rainfall_events"].append([])
                year_results[year]["event_dates"].append([])
                continue

            # Select rainfall and coordinates of time (day, subday_12mins) for the current simulation
            rain_flat = rainfall.values
            # days = ds_12min['day'].values
            # subdays = ds_12min['subday_12mins'].values

            # Flatten the arrays for easy processing
            # rain_flat = rain_values.flatten()
            # days_flat = np.repeat(days, len(subdays))
            # subdays_flat = np.tile(subdays, len(days))

            # Find events where rainfall exceeds the threshold
            indices = np.where(rain_flat >= threshold)[0]
            events = rain_flat[indices]
            # event_dates_row = [(days_flat[i], subdays_flat[i]) for i in indices]
            event_dates_row = rainfall.index[indices]

            # Append results for the current year
            year_results[year]["event_counts"].append(len(events))
            year_results[year]["rainfall_events"].append(events.tolist())
            year_results[year]["event_dates"].append(event_dates_row)
        
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
    return Debris_Flow_Data

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
    - rainfall (Series-like): 30 minute rainfall data (mm)
    - catchment (str): Name of the catchment to process. If None, 
    process all catchments.
    - recorders (dict): OPTIONAL: Dictionary of recorder functions to 
    use during the simulation.

    --------------------------------------------------------------------
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

    ):
    pass