from affine import Affine
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import os
import logging

from fire_impacts.pre.util import read_aligned, read_raster
logger = logging.getLogger(__name__)

from fire_impacts.pre import FireImpactsProject

DNBR_SEVERITY_THRESHOLD = 400
EMPIRICAL_COEFFICIENT = 0.082
M2_TO_HA=1e-4
MG_TO_KG=1e-6

def compute_klscp_layer(proj:FireImpactsProject, catchment:str, support_practice_factor:float=1.0):
    #.....................................................................................................................
    #Step3: calculate a base layer for rusle (this is a part of equastion to make analysis quicker)
    # Read all input rasters and ensure they align
    c_factor_path = proj.catchment_path(catchment,'Erodibility','C_factor_adjusted.tif')
    k_factor_path = proj.catchment_path(catchment,'Erodibility','K_factor_adjusted.tif')
    ls_factor_path = proj.catchment_path(catchment,'Erodibility','LS_factor.tif')
    with rasterio.open(c_factor_path) as c_factor, \
         rasterio.open(k_factor_path) as k_factor, \
         rasterio.open(ls_factor_path) as ls_factor:

        # Read data as arrays
        c_array = c_factor.read(1)
        k_array = k_factor.read(1)
        ls_array = ls_factor.read(1)

        # Copy the profile metadata before exiting the `with` block
        profile = ls_factor.meta.copy() # use a lsyer that nodata values =np.nan

        # Perform the multiplication for RUSLE base layer
        base = c_array * k_array * ls_array
        base = base * support_practice_factor

        # Save the result as a raster layer
        base_raster_path = proj.catchment_path(catchment,'Erodibility','KLSCP.tif')
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        with rasterio.open(base_raster_path, 'w', **profile) as dst:
            dst.write(base, 1)

def _rusle_parameter_grids(project:FireImpactsProject, catchment:str):
    cell_area_m2 = project.cell_area(catchment)
    cell_area_ha = cell_area_m2 * M2_TO_HA  # Convert to hectares

    compute_klscp_layer(project,catchment)
    klscp, transform, crs = read_raster(project.catchment_path(catchment,'Erodibility','KLSCP.tif'))
    sdr,_,_ = read_raster(project.catchment_path(catchment,'Delivery','SDR.tif'))
    dnbr = read_aligned(project.catchment_path(catchment,'FireSeverity','dNBR.tif'),transform,crs,klscp.shape)
    return klscp, sdr, dnbr, cell_area_ha, transform

def lumped_daily_rusle(project:FireImpactsProject, rainfall, catchment=None):
    """
    Calculates RUSLE and SDR erosion values for sub-catchments based on 30min rainfall sequence.

    Parameters:
    - project (fire_impacts.FireImpactsProject): Current project
    - rainfall (Series-like): 30 minute rainfall data
    - catchment (str): Name of the catchment to process. If None, process all catchments.

    Returns:
    - RUSLE_df (DataFrame): DataFrame summarizing RUSLE and sediment delivery results.
    """
    if catchment is None:
        return project.for_each_catchment(lambda c: lumped_daily_rusle(project,rainfall,c))

    RUSLE_df = calculate_lumped_rusle(project.subcatchment_boundaries(catchment), rainfall, *_rusle_parameter_grids(project,catchment))

    logger.info('Done')
    return RUSLE_df

def gridded_total_rusle(project:FireImpactsProject, rainfall, catchment=None):
    if catchment is None:
        return project.for_each_catchment(lambda c: lumped_daily_rusle(project,rainfall,c))
    result = None
    boundary = project.subcatchment_boundaries(catchment).iloc[0].geometry

    total_eroded = None
    total_delivered = None
    params = _rusle_parameter_grids(project,catchment)
    for day_data in generate_rusle_for_feature([boundary], rainfall, *params):
        day, _, _, _, \
        daily_RUSLE, daily_SDR, \
        _, _, _, _ = day_data
        logger.info('Processing day %s',day)
        if total_eroded is None:
            total_eroded = daily_RUSLE
            total_delivered = daily_SDR
        else:
            total_eroded += daily_RUSLE
            total_delivered += daily_SDR

    logger.info('Done')
    return total_eroded, total_delivered, params[-1]

def calculate_lumped_rusle(subcatchments:gpd.GeoDataFrame, rainfall:pd.DataFrame, klscp:np.array, sdr:np.array, dnbr:np.array, cell_area_ha:float, transform:Affine):
    daily_erosion = []
    # Read catchment shapefile
    # Loop through each catchment
    for ind, subcatchment in subcatchments.iterrows():
        logger.info('Processing subcatchment %d',ind+1)
        #catchment_name = catchment['PRIMARY_CA']  # Replace 'PRIMARY_CA' with the appropriate field for catchment names
        geometry = [subcatchment['geometry']]

        for day_data in generate_rusle_for_feature(geometry, rainfall, klscp, sdr, dnbr, cell_area_ha, transform):
            day, daily_total_rain, max_intensity, max_erosivity, \
            daily_RUSLE, daily_SDR, \
            daily_RUSLE_below_threshold, daily_RUSLE_above_threshold, \
            daily_SDR_below_threshold, daily_SDR_above_threshold = day_data

            # Store the daily result in the dictionary
            daily_erosion.append({
                'Sub-catchment': f"Sub_{ind+1}",
                # 'Simulation': sim.item()+1,
                'Day': day,
                'Rainfall (total daily)': daily_total_rain,
                'Max Rain Intensity (30 mins)': max_intensity,
                'Max Erosivity (30 mins)': max_erosivity,
                'RUSLE': np.nansum(daily_RUSLE),
                'RUSLE_SDR': np.nansum(daily_SDR),
                'RUSLE (Low severity)': np.nansum(daily_RUSLE_below_threshold),
                'RUSLE (High severity)': np.nansum(daily_RUSLE_above_threshold),
                'RUSLE_SDR (Low severity)': np.nansum(daily_SDR_below_threshold),
                'RUSLE_SDR (High severity)': np.nansum(daily_SDR_above_threshold),
            })
        """
        # Save the output raster
        rusle_path = os.path.join(output_path, f"{catchment_name}_RUSLE_sim{sim.item()}_{day.item()}.tif")
        with rasterio.open(rusle_path, 'w', **profile) as dst:
            dst.write(daily_RUSLE, 1)

        sdr_rusle_path = os.path.join(output_path, f"{catchment_name}_SDR_RUSLE_sim{sim.item()}_{day.item()}.tif")
        with rasterio.open(sdr_rusle_path, 'w', **profile) as dst:
            dst.write(daily_SDR_RUSLE, 1)
        """
    # Convert the results to a DataFrame for analysis or saving
    RUSLE_df = pd.DataFrame(daily_erosion)
    # Read constituent data and convert it to ratio (its in mg per kg)
    # constituent_df = pd.read_csv(Constituent_path)
    RUSLE_df = compute_particulates(RUSLE_df)

    # Save to a CSV file if needed
    RUSLE_df=RUSLE_df.round(1)
    logger.info('Done')
    return RUSLE_df

def generate_rusle_for_feature(geometry:list, rainfall:pd.DataFrame, klscp:np.array, sdr:np.array, dnbr:np.array, cell_area_ha:float, transform:Affine):
    '''
    Calculates RUSLE and SDR erosion values for sub-catchments based on rainfall simulations.

    Parameters:
    - geometry (list): List of shapely geometries representing the sub-catchment.
    - rainfall (pd.Series like): Series with 30-minute rainfall data.
    - klscp (np.array): Raster layer with KLSCP values.
    - sdr (np.array): Raster layer with SDR values.
    - dnbr (np.array): Raster layer with dNBR values.
    - cell_area_ha (float): Area of each cell in hectares.
    - transform (Affine): Affine transformation for the raster layers.

    klscp, sdr and dnbr should have the same shape and transformation.

    Yield:
    - Tuple with daily results for the sub-catchment:
      day (datetime), daily_total_rain (float), max_intensity (float), max_erosivity (float),
      daily_RUSLE (np.array), daily_SDR (np.array),
      daily_RUSLE_below_threshold (np.array), daily_RUSLE_above_threshold (np.array),
      daily_SDR_below_threshold (np.array), daily_SDR_above_threshold (np.array)
    '''
    mask = rasterio.features.rasterize(geometry, transform=transform, fill=np.nan, dtype=np.float32, out_shape=klscp.shape)
    klscp_masked = klscp*mask
    sdr_masked = sdr*mask
    dnbr_masked = dnbr*mask

    days = pd.Series(rainfall.index.date).drop_duplicates()
    for day in days:
        rainfall_data = rainfall[rainfall.index.date==day]
        # Initialize daily arrays for RUSLE and SDR_RUSLE
        daily_RUSLE = np.zeros_like(klscp_masked, dtype=np.float32)
        daily_SDR = np.zeros_like(klscp_masked, dtype=np.float32)
        daily_total_rain = 0.0
        max_intensity = 0.0
        max_erosivity = 0.0

        # Loop over each 30-min interval
        for subday in rainfall_data.values:
            # Get rainfall amount (∆V_r) during the 30-min period
            # delta_v_r = rainfall_data.sel(subday_30mins=subday).values
            delta_v_r = subday
            daily_total_rain+=delta_v_r

            # Skip calculations if delta_v_r is 0
            if delta_v_r == 0: # What if spatial? max?
                continue

            # Calculate rainfall intensity (∆V_r / ∆t_r)
            intensity = delta_v_r / 0.5  # mm/hr
            max_intensity = max(max_intensity,intensity)

            # Calculate unit kinetic energy (e_r)
            e_r = 0.29 * (1 - 0.72 * np.exp(-EMPIRICAL_COEFFICIENT * intensity))

            # Calculate energy (E)
            E = e_r * delta_v_r

            # Calculate erosivity factor (R)
            R = E * intensity
            max_erosivity = max(max_erosivity,R)

            # Calculate RUSLE
            # TODO: sediment eroded? kg? t?
            RUSLE = (R * klscp_masked) * cell_area_ha # Total erosion in tonnes per hectare
            daily_RUSLE += RUSLE  # Accumulate RUSLE for the day

            # Calculate SDR_RUSLE
            # TODO: sediment delivered? kg? t?
            SDR_RUSLE = RUSLE * sdr_masked  # Delivered erosion/sediment
            daily_SDR += SDR_RUSLE  # Accumulate SDR_RUSLE for the day


        # Mask daily_RUSLE and daily_SDR_RUSLE for dNBR values
        dnbr_below_threshold = dnbr_masked < DNBR_SEVERITY_THRESHOLD
        dnbr_above_threshold = dnbr_masked >= DNBR_SEVERITY_THRESHOLD

        daily_RUSLE_below_threshold = np.where(dnbr_below_threshold, daily_RUSLE, 0)
        daily_RUSLE_above_threshold = np.where(dnbr_above_threshold, daily_RUSLE, 0)

        daily_SDR_below_threshold = np.where(dnbr_below_threshold, daily_SDR, 0)
        daily_SDR_above_threshold = np.where(dnbr_above_threshold, daily_SDR, 0)

        yield((day,daily_total_rain,max_intensity,max_erosivity,daily_RUSLE,daily_SDR,daily_RUSLE_below_threshold,daily_RUSLE_above_threshold,daily_SDR_below_threshold,daily_SDR_above_threshold))

def compute_particulates(rusle_df,constituents_df=None):
    if constituents_df is None:
        constituent_path = os.path.join(os.path.dirname(__file__),'..','..','data','ash_constituents.csv')
        constituents_df = pd.read_csv(constituent_path)

    # Iterate through each constituent and severity
    for _, row in constituents_df.iterrows():
        particulate = row['Particulate constituent (ash)']
        low_severity = row['Low severity- mean amount (mgkg-1)'] * MG_TO_KG
        high_severity = row['High severity- mean amount (mgkg-1)'] * MG_TO_KG

        # Define new column name
        column_name = f"{particulate} (Tonne)"
        rusle_df[column_name] = (rusle_df['RUSLE_SDR (Low severity)'] * low_severity) + (rusle_df['RUSLE_SDR (High severity)'] * high_severity)
    return rusle_df


