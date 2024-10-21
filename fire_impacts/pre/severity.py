'''
This module calculates fire severity (NBR, dNBR) before and after the fire.
'''

import os
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from datetime import datetime, timedelta
import pystac_client
import odc.stac
from dea_tools.bandindices import calculate_indices
import logging
from .project import FireImpactsProject
logger = logging.getLogger(__name__)

CATALOG=None
def init_catalog(url:str="https://explorer.dea.ga.gov.au/stac"):
    #Connect to the DEA Explorer STAC API to allow searching for data
    global CATALOG
    if CATALOG is None:
        CATALOG = pystac_client.Client.open(url)
        odc.stac.configure_rio(
            cloud_defaults=True,
            aws={"aws_unsigned": True}
        )

def calculate_fire_severity(project:FireImpactsProject, catchment:str, fire_start_date, fire_end_date,
                            start_date_pre=None, end_date_post=None, collection_id=('ga_s2am_ard_3','ga_s2bm_ard_3'),
                            max_cloud_cover=20, resolution_input=20, bbox=None):
    '''
    This function calculates fire severity (NBR, dNBR) before and after the fire.

    Parameters:
    - project (dict): Dictionary with folder paths initialized for the project.
    - catchment (str): Name of the registted catchment to process.
    - fire_start_date (str): The date when the fire started.
    - fire_end_date (str): The date when the fire ended.
    - start_date_pre (str): The start date for pre-fire data. (Default is 90 days before fire_start_date)
    - end_date_post (str): The end date for post-fire data. (Default is 90 days after fire_end_date)
    - collection_id (list): Collection IDs (e.g., 'ga_s2am_ard_3').
    - max_cloud_cover (int): Maximum cloud cover percentage.
    - resolution_input (int): Resolution in meters.
    - bbox (list): Bounding box for the catchment area.

    Returns:
    - None. Saves NBR, dNBR, and metadata files to disk.
    '''
    if CATALOG is None:
        init_catalog()

    shapefile_path = project.boundary_files[catchment]
    def date_rel(date:str, days:int):
        return (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')
    # Get pre and post fire date ranges
    end_date_pre = date_rel(fire_start_date,-1)
    start_date_post = date_rel(fire_end_date,1)
    if start_date_pre is None:
        logger.info('No start date for pre-fire data provided. Defaulting to 90 days before fire start date.')
        start_date_pre = date_rel(fire_start_date,-90)
    if end_date_post is None:
        logger.info('No end date for post-fire data provided. Defaulting to 90 days after fire end date.')
        end_date_post = date_rel(fire_end_date,90)

    if bbox is None:
        logger.info('Bounding box not provided. Calculating bounding box from shapefile with 10km buffer.')
        bbox = project.catchment_bounds(catchment,10)

    # Get the correct collection type
    # collection_type = ['ga_ls_3', 'ga_s2_3', 'ga_gm_3']
    if 'ga_s2' in collection_id[0]:
        selected_collection_type = 'ga_s2_3'
    elif 'ga_ls' in collection_id[0]:
        selected_collection_type = 'ga_ls_3'
    else:
        raise ValueError(f"Unsupported collection type: {collection_id[0]}")

    # Define the filter for cloud cover
    filter_query = f"eo:cloud_cover < {max_cloud_cover}"

    # Load the catchment shapefile and prepare folder
    catchment_folder = project.catchment_path(catchment,'FireSeverity')
    os.makedirs(catchment_folder, exist_ok=True)
    gdf = gpd.read_file(shapefile_path)

    def calc_nbr(datetime,label,use_mask=True):
        query = CATALOG.search(
            bbox=bbox,
            collections=collection_id,
            datetime=datetime,
            filter=filter_query,
            sortby="eo:cloud_cover",
        )
        items = list(query.items())
        logger.info(f"Found: {len(items):d} datasets")
        bands = ['nbart_nir_1', 'nbart_swir_3']
        if use_mask:
            bands.append('oa_s2cloudless_mask') # oa_s2cloudless_mask is not used for POST fire. WHY?
        ds = odc.stac.load(
            items,
            bands=bands,
            crs=gdf.crs,
            resolution=resolution_input,
            groupby="solar_day",
            bbox=bbox,
        )
        image_metadata = extract_image_metadata(items, ds['time'].values, resolution_input, label)

        nbr = calculate_indices(ds, index='NBR', collection=selected_collection_type, drop=False)
        image = nbr.median(dim='time')
        nbr_ds = image.NBR
        nbr_ds.rio.write_crs(gdf.crs).rio.to_raster(os.path.join(catchment_folder, f"{label}_NBR.tif"))

        return image_metadata, nbr_ds

    # Calculate Pre-fire NBR
    logger.info(f'Calculating Pre_fire NBR for {catchment}')
    pre_image_metadata, prefire_NBR = calc_nbr(f"{start_date_pre}/{end_date_pre}",'Prefire',use_mask=True)


    # Calculate Post-fire NBR
    logger.info(f'Calculating Post_fire NBR for {catchment}')
    post_image_metadata, postfire_NBR = calc_nbr(f"{start_date_post}/{end_date_post}",'Postfire',use_mask=False)

    # Combine Pre-fire and Post-fire metadata
    combined_metadata = pre_image_metadata + post_image_metadata  # Merging both pre and post metadata

    # Save combined metadata to a single CSV
    save_metadata_to_csv(combined_metadata, catchment_folder, "Satellite_image_information_combined.csv")

    # Calculate dNBR
    logger.info(f'Calculating dNBR for {catchment}')
    delta_NBR = prefire_NBR - postfire_NBR
    delta_NBR.rio.write_crs(gdf.crs).rio.to_raster(os.path.join(catchment_folder, "dNBR.tif"))
    logger.info('Processes are completed')

def extract_image_metadata(items, valid_times, resolution_input, fire_status):
    valid_times_str = [pd.to_datetime(time).isoformat() + 'Z' for time in valid_times]
    metadata = [
        {
            "Pre/Post-fire": fire_status,
            "Satellite Name": item.properties.get("platform"),
            "Product ID": item.collection_id,
            "Time of Image": item.properties.get("datetime"),
            "Cloud Cover (%)": round(item.properties.get("eo:cloud_cover"), 1),
            "Resolution": resolution_input
        }
        for item in items
        if item.properties.get("datetime") in valid_times_str
    ]
    return metadata

def save_metadata_to_csv(metadata, folder, filename):
    pd.DataFrame(metadata).to_csv(os.path.join(folder, filename))
