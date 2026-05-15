# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""}
from glob import glob
import os
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('pyogrio._io').setLevel(logging.WARNING)
logging.getLogger('distributed.core').setLevel(logging.WARNING)
logging.getLogger('distributed.scheduler').setLevel(logging.WARNING)
logging.getLogger('distributed.nanny').setLevel(logging.WARNING)
logging.getLogger('distributed.diskutils').setLevel(logging.WARNING)
logging.getLogger('distributed.http.proxy').setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from dask import compute, delayed
from dask.distributed import Client
import time

from fire_impacts.pre.project import  initialise_project, find_all_shapefiles
from fire_impacts.pre.topography import extract_catchment_dems, extract_headwaters
from fire_impacts.context import RunContext

# %%
HW_THRESHOLD=20000 # 20,000 m^2 (2ha)
WORKING_DIRECTORY=r''
DEM_PATH=r'' # Add path to DEM (eg to SRTM derived DEM-H)
CATCHMENT_SHAPEFILE_PATH=r'' # Path to a directory containing one or more shapefiles, each containing the boundary of a catchment of interest

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Get the catchment DEM (masking the SRTM DEM by the catchment shapefile)

# %% editable=true slideshow={"slide_type": ""}
fire_impacts_project = initialise_project(WORKING_DIRECTORY, find_all_shapefiles(CATCHMENT_SHAPEFILE_PATH), exist_ok=False,clear=True)
for ctx in RunContext.enumerate_catchments(fire_impacts_project):
    extract_catchment_dems(ctx, DEM_PATH)

print('===================================================================================')
print(f'Masked DEMs were saved in {fire_impacts_project["Catchments_DEM"]} and Catchment_DEM folders')

# %%

# %% editable=true slideshow={"slide_type": ""}
# Get the path to the 'Catchments_DEM' folder
catchments_dem_folder = fire_impacts_project['Catchments_DEM']
clipped_raster_paths = []
# Collecting all raster file paths
for root, dirs, files in os.walk(catchments_dem_folder):
    for file in files:
        if file.endswith('.tif'):
            raster_path = os.path.join(root, file)
            clipped_raster_paths.append(raster_path)
# Reading and plotting each raster file
for raster_path in clipped_raster_paths:
    with rio.open(raster_path) as src:
        data = src.read(1)
        no_data_value = src.nodata
        if no_data_value is not None:
            data = np.where(data == no_data_value, np.nan, data)  # Replace NoData values with NaN
        transform = src.transform
        # Extract the file name without extension for the plot title
        file_name = os.path.splitext(os.path.basename(raster_path))[0].replace('_', ' ')
        plt.figure(figsize=(5, 5))
        img = plt.imshow(data, cmap='viridis', extent=(
            transform[2], transform[2] + transform[0] * data.shape[1],
            transform[5] + transform[4] * data.shape[0], transform[5]
        ))
        plt.title(f'{file_name}', fontsize=12)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        cbar = plt.colorbar(img, label='Elevation')
        plt.show()

# %% [markdown]
# ### Import and list the catchment DEM files, and then delinate headwater (area=2 he)

# %% editable=true slideshow={"slide_type": ""}
# Initialize Dask client
client = Client()
extract_headwaters_d = delayed(extract_headwaters) # Create DASK version of function for running multiple in parallel
# Build one delayed task per catchment via enumerate_catchments
tasks = [
    extract_headwaters_d(ctx, HW_THRESHOLD)
    for ctx in RunContext.enumerate_catchments(fire_impacts_project)
]

# %%

# %% editable=true slideshow={"slide_type": ""}
st = time.time()

# Execute the tasks in parallel
results = compute(*tasks)

# Save all results into a dictionary
WHs_data = {name: hw_data for name, hw_data in results}
print('HWs are delineated and saved in the Topography/catch name/ HW_SHPs and HW_Rasters')

# get the end time
et = time.time()
elapsed_time_h = round((et - st) / 3600, 2)  # get the execution time
elapsed_time_m = round((et - st) / 60, 2)  # get the execution time
print('Execution time:', elapsed_time_h, 'hours')
print('Execution time:', elapsed_time_m, 'minutes')

# %%
WHs_data['DR_Primary_Catchment_Thomson']
#WHs_data['Upper_Yarra']

# %%

# %%
