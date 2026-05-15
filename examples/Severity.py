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

# %%
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import os
import geopandas as gpd

from fire_impacts.pre.project import initialise_project, find_all_shapefiles
from fire_impacts.pre.severity import calculate_fire_severity
from fire_impacts.context import RunContext


# %%
# Define working directory and shapefile path
WORKING_DIRECTORY = './fire-impacts-data'
CATCHMENT_SHAPEFILE_PATH = '' # Set path to catchment boundary shapefiles
FIRE_START_DATE = '2019-01-15'  # Set fire start date (the date that fire started)
FIRE_END_DATE = '2019-03-07'    # Set fire end date (the date that fire ended)



# %%
# Get all shapefiles and bounding boxes
shapefiles = find_all_shapefiles(CATCHMENT_SHAPEFILE_PATH)

# Initialize project folders with exist_ok=True to avoid FileExistsError
project_folders = initialise_project(WORKING_DIRECTORY, catchment_shapefiles=shapefiles, exist_ok=True)

for ctx in RunContext.enumerate_events(
    project_folders, event=FIRE_START_DATE,
):
    calculate_fire_severity(
        ctx,
        fire_start_date=FIRE_START_DATE,
        fire_end_date=FIRE_END_DATE,
    )

# %%
