'''
This module contains the classes and functions that are used to manage the data for the fire_impacts module.
'''

import os
from glob import glob
from pathlib import Path
import shutil
import rasterio as rio
import rasterstats as rs
import geopandas as gpd
import pandas as pd
import json
import logging
logger = logging.getLogger(__name__)

PER_CATCHMENT_FOLDERS = [
    'Topography',
    'FireSeverity',
    'Soils',
    'Erodibility'
]
STATS=['mean', 'max', 'min', 'median', 'std']

class FireImpactsProject(object):
    '''Objects representing the project folder structure for a fire impacts study.'''
    def __init__(self,project_path,exist_ok=False,clear=False):
        '''
        Initialise a project object from a given project path based on the file found in the project path.

        Keeps track of data related to one or more catchments. Register catchments using the add_catchment or add_all_catchments methods.

        Parameters:
        - project_path (str): Path to the project folder.
        - exist_ok (bool): If True, do not raise an error if the project folder already exists.
        - clear (bool): If True, clear the project folder if it already exists.
        '''
        self.project_path = project_path
        self.catchments = []
        self.boundary_files = {}
        self.source_data = {}

        if clear or not exist_ok:
          self.initialise_project(project_path,exist_ok=exist_ok,clear=clear)
        else:
          try:
              self.load_project()
          except:
              self.initialise_project(project_path,exist_ok=exist_ok,clear=clear)


    def _settings_fn(self):
        return os.path.join(self.project_path,'settings.json')

    def _settings(self):
        return dict(catchments=self.catchments,source_data=self.source_data,boundary_files=self.boundary_files)

    def _write(self):
        with open(self._settings_fn(),'w') as f:
            json.dump(self._settings(),f,indent=2)

    def catchment_path(self,catchment_name=None,*args):
        '''
        Expand a path relative to the project folder to a full path.

        Parameters:
        - catchment_name (str): Name of the catchment to expand the path for. If not provided, the path will be expanded to the main Catchments folder.
        - args (list): Additional path components to expand.
        '''
        base = os.path.join(self.project_path,'Catchments')
        if catchment_name is None:
            assert len(args) == 0, 'Cannot specify additional arguments without a catchment name.'
            return base
        return os.path.join(base,catchment_name,*args)

    def load_project(self):
        '''
        (re)Load the project settings from the settings file.
        '''
        with open(self._settings_fn(),'r') as f:
            settings = json.load(f)
        self.catchments = settings.get('catchments',[])
        self.source_data = settings.get('source_data',{})
        self.boundary_files = settings.get('boundary_files',{})
        self.ensure_catchment_folders()

    def add_catchment(self,catchment_shapefile:str|Path,name=None,replace_existing=False):
        '''
        Register a new catchment in the project.

        Parameters:
        - catchment_shapefile (str): Path to the shapefile defining the catchment boundary.
        - name (str): Name to use for the catchment. If not provided, the name will be derived from the shapefile name.
        - replace_existing (bool): If True, replace an existing catchment with the same name.
                                   If False, raise an error if a catchment with the same name already exists.
        '''
        if name is None:
            name = os.path.splitext(os.path.basename(catchment_shapefile))[0]
        if name in self.catchments and not replace_existing:
            raise ValueError(f'Catchment {name} already exists in project.')
        self.catchments.append(name)
        self.boundary_files[name] = str(catchment_shapefile)
        self.ensure_catchment_folders(name)
        self._write()

    def ensure_catchment_folders(self,catchment_name:str=None):
        if catchment_name is None:
            for catchment in self.catchments:
                self.ensure_catchment_folders(catchment)
            return
        catchment_path = self.catchment_path(catchment_name)
        for folder in PER_CATCHMENT_FOLDERS:
            os.makedirs(os.path.join(catchment_path,folder),exist_ok=True)

    def add_all_catchments(self,catchment_shapefiles):
        '''
        Register all catchments in the project from a list of shapefiles.

        Parameters:
        - catchment_shapefiles (list): List of paths to the shapefiles defining the catchment boundaries.

        Note: This method will replace any existing catchments in the project.
        '''
        for shapefile in catchment_shapefiles:
            logger.info('Adding catchment from: %s',shapefile)
            self.add_catchment(shapefile,replace_existing=True)

    def initialise_project(self,project_path,exist_ok=False,clear=False):
        if not clear and os.path.exists(project_path):
            raise FileExistsError(f'Project folder already exists: {project_path}')
        if clear and os.path.exists(project_path):
            logger.info('Clearing existing project folder: %s',project_path)
            shutil.rmtree(project_path)
        os.makedirs(self.catchment_path(),exist_ok=exist_ok)
        self._write()

    def catchment_bounds(self,catchment:str, buffer_distance_km:float=10):
        '''
        Get the bounding box for a catchment with an (optional) buffer distance in approximate kilometres.
        '''
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        bbox = gdf_wgs84.total_bounds

        # Convert 10 km to degrees (approximate conversion, 1 degree = 111 km)
        buffer_degrees = buffer_distance_km / 111  # This is an approximation for small distances

        # Apply buffer to the bounding box
        bbox_with_buffer = [
            bbox[0] - buffer_degrees,  # minx with buffer
            bbox[1] - buffer_degrees,  # miny with buffer
            bbox[2] + buffer_degrees,  # maxx with buffer
            bbox[3] + buffer_degrees   # maxy with buffer
        ]
        return bbox_with_buffer

    def catchment_crs(self,catchment:str):
        '''
        Get the CRS for a catchment from the catchment boundary coverage.
        '''
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        return gdf.crs

    def for_each_catchment(self,fn:callable):
        '''
        Run a function for each catchment in the project.

        Parameters:
        - fn (callable): Function to run for each catchment. The function should take a single argument (the catchment name) and optionally return a value.

        Returns:
        - dict: Dictionary containing the results of the function for each catchment.
        '''
        logger.info('Processing %d catchments',len(self.catchments))
        return {catchment:fn(catchment) for catchment in self.catchments}

    def plot_catchment_raster(self,*args,catchment=None,figure=None):
        '''
        '''
        if figure is None:
            from matplotlib import pyplot as plt
            figure = plt.figure()

        if catchment is None:
            self.for_each_catchment(lambda c:self.plot_catchment_raster(*args,catchment=c,figure=figure))
            return

        import rasterio as rio
        import os
        import numpy as np
        raster_path = self.catchment_path(catchment,*args)
        if not raster_path.endswith('.tif'):
            raster_path += '.tif'

        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)

        with rio.open(raster_path) as src:
            data = src.read(1)
            no_data_value = src.nodata
            if no_data_value is not None:
                data = np.where(data == no_data_value, np.nan, data)  # Replace NoData values with NaN
            transform = src.transform
            file_name = os.path.splitext(os.path.basename(raster_path))[0].replace('_', ' ')
            ax = figure.add_subplot()
            img = ax.imshow(data, cmap='viridis', extent=(
                transform[2], transform[2] + transform[0] * data.shape[1],
                transform[5] + transform[4] * data.shape[0], transform[5]
            ))
            ax.set_title(f'{catchment} {file_name}', fontsize=12)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            cbar = figure.colorbar(img, label=args[-1].split('.')[0])
            gdf.plot(ax=ax, facecolor='none', edgecolor='red')
            return

def find_all_shapefiles(base_directory):
    '''
    Find all shapefiles in a directory and its subdirectories.
    '''
    assert os.path.isdir(base_directory), f"Directory not found: {base_directory}"
    shapefiles = glob(os.path.join(base_directory, '**','*.shp'),recursive=True)
    return shapefiles

def summary_stats(project:FireImpactsProject,catchment_name=None):
    '''
    Calculate summary statistics for a catchment from pre-processed raster data.

    Parameters:
    - project (FireImpactsProject): Project object containing the catchment data.
    - catchment_name (str): Name of the catchment to process. If not provided, process all catchments in the project.

    Returns:
    - pd.DataFrame: DataFrame containing the summary statistics for the catchment (if catchment_name is provided), OR
    - dict: Dictionary of DataFrames containing the summary statistics for each catchment.
    '''
    if isinstance(project,str):
        project = FireImpactsProject(project)
    if catchment_name is None:
        return project.for_each_catchment(lambda c:summary_stats(project,c))

    headwaters_path = project.catchment_path(catchment_name,'Topography','Headwaters.shp')
    gdf = gpd.read_file(headwaters_path)
    # Initialize a list to store the results
    results = []

    sources = [
        ('Slope',('Topography','Slope.tif')),
        ('dNBR',('FireSeverity','dNBR.tif')),
        ('Aridity',('Soils','Aridity.tif')),
        # ('Rain','Rain','Rainfall.tif')
    ]

    soil_path = project.catchment_path(catchment_name,'Soils')
    for fn in os.listdir(soil_path):
        abs_fn = os.path.join(soil_path,fn)
        if not os.path.isdir(abs_fn):
            continue

        for child_fn in os.listdir(abs_fn):
            if child_fn.endswith('.tif'):
                sources.append((child_fn.replace('.tif',''),('Soils',fn,child_fn)))

    # Process each polygon in the shapefile
    result = {
        'ID':gdf['ID']
    }

    logger.info('Processing %d polygons for %d layers in %s',len(gdf),len(sources),catchment_name)
    for label, path in sources:
        logging.info('Processing %s from %s',label,path[-1])
        stats = get_zonal_stats(gdf, project.catchment_path(catchment_name,*path),label)
        for k in STATS:
            result[f'{label}_{k}'] = [s[k] for s in stats]

    extracted_data = pd.DataFrame(result)

    csv_path=project.catchment_path(catchment_name, 'Soil_Slope_Aridity_dNBR.csv')
    extracted_data.to_csv(csv_path, index=False)


def get_zonal_stats(gdf, raster_path,label):
    '''Function to get stats for a given polygon and raster'''
    with rio.open(raster_path) as src:
        assert src.crs == gdf.crs, f"CRS mismatch: {src.crs} != {gdf.crs}"
        stats = rs.zonal_stats(
            gdf,
            raster_path,
            stats=STATS,
            nodata=src.nodata or -9999
        )
    return stats
