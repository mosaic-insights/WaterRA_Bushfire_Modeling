import os
from glob import glob
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
    'Soils'
]
STATS=['mean', 'max', 'min', 'median', 'std']

class FireImpactsProject(object):
    '''Objects representing the project folder structure for a fire impacts study.'''
    def __init__(self,project_path,exist_ok=False,clear=False):
        '''
        Initialise a project object from a given project path based on the file found in the project path.
        '''
        self.project_path = project_path
        self.catchments = []
        self.boundary_files = {}
        self.source_data = {}

        if clear or not exist_ok:
          self.initialise_project(project_path,exist_ok=exist_ok,clear=clear)
        else:
          try:
              self.load_project(project_path)
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
        base = os.path.join(self.project_path,'Catchments')
        if catchment_name is None:
            return base
        return os.path.join(base,catchment_name,*args)

    def load_project(self,project_path):
        with open(self._settings_fn(),'r') as f:
            settings = json.load(f)
        self.catchments = settings.get('catchments',[])
        self.source_data = settings.get('source_data',{})
        self.boundary_files = settings.get('boundary_files',{})

    def add_catchment(self,catchment_shapefile,name=None,replace_existing=False):
        if name is None:
            name = os.path.splitext(os.path.basename(catchment_shapefile))[0]
        if name in self.catchments and not replace_existing:
            raise ValueError(f'Catchment {name} already exists in project.')
        self.catchments.append(name)
        self.boundary_files[name] = catchment_shapefile
        catchment_path = self.catchment_path(name)
        for folder in PER_CATCHMENT_FOLDERS:
            os.makedirs(os.path.join(catchment_path,folder),exist_ok=True)
        self._write()

    def add_all_catchments(self,catchment_shapefiles):
        for shapefile in catchment_shapefiles:
            logger.info('Adding catchment from: %s',shapefile)
            self.add_catchment(shapefile)

    def initialise_project(self,project_path,exist_ok=False,clear=False):
        if not clear and os.path.exists(project_path):
            raise FileExistsError(f'Project folder already exists: {project_path}')
        if clear and os.path.exists(project_path):
            logger.info('Clearing existing project folder: %s',project_path)
            shutil.rmtree(project_path)
        os.makedirs(self.catchment_path(),exist_ok=exist_ok)
        self._write()

    def catchment_bounds(self,catchment:str, buffer_distance_km:float=10):
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
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        return gdf.crs

def find_all_shapefiles(base_directory):
    '''
    Find all shapefiles in a directory and its subdirectories.
    '''
    assert os.path.isdir(base_directory), f"Directory not found: {base_directory}"
    shapefiles = glob(os.path.join(base_directory, '**','*.shp'),recursive=True)
    return shapefiles

def summary_stats(project:FireImpactsProject,catchment_name=None):
    if isinstance(project,str):
        project = FireImpactsProject(project)
    if catchment_name is None:
        return {catchment_name:summary_stats(project,catchment_name) for catchment_name in project.catchments}

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
