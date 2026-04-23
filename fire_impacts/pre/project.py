'''
This module contains the classes and functions that are used to manage the data for the fire_impacts module.
'''

import os
from glob import glob
from pathlib import Path
import shutil
import rasterio as rio
import numpy as np
import geopandas as gpd
import pandas as pd
import json
import logging
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# Get the top-level util specifically using importlib:
import importlib
toputil = importlib.import_module('fire_impacts.util') 
const = importlib.import_module('fire_impacts.const')
logger = logging.getLogger(__name__)

# These are the default directories that need to exist inside every 
#catchments directory:
PER_CATCHMENT_FOLDERS = const.PER_CATCHMENT_FOLDERS

STATS = const.STATS
APPROX_KM_PER_DEGREE = const.APPROX_KM_PER_DEGREE  

# State exactly what dtypes we're happy to save rasters in:
default_dtypes_raster = {
    'int': rio.int32,
    'float': rio.float32
    }
# Convert numpy one-character dtype.kind attributes into more general 
#descriptors that will map in default_dtypes_raster:
numpy_kind_to_desc = const.numpy_kind_to_desc


###############################################################################
####### FireImpactsProject ####################################################
###############################################################################
class FireImpactsProject(object):
    """
    Object representing the project folder structure for a fire impacts 
    study.
    --------------------------------------------------------------------
    Notes:
    - Keeps track of data related to one or more catchments
    - Register catchments using the add_catchment or add_all_catchments 
    methods.
    --------------------------------------------------------------------
    """
    ###########################################################################
    def __init__(self,project_path,exist_ok=False,clear=False):
        """
        Initialise a project object from a given project path based on 
        the file found in the project path.
        
        Parameters:
        - project_path (str): Path to the project folder.
        - exist_ok (bool): If True, do not raise an error if the 
        project folder already exists.
        - clear (bool): If True, clear the project folder if it already 
        exists.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        norm_path = os.path.normpath(project_path)
        self.project_path = norm_path
        self.catchments = []
        self.boundary_files = {}
        self.source_data = {}

        # If the user has said to clear the existing folder OR they have
        #said to proceed with loading a new project even if there is
        #already a folder there:
        if clear or not exist_ok:
            self.initialise_project(norm_path,exist_ok=exist_ok,clear=clear)
        else:
            try:
                self.load_project()
            except:
                self.initialise_project(
                    norm_path,exist_ok=exist_ok,clear=clear
                    )

        self.load_vis_defaults()
        self.load_name_defaults()

    ###########################################################################
    def _settings_fn(self):
        """
        Load settings from a .json file in the project folder
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        return os.path.join(self.project_path,'settings.json')

    ###########################################################################
    def _settings(self):
        """
        Get the settings from the project that are required for the
        settings.json file
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        settings_dict = dict(
            catchments=self.catchments,
            source_data=self.source_data,
            boundary_files=self.boundary_files
            )
        return settings_dict

    ###########################################################################
    def _write(self):
        """
        Write the settings from the current project to a .json file
        ----------------------------------------------------------------
        Notes:
        - All this really does is save the paths for the boundary files
        in a json-friendly way
        ----------------------------------------------------------------
        """
        # Use _settings_fn() to get the path in write mode:
        with open(self._settings_fn(),'w') as f:
            json.dump(self._settings(),f,indent=2)

    ###########################################################################
    def catchment_path(self,catchment_name=None,*args):
        """
        Expand a path based on key words to a usable path relative to 
        a particular catchment

        Parameters:
        - catchment_name (str): Name of the catchment to expand the 
        path for. If not provided, the path will be expanded to the 
        main Catchments folder.
        - args (list): Additional path components to expand.

        Returns:
        - Path to the catchment folder, or base as a fallback
        ----------------------------------------------------------------
        Notes:
        - Args should correspond to subfolder names, for example 
        'Erodibility', 'KLSCP.tif' gives the full path to that file
        ----------------------------------------------------------------
        """
        # Every project will have a Catchments folder:
        base = os.path.join(self.project_path,'Catchments')
        # If they haven't provided additional arguments, just return
        #the top level:
        if catchment_name is None:
            assert len(args) == 0, (
                'Cannot specify additional arguments without a '
                'catchment name.'
                )
            return base
        return os.path.join(base,catchment_name,*args)

    ###########################################################################
    def load_project(self):
        """
        (re)Load the project settings from the settings file.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Get the current path and load the settings file in:
        with open(self._settings_fn(),'r') as f:
            settings = json.load(f)
        # Get the names of the current catchments and assign to 
        #class instance:
        self.catchments = settings.get('catchments',[])
        # If Source data has been loaded, assign to class instance:
        self.source_data = settings.get('source_data',{})
        # Load boundary files to class instance:
        self.boundary_files = settings.get('boundary_files',{})


        self.ensure_catchment_folders()
        self.load_vis_defaults()
        self.load_name_defaults()

    ###########################################################################
    def add_catchment(
        self,
        catchment_shapefile:str|Path,
        name=None,
        replace_existing=False
        ):
        """
        Register a new catchment in the project

        Parameters:
        - catchment_shapefile: string or Path object pointing to a 
        shapefile of the catchment boundary
        - name: name to use for the catchment. If not provided, the 
        name will be derived from the shapefile name.
        - replace_existing: whether the shapefile should be overwritten 
        if it already exists
        ----------------------------------------------------------------
        Notes:
        - replace_existing behaviour:
            - If True, replace an existing catchment with the same name
            - If False, raise an error if a catchment with the same 
            name already exists.
        ----------------------------------------------------------------
        """
        # If a name hasn't been specified, derive one from the 
        #shapefile name:
        if name is None:
            name = os.path.splitext(os.path.basename(catchment_shapefile))[0]
        
        # Check if the catchment is already there:
        have_already = name in self.catchments
        # If so, and the user hasn't said to replace, raise an error:
        if have_already and not replace_existing:
            raise ValueError(
                f'Catchment {name} already exists in project.'
                )
        # If the catchment isn't already there, add its name to the 
        #list of catchments in the class instance
        if not have_already:
            self.catchments.append(name)
        # Add an entry to the current instance's boundary files 
        #dictionary pointing to the current catchment shapefile path:
        self.boundary_files[name] = str(catchment_shapefile)

        # Create the standard set of subcatchment folders for this new
        #catchment:
        self.ensure_catchment_folders(name)
        # Update the settings.json file so it includes the new 
        #catchment:
        self._write()

    ###########################################################################
    def add_subcatchments(
        self,
        catchment_name:str,
        subcatch_shapefile_path:str,
        id_cols:list=[]
        ):
        """
        Load subcatchments from a shapefile.

        Parameters:
        ----------------------------------------------------------------
        Notes:
        - Reproject to the catchment crs
        - Clip to the catchment boundary
        - Keep identifying attributes
        - Load into class instance as geodataframe
        - Add path to boundary files and update settings.json
        - Save processed boundary as shapefile in the subcatchments 
        folder

        TODO:
        - Handle slivers near edge of catchment boundary
        ----------------------------------------------------------------
        """
        # Read in the proposed subcatchments:
        in_gdf = gpd.read_file(subcatch_shapefile_path)
        # Check and compare CRS of subcatchment and existing catchment:
        subcatch_crs = in_gdf.crs
        catch_crs = self.catchment_crs(catchment_name)
        if subcatch_crs != catch_crs:
            catch_epsg = catch_crs.to_epsg()
            subcatch_epsg = subcatch_crs.to_epsg()
            logger.info(
                f'Subcatchment shapefile CRS is EPSG: {subcatch_epsg}. '
                'Reprojecting/transforming to the catchment CRS which '
                f'is EPSG: {catch_epsg}.'
                )
            int_gdf = in_gdf.to_crs(catch_crs)
        else:
            int_gdf = in_gdf

        # Check that there is at least some overlap in the bounding 
        #boxes of the newly 
        catch_gdf = self.catchment_boundary(catchment_name)
        logger.info(
            'Clipping subcatchments to the catchment polygon...'
            )
        # Clip the subcatchments to the catchment boundary
        subcatch_clipped = int_gdf.clip(catch_gdf)
        # Raise an error if there's no shared area:
        if subcatch_clipped.empty:
            raise ValueError(
                'Only subcatchment areas within the catchment boundary '
                'can be processed, but there were none left after '
                'clipping.'
                )
        # Add original subcatchment geodataframe to boundary files:
        key_name = catchment_name + '_' + 'subcatchments'
        self.boundary_files[key_name] = subcatch_shapefile_path
        self._write()

        # Get only the useful columns, plus geometry:
        good_cols = id_cols + [subcatch_clipped.geometry.name]
        int_gdf = subcatch_clipped[good_cols]

        #Use the index as the internal integer subcatchment id (sc_ID)
        out_gdf = int_gdf.reset_index(drop=False, names=self.subcatchment_id)

        # Save the clipped subcatchments to the subcatchments folder:
        save_path = self.catchment_path(catchment_name, 'Subcatchments')
        key_file_name = key_name + '.shp'
        key_file_path = os.path.join(save_path, key_file_name)
        out_gdf.to_file(key_file_path)
        logger.info(
            'Saved clipped subcatchments shapefile in the catchment '
            f'crs to {key_file_path}'
            )

    ###########################################################################
    def ensure_catchment_folders(self,catchment_name:str=None):
        """
        Make sure the project directory structure is as expected. 
        Create the required folders if they don't already exist.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Run for all catchments if none is specified:
        if catchment_name is None:
            for catchment in self.catchments:
                self.ensure_catchment_folders(catchment)
            return
        # Get the catchment-level folder:
        catchment_path = self.catchment_path(catchment_name)
        # Go throuh each of the required folders:
        for folder in PER_CATCHMENT_FOLDERS:
            # Make a new one if it's not already there:
            os.makedirs(os.path.join(catchment_path,folder),exist_ok=True)

    ###########################################################################
    def add_all_catchments(self,catchment_shapefiles):
        """
        Register all catchments in the project from a list of 
        shapefiles.

        Parameters:
        - catchment_shapefiles (list): List of paths to the shapefiles 
        defining the catchment boundaries.
        ----------------------------------------------------------------
        Notes:
        - This method will replace any existing catchments in the 
        project.
        ----------------------------------------------------------------
        """
        for shapefile in catchment_shapefiles:
            logger.info('Adding catchment from: %s',shapefile)
            self.add_catchment(shapefile,replace_existing=True)

    ###########################################################################
    def initialise_project(self,project_path,exist_ok=False,clear=False):
        """
        Load a brand new project in the specified path. Throw an error 
        if it already exists but user has said not to clear.

        Parameters:
        - project_path: path to the desired location, including the 
        desired project name as the final folder
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If there is already a folder and the user has said NOT to
        #clear it:
        if not clear and os.path.exists(project_path):
            raise FileExistsError(
                f'Project folder already exists: {project_path}'
                )
        # If there is already a folder and the user as said it's ok to
        #clear its contents:
        if clear and os.path.exists(project_path) and not exist_ok:
            logger.info('Clearing existing project folder: %s',project_path)
            # Remove the directory and all of its contents:
            shutil.rmtree(project_path)
        # Create a new folder in the location of the project path:
        os.makedirs(self.catchment_path(),exist_ok=exist_ok)
        # Write the settings (which will initially be shells):
        self._write()

    ###########################################################################
    def catchment_boundary(self,catchment:str) -> gpd.GeoDataFrame:
        """
        Get the catchment boundary as a GeoDataFrame.

        Parameters:
        - catchment: name of the catchment loaded into the class 
        instance

        Returns:
        - Geodataframe of the catchment boundary file
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        return gdf

    ###########################################################################
    def get_subcatchments(self,catchment:str) -> gpd.GeoDataFrame:
        """
        Get the subcatchment boundaries as a GeoDataFrame.

        Parameters:
        - Name of the catchment to get subcatchments for

        Returns:
        - Geodataframe of subcatchments if it exists, otherwise just 
        the catchment boundary itself
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Load basic path elements:
        shape_name = catchment + '_subcatchments.shp'
        project_folder = 'Subcatchments'
        new_id_col_name = self.subcatchment_id
        # Read in the shapefile:
        gdf = self.get_catchment_polygons(
            catchment,
            project_folder,
            shape_name,
            new_id_col_name
            )
        
        return gdf
    
    ###########################################################################
    def get_headwaters(self, catchment:str) -> gpd.GeoDataFrame:
        """
        Get the headwater boundaries and basic attributes as a 
        GeoDataFrame.

        Parameters:
        - Name of the catchment to get subcatchments for

        Returns:
        - Geodataframe of headwaters
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Load basic path elements:
        shape_name = 'Headwaters.shp'
        project_folder = 'Topography'
        new_id_col_name = self.headwater_id
        # Read in the shapefile:
        gdf = self.get_catchment_polygons(
            catchment,
            project_folder,
            shape_name,
            new_id_col_name
            )
        return gdf

        
    ###########################################################################
    def get_catchment_polygons(
        self,
        catchment:str,
        folder:str,
        poly_file_name:str,
        auto_id_col_name:str
        ) -> gpd.GeoDataFrame:
        """
        Read a shapefile containing relevant polygons in a catchment and
        return a GeoDataFrame
        ----------------------------------------------------------------
        Notes:
        - First check if headwaters have been created.
        - Return the GeoDataFrame wiht all the inherent columns
        ----------------------------------------------------------------
        """
        # Get the path to the shapefile:
        shapefile_path = self.catchment_path(
            catchment,
            folder,
            poly_file_name
            )
        # Check that it exists and if so, return it:
        if os.path.exists(shapefile_path):
            gdf = gpd.read_file(shapefile_path)
            # Only create the ID column if the shapefile doesn't already
            # have one. extract_headwaters() writes hw_ID as 1-based
            # integers; blindly overwriting it with gdf.index (0-based)
            # would cause a one-position mismatch when merging against
            # any CSV that was built from the shapefile's own IDs.
            if auto_id_col_name not in gdf.columns:
                gdf[auto_id_col_name] = gdf.index
            return gdf
        # Otherwise return None
        else:
            raise FileNotFoundError(
                f'Catchment polygons ({poly_file_name}) were requested '
                f'from project.get_catchment_polygons() for {catchment}, '
                'but they appear not to be loaded yet. Use '
                'project.add_subcatchments() or '
                'topography.extract_headwaters() first.'
                )

    ###########################################################################
    def catchment_bounds(self,catchment:str, buffer_distance_km:float=10):
        """
        Get the bounding box for a catchment in WGS84 with an 
        (optional) buffer distance in approximate kilometres.

        Parameters:
        - catchment: name of the catchment to get the bounding box of
        - buffer_distance_km: number of kilometres beyond the 
        catchment's boundary to buffer before getting the bounding box

        Returns:
        - List of min/max longitude/latitude of the buffered catchment 
        boundary
        ----------------------------------------------------------------
        Notes:
        - Buffer is important to allow for differences in projections 
        etc.
        - Primarily used for getting satellite data through dea-tools
        ----------------------------------------------------------------
        """
        # Get the catchmetn boundary and transform to WGS84:
        gdf = self.catchment_boundary(catchment)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        # Get the extent from the transformed geodataframe:
        bbox = gdf_wgs84.total_bounds

        # Convert 10 km to degrees (approximate conversion, 1 degree = 
        #111 km)
        buffer_degrees = buffer_distance_km / APPROX_KM_PER_DEGREE

        # Apply buffer to the bounding box
        bbox_with_buffer = [
            bbox[0] - buffer_degrees,  # minx with buffer
            bbox[1] - buffer_degrees,  # miny with buffer
            bbox[2] + buffer_degrees,  # maxx with buffer
            bbox[3] + buffer_degrees   # maxy with buffer
        ]
        return bbox_with_buffer

    ###########################################################################
    def catchment_crs(self,catchment:str):
        """
        Get the CRS for a catchment from the catchment boundary

        Parameters:
        - catchment: name of the catchment

        Returns:
        - GeoPandas crs object
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        gdf = self.catchment_boundary(catchment)
        return gdf.crs

    ###########################################################################
    def cell_area(self,catchment:str=None):
        """
        Get the cell area of the DEM for a catchment

        Parameters:
        - catchment: name of the catchment
        
        Returns:
        - Planar area of one cell in the catchment DEM, in the units of 
        that DEM
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If no catchment sspecified, process for all catchments:
        if catchment is None:
            return self.for_each_catchment(
                lambda c:self.cell_area(catchment=c)
                )

        # Get the path for the DEM for the current cathment:
        fn = self.catchment_path(catchment,'Topography','DEM.tif')
        
        # Open the DEM in rasterio:
        with rio.open(fn) as src:
            # Get the transform, which has the cell sizes
            transform = src.transform
            # Compute width *  height and return the positive value:
            return abs(transform.a * transform.e)

    ###########################################################################
    def for_each_catchment(self,fn:callable):
        """
        Run a function for each catchment in the project.

        Parameters:
        - fn (callable): Function to run for each catchment. The
        function should take a single argument (the catchment name)
        and optionally return a value.

        Returns:
        - dict: Dictionary containing the results of the function for
        each catchment.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        logger.info('Processing %d catchments',len(self.catchments))
        return {catchment:fn(catchment) for catchment in self.catchments}

    ###########################################################################
    def load_vis_defaults(self):
        """
        Helper function to load certain values for default
        visualisations
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        from matplotlib.colors import LogNorm
        self.vis_DEM = {
            'cmap': 'viridis',
            'measure': 'Elevation',
            'units': 'm',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': 'DEM'
        }

        self.vis_slope = {
            'cmap': 'viridis',
            'measure': 'Slope',
            'units': '°',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': 'Slope'
        }

        self.vis_flow_accum = {
            'cmap': 'viridis',
            'measure': 'Contributing areas',
            'units': 'count',
            'norm': 'log',
            'vmin': 10,
            'cbar_extend': 'min',
            'title_varname': 'Flow Accumulation'
        }

        self.vis_dNBR = {
            'cmap': 'inferno',
            'measure': 'ΔNBR',
            'units': 'raw',
            'title_varname': 'ΔNBR',
            'norm': 'linear',
            'cbar_extend': 'neither'
        }

        self.vis_i12_crit = {
            'cmap': 'plasma_r',
            'measure': '12-minute intensity threshold',
            'units': 'mm/hr',
            'title_varname': 'Rain Intensity I12 Crit',
            'norm': 'linear',
            'cbar_extend': 'neither'
        }

        self.vis_num_debris_flow_events = {
            'cmap': 'Reds',
            'measure': 'Debris Flow Events',
            'units': 'count',
            'title_varname': 'Debris Flow Events',
            'norm': 'boundary',
            'cbar_extend': 'neither',
        }

        self.vis_aridity = {
            'cmap': 'cividis',
            'measure': 'Aridity Factor',
            'units': 'wet → dry',
            'title_varname': 'Aridity',
            'norm': 'linear',
            'cbar_extend': 'neither'
        }

        self.vis_erosion = {
            'cmap': 'cividis',
            'measure': 'Erosion',
            'units': 'tonnes per cell',
            'title_varname': '',
            'norm': 'linear',
            'cbar_extend': 'neither'
        }

        self.vis_delivered = {
            'cmap': 'cividis',
            'measure': 'Sediment Delivery',
            'units': 'tonnes per cell',
            'title_varname': '',
            'norm': 'linear',
            'cbar_extend': 'neither'
        }

        self.vis_debris_mass = {
            'cmap': 'cividis',
            'measure': 'Available Debris Mass',
            'units': 'Kg',
            'title_varname': 'Debris Flow Mass',
            'norm': 'log',
            'vmin': 0,
            'cbar_extend': 'neither'
            }

    ###########################################################################
    def get_vis_params(self, file_or_col_name:str):
        """
        Get the appropriate visualisation parameters based on the name 
        of a raster file OR the name of a column in a table
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        input_string = file_or_col_name.lower().strip().replace(' ', '_')
        clay_mass_fmt = const.DEBRIS_MASS_FIELD.lower().strip().replace(' ', '_')

        # Fallback values:
        default_params = {
            'cmap': 'viridis',
            'measure': 'Undefined',
            'units': 'n/a',
            'norm': None,
            'cbar_extend': 'neither',
            'title_varname': ''
            }
        
        # Dictionary linking 
        param_dict = {
            'slope': self.vis_slope,
            'flow_acc': self.vis_flow_accum,
            'masked_dnbr': self.vis_dNBR,
            'dnbr': self.vis_dNBR,
            'i12_crit': self.vis_i12_crit,
            'num_events': self.vis_num_debris_flow_events,
            'aridity': self.vis_aridity,
            'erosion': self.vis_erosion,
            'delivered': self.vis_delivered,
            'dem': self.vis_DEM,
            clay_mass_fmt: self.vis_debris_mass,
            'plain': default_params
            }
        
        # Return the vis_params attribute if the input string matches:
        for key, value in param_dict.items():
            if key in input_string:
                return value
        
        logger.info(
            'Visualisation parameters not found for '
            f'{file_or_col_name}. Falling back to defaults.'
            )
        return default_params
        


    ###########################################################################
    def load_name_defaults(self):
        """
        Load useful default field names to be accessed later
        ----------------------------------------------------------------
        Notes:
        - This may no longer be needed with the const.py module
        ----------------------------------------------------------------
        """
        # ID fields:
        self.headwater_id = const.HW_ID
        self.subcatchment_id = const.SC_ID

    ###########################################################################
    def plot_catchment_raster(
        self,
        *args,
        catchment=None,
        existing_figure=None,
        axes_index=None,
        new_subplot:bool=True
        ):
        """
        Plot the requested raster for catchment(s)

        Parameters:
        - *args:
        - catchment (str): name of the catchment to
        plot. If none, each catchment in the current project will be
        plotted.
        - figure (mpl.figure): matplotlib figure object within
        which all the plots will be created. This function will create
        one if not provided.
        - axes_index (int): Of the axes objects that belong to this
        figure (if provided), the index of the one to draw on.
        - new_subplot (bool): whether a new subplot needs to be created
        as part of this call
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # If a figure object is not provided, create an empty one:
        if existing_figure is None:
            from matplotlib import pyplot as plt
            figure = plt.figure()
            # Case where neither figure nor subplot are provided:
            if axes_index is None:
                # If no figure was provided, we also need a subplot index
                #to provide later:
                axes_index = 0
        else:
            # Use the existing figure if it's been provided:
            figure = existing_figure
            if axes_index is None:
                # If an axes index isn't provided, use the end one:
                axes_index = len(figure.axes) - 1
            # If they have provided an axes index but have also
            #requested a new subplot, display a warning:
            else:
                if new_subplot:
                    logger.warning(
                        'project.plot_catchment_raster() received axes '
                        f'{axes_index} but a new subplot was also '
                        'requested via new_subplot=True. This is '
                        'contradictory and will most likely produce '
                        'an undseired result, like plots partially '
                        'overlapping each other.'
                        )


        # If a catchment has not been specified, create a subplot for
        #each catchment in the current project:
        if catchment is None:
            figure.subplots(
                nrows=len(self.catchments), #One subplot per catchment
                ncols=1 #Stack vertically
                )
            # Get a -something- for each catchment in the project
            self.for_each_catchment(
                lambda c:self.plot_catchment_raster(
                    *args,
                    catchment=c,
                    existing_figure=figure,
                    axes_index=self.catchments.index(c),
                    new_subplot=False
                    )
                )
            return
        # If catchment is not none we need to add a subplot:
        else:
            if new_subplot:
                # Get the number of subplots already:
                num_subs_already = len(figure.axes)
                figure.add_subplot(
                    num_subs_already + 1, #Num rows, add one to existing
                    1, #Num cols
                    num_subs_already + 1 #index, last row
                    )
        import rasterio as rio
        import os
        import numpy as np

        raster_path = self.catchment_path(catchment,*args)
        if not raster_path.endswith('.tif'):
            raster_path += '.tif'

        gdf = self.catchment_boundary(catchment)

        file_name = args[-1]

        vis_params = self.get_vis_params(file_name)

        useful_filename_part = file_name.split('.')[0].lower()
        if 'erosion' in file_name:
            title = toputil.get_erosion_title(
                useful_filename_part, 'erosion'
                )
            vis_params['title_varname'] = title
        elif 'delivered' in file_name:
            title = toputil.get_erosion_title(
                useful_filename_part, 'delivered'
                )
            vis_params['title_varname'] = title
            
        catch_name = toputil.clean_chart_title(catchment)
        chart_title = catch_name + ': ' + vis_params['title_varname']

        ax = figure.axes[axes_index]
        img, this_crs, cbar = toputil.plot_spatial_raster(
            ax,
            raster_path,
            vis_params,
            title=chart_title,
            colourbar=True,
            clip_geometry=gdf
            )

        # Get the coordinate reference of the raster so we can
        #extract relevant info
        if this_crs.is_projected:
            these_units = this_crs.linear_units + 's'
        elif this_crs.is_geographic:
            # Assumes degrees are the only relevant angular unit:
            these_units = 'degrees'

        # Aesthetics:
        toputil.mapify_axes(ax, this_crs, these_units)

        # Add the catchment boundary:
        plot_catchment_boundary(self, catchment, ax)

        return

    ###########################################################################
    def get_saved_data(
        self,
        catchment:str,
        type:str|None,
        name:str,
        format:str='csv'
        ) -> pd.DataFrame:
        """
        Get a file saved within the catchment's folder structure
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Get the subfolder for the requested catchment:
        if type is None:
            data_table_loc = self.catchment_path(catchment)
        # If a data type has been requested e.g. DebrisFlow, go to that
        #subfolder:
        else:
            data_table_loc = self.catchment_path(catchment, type)
        # Get the actual path the the file: 
        data_table_path = os.path.join(
            data_table_loc,
            name
            ) + '.' + format
        
        # Try reading in the csv:
        df = pd.read_csv(data_table_path)

        return df
        

    ###########################################################################
    def plot_catchment_polygons(
        self,
        catchment:str,
        polygons:gpd.GeoDataFrame,
        colour_col:str,
        vis_params:dict,
        title:str,
        non_geo_data:pd.DataFrame | None=None,
        id_col:str | None=None,
        existing_figure=None,
        existing_axes=None
        ):
        """
        Plot catchment polygons, 
        optionally coloured by a specific column.
        ----------------------------------------------------------------
        Notes:
        - individual methods (headwaters, subcatchments) should get the 
        relevant geodataframe and non-spatial data, and join them 
        together
        - Then they should call this method to do the actual plotting.
        ----------------------------------------------------------------
        """

        # Work out which figure/axes to use:
        fig, ax = toputil.fig_ax_admin(existing_figure, existing_axes)

        # Call the vector plotting function:
        this_crs, cbar, ax = toputil.plot_spatial_vector(
            ax,
            polygons,
            vis_params,
            title,
            symbol_data=non_geo_data,
            id_col_name=id_col,
            data_col_name=colour_col
            )

        

        # Set a grey background for plots to aid readbility:
        ax.set_facecolor('#D3D3D3')

        # Add scalebar or ticks as appropriate:
        these_units = this_crs.axis_info[0].unit_name
        toputil.mapify_axes(ax, this_crs, these_units)
        # Add the catchment boundary:
        plot_catchment_boundary(self, catchment, ax)

    ###########################################################################
    def plot_headwaters(
        self,
        catchment:str,
        colour_col:str|None=None,
        table:pd.DataFrame | None=None,
        data_type:str='DebrisFlow',
        existing_figure=None,
        existing_axes=None
        ):
        """
        Plot the headwaters coloured by a specified data value

        Parameters:
        - catchment (str): name of the catchment within the current
        project
        - colour_col (str): name of the column in the .csv file which
        is to be used to colour the headwaters
        - table (pd.DataFrame): OPTIONAL: DataFrame containing the
        data to plot. If not provided, the function will attempt to
        load a data table from file.
        - data_type (str): name of the output type, which will generally
        be DebrisFlow but could be RUSLE or similar later
        - data_format (str): three-letter extension relevant to the file
        type being read for the non-spatial data.
        - existing figure (mpl.figure): matplotlib figure object to
        include the new chart on, if desired. One will be created if
        not.
        - existing axes (mpl.axes): matplotlib axes object to plot the
        new data onto, if desired. One will be created if not.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Get the headwater polygons:
        headwaters_gdf = self.get_headwaters(catchment)

        if data_type == 'DebrisFlow':
            data_folder = data_type
            data_file_name = 'DebrisFlowData'
        else:
            data_folder = None
            data_file_name = 'Soil_Slope_Aridity_dNBR_headwaters'
        
        # If no colour column or data table is provided, skip data
        # loading entirely and render plain shapes.
        if colour_col is None and table is None:
            non_geo_data = None
        else:
            non_geo_data = self.get_table_safely(
                colour_col=colour_col,
                data_type=data_folder,
                data_file=data_file_name,
                catchment=catchment,
                allow_basic=False,
                table=table
                )

        if non_geo_data is not None and colour_col in non_geo_data.columns:
            # Get a subset of just the ID coloumn and the colour column:
            id_col = self.headwater_id
            ng_for_join = non_geo_data[[id_col, colour_col]]
            actual_colour_col = colour_col
            column_for_title = colour_col
        else:
            id_col = None
            ng_for_join = None
            actual_colour_col = None
            column_for_title = '(plain)'
        if colour_col is not None:
            vis_params = self.get_vis_params(colour_col)
        else:
            vis_params = self.get_vis_params('Plain')

        ax_title = toputil.make_axes_title(
            catchment,
            'Headwaters',
            vis_params['title_varname'],
            column_for_title
            )
        
        self.plot_catchment_polygons(
            catchment=catchment,
            polygons=headwaters_gdf,
            colour_col=actual_colour_col,
            vis_params=vis_params,
            title=ax_title,
            non_geo_data=ng_for_join,
            id_col=id_col,
            existing_figure=existing_figure,
            existing_axes=existing_axes
            )

    ###########################################################################
    def plot_subcatchments(
        self,
        catchment: str,
        colour_col: str,
        data_type: str | None = None,
        data_file: str | None = None,
        table: pd.DataFrame | None = None,
        existing_figure=None,
        existing_axes=None
        ):
        """
        Plot subcatchment polygons coloured by a specified data column.

        Parameters:
        - catchment (str): name of the catchment in the current project
        - colour_col (str): column name to use for polygon colouring
        - data_type (str): OPTIONAL subfolder name under the catchment
          directory where the data CSV lives (e.g. 'Results',
          'DebrisFlow'). Auto-detected from colour_col if not given.
        - data_file (str): OPTIONAL CSV file name (without extension).
          Auto-detected from colour_col if not given.
        - table (pd.DataFrame): OPTIONAL pre-loaded data table. Skips
          file loading if provided.
        - existing_figure: matplotlib figure to plot onto
        - existing_axes: matplotlib axes to plot onto
        ----------------------------------------------------------------
        Notes:
        - Auto-detection rules (applied when data_file is not supplied):
            - colour_col contains 'erosion' or 'delivered':
              reads rusle_subcatchment_summary.csv from Results/
            - colour_col contains 'events', 'debris', 'mass', or
              'i12': reads DebrisFlowData_subcatchments.csv from
              DebrisFlow/
            - otherwise: reads Soil_Slope_Aridity_dNBR_subcatchments
              from the catchment root (original behaviour)
        - Shorthand column names are accepted:
            - 'erosion_y1' resolves to 'erosion_y1_sum'
            - 'peak_erosion_y1' resolves to 'peak_erosion_y1_mean'
            - 'mass' resolves to the debris mass column
              (const.DEBRIS_MASS_FIELD)
        - Calling convention: (catchment, data_folder, colour_col)
          is also accepted as a positional form for backwards
          compatibility, e.g.
          plot_subcatchments(name, 'DebrisFlow', 'Year1_num_events')
        ----------------------------------------------------------------
        """
        # Support positional calling convention
        # (catchment, folder_name, colour_col). If colour_col looks
        # like a data folder rather than a column name and data_type
        # has been provided, the user probably passed them in the old
        # order — swap them:
        _known_folders = {
            'DebrisFlow', 'Results', 'Topography', 'Soils',
            'Erodibility', 'Delivery', 'Subcatchments'
            }
        if colour_col in _known_folders and data_type is not None:
            colour_col, data_type = data_type, colour_col

        # Auto-detect data_file from colour_col when not supplied.
        # data_type is also set here if not already provided:
        if data_file is None and table is None:
            col_lower = colour_col.lower()
            if any(k in col_lower for k in ('erosion', 'delivered')):
                # RUSLE and sediment delivery outputs — produced by
                # aggregate_rusle_to_subcatchments(), in Results/:
                if data_type is None:
                    data_type = const.RESULTS_FOLDER_NAME
                data_file = const.RUSLE_SC_SUMMARY_NAME
            elif any(k in col_lower
                     for k in ('events', 'debris', 'mass', 'i12')):
                # Debris flow outputs — produced by
                # aggregate_debris_flow_summary_to_subcatchments(),
                # in DebrisFlow/:
                if data_type is None:
                    data_type = 'DebrisFlow'
                data_file = const.DEBRIS_SC_SUMMARY_NAME
            else:
                # Fall back to soil/slope/aridity summary:
                if data_type is None:
                    data_type = ''
                data_file = 'Soil_Slope_Aridity_dNBR_subcatchments'

        subcatch_gdf = self.get_subcatchments(catchment)

        # Get the non-spatial data
        non_geo_data = self.get_table_safely(
            colour_col=colour_col,
            data_type=data_type,
            data_file=data_file,
            catchment=catchment,
            allow_basic=True,
            table=table
            )

        # Resolve shorthand column names before looking up the data:
        if non_geo_data is not None:
            # 'mass' → the actual debris mass delivery column:
            if colour_col.lower().strip() == 'mass':
                colour_col = const.DEBRIS_MASS_FIELD
            # Bare column names → append the default aggregation
            # suffix. Rules:
            #   i12 columns   → _min (most vulnerable headwater)
            #   peak rasters  → _mean
            #   total rasters → _sum
            if colour_col not in non_geo_data.columns:
                _cl = colour_col.lower()
                if 'i12' in _cl:
                    suffix = '_min'
                elif 'peak' in _cl:
                    suffix = '_mean'
                else:
                    suffix = '_sum'
                candidate = colour_col + suffix
                if candidate in non_geo_data.columns:
                    logger.info(
                        'Column %s not found; resolving to %s.',
                        colour_col, candidate
                        )
                    colour_col = candidate

        id_col = self.subcatchment_id
        if non_geo_data is not None:
            ng_for_join = non_geo_data[[id_col, colour_col]]
        else:
            ng_for_join = None

        vis_params = self.get_vis_params(colour_col)

        # Copy before modifying — vis_params dicts are shared instance
        # attributes and must not be mutated in-place:
        vis_params = vis_params.copy()

        # Set title and units for erosion/delivery columns. The
        # aggregation suffix on the column name (_sum or _mean) tells
        # us exactly what was computed, so we can label it precisely:
        col_lower = colour_col.lower()
        if 'erosion' in col_lower or 'delivered' in col_lower:
            var_type = (
                'Erosion' if 'erosion' in col_lower else 'Delivered'
            )
            year = (
                'Year 1' if 'y1' in col_lower
                else 'Year 2' if 'y2' in col_lower
                else ''
            )
            if colour_col.endswith('_mean'):
                # Peak rasters: each cell stores the max 30-min value.
                # Zonal stat is mean across cells — 'mean tonnes per
                # cell' distinguishes it from a catchment total:
                agg = 'Peak 30-min'
                vis_params['units'] = 'mean peak tonnes per cell'
            else:
                # Total rasters: each cell stores cumulative tonnes.
                # Zonal stat is a sum — i.e. total tonnes eroded
                # within the subcatchment:
                agg = 'Total'
                vis_params['units'] = 'total tonnes'
            vis_params['title_varname'] = (
                f'{agg} {var_type} {year}'.strip()
            )

        elif 'i12' in col_lower:
            # I12 threshold columns: suffix tells us min or mean.
            # Year is encoded as 'year_1' / 'year_2' in the col name:
            year = (
                'Year 1' if 'year_1' in col_lower
                else 'Year 2' if 'year_2' in col_lower
                else ''
            )
            stat_desc = (
                'Min' if col_lower.endswith('_min') else 'Mean'
            )
            vis_params['title_varname'] = (
                f'{stat_desc} I12 Threshold {year}'.strip()
            )
            # Units are already 'mm/hr' from vis_i12_crit — correct.

        # Build the axes title. When title_varname is fully specified,
        # use it directly — make_axes_title sniffs year/agg from the
        # column name and would duplicate them for suffixed columns
        # like 'peak_erosion_y1_mean'. Fall back to make_axes_title
        # only for columns with no recognised title_varname:
        if vis_params.get('title_varname'):
            catch_label = toputil.clean_chart_title(catchment)
            ax_title = (
                f'{catch_label} Subcatchments: '
                f'{vis_params["title_varname"]}'
            )
        else:
            ax_title = toputil.make_axes_title(
                catchment,
                'Subcatchments',
                vis_params['title_varname'],
                colour_col
                )

        self.plot_catchment_polygons(
            catchment=catchment,
            polygons=subcatch_gdf,
            colour_col=colour_col,
            vis_params=vis_params,
            title=ax_title,
            non_geo_data=ng_for_join,
            id_col=id_col,
            existing_figure=existing_figure,
            existing_axes=existing_axes
            )
            
    ###########################################################################
    def get_table_safely(
        self,
        colour_col:str,
        data_type:str,
        data_file:str,
        catchment:str,
        allow_basic:bool,
        table:pd.DataFrame | None=None
        ):
        """
        Perform basic sanity checks and retrieve a non-spatial table 
        for use when plotting polygons like headwaters or subcatchments.
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        if table is None:
            # Try to get the data using the specified data type/file:
            try:
                non_geo_data = self.get_saved_data(
                    catchment=catchment,
                    type=data_type,
                    name=data_file
                    )
            # If nothing is found we'll still plot the subcatchments 
            #but all the same colour:
            except FileNotFoundError:
                if allow_basic:
                    logger.info(
                        'Plotting polygons was requested with no '
                        'data to colour the shapes with. Proceeding '
                        'to plot boundaries with uniform '
                        'colours.'
                        )
                    non_geo_data = None
                else:
                    raise
        else:
            non_geo_data = table
        # Raise an error if there is data available but the required 
        #column is not there:
        if non_geo_data is not None:
            if colour_col not in non_geo_data.columns:
                logger.warning(
                    'project.plot_subcatchments() was asked to colour '
                    f'the map based on {colour_col}, but data table '
                    f'only had the following:\n {non_geo_data.columns}. '
                    'plotting will proceed but with all symbols the ' \
                    'same.'
                    )
        return non_geo_data

    ###########################################################################
    def thresh_sev_scatter(
        self,
        catchment:str,
        existing_figure = None,
        existing_axes = None,
        width=12,
        height=8,
        dpi=600
        ):
        """
        Produce a scatter plot of year 1 and year 2 critical rainfall
        intensity thresholds (x-axis) vs. mean dNBR (y-axis) for each
        headwater.

        Parameters:
        - catchment (str): name of the catchment within the current
        project
        - existing figure (mpl.figure): matplotlib figure object to
        include the new chart on, if desired. One will be created if
        not.
        - existing axes (mpl.axes): matplotlib axes object to plot the
        new data onto, if desired. One will be created if not.
        - width (numeric): desired width of the figure in inches
        - height (numeric): desired height of the figure in inches
        - dpi (int): desired resolution of the figure in dots per inch
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        # Get the location of the debris flow data for the current
        #catchment:
        folder = self.catchment_path(catchment, 'DebrisFlow')
        file = 'DebrisFlowData.csv'
        path = os.path.join(folder, file)

        # Make sure it actually existsa and if not, throw an error:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                'project.thresh_sev_scatter() requires debris flow data '
                'to be loaded. Run debris.debris_flow() which will save '
                f'the required data in a csv here:\n{path}'
            )
        non_geo_data = pd.read_csv(path)
        # Prepare the data for plotting:
        x1_col = 'I12_crit_mean_Year_1'
        x2_col = 'I12_crit_mean_Year_2'
        y_col = 'dNBR_mean'
        data_for_scatter = non_geo_data[[x1_col, x2_col, y_col]].dropna()

        # Get median values for axes lines:
        median_x1_col = data_for_scatter[x1_col].median()
        median_x2_col = data_for_scatter[x2_col].median()
        median_y_col = data_for_scatter[y_col].median()

        # Colours for each years' data:
        col_year_1 = '#800080' #purple
        col_year_2 = '#696969' #grey

        sfig, sax = toputil.fig_ax_admin(existing_figure, existing_axes)

        # Set size and resolution parameters for figure:
        sfig.set_size_inches(width, height)
        sfig.set_dpi(dpi)

        # Plot year 1 and then year 2 values:
        splot1 = sax.scatter(
            x=data_for_scatter[x1_col],
            y=data_for_scatter[y_col],
            marker='x',
            color=col_year_1,
            label='Year 1'
            )
        splot2 = sax.scatter(
            x=data_for_scatter[x2_col],
            y=data_for_scatter[y_col],
            marker='o',
            color=col_year_2,
            label='Year 2'
        )
        sax.set_ylim(0, 1000)

        # Vertical lines for critical rainfall threshold medians for
        #each year:
        x1_col_med = sax.axvline(
            x=median_x1_col,
            label='I12 crit. rain threshold: y1 median',
            ls='--',
            c=col_year_1
            )
        x2_col_med = sax.axvline(
            x=median_x2_col,
            label='I12 crit. rain threshold: y2 median',
            ls='--',
            c=col_year_2
            )
        # Horizontal line for dNBR median (will be 0 inles)
        y_col_med = sax.axhline(
            y=median_y_col, label=f'dNBR median', ls=':', c='grey'
            )

        # Aesthetics:
        sax.set_title(
            'Scatter plot of mean dNBR vs year 1 critical rainfall '
            f'for {toputil.clean_chart_title(catchment)} headwaters'
            )
        sax.set_xlabel(
            'I12 critical threshold for debris flow'
            )
        sax.set_ylabel('Mean dNBR')

        # Add legend:
        this_leg = sax.legend(
            loc='upper left',
            bbox_to_anchor=(1.0, 1.0)
            )
        return sfig
    
    ###########################################################################
    def get_fire_end_date(self, catchment):
        """
        Get the end date of the fire for a oarticular catchment as a 
        pandas datetime
        ----------------------------------------------------------------
        ----------------------------------------------------------------
        """
        fire_meta_path = self.catchment_path(
            catchment,
            const.FIRE_SEVERITY_FOLDER_NAME,
            'FireMeta.csv'
            )
        fire_meta = pd.read_csv(fire_meta_path, index_col=0)
        end_date_iso = fire_meta.loc['end_date','Value']
        return pd.to_datetime(end_date_iso)


###########################################################################
def plot_catchment_boundary(
    project:FireImpactsProject,
    catchment:str,
    axes,
    new_legend=True
    ):
    """
    Plot the the catchment boundary on an axes object and add a
    a legend
    --------------------------------------------------------------------
    TODO: Move this to top util and use plot_spatial_vector() for most
    steps.
    For now, We'll keep this separate as the way it plots and the way
    it gets the data are both somewhat different.
    --------------------------------------------------------------------
    """
    # Set the colour for the line:
    catch_bound_colour = 'red'
    # Get the actual boundary:
    gdf = project.catchment_boundary(catchment)
    # Plot on the provided axes:
    gdf.plot(ax=axes, facecolor='none', edgecolor=catch_bound_colour)
    # Dummy line for legend:
    dummy_line = [
        mlines.Line2D(
            [], #Empty x-data
            [], #Empty y-data
            color=catch_bound_colour
            )
        ]
    if new_legend:
        this_leg = axes.legend(
            dummy_line,
            ['Catchment Boundary'], #Legend label
            fontsize='xx-small'
            )

###############################################################################
def get_vis_dx(ax, crs):
    """
    Return an appropriate visualisation dx value (map units per pixel)
    to help in making a scalebar
    --------------------------------------------------------------------
    Notes:
    - Probably only required if for some reason extent property hasn't
    already been set either by matplotlib or geopandas.
    --------------------------------------------------------------------
    """
    if not crs.is_projected:
        logger.warning(
            'get_vis_dx only accepts projected crs objects currently. '
            'dx not returned.'
            )
        return None
    # Get the width of the current axes in the map's units:
    x_range = ax.get_xlim()
    x_min = x_range[0]
    x_max = x_range[1]
    ax_width_map_units = x_max - x_min

    # Get the width of the current axes in pixels:
    ax_bbox_pixels = ax.get_window_extent()
    ax_width_px = ax_bbox_pixels.width

    # Calculate map units per pixel:
    map_units_per_pixel = ax_width_map_units / ax_width_px
    return map_units_per_pixel

def find_all_shapefiles(base_directory):
    '''
    Find all shapefiles in a directory and its subdirectories.
    '''
    assert os.path.isdir(base_directory), f"Directory not found: {base_directory}"
    shapefiles = glob(os.path.join(base_directory, '**','*.shp'),recursive=True)
    return shapefiles

###############################################################################
def summary_stats(
    project:FireImpactsProject,
    catchment_name=None,
    zone_type='headwaters'
    ):
    """
    Calculate summary statistics for a catchment from pre-processed 
    raster data.

    Parameters:
    - project (FireImpactsProject): Project object containing the 
    catchment data.
    - catchment_name (str): Name of the catchment to process. If not 
    provided, process all catchments in the project.

    Returns:
    - pd.DataFrame: DataFrame containing the summary statistics for the 
    catchment (if catchment_name is provided), OR
    - dict: Dictionary of DataFrames containing the summary statistics 
    for each catchment.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Check that we've been asked for something we can do:
    acceptable_zones = ['headwaters', 'subcatchments']
    requested_zone = zone_type.strip().lower()
    if requested_zone not in acceptable_zones:
        raise ValueError(
            'project.summary_stats() was asked to compute stats for '
            f'{zone_type}. This is not currently coded for, please use '
            f'one of: {acceptable_zones}'
            )
    # If we've been given a string instead of an actual project object,
    #try initialising/loading a project with the given name:
    if isinstance(project,str):
        project = FireImpactsProject(project)
    # Process for all catchments if none was specified:
    if catchment_name is None:
        return project.for_each_catchment(lambda c:summary_stats(project,c))
    
    if requested_zone == 'subcatchments':
        id_col_name = project.subcatchment_id
        zones_gdf = project.get_subcatchments(catchment_name)
    else:
        id_col_name = project.headwater_id
        headwaters_path = project.catchment_path(
            catchment_name,
            'Topography',
            'Headwaters.shp'
            )
        zones_gdf = gpd.read_file(headwaters_path)

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
        id_col_name: zones_gdf[id_col_name]
    }

    logger.info('Processing %d polygons for %d layers in %s',len(zones_gdf),len(sources),catchment_name)
    for label, path in sources:
        logging.info('Processing %s from %s',label,path[-1])
        stats = toputil.get_zonal_stats(
            zones_gdf,
            project.catchment_path(catchment_name,*path),
            label
            )
        for k in STATS:
            result[f'{label}_{k}'] = [s[k] for s in stats]

    extracted_data = pd.DataFrame(result)

    # Convert dNBR values to a set of standardised numbers [0, 1000]:
    for stat in STATS:
        this_col_name = 'dNBR_' + stat
        extracted_data[this_col_name] = format_dNBR(
            extracted_data[this_col_name]
            )


    csv_path=project.catchment_path(
        catchment_name,
        f'Soil_Slope_Aridity_dNBR_{zone_type}.csv'
        )
    extracted_data.to_csv(csv_path, index=False)

    return extracted_data

###############################################################################
def format_dNBR(series:pd.Series):
    """
    Convert dNBR values between -1 and 1, to standardised values
    between 0 and 1000. Values below 0 are set to 0.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    return series.clip(lower=0).mul(1000)

###############################################################################
def save_catchment_raster(
    project:FireImpactsProject,
    catchment_name:str,
    file_name:str,
    section:str,
    data,
    meta
    ):
    """
    
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Build the path:
    out_path = project.catchment_path(
        catchment_name,
        section,
        f'{file_name}.tif'
        )
    
    # Set standard data type parameters for all our rasters:
    final_meta = meta.copy()
    in_dtype = final_meta['dtype']
    in_dtype_kind = np.dtype(in_dtype).kind
    out_dtype = default_dtypes_raster[numpy_kind_to_desc[in_dtype_kind]]

    # Update the metadata with the final dtype
    final_meta.update(dtype=out_dtype, count=1)

    # Try writing the file. Return True with location if successful, 
    #or false with the error message if not.
    try:
        with rio.open(out_path, 'w', **final_meta) as dst:
            dst.write(data.astype(out_dtype), 1)
        result = True
        result_string = f'Saved raster to {out_path}'
    except Exception as e:
        result = False
        result_string = f'Could not save raster: {e}'
    
    return result, result_string
    






