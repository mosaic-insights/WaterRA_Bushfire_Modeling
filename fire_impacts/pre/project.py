'''
This module contains the classes and functions that are used to manage the data for the fire_impacts module.
'''

import os
from glob import glob
from pathlib import Path
import shutil
import rasterio as rio
import rasterstats as rs
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
logger = logging.getLogger(__name__)

# These are the default directories that need to exist inside every 
#catchments directory:
PER_CATCHMENT_FOLDERS = [
    'Topography',
    'FireSeverity',
    'Soils',
    'Erodibility',
    'Delivery',
    'Subcatchments'
]

STATS=['mean', 'max', 'min', 'median', 'std']
APPROX_KM_PER_DEGREE = 111  # Approximate conversion factor from degrees to kilometers


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
        out_gdf = subcatch_clipped[good_cols]
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
        if clear and os.path.exists(project_path):
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
        # Assume there is a shapefile with all the subcatchment 
        #boundaries in the Topography folder for the current project:
        shapefile_path = self.catchment_path(
            catchment,
            'Subcatchments',
            'Subcatchments.shp'
            )
        
        # If there is a subcatchments shapefile, load it as a 
        #GeoDataFrame and return it:
        if os.path.exists(shapefile_path):
            gdf = gpd.read_file(shapefile_path)
            return gdf
        # Otherwise return None
        else:
            logger.warning(
                'Subcatchment boundaries were requested from '
                f'project.get_subcatchments() for {catchment}, '
                'but they appear not to be loaded yet. Use '
                'project.add_subcatchments() first.'
                )
            return None

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
            'title_varname': 'Rain Intensity',
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

        useful_filename_part = args[-1].split('.')[0].lower()

        if useful_filename_part == 'dem':
            vis_params = self.vis_DEM
        elif useful_filename_part == 'slope':
            vis_params = self.vis_slope
        elif useful_filename_part == 'flow_accumulation':
            vis_params = self.vis_flow_accum
        elif useful_filename_part == 'aridity':
            vis_params = self.vis_aridity
        elif useful_filename_part == 'dnbr':
            vis_params = self.vis_dNBR
        elif useful_filename_part == 'erosion_y1':
            vis_params = self.vis_erosion
            vis_params['title_varname'] = 'Total Erosion Year 1'
        elif useful_filename_part == 'erosion_y2':
            vis_params = self.vis_erosion
            vis_params['title_varname'] = 'Total Erosion Year 2'
        elif useful_filename_part == 'peak_erosion_y1':
            vis_params = self.vis_erosion
            vis_params['title_varname'] = 'Peak 30-min Erosion Year 1'
        elif useful_filename_part == 'peak_erosion_y2':
            vis_params = self.vis_erosion
            vis_params['title_varname'] = 'Peak 30-min Erosion Year 2'
        else:
            vis_params = {
                'cmap': 'viridis',
                'measure': 'Undefined',
                'units': 'n/a',
                'norm': None,
                'cbar_extend': 'neither',
                'title_varname': ''
                }
            
        catch_name = toputil.clean_chart_title(catchment)
        chart_title = catch_name + ': ' + vis_params['title_varname']



        ax = figure.axes[axes_index]
        img, this_crs, cbar = toputil.plot_spatial_raster(
            ax,
            raster_path,
            vis_params,
            title=chart_title,
            colourbar=True
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
    def plot_headwaters(
        self,
        catchment:str,
        colour_col:str,
        table:pd.DataFrame=None,
        data_type:str='DebrisFlow',
        data_format:str='csv',
        existing_figure=None,
        existing_axes=None,
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
        # Check that headwaters have been computed. If not, raise error:
        hw_shape_loc = self.catchment_path(
            catchment,
            'Topography',
            )
        hw_shape_path = os.path.join(hw_shape_loc, 'Headwaters.shp')
        if not os.path.isfile(hw_shape_path):
            raise FileNotFoundError(
                'project.plot_headwaters() was called but Headwaters.shp '
                f'does not exist in the Topography folder for {catchment}. '
                'Run topography.extract_headwaters() first for the current '
                'catchment.'
            )

        if table is None:
            # Check that the data table that has been passed already exists:
            data_table_loc = self.catchment_path(catchment, data_type)
            data_table_path = os.path.join(
                data_table_loc,
                data_type
                ) + 'Data.' + data_format

            if not os.path.isfile(data_table_path):
                raise FileNotFoundError(
                    'project.plot_headwaters() was called requeting to '
                    f'plot data from {data_type} as the variable. Full '
                    f'path checked for was {data_table_path}.'
                )

            # Get the data table path and check that the requested column
            #exists:
            non_geo_data = pd.read_csv(data_table_path)
        else:
            non_geo_data = table

        if colour_col not in non_geo_data.columns:
            raise ValueError(
                'project.plot_headwaters() was asked to colour the map '
                f'based on {colour_col}, but data table only had the '
                f'following:\n {non_geo_data.columns}'
            )
        # Get a subset of just the ID coloumn and the colour column:
        id_col = 'ID'
        ng_for_join = non_geo_data[[id_col, colour_col]]

        # Choose visualisation parameters based on the colour column:
        if colour_col[:4].lower() == 'dnbr':
            vis_params = self.vis_dNBR
        elif colour_col[:8].lower() == 'i12_crit':
            vis_params = self.vis_i12_crit
        elif colour_col[-10:].lower() == 'num_events':
            vis_params = self.vis_num_debris_flow_events
        # Default parameters fallback:
        else:
            vis_params = {
                'cmap': 'inferno',
                'measure': 'Undefined',
                'units': 'n/a',
                'norm': None,
                'cbar_extend': 'neither',
                'title_varname': '-'
            }

        # Format the chart title to use the colour column in some way:
        #TODO: This needs to be more dynamic, currently I'm just 
        #building it to handle the cases I've seen so far.
        non_under = colour_col.split('_')
        if len(non_under) == 2:
            var_qual = non_under[1].title()
        elif len(non_under) > 2:
            var_qual = non_under[0].title()
        else:
            var_qual = ''

        # Generate a relevant title:
        varname = (
            vis_params['title_varname']
            + ' '
            + var_qual
            ).strip() # Variable name part of title
        catch_title = toputil.clean_chart_title(catchment)
        ax_title = (
            f'{catch_title} Headwaters: {varname}'
            )

        # Call the vector plotting function:
        this_crs, cbar, existing_axes = toputil.plot_spatial_vector(
            existing_axes,
            hw_shape_path,
            vis_params,
            ax_title,
            symbol_data=ng_for_join,
            id_col_name=id_col,
            data_col_name=colour_col
            )

        # Work out which figure/axes to use:
        fig, ax = toputil.fig_ax_admin(existing_figure, existing_axes)

        # Handle whether a catchment is specified and if not, plot for all
        #catchments just like plot_catchment_rasters():
        # TODO: add this code

        # Set a grey background for headwater plots to aid readbility:
        ax.set_facecolor('#D3D3D3')


        # Add scalebar or ticks as appropriate:
        these_units = this_crs.axis_info[0].unit_name
        toputil.mapify_axes(ax, this_crs, these_units)
        # Add the catchment boundary:
        plot_catchment_boundary(self, catchment, ax)

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
        y_col = 'dNBR_mean_fmtd'
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

    # Convert dNBR values to a set of standardised numbers [0, 1000]:
    for stat in STATS:
        this_col_name = 'dNBR_' + stat
        new_col_name = this_col_name + '_fmtd'
        extracted_data[new_col_name] = format_dNBR(extracted_data[this_col_name])


    csv_path=project.catchment_path(catchment_name, 'Soil_Slope_Aridity_dNBR.csv')
    extracted_data.to_csv(csv_path, index=False)

    return extracted_data


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

###############################################################################
def format_dNBR(series:pd.Series):
    """
    Convert dNBR values between -1 and 1, to standardised values
    between 0 and 1000. Values below 0 are set to 0.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    return series.clip(lower=0).mul(1000)






