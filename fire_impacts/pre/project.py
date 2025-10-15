'''
This module contains the classes and functions that are used to manage the data for the fire_impacts module.
'''

import os
import re
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
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

PER_CATCHMENT_FOLDERS = [
    'Topography',
    'FireSeverity',
    'Soils',
    'Erodibility',
    'Delivery'
]

STATS=['mean', 'max', 'min', 'median', 'std']
APPROX_KM_PER_DEGREE = 111  # Approximate conversion factor from degrees to kilometers

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
        
        TODO:
        - make this more resilient to different formats of paths used by
        the OS and by the user's input.
        '''
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
              self.initialise_project(norm_path,exist_ok=exist_ok,clear=clear)

        self.load_vis_defaults()


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
        """
        Docstring placeholder
        """
        # If there is already a folder and the user has said NOT to 
        #clear it:
        if not clear and os.path.exists(project_path):
            raise FileExistsError(f'Project folder already exists: {project_path}')
        # If there is already a folder and the user as said it's ok to 
        #clear its contents:
        if clear and os.path.exists(project_path):
            logger.info('Clearing existing project folder: %s',project_path)
            # Remove the directory and all of its contents:
            shutil.rmtree(project_path)
        os.makedirs(self.catchment_path(),exist_ok=exist_ok)
        self._write()

    def catchment_boundary(self,catchment:str) -> gpd.GeoDataFrame:
        '''
        Get the catchment boundary as a GeoDataFrame.
        '''
        shapefile_path = self.boundary_files[catchment]
        gdf = gpd.read_file(shapefile_path)
        return gdf

    def subcatchment_boundaries(self,catchment:str) -> gpd.GeoDataFrame:
        '''
        Get the subcatchment boundaries as a GeoDataFrame.
        '''
        shapefile_path = self.catchment_path(catchment,'Topography','Subcatchments.shp')
        if os.path.exists(shapefile_path):
          gdf = gpd.read_file(shapefile_path)
          return gdf

        return self.catchment_boundary(catchment)

    def catchment_bounds(self,catchment:str, buffer_distance_km:float=10):
        '''
        Get the bounding box for a catchment in WGS84 with an (optional) buffer distance in approximate kilometres.
        '''
        gdf = self.catchment_boundary(catchment)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        bbox = gdf_wgs84.total_bounds

        # Convert 10 km to degrees (approximate conversion, 1 degree = 111 km)
        buffer_degrees = buffer_distance_km / APPROX_KM_PER_DEGREE

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
        gdf = self.catchment_boundary(catchment)
        return gdf.crs

    def cell_area(self,catchment:str=None):
        if catchment is None:
            return self.for_each_catchment(lambda c:self.cell_area(catchment=c))

        fn = self.catchment_path(catchment,'Topography','DEM.tif')
        with rio.open(fn) as src:
            transform = src.transform
            return abs(transform.a * transform.e)

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
            'cbar_extend': 'neither'
        }

        self.vis_slope = {
            'cmap': 'viridis',
            'measure': 'Slope',
            'units': '°',
            'norm': None,
            'cbar_extend': 'neither'
        }

        self.vis_flow_accum = {
            'cmap': 'viridis',
            'measure': 'Contributing areas',
            'units': 'count',
            'norm': LogNorm(vmin=10),
            'cbar_extend': 'min'
        }

        self.vis_dNBR = {
            'cmap': 'inferno',
            'measure': 'ΔNBR',
            'units': 'n/a',
            'title_varname': 'ΔNBR'
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
                axes_index = len(figure.axes)

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

        if args[-1].split('.')[0] == 'DEM':
            vis_params = self.vis_DEM
        elif args[-1].split('.')[0].lower() == 'slope':
            vis_params = self.vis_slope
        elif args[-1].split('.')[0].lower() == 'flow_accumulation':
            vis_params = self.vis_flow_accum
        else:
            vis_params = {
                'cmap': 'viridis',
                'measure': 'Undefined',
                'units': 'n/a',
                'norm': None,
                'cbar_extend': 'neither'
                }

        with rio.open(raster_path) as src:
            data = src.read(1)
            no_data_value = src.nodata
            if no_data_value is not None:
                data = np.where(data == no_data_value, np.nan, data)  # Replace NoData values with NaN
            transform = src.transform

            
            
            file_name = os.path.splitext(os.path.basename(raster_path))[0].replace('_', ' ')
            ax = figure.axes[axes_index] 
            img = ax.imshow(
                data,
                cmap=vis_params['cmap'],
                norm=vis_params['norm'],
                extent=(
                    transform[2],
                    transform[2] + transform[0] * data.shape[1],
                    transform[5] + transform[4] * data.shape[0],
                    transform[5]
                    )
                )

            title = clean_chart_title(catchment)
            ax.set_title(f'{title}: {file_name}', fontsize='large')
            
            cbar = figure.colorbar(
                img,
                label=f'{vis_params['measure']} ({vis_params['units']})',
                extend=vis_params['cbar_extend']
                )
            
            # Add the catchment boundary:
            plot_catchment_boundary(self, catchment, ax)

            # Get the coordinate reference of the raster so we can
            #extract relevant info
            this_crs = src.crs
            if this_crs.is_projected:
                these_units = this_crs.linear_units + 's'
            elif this_crs.is_geographic:
                these_units = this_crs.angular_units + 's'
            
            # Aesthetics:
            mapify_axes(ax, this_crs, these_units)

            
            
            return

    ###########################################################################
    def plot_headwaters(
            self,
            catchment:str,
            colour_col:str,
            data_type:str='DebrisFlow',
            data_format:str='csv',
            existing_figure=None,
            existing_axes=None,
            ):
        """
        Plot the headwaters coloured by a specified data value
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
        if colour_col not in non_geo_data.columns:
            raise ValueError(
                'project.plot_headwaters() was asked to colour the map '
                f'based on {colour_col}, but data table only had the '
                f'following:\n {non_geo_data.columns}'
            )
        ng_for_join = non_geo_data[['ID', colour_col]]
        
        headwater_shapes = gpd.read_file(hw_shape_path)
        geom_with_data = pd.merge(
            headwater_shapes,
            ng_for_join,
            on='ID'
            )
        
        # Handle existing figures and/or subplots in a similar way to
        #plot_catchment_rasters():
        if existing_figure is None and existing_axes is None:
            fig, ax = plt.subplots()
        # Handle if figure is provided but no axes (check subplots):

        # If axes is provided with no figure that is fine.
        # If both a figure and an axes are provided, just use those:
        else:
            fig = existing_figure
            ax = existing_axes

        #

        # Handle whether a catchment is specified and if not, plot for all
        #catchments just like plot_catchment_rasters():

        if colour_col[:4].lower() == 'dnbr':
            vis_params = self.vis_dNBR
        else:
            vis_params = {
                'cmap': 'inferno',
                'measure': 'Undefined',
                'units': 'n/a',
                'norm': None,
                'cbar_extend': 'neither',
                'title_varname': '-'
            }

        # Then actually plot the headwaters based on the specified value.
        geom_with_data.plot(
            ax=ax,
            column=colour_col,
            cmap=vis_params['cmap']
            )
        
        # Set the axis limits and aspects so the scalebar works:
        
        # Normalise colormap applicator for the current data column:
        normer = plt.Normalize(
            vmin=geom_with_data[colour_col].min(),
            vmax=geom_with_data[colour_col].max()
        )
        # Add a mapper to create our colorbar with:
        mapper = mpl.cm.ScalarMappable(
            cmap=vis_params['cmap'],
            norm=normer
        )
        
        cbar = fig.colorbar(
            mapper,
            ax=ax,
            label=vis_params['measure'],
            extend='neither'
            )
        
        # Aesthetics:
        varname = (
            vis_params['title_varname'] 
            + ' ' 
            + colour_col.split('_')[1].title()
            )
        ax.set_facecolor('#D3D3D3')
        title = clean_chart_title(catchment)
        ax.set_title(f'{title} Headwaters: {varname}', fontsize='large')

        # Add scalebar or ticks as appropriate:
        this_crs = geom_with_data.crs
        these_units = this_crs.axis_info[0].unit_name
        mapify_axes(ax, this_crs, these_units)
        # Add the catchment boundary:
        plot_catchment_boundary(self, catchment, ax)

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

###############################################################################
def mapify_axes(
    ax,
    crs,
    units:str,
    ):
    """
    Settings for maps based on whether the data is in a projected or
    geographic coordinate system
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    
    if crs.is_projected:
        # No ticks for a projects CS, we'll use a scalebar
        #instead:
        ax.set_xticks([])
        ax.set_yticks([])
        from matplotlib_scalebar.scalebar import ScaleBar

        # Set the font size for the scalebar text
        sb_fontprops = {
            'size': 'xx-small'
            }

        these_units = units[0]
        # Create the scalebar object:
        this_scalebar = ScaleBar(
            dx=1, #size of one pixel
            units=these_units, #units of the pixel size
            loc='lower left',
            font_properties=sb_fontprops,
            box_alpha=0.5
            )
        # Plot the scalebar onto the map:
        ax.add_artist(this_scalebar)

    if crs.is_geographic:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

###############################################################################
def clean_chart_title(text):
    """
    Removes underscores, ending-EPSG codes, camel-case
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Remove trailing underscores etc. (EPSG code):
    int_title = re.sub(r'_\d+$', '', text)
    # Expand camel case to spaced words:
    int_title = re.sub(r'(?<!^)(?=[A-Z])', ' ', int_title)

    title = int_title.replace('_', ' ').strip()
    return title

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


