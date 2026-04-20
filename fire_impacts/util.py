import os
import re
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import rasterio as rio
import rasterstats as rs

import importlib
constants = importlib.import_module('fire_impacts.const')

STATS=constants.STATS

logger = logging.getLogger(__name__)

def retry(fn,retries=5,initial_delay=8,delay_scale=3,specific_exceptions=None):
    import time

    try:
        return fn()
    except Exception as e:
        if retries<=0:
            raise e

        if specific_exceptions is not None:
            if e.__class__ not in specific_exceptions:
                raise e

        logger.warning('Failed with %s. Retrying after %d seconds'%(str(e),initial_delay))
        time.sleep(initial_delay)
        return retry(fn,retries-1,initial_delay*delay_scale,delay_scale,specific_exceptions)

###############################################################################
def package_data_path(fn=None):
    """
    Point to where static lookup tables are currently stored in the 
    package and join them to a specified file name to produce a usable 
    path
    """
    dirname = os.path.join(os.path.dirname(__file__),'..','data')
    if fn is None:
        return dirname
    return os.path.join(dirname,fn)

###############################################################################
def load_package_data(fn):
    """
    For static package lookup tables, get the full path/filename.ext 
    and then make sure the output is a csv.
    """
    fn = package_data_path(fn)
    if fn.endswith('.csv'):
        logger.info(f"Loading data from {fn}")
        import pandas as pd
        return pd.read_csv(fn)
    logger.error(f"Unsupported file type: {fn}")
    return None

def file_matching_all(path,*substrings):
    """Check if a file contains all substrings and return a list of matches"""
    files = os.listdir(path)
    return [fn for fn in files if all(p in fn for p in substrings)]

def unique_file_matching(path,*substrings,extension=None):
    """Check if a single file contains all substrings and return the unique match"""
    matches = file_matching_all(path,*substrings)
    if extension is not None:
        matches = [fn for fn in matches if fn.endswith(extension)]
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found in {path} matching patterns: {substrings}")
    elif len(matches) > 1:
        raise FileExistsError(f"Multiple files found in {path} matching patterns: {substrings}")
    return matches[0]

###############################################################################
def check_acceptable_param(param:str, acceptable_types) -> str:
    """
    Check that a string used for a function parameter is coded for, and 
    return it formatted in a standard way
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    cleaned_param = param.lower().strip()
    if cleaned_param not in acceptable_types:
        raise ValueError(
            f'Received argument of {param} for a function, but it must '
            f'be one of {acceptable_types}.'
            )
    else:
        return cleaned_param

###############################################################################
def date_rel(date:str, days:int):
    """
    Helper function to calculate date differences by number of days
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    new_date = (
        datetime.strptime(date, '%Y-%m-%d')
        + timedelta(days=days)
        ).strftime('%Y-%m-%d')
    return new_date

###############################################################################
def fig_ax_admin(
    ex_figure=None,
    ex_axes=None,
    new_subplot:bool=True,
    ex_ax_idx=None
    ):
    """
    For visualisations, determine plotting behaviour based on whether 
    the user provides an existing figure and/or axes

    Parameters:
    - figure (mpl.figure): existing matplotlib figure object if 
    provided to the calling function
    - axes (mpl.axes): existing matplotlib axes object if provided to 
    the calling function
    - new_subplot (bool): Whether a new subplot is to be created
    - ex_ax_idx: index of the existing axes to plot on. Required if a
    figure is provided but no axes object, but an existing axes is
    to be drawn on.

    Returns:
    - The existing matplotlib figure if provided by the user, otherwise
    a brand new one
    - The existing matplotlib axes if provided by the user, otherwise a
    brand new one
    --------------------------------------------------------------------
    Notes:
    - Assumes that if the user provides an axes, that it is not
    figureless.
    - This is designed to allow provision of an integer axes index 
    instead of an axes object if desired.
    --------------------------------------------------------------------
    """
    # Create both figure and axes if we haven't been provided with them:
    if ex_figure is None and ex_axes is None:
        out_fig, out_ax = plt.subplots()

    #-----  This is the tricky case --------
    # If we're given a figure but no axes:
    elif ex_axes is None:
        out_fig = ex_figure
        # If a new subplot is requested, just add it:
        if new_subplot:
            out_ax = out_fig.add_subplot()
        # If we're not adding a new subplot, use the user's provided
        #index to decide which axes to plot on:
        else:
            # If the user hasn't specified an index, we'll use the last
            #one:
            if ex_ax_idx is None:
                out_ax_idx = len(out_fig.axes) + 1
            # Otherwise use what they specified:
            else:
                out_ax_idx = ex_ax_idx
            # Get the axes with the provided index:
            out_ax = out_fig.axes[out_ax_idx]
            
    # If axes but no figure, get the parent figure of the axes:
    elif ex_figure is None:
        out_fig = ex_axes.figure
        out_ax = ex_axes
    # If we've been provided with both, just use those:
    else:
        out_fig = ex_figure
        out_ax = ex_axes
    
    return out_fig, out_ax

###############################################################################
def mapify_axes(
    ax,
    crs,
    units:str,
    ):
    """
    Settings for maps based on whether the data is in a projected or
    geographic coordinate system

    Parameters:
    - ax (mpl.axes): matplotlib axes object being uses as a map
    - crs: crs object which can be a GeoDataFrame.crs (for vectors) or
    rasterio's pyplot crs object. 
    - units (str): text describing the units use by the axes object
    --------------------------------------------------------------------
    Notes:
    - The crs object can be more flexible; all it needs is a boolean
    is_projected attribute which equals True for projected CRS and False
    for geographic.
    - For projected CRS, units is assumed to be 'metres', and 'm' will
    be passed to the scalebar indicating metres. If your PCS is not in 
    metres, this may cause the scalebar to fail altogether or have an
    incorrect label.
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
        # Set number format to always two decimal places:
        this_tick_label_formatter = mpl.ticker.FormatStrFormatter('%.2f')
        ax.xaxis.set_major_formatter(this_tick_label_formatter)
        ax.yaxis.set_major_formatter(this_tick_label_formatter)
        # 3-5 ticks on x-axis
        tick_number_formatter_x = mpl.ticker.MaxNLocator(
            min_n_ticks=3,
            nbins=5
            )
        ax.xaxis.set_major_locator(tick_number_formatter_x)
        # Same on y-axis
        tick_number_formatter_y = mpl.ticker.MaxNLocator(
            min_n_ticks=3,
            nbins=5
            )
        ax.yaxis.set_major_locator(tick_number_formatter_y)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

###############################################################################
def fit_multi_figs(fig):
    """
    Adjusts size of figure to fit its axes nicely.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    pass

###############################################################################
def make_axes_title(
    catchment_name:str,
    area_type:str,
    var_name:str,
    colour_column_name:str,
    ) -> str:
    """
    
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    catch_title = clean_chart_title(catchment_name)
    area_title = area_type.title().strip()
    # Most columns will have an aggregation type which is separate to 
    #the variable name, which we still want to keep if it exists:
    clean_name = colour_column_name.replace('_', '').lower().strip()
    for stat in STATS:
        if stat in clean_name:
            agg = stat.title()
            break
        else:
            agg = ''
    
    # Include a year in the title if it's part of the column name:
    if 'year' in clean_name:
        if 'year1' in clean_name:
            year = 'Year 1'
        elif 'year2' in clean_name:
            year = 'Year 2'
        else:
            year = ''
    else:
        year = ''
    
    # Put the title together then clean extra spaces:
    base_title = f'{catch_title} {area_title}: {var_name} {year} {agg}'
    neat_title = ' '.join(base_title.split())
    
    return neat_title

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

###########################################################################
def get_cmap_normer(
    data,
    scale:str,
    min_val=None,
    max_val=None,
    clipped=False,
    clipped_pct=(2, 98)
    ):
    """
    Create a colourmap that can be used for both normalised and log
    scales, and can be used for an image itself and its colourbar.

    Parameters:
    - data: array of values that will be mapped
    - scale: 'log' if log scale is desired, otherwise will be linear
    - min_val: desired value for the minimum of the colour range
    - max_val: desired value for the maximum of the colour range
    - clipped: whether to clip the most extreme values by percentile

    Returns:
    - matplotlib Normalize object which maps the values in the data in
    a way that can be used for colourmaps for plots and colourbars.
    --------------------------------------------------------------------
    Notes:
    - 
    --------------------------------------------------------------------
    """
    arr1 = np.asanyarray(data)

    # Select finite and unmasked values only:
    if np.ma.isMaskedArray(arr1):
        finite = arr1[
            np.isfinite(#this gets all non-infinite values
                arr1.filled(np.nan) #This fills masked values with nan
                )
            ].compressed()
    else:
        finite = arr1[np.isfinite(arr1)]

    # Raise error if no non-infinite, non-masked values
    if finite.size == 0:
        raise ValueError(
            'util.get_cmap_normer() received an array with no valid '
            'values.'
        )
    
    # Get vmin and vmax values from the data if not provided:
    if min_val is None or max_val is None:
        if clipped:
            # Calculate the low and high percentiles provided in the
            #clipped_pct tuple:
            lo, hi = np.nanpercentile(finite, clipped_pct)
            # Use the calculated values if the user hasn't specified:
            vmin = lo if min_val is None else min_val
            vmax = hi if max_val is None else max_val
        else:
            vmin = np.nanmin(finite) if min_val is None else min_val
            vmax = np.nanmax(finite) if max_val is None else max_val
    # If both values are provided, make sure they're valid:
    else:
        # If one of the values is invalid:
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError(
                'util.get_cmap_normer() received invalid values ('
                f'{min_val} to {max_val}) for min_val and/or max_val '
                'arguments.'
            )
        elif max_val <= min_val:
            raise ValueError(
                'util.get_cmap_normer() received a max value of '
                f'{max_val}, which is not larger than the min value '
                f'of {min_val}.'
            )
        else:
            vmin = min_val
            vmax = max_val

    # For now we're going to make everything linear unless the user
    #specifies logarithmic:
    if scale is None or scale.lower().strip() == 'linear':
        return Normalize(vmin=vmin, vmax=vmax)
    elif scale.lower().strip() == 'boundary':
        # Update the vmin and max to integers:
        vmin = int(vmin)
        vmax = int(vmax)
        bounds = np.arange((vmin - 0.5), (vmax + 1.5), 1)
        num_boundaries = len(bounds)
        return mpl.colors.BoundaryNorm(bounds, num_boundaries)


    # Handle logarithmic scale in a safe way;
    else:
        # Lift vmin above 0 if it's not already, and check that not all
        #the values are non-positive
        if vmin <= 0:
            # Check for positive values first:
            posvals = finite[finite > 0]
            if posvals.size == 0:
                raise ValueError(
                    'Log colour scale requires some positive values; ' 
                    'none were found.'
                    )
            # Use the minimum positive value as the new minimum
            vmin = np.nanmin(posvals)
        if vmax <= 0:
            raise ValueError(
                'Log color scale requires vmax > 0'
                )
        return LogNorm(vmin=vmin, vmax=vmax)
    
###############################################################################
def insert_colourbar(axes, normaliser, vis_params):
    """
    Insert a relevant colourbar that fits nicely with a raster plot

    Parameters:
    - axes: matplotlib axes object which the colourbar is for
    - normaliser: matplotlib Normalize object for the data
    - vis_params: dictionary of relevant visualisation settings
    --------------------------------------------------------------------
    Notes:
    - Requires matplotlib toolkits (mpl_toolkits)
    - use util.get_cmap_normer() first to ensure the colourbar scale
    matches that of the plot
    --------------------------------------------------------------------
    """
    # Get the width and height of the figure:
    width = abs(axes.get_xlim()[1] - axes.get_xlim()[0])
    height = abs(axes.get_ylim()[1] - axes.get_ylim()[0])
    # Put the colourbar on the right unless the plot is notably
    #landscape in proportions. Exception is if it's projected,
    #in which case colourbar on the bottom looks bad.
    if width / height >= 1.5 and axes.loaded_crs.is_projected:
        position = 'bottom'
    else:
        position = 'right'
    
    norm_type = vis_params['norm']
    cmap_name = vis_params['cmap']

    # Create a special discrete mapper if we're using a boundary 
    #normaliser:
    if norm_type == 'boundary':
        num_colours = normaliser.N - 1
        # Get a version of the colourmap with the specific number of 
        #colours needed:
        boundary_cmap = plt.cm.get_cmap(cmap_name, num_colours)
        # Create the mappable object:
        mappable = ScalarMappable(norm=normaliser, cmap=boundary_cmap)

        max_ticks = 10
        # Update the ticks to go in the centre of the discrete colours:
        bounds = normaliser.boundaries
        centres = 0.5 * (bounds[:-1] + bounds[1:])
        int_labels = np.round(centres).astype(int)
        if num_colours > 1:
            
            int_min, int_max = int_labels[0], int_labels[-1]

            # If the number of labels is less than the maximum, show all:
            if num_colours <= max_ticks:
                vals_to_show = np.arange(int_min, int_max + 1)
            # Otherwise, work out a roughly optimal spacing:
            else:
                int_between_ticks = np.ceil(num_colours / (max_ticks - 1))
                vals_to_show = np.arange(int_min, int_max + 1, int_between_ticks)
                # If we haven't naturally included the maximum, add it in again:
                if vals_to_show[-1] != int_max:
                    vals_to_show = np.append(vals_to_show, int_max)
            # Make a lookup of what labels to use for what ticks:
            val_centre_dict = dict(zip(int_labels, centres))
            labels_to_show = vals_to_show.astype(int)
            tick_positions = [val_centre_dict[v] for v in vals_to_show]
        else:
            tick_positions = centres
            labels_to_show = centres.astype(int)

    # Otherwise we're using a normal continuous one:
    else:
        # Create a mappable object using the previously-created normaliser:
        mappable = ScalarMappable(norm=normaliser, cmap=vis_params['cmap'])
    
    mappable.set_array([]) #avoid warnings

    # Create a divider to manage spacing of axis and colourbar:
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(position, size='5%', pad=0.05)

    # Create the colourbar:
    cbar = axes.figure.colorbar(
        mappable,
        cax=cax,
        location=position,
        label=f"{vis_params['measure']} ({vis_params['units']})",
        extend=vis_params['cbar_extend']
        )
    if norm_type == 'boundary':
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(labels_to_show)
        cbar.minorticks_off()

    axes.custom_cbar = cbar
    axes.custom_cax = cax

    return cbar

###############################################################################
def plot_spatial_raster(
    existing_axes,
    full_raster_path:str,
    vis_params:dict,
    title:str,
    colourbar:bool=True,
    clip_geometry=None
    ):
    """
    Plot a raster in a standardised way

    Parameters:
    - existing_axes: matplotlib axes to plot onto
    - full_raster_path (str): path to the raster file
    - vis_params (dict): dictionary of visualisation parameters
    - title (str): title for the axes
    - colourbar (bool): whether to add a colourbar
    - clip_geometry (GeoDataFrame): optional boundary to clip the raster
      to in-memory before plotting. Cells outside the boundary are masked
      out, and the colourmap range is derived only from in-boundary
      values. The GeoDataFrame is reprojected to match the raster CRS if
      needed.

    Returns:
    - img, the raster image artist created by this function
    --------------------------------------------------------------------
    Notes:
    - The image artist returned by this function can be used to set
    colourmaps etc.
    - Requires rasterio and numpy
    --------------------------------------------------------------------
    """
    # Ensure we have a valid figure and axes for plotting:
    ax = existing_axes

    # Open the raster and start building the plot:
    with rio.open(full_raster_path) as src:
        # Optionally clip the raster to the supplied boundary in-memory.
        # rasterio.mask requires the geometry CRS to match the raster,
        # so reproject if needed.
        if clip_geometry is not None:
            from rasterio.mask import mask as rio_mask
            boundary = clip_geometry.to_crs(src.crs)
            shapes = [geom.__geo_interface__
                      for geom in boundary.geometry]
            # filled=False returns a numpy masked array, which works
            # for any dtype (including int). We convert to float and
            # set masked cells to NaN below.
            clipped, transform = rio_mask(
                src, shapes, crop=True, filled=False)
            data = clipped[0].astype(float)
            data[clipped[0].mask] = np.nan
        else:
            data = src.read(1).astype(float)
            transform = src.transform

        # Replace any raster nodata value with NaN:
        no_data_value = src.nodata
        if no_data_value is not None:
            data = np.where(data == no_data_value, np.nan, data)

        # Grab the crs while we have it:
        this_crs = src.crs

        # Tie the vis params and crs to the axes for access elsewhere:
        ax.loaded_vis_params = vis_params
        ax.loaded_crs = this_crs
        
        # Get the minimum value from vis_params if there is one:
        try:
            req_min = vis_params['vmin']
        except KeyError:
            req_min = None
        # Same for the maximum value:
        try:
            req_max = vis_params['vmax']
        except KeyError:
            req_max = None

        # Get a Normalize object to handle colourmapping:
        this_normaliser = get_cmap_normer(
            data,
            vis_params['norm'],
            min_val=req_min,
            max_val=req_max
            )

        # Plot the raster values onto the axes:
        img = ax.imshow(
            data,
            cmap=vis_params['cmap'],
            norm=this_normaliser,
            extent=(
                transform[2],
                transform[2] + transform[0] * data.shape[1],
                transform[5] + transform[4] * data.shape[0],
                transform[5]
                )
            )
        
        if colourbar:
            this_cbar = insert_colourbar(
                ax,
                this_normaliser,
                vis_params
                )
        else:
            this_cbar=None

        existing_axes.set_title(title)

        return img, this_crs, this_cbar

###############################################################################
def plot_spatial_vector(
    existing_axes,
    vector_path_or_data:str | gpd.GeoDataFrame,
    vis_params:dict,
    title:str, #title for this axes
    legend:bool=False,
    label:str=None, #Label to go in the legend
    colourbar:bool=True,
    symbol_data=None,
    id_col_name:str='ID',
    data_col_name=None
    ):
    """
    Plot a vector in a standardised way
    --------------------------------------------------------------------
    Notes:
    - We should assume for now that we're getting two things. A 
    polygon file with areas, and (optionally) a DataFrame with 
    values for symbolising the polygons.
    - If a data-based fill is required, this function should
    receive a two-column dataframe. It will have an ID in the first
    column and the value in the second. Will be joined to the 
    spatial file by the ID.
    --------------------------------------------------------------------
    """
    if isinstance(vector_path_or_data, str):
        # Read in the spatial data file:
        shapes = gpd.read_file(vector_path_or_data)
    elif isinstance(vector_path_or_data, gpd.GeoDataFrame):
        shapes = vector_path_or_data
    else:
        raise ValueError(
            'util.plot_spatial_vector() requires either a path to a '
            'shapefile, or a GeoDataFrame, as the vector_path_or_data '
            f'parameter, but received {vector_path_or_data}'
        )
    # Get useful metadata:
    this_crs = shapes.crs

    existing_axes.loaded_vis_params = vis_params
    existing_axes.loaded_crs = this_crs

    # Get relevant values
    norm_type = vis_params['norm']
    cmap_name = vis_params['cmap']

    # If a symbolisation DataFrame is provided:
    if symbol_data is not None:
        # Merge in id_col_name so we have the value for each vector
        #feature to use for symbolising:
        geom_with_data = pd.merge(
            shapes,
            symbol_data,
            on=id_col_name
            )
        colour_col = data_col_name

        # Get a normaliser to use for both plot and colourbar:
        normer = get_cmap_normer(
            data=symbol_data[colour_col],
            scale=vis_params['norm']
            )
        min_plot_val = normer.vmin
        max_plot_val = normer.vmax

        # Create a special discrete mapper if we're using a boundary 
        #normaliser:
        if norm_type == 'boundary':
            num_colours = normer.N - 1
            # Get a version of the colourmap with the specific number of 
            #colours needed:
            boundary_cmap = plt.cm.get_cmap(cmap_name, num_colours)
            # Create the mappable object:
            use_this_cmap = boundary_cmap
        else:
            use_this_cmap = cmap_name

        thing_to_plot=geom_with_data
    # Populate empty values for symbolisations
    else:
        colour_col = None
        use_this_cmap=None
        normer=None
        thing_to_plot = shapes

    # Use Geopandas' built-in plot method:
    existing_axes = thing_to_plot.plot(
        ax=existing_axes,
        column=colour_col,
        cmap=use_this_cmap,
        norm=normer
        )
    
    # If we're symbolising by column and a colourbar is requested,
    #add one:
    if symbol_data is not None and colourbar:
        this_cbar = insert_colourbar(
            axes=existing_axes,
            normaliser=normer,
            vis_params=vis_params
            )
    else:
        this_cbar = None
    
    existing_axes.set_title(title)

    return this_crs, this_cbar, existing_axes

###########################################################################
def get_erosion_title(file_or_col:str, type:str):
    """
    Construct the 'title varname' attribute when plotting the 
    different types of erosion outputs
    ----------------------------------------------------------------
    ----------------------------------------------------------------
    """
    if 'y1' in file_or_col:
        year = 'Year 1'
    elif 'y2' in file_or_col:
        year = 'Year 2'
    else:
        year = '-'

    if 'peak' in file_or_col:
        agg = 'Peak 30-min'
    elif 'total' in file_or_col:
        agg = 'Total'
    else:
        agg = '-'

    meas = type.title()
    title = f'{agg} {meas} {year}'

    return title

###############################################################################
def get_zonal_stats(gdf, raster_path, label, extra_stats=None,
                    all_touched=False, stats=None):
    """
    Compute zonal statistics for each polygon in *gdf* against a raster.

    Parameters
    ----------
    gdf : GeoDataFrame
        Zone polygons.
    raster_path : str
        Path to the raster file.
    label : str
        Label for logging.
    extra_stats : list of str, optional
        Additional rasterstats statistics to include in the output
        (e.g. ``['count', 'nodata']``).  These are appended to the
        standard STATS list.
    all_touched : bool
        If True, include every raster cell touched by a geometry, not
        just those with centres inside the polygon.  Useful for coarse
        rasters where small zones may otherwise have zero pixel overlap.
    stats : list of str, optional
        If given, replaces the default STATS list entirely (and ignores
        *extra_stats*).  Use when only a single aggregation is needed.

    Returns
    -------
    list of dict
        One dict per zone with keys for each requested statistic.
    """
    if stats is not None:
        requested = list(stats)
    else:
        requested = list(STATS)
        if extra_stats:
            requested = requested + [s for s in extra_stats if s not in requested]


    with rio.open(raster_path) as src:
        logger.info(
            f'Getting zonal stats for raster in EPSG:{src.crs.to_epsg()}.'
            f'Zonal vector is in EPSG:{gdf.crs.to_epsg()}.'
            )
        if src.crs != gdf.crs:
            logger.info(f'Reprojecting zones to {src.crs.to_epsg()}...')
            temp_gdf = gdf.to_crs(src.crs)
        else:
            temp_gdf = gdf

        # Determine the effective nodata value.  For float rasters
        # without a declared nodata, use NaN — this is the standard
        # convention and ensures NaN pixels in soil/aridity grids are
        # correctly excluded from statistics.
        nd = src.nodata
        if nd is None and np.issubdtype(src.dtypes[0], np.floating):
            nd = float('nan')
        elif nd is None:
            nd = -9999

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Warning: converting a masked element to nan',
                category=UserWarning,
            )
            zstats = rs.zonal_stats(
                temp_gdf,
                raster_path,
                stats=requested,
                nodata=nd,
                all_touched=all_touched,
                )
    return zstats

