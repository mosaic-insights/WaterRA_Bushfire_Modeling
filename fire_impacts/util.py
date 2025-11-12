import os
import re
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio as rio
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

def package_data_path(fn=None):
    dirname = os.path.join(os.path.dirname(__file__),'..','data')
    if fn is None:
        return dirname
    return os.path.join(dirname,fn)


def load_package_data(fn):
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

def unique_file_matching(path,*substrings):
    """Check if a single file contains all substrings and return the unique match"""
    matches = file_matching_all(path,*substrings)
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found in {path} matching patterns: {substrings}")
    elif len(matches) > 1:
        raise FileExistsError(f"Multiple files found in {path} matching patterns: {substrings}")
    return matches[0]

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
    if scale.lower().strip() != 'log':
        return Normalize(vmin=vmin, vmax=vmax)
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
    
###########################################################################
def insert_colourbar(axes, normaliser, vis_params):
    """
    Insert a relevant colourbar that fits nicely with a raster plot
    --------------------------------------------------------------------
    Notes:
    - Requires matplotlib toolkits (mpl_toolkits)
    --------------------------------------------------------------------
    """
    

    # Create a divider to manage spacing of axis and colourbar:
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Create a mappable object using the previously-created normaliser:
    mappable = ScalarMappable(norm=normaliser, cmap=vis_params['cmap'])
    mappable.set_array([]) #avoid warnings

    cbar = axes.figure.colorbar(
        mappable,
        cax=cax,
        label=f'{vis_params['measure']} ({vis_params['units']})',
        extend=vis_params['cbar_extend']
        )
    return cbar

###########################################################################
def plot_spatial_raster(
    existing_axes,
    full_raster_path:str,    
    vis_params:dict,
    colourbar:bool=True
    ):
    """
    Plot a raster in a standardised way

    Parameters:
    - vis_params (dict): dictionary of visualisation parameters

    Returns:
    - img, the raster image artist created by this function
    ----------------------------------------------------------------
    Notes:
    - The image artist returned by this function can be used to set
    colourmaps etc.
    - Requires rasterio and numpy
    ----------------------------------------------------------------
    """
    # Ensure we have a valid figure and axes for plotting:
    ax = existing_axes

    # Open the raster and start building the plot:
    with rio.open(full_raster_path) as src:
        # Get relevant data and metadata:
        data = src.read(1)
        no_data_value = src.nodata
        if no_data_value is not None:
            # Replace NoData values with NaN
            data = np.where(data == no_data_value, np.nan, data)
        transform = src.transform
        
        # Get a Normalize object to handle colourmapping:
        this_normaliser = get_cmap_normer(data, vis_params['norm'])

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

        # Grab the crs while we have it:
        this_crs = src.crs

        return img, this_crs, this_cbar

###########################################################################
def plot_spatial_vector(
    
    ):
    """
    Plot a vector in a standardised way
    ----------------------------------------------------------------
    ----------------------------------------------------------------
    """
    pass


