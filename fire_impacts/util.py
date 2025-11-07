import os
import re
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
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
def fig_ax_admin(ex_figure=None, ex_axes=None):
    """
    For visualisations, determine plotting behaviour based on whether 
    the user provides an existing figure and/or axes

    Parameters:
    - figure (mpl.figure): existing matplotlib figure object if 
    provided to the calling function
    - axes (mpl.axes): existing matplotlib axes object if provided to 
    the calling function

    Returns:
    - The existing matplotlib figure if provided by the user, otherwise
    a brand new one
    - The existing matplotlib axes if provided by the user, otherwise a
    brand new one
    --------------------------------------------------------------------
    Notes:
    - Assumes that if the user provides an axes, that it is not
    figureless.
    - If a figure is provided but no axes, will add a new axes object.
    This may result in undesired behaviour. TODO: add functionality to
    try filling the last emtpy subplot before creating a new axes.
    --------------------------------------------------------------------
    """
    # Create both figure and axes if we haven't been provided with them:
    if ex_figure is None and ex_axes is None:
        out_fig, out_ax = plt.subplots()

    # If we're given a figure but no axes, add a subplot:
    elif ex_axes is None:
        out_fig = ex_figure
        out_ax = out_fig.add_subplot()
    
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
