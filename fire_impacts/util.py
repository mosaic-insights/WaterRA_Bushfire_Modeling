"""
Shared utility functions for visualisation, raster handling, and
general data manipulation within the fire_impacts package.
"""

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

from . import const as constants

STATS = constants.STATS

logger = logging.getLogger(__name__)


###############################################################################
def retry(
    fn,
    retries=5,
    initial_delay=8,
    delay_scale=3,
    specific_exceptions=None,
    ):
    """
    Call fn() and retry on failure with exponential back-off.

    Parameters:
    - fn: Callable to attempt.
    - retries: Maximum number of retries before re-raising.
    - initial_delay: Seconds to wait before the first retry.
    - delay_scale: Multiplier applied to the delay after each retry.
    - specific_exceptions: If given, only retry on these exception
      classes; all others are re-raised immediately.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    import time

    try:
        return fn()
    except Exception as e:
        # If all retries are exhausted, give up and re-raise:
        if retries <= 0:
            raise e

        # Re-raise immediately for exception types not in the
        # allowed list:
        if specific_exceptions is not None:
            if e.__class__ not in specific_exceptions:
                raise e

        logger.warning(
            f'Failed with {e}. Retrying after {initial_delay} seconds'
            )
        time.sleep(initial_delay)
        return retry(
            fn, retries - 1, initial_delay * delay_scale,
            delay_scale, specific_exceptions,
            )


###############################################################################
def package_data_path(fn=None):
    """
    Return the path to the package's static data directory.

    Parameters:
    - fn: Optional filename to join onto the data directory path.

    Returns:
    - Full path to the data directory, or to the specified file
      within it.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    dirname = os.path.join(os.path.dirname(__file__), '..', 'data')
    if fn is None:
        return dirname
    return os.path.join(dirname, fn)


###############################################################################
def load_package_data(fn):
    """
    Load a static package lookup table by filename.

    Parameters:
    - fn: Filename of the data file within the package data directory.

    Returns:
    - DataFrame if the file is a CSV, otherwise None.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    fn = package_data_path(fn)
    if fn.endswith('.csv'):
        logger.info(f'Loading data from {fn}')
        import pandas as pd
        return pd.read_csv(fn)
    logger.error(f'Unsupported file type: {fn}')
    return None


###############################################################################
def file_matching_all(path, *substrings):
    """
    Return all files in a directory whose names contain every substring.

    Parameters:
    - path: Directory to list.
    - substrings: One or more substrings that must all appear in the
      filename.

    Returns:
    - List of matching filenames (not full paths).
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    files = os.listdir(path)
    return [fn for fn in files if all(p in fn for p in substrings)]


###############################################################################
def unique_file_matching(path, *substrings, extension=None):
    """
    Return the single file in a directory matching all given substrings.

    Parameters:
    - path: Directory to search.
    - substrings: One or more substrings that must all appear in the
      filename.
    - extension: If provided, also filter by this file extension
      (e.g. '.tif').

    Returns:
    - Filename of the unique match (not the full path).
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    matches = file_matching_all(path, *substrings)
    if extension is not None:
        matches = [fn for fn in matches if fn.endswith(extension)]
    if len(matches) == 0:
        raise FileNotFoundError(
            f'No file found in {path} matching patterns: {substrings}'
            )
    elif len(matches) > 1:
        raise FileExistsError(
            f'Multiple files found in {path} matching patterns: '
            f'{substrings}'
            )
    return matches[0]


###############################################################################
def check_acceptable_param(param: str, acceptable_types) -> str:
    """
    Validate a string parameter value and return it normalised.

    Parameters:
    - param: Parameter value to check.
    - acceptable_types: Collection of valid lower-cased strings.

    Returns:
    - The parameter value, stripped and lower-cased.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    cleaned_param = param.lower().strip()
    if cleaned_param not in acceptable_types:
        raise ValueError(
            f'Received argument of {param} for a function, but it '
            f'must be one of {acceptable_types}.'
            )
    else:
        return cleaned_param


###############################################################################
def date_rel(date: str, days: int):
    """
    Shift a date string by a number of days.

    Parameters:
    - date: Date string in 'YYYY-MM-DD' format.
    - days: Number of days to add (negative to subtract).

    Returns:
    - New date string in 'YYYY-MM-DD' format.
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
    new_subplot: bool = True,
    ex_ax_idx=None,
    ):
    """
    Resolve a matplotlib figure and axes for a visualisation call.

    Parameters:
    - ex_figure: Existing matplotlib figure. If both ex_figure and
      ex_axes are None, a new figure and axes are created.
    - ex_axes: Existing matplotlib axes. If provided without a
      figure, the parent figure is taken from the axes object.
    - new_subplot: If True and a figure is given without axes, add a
      new subplot rather than selecting an existing one.
    - ex_ax_idx: Index of the existing axes to use when new_subplot
      is False and no axes object is provided.

    Returns:
    - Resolved matplotlib figure.
    - Resolved matplotlib axes.
    --------------------------------------------------------------------
    Notes:
    - Assumes that if ex_axes is provided it already belongs to a
      figure.
    - Accepts an integer axes index via ex_ax_idx as an alternative
      to passing an axes object directly.
    --------------------------------------------------------------------
    """
    # Create both figure and axes if neither is provided:
    if ex_figure is None and ex_axes is None:
        out_fig, out_ax = plt.subplots()

    # Figure provided but no axes - resolve which axes to draw on:
    elif ex_axes is None:
        out_fig = ex_figure
        if new_subplot:
            # Add a fresh subplot to the existing figure:
            out_ax = out_fig.add_subplot()
        else:
            # Select an existing axes by index; default to the last
            # one if no index is specified:
            if ex_ax_idx is None:
                out_ax_idx = len(out_fig.axes) + 1
            else:
                out_ax_idx = ex_ax_idx
            out_ax = out_fig.axes[out_ax_idx]

    # Axes provided but no figure - get the parent figure:
    elif ex_figure is None:
        out_fig = ex_axes.figure
        out_ax = ex_axes

    # Both provided - use them directly:
    else:
        out_fig = ex_figure
        out_ax = ex_axes

    return out_fig, out_ax


###############################################################################
def mapify_axes(
    ax,
    crs,
    units: str,
    ):
    """
    Apply map-appropriate axis formatting for a given coordinate system.

    Parameters:
    - ax: matplotlib axes object to format.
    - crs: CRS object with a boolean is_projected attribute (e.g.
      a GeoDataFrame.crs or a rasterio CRS).
    - units: Text describing the axis units (used for scalebar
      configuration on projected CRS).
    --------------------------------------------------------------------
    Notes:
    - For projected CRS, ticks are hidden and a scalebar is added.
      units is assumed to be metres; a non-metre PCS may produce an
      incorrect or missing scalebar label.
    - For geographic CRS, tick labels are formatted to two decimal
      places and longitude/latitude axis labels are added.
    --------------------------------------------------------------------
    """
    if crs.is_projected:
        # Projected CRS: suppress ticks and add a scalebar instead:
        ax.set_xticks([])
        ax.set_yticks([])
        from matplotlib_scalebar.scalebar import ScaleBar

        sb_fontprops = {'size': 'xx-small'}
        these_units = units[0]

        # Create the scalebar (dx=1 means one pixel = one map unit):
        this_scalebar = ScaleBar(
            dx=1,
            units=these_units,
            loc='lower left',
            font_properties=sb_fontprops,
            box_alpha=0.5,
            )
        ax.add_artist(this_scalebar)

    if crs.is_geographic:
        # Geographic CRS: format tick labels to two decimal places:
        this_tick_label_formatter = mpl.ticker.FormatStrFormatter(
            '%.2f'
            )
        ax.xaxis.set_major_formatter(this_tick_label_formatter)
        ax.yaxis.set_major_formatter(this_tick_label_formatter)

        # Aim for 3-5 ticks on each axis:
        tick_number_formatter_x = mpl.ticker.MaxNLocator(
            min_n_ticks=3, nbins=5
            )
        ax.xaxis.set_major_locator(tick_number_formatter_x)

        tick_number_formatter_y = mpl.ticker.MaxNLocator(
            min_n_ticks=3, nbins=5
            )
        ax.yaxis.set_major_locator(tick_number_formatter_y)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


###############################################################################
def fit_multi_figs(fig):
    """
    Adjust the size of a figure to fit its axes nicely.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    pass


###############################################################################
def make_axes_title(
    catchment: str,
    area_type: str,
    var_name: str,
    colour_column_name: str,
    ) -> str:
    """
    Build a standardised axes title from catchment, area, and variable
    information.

    Parameters:
    - catchment: Name of the catchment (underscores and EPSG
      codes are cleaned up automatically).
    - area_type: Spatial unit type, e.g. 'Headwaters'.
    - var_name: Variable or measure name to include in the title.
    - colour_column_name: Column name used for colouring; year and
      aggregation type are extracted from this where present.

    Returns:
    - Cleaned title string with extra spaces collapsed.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    catch_title = clean_chart_title(catchment)
    area_title = area_type.title().strip()

    # Extract the aggregation type from the column name if present.
    # Most columns carry a stat suffix (e.g. _mean, _sum) which is
    # worth including in the title alongside the variable name:
    clean_name = colour_column_name.replace('_', '').lower().strip()
    for stat in STATS:
        if stat in clean_name:
            agg = stat.title()
            break
        else:
            agg = ''

    # Include a year label if the column name encodes one:
    if 'year' in clean_name:
        if 'year1' in clean_name:
            year = 'Year 1'
        elif 'year2' in clean_name:
            year = 'Year 2'
        else:
            year = ''
    else:
        year = ''

    # Assemble the title and collapse any duplicate spaces:
    base_title = (
        f'{catch_title} {area_title}: {var_name} {year} {agg}'
        )
    neat_title = ' '.join(base_title.split())

    return neat_title


###############################################################################
def clean_chart_title(text):
    """
    Clean a raw string for use as a chart title.

    Parameters:
    - text: Raw string (e.g. a catchment folder name).

    Returns:
    - Title string with trailing EPSG codes removed and underscores
      and camel case expanded to spaces.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Remove a trailing numeric EPSG code (e.g. '_28355'):
    int_title = re.sub(r'_\d+$', '', text)
    # Expand camel case to space-separated words:
    int_title = re.sub(r'(?<!^)(?=[A-Z])', ' ', int_title)
    title = int_title.replace('_', ' ').strip()
    return title


###############################################################################
def get_cmap_normer(
    data,
    scale: str,
    min_val=None,
    max_val=None,
    clipped=False,
    clipped_pct=(2, 98),
    ):
    """
    Create a matplotlib Normalize object for colourmapping.

    Parameters:
    - data: Array of values to be mapped.
    - scale: 'log' for logarithmic scale; 'boundary' for a discrete
      integer scale; any other value gives a linear scale.
    - min_val: Desired minimum of the colour range. Derived from data
      if not provided.
    - max_val: Desired maximum of the colour range. Derived from data
      if not provided.
    - clipped: If True, derive vmin/vmax from percentiles of the data
      rather than the full range.
    - clipped_pct: (low, high) percentile tuple used when clipped is
      True.

    Returns:
    - matplotlib Normalize (or BoundaryNorm / LogNorm) object for
      use with colourmaps and colourbars.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    arr1 = np.asanyarray(data)

    # Select finite and unmasked values only:
    if np.ma.isMaskedArray(arr1):
        finite = arr1[
            np.isfinite(
                arr1.filled(np.nan)  # fill masked values with nan
                )
            ].compressed()
    else:
        finite = arr1[np.isfinite(arr1)]

    # Raise an error if no usable values remain:
    if finite.size == 0:
        raise ValueError(
            'util.get_cmap_normer() received an array with no valid '
            'values.'
            )

    # Derive vmin and vmax from the data when not provided:
    if min_val is None or max_val is None:
        if clipped:
            # Use the provided low/high percentiles as the range:
            lo, hi = np.nanpercentile(finite, clipped_pct)
            vmin = lo if min_val is None else min_val
            vmax = hi if max_val is None else max_val
        else:
            vmin = np.nanmin(finite) if min_val is None else min_val
            vmax = np.nanmax(finite) if max_val is None else max_val

        # Degenerate case: all values are equal (or clipped
        # percentiles coincide). Matplotlib silently expands such a
        # Normalize to +/-0.1, producing nonsensical negatives for
        # non-negative measures like probabilities. Widen to a useful
        # range instead: [0, 1] when the constant value is in that
        # interval, otherwise a unit window centred on the value:
        if vmin == vmax:
            if 0.0 <= vmin <= 1.0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = vmin - 0.5, vmin + 0.5

    # Validate explicitly provided vmin/vmax:
    else:
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError(
                'util.get_cmap_normer() received invalid values ('
                f'{min_val} to {max_val}) for min_val and/or max_val.'
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

    # Return the appropriate normaliser for the requested scale type:
    if scale is None or scale.lower().strip() == 'linear':
        return Normalize(vmin=vmin, vmax=vmax)

    elif scale.lower().strip() == 'boundary':
        # Discrete integer bins centred on whole numbers:
        vmin = int(vmin)
        vmax = int(vmax)
        bounds = np.arange((vmin - 0.5), (vmax + 1.5), 1)
        num_boundaries = len(bounds)
        return mpl.colors.BoundaryNorm(bounds, num_boundaries)

    else:
        # Logarithmic scale: lift vmin above zero if needed and verify
        # that at least some positive values exist in the data:
        if vmin <= 0:
            posvals = finite[finite > 0]
            if posvals.size == 0:
                raise ValueError(
                    'Log colour scale requires some positive values; '
                    'none were found.'
                    )
            vmin = np.nanmin(posvals)
        if vmax <= 0:
            raise ValueError(
                'Log color scale requires vmax > 0'
                )
        return LogNorm(vmin=vmin, vmax=vmax)


###############################################################################
def insert_colourbar(axes, normaliser, vis_params):
    """
    Add a fitted colourbar to a spatial plot.

    Parameters:
    - axes: matplotlib axes that the colourbar is associated with.
    - normaliser: matplotlib Normalize object for the data.
    - vis_params: Dict of visualisation parameters (must include
      'norm', 'cmap', 'measure', 'units', 'cbar_extend').

    Returns:
    - matplotlib Colorbar object.
    --------------------------------------------------------------------
    Notes:
    - Use get_cmap_normer() first so the colourbar scale matches the
      plot.
    - Colourbar is placed on the right for roughly square or portrait
      plots, and on the bottom for wide landscape projected plots.
    --------------------------------------------------------------------
    """
    # Decide placement based on plot aspect ratio:
    width = abs(axes.get_xlim()[1] - axes.get_xlim()[0])
    height = abs(axes.get_ylim()[1] - axes.get_ylim()[0])
    if width / height >= 1.5 and axes.loaded_crs.is_projected:
        position = 'bottom'
    else:
        position = 'right'

    norm_type = vis_params['norm']
    cmap_name = vis_params['cmap']

    if norm_type == 'boundary':
        # Build a discrete colourmap and compute tick positions at
        # the centre of each colour band:
        num_colours = normaliser.N - 1
        boundary_cmap = plt.cm.get_cmap(cmap_name, num_colours)
        mappable = ScalarMappable(norm=normaliser, cmap=boundary_cmap)

        max_ticks = 10
        bounds = normaliser.boundaries
        centres = 0.5 * (bounds[:-1] + bounds[1:])
        int_labels = np.round(centres).astype(int)

        if num_colours > 1:
            int_min, int_max = int_labels[0], int_labels[-1]
            # Show all ticks if there are few enough; otherwise space
            # evenly and ensure the maximum is always included:
            if num_colours <= max_ticks:
                vals_to_show = np.arange(int_min, int_max + 1)
            else:
                int_between_ticks = np.ceil(
                    num_colours / (max_ticks - 1)
                    )
                vals_to_show = np.arange(
                    int_min, int_max + 1, int_between_ticks
                    )
                if vals_to_show[-1] != int_max:
                    vals_to_show = np.append(vals_to_show, int_max)
            val_centre_dict = dict(zip(int_labels, centres))
            labels_to_show = vals_to_show.astype(int)
            tick_positions = [
                val_centre_dict[v] for v in vals_to_show
                ]
        else:
            tick_positions = centres
            labels_to_show = centres.astype(int)

    # Continuous colourbar (linear or log):
    else:
        mappable = ScalarMappable(
            norm=normaliser, cmap=vis_params['cmap']
            )

    mappable.set_array([])  # suppress matplotlib warning

    # Attach the colourbar neatly using make_axes_locatable:
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(position, size='5%', pad=0.05)

    cbar = axes.figure.colorbar(
        mappable,
        cax=cax,
        location=position,
        label=f"{vis_params['measure']} ({vis_params['units']})",
        extend=vis_params['cbar_extend'],
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
    full_raster_path: str,
    vis_params: dict,
    title: str,
    colourbar: bool = True,
    clip_geometry=None,
    ):
    """
    Plot a raster onto matplotlib axes in a standardised way.

    Parameters:
    - existing_axes: matplotlib axes to plot onto.
    - full_raster_path: Path to the raster file.
    - vis_params: Dict of visualisation parameters (cmap, norm, vmin,
      vmax, cbar_extend, measure, units, and optionally
      scale_to_per_ha).
    - title: Title for the axes.
    - colourbar: If True, add a colourbar.
    - clip_geometry: Optional GeoDataFrame to clip the raster to
      in-memory before plotting. Cells outside the boundary are
      masked; the colourmap range is derived from in-boundary values
      only. Reprojected to the raster CRS if needed.

    Returns:
    - img: The raster image artist.
    - this_crs: CRS of the raster.
    - this_cbar: Colourbar object, or None if colourbar is False.
    --------------------------------------------------------------------
    Notes:
    - The returned image artist can be used to adjust colourmaps
      after the fact.
    --------------------------------------------------------------------
    """
    ax = existing_axes

    with rio.open(full_raster_path) as src:
        # Optionally clip the raster to the supplied boundary. rasterio
        # requires geometry CRS to match the raster, so reproject if
        # needed. filled=False gives a masked array; masked cells are
        # set to NaN below so they are excluded from colour scaling:
        if clip_geometry is not None:
            from rasterio.mask import mask as rio_mask
            boundary = clip_geometry.to_crs(src.crs)
            shapes = [
                geom.__geo_interface__ for geom in boundary.geometry
                ]
            clipped, transform = rio_mask(
                src, shapes, crop=True, filled=False
                )
            data = clipped[0].astype(float)
            data[clipped[0].mask] = np.nan
        else:
            data = src.read(1).astype(float)
            transform = src.transform

        # Replace any declared nodata value with NaN:
        no_data_value = src.nodata
        if no_data_value is not None:
            data = np.where(data == no_data_value, np.nan, data)

        # Optionally rescale from per-cell to per-hectare units.
        # Cell area in hectares comes from the affine transform:
        if vis_params.get('scale_to_per_ha'):
            cell_area_ha = (
                transform[0] * abs(transform[4])
                ) / 10000
            data = data / cell_area_ha

        this_crs = src.crs

        # Attach vis params and CRS to the axes for downstream access:
        ax.loaded_vis_params = vis_params
        ax.loaded_crs = this_crs

        # Get the requested colour range from vis_params if provided:
        req_min = vis_params.get('vmin')
        req_max = vis_params.get('vmax')

        # Build a Normalize object for colourmapping:
        this_normaliser = get_cmap_normer(
            data,
            vis_params['norm'],
            min_val=req_min,
            max_val=req_max,
            )

        # Plot the raster as an image; the spatial extent is derived
        # from the affine transform:
        img = ax.imshow(
            data,
            cmap=vis_params['cmap'],
            norm=this_normaliser,
            extent=(
                transform[2],
                transform[2] + transform[0] * data.shape[1],
                transform[5] + transform[4] * data.shape[0],
                transform[5],
                ),
            )

        if colourbar:
            this_cbar = insert_colourbar(
                ax, this_normaliser, vis_params
                )
        else:
            this_cbar = None

        existing_axes.set_title(title)

        return img, this_crs, this_cbar


###############################################################################
def plot_spatial_vector(
    existing_axes,
    vector_path_or_data: str | gpd.GeoDataFrame,
    vis_params: dict,
    title: str,
    legend: bool = False,
    label: str = None,
    colourbar: bool = True,
    symbol_data=None,
    id_col_name: str = 'ID',
    data_col_name=None,
    ):
    """
    Plot a vector layer onto matplotlib axes in a standardised way.

    Parameters:
    - existing_axes: matplotlib axes to plot onto.
    - vector_path_or_data: Path to a shapefile, or a GeoDataFrame.
    - vis_params: Dict of visualisation parameters.
    - title: Title for the axes.
    - legend: Whether to add a legend.
    - label: Label string for the legend entry.
    - colourbar: If True and symbol_data is provided, add a colourbar.
    - symbol_data: Optional DataFrame for symbolising polygons. Must
      contain id_col_name and data_col_name columns.
    - id_col_name: Column name used to join symbol_data to the vector
      layer.
    - data_col_name: Column in symbol_data containing the values to
      use for polygon colouring.

    Returns:
    - this_crs: CRS of the vector data.
    - this_cbar: Colourbar object, or None.
    - existing_axes: The axes after plotting.
    --------------------------------------------------------------------
    Notes:
    - When symbol_data is None, polygons are plotted without a data
      fill.
    --------------------------------------------------------------------
    """
    # Load the vector data from a file path or use directly:
    if isinstance(vector_path_or_data, str):
        shapes = gpd.read_file(vector_path_or_data)
    elif isinstance(vector_path_or_data, gpd.GeoDataFrame):
        shapes = vector_path_or_data
    else:
        raise ValueError(
            'util.plot_spatial_vector() requires either a path to a '
            'shapefile, or a GeoDataFrame, as the vector_path_or_data '
            f'parameter, but received {vector_path_or_data}'
            )

    this_crs = shapes.crs
    existing_axes.loaded_vis_params = vis_params
    existing_axes.loaded_crs = this_crs

    norm_type = vis_params['norm']
    cmap_name = vis_params['cmap']

    if symbol_data is not None:
        # Merge the symbolisation data onto the geometry by id column:
        geom_with_data = pd.merge(
            shapes, symbol_data, on=id_col_name
            )
        colour_col = data_col_name
        geom_with_data.to_csv(
            '\\zz_TempDump\\geom_with_data.csv', index=False
            )

        # Build a normaliser, honouring any fixed vmin/vmax from
        # vis_params so callers can lock the colour range:
        normer = get_cmap_normer(
            data=symbol_data[colour_col],
            scale=vis_params['norm'],
            min_val=vis_params.get('vmin'),
            max_val=vis_params.get('vmax'),
            )

        # Use a discrete colourmap for boundary norm:
        if norm_type == 'boundary':
            num_colours = normer.N - 1
            boundary_cmap = plt.cm.get_cmap(cmap_name, num_colours)
            use_this_cmap = boundary_cmap
        else:
            use_this_cmap = cmap_name

        thing_to_plot = geom_with_data

    else:
        # No symbolisation data - plot plain polygon outlines:
        colour_col = None
        use_this_cmap = None
        normer = None
        thing_to_plot = shapes

    # Plot using GeoPandas' built-in plot method:
    existing_axes = thing_to_plot.plot(
        ax=existing_axes,
        column=colour_col,
        cmap=use_this_cmap,
        norm=normer,
        )

    # Add a colourbar when symbolising by data column:
    if symbol_data is not None and colourbar:
        this_cbar = insert_colourbar(
            axes=existing_axes,
            normaliser=normer,
            vis_params=vis_params,
            )
    else:
        this_cbar = None

    existing_axes.set_title(title)

    return this_crs, this_cbar, existing_axes


###############################################################################
def get_erosion_title(file_or_col: str, type: str):
    """
    Build the title_varname string for an erosion or delivery plot.

    Parameters:
    - file_or_col: Raster filename or column name (year and
      aggregation type are detected via keyword matching).
    - type: Measure type label, e.g. 'erosion' or 'delivered'.

    Returns:
    - Title string combining aggregation type, measure, and year.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Detect year from the filename or column name:
    if 'y1' in file_or_col:
        year = 'Year 1'
    elif 'y2' in file_or_col:
        year = 'Year 2'
    else:
        year = '-'

    # Detect whether this is a peak or total aggregation:
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
def get_zonal_stats(
    gdf,
    raster_path,
    label,
    extra_stats=None,
    all_touched=False,
    stats=None,
    ):
    """
    Compute zonal statistics for each polygon in gdf against a raster.

    Parameters:
    - gdf: GeoDataFrame of zone polygons.
    - raster_path: Path to the raster file.
    - label: Label string used in log messages.
    - extra_stats: Additional rasterstats statistics to include (e.g.
      ['count', 'nodata']). Appended to the standard STATS list.
    - all_touched: If True, include every raster cell touched by a
      polygon, not just those with centres inside it. Useful for
      coarse rasters where small zones may have zero pixel overlap.
    - stats: If given, replaces the default STATS list entirely
      (extra_stats is then ignored).

    Returns:
    - List of dicts, one per zone, with a key for each requested
      statistic.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Build the list of statistics to request from rasterstats:
    if stats is not None:
        requested = list(stats)
    else:
        requested = list(STATS)
        if extra_stats:
            requested = requested + [
                s for s in extra_stats if s not in requested
                ]

    with rio.open(raster_path) as src:
        logger.info(
            f'Getting zonal stats for raster in '
            f'EPSG:{src.crs.to_epsg()}. '
            f'Zonal vector is in EPSG:{gdf.crs.to_epsg()}.'
            )
        # Reproject zones to the raster CRS if they differ:
        if src.crs != gdf.crs:
            logger.info(
                f'Reprojecting zones to {src.crs.to_epsg()}...'
                )
            temp_gdf = gdf.to_crs(src.crs)
        else:
            temp_gdf = gdf

        # Determine the effective nodata value. For float rasters
        # without a declared nodata, use NaN - this is the standard
        # convention and ensures NaN pixels are correctly excluded:
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
