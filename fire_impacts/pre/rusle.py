from fire_impacts.pre.util import clip_and_reproject_raster, read_raster, read_aligned
import rasterio as rio
from .project import FireImpactsProject
from .topography import D8_FLOW_DIRECTIONS
from .data_sources import CSIRO_C_FACTOR_GRID, CSIRO_K_FACTOR_GRID
from pysheds.grid import Grid
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


def compute_adjusted_k_c(proj: FireImpactsProject, catchment: str, c_factor_fn: str = None, k_factor_fn: str = None):
    if catchment is None:
        proj.for_each_catchment(lambda c: compute_adjusted_k_c(
            proj, c, c_factor_fn, k_factor_fn))
        return

    # bounds = proj.catchment_bounds(catchment)
    shp = proj.boundary_files[catchment]

    if c_factor_fn is None:
        c_factor_fn = CSIRO_C_FACTOR_GRID
    clip_and_reproject_raster(c_factor_fn, shp, proj.catchment_path(
        catchment, 'Erodibility', 'C_factor.tif'))

    if k_factor_fn is None:
        k_factor_fn = CSIRO_K_FACTOR_GRID
    clip_and_reproject_raster(k_factor_fn, shp, proj.catchment_path(
        catchment, 'Erodibility', 'K_factor.tif'))

    dem_fn = proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    dem, dem_meta = read_raster(dem_fn)
    dem_transform = dem_meta['transform']
    dem_crs = dem_meta['crs']
    dNBR = read_aligned(proj.catchment_path(
        catchment, 'FireSeverity', 'dNBR.tif'), dem_transform, dem_crs, dem.shape)
    Cbase = read_aligned(proj.catchment_path(
        catchment, 'Erodibility', 'C_factor.tif'), dem_transform, dem_crs, dem.shape)
    Kbase = read_aligned(proj.catchment_path(
        catchment, 'Erodibility', 'K_factor.tif'), dem_transform, dem_crs, dem.shape)
    AI = read_aligned(proj.catchment_path(catchment, 'Soils',
                      'Aridity.tif'), dem_transform, dem_crs, dem.shape)

    t = 1
    x_c = 0.4
    x_k = 1
    Kfire = 0.081

    CdNBR = dNBR * 1000
    CdNBR[CdNBR < 0] = 0
    CdNBR[CdNBR > 400] = 0.081
    dNBRmask = (CdNBR > 0) & (CdNBR <= 400)
    CdNBR[dNBRmask] = Cbase[dNBRmask] + \
        ((0.081 - Cbase[dNBRmask]) * (CdNBR[dNBRmask] / 400))

    C = (CdNBR - Cbase) * np.exp(-t / (x_c * AI)) + Cbase
    K = (Kfire - Kbase) * np.exp(-t / (x_k * AI)) + Kbase

    out_meta = {
        'driver': 'GTiff',
        'height': dem.shape[0],
        'width': dem.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': dem_crs,
        'transform': dem_transform,
        'compress': 'lzw',
        'nodata': np.nan
    }
    with rio.open(proj.catchment_path(catchment, 'Erodibility', 'C_factor_adjusted.tif'), 'w', **out_meta) as dest:
        dest.write(C, 1)

    with rio.open(proj.catchment_path(catchment, 'Erodibility', 'K_factor_adjusted.tif'), 'w', **out_meta) as dest:
        dest.write(K, 1)

def _topographic_indices(project: FireImpactsProject,catchment: str):
    dem_path = project.catchment_path(catchment, 'Topography', 'DEM.tif')
    grid = Grid.from_raster(dem_path)
    dem_grid = grid.read_raster(dem_path)
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.fill_depressions(dem_filled)
    inflated_dem = grid.resolve_flats(dem_filled)
    # Calculate flow direction and accumulation
    fdir = grid.flowdir(inflated_dem, dirmap=D8_FLOW_DIRECTIONS)
    acc = grid.accumulation(fdir, dirmap=D8_FLOW_DIRECTIONS)
    return grid, fdir, acc

def compute_lsi(project: FireImpactsProject, catchment=None):
    """
    Calculate slope, aspect, specific catchment area, and LSi factor from a DEM (Digital Elevation Model).
    The LSi factor is then saved as a raster file in the specified output path.

    Parameters:
    project (fire_impacts.FireImpactsProject): Current project
    catchment (str): Name of the catchment to process. If None, process all catchments.

    Returns:
    slope_degrees (numpy array): Slope values in degrees for each pixel.
    slope_percent (numpy array): Slope values in percentage for each pixel.
    aspect_radians (numpy array): Aspect values (direction of steepest slope) in radians for each pixel.
    area (numpy array): Specific catchment area (Ai_in) for each pixel.
    LSi (numpy array): The calculated slope length-gradient factor (LSi) for each pixel.
    """

    if catchment is None:
        return project.for_each_catchment(lambda c: compute_lsi(project, c))

    logger.info('Computed LSI for catchment: %s', catchment)
    dem_path = project.catchment_path(catchment, 'Topography', 'DEM.tif')
    # Open the DEM raster and extract the elevation data
    with rio.open(dem_path) as src:
        dem_data = src.read(1)  # Read the first band (DEM data)
        transform = src.transform  # Affine transformation for pixel to geographic coordinates
        nodata = src.nodata  # NoData value used in the DEM

    # Get pixel resolution (grid cell size)
    xres = transform[0]  # Width of a pixel (east-west direction)
    yres = abs(transform[4])  # Height of a pixel (north-south direction)
    pixel_area = xres * yres  # Area of a single pixel

    # Replace NoData values with NaN for numerical operations
    dem_data = np.where(dem_data == nodata, np.nan, dem_data)

    # Calculate slope using the gradient of the elevation data
    # Elevation gradient in x and y directions
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)  # Slope as a ratio (rise/run)
    slope_radians = np.arctan(slope_ratio)  # Slope in radians
    slope_degrees = np.degrees(slope_radians)  # Slope in degrees
    slope_percent = slope_ratio * 100  # Slope as a percentage

    # Calculate the aspect (direction of steepest descent) using arctan2
    aspect_radians = np.arctan2(dz_dy, -dz_dx)  # Aspect in radians
    # Ensure aspect is in the range [0, 2π]
    # 0 radians (or 0°) indicates a north-facing slope
    aspect_radians = np.where(
        aspect_radians < 0, 2 * np.pi + aspect_radians, aspect_radians)
    # Initialize a Pysheds Grid object and read the DEM data into it                          # π/2 radians (or 90°) indicates an east-facing slope.
    # π radians (or 180°) indicates a south-facing slope.
    _, _, acc = _topographic_indices(project,catchment)
    acc_data = np.array(acc, dtype=np.float32)  # Convert to numpy array
    # Estimate specific catchment area (Ai_in) in meter
    specific_area = np.sqrt(acc_data * pixel_area)
    # Cap specific_area values to a maximum of 141
    # set a max for slope length (sqrt(area in m2)) to avoid overestimation of the LS factor in heterogeneous landscapes
    specific_area = np.where(specific_area > 141, 141, specific_area)

    # Calculate the aspect length (xi) as the sum of the absolute sine and cosine of the aspect angle
    xi = np.abs(np.sin(aspect_radians)) + np.abs(np.cos(aspect_radians))

    # Calculate slope factor (Si) based on slope percentage and slope in degrees
    # Initialize Si with the same shape as slope_percent
    Si = np.zeros_like(slope_percent)
    Si[slope_percent < 9] = 10.8 * \
        np.sin(slope_radians[slope_percent < 9]) + 0.03  # For slopes < 9%
    Si[slope_percent >= 9] = 16.8 * \
        np.sin(slope_radians[slope_percent >= 9]) - 0.50  # For slopes >= 9%

    # Calculate the RUSLE length exponent factor (m) based on slope percentage
    # Initialize m with the same shape as slope_percent
    m = np.zeros_like(slope_percent)
    m[slope_percent <= 1] = 0.2
    m[(slope_percent > 1) & (slope_percent <= 3.5)] = 0.3
    m[(slope_percent > 3.5) & (slope_percent <= 5)] = 0.4
    m[(slope_percent > 5) & (slope_percent <= 9)] = 0.5

    # For slopes greater than 9%, use a more detailed calculation
    mask = slope_percent > 9
    if np.any(mask):
        # Convert percentage to radians
        slope_radians_high = np.arctan(slope_percent[mask] / 100)
        beta = (np.sin(slope_radians_high) / 0.0896) / \
            ((3 * np.sin(slope_radians_high)**0.8) + 0.56)
        m[mask] = beta / (1 + beta)

    # Calculate LSi factor (slope length-gradient factor)
    D = np.sqrt(pixel_area)  # Grid cell dimension (same as pixel size in DEM)
    LSi = Si * (((specific_area + D**2)**(m + 1)) - (specific_area **
                (m + 1))) / ((D**(m + 2)) * (xi**m) * (22.13**m))

    # Open the DEM raster to retrieve the metadata for writing the output raster
    with rio.open(dem_path) as src:
        dem_meta = src.meta.copy()  # Copy metadata from the DEM

    # Update metadata to set the output raster data type and NoData value
    nodata_value = 0.0  # NoData value for LSi raster
    dem_meta.update({
        'dtype': rio.float32,  # Set data type to float32
        'nodata': nodata_value      # Set NoData value
    })

    # Replace NaN values in LSi with the NoData value before saving
    LSi = np.where(np.isnan(LSi), nodata_value, LSi)

    # Create output file path for the LSi raster
    lsi_path = project.catchment_path(catchment, 'Erodibility','LS_factor.tif')

    # Write the LSi raster to the specified path
    with rio.open(lsi_path, "w", **dem_meta) as out_dst:
        out_dst.write(LSi.astype(np.float32), 1)  # Write LSi data as float32

    logger.info("LS factor computed for catchment: %s",catchment)  # Print a message to indicate completion

    return slope_degrees, slope_percent, aspect_radians, specific_area, LSi  # specific_area?


DEFAULT_MAX_SDR=0.8
DEFAULT_IC0=0.5
DEFAULT_K=1

def compute_sediment_delivery_ratio(project: FireImpactsProject, catchment=None,max_sdr=DEFAULT_MAX_SDR,ic0=DEFAULT_IC0,k=DEFAULT_K):
    """
    Main function to calculate SDR, slope, flow direction, flow accumulation, and save the SDR raster.

    Parameters:
    project (fire_impacts.FireImpactsProject): Current project
    catchment (str): Name of the catchment to process. If None, process all catchments.
    max_sdr (float): Maximum value for the Sediment Delivery Ratio (SDR). Default is 0.8.
    ic0 (float): Initial value for the Connectivity Index (IC). Default is 0.5.
    k (float): Parameter for the SDR calculation. Default is 1.

    Returns:
    slope_ratio (numpy array): Slope as a ratio (unitless).
    slope_degrees (numpy array): Slope in degrees.
    distance_to_stream (numpy array): Distance (downslope length) of each to the nearest stream
    Dup (numpy array): Upslope area component.
    Ddn (numpy array): Donslope component.
    IC (numpy array): Connectivity Index for each pixel.
    SDR (numpy array): Sediment Delivery Ratio array for each pixel.
    """

    if catchment is None:
        return project.for_each_catchment(lambda c: compute_sediment_delivery_ratio(project, c,max_sdr,ic0,k))

    logger.info('Computing Sediment Delivery Ratio for catchment: %s', catchment)
    dem_path = project.catchment_path(catchment, 'Topography', 'DEM.tif')
    # Step 1: Read the DEM and calculate flow direction, accumulation, area, and slope ratio
    with rio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
        transform = src.transform
        nodata = src.nodata
        dem_profile = src.profile  # Get the metadata profile for saving

    xres = transform[0]  # Width of a pixel (east-west direction)
    yres = abs(transform[4])  # Height of a pixel (north-south direction)
    pixel_area = xres * yres  # Area of a single pixel

    # Handle NoData values
    null_mask = dem_data == nodata
    dem_data = np.where(null_mask, np.nan, dem_data) # replace -3.4028235e+38 with np.nan

    # Initialize pysheds Grid and calculate flow dir and flow accumulation and area
    grid, fdir, acc = _topographic_indices(project,catchment)
    # grid = Grid.from_raster(dem_path)
    # dem_grid = grid.read_raster(dem_path)
    # dem_filled = grid.fill_pits(dem_grid)
    # dem_filled = grid.fill_depressions(dem_filled)
    # inflated_dem = grid.resolve_flats(dem_filled)
    # # Calculate flow direction and accumulation
    # fdir = grid.flowdir(inflated_dem, dirmap=D8_FLOW_DIRECTIONS)
    # acc = grid.accumulation(fdir, dirmap=D8_FLOW_DIRECTIONS)

    # Calculate the area
    acc_data = np.array(acc, dtype=np.float32) # get the array data from raster data
    area = acc_data * pixel_area  # This is area in meter square (Multiply the flow accumulation by the resolution to get the area)

    # Calculate slope using numpy gradient
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)
    # Apply slope thresholds for connectivity index calculations
    # Set Sth values less than 0.005 to 0.005
    Sth = np.where(slope_ratio < 0.005, 0.005, np.where(slope_ratio <= 1, slope_ratio, 1)) # this also sets nan to 1, which we will return them to nan in the next two lines
    nan_mask = np.isnan(slope_ratio) # Mask the NaN valuesfrom slope_ratio
    Sth[nan_mask] = np.nan # Restore NaN values by applying the mask

    # Calculate the average thresholded slope gradient of the upslope contributing area (m/m)
    Sth_path = project.catchment_path(catchment,'Delivery', 'Sth.tif')
    with rio.open(Sth_path, 'w', **dem_meta) as dest:
        dest.write(Sth.astype('float32'), 1)
    # Read the temporary rasters into pysheds as Raster objects
    Sth_raster = grid.read_raster(Sth_path)
    acc_Sth = grid.accumulation(fdir=fdir, weights=Sth_raster) # this is accumulated c factor for each cell
    acc_Sth_arr = np.array(acc_Sth, dtype=np.float32)
    acc_no0 = np.where(acc_data == 0, np.nan, acc_data)  # avoid divide by zero
    Av_Sth = acc_Sth_arr / acc_no0 # This is avarage of Cth in each cell

    # --------------------------------------------------------------------------------------------------------------------------
    # Step 2: Read the C-factor raster and calculate Cth
    c_factor_path = project.catchment_path(catchment,'Erodibility', 'C_factor_adjusted.tif')
    with rio.open(c_factor_path) as c_factor_src:
        c_factor = c_factor_src.read(1)

    # Set a C factor thresholded (set values less than 0.001 to 0.001) and calculate verage thresholded C factor of the upslope contributing area
    Cth = np.where(c_factor < 0.001, 0.001, c_factor) # this is an array, it shoud be converted to a raster to use in pyshed grid analysis
    Cth_path = project.catchment_path(catchment,'Delivery', 'Cth.tif') # Temporary file paths
    # Write Cth arrays to temporary raster files
    dem_meta.update(dtype='float32')
    with rio.open(Cth_path, 'w', **dem_meta) as dest:
        dest.write(Cth.astype('float32'), 1)
    # Read the temporary rasters into pysheds as Raster objects
    Cth_raster = grid.read_raster(Cth_path)
    # Calculate the average thresholded C factor of the upslope contributing area
    acc_Cth = grid.accumulation(fdir=fdir, weights=Cth_raster) # this is accumulated c factor for each cell
    acc_Cth_arr = np.array(acc_Cth, dtype=np.float32)
    Av_Cth = acc_Cth_arr / acc_no0  # This is avarage of Cth in each cell
    #----------------------------------------------------------------------------------------------------------
    # Step 3: calculate downslope path distance to the nearest stream
    # Create stream network (channel domain) based on a flow accumulation threshold (e.g., 26 cells)
    #streams = acc > 26 # defines stream cells based on a flow accumulation threshold (e.g., 26 cells)
    streams = (area > 1.3e4) # define stream cells (True) based on area
    stream_cells = np.where(streams)  # returns the indices (row, col) of all cells in the streams array that have a value of True (as a tuple of two arrays)
    streams_path = project.catchment_path(catchment,'Delivery', 'Streams.tif')
    with rio.open(streams_path, 'w', **dem_meta) as dest:
        dest.write(streams.astype(np.uint8),1)
    # Initialize output array for storing downslope distances
    distance_to_stream = np.full_like(dem_data, 0)
    Ddn = np.full_like(dem_data, 0.0, dtype=np.float32)  # Initialize the Ddn array with zeros

    # Initialize a list of stream cells to start from
    st_indices = list(zip(stream_cells[0], stream_cells[1])) # get list of stream cell coordinates/indexes

    # map neighboring cell for D8 flow directions (north, northeast, east, southeast, south, southwest, west, northwest)
    dy = np.array([-1, -1, 0, 1, 1, 1, 0, -1])  # row (move to north (-1) or south (1), or the diagonals) so dx=0 and dy=1 is south;  dx=-1 and dy=1 is southwest
    dx = np.array([0, 1, 1, 1, 0, -1, -1, -1])  # column (move to east (1) or west (-1), or the diagonals) so dx=1 and dy=0 is east;  dx=1 and dy=-1 is northeast
    # calculate diagonal distance
    diag_cell_size = (xres**2 + yres**2) ** 0.5  # calculate diagonal distance (approximatly 41 m for resultion 29 m)
    # Create an array to store the distances between the grid cell and its eight neighboring cells for D8 flow direction
    grid_lengths = np.array([yres, diag_cell_size, xres, diag_cell_size,
                             yres, diag_cell_size, xres, diag_cell_size])  # array([28.02902639, 39.63902926, 28.02902639, 39.63902926, 28.02902639,
                                                                                                 # 39.63902926, 28.02902639, 39.63902926])
    # Track visited cells (creatw a boolean array to keep track whether each cell in the grid has already been processed during the downslope distance calculation)
    visited = np.zeros_like(dem_data, dtype=bool) # creates visited array with the same shape and size as the dem array with all cells set to False
    visited[stream_cells] = True  # where False represents "not visited" and True represents "visited."

    # Start the downslope path traversal to calculate Ddn
    while st_indices:
        # Get the current cell from the st_indices
        row, col = st_indices.pop(0)
        current_distance = distance_to_stream[row, col] # at initial (first iteration) the current distance is 0, after that gets a new distance in each iteration

        # Check the 8 neighbors
        for i in range(8):  # D8
            new_row = row + dy[i]
            new_col = col + dx[i]

            # Ensure the neighbor is within the grid bounds
            if 0 <= new_row < dem_data.shape[0] and 0 <= new_col < dem_data.shape[1]:
                # Check if the flow direction leads to this neighbor cell
                if fdir[new_row, new_col] == D8_FLOW_DIRECTIONS[(i + 4) % 8]:
                    # If the neighbor hasn't been visited, calculate its Ddn contribution
                    if not visited[new_row, new_col]: # check if the neighboring cell (at new_row, new_col) has already been visited (If the cell hasn't been visited, the distance to the stream for this cell will be calculated)
                        visited[new_row, new_col] = True # mark the neighboring cell as "visited" so that it won't be processed again in future iterations
                        # Calculate the Ddn component for the cell
                        if Cth[new_row, new_col] > 0 and Sth[new_row, new_col] > 0:
                            downslope_component = grid_lengths[i] / (Cth[new_row, new_col] * Sth[new_row, new_col])
                        else:
                            downslope_component = 0

                        # Accumulate the Ddn value
                        Ddn[new_row, new_col] = Ddn[row, col] + downslope_component

                        # Update distance to stream
                        distance_to_stream[new_row, new_col] = current_distance + grid_lengths[i]

                        # Add the neighbor to the queue
                        st_indices.append((new_row, new_col))

    # Mask zeros outside the catchment boundary by checking against the nodata values in the DEM
    distance_to_stream[null_mask] = np.nan
    # Update the metadata to float32 and set nodata value
    dem_profile.update(dtype=rio.float32, nodata=np.nan)
    dist_path = project.catchment_path(catchment,'Delivery', 'Distance_to_stream.tif')
    with rio.open(dist_path, 'w', **dem_profile) as dst:
        dst.write(distance_to_stream.astype(rio.float32), 1)

    # Mask zeros outside the catchment boundary by checking against the nodata values in the DEM
    Ddn[null_mask] = np.nan
    Ddn_path = project.catchment_path(catchment,'Delivery', 'Ddn.tif')
    with rio.open(Ddn_path, 'w', **dem_profile) as dst:
        dst.write(Ddn.astype(rio.float32), 1)
    # ------------------------------------------------------------------------------------------------------------
    # Step 4: calculate the upslope component Dup, Connectivity Index (IC) and then SDR
    # Calculate the upslope component Dup
    Dup = Av_Cth * Av_Sth * np.sqrt(area)
    # Mask zeros outside the catchment boundary by checking against the nodata values in the DEM
    Dup[null_mask] = np.nan
    Dup_path = project.catchment_path(catchment,'Delivery', 'Dup.tif')
    with rio.open(Dup_path, 'w', **dem_profile) as dst:
        dst.write(Dup.astype(rio.float32), 1)
    EPS = 1 #A lower bound is set to avoid infinite values for IC.
    Ddn = np.where(Ddn <= 0, EPS, Ddn)
    # Calculate Connectivity Index (IC)
    IC = np.log10(Dup / Ddn)
    # Mask zeros outside the catchment boundary by checking against the nodata values in the DEM
    IC[null_mask] = np.nan
    IC_path = project.catchment_path(catchment,'Delivery', 'IC.tif')
    with rio.open(IC_path, 'w', **dem_profile) as dst:
        dst.write(IC.astype(rio.float32), 1)

    # Calculate Sediment Delivery Ratio (SDR)
    SDR = max_sdr / (1 + np.exp((ic0 - IC) / k))

    # Save SDR output as a raster
    output_sdr_path = project.catchment_path(catchment,'Delivery', "SDR.tif")
    dem_profile.update(dtype=rio.float32)  # Update profile for output
    with rio.open(output_sdr_path, 'w', **dem_profile) as dst:
        dst.write(SDR.astype(np.float32), 1)

    logger.info("Sediment Delivery Ratio computed for catchment: %s", catchment)

    return slope_ratio, fdir, acc, distance_to_stream, IC, Dup, Ddn, SDR