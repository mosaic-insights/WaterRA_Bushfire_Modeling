from fire_impacts.pre.util import clip_and_reproject_raster, read_raster, read_aligned
import rasterio as rio
from .project import FireImpactsProject
from pysheds.grid import Grid
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


def compute_adjusted_k_c(proj: FireImpactsProject, catchment: str, c_factor_fn: str, k_factor_fn: str):
    if catchment is None:
        proj.for_each_catchment(lambda c: compute_adjusted_k_c(
            proj, c, c_factor_fn, k_factor_fn))
        return

    # bounds = proj.catchment_bounds(catchment)
    shp = proj.boundary_files[catchment]
    clip_and_reproject_raster(c_factor_fn, shp, proj.catchment_path(
        catchment, 'Erodibility', 'C_factor.tif'))
    clip_and_reproject_raster(k_factor_fn, shp, proj.catchment_path(
        catchment, 'Erodibility', 'K_factor.tif'))

    dem_fn = proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    dem, dem_transform, dem_crs = read_raster(dem_fn)
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
    grid = Grid.from_raster(dem_path)
    # 3π/2 radians (or 270°) indicates a west-facing slope.
    dem_grid = grid.read_raster(dem_path)

    # Fill pits and depressions in the DEM to ensure correct flow direction
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.fill_depressions(dem_filled)
    # Ensure that flat areas drain correctly
    inflated_dem = grid.resolve_flats(dem_filled)

    # Define the flow direction mapping (dirmap) and calculate flow direction
    # Directional map for flow direction
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # Calculate flow accumulation (number of upstream cells contributing to each cell)
    acc = grid.accumulation(fdir, dirmap=dirmap)
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
    D = xres  # Grid cell dimension (same as pixel size in DEM)
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
