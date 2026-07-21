"""
RUSLE pre-processing: fire-adjusted C/K factors, LSI, and SDR computation.

Computes the fire-adjusted cover (C) and erodibility (K) factors from
dNBR and aridity, the slope-length-gradient (LSI) factor from the DEM,
and the Sediment Delivery Ratio (SDR) from the hydrological connectivity
index.  All outputs are written to the project's Erodibility and Delivery
folders.
"""

from fire_impacts.pre.util import (
    clip_and_reproject_raster, read_raster, read_aligned
)
from fire_impacts import const as c
import rasterio as rio
from .project import FireImpactsProject
from .topography import D8_FLOW_DIRECTIONS
from .data_sources import CSIRO_C_FACTOR_GRID, CSIRO_K_FACTOR_GRID
from pysheds.grid import Grid
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adjusted K and C factor computation
# ---------------------------------------------------------------------------

def compute_adjusted_k_c(
    proj: FireImpactsProject,
    catchment: str,
    c_factor_fn: str = None,
    k_factor_fn: str = None,
    compute_lsi_factor: bool = True,
    compute_sdr: bool = True,
    recovery_times=None,
):
    """
    Compute fire-adjusted C and K factors and prepare RUSLE inputs.

    Parameters:
    - proj: FireImpactsProject instance.
    - catchment: Name of catchment to process. If None, run for
      all catchments.
    - c_factor_fn: Path to C-factor raster. Defaults to CSIRO
      grid.
    - k_factor_fn: Path to K-factor raster. Defaults to CSIRO
      grid.
    - compute_lsi_factor: If True, also compute the LSI factor.
    - compute_sdr: If True, also compute the SDR.

    Returns:
    - None. Outputs are written to project raster files.
    """
    if catchment is None:
        proj.for_each_catchment(lambda catchment_name: compute_adjusted_k_c(
            proj,
            catchment_name,
            c_factor_fn=c_factor_fn,
            k_factor_fn=k_factor_fn,
            compute_lsi_factor=compute_lsi_factor,
            compute_sdr=compute_sdr,
            recovery_times=recovery_times,
        ))
        return

    shp = proj.boundary_files[catchment]

    # Create a constant C-factor layer with value 0.01 for all valid DEM cells.
    c_factor_out = proj.catchment_path(catchment, 'Erodibility', 'C_factor.tif')

    if c_factor_fn is None:
        dem_fn = proj.catchment_path(catchment, 'Topography', 'DEM.tif')

        with rio.open(dem_fn) as dem_src:
            dem_data = dem_src.read(1)
            dem_meta = dem_src.meta.copy()
            dem_nodata = dem_src.nodata

        C_default = np.full(dem_data.shape, 0.01, dtype=np.float32)

        if dem_nodata is not None:
            C_default = np.where(dem_data == dem_nodata, np.nan, C_default)

        dem_meta.update({
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw"
        })

        with rio.open(c_factor_out, "w", **dem_meta) as dest:
            dest.write(C_default, 1)

    else:
        # If the user provides a C-factor raster, clip/reproject it to the catchment.
        clip_and_reproject_raster(
            c_factor_fn,
            shp,
            c_factor_out
        )

    if k_factor_fn is None:
        k_factor_fn = CSIRO_K_FACTOR_GRID
    clip_and_reproject_raster(
        k_factor_fn, shp,
        proj.catchment_path(catchment, 'Erodibility', 'K_factor.tif')
    )

    dem_fn = proj.catchment_path(catchment, 'Topography', 'DEM.tif')
    dem, dem_meta = read_raster(dem_fn)
    dem_transform = dem_meta['transform']
    dem_crs = dem_meta['crs']
    dNBR = read_aligned(
        proj.catchment_path(
            catchment, 'FireSeverity', 'masked_dNBR.tif'),
        dem_transform, dem_crs, dem.shape
    )
    Cbase = read_aligned(
        proj.catchment_path(catchment, 'Erodibility', 'C_factor.tif'),
        dem_transform, dem_crs, dem.shape
    )
    Kbase = read_aligned(
        proj.catchment_path(catchment, 'Erodibility', 'K_factor.tif'),
        dem_transform, dem_crs, dem.shape
    )
    AI = read_aligned(
        proj.catchment_path(catchment, 'Soils', 'Aridity.tif'),
        dem_transform, dem_crs, dem.shape
    )

    # Model parameters
    if recovery_times is None:
        recovery_times = c.DEFAULT_RECOVERY_TIMES
    x_c = 0.4
    x_k = 1
    Kfire = 0.081
    Cpeak = 0.35

    # Compute fire-adjusted C factor using dNBR
    CdNBR = dNBR * 1000
    CdNBR[CdNBR < 0] = 0
    dNBRmask = (CdNBR > 0) & (CdNBR <= 400)
    CdNBR[dNBRmask] = (
        Cbase[dNBRmask]
        + ((Cpeak - Cbase[dNBRmask]) * (CdNBR[dNBRmask] / 400))
    )
    CdNBR[CdNBR > 400] = Cpeak
    
    # LS factor does not depend on T, compute it once only.
    if compute_lsi_factor:
        compute_lsi(proj, catchment)

    for recovery_time in recovery_times:
        suffix = c.recovery_time_suffix(recovery_time)

        logger.info(
            "Computing fire-adjusted C/K factors for catchment %s at T=%s years",
            catchment, recovery_time
        )

        C = (
            (CdNBR - Cbase)
            * np.exp(-recovery_time / (x_c * AI))
            + Cbase
        )
        K = (
            (Kfire - Kbase)
            * np.exp(-recovery_time / (x_k * AI))
            + Kbase
        )

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

        c_out = proj.catchment_path(
            catchment, 'Erodibility', f'C_factor_adjusted_{suffix}.tif')
        with rio.open(c_out, 'w', **out_meta) as dest:
            dest.write(C.astype(np.float32), 1)

        k_out = proj.catchment_path(
            catchment, 'Erodibility', f'K_factor_adjusted_{suffix}.tif')
        with rio.open(k_out, 'w', **out_meta) as dest:
            dest.write(K.astype(np.float32), 1)

        if compute_sdr:
            compute_sediment_delivery_ratio(
                proj,
                catchment,
                c_factor_path=c_out,
                output_suffix=suffix,
            )

# ---------------------------------------------------------------------------
# Topographic index helper
# ---------------------------------------------------------------------------

def _topographic_indices(
    project: FireImpactsProject,
    catchment: str,
):
    """
    Compute D8 flow direction and accumulation for a catchment DEM.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment: Name of the catchment to process.

    Returns:
    - grid: Pysheds Grid initialised from the DEM.
    - fdir: Flow direction raster.
    - acc: Flow accumulation raster.
    """
    dem_path = project.catchment_path(
        catchment, 'Topography', 'DEM.tif')
    grid = Grid.from_raster(dem_path)
    dem_grid = grid.read_raster(dem_path)
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.fill_depressions(dem_filled)
    inflated_dem = grid.resolve_flats(dem_filled)
    # Calculate flow direction and accumulation
    fdir = grid.flowdir(inflated_dem, dirmap=D8_FLOW_DIRECTIONS)
    acc = grid.accumulation(fdir, dirmap=D8_FLOW_DIRECTIONS)
    return grid, fdir, acc


# ---------------------------------------------------------------------------
# LSI factor computation
# ---------------------------------------------------------------------------

def compute_lsi(project: FireImpactsProject, catchment=None):
    """
    Calculate the LSi (slope length-gradient) factor from a DEM.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment: Name of catchment to process. If None, process
      all catchments.

    Returns:
    - slope_degrees: Slope in degrees for each pixel.
    - slope_percent: Slope as a percentage for each pixel.
    - aspect_radians: Aspect (direction of steepest slope) in
      radians for each pixel.
    - specific_area: Specific catchment area (Ai_in) in metres
      for each pixel.
    - LSi: Slope length-gradient factor for each pixel.
    """
    if catchment is None:
        return project.for_each_catchment(
            lambda c: compute_lsi(project, c))

    logger.info('Computing LSI factor for catchment: %s', catchment)
    dem_path = project.catchment_path(
        catchment, 'Topography', 'DEM.tif')

    # Open the DEM raster and extract the elevation data
    with rio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        nodata = src.nodata

    # Get pixel resolution (grid cell size)
    xres = transform[0]       # east-west pixel width
    yres = abs(transform[4])  # north-south pixel height
    pixel_area = xres * yres

    # Replace NoData values with NaN for numerical operations
    dem_data = np.where(dem_data == nodata, np.nan, dem_data)

    # Calculate slope from elevation gradients
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_radians = np.arctan(slope_ratio)
    slope_degrees = np.degrees(slope_radians)
    slope_percent = slope_ratio * 100

    # Calculate aspect (direction of steepest descent).
    # Convention: 0 rad = north-facing, π/2 rad = east-facing,
    # π rad = south-facing.
    aspect_radians = np.arctan2(dz_dy, -dz_dx)
    # Ensure aspect is in the range [0, 2π]
    aspect_radians = np.where(
        aspect_radians < 0,
        2 * np.pi + aspect_radians,
        aspect_radians
    )

    # Compute flow accumulation via pysheds
    _, _, acc = _topographic_indices(project, catchment)
    acc_data = np.array(acc, dtype=np.float32)

    # Estimate specific catchment area (Ai_in) in metres
    specific_area = np.sqrt(acc_data * pixel_area)
    # Cap slope length to avoid LS factor overestimation in
    # heterogeneous landscapes (sqrt(area in m2) ≤ 141 m)
    specific_area = np.where(specific_area > 141, 141, specific_area)

    # Aspect length factor (xi)
    xi = (
        np.abs(np.sin(aspect_radians))
        + np.abs(np.cos(aspect_radians))
    )

    # Slope factor (Si), split by slope percentage threshold
    Si = np.zeros_like(slope_percent)
    Si[slope_percent < 9] = (
        10.8 * np.sin(slope_radians[slope_percent < 9]) + 0.03
    )
    Si[slope_percent >= 9] = (
        16.8 * np.sin(slope_radians[slope_percent >= 9]) - 0.50
    )

    # RUSLE length exponent (m), based on slope percentage class
    m = np.zeros_like(slope_percent)
    m[slope_percent <= 1] = 0.2
    m[(slope_percent > 1) & (slope_percent <= 3.5)] = 0.3
    m[(slope_percent > 3.5) & (slope_percent <= 5)] = 0.4
    m[(slope_percent > 5) & (slope_percent <= 9)] = 0.5

    # For slopes > 9%, use the beta-based McCool et al. formula
    mask = slope_percent > 9
    if np.any(mask):
        slope_radians_high = np.arctan(slope_percent[mask] / 100)
        beta = (
            (np.sin(slope_radians_high) / 0.0896)
            / (3 * np.sin(slope_radians_high)**0.8 + 0.56)
        )
        m[mask] = beta / (1 + beta)

    # Calculate LSi factor (slope length-gradient factor)
    D = np.sqrt(pixel_area)
    LSi = (
        Si
        * (
            ((specific_area + D**2)**(m + 1))
            - (specific_area**(m + 1))
        )
        / ((D**(m + 2)) * (xi**m) * (22.13**m))
    )

    # Write output raster, replacing NaN with the nodata value
    nodata_value = 0.0
    with rio.open(dem_path) as src:
        dem_meta = src.meta.copy()
    dem_meta.update({'dtype': rio.float32, 'nodata': nodata_value})
    LSi = np.where(np.isnan(LSi), nodata_value, LSi)

    lsi_path = project.catchment_path(
        catchment, 'Erodibility', 'LS_factor.tif')
    with rio.open(lsi_path, 'w', **dem_meta) as out_dst:
        out_dst.write(LSi.astype(np.float32), 1)

    logger.info('LS factor computed for catchment: %s', catchment)

    return slope_degrees, slope_percent, aspect_radians, specific_area, LSi


# ---------------------------------------------------------------------------
# Sediment Delivery Ratio computation
# ---------------------------------------------------------------------------

DEFAULT_MAX_SDR = 0.8
DEFAULT_IC0 = 0.5
DEFAULT_K = 1


def compute_sediment_delivery_ratio(
    project: FireImpactsProject,
    catchment=None,
    max_sdr=DEFAULT_MAX_SDR,
    ic0=DEFAULT_IC0,
    k=DEFAULT_K,
    c_factor_path=None,
    output_suffix=None,
):
    """
    Calculate the Sediment Delivery Ratio (SDR) for a catchment.

    Parameters:
    - project: FireImpactsProject instance.
    - catchment: Name of catchment to process. If None, process
      all catchments.
    - max_sdr: Maximum SDR value. Default is 0.8.
    - ic0: Calibration parameter for the IC-SDR relationship.
      Default is 0.5.
    - k: Shape parameter for the IC-SDR relationship.
      Default is 1.

    Returns:
    - slope_ratio: Slope as a dimensionless ratio.
    - fdir: D8 flow direction raster.
    - acc: Flow accumulation raster.
    - distance_to_stream: Downslope distance to nearest stream
      for each cell.
    - IC: Connectivity index for each cell.
    - Dup: Upslope component of the connectivity index.
    - Ddn: Downslope component of the connectivity index.
    - SDR: Sediment Delivery Ratio for each cell.
    """
    if catchment is None:
        return project.for_each_catchment(
            lambda c: compute_sediment_delivery_ratio(
                project, c, max_sdr, ic0, k, c_factor_path, output_suffix))

    logger.info(
        'Computing Sediment Delivery Ratio for catchment: %s',
        catchment)

    # ------------------------------------------------------------------
    # Step 1: Read DEM and compute flow direction, accumulation, slope
    # ------------------------------------------------------------------
    dem_path = project.catchment_path(
        catchment, 'Topography', 'DEM.tif')
    with rio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
        transform = src.transform
        nodata = src.nodata
        dem_profile = src.profile

    xres = transform[0]       # east-west pixel width
    yres = abs(transform[4])  # north-south pixel height
    pixel_area = xres * yres

    # Replace nodata with NaN
    null_mask = dem_data == nodata
    dem_data = np.where(null_mask, np.nan, dem_data)

    # Compute flow direction and accumulation via pysheds
    grid, fdir, acc = _topographic_indices(project, catchment)
    logger.info('Flow direction and accumulation computed')

    acc_data = np.array(acc, dtype=np.float32)
    # Area in square metres (flow accumulation × pixel area)
    area = acc_data * pixel_area

    # Compute slope and apply thresholds for connectivity index.
    # Slope is clamped to [0.005, 1]; NaN cells are restored below.
    dz_dx, dz_dy = np.gradient(dem_data, xres, yres)
    slope_ratio = np.sqrt(dz_dx**2 + dz_dy**2)
    Sth = np.where(
        slope_ratio < 0.005, 0.005,
        np.where(slope_ratio <= 1, slope_ratio, 1)
    )
    nan_mask = np.isnan(slope_ratio)
    Sth[nan_mask] = np.nan

    # Accumulate thresholded slope over upslope contributing area
    Sth_path = project.catchment_path(
        catchment, 'Delivery', 'Sth.tif')
    with rio.open(Sth_path, 'w', **dem_meta) as dest:
        dest.write(Sth.astype('float32'), 1)
    Sth_raster = grid.read_raster(Sth_path)
    acc_Sth = grid.accumulation(fdir=fdir, weights=Sth_raster)
    acc_Sth_arr = np.array(acc_Sth, dtype=np.float32)
    # Avoid divide-by-zero when computing upslope averages
    acc_no0 = np.where(acc_data == 0, np.nan, acc_data)
    Av_Sth = acc_Sth_arr / acc_no0

    logger.info('Upslope slope averages (Sth) computed')

    # ------------------------------------------------------------------
    # Step 2: C-factor — compute thresholded upslope averages
    # ------------------------------------------------------------------
    suffix_text = f"_{output_suffix}" if output_suffix else ""
    with rio.open(c_factor_path) as c_factor_src:
        c_factor = c_factor_src.read(1)

    # Threshold C factor to a minimum of 0.001
    Cth = np.where(c_factor < 0.001, 0.001, c_factor)
    Cth_path = project.catchment_path(
        catchment, 'Delivery', f'Cth{suffix_text}.tif')
    dem_meta.update(dtype='float32')
    with rio.open(Cth_path, 'w', **dem_meta) as dest:
        dest.write(Cth.astype('float32'), 1)
    Cth_raster = grid.read_raster(Cth_path)
    acc_Cth = grid.accumulation(fdir=fdir, weights=Cth_raster)
    acc_Cth_arr = np.array(acc_Cth, dtype=np.float32)
    Av_Cth = acc_Cth_arr / acc_no0

    logger.info('Upslope C-factor averages (Cth) computed')

    # ------------------------------------------------------------------
    # Step 3: Downslope path distance to nearest stream
    # ------------------------------------------------------------------

    # Define stream network based on contributing area threshold
    streams = area > 1.3e4
    stream_cells = np.where(streams)
    streams_path = project.catchment_path(
        catchment, 'Delivery', 'Streams.tif')
    with rio.open(streams_path, 'w', **dem_meta) as dest:
        dest.write(streams.astype(np.uint8), 1)

    # Initialise output arrays
    distance_to_stream = np.full_like(dem_data, 0)
    Ddn = np.full_like(dem_data, 0.0, dtype=np.float32)

    # D8 neighbour offsets (N, NE, E, SE, S, SW, W, NW)
    dy = np.array([-1, -1, 0, 1, 1,  1,  0, -1])
    dx = np.array([ 0,  1, 1, 1, 0, -1, -1, -1])
    diag_cell_size = (xres**2 + yres**2) ** 0.5
    grid_lengths = np.array([
        yres, diag_cell_size, xres, diag_cell_size,
        yres, diag_cell_size, xres, diag_cell_size,
    ])

    # BFS outward from stream cells to accumulate Ddn
    visited = np.zeros_like(dem_data, dtype=bool)
    visited[stream_cells] = True
    st_indices = list(zip(stream_cells[0], stream_cells[1]))

    logger.info(
        'Computing downslope path distances for catchment: %s'
        ' (%d stream seed cells)...',
        catchment, len(st_indices))

    while st_indices:
        row, col = st_indices.pop(0)
        current_distance = distance_to_stream[row, col]

        for i in range(8):
            new_row = row + dy[i]
            new_col = col + dx[i]

            # Ensure the neighbour is within the grid bounds
            if (0 <= new_row < dem_data.shape[0]
                    and 0 <= new_col < dem_data.shape[1]):
                # Check if the flow direction leads to this neighbour
                if (fdir[new_row, new_col]
                        == D8_FLOW_DIRECTIONS[(i + 4) % 8]):
                    if not visited[new_row, new_col]:
                        visited[new_row, new_col] = True
                        if (Cth[new_row, new_col] > 0
                                and Sth[new_row, new_col] > 0):
                            downslope_component = (
                                grid_lengths[i]
                                / (Cth[new_row, new_col]
                                   * Sth[new_row, new_col])
                            )
                        else:
                            downslope_component = 0
                        Ddn[new_row, new_col] = (
                            Ddn[row, col] + downslope_component)
                        distance_to_stream[new_row, new_col] = (
                            current_distance + grid_lengths[i])
                        st_indices.append((new_row, new_col))

    logger.info('Downslope path distances computed')

    # Mask nodata cells and write Ddn and distance outputs
    distance_to_stream[null_mask] = np.nan
    dem_profile.update(dtype=rio.float32, nodata=np.nan)
    dist_path = project.catchment_path(
        catchment, 'Delivery', 'Distance_to_stream.tif')
    with rio.open(dist_path, 'w', **dem_profile) as dst:
        dst.write(distance_to_stream.astype(rio.float32), 1)

    Ddn[null_mask] = np.nan
    Ddn_path = project.catchment_path(
        catchment, 'Delivery', f'Ddn{suffix_text}.tif')
    with rio.open(Ddn_path, 'w', **dem_profile) as dst:
        dst.write(Ddn.astype(rio.float32), 1)

    # ------------------------------------------------------------------
    # Step 4: Upslope component, Connectivity Index, and SDR
    # ------------------------------------------------------------------

    # Calculate the upslope component Dup
    Dup = Av_Cth * Av_Sth * np.sqrt(area)
    Dup[null_mask] = np.nan
    Dup_path = project.catchment_path(
        catchment, 'Delivery', f'Dup{suffix_text}.tif')
    with rio.open(Dup_path, 'w', **dem_profile) as dst:
        dst.write(Dup.astype(rio.float32), 1)

    # Lower bound on Ddn to avoid log10(0) in IC calculation
    EPS = 1
    Ddn = np.where(Ddn <= 0, EPS, Ddn)

    # Calculate Connectivity Index (IC)
    IC = np.log10(Dup / Ddn)
    IC[null_mask] = np.nan
    IC_path = project.catchment_path(
        catchment, 'Delivery', f'IC{suffix_text}.tif')
    with rio.open(IC_path, 'w', **dem_profile) as dst:
        dst.write(IC.astype(rio.float32), 1)

    logger.info('Connectivity index (IC) computed')

    # Calculate and save Sediment Delivery Ratio
    SDR = max_sdr / (1 + np.exp((ic0 - IC) / k))
    output_sdr_path = project.catchment_path(
        catchment, 'Delivery', f'SDR{suffix_text}.tif')
    dem_profile.update(dtype=rio.float32)
    with rio.open(output_sdr_path, 'w', **dem_profile) as dst:
        dst.write(SDR.astype(np.float32), 1)

    logger.info(
        'Sediment Delivery Ratio computed for catchment: %s',
        catchment)

    return slope_ratio, fdir, acc, distance_to_stream, IC, Dup, Ddn, SDR
