"""
RUSLE pre-processing: fire-adjusted C/K factors, LSI, and SDR computation.

Computes the fire-adjusted cover (C) and erodibility (K) factors from
dNBR and aridity, the slope-length-gradient (LSI) factor from the DEM,
and the Sediment Delivery Ratio (SDR) from the hydrological connectivity
index.  All outputs are written to the project's Erodibility and Delivery
folders.
"""

from fire_impacts.pre.util import (
    clip_and_reproject_raster, read_raster_masked, read_aligned_like,
    write_raster, slope_from_dem, dem_flow_layers, upslope_weighted_mean
)
from fire_impacts import const as c
from .project import FireImpactsProject
from ..context import RunContext
from .topography import D8_FLOW_DIRECTIONS
from .data_sources import CSIRO_C_FACTOR_GRID, CSIRO_K_FACTOR_GRID
import numpy as np
import os
import logging
import warnings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adjusted K and C factor computation
# ---------------------------------------------------------------------------

def compute_adjusted_k_c(
    ctx: RunContext,
    c_factor_fn: str = None,
    k_factor_fn: str = None,
    compute_lsi_factor: bool = True,
    compute_sdr: bool = True,
    recovery_breakpoints=None,
    recovery_times=None,
):
    """
    Compute fire-adjusted C and K factors and prepare RUSLE inputs.

    Parameters:
    - ctx: event-level RunContext identifying the catchment + event.
    - c_factor_fn: Path to C-factor raster. When None, a constant
      C factor of 0.01 is written over the valid DEM cells.
    - k_factor_fn: Path to K-factor raster. Defaults to CSIRO grid.
    - compute_lsi_factor: If True, also compute the LSI factor
      (static — written at catchment level).
    - compute_sdr: If True, also compute the per-recovery SDRs (event
      level) and the baseline SDR (catchment level).
    - recovery_breakpoints: Monotonic array of years-since-fire boundaries
      defining the recovery windows (n+1 breakpoints -> n windows; window
      i is modelled at recovery time b_i). Defaults to
      const.DEFAULT_RECOVERY_BREAKPOINTS. Persisted into the event's
      event.json so the simulation step doesn't need it re-specified.
    - recovery_times: Deprecated. The old list of window-start times; if
      given it is converted to breakpoints using the default interval.

    Returns:
    - None. Outputs are written to project raster files.
    """
    ctx.validate()
    project = ctx.project
    catchment = ctx.catchment

    shp = project.boundary_files[catchment]

    dem_fn = ctx.catchment_path('Topography', 'DEM.tif')
    dem_grid = read_raster_masked(dem_fn)

    # Base C and K factors are static (do not depend on the fire) and
    # live at the catchment level.
    os.makedirs(ctx.catchment_path('Erodibility'), exist_ok=True)
    c_factor_out = ctx.catchment_path('Erodibility', 'C_factor.tif')

    if c_factor_fn is None:
        # Create a constant C-factor layer of 0.01 over valid DEM cells.
        C_default = np.full(dem_grid.shape, 0.01, dtype=np.float32)
        C_default = np.where(dem_grid.nodata_mask, np.nan, C_default)
        write_raster(c_factor_out, C_default, dem_grid.meta())

    else:
        # A user-supplied C-factor raster is clipped to the catchment.
        clip_and_reproject_raster(c_factor_fn, shp, c_factor_out)

    if k_factor_fn is None:
        k_factor_fn = CSIRO_K_FACTOR_GRID
    clip_and_reproject_raster(
        k_factor_fn, shp,
        ctx.catchment_path('Erodibility', 'K_factor.tif'),
    )

    # dNBR is fire-event-specific.
    dNBR = read_aligned_like(
        ctx.event_path('FireSeverity', 'masked_dNBR.tif'), dem_grid,
    )
    Cbase = read_aligned_like(
        ctx.catchment_path('Erodibility', 'C_factor.tif'), dem_grid,
    )
    Kbase = read_aligned_like(
        ctx.catchment_path('Erodibility', 'K_factor.tif'), dem_grid,
    )
    AI = read_aligned_like(
        ctx.catchment_path('Soils', 'Aridity.tif'), dem_grid,
    )

    # Model parameters
    # Resolve recovery breakpoints (a single monotonic array). recovery_times
    # is deprecated: convert it to breakpoints by closing the final window
    # with the default interval.
    if recovery_times is not None:
        warnings.warn(
            "recovery_times is deprecated; pass recovery_breakpoints "
            "(a single monotonic array) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if recovery_breakpoints is None:
            recovery_breakpoints = c.breakpoints_from_times_and_interval(
                recovery_times, c.DEFAULT_RECOVERY_INTERVAL_YEARS)
    if recovery_breakpoints is None:
        recovery_breakpoints = c.DEFAULT_RECOVERY_BREAKPOINTS
    # Each recovery window is modelled at its start time (C/K evaluated at
    # the window start).
    recovery_start_times = [
        start for start, _ in c.recovery_windows(recovery_breakpoints)
    ]
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

    # Flow direction/accumulation are needed by both the LS factor and
    # every SDR below. Condition the DEM once and share the result rather
    # than recomputing it per output (previously it was recomputed for
    # the LS factor and again for every per-recovery and baseline SDR).
    topo = None
    if compute_lsi_factor or compute_sdr:
        topo = dem_flow_layers(dem_fn)

    # LS factor is static (independent of both the fire and the recovery
    # time) — written once at the catchment level.
    if compute_lsi_factor:
        compute_lsi(ctx, topo=topo)

    # The fire-adjusted layers are per-event and per-recovery-time.
    event_erod_dir = ctx.event_path('Erodibility')
    os.makedirs(event_erod_dir, exist_ok=True)

    out_meta = dem_grid.meta()

    for recovery_time in recovery_start_times:
        suffix = c.recovery_time_suffix(recovery_time)

        logger.info(
            'Computing fire-adjusted C/K factors for catchment %s '
            'event %s at T=%s years',
            catchment, ctx.event, recovery_time
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

        c_out = os.path.join(
            event_erod_dir, f'C_factor_adjusted_{suffix}.tif')
        write_raster(c_out, C, out_meta)

        k_out = os.path.join(
            event_erod_dir, f'K_factor_adjusted_{suffix}.tif')
        write_raster(k_out, K, out_meta)

        if compute_sdr:
            compute_sediment_delivery_ratio(
                ctx,
                c_factor_path=c_out,
                output_suffix=suffix,
                topo=topo,
            )

    # Also compute the baseline (no-fire) SDR from the unadjusted C factor,
    # so the baseline simulation has SDR_baseline.tif available without a
    # separate step. It uses the base C_factor.tif and so is
    # fire-independent: written at catchment scope (see
    # compute_sediment_delivery_ratio).
    if compute_sdr:
        compute_sediment_delivery_ratio(
            ctx, output_suffix='baseline', topo=topo,
        )

    # Persist the recovery breakpoints into the event definition so the
    # simulation step can read them back instead of re-specifying them.
    ctx.set_recovery_breakpoints(recovery_breakpoints)

# ---------------------------------------------------------------------------
# Topographic index helper
# ---------------------------------------------------------------------------

def _topographic_indices(ctx: RunContext):
    """
    Compute D8 flow direction and accumulation for the context's catchment.

    Returns:
    - grid: Pysheds Grid initialised from the DEM.
    - fdir: Flow direction raster.
    - acc: Flow accumulation raster.
    """
    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    return dem_flow_layers(dem_path, dirmap=D8_FLOW_DIRECTIONS)


# ---------------------------------------------------------------------------
# LSI factor computation
# ---------------------------------------------------------------------------

def compute_lsi(ctx: RunContext, topo=None):
    """
    Calculate the LSi (slope length-gradient) factor from a DEM.

    LS factor is static (does not depend on the fire), so only
    ctx.catchment is used; the event/ensemble fields are ignored.

    Parameters:
    - ctx: catchment-only RunContext.
    - topo: optional precomputed (grid, fdir, acc) tuple from
      dem_flow_layers() for the catchment DEM, to avoid reconditioning
      the DEM when the caller already has one.

    Returns:
    - slope_degrees: Slope in degrees for each pixel.
    - slope_percent: Slope as a percentage for each pixel.
    - aspect_radians: Aspect (direction of steepest slope) in
      radians for each pixel.
    - specific_area: Specific catchment area (Ai_in) in metres
      for each pixel.
    - LSi: Slope length-gradient factor for each pixel.
    """
    catchment = ctx.catchment
    logger.info('Computing LSI factor for catchment: %s', catchment)
    dem_path = ctx.catchment_path('Topography', 'DEM.tif')

    # Open the DEM raster (nodata comes back as NaN)
    dem_grid = read_raster_masked(dem_path)
    dem_data = dem_grid.data

    # Get pixel resolution (grid cell size)
    xres = dem_grid.xres
    yres = dem_grid.yres
    pixel_area = dem_grid.pixel_area

    # Calculate slope from elevation gradients
    slope_ratio, dz_dx, dz_dy = slope_from_dem(dem_data, xres, yres)
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

    # Compute flow accumulation via pysheds (or reuse the caller's)
    if topo is None:
        topo = _topographic_indices(ctx)
    _, _, acc = topo
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
    LSi = np.where(np.isnan(LSi), nodata_value, LSi)

    lsi_path = ctx.catchment_path('Erodibility', 'LS_factor.tif')
    write_raster(lsi_path, LSi, dem_grid.meta(nodata=nodata_value))

    logger.info('LS factor computed for catchment: %s', catchment)

    return slope_degrees, slope_percent, aspect_radians, specific_area, LSi


# ---------------------------------------------------------------------------
# Sediment Delivery Ratio computation
# ---------------------------------------------------------------------------

DEFAULT_MAX_SDR = 0.8
DEFAULT_IC0 = 0.5
DEFAULT_K = 1


def compute_sediment_delivery_ratio(
    ctx: RunContext,
    max_sdr=DEFAULT_MAX_SDR,
    ic0=DEFAULT_IC0,
    k=DEFAULT_K,
    c_factor_path=None,
    output_suffix=None,
    topo=None,
):
    """
    Calculate the Sediment Delivery Ratio (SDR) for the context's event.

    Parameters:
    - ctx: event-level RunContext. SDR depends on the fire-adjusted C
      factor, so all Delivery/ outputs are written under
      Events/<event>/Delivery/.
    - max_sdr: Maximum SDR value. Default is 0.8.
    - ic0: Calibration parameter for the IC-SDR relationship.
      Default is 0.5.
    - k: Shape parameter for the IC-SDR relationship.
      Default is 1.
    - c_factor_path: Path to the C-factor raster used in the
      connectivity calculation. When None, falls back to the
      catchment's base C_factor.tif and produces the baseline
      (no-fire) SDR. compute_adjusted_k_c passes a recovery-specific
      C_factor_adjusted_<suffix>.tif to build per-recovery SDRs.
    - output_suffix: Suffix for the output SDR (and intermediate)
      rasters, e.g. 't0' -> SDR_t0.tif. Defaults to 'baseline' when
      no c_factor_path is supplied so the output matches the layer the
      baseline simulation reads (SDR_baseline.tif).
    - topo: optional precomputed (grid, fdir, acc) tuple from
      dem_flow_layers() for the catchment DEM. compute_adjusted_k_c
      passes this so the DEM is conditioned once rather than per SDR.

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
    ctx.validate()
    catchment = ctx.catchment

    # Without a C-factor raster there is no single "adjusted" C factor to
    # fall back on (there is now one per recovery time), so default to the
    # base C_factor.tif and write the baseline SDR — the layer the
    # baseline (no-fire) simulation reads.
    baseline = c_factor_path is None
    if baseline:
        c_factor_path = ctx.catchment_path('Erodibility', 'C_factor.tif')
        if output_suffix is None:
            output_suffix = 'baseline'
    suffix_text = f"_{output_suffix}" if output_suffix else ""

    logger.info(
        'Computing Sediment Delivery Ratio for catchment: %s (event %s)',
        catchment, ctx.event)

    # The per-recovery SDRs depend on the fire-adjusted C factor and are
    # written per event; the baseline SDR is derived from the base C
    # factor, so it is fire-independent and lives at catchment scope.
    delivery_dir = (
        ctx.catchment_path('Delivery') if baseline
        else ctx.event_path('Delivery')
    )
    os.makedirs(delivery_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Read DEM and compute flow direction, accumulation, slope
    # ------------------------------------------------------------------
    dem_path = ctx.catchment_path('Topography', 'DEM.tif')
    dem_grid = read_raster_masked(dem_path)
    dem_data = dem_grid.data
    null_mask = dem_grid.nodata_mask

    xres = dem_grid.xres
    yres = dem_grid.yres
    pixel_area = dem_grid.pixel_area
    out_meta = dem_grid.meta()

    # Compute flow direction and accumulation via pysheds (or reuse the
    # caller's precomputed layers)
    if topo is None:
        topo = _topographic_indices(ctx)
    grid, fdir, acc = topo
    logger.info('Flow direction and accumulation computed')

    acc_data = np.array(acc, dtype=np.float32)
    # Area in square metres (flow accumulation × pixel area)
    area = acc_data * pixel_area

    # Compute slope and apply thresholds for connectivity index.
    # Slope is clamped to [0.005, 1]; NaN cells are restored below.
    slope_ratio, _, _ = slope_from_dem(dem_data, xres, yres)
    Sth = np.where(
        slope_ratio < 0.005, 0.005,
        np.where(slope_ratio <= 1, slope_ratio, 1)
    )
    nan_mask = np.isnan(slope_ratio)
    Sth[nan_mask] = np.nan

    # Mean slope over each cell's upslope contributing area. Sth is NaN
    # wherever slope is undefined: the one-cell halo of valid DEM cells
    # bordering nodata (np.gradient needs all neighbours) and any interior
    # DEM voids. A raw NaN weight poisons the pysheds accumulation for
    # every downstream cell, so the biggest streams — largest upslope area,
    # hence the highest chance of a NaN somewhere above them — come back
    # NaN. upslope_weighted_mean zeroes those cells and divides by a count
    # of the cells that actually contributed, dropping them from the
    # average rather than nulling the whole downstream path.
    #
    # Sth.tif keeps the raw (NaN-carrying) slope for inspection.
    Sth_path = os.path.join(delivery_dir, 'Sth.tif')
    write_raster(Sth_path, Sth, out_meta)
    Sth_raster = grid.read_raster(Sth_path)
    Av_Sth = upslope_weighted_mean(grid, fdir, Sth_raster)

    logger.info('Upslope slope averages (Sth) computed')

    # ------------------------------------------------------------------
    # Step 2: C-factor — compute thresholded upslope averages
    # ------------------------------------------------------------------
    # Align the C factor to the DEM grid: every other array here (Sth, Ddn,
    # fdir, the flow accumulation) lives on the DEM grid, and the downslope
    # BFS below indexes Cth with DEM-grid coordinates. A raw read() of a
    # C factor stored at a coarser native resolution would be a different
    # shape and raise IndexError. read_aligned resamples it to match.
    c_factor = read_aligned_like(c_factor_path, dem_grid)

    # Threshold C factor to a minimum of 0.001
    Cth = np.where(c_factor < 0.001, 0.001, c_factor)
    Cth_path = os.path.join(
        delivery_dir, f'Cth{suffix_text}.tif')
    write_raster(Cth_path, Cth, out_meta)
    # Same sanitised upslope mean as Sth: a C-factor raster that doesn't
    # cover the full DEM extent would carry NaN and poison the streams the
    # same way, so drop those cells from the average rather than the whole
    # downstream path.
    Cth_raster = grid.read_raster(Cth_path)
    Av_Cth = upslope_weighted_mean(grid, fdir, Cth_raster)

    logger.info('Upslope C-factor averages (Cth) computed')

    # ------------------------------------------------------------------
    # Step 3: Downslope path distance to nearest stream
    # ------------------------------------------------------------------

    # Define stream network based on contributing area threshold
    streams = area > 1.3e4
    stream_cells = np.where(streams)
    streams_path = os.path.join(delivery_dir, 'Streams.tif')
    write_raster(streams_path, streams, out_meta)

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
    dist_path = os.path.join(delivery_dir, 'Distance_to_stream.tif')
    write_raster(dist_path, distance_to_stream, out_meta)

    Ddn[null_mask] = np.nan
    Ddn_path = os.path.join(
        delivery_dir, f'Ddn{suffix_text}.tif')
    write_raster(Ddn_path, Ddn, out_meta)

    # ------------------------------------------------------------------
    # Step 4: Upslope component, Connectivity Index, and SDR
    # ------------------------------------------------------------------

    # Calculate the upslope component Dup
    Dup = Av_Cth * Av_Sth * np.sqrt(area)
    Dup[null_mask] = np.nan
    Dup_path = os.path.join(
        delivery_dir, f'Dup{suffix_text}.tif')
    write_raster(Dup_path, Dup, out_meta)

    # Lower bound on Ddn to avoid log10(0) in IC calculation
    EPS = 1
    Ddn = np.where(Ddn <= 0, EPS, Ddn)

    # Calculate Connectivity Index (IC)
    IC = np.log10(Dup / Ddn)
    IC[null_mask] = np.nan
    IC_path = os.path.join(
        delivery_dir, f'IC{suffix_text}.tif')
    write_raster(IC_path, IC, out_meta)

    logger.info('Connectivity index (IC) computed')

    # Calculate and save Sediment Delivery Ratio
    SDR = max_sdr / (1 + np.exp((ic0 - IC) / k))
    output_sdr_path = os.path.join(
        delivery_dir, f'SDR{suffix_text}.tif')
    write_raster(output_sdr_path, SDR, out_meta)

    logger.info(
        'Sediment Delivery Ratio computed for catchment: %s',
        catchment)

    return slope_ratio, fdir, acc, distance_to_stream, IC, Dup, Ddn, SDR
