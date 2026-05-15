"""
Topographic pre-processing: DEM extraction, slope, flow routing, and
headwater delineation.

Functions here use pysheds for hydrological enforcement and rasterio
for raster I/O.  Outputs are written to each catchment's Topography
folder within the FireImpactsProject directory structure.
"""

import os
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import geopandas as gpd
import rasterio as rio
from shapely.geometry import shape, Point, LineString
from shapely.strtree import STRtree
from rasterio.features import shapes
from pysheds.view import Raster as PyshedsRaster
from pysheds.grid import Grid
from .project import FireImpactsProject, save_catchment_raster
from ..context import RunContext
from .util import *
import copy
import logging

logger = logging.getLogger(__name__)

from ..const import *


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def ftoi(x, dp=5):
    """Round x to dp decimal places and return as an integer."""
    return int(round(x, dp))


# ---------------------------------------------------------------------------
# DEM extraction
# ---------------------------------------------------------------------------

def extract_catchment_dems(
    ctx: RunContext,
    dem_path=None,
    target_resolution=None,
):
    """
    Extract a DEM for the context's catchment from a regional DEM.

    Parameters:
    - ctx: catchment-only RunContext (event/ensemble are unused).
    - dem_path: path to the regional DEM file; if None, downloads the
      DEMH mosaic from AWS.
    - target_resolution: desired output pixel resolution; defaults to
      automatic selection.

    Returns:
    - None.  Writes DEM.tif to the catchment's Topography folder.
    """
    catchment = ctx.catchment
    shapefile = ctx.project.boundary_files[catchment]
    logger.info("Extracting DEM for catchment: %s", catchment)

    output_path = ctx.catchment_path("Topography", "DEM.tif")

    if dem_path is None:
        logger.info(
            "No DEM path provided — downloading DEMH from AWS "
            "for catchment: %s",
            catchment,
        )
        from .data_sources import DEMH
        fn = DEMH
    else:
        fn = dem_path

    # Clip and reproject the raster with the shapefile
    clip_and_reproject_raster(
        fn, shapefile, output_path,
        target_resolution=target_resolution,
    )


# ---------------------------------------------------------------------------
# Stream network helpers
# ---------------------------------------------------------------------------

def calculate_movement_distance(point, spatial_index, lines):
    """
    Calculate the movement distance from a pour point to a stream end.

    Parameters:
    - point: shapely Point representing the pour point.
    - spatial_index: shapely STRtree built from the stream line segments.
    - lines: list of shapely LineString objects (one per branch).

    Returns:
    - displacement: straight-line distance from the pour point to the
      end of the branch, or None if no match found.
    - movement_distance: distance along the branch from the pour point
      to its end, or 0 if no match found.
    - start_point: shapely Point at the start of the matched branch.
    - end_point: shapely Point at the end of the matched branch.
    """
    # Apply a small buffer around the point to handle snapping errors
    buffered_point = point.buffer(1)

    # Query the spatial index to get possible matching features
    possible_matches_indices = spatial_index.query(buffered_point)

    movement_distance = 0
    displacement = None
    end_point = None

    for idx in possible_matches_indices:
        line = lines[idx]

        if line.intersects(buffered_point):
            start_point_coords = list(line.coords)[0]
            start_point = Point(start_point_coords)

            end_point_coords = list(line.coords)[-1]
            end_point = Point(end_point_coords)

            # Straight-line displacement from pour point to end
            displacement = np.sqrt(
                (end_point.x - point.x) ** 2
                + (end_point.y - point.y) ** 2
            )

            # Actual movement distance along the branch from pour point
            found_pour_point = False
            coords = list(line.coords)

            for i in range(len(coords) - 1):
                if found_pour_point or np.allclose(
                    coords[i], [point.x, point.y]
                ):
                    found_pour_point = True
                    segment_distance = np.sqrt(
                        (coords[i + 1][0] - coords[i][0]) ** 2
                        + (coords[i + 1][1] - coords[i][1]) ** 2
                    )
                    movement_distance += segment_distance
                # Stop once we reach the end point
                if np.allclose(coords[i + 1], end_point_coords):
                    break
            break  # Exit once the relevant branch is found

    return displacement, movement_distance, start_point, end_point


def find_nearest_index(x, y, transform):
    """
    Return the (row, col) grid index nearest to the given coordinates.

    Parameters:
    - x: x coordinate in the raster CRS.
    - y: y coordinate in the raster CRS.
    - transform: rasterio Affine transform for the grid.

    Returns:
    - (row, col) tuple of integer grid indices.
    """
    col, row = ~transform * (x, y)
    col, row = int(np.round(col)), int(np.round(row))
    return row, col


def get_adjacent_cells(row, col):
    """
    Return the 3x3 neighbourhood of grid indices around (row, col).

    Parameters:
    - row: row index of the centre cell.
    - col: column index of the centre cell.

    Returns:
    - List of (row, col) tuples for the 9-cell neighbourhood including
      the centre cell itself.
    """
    return [
        (row + i, col + j) for i in range(-1, 2) for j in range(-1, 2)
    ]


def find_closest_to_threshold(acc, row, col, threshold_cells):
    """
    Find the cell in a 3x3 neighbourhood whose accumulation is closest
    to threshold_cells.

    Parameters:
    - acc: 2-D numpy array of flow accumulation values.
    - row: row index of the starting cell.
    - col: column index of the starting cell.
    - threshold_cells: target cell count to snap to.

    Returns:
    - (row, col) tuple of the cell with the closest accumulation value.
    """
    adjacent_indices = get_adjacent_cells(row, col)
    closest_cell = (row, col)
    min_diff = abs(acc[row, col] - threshold_cells)

    for adj_row, adj_col in adjacent_indices:
        if 0 <= adj_row < acc.shape[0] and 0 <= adj_col < acc.shape[1]:
            diff = abs(acc[adj_row, adj_col] - threshold_cells)
            if diff < min_diff:
                min_diff = diff
                closest_cell = (adj_row, adj_col)

    return closest_cell


# ---------------------------------------------------------------------------
# Slope computation
# ---------------------------------------------------------------------------

def dem_to_slope(
    ctx: RunContext,
    dem: str | tuple,
    gradient: bool = False,
    hydro: bool = False,
    save: bool = True,
    crs_unit_to_metres: float = None,
):
    """
    Convert a DEM to a slope raster (degrees or gradient).

    Parameters:
    - ctx: catchment-only RunContext.
    - dem: path string to the DEM file, or a (data, meta) tuple of
      rasterio objects for an in-memory raster.
    - gradient: if True, return the raw terrain gradient instead of
      degrees.
    - hydro: if True, use the hydrologically-enforced output filename.
      Has no effect if save is False.
    - save: if True, write the slope raster to the Topography folder.
    - crs_unit_to_metres: conversion factor from CRS units to metres;
      required if the DEM CRS is not in metres.  Defaults to the
      approximate degrees-to-metres constant.

    Returns:
    - data: 2-D numpy array of slope values.
    - meta: rasterio metadata dict for the output raster.
    """
    # If we've been given a tuple, assume it is (data, meta)
    if isinstance(dem, tuple):
        data, meta = dem
    # If it's a string, read in the raster
    elif isinstance(dem, str):
        data, meta = read_raster(dem)
    else:
        raise ValueError(
            "topography.dem_to_slope() requires either a string path "
            "pointing to a readable DEM, or a tuple of (data, meta) "
            f"rasterio objects. Received {dem}"
        )
    meta2 = meta.copy()
    # Get raster attributes for easy access
    transform = meta["transform"]
    crs = meta["crs"]
    nodata = meta["nodata"]

    pix_width = transform[0]
    pix_height = abs(transform[4])
    pix_planar_area = pix_width * pix_height
    data_present = np.where(data == nodata, np.nan, data)

    # Handle conversion of units to metres so that units are standardised
    if crs.linear_units not in CRS_METRE_UNITS:
        # If units are not metres and no conversion was specified,
        # assume degrees and convert using the approximate constant.
        if crs_unit_to_metres is None:
            crs_unit_to_metres = APPROX_DEGREES_TO_METRES
        logger.warning(
            "CRS should be in metres, was %s. "
            "Applying crs_unit_to_metres conversion (%s)",
            crs.linear_units, crs_unit_to_metres,
        )
        pix_planar_area *= crs_unit_to_metres ** 2
        pix_width *= crs_unit_to_metres
        pix_height *= crs_unit_to_metres

    # Horizontal and vertical gradients along each cell axis
    horiz_grad, vert_grad = np.gradient(
        data_present, pix_width, pix_height
    )
    terrain_grad = np.sqrt(horiz_grad ** 2 + vert_grad ** 2)

    # Convert to degrees unless the caller wants raw gradient
    terr_slope_rad = np.arctan(terrain_grad)
    terr_slope_deg = np.degrees(terr_slope_rad)

    if gradient:
        final_data = terrain_grad
    else:
        final_data = terr_slope_deg

    # Save the slope raster if requested
    if save:
        # Output filename depends on whether a hydro-enforced DEM was used
        if hydro:
            file_name = SLOPE_HYDRO_FN
        else:
            file_name = SLOPE_FN
        success, message = save_catchment_raster(
            project=ctx.project,
            catchment=ctx.catchment,
            file_name=file_name,
            section="Topography",
            data=final_data,
            meta=meta2,
        )
        logger.info(message)

    return final_data, meta2


# ---------------------------------------------------------------------------
# Hydrological enforcement
# ---------------------------------------------------------------------------

def hydro_force_dem(dem_path: str):
    """
    Apply hydrological enforcement to a DEM: fix pits, depressions, and
    flats.

    Parameters:
    - dem_path: path to the DEM to be processed.

    Returns:
    - inflated_dem: pysheds Raster with pits, depressions, and flats
      resolved.
    - grid: pysheds Grid object used for subsequent routing operations.
    """
    grid = Grid.from_raster(dem_path)
    logger.info(
        "Creating hydrologically-enforced DEM from %s", dem_path
    )
    dem = grid.read_raster(dem_path)

    # Apply hydrological fixes using pysheds
    logger.info("Filling pits")
    fill_dem = grid.fill_pits(dem)
    logger.info("Filling depressions")
    flooded_dem = grid.fill_depressions(fill_dem)
    logger.info("Resolving flats")
    inflated_dem = grid.resolve_flats(flooded_dem)

    return inflated_dem, grid


# ---------------------------------------------------------------------------
# Flow routing
# ---------------------------------------------------------------------------

def rio_to_pysheds(
    data,
    meta,
    filename,
    dirmap: tuple = D8_FLOW_DIRECTIONS,
    routing: str = FLOW_ROUTING_TYPE,
) -> PyshedsRaster:
    """
    Convert rasterio data and meta to a pysheds Raster object.

    Reads the viewfinder from an existing file on disk, updates it with
    the nodata value from meta, and wraps the provided numpy array.

    Parameters:
    - data: numpy array of raster values.
    - meta: rasterio metadata dict (must include 'nodata').
    - filename: path to an existing raster used to initialise the
      pysheds Grid and derive the viewfinder.
    - dirmap: D8 flow direction mapping tuple.
    - routing: routing method string (e.g. 'd8').

    Returns:
    - PyshedsRaster wrapping data with the viewfinder from filename.
    """
    grid = Grid.from_raster(filename)
    interim = grid.read_raster(filename)

    # Get the viewfinder from the grid and update its nodata value
    vf = interim.viewfinder.copy()
    vf.nodata = meta["nodata"]

    out_Raster = PyshedsRaster(
        input_array=data,
        viewfinder=vf,
        metadata={
            "dirmap": dirmap,
            "routing": routing,
        },
    )

    return out_Raster


def compute_flow_dir(
    hydro_dem: ArrayLike,
    hydro_meta: dict,
    grid: Grid,
    dirmap: tuple,
    ctx: RunContext,
    save: bool = True,
    routing: str = FLOW_ROUTING_TYPE,
) -> tuple[PyshedsRaster, dict, Grid]:
    """
    Compute a flow direction raster from a hydrologically enforced DEM.

    Parameters:
    - hydro_dem: pysheds Raster of the hydrologically enforced DEM.
    - hydro_meta: rasterio metadata dict for the DEM.
    - grid: pysheds Grid object from hydro_force_dem.
    - dirmap: D8 flow direction mapping tuple.
    - ctx: catchment-only RunContext.
    - save: if True, write the flow direction raster to disk.
    - routing: routing method string (e.g. 'd8').

    Returns:
    - flow_dir_Raster: pysheds Raster of flow directions.
    - flow_dir_meta: rasterio metadata dict for the flow direction raster.
    - grid: the same pysheds Grid object (passed through).
    """
    logger.info("Computing flow direction")
    fdir = grid.flowdir(hydro_dem, dirmap=dirmap, routing=routing)

    in_nodata = hydro_meta["nodata"]
    out_nodata = np.int32(NODATA_VAL_INT)

    # Update rasterio metadata for the flow direction output
    flow_dir_meta = hydro_meta.copy()
    flow_dir_meta.update(dtype=np.int32, nodata=out_nodata, count=1)

    # Update the pysheds viewfinder with the integer nodata value
    flow_dir_vf = fdir.viewfinder.copy()
    flow_dir_vf.nodata = out_nodata

    # Replace input nodata values with an integer sentinel
    flow_dir_data = np.where(
        fdir == in_nodata,
        out_nodata,
        fdir,
    ).astype(np.int32)

    flow_dir_Raster = PyshedsRaster(
        input_array=flow_dir_data,
        viewfinder=flow_dir_vf,
        metadata={
            "dirmap": dirmap,
            "routing": routing,
        },
    )

    if save:
        success, message = save_catchment_raster(
            project=ctx.project,
            catchment=ctx.catchment,
            file_name=FLOW_DIRECTION_FN,
            section="Topography",
            data=flow_dir_data,
            meta=flow_dir_meta,
        )
        logger.info(message)

    return flow_dir_Raster, flow_dir_meta, grid


def compute_flow_accum(
    flow_dir_data: ArrayLike,
    flow_dir_meta: dict,
    grid: Grid,
    dirmap: tuple,
    ctx: RunContext,
    save: bool = True,
    routing: str = FLOW_ROUTING_TYPE,
) -> tuple[PyshedsRaster, dict, Grid]:
    """
    Compute a flow accumulation raster from a flow direction raster.

    Parameters:
    - flow_dir_data: pysheds Raster or array of flow direction values.
    - flow_dir_meta: rasterio metadata dict for the flow direction raster.
    - grid: pysheds Grid object from compute_flow_dir.
    - dirmap: D8 flow direction mapping tuple.
    - ctx: catchment-only RunContext.
    - save: if True, write the flow accumulation raster to disk.
    - routing: routing method string (e.g. 'd8').

    Returns:
    - flow_acc_data: numpy array of flow accumulation values.
    - flow_acc_meta: rasterio metadata dict for the output raster.
    - grid: the same pysheds Grid object (passed through).

    ------------------------------------------------------------------------
    Notes:
    - Assumes the input flow direction raster has correct integer dtypes
      and nodata values.
    ------------------------------------------------------------------------
    """
    logger.info("Computing flow accumulation")
    flow_acc_data = grid.accumulation(
        flow_dir_data,
        dirmap=dirmap,
        routing=routing,
    )
    flow_acc_meta = flow_dir_meta.copy()

    if save:
        success, message = save_catchment_raster(
            project=ctx.project,
            catchment=ctx.catchment,
            file_name=FLOW_ACCUMULATION_FN,
            section="Topography",
            data=flow_acc_data,
            meta=flow_acc_meta,
        )
        logger.info(message)

    return flow_acc_data, flow_acc_meta, grid


# ---------------------------------------------------------------------------
# Headwater delineation
# ---------------------------------------------------------------------------

def extract_headwaters(
    ctx: RunContext,
    threshold_m2: float = DEFAULT_HW_THRESHOLD,
):
    """
    Delineate headwaters for a catchment based on a flow accumulation
    threshold.

    Parameters:
    - ctx: catchment-only RunContext.
    - threshold_m2: contributing area threshold in square metres for
      headwater delineation (default DEFAULT_HW_THRESHOLD).

    Returns:
    - DataFrame containing headwater summary data.

    ------------------------------------------------------------------------
    Notes:
    - Also writes slope.tif as a side-effect of calling dem_to_slope.
    - Writes Headwaters.shp, Headwaters.tif, Headwaters.csv,
      Flow_accumulation.tif, and Stream_Network.tif to the catchment's
      Topography folder.
    ------------------------------------------------------------------------
    """
    catchment = ctx.catchment
    new_hw_id_field = ctx.project.headwater_id

    logger.info("Extracting headwaters for catchment: %s", catchment)
    dem_fn = ctx.catchment_path("Topography", "DEM.tif")

    # Compute and save slope as a side-effect (kept for workflow compatibility)
    slope_ras, meta = dem_to_slope(ctx, dem=dem_fn)

    crs = meta["crs"]
    transform = meta["transform"]
    x_res = transform[0]
    y_res = abs(transform[4])
    res_sq = x_res * y_res

    # Hydrologically enforce the DEM
    prepared_dem, grid = hydro_force_dem(dem_fn)

    # Compute flow direction and accumulation
    flow_dir_data, flow_dir_meta, grid = compute_flow_dir(
        hydro_dem=prepared_dem,
        hydro_meta=meta,
        grid=grid,
        dirmap=D8_FLOW_DIRECTIONS,
        ctx=ctx,
    )
    flow_acc_data, flow_acc_meta, grid = compute_flow_accum(
        flow_dir_data=flow_dir_data,
        flow_dir_meta=flow_dir_meta,
        grid=grid,
        dirmap=D8_FLOW_DIRECTIONS,
        ctx=ctx,
    )

    threshold_cells = int(threshold_m2 / res_sq)
    logger.info(
        "Threshold # cells: %d (%f m^2)", threshold_cells, threshold_m2
    )

    mask_at_threshold = flow_acc_data == threshold_cells
    mask_above_threshold = flow_acc_data >= threshold_cells

    # Extract the river network above the accumulation threshold
    logger.info("Extracting river network")
    branches = grid.extract_river_network(
        flow_dir_data,
        mask_above_threshold,
        dirmap=D8_FLOW_DIRECTIONS,
        nodata_out=np.int64(0),
    )

    # Save the stream network raster
    stream_network_file = ctx.catchment_path(
        "Topography", "Stream_Network.tif",
    )
    stream_meta = meta.copy()
    stream_meta.update({
        "dtype": "int32",
        "count": 1,
        "nodata": NODATA_VAL_INT,
    })

    stream_network_array = (
        np.ones_like(flow_acc_data, dtype=np.int32) * -9999
    )

    for feature in branches["features"]:
        coords = np.array(feature["geometry"]["coordinates"])
        for x, y in coords:
            col, row = ~transform * (x, y)
            col, row = ftoi(col, 0), ftoi(row, 0)
            if (
                0 <= col < stream_network_array.shape[1]
                and 0 <= row < stream_network_array.shape[0]
            ):
                stream_network_array[row, col] = 1

    with rio.open(stream_network_file, "w", **stream_meta) as dst:
        dst.write(stream_network_array, 1)
    logger.info("Saved Stream Network to: %s", stream_network_file)

    # Build a spatial index for quick branch lookups
    logger.info(
        "Building spatial index of %d branches",
        len(branches["features"]),
    )
    lines = [
        LineString(branch["geometry"]["coordinates"])
        for branch in branches["features"]
    ]
    spatial_index = STRtree(lines)

    # Compute stream order to identify first-order (headwater) branches
    stream_order = grid.stream_order(
        flow_dir_data,
        mask_above_threshold,
        dirmap=D8_FLOW_DIRECTIONS,
        method="strahler",
    )

    logger.info("Snapping start points to stream heads")
    start_xs = [list(l.coords)[0][0] for l in lines]
    start_ys = [list(l.coords)[0][1] for l in lines]

    logger.info("Processing %d line segments", len(lines))

    geometries = []
    records = []
    subcatchment_raster = np.zeros_like(slope_ras, dtype=np.int16)

    idx = 1
    count = 0
    for line, x, y in zip(lines, start_xs, start_ys):
        count += 1
        catchment_id = idx
        if count % 100 == 0:
            logger.info(
                "Processing branch %d/%d", count, len(lines)
            )

        start_point = Point([x, y])

        # Find the grid index for this pour point
        row, col = find_nearest_index(x, y, grid.affine)

        # Skip branches with stream order greater than 1
        if stream_order[row, col] > 1:
            continue

        # If the pour point has higher accumulation than the threshold,
        # snap to the nearest cell that is closest to the threshold.
        if flow_acc_data[row, col] > threshold_cells:
            row, col = find_closest_to_threshold(
                flow_acc_data, row, col, threshold_cells
            )
            x, y = grid.affine * (col, row)

        pp_flow_acc = flow_acc_data[row, col]
        grid_1 = copy.deepcopy(grid)

        catch = grid_1.catchment(
            x=x, y=y,
            fdir=flow_dir_data,
            dirmap=D8_FLOW_DIRECTIONS,
            xytype="coordinate",
        )
        catchment_view = grid_1.view(catch)
        catchment_cells = catchment_view * catchment_id
        subcatchment_raster += catchment_cells

        displacement, movement_distance, _, end_point = (
            calculate_movement_distance(start_point, spatial_index, lines)
        )

        # Convert catchment view to a polygon geometry
        catchment_view = np.array(catchment_view, dtype=np.int16)
        shapes_generator = shapes(catchment_view, transform=transform)
        all_geometries = [
            shape(geom)
            for geom, value in shapes_generator
            if value == 1
        ]
        combined_geometry = (
            gpd.GeoSeries(all_geometries).unary_union
            if len(all_geometries) > 1
            else all_geometries[0]
        )
        geometries.append(combined_geometry)

        records.append({
            new_hw_id_field: catchment_id,
            "Area_m2": round(combined_geometry.area, 0),
            "Area_ha": round(combined_geometry.area / 10000, 1),
            "PP_Flow_acc": pp_flow_acc,
            "PourPt_X": x,
            "PourPt_Y": y,
            "Dist": round(displacement, 1) if displacement else 0,
            "Move_dist": (
                round(movement_distance, 1) if movement_distance else 0
            ),
            "X_EndP": end_point.x if end_point else None,
            "Y_EndP": end_point.y if end_point else None,
        })
        idx += 1

    logger.info(
        "Headwaters extraction completed for catchment: %s", catchment
    )

    # Save as a GeoDataFrame / shapefile
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    shp_output_path = ctx.catchment_path("Topography", "Headwaters.shp")
    logger.info(
        "Writing headwaters data to shapefile: %s", shp_output_path
    )
    gdf.to_file(shp_output_path, driver="ESRI Shapefile")

    # Save the subcatchment raster
    subcatchment_raster[subcatchment_raster == 0] = -9999
    meta.update({
        "driver": "GTiff",
        "height": subcatchment_raster.shape[0],
        "width": subcatchment_raster.shape[1],
        "transform": transform,
        "crs": crs,
        "nodata": -9999,
    })
    output_raster_path = ctx.catchment_path("Topography", "Headwaters.tif")
    with rio.open(output_raster_path, "w", **meta) as dst:
        dst.write(subcatchment_raster, 1)

    # Save the headwater summary CSV
    hw_data = pd.DataFrame.from_records(records)
    csv_path = ctx.catchment_path("Topography", "Headwaters.csv")
    logger.info("Writing summary data to CSV file: %s", csv_path)
    hw_data.to_csv(csv_path, index=False)

    return hw_data
