import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from shapely.geometry import shape, Point, LineString
from shapely.strtree import STRtree
from rasterio.features import shapes
from pysheds.grid import Grid
from .project import FireImpactsProject
from .util import clip_and_reproject_raster
import copy
import logging
logger = logging.getLogger(__name__)

DEFAULT_HW_THRESHOLD=20000
APPROX_DEGREES_TO_METRES=111000
D8_FLOW_DIRECTIONS = (64, 128, 1, 2, 4, 8, 16, 32) # (north, northeast, east, southeast, south, southwest, west, northwest)
CRS_METRE_UNITS={'m','meter','meters','metre','metres'}

def ftoi(x,dp=5):
    return int(round(x,dp))

def extract_catchment_dems(project:FireImpactsProject,dem_path,target_resolution=None):
  '''
  Extract DEMs for all catchments in the project from a single regional DEM.

  Parameters:
  - project (dict): Dictionary containing the project folder structure.
  - dem_path (str): Path to the regional DEM file.
  - target_resolution (tuple): OPTIONAL: Desired resolution for the output rasters. Default to automatic selection of resolution
  '''
  catchments = project.catchments
  logger.info('Extracting %d catchment DEMs',len(catchments))
  for catchment in catchments:
      shapefile = project.boundary_files[catchment]
      logger.info('Extracting DEM for catchment: %s',catchment)

      # Construct the output file names
      output_path = os.path.join(project.catchment_path(catchment), 'Topography', 'DEM.tif')

      # Clip and reproject the raster with the shapefile
      clip_and_reproject_raster(dem_path, shapefile, output_path, target_resolution=target_resolution)

def calculate_movement_distance(point, spatial_index, lines):
    '''
    Function to calculate movement distance for a single pour point
    '''
    # Apply a small buffer around the point (e.g., 1 meter)
    buffered_point = point.buffer(1)  # Adjust the buffer size as necessary

    # Query the spatial index to get possible matching features
    possible_matches_indices = spatial_index.query(buffered_point)

    movement_distance = 0
    displacement = None
    end_point = None

    for idx in possible_matches_indices:
        # Retrieve the corresponding LineString using the index
        line = lines[idx]

        # Check if the buffered point intersects the line
        if line.intersects(buffered_point):
            # Get the start point (pour point) of the feature
            start_point_coords = list(line.coords)[0]
            start_point = Point(start_point_coords)

            # Get the end point of the feature
            end_point_coords = list(line.coords)[-1]
            end_point = Point(end_point_coords)  # Convert to Shapely Point

            # Calculate displacement
            displacement = np.sqrt((end_point.x - point.x)**2 + (end_point.y - point.y)**2)  # Straight-line distance

            # Calculate actual movement distance along the feature
            found_pour_point = False  # Start from the pour point to the end point
            coords = list(line.coords)

            for i in range(len(coords) - 1):
                if found_pour_point or np.allclose(coords[i], [point.x, point.y]):
                    found_pour_point = True
                    # Calculate distance between consecutive points
                    segment_distance = np.sqrt((coords[i+1][0] - coords[i][0])**2 + (coords[i+1][1] - coords[i][1])**2)
                    movement_distance += segment_distance
                # Stop once we reach the end point
                if np.allclose(coords[i+1], end_point_coords):
                    break
            break  # Exit the loop once the relevant branch is found
    return displacement, movement_distance, start_point, end_point


# Find the nearest index in the grid based on coordinates
def find_nearest_index(x, y, transform):
    col, row = ~transform * (x, y)  # Convert coordinates to grid indices
    col, row = int(np.round(col)), int(np.round(row))  # Round to the nearest integer
    return row, col

def get_adjacent_cells(row, col):
    # Get the indices of the adjacent cells including the center cell itself
    adjacent_indices = [(row + i, col + j) for i in range(-1, 2) for j in range(-1, 2)]
    return adjacent_indices

def find_closest_to_threshold(acc, row, col, threshold_cells):
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

def extract_headwaters(project:FireImpactsProject,name:str=None,threshold_m2:float=DEFAULT_HW_THRESHOLD,crs_unit_to_metres:float=None):
    '''
    Delinate headwaters for a catchment based on a flow accumulation threshold.

    Parameters:
    - project (dict): Dictionary containing the project folder structure.
    - name (str): Name of the catchment to process. If None, process all catchments.
    - threshold_m2 (float): Threshold area in square meters for headwaters. Default is 20,000 m^2.

    Writes:
    - Headwaters.shp: Shapefile containing the headwaters polygons.
    - Headwaters.tif: Raster file containing the headwaters polygons.
    - Headwaters.csv: CSV file containing the headwaters summary
    - Flow_accumulation.tif: Raster file containing the flow accumulation data.
    - Stream_Network.tif: Raster file containing the stream network data.
    - Slope.tif: Raster file containing the slope data.

    Returns:
    - pd.DataFrame: DataFrame containing the headwaters summary data.
    '''
    # Extract CRS and transform and copy meta from DEM to write headwaters
    if name is None:
        return project.for_each_catchment(lambda c: extract_headwaters(project,c,threshold_m2))

    logger.info(f'Extracting headwaters for catchment: {name}')
    dem_fn = project.catchment_path(name,'Topography','DEM.tif')
    with rio.open(dem_fn) as src:
        logging.info('Computing catchment slope')
        crs = src.crs
        transform = src.transform
        meta = src.meta.copy()
        resolution = src.res
        res_sq = resolution[0] * resolution[1]
        # Calculate and save slope
        dem_data = src.read(1)  # Read the first band (DEM values)
        # Calculate gradient in the x and y directions
        x_res, y_res = src.res
        if crs.linear_units not in CRS_METRE_UNITS:
          if crs_unit_to_metres is None:
              crs_unit_to_metres = APPROX_DEGREES_TO_METRES
          logger.warning('CRS should be in meters, was %s. Applying crs_unit_to_metres conversion (%f)',src.crs.linear_units,crs_unit_to_metres)
          res_sq *= crs_unit_to_metres**2
          x_res *= crs_unit_to_metres
          y_res *= crs_unit_to_metres
        dx, dy = np.gradient(dem_data, x_res, y_res)
        # Calculate slope in degrees
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180.0 / np.pi)
        # Save slope as a new GeoTIFF
        slope_profile = src.profile
        slope[dem_data==meta['nodata']] = meta['nodata']
        slope_profile.update(dtype=rio.float32, count=1)
        output_slope_path = project.catchment_path(name,'Topography','Slope.tif')
        with rio.open(output_slope_path, 'w', **slope_profile) as slope_dataset:
            slope_dataset.write(slope.astype(rio.float32), 1)

    threshold_cells = int(threshold_m2 / res_sq)
    logger.info('Threshold # cells: %d (%f m^2)', threshold_cells, threshold_m2)

    grid = Grid.from_raster(dem_fn)
    logger.info('Loading DEM from %s',dem_fn)
    dem = grid.read_raster(dem_fn)

    logger.info('Filling pits')
    fill_dem = grid.fill_pits(dem)  # Fill pits in DEM
    logger.info('Filling depressions')
    flooded_dem = grid.fill_depressions(fill_dem)  # Fill depressions in DEM
    logger.info('Resolving flats')
    inflated_dem = grid.resolve_flats(flooded_dem)  # Resolve flats in DEM

    logger.info('Computing flow directions')
    fdir = grid.flowdir(inflated_dem, dirmap=D8_FLOW_DIRECTIONS)  # Compute flow directions # each cell routes to only one of its nearest neighbors
    fdir = fdir.astype(float)
    logger.info('Computing flow accumulation')
    acc = grid.accumulation(fdir, dirmap=D8_FLOW_DIRECTIONS)  # Calculate flow accumulation
    # Save flow accumulation as Flow_acc
    flow_acc_file = project.catchment_path(name,'Topography','Flow_accumulation.tif')
    acc_meta = meta.copy()
    acc_meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': -9999
    })
    with rio.open(flow_acc_file, 'w', **acc_meta) as dst:
        dst.write(acc.astype(np.float32), 1)
    logger.info(f'Saved Flow Accumulation to: {flow_acc_file}')
    mask_at_threshold = acc == threshold_cells
    mask_above_threshold = acc >= threshold_cells  # need to be equal or greater then threshold
    # Extract river network based on flow accumulation threshold
    logger.info('Extracting river network')
    branches = grid.extract_river_network(fdir, mask_above_threshold, dirmap=D8_FLOW_DIRECTIONS,nodata_out=np.int64(0)) # mask if the flow acc is less than threshold
    # Save the stream network as Stream_Network.tif
    stream_network_file = project.catchment_path(name,'Topography','Stream_Network.tif')
    stream_meta = meta.copy()
    stream_meta.update({
        'dtype': 'int32',
        'count': 1,
        'nodata': -9999
    })

    # Create an empty array for stream network output
    stream_network_array = np.ones_like(acc, dtype=np.int32) * -9999

    for feature in branches['features']:
        # print(feature)
        # assert False
        coords = np.array(feature['geometry']['coordinates'])
        # Convert coordinates to row/col indices and update stream network array
        for x, y in coords:
            col, row = ~transform * (x, y)
            col, row = ftoi(col,0),ftoi(row,0) # OR round?
            if 0 <= col < stream_network_array.shape[1] and 0 <= row < stream_network_array.shape[0]:
                stream_network_array[row, col] = 1  # Mark the stream cells

    with rio.open(stream_network_file, 'w', **stream_meta) as dst:
        dst.write(stream_network_array, 1)
    logger.info(f'Saved Stream Network to: {stream_network_file}')

    # Build a spatial index for the LineStrings in branches
    logger.info('Building spatial index of %d branches',len(branches['features']))
    lines = [LineString(branch['geometry']['coordinates']) for branch in branches['features']]
    spatial_index = STRtree(lines)
    stream_order = grid.stream_order(fdir, mask_above_threshold, dirmap=D8_FLOW_DIRECTIONS, method='strahler') # get the stream order to filter the headwaters for first stream order

    logger.info('Snapping start points to stream heads')
    start_xs = [list(l.coords)[0][0] for l in lines]
    start_ys = [list(l.coords)[0][1] for l in lines]

    logger.info('Processing %d line segments',len(lines))
    # Iterate through each branch in the river network

    geometries = []
    records = []
    subcatchment_raster = np.zeros_like(dem,dtype=np.int16)

    idx=1
    count = 0
    for line,x,y in zip(lines,start_xs,start_ys):
        count += 1
        catchment_id = idx
        if (count) % 100 == 0:
            logger.info('Processing branch %d/%d',(count), len(lines))

        # Get the pour point (start point) from the river network branch
        # start_point_coords = list(line.coords)[0]
        start_point = Point([x,y])

        # Find the nearest index for the start point in the grid
        row, col = find_nearest_index(x, y, grid.affine)
        # Check the stream order at this location
        if stream_order[row, col] > 1:
            continue  # Skip to the next line if stream order is greater than 1

        # If the flow accumulation in the pour point is greater then threshold, get the an adjesent cell with the closest flow acc to the thrshold
        if acc[row, col] > threshold_cells:
            # Find the closest cell with flow accumulation near to threshold
            row, col = find_closest_to_threshold(acc, row, col, threshold_cells)
            x, y = grid.affine * (col, row)  # Update the coordinates
        # Get the flow accumulation at the pour point and save it in the dataframe
        pp_flow_acc = acc[row, col]
        grid_1 = copy.deepcopy(grid)

        catch = grid_1.catchment(x=x, y=y, fdir=fdir, dirmap=D8_FLOW_DIRECTIONS, xytype='coordinate') # snap the
        catchment_view = grid_1.view(catch)
        catchment_cells = catchment_view * catchment_id
        subcatchment_raster += catchment_cells

        # Calculate movement distance using the optimized function
        displacement, movement_distance, _, end_point = calculate_movement_distance(start_point, spatial_index, lines)

        # Convert catchment to GeoDataFrame
        catchment_view = np.array(catchment_view, dtype=np.int16)
        shapes_generator = shapes(catchment_view, transform=transform)
        all_geometries = [shape(geom) for geom, value in shapes_generator if value == 1]
        combined_geometry = gpd.GeoSeries(all_geometries).unary_union if len(all_geometries) > 1 else all_geometries[0]
        geometries.append(combined_geometry)

        records.append({
            'ID': catchment_id,
            'Area_m2': round(combined_geometry.area, 0),
            'Area_ha': round(combined_geometry.area / 10000, 1),
            'PP_Flow_acc': pp_flow_acc,
            'PourPt_X': x,
            'PourPt_Y': y,
            'Dist': round(displacement, 1) if displacement else 0,
            'Move_dist': round(movement_distance, 1) if movement_distance else 0,
            'X_EndP': end_point.x if end_point else None,
            'Y_EndP': end_point.y if end_point else None
        })
        idx +=1

    logger.info('Headwaters extraction completed for catchment: %s',name)
    # Save the data as a DataFrame in a CSV file
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
    shp_output_path = project.catchment_path(name,'Topography','Headwaters.shp')
    logger.info('Writing headwaters data to shapefile: %s',shp_output_path)
    gdf.to_file(shp_output_path, driver='ESRI Shapefile')

    subcatchment_raster[subcatchment_raster == 0] = -9999
    meta.update({
        'driver': 'GTiff',
        'height': subcatchment_raster.shape[0],
        'width': subcatchment_raster.shape[1],
        'transform': transform,
        'crs': crs,
        'nodata': -9999
    })
    # Specify the path where the clipped catchment raster file will be saved
    output_raster_path = project.catchment_path(name,'Topography','Headwaters.tif')
    with rio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(subcatchment_raster, 1)
    hw_data = pd.DataFrame.from_records(records)

    csv_path = project.catchment_path(name,'Topography','Headwaters.csv')
    logger.info('Writing summary data to CSV file: %s',csv_path)
    hw_data.to_csv(csv_path, index=False)

    return hw_data


