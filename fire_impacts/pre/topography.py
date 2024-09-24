import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping, shape, Point, LineString
from shapely.strtree import STRtree
from rasterio.features import shapes
from affine import Affine
from pysheds.grid import Grid
from scipy.ndimage import label
import copy
import logging
logger = logging.getLogger(__name__)

def extract_catchment_dems(project,dem_path,target_resolution=None):
  '''
  Extract DEMs for all catchments in the project from a single regional DEM.

  Parameters:
  - project (dict): Dictionary containing the project folder structure.
  - dem_path (str): Path to the regional DEM file.
  - target_resolution (tuple): OPTIONAL: Desired resolution for the output rasters. Default to automatic selection of resolution
  '''
  shapefile_paths = project['Catchment_Shapefiles']
  logger.info('Extracting %d catchment DEMs',len(shapefile_paths))
  for shapefile in shapefile_paths:
      shapefile_filename = os.path.splitext(os.path.basename(shapefile))[0]
      logger.info('Extracting DEM for catchment: %s',shapefile)

      # Construct the output file names
      output_path1 = os.path.join(project[shapefile_filename], 'Catchment_Files', f'{shapefile_filename}_DEM.tif')
      output_path2 = os.path.join(project['Catchments_DEM'], f'{shapefile_filename}_DEM.tif')

      # Clip and reproject the raster with the shapefile
      clip_and_reproject_raster(dem_path, shapefile, [output_path1, output_path2], target_resolution=target_resolution)

def clip_and_reproject_raster(raster_file, shapefile, output_files, target_resolution=None):
    """
    Clips a raster file using a shapefile and reprojects the clipped raster to the CRS of the shapefile.

    Parameters:
    - raster_file (str): Path to the input raster file.
    - shapefile (str): Path to the shapefile for clipping.
    - output_files (list): List of paths to the output reprojected raster files.
    - target_resolution (tuple): OPTIONAL: Desired resolution for the output rasters. Default to automatic selection of resolution.
    """
    # Read the raster file to get its CRS and resolution
    with rio.open(raster_file) as src:
        raster_crs = src.crs
        raster_res = src.res  # Get the resolution of the input raster in original CRS

    # Read the shapefile
    catchment = gpd.read_file(shapefile)
    # Get the CRS of the shapefile
    shapefile_crs = catchment.crs.to_string()
    # Ensure the shapefile is in the same CRS as the raster before clipping
    catchment = catchment.to_crs(raster_crs)
    # Read the raster file
    with rio.open(raster_file) as src:
        # Clip the raster with the shapefile
        out_image, out_transform = mask(src, catchment.geometry.apply(mapping), crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "crs": src.crs})

    # Write the clipped raster to a temporary file
    temp_file = 'clipped_temp.tif'
    with rio.open(temp_file, 'w', **out_meta) as dest:
        dest.write(out_image)

    # Reproject the clipped raster to the CRS of the shapefile with the target resolution
    with rio.open(temp_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, shapefile_crs, src.width, src.height, *src.bounds, resolution=target_resolution)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': shapefile_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'res': target_resolution  # Set the target resolution explicitly
        })

        for output_file in output_files:
            with rio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=shapefile_crs,
                        resampling=Resampling.nearest)
        
    # Calculate the slope and save it 
    with rio.open(temp_file) as src:
        dem_data = src.read(1)
        cellsize_x = src.transform[0]
        cellsize_y = -src.transform[4]

        # Calculate the slope using numpy gradients
        dz_dx, dz_dy = np.gradient(dem_data, cellsize_x, cellsize_y)
        slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_degrees = np.degrees(slope_radians)

        # Update metadata for slope file
        slope_meta = src.meta.copy()
        slope_meta.update({
            'dtype': 'float32',
            'count': 1
        })

        # Save the slope layer
        slope_file = output_files[0].replace('_DEM.tif', '_Slope.tif')
        with rio.open(slope_file, 'w', **slope_meta) as dst:
            dst.write(slope_degrees.astype(np.float32), 1)
    # Clean up temporary file
    os.remove(temp_file)

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

def extract_headwaters(project,name,threshold_m2):
    # Extract CRS and transform and copy meta from DEM to write headwaters
    logger.info(f'Extracting headwaters for catchment: {name}')
    dem_fn = os.path.join(project['Catchments_DEM'],f'{name}_DEM.tif')
    with rio.open(dem_fn) as src:
        crs = src.crs
        transform = src.transform
        meta = src.meta.copy()
        resolution = src.res
        res_sq = resolution[0] * resolution[1]
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
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)  # Specify directional mapping # Determine D8 flow directions from DEM
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)  # Compute flow directions # each cell routes to only one of its nearest neighbors
    fdir = fdir.astype(float)
    logger.info('Computing flow accumulation')
    acc = grid.accumulation(fdir, dirmap=dirmap)  # Calculate flow accumulation
    # Save flow accumulation as Flow_acc
    flow_acc_file = os.path.join(project['Topography'], name, 'Catchment_DEM', 'Flow_accumulation.tif')
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
    branches = grid.extract_river_network(fdir, mask_above_threshold, dirmap=dirmap) # mask if the flow acc is less than threshold
    # Save the stream network as Stream_Network.tif
    stream_network_file = os.path.join(project['Topography'], name, 'Catchment_DEM', 'Stream_Network.tif')
    stream_meta = meta.copy()
    stream_meta.update({
        'dtype': 'int32',
        'count': 1,
        'nodata': -9999
    })
    
    # Create an empty array for stream network output
    stream_network_array = np.zeros_like(acc, dtype=np.int32)
    
    for feature in branches['features']:
        coords = np.array(feature['geometry']['coordinates'])
        # Convert coordinates to row/col indices and update stream network array
        for x, y in coords:
            col, row = ~transform * (x, y)
            col, row = int(col), int(row)
            if 0 <= col < stream_network_array.shape[1] and 0 <= row < stream_network_array.shape[0]:
                stream_network_array[row, col] = 1  # Mark the stream cells
    
    with rio.open(stream_network_file, 'w', **stream_meta) as dst:
        dst.write(stream_network_array, 1)
    logger.info(f'Saved Stream Network to: {stream_network_file}')
    
    # Build a spatial index for the LineStrings in branches
    logger.info('Building spatial index of %d branches',len(branches['features']))
    lines = [LineString(branch['geometry']['coordinates']) for branch in branches['features']]
    spatial_index = STRtree(lines)
    stream_order = grid.stream_order(fdir, mask_above_threshold, dirmap=dirmap, method='strahler') # get the stream order to filter the headwaters for first stream order

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

        catch = grid_1.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, xytype='coordinate') # snap the
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
    shp_output_path = os.path.join(project['Topography'], name, 'HW_SHPs', f'{name}.shp')
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
    output_raster_path = os.path.join(project['Topography'], name, 'HW_Rasters', f'{name}.tif')
    with rio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(subcatchment_raster, 1)
    hw_data = pd.DataFrame.from_records(records)

    # hw_data = pd.concat(WH_df, ignore_index=True)
    csv_path = os.path.join(project['Topography'], name, f'{name}.csv')
    logger.info('Writing summary data to CSV file: %s',csv_path)
    hw_data.to_csv(csv_path, index=False)

    return name, hw_data


