import os
from glob import glob
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping, shape, Point, LineString
from shapely.strtree import STRtree
from rasterio.features import shapes
from pyproj import Proj
import elevation
from pysheds.grid import Grid
from scipy.ndimage import label
import copy
import shutil
import time
# import warnings
import logging
logger = logging.getLogger(__name__)

def initialise_project(directory:str,catchment_shapefiles=[],exist_ok=False,clear=False):
    '''
    Initialise a folder structure to contain working data for fire impacts studies.

    Parameters:
    - directory (str): Path to the directory where the project folder will be created.
    - catchment_shapefiles (list): List of paths to shapefiles representing catchments.
    - exist_ok (bool): OPTIONAL: If False, raise an error if the project folder already exists. Default is False.
    - clear (bool): OPTIONAL: If True, clear the project folder if it already exists. Default is False.
    '''
    folder_paths = {}
    # Define initial folder structure
    subfolders = ['Output', 'Output/Topography', 'Output/Topography/Catchments_DEM']
    # Create folders and store their paths
    for folder in subfolders:
        path = os.path.join(directory, folder)
        if os.path.exists(path) and clear:
            logger.info('Clearing folder: %s',path)
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=exist_ok)
        folder_name = folder.split('/')[-1]
        folder_paths[folder_name] = path
        # globals()[folder_name] = path

    folder_paths['Catchment_Shapefiles'] = catchment_shapefiles
    folder_paths['Catchment_Names'] = [os.path.splitext(os.path.basename(shapefile))[0] for shapefile in catchment_shapefiles]
    catchment_folders = create_catchment_folders(folder_paths)
    folder_paths.update(catchment_folders)
    return folder_paths

def create_catchment_folders(project):
    catchment_names = project['Catchment_Names']
    topography_path = project['Topography']
    folder_paths = {}
    for catch_name in catchment_names:
        main_path = os.path.join(topography_path, catch_name)
        os.makedirs(main_path, exist_ok=True)
        # Create subfolders inside each main folder
        subfolders = ['HW_SHPs', 'HW_Rasters', 'Catchment_DEM']
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
        folder_name = catch_name.split('/')[-1]
        folder_paths[folder_name] = main_path
        # globals()[folder_name] = main_path
    return folder_paths

def find_all_shapefiles(base_directory):
    '''
    Find all shapefiles in a directory and its subdirectories.
    '''
    assert os.path.isdir(base_directory), f"Directory not found: {base_directory}"
    shapefiles = glob(os.path.join(base_directory, '**','*.shp'),recursive=True)
    return shapefiles

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
      output_path1 = os.path.join(project[shapefile_filename], 'Catchment_DEM', f'{shapefile_filename}_DEM.tif')
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
    mask_at_threshold = acc == threshold_cells
    mask_above_threshold = acc > threshold_cells
    # Extract river network based on flow accumulation threshold
    logger.info('Extracting river network')
    branches = grid.extract_river_network(fdir, mask_above_threshold, dirmap=dirmap) # mask if the flow acc is less than threshold

    # Build a spatial index for the LineStrings in branches
    logger.info('Building spatial index of %d branches',len(branches['features']))
    lines = [LineString(branch['geometry']['coordinates']) for branch in branches['features']]
    spatial_index = STRtree(lines)

    # Initialize result storage
    WH_df = []

    logger.info('Snapping start points to stream heads')
    start_xs = [list(l.coords)[0][0] for l in lines]
    start_ys = [list(l.coords)[0][1] for l in lines]
    start_xy = np.column_stack([start_xs, start_ys])
    new_xy = grid.snap_to_mask(mask_at_threshold, start_xy)

    logger.info('Processing %d line segments',len(lines))
    # Iterate through each branch in the river network
    for idx, (line,x,y,(x_snap,y_snap)) in enumerate(zip(lines,start_xs,start_ys,new_xy)):
        if (idx + 1) % 100 == 0:
            logger.info('Processing branch %d/%d',(idx + 1), len(lines))
        # Get the pour point (start point) from the river network branch
        # start_point_coords = list(line.coords)[0]
        start_point = Point([x,y])
        # x, y = start_point.x, start_point.y

        # Snap the point to the nearest stream
        grid_1 = copy.deepcopy(grid)
        # x_snap, y_snap = grid_1.snap_to_mask(mask_at_threshold, (x, y)) # snap the pour point for a cell that is 26 (to get 2 heactares headwaters)

        catch = grid_1.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
        grid_1.clip_to(catch)
        clipped_catch = grid_1.view(catch)

        # Write headwaters as a raster
        meta.update({
            'driver': 'GTiff',
            'height': clipped_catch.shape[0],
            'width': clipped_catch.shape[1],
            'transform': transform,
            'crs': crs
        })
        # Specify the path where the clipped catchment raster file will be saved
        output_raster_path = os.path.join(project['Topography'], name, 'HW_Rasters', f'ID-{idx + 1}.tif')
        with rio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(clipped_catch, 1)

        # Calculate movement distance using the optimized function
        displacement, movement_distance, _, end_point = calculate_movement_distance(start_point, spatial_index, lines)

        # Convert catchment to GeoDataFrame
        clipped_catch = np.array(clipped_catch, dtype=np.int16)
        shapes_generator = shapes(clipped_catch, transform=transform)
        all_geometries = [shape(geom) for geom, value in shapes_generator if value == 1]
        combined_geometry = gpd.GeoSeries(all_geometries).unary_union if len(all_geometries) > 1 else all_geometries[0]

        gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=crs)
        gdf['ID'] = idx + 1
        gdf['Area_m2'] = round(gdf['geometry'].area, 0)
        gdf['Area_ha'] = round(gdf['Area_m2'] / 10000, 1)
        gdf['PourPt_X'] = x
        gdf['PourPt_Y'] = y
        gdf['Dist'] = round(displacement, 1) if displacement else 0
        gdf['Move_dist'] = round(movement_distance, 1) if movement_distance else 0
        if end_point:
            gdf['X_EndP'] = end_point.x
            gdf['Y_EndP'] = end_point.y
        else:
            gdf['X_EndP'] = None
            gdf['Y_EndP'] = None

        # Save the GeoDataFrame as a shapefile
        shp_output_path = os.path.join(project['Topography'], name, 'HW_SHPs', f'ID-{idx + 1}.shp')
        gdf.to_file(shp_output_path, driver='ESRI Shapefile')

        WH_df.append(gdf.iloc[:, 1:])

    logger.info('Headwaters extraction completed for catchment: %s',name)
    # Save the data as a DataFrame in a CSV file
    hw_data = pd.concat(WH_df, ignore_index=True)
    csv_path = os.path.join(project['Topography'], name, f'{name}.csv')
    logger.info('Writing summary data to CSV file: %s',csv_path)
    hw_data.to_csv(csv_path, index=False)

    return name, hw_data


