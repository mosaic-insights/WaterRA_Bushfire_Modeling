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
from .util import *
import copy
import logging
logger = logging.getLogger(__name__)
from ..const import *

def ftoi(x,dp=5):
    return int(round(x,dp))

def extract_catchment_dems(
        project:FireImpactsProject,
        dem_path=None,
        target_resolution=None
        ):
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

      if dem_path is None:
          logger.info('No DEM path provided, downloading DEMH data from AWS for catchment: %s',catchment)
          from .data_sources import DEMH
          fn = DEMH
      else:
          fn = dem_path
      # Clip and reproject the raster with the shapefile
      clip_and_reproject_raster(fn, shapefile, output_path, target_resolution=target_resolution)

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

###############################################################################
def dem_to_slope(
    project:FireImpactsProject,
    dem:str | tuple,
    catchment_name:str,
    gradient:bool=False,
    hydro:bool=False,
    save:bool=True,
    crs_unit_to_metres:float=None
    ):
    """
    Convert a DEM to a slope raster. Can be either raw, or a 
    hydrologically-enforced DEM.

    Parameters:
    - project (FireImpactsProject): the current project which handles 
    directory locations
    - dem: This can be either of the following:
        - A path including filename and extension that points to the 
        DEM, as a string
        - a tuple of rasterio (data, meta) objects for an in-memory 
        raster
    - catchment_name (string): name of the catchment this DEM is for
    - hydro (bool, default False): whether this is a 
    hydrologically-enforced DEM. Only effects the output file name, so 
    will have no impact of save is set to False
    - save (bool, default True): Whether the output slope DEM should 
    be saved as a GeoTIFF
    - crs_unit_to_metres (float): if the units of the DEM's crs are not 
    metres, the number of those units per metre.

    Returns:
    - tuple of rasterio objects as (data, meta)
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # If we've been given a tuple, assume it is (data, meta)
    if isinstance(dem, tuple):
        data, meta = dem
    #If it's a string, read in the raster:
    elif isinstance(dem, str):
        data, meta = read_raster(dem)
    else:
        raise ValueError(
            'topography.dem_to_slope() requires either a string path '
            'pointing to a readable DEM, or a tuple of (data, meta) '
            f'rasterio objects. Received {dem}'
        )
    meta2 = meta.copy()
    # Get other raster attributes for easy access:
    transform = meta['transform']
    crs = meta['crs']
    nodata = meta['nodata']

    pix_width = transform[0]
    pix_height = abs(transform[4])
    pix_planar_area = pix_width * pix_height
    data_present = np.where(data == nodata, np.nan, data)

    # Handle conversion of units to metres so that units are 
    #standardised:
    if crs.linear_units not in CRS_METRE_UNITS:
        # If units are not metres, and a conversion was not specified, 
        #assume it's in degrees and convert from that:
        if crs_unit_to_metres is None:
            crs_unit_to_metres = APPROX_DEGREES_TO_METRES
        logger.warning(
            f'CRS should be in meters, was {crs.linear_units}. '
            'Applying crs_unit_to_metres conversion '
            f'({crs_unit_to_metres})'
            )
        pix_planar_area *= crs_unit_to_metres**2
        pix_width *= crs_unit_to_metres
        pix_height *= crs_unit_to_metres

    # Get the horizontal and vertical gradients for each cell to its 
    #neighbours along that plane. 
    horiz_grad, vert_grad = np.gradient(data_present, pix_width, pix_height)
    # Get the actual terrain gradient:
    terrain_grad = np.sqrt(horiz_grad**2 + vert_grad**2)

    # Get a version in degrees:
    terr_slope_rad = np.arctan(terrain_grad)
    terr_slope_deg = np.degrees(terr_slope_rad)

    if gradient:
        final_data = terrain_grad
    else:
        final_data = terr_slope_deg

    # Save the slope raster if that's been requested:
    if save:
        # Set output file name based on whether we've used a 
        #hydrologically enforced DEM for this slope raster:
        if hydro:
            file_name = SLOPE_HYDRO_FN
        else:
            file_name = SLOPE_FN
        success, message = save_catchment_raster(
            project=project,
            catchment_name=catchment_name,
            file_name=file_name,
            section='Topography',
            data=final_data,
            meta=meta2
            )
        logger.info(message)
        
    # Return the raster data and its metadata:
    return final_data, meta2

###############################################################################
def hydro_force_dem(dem_path:str):
    """
    Apply hydrological enforcements to a DEM: Fix pits, depressions, 
    and flats.

    Parameters:
    - dem_path (string): Path to the DEM which is to be processed.

    Returns:
    - pysheds grid object which is a dem that can be used for 
    hydrology operations.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Create a pysheds grid object from the DEM:
    grid = Grid.from_raster(dem_path)
    logger.info(
        'Creating hydrologically-enforced DEM from %s', dem_path
        )
    dem = grid.read_raster(dem_path)

    # Apply the hydrological fixes using pysheds:
    logger.info('Filling pits')
    fill_dem = grid.fill_pits(dem)
    logger.info('Filling depressions')
    flooded_dem = grid.fill_depressions(fill_dem) 
    logger.info('Resolving flats')
    inflated_dem = grid.resolve_flats(flooded_dem)

    return inflated_dem, grid

###############################################################################
def rio_to_pysheds(
    data,
    meta,
    filename,
    dirmap:tuple=D8_FLOW_DIRECTIONS,
    routing:str=FLOW_ROUTING_TYPE
    ) -> PyshedsRaster:
    """
    Convert rasterio data and meta objects to a Pysheds raster

    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    # Create and populate the pyshed Grid
    grid = Grid.from_raster(filename)
    interim = grid.read_raster(filename)

    # Get the viewfinder from the grid, and update its metadata as 
    #required:
    vf = interim.viewfinder.copy()
    vf.nodata = meta['nodata']

    # Create the PyshedRaster object:
    out_Raster = PyshedsRaster(
        input_array=data,
        viewfinder=vf,
        metadata={
            'dirmap': dirmap,
            'routing': routing
            }
        )
    
    return out_Raster

###############################################################################
def compute_flow_dir(
    hydro_dem:ArrayLike,
    hydro_meta:dict,
    grid:Grid,
    dirmap:tuple,
    project:FireImpactsProject,
    catchment_name:str,
    save:bool=True,
    routing:str=FLOW_ROUTING_TYPE
    ) -> tuple[PyshedsRaster, dict, Grid]:
    """
    Compute a flow direction raster from a hydrologically enforced DEM

    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
        
    # Compute the flow direction grid:
    logger.info('Computing flow direction')
    fdir = grid.flowdir(hydro_dem, dirmap=dirmap, routing=routing)

    # Define input and output nodata values:
    in_nodata = hydro_meta['nodata']
    out_nodata = np.int32(NODATA_VAL_INT)

    # Update the rasterio metadata:
    flow_dir_meta = hydro_meta.copy()
    flow_dir_meta.update(dtype=np.int32, nodata=out_nodata, count=1)

    # Update the pysheds viewfinder with the target nodata value:
    flow_dir_vf = fdir.viewfinder.copy()
    flow_dir_vf.nodata = out_nodata

    # Replace input nodata values with a useful integer one:
    flow_dir_data = np.where(
        fdir == in_nodata,
        out_nodata,
        fdir
        ).astype(np.int32)
    
    flow_dir_Raster = PyshedsRaster(
        input_array=flow_dir_data,
        viewfinder=flow_dir_vf,
        metadata={
            'dirmap': dirmap,
            'routing': routing
            }
        )
    
    # Save the raster to file if requested:
    if save:
        success, message = save_catchment_raster(
            project=project,
            catchment_name=catchment_name,
            file_name=FLOW_DIRECTION_FN,
            section = 'Topography',
            data=flow_dir_data,
            meta=flow_dir_meta
            )
        logger.info(message)
    
    return flow_dir_Raster, flow_dir_meta, grid

###############################################################################
def compute_flow_accum(
    flow_dir_data:ArrayLike,
    flow_dir_meta:dict,
    grid:Grid,
    dirmap:tuple,
    project:FireImpactsProject,
    catchment_name:str,
    save:bool=True,
    routing:str=FLOW_ROUTING_TYPE
    ) -> tuple[PyshedsRaster, dict, Grid]:
    """
    Compute a flow accumulation raster from a flow direction raster

    --------------------------------------------------------------------
    Notes:
    - Assues the input flow direction has correct integer dtypes and 
    nodata values.
    --------------------------------------------------------------------
    """
    # Compute the flow accumulation grid:
    logger.info('Computing flow accumulation')
    flow_acc_data = grid.accumulation(
        flow_dir_data,
        dirmap=dirmap,
        routing=routing) 
    flow_acc_meta = flow_dir_meta.copy()

    # Save the raster to file if requested:
    if save:
        success, message = save_catchment_raster(
            project=project,
            catchment_name=catchment_name,
            file_name=FLOW_ACCUMULATION_FN,
            section='Topography',
            data=flow_acc_data,
            meta=flow_acc_meta
        )
        logger.info(message)

    return flow_acc_data, flow_acc_meta, grid
    
###############################################################################
def extract_headwaters(
    project:FireImpactsProject,
    name:str | None=None,
    threshold_m2:float=DEFAULT_HW_THRESHOLD
    ):
    """
    Delinate headwaters for a catchment based on a flow accumulation 
    threshold.

    Parameters:
    - project (FireImpactsProject): project object for directory 
    structure
    - name (str): Name of the catchment to process. If None, process 
    all catchments.
    - threshold_m2 (float): Threshold area in square meters for 
    headwaters. Default is 20,000 m^2.

    Returns:
    - Dataframe containing headwater summary data

    Writes:
    - Headwaters.shp: Shapefile containing the headwaters polygons.
    - Headwaters.tif: Raster file containing the headwaters polygons.
    - Headwaters.csv: CSV file containing the headwaters summary
    - Flow_accumulation.tif: Raster file containing the flow 
    accumulation data.
    - Stream_Network.tif: Raster file containing the stream network 
    data.
    - Slope.tif: Raster file containing the slope data.
    --------------------------------------------------------------------
    TODO: move the writing of slope.tif out of this function to 
    somewhere it's needed.
    --------------------------------------------------------------------
    """
    new_hw_id_field = project.headwater_id
    # Extract CRS and transform and copy meta from DEM to write headwaters
    if name is None:
        return project.for_each_catchment(
            lambda c: extract_headwaters(project,c,threshold_m2)
            )

    logger.info(f'Extracting headwaters for catchment: {name}')
    dem_fn = project.catchment_path(name,'Topography','DEM.tif')
    
    # Placeholder for consistency with old workflow, so calling this 
    #function still saves the slope file:
    slope_ras, meta = dem_to_slope(
        project=project,
        dem=dem_fn,
        catchment_name=name
        )

    crs = meta['crs']
    transform = meta['transform']
    x_res = transform[0]
    y_res = abs(transform[4])
    res_sq = x_res * y_res

    # Hyrdologically enforce the DEM:
    prepared_dem, grid = hydro_force_dem(dem_fn)
    # Get a flow direction raster:
    flow_dir_data, flow_dir_meta, grid = compute_flow_dir(
        hydro_dem=prepared_dem,
        hydro_meta=meta,
        grid=grid,
        dirmap=D8_FLOW_DIRECTIONS,
        project=project,
        catchment_name=name
        )
    # Get a flow accumulation raster:
    flow_acc_data, flow_acc_meta, grid = compute_flow_accum(
        flow_dir_data=flow_dir_data,
        flow_dir_meta=flow_dir_meta,
        grid=grid,
        dirmap=D8_FLOW_DIRECTIONS,
        project=project,
        catchment_name=name
        )


    
    threshold_cells = int(threshold_m2 / res_sq)
    logger.info('Threshold # cells: %d (%f m^2)', threshold_cells, threshold_m2)

    mask_at_threshold = flow_acc_data == threshold_cells
    mask_above_threshold = flow_acc_data >= threshold_cells  # need to be equal or greater then threshold
    # Extract river network based on flow accumulation threshold
    logger.info('Extracting river network')
    branches = grid.extract_river_network(flow_dir_data, mask_above_threshold, dirmap=D8_FLOW_DIRECTIONS,nodata_out=np.int64(0)) # mask if the flow acc is less than threshold
    # Save the stream network as Stream_Network.tif
    stream_network_file = project.catchment_path(name,'Topography','Stream_Network.tif')
    stream_meta = meta.copy()
    stream_meta.update({
        'dtype': 'int32',
        'count': 1,
        'nodata': NODATA_VAL_INT
    })

    # Create an empty array for stream network output
    stream_network_array = np.ones_like(flow_acc_data, dtype=np.int32) * -9999

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
    stream_order = grid.stream_order(flow_dir_data, mask_above_threshold, dirmap=D8_FLOW_DIRECTIONS, method='strahler') # get the stream order to filter the headwaters for first stream order

    logger.info('Snapping start points to stream heads')
    start_xs = [list(l.coords)[0][0] for l in lines]
    start_ys = [list(l.coords)[0][1] for l in lines]

    logger.info('Processing %d line segments',len(lines))
    # Iterate through each branch in the river network

    geometries = []
    records = []
    subcatchment_raster = np.zeros_like(slope_ras,dtype=np.int16)

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
        if flow_acc_data[row, col] > threshold_cells:
            # Find the closest cell with flow accumulation near to threshold
            row, col = find_closest_to_threshold(flow_acc_data, row, col, threshold_cells)
            x, y = grid.affine * (col, row)  # Update the coordinates
        # Get the flow accumulation at the pour point and save it in the dataframe
        pp_flow_acc = flow_acc_data[row, col]
        grid_1 = copy.deepcopy(grid)

        catch = grid_1.catchment(x=x, y=y, fdir=flow_dir_data, dirmap=D8_FLOW_DIRECTIONS, xytype='coordinate') # snap the
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
            new_hw_id_field: catchment_id,
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


