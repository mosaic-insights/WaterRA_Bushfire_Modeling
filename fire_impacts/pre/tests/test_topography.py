from .util import *
import rasterio as rio
from fire_impacts.pre import topography

def get_raster_array(fn):
  with rio.open(fn) as src:
    return src.read(1)

def test_headwater_delineation(get_project,get_file):
  proj = get_project()

  topography.extract_catchment_dems(proj, get_file('example_dem.tif'))
  fn = proj.catchment_path('example_small_catchment','Topography','DEM.tif')
  assert Path(fn).exists(), 'DEM not extracted'

  topography.extract_headwaters(proj, 'example_small_catchment')
  headwater_fn = proj.catchment_path('example_small_catchment','Topography','Headwaters.tif')
  assert Path(headwater_fn).exists(), 'Headwaters not extracted'

  flow_accum_fn = proj.catchment_path('example_small_catchment','Topography','Flow_accumulation.tif')
  assert Path(flow_accum_fn).exists(), 'Flow accumulation not extracted'

  headwater_arr = get_raster_array(headwater_fn)
  flow_accum_arr = get_raster_array(flow_accum_fn)
  HEADWATER_FLOW_ACCUMULATION=21

  headwater_cells = flow_accum_arr==HEADWATER_FLOW_ACCUMULATION
  expected_num_headwater_cells = headwater_cells.sum()
  headwater_cell_values = headwater_arr[headwater_cells]
  num_headwater_cells = (headwater_cell_values>=0).sum()
  if num_headwater_cells!=expected_num_headwater_cells:
    assert False,'Headwater cells not delineated: expected %d, got %d'%(expected_num_headwater_cells,num_headwater_cells)



