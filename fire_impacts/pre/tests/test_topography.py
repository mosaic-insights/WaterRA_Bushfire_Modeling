from .util import *
import rasterio as rio
from fire_impacts import const
from fire_impacts.pre import topography
from fire_impacts.context import RunContext

def get_raster_array(fn):
  with rio.open(fn) as src:
    return src.read(1)

def get_threshold_cells(fn):
  '''
  Contributing area threshold, in cells, for the grid that fn sits on.
  Mirrors the conversion in extract_headwaters() so the test tracks the
  DEM resolution instead of hard coding a cell count.
  '''
  with rio.open(fn) as src:
    res_sq = src.transform[0] * abs(src.transform[4])
  return int(const.DEFAULT_HW_THRESHOLD / res_sq)

def test_headwater_delineation(get_project,get_file):
  proj = get_project()
  ctx = RunContext.solo_catchment(proj)

  topography.extract_catchment_dems(ctx, get_file(DEM_FILE))
  fn = proj.catchment_path(CATCHMENT,'Topography','DEM.tif')
  assert Path(fn).exists(), 'DEM not extracted'

  topography.extract_headwaters(ctx)
  headwater_fn = proj.catchment_path(CATCHMENT,'Topography','Headwaters.tif')
  assert Path(headwater_fn).exists(), 'Headwaters not extracted'

  flow_accum_fn = proj.catchment_path(CATCHMENT,'Topography','Flow_accumulation.tif')
  assert Path(flow_accum_fn).exists(), 'Flow accumulation not extracted'

  headwater_arr = get_raster_array(headwater_fn)
  flow_accum_arr = get_raster_array(flow_accum_fn)
  threshold_cells = get_threshold_cells(flow_accum_fn)

  headwater_cells = flow_accum_arr==threshold_cells
  expected_num_headwater_cells = headwater_cells.sum()
  assert expected_num_headwater_cells, \
    'No cell sits exactly on the %d cell threshold - test proves nothing'%threshold_cells

  headwater_cell_values = headwater_arr[headwater_cells]
  num_headwater_cells = (headwater_cell_values>=0).sum()
  if num_headwater_cells!=expected_num_headwater_cells:
    assert False,'Headwater cells not delineated: expected %d, got %d'%(expected_num_headwater_cells,num_headwater_cells)
