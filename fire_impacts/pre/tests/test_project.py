
from fire_impacts import FireImpactsProject
from .util import *

def test_create_project(tmp_path):
  proj_dir = tmp_path / 'project'
  assert not proj_dir.exists()
  proj = FireImpactsProject(proj_dir, exist_ok=False,clear=False)
  assert proj_dir.exists()

  other_prj = FireImpactsProject(proj_dir, exist_ok=True,clear=False)

def test_add_catchment(tmp_path,get_file):
  proj_dir = tmp_path / 'project'
  assert not proj_dir.exists()
  proj = FireImpactsProject(proj_dir, exist_ok=False,clear=False)
  fn = get_file(CATCHMENT_FILE)
  proj.add_catchment(fn)
  catch_path = proj_dir / 'Catchments' / CATCHMENT / 'Topography'
  assert catch_path.exists(), 'Catchment directories not created'

