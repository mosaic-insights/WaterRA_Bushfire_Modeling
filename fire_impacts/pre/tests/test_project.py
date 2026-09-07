
from fire_impacts import FireImpactsProject
from .util import *

def test_create_project(tmp_path):
  proj_dir = tmp_path / 'project'
  assert not proj_dir.exists()
  proj = FireImpactsProject(proj_dir, exist_ok=False,clear=False)
  assert proj_dir.exists()

  other_prj = FireImpactsProject(proj_dir, exist_ok=True,clear=False)

def test_reopening_a_project_without_clearing_keeps_its_contents(tmp_path):
  """`clear = false` in study.toml has to mean "reopen this project".

  The folder is already there on the very first PrepareData run -
  `fire-impacts new` leaves a settings.json and a Catchments/ behind - so
  a construction that treats an existing folder as an obstacle makes the
  setting unusable rather than merely strict.
  """
  proj_dir = tmp_path / 'project'

  first = FireImpactsProject(proj_dir, clear=False, exist_ok=True)
  first.catchments.append('Kept')
  first._write()

  second = FireImpactsProject(proj_dir, clear=False, exist_ok=True)
  assert second.catchments == ['Kept'], \
    'reopening with clear=False must not re-initialise the project'

def test_clearing_a_project_still_wipes_what_was_registered(tmp_path):
  """The default path, unchanged: clear=True drops the catchments that
  were registered rather than reopening them."""
  proj_dir = tmp_path / 'project'

  first = FireImpactsProject(proj_dir, clear=True, exist_ok=False)
  first.catchments.append('Gone')
  first._write()

  second = FireImpactsProject(proj_dir, clear=True, exist_ok=False)
  assert second.catchments == [], \
    'clear=True must not carry the old catchments forward'

def test_add_catchment(tmp_path,get_file):
  proj_dir = tmp_path / 'project'
  assert not proj_dir.exists()
  proj = FireImpactsProject(proj_dir, exist_ok=False,clear=False)
  fn = get_file(CATCHMENT_FILE)
  proj.add_catchment(fn)
  catch_path = proj_dir / 'Catchments' / CATCHMENT / 'Topography'
  assert catch_path.exists(), 'Catchment directories not created'

