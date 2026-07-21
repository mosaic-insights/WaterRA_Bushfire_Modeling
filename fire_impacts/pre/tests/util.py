from pathlib import Path
import pytest
from fire_impacts import FireImpactsProject

# The example dataset in test_data/. add_catchment() derives the catchment
# name from the file's basename, so CATCHMENT tracks CATCHMENT_FILE.
CATCHMENT_FILE = 'EgSmallCatchment_7899.shp'
CATCHMENT = 'EgSmallCatchment_7899'
DEM_FILE = 'DEM_10m_EgSmallCatchment_7899.tif'


@pytest.fixture()
def get_file():
    def _(file_path:str):
        import fire_impacts
        return (Path(fire_impacts.__file__).parent.parent / 'test_data' / file_path)
    return _


@pytest.fixture()
def get_project(tmp_path, get_file):
    def _():
        proj_dir = tmp_path / 'project'
        assert not proj_dir.exists()
        proj = FireImpactsProject(proj_dir, exist_ok=False,clear=False)
        proj.add_catchment(get_file(CATCHMENT_FILE))
        return proj
    return _
