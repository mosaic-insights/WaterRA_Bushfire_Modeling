from pathlib import Path
import pytest
from fire_impacts import FireImpactsProject

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
        proj.add_catchment(get_file('example_small_catchment.json'))
        return proj
    return _
