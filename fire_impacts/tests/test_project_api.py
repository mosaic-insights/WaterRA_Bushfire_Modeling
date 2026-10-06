"""
`new_project`, `update_project` and `project_status` are the Jupyter/REPL
equivalent of `fire-impacts new`/`update`/`status` - the same logic,
called directly as Python instead of through the CLI. These tests mirror
`test_study_cli.py`'s scenarios one-for-one, calling the functions
instead of invoking `typer`'s CliRunner.
"""

import json
import os

import pytest

from fire_impacts.project_api import new_project, update_project
from fire_impacts.project_api import project_status
from fire_impacts.study import CONFIG_NAME


def test_new_writes_a_study_config(tmp_path):
    project = tmp_path / 'proj'
    written = new_project(str(project), notebooks=False)
    assert (project / CONFIG_NAME).exists()
    assert os.path.basename(written) == CONFIG_NAME


def test_new_seeds_the_config_with_the_example(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'EgSmallCatchment_7899' in text
    assert '2019_fire' in text


def test_new_project_prints_where_the_config_was_written(tmp_path, capsys):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    captured = capsys.readouterr()
    assert CONFIG_NAME in captured.out


def test_a_new_project_has_notebooks_that_can_find_their_config(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project))

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        assert (project / f'{name}.py').exists()
    assert (project / CONFIG_NAME).exists()

    from fire_impacts.study import check_study
    report = check_study(str(project))
    assert report['unknown'] == []
    assert report['required_missing'] == []


def test_update_leaves_an_existing_config_alone(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    (project / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')

    result = update_project(str(project))

    assert (project / CONFIG_NAME).read_text(encoding='utf-8') == '# mine\n'
    assert result.migrated_config is False


def test_update_creates_a_config_for_a_project_that_predates_it(tmp_path):
    """Such a project has notebooks but no config, and after the update its
    notebooks call load_study()."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    result = update_project(str(project))

    assert (project / CONFIG_NAME).exists()
    assert result.migrated_config is True
    # Not re-checked the same call it was written in - matches
    # cli.update's own behaviour (see project_api.update_project).
    assert result.report is None


def test_update_report_reflects_unset_settings_of_an_existing_config(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)

    result = update_project(str(project))

    assert result.migrated_config is False
    assert result.report is not None
    assert isinstance(result.report['unset'], list)


def test_update_raises_for_a_missing_directory(tmp_path):
    missing = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError, match='fire-impacts new'):
        update_project(str(missing))


def test_update_recovers_names_from_an_existing_catchment(tmp_path):
    """The seeding path that actually matters: a project created before
    study.toml existed, with a real catchment, event and ensemble on
    disk, migrated for the first time."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    (project / 'settings.json').write_text(
        json.dumps({'catchments': ['Avon']}), encoding='utf-8')
    events_dir = project / 'Catchments' / 'Avon' / 'Events' / '2019_fire'
    events_dir.mkdir(parents=True)
    ensembles_dir = (
        project / 'Catchments' / 'Avon' / 'Ensembles' / 'stochastic_v2')
    ensembles_dir.mkdir(parents=True)

    update_project(str(project))

    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'name = "Avon"' in text
    assert 'name = "2019_fire"' in text
    assert 'name = "stochastic_v2"' in text
    assert text.count('could not be recovered from the project') == 2


def test_status_reports_a_typo_in_the_config(tmp_path, capsys):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    (project / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')

    result = project_status(str(project))
    captured = capsys.readouterr()

    assert 'naem' in captured.out
    assert "Did you mean 'name'" in captured.out
    assert any('naem' in u for u in result.report['unknown'])


def test_status_reports_a_missing_config(tmp_path, capsys):
    """Both branches of the report name the file, so looking only for
    CONFIG_NAME would pass against a report that never set missing_file
    at all. The report has to say which branch it took, and name the
    command that fixes it."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    result = project_status(str(project))
    captured = capsys.readouterr()

    assert CONFIG_NAME in captured.out
    assert 'missing' in captured.out
    assert 'fire-impacts update' in captured.out
    assert result.report['missing_file'] is True


def test_status_does_not_call_a_config_that_is_there_missing(tmp_path, capsys):
    """The other half. Without it, a status that printed 'missing'
    unconditionally would still satisfy the test above."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)

    project_status(str(project))
    captured = capsys.readouterr()

    assert f'{CONFIG_NAME:<24} present' in captured.out
    assert 'missing' not in captured.out


def test_status_raises_for_a_missing_directory(tmp_path):
    missing = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError):
        project_status(str(missing))


def test_status_returns_the_notebook_states_too(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project))

    result = project_status(str(project))

    names = {s.name for s in result.states}
    assert names == {'PrepareData', 'Simulation', 'SimulationEnsemble',
                      'SourceIntegration'}


def test_the_functions_are_importable_from_the_package_top_level():
    import fire_impacts
    assert fire_impacts.new_project is new_project
    from fire_impacts.project_api import update_project, project_status
    assert fire_impacts.update_project is update_project
    assert fire_impacts.project_status is project_status
