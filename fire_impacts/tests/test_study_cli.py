"""
`fire-impacts new` and `update` look after study.toml.

The rule that matters: the file is created when it is absent and never
touched when it is present. It is the one file in a project that holds
the user's own decisions in a form this package does not replace.
"""

import json
import os

from typer.testing import CliRunner

from fire_impacts.cli import app
from fire_impacts.study import CONFIG_NAME

runner = CliRunner()


def test_new_writes_a_study_config(tmp_path):
    project = tmp_path / 'proj'
    result = runner.invoke(app, ['new', str(project), '--no-notebooks'])
    assert result.exit_code == 0, result.output
    assert (project / CONFIG_NAME).exists()
    assert CONFIG_NAME in result.output


def test_new_seeds_the_config_with_the_example(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'EgSmallCatchment_7899' in text
    assert '2019_fire' in text


def test_update_leaves_an_existing_config_alone(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    (project / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')

    result = runner.invoke(app, ['update', str(project)])
    assert result.exit_code == 0, result.output
    assert (project / CONFIG_NAME).read_text(encoding='utf-8') == '# mine\n'


def test_update_creates_a_config_for_a_project_that_predates_it(tmp_path):
    """Such a project has notebooks but no config, and after the update its
    notebooks call load_study()."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    os.remove(project / CONFIG_NAME)

    result = runner.invoke(app, ['update', str(project)])
    assert (project / CONFIG_NAME).exists()
    assert CONFIG_NAME in result.output


def test_status_reports_a_typo_in_the_config(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    (project / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')

    result = runner.invoke(app, ['status', str(project)])
    assert 'naem' in result.output
    assert "Did you mean 'name'" in result.output


def test_status_reports_a_missing_config(tmp_path):
    """Both branches of the report name the file, so looking only for
    `CONFIG_NAME` would pass against a check_study that never set
    missing_file at all. The report has to say which branch it took, and
    name the command that fixes it."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    os.remove(project / CONFIG_NAME)

    result = runner.invoke(app, ['status', str(project)])
    assert CONFIG_NAME in result.output
    assert 'missing' in result.output
    assert 'fire-impacts update' in result.output


def test_status_does_not_call_a_config_that_is_there_missing(tmp_path):
    """The other half. Without it, a status command that printed
    'missing' unconditionally would still satisfy the test above."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    assert (project / CONFIG_NAME).exists()

    result = runner.invoke(app, ['status', str(project)])
    assert result.exit_code == 0, result.output
    assert f'{CONFIG_NAME:<24} present' in result.output
    assert 'missing' not in result.output


def test_update_recovers_names_from_an_existing_catchment(tmp_path):
    """The seeding path that actually matters: a project created before
    study.toml existed, with a real catchment, event and ensemble on
    disk, migrated for the first time.

    `new --no-notebooks` never registers a catchment, so every other
    test in this file only exercises the empty-project fallback; this
    one builds a project FireImpactsProject itself would recognise.
    """
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    os.remove(project / CONFIG_NAME)

    (project / 'settings.json').write_text(
        json.dumps({'catchments': ['Avon']}), encoding='utf-8')
    events_dir = project / 'Catchments' / 'Avon' / 'Events' / '2019_fire'
    events_dir.mkdir(parents=True)
    # Deliberately not 'stochastic' - that is EnsembleSettings' own
    # default, so a name equal to it would leave the assertion below
    # unable to tell a seeded value from an unseeded, commented-out one.
    ensembles_dir = (
        project / 'Catchments' / 'Avon' / 'Ensembles' / 'stochastic_v2')
    ensembles_dir.mkdir(parents=True)

    result = runner.invoke(app, ['update', str(project)])
    assert result.exit_code == 0, result.output

    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'name = "Avon"' in text
    assert 'name = "2019_fire"' in text
    assert 'name = "stochastic_v2"' in text
    # The two settings a project never records are still flagged, even
    # though the catchment itself was recovered:
    assert text.count('could not be recovered from the project') == 2


def test_a_new_project_has_notebooks_that_can_find_their_config(tmp_path):
    """The whole point: `fire-impacts new` produces a project where the
    notebooks' first cell works."""
    project = tmp_path / 'proj'
    result = runner.invoke(app, ['new', str(project)])
    assert result.exit_code == 0, result.output

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        assert (project / f'{name}.py').exists()
    assert (project / CONFIG_NAME).exists()

    from fire_impacts.study import check_study
    report = check_study(str(project))
    assert report['unknown'] == []
    assert report['required_missing'] == []


def test_every_template_loads_the_study(tmp_path):
    """A template that forgot its settings block would silently keep its
    hard-coded example values."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project)])

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        text = (project / f'{name}.py').read_text(encoding='utf-8')
        assert 'from fire_impacts.study import load_study' in text, name
        assert "load_study('.')" in text, name
