"""
Refreshing a project's template notebooks without losing the user's work.

`fire-impacts update` overwrites the notebooks in a user's project, so
the thing worth testing hard is the distinction it makes before doing
so: a pristine copy of a template can be replaced outright, an edited
one has to be backed up first, and merely *running* a notebook is not
an edit.
"""

import json
import os

import jupytext
import pytest

from fire_impacts import notebooks as nb


TEMPLATE_A = '''# %% [markdown]
# # First template

# %%
value = 1
'''

TEMPLATE_B = '''# %% [markdown]
# # First template

# %%
value = 2

# %%
extra = 'added in a later release'
'''

TEMPLATE_OTHER = '''# %%
other = True
'''


@pytest.fixture()
def templates(tmp_path, monkeypatch):
    """
    Stand in a controllable template directory for the shipped one.

    Returns a callable that writes a named template, so a test can
    publish a "new release" of one mid-way through.
    """
    directory = tmp_path / 'templates'
    directory.mkdir()
    monkeypatch.setattr(nb, 'template_dir', lambda: str(directory))

    def publish(name, content):
        (directory / f'{name}.py').write_text(content)

    publish('Alpha', TEMPLATE_A)
    return publish


@pytest.fixture()
def project(tmp_path):
    """An empty project directory."""
    path = tmp_path / 'project'
    path.mkdir()
    return str(path)


def actions(states):
    """Map notebook name to the action recorded against it."""
    return {s.name: s.action for s in states}


def read_notebook(project, name):
    """Load a project notebook as parsed JSON."""
    with open(os.path.join(project, f'{name}.ipynb')) as f:
        return json.load(f)


def write_notebook(project, name, doc):
    """Write a parsed notebook back into a project."""
    with open(os.path.join(project, f'{name}.ipynb'), 'w') as f:
        json.dump(doc, f)


def edit_notebook(project, name):
    """Make a substantive edit to a project notebook."""
    doc = read_notebook(project, name)
    doc['cells'][-1]['source'] = ['value = 99  # my calibration\n']
    write_notebook(project, name, doc)


def run_notebook(project, name):
    """
    Simulate executing a notebook: outputs, execution counts, a kernel.

    None of this is a change to the notebook's content, and none of it
    should register as an edit.
    """
    doc = read_notebook(project, name)
    for cell in doc['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = 7
            cell['outputs'] = [{
                'output_type': 'stream',
                'name': 'stdout',
                'text': ['1\n'],
                }]
    doc['metadata']['kernelspec'] = {
        'name': 'conda-env-fire', 'display_name': 'fire',
        }
    write_notebook(project, name, doc)


class TestInstalling:

    def test_installs_script_and_notebook(self, templates, project):
        nb.refresh_notebooks(project)

        assert os.path.exists(os.path.join(project, 'Alpha.py'))
        assert os.path.exists(os.path.join(project, 'Alpha.ipynb'))

    def test_records_what_it_installed(self, templates, project):
        nb.refresh_notebooks(project)

        record = nb.read_manifest(project)['Alpha']
        assert record['py'] == nb.template_fingerprints('Alpha')['py']
        assert record['ipynb'] == nb.template_fingerprints('Alpha')['ipynb']
        assert record['installed']

    def test_adds_templates_released_later(self, templates, project):
        nb.refresh_notebooks(project)
        templates('Beta', TEMPLATE_OTHER)

        assert actions(nb.refresh_notebooks(project)) == {
            'Alpha': nb.ACTION_CURRENT, 'Beta': nb.ACTION_INSTALL,
            }

    def test_creates_the_project_directory(self, templates, tmp_path):
        path = str(tmp_path / 'brand-new')
        nb.refresh_notebooks(path)

        assert os.path.exists(os.path.join(path, 'Alpha.ipynb'))


class TestDetectingEdits:

    def test_untouched_notebook_needs_nothing(self, templates, project):
        nb.refresh_notebooks(project)

        assert actions(nb.plan_update(project)) == {
            'Alpha': nb.ACTION_CURRENT,
            }

    def test_running_a_notebook_is_not_an_edit(self, templates, project):
        nb.refresh_notebooks(project)
        run_notebook(project, 'Alpha')

        state, = nb.plan_update(project)
        assert not state.edited
        assert state.action == nb.ACTION_CURRENT

    def test_editing_a_cell_is_an_edit(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')

        state, = nb.plan_update(project)
        assert state.edited
        assert state.action == nb.ACTION_BACKUP

    def test_editing_the_paired_script_is_an_edit(self, templates, project):
        nb.refresh_notebooks(project)
        script = os.path.join(project, 'Alpha.py')
        with open(script, 'a') as f:
            f.write('\n# %%\nmine = True\n')

        assert nb.plan_update(project)[0].edited

    def test_new_template_alone_is_not_an_edit(self, templates, project):
        nb.refresh_notebooks(project)
        templates('Alpha', TEMPLATE_B)

        state, = nb.plan_update(project)
        assert not state.edited
        assert state.action == nb.ACTION_REPLACE

    def test_line_endings_are_not_an_edit(self, templates, project):
        nb.refresh_notebooks(project)
        script = os.path.join(project, 'Alpha.py')
        with open(script, 'rb') as f:
            content = f.read()
        with open(script, 'wb') as f:
            f.write(content.replace(b'\n', b'\r\n'))

        assert not nb.plan_update(project)[0].edited

    def test_deleted_notebook_is_regenerated(self, templates, project):
        nb.refresh_notebooks(project)
        os.remove(os.path.join(project, 'Alpha.ipynb'))

        assert nb.plan_update(project)[0].action == nb.ACTION_REPLACE
        nb.refresh_notebooks(project)
        assert os.path.exists(os.path.join(project, 'Alpha.ipynb'))

    def test_unparseable_notebook_is_treated_as_edited(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        with open(os.path.join(project, 'Alpha.ipynb'), 'w') as f:
            f.write('this is not JSON')

        assert nb.plan_update(project)[0].edited


class TestRefreshing:

    def test_edited_notebook_is_backed_up_then_replaced(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)

        state, = nb.refresh_notebooks(project)

        # The user's version is recoverable...
        backup, = [p for p in state.backed_up if p.endswith('.ipynb')]
        assert 'my calibration' in open(backup).read()

        # ...and the project now has the new template.
        assert 'added in a later release' in open(
            os.path.join(project, 'Alpha.py')
            ).read()

    def test_backup_keeps_both_halves_of_the_pair(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')

        state, = nb.refresh_notebooks(project)
        assert sorted(os.path.basename(p) for p in state.backed_up) == [
            'Alpha.ipynb', 'Alpha.py',
            ]

    def test_backups_land_under_the_backup_folder(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')

        state, = nb.refresh_notebooks(project)
        folder = os.path.dirname(state.backed_up[0])
        assert os.path.dirname(folder) == os.path.join(
            project, nb.BACKUP_FOLDER_NAME
            )

    def test_one_backup_folder_per_refresh(self, templates, project):
        templates('Beta', TEMPLATE_OTHER)
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        edit_notebook(project, 'Beta')

        states = nb.refresh_notebooks(project)
        folders = {os.path.dirname(p)
                   for s in states for p in s.backed_up}
        assert len(folders) == 1

    def test_unedited_notebook_is_replaced_without_a_backup(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        templates('Alpha', TEMPLATE_B)

        state, = nb.refresh_notebooks(project)
        assert state.backed_up == []
        assert not os.path.exists(
            os.path.join(project, nb.BACKUP_FOLDER_NAME)
            )

    def test_refreshing_updates_the_record(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)
        nb.refresh_notebooks(project)

        # The edit is gone, so a second refresh has nothing left to do.
        assert actions(nb.plan_update(project)) == {
            'Alpha': nb.ACTION_CURRENT,
            }

    def test_dry_run_changes_nothing(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)
        before = open(os.path.join(project, 'Alpha.ipynb')).read()

        states = nb.refresh_notebooks(project, dry_run=True)

        assert actions(states) == {'Alpha': nb.ACTION_BACKUP}
        assert open(os.path.join(project, 'Alpha.ipynb')).read() == before
        assert not os.path.exists(
            os.path.join(project, nb.BACKUP_FOLDER_NAME)
            )

    def test_only_new_leaves_existing_notebooks_alone(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)
        templates('Beta', TEMPLATE_OTHER)
        before = open(os.path.join(project, 'Alpha.ipynb')).read()

        states = nb.refresh_notebooks(project, only_new=True)

        assert actions(states) == {
            'Alpha': nb.ACTION_SKIP, 'Beta': nb.ACTION_INSTALL,
            }
        assert open(os.path.join(project, 'Alpha.ipynb')).read() == before

    def test_no_backup_overwrites_without_keeping_a_copy(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)

        state, = nb.refresh_notebooks(project, backup=False)

        assert state.backed_up == []
        assert not os.path.exists(
            os.path.join(project, nb.BACKUP_FOLDER_NAME)
            )
        assert 'added in a later release' in open(
            os.path.join(project, 'Alpha.py')
            ).read()

    def test_named_subset_only(self, templates, project):
        templates('Beta', TEMPLATE_OTHER)
        nb.refresh_notebooks(project, names=['Alpha'])

        assert not os.path.exists(os.path.join(project, 'Beta.py'))


class TestProjectsPredatingTheManifest:
    """
    Projects created before installs were recorded have nothing to
    compare against, so the fallback is the current template: identical
    means pristine, anything else is assumed to be the user's work.
    """

    def test_untracked_pristine_copy_is_left_alone(
        self, templates, project,
        ):
        nb.refresh_notebooks(project)
        os.remove(nb.manifest_path(project))

        assert actions(nb.plan_update(project)) == {
            'Alpha': nb.ACTION_CURRENT,
            }

    def test_untracked_edit_is_backed_up(self, templates, project):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        os.remove(nb.manifest_path(project))

        state, = nb.plan_update(project)
        assert state.action == nb.ACTION_BACKUP
        assert not state.recorded

    def test_untracked_old_template_is_backed_up(self, templates, project):
        # Indistinguishable from an edit without a record, so it is
        # treated as one - a spare copy is the harmless outcome.
        nb.refresh_notebooks(project)
        os.remove(nb.manifest_path(project))
        templates('Alpha', TEMPLATE_B)

        assert nb.plan_update(project)[0].action == nb.ACTION_BACKUP

    def test_unreadable_manifest_is_ignored(self, templates, project):
        nb.refresh_notebooks(project)
        with open(nb.manifest_path(project), 'w') as f:
            f.write('{ truncated')

        assert nb.read_manifest(project) == {}
        assert actions(nb.plan_update(project)) == {
            'Alpha': nb.ACTION_CURRENT,
            }


class TestShippedTemplates:
    """The real templates, not the fixtures."""

    def test_every_template_is_valid_jupytext(self):
        names = nb.template_names()
        assert 'PrepareData' in names

        for name in names:
            notebook = jupytext.read(nb.template_path(name), fmt='py:percent')
            assert notebook.cells

    def test_fingerprints_are_stable(self):
        first = nb.template_fingerprints('PrepareData')
        assert first == nb.template_fingerprints('PrepareData')

    def test_templates_ship_inside_the_package(self):
        # They have to live in the package to survive installation: a
        # top-level templates/ directory exists in a source checkout but
        # not in an installed wheel, which left `fire-impacts new` with
        # nothing to copy.
        import fire_impacts

        package = os.path.dirname(os.path.abspath(fire_impacts.__file__))
        assert nb.template_dir() == os.path.join(package, 'templates')


class TestCommandLine:
    """The `fire-impacts` commands, as a user would invoke them."""

    @pytest.fixture()
    def run(self, templates):
        from typer.testing import CliRunner
        from fire_impacts import cli

        runner = CliRunner()
        return lambda *args: runner.invoke(cli.app, list(args))

    def test_update_refuses_a_path_that_is_not_a_project(self, run, tmp_path):
        result = run('update', str(tmp_path / 'nowhere'))

        assert result.exit_code != 0

    def test_update_backs_up_an_edited_notebook(
        self, run, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        templates('Alpha', TEMPLATE_B)

        assert run('update', project).exit_code == 0

        backups = os.path.join(project, nb.BACKUP_FOLDER_NAME)
        stamp, = os.listdir(backups)
        assert 'my calibration' in open(
            os.path.join(backups, stamp, 'Alpha.ipynb')
            ).read()

    def test_dry_run_leaves_the_project_untouched(
        self, run, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')
        before = open(os.path.join(project, 'Alpha.ipynb')).read()

        assert run('update', project, '--dry-run').exit_code == 0
        assert open(os.path.join(project, 'Alpha.ipynb')).read() == before

    def test_status_reports_an_edited_notebook(
        self, run, templates, project,
        ):
        nb.refresh_notebooks(project)
        edit_notebook(project, 'Alpha')

        result = run('status', project)
        assert result.exit_code == 0
        assert 'edited' in result.stdout

    def test_status_reports_an_untouched_notebook(
        self, run, templates, project,
        ):
        nb.refresh_notebooks(project)

        result = run('status', project)
        assert 'up to date' in result.stdout
