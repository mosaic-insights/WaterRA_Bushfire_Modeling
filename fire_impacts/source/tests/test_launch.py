"""
Getting a Veneer connection from the [source] settings in study.toml.

Nothing here starts Source: veneer-py's command-line helpers are replaced
with fakes that record what they were asked to do. What is under test is
the decision this module makes - connect, or build and start a command
line - and that every misconfiguration names the setting to fix rather
than failing later inside veneer-py with a message about a path.
"""

import json
import os
from dataclasses import replace

import pytest

from fire_impacts.source import launch, veneer_config
from fire_impacts.study import SourceIntegrationSettings, StudyConfigError

VENEER_EXE = 'FlowMatters.Source.VeneerCmd.exe'


class FakeVeneer:
    """Stands in for veneer.Veneer; records the port it was opened on."""

    def __init__(self, port=9876, plugins=None):
        self.port = port
        self.plugins = plugins

    def status(self):
        # Veneer's root endpoint, which reports plugins as 'PluginsLoaded'.
        # veneer-py's scenario_info() fetches the same document.
        if self.plugins is None:
            return {}
        return {'PluginsLoaded': self.plugins}

    scenario_info = status


class FakeManage:
    """Records calls to the veneer.manage functions launch.py uses."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.built = []
        self.started = []
        self.killed = []
        self.bound_port = 9877

    def create_command_line(self, veneer_path, source_version=None,
                            source_path=None, dest=None, force=True,
                            **kwargs):
        self.built.append(dict(veneer_path=veneer_path,
                               source_version=source_version,
                               source_path=source_path, dest=dest,
                               force=force))
        os.makedirs(dest, exist_ok=True)
        exe = os.path.join(dest, VENEER_EXE)
        open(exe, 'w').close()
        return exe

    def start(self, **kwargs):
        self.started.append(kwargs)
        return ['process'], [self.bound_port], [None]

    def kill_all_now(self, processes):
        self.killed.append(list(processes))


@pytest.fixture()
def manage(tmp_path, monkeypatch):
    fake = FakeManage(tmp_path)
    monkeypatch.setattr(launch, 'create_command_line',
                        fake.create_command_line)
    monkeypatch.setattr(launch, 'start', fake.start)
    monkeypatch.setattr(launch, 'kill_all_now', fake.kill_all_now)
    monkeypatch.setattr(launch, 'Veneer', FakeVeneer)
    return fake


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    return str(path)


@pytest.fixture()
def project(tmp_path):
    return touch(tmp_path / 'models' / 'Cat.rsproj')


@pytest.fixture()
def prebuilt(tmp_path):
    return touch(tmp_path / 'cmd' / VENEER_EXE)


@pytest.fixture()
def install(tmp_path):
    """A Source install and a Veneer release, as directories."""
    source_dir = tmp_path / 'eWater' / 'Source 5.42.0.12345'
    veneer_dir = tmp_path / 'Veneer' / 'Source 5.42'
    source_dir.mkdir(parents=True)
    veneer_dir.mkdir(parents=True)
    return str(source_dir), str(veneer_dir)


def settings(**kwargs):
    return replace(SourceIntegrationSettings(), **kwargs)


# ---------------------------------------------------------------------------
# Connecting to a Source that is already open
# ---------------------------------------------------------------------------

def test_without_a_project_file_it_connects_to_the_open_source(manage):
    session = launch.open_source(settings(port=9880))

    assert not session.launched
    assert session.v.port == 9880
    assert manage.started == []


def test_closing_a_connection_to_an_open_source_leaves_it_running(manage):
    session = launch.open_source(settings())
    session.close()

    assert manage.killed == []


@pytest.mark.parametrize('name', [
    'veneer_command_line', 'source_dir', 'veneer_dir', 'command_line_dir'])
def test_command_line_settings_without_a_project_file_are_refused(
        manage, tmp_path, name):
    """Ignoring them would leave someone believing the notebook started a
    command line for them, when it is driving whatever Source is open."""
    with pytest.raises(StudyConfigError) as exc:
        launch.open_source(settings(**{name: str(tmp_path)}))
    assert f'source.{name}' in str(exc.value)
    assert 'source.project_file' in str(exc.value)


def test_plugins_without_a_project_file_are_refused(manage, tmp_path):
    with pytest.raises(StudyConfigError, match='source.plugins'):
        launch.open_source(settings(plugins=[touch(tmp_path / 'A.dll')]))


# ---------------------------------------------------------------------------
# Choosing a command line
# ---------------------------------------------------------------------------

def test_a_prebuilt_command_line_is_used_as_it_is(manage, project, prebuilt):
    exe = launch.veneer_command_line(
        settings(project_file=project, veneer_command_line=prebuilt))

    assert exe == prebuilt
    assert manage.built == []


def test_a_missing_prebuilt_command_line_names_the_setting(
        manage, project, tmp_path):
    with pytest.raises(StudyConfigError, match='source.veneer_command_line'):
        launch.veneer_command_line(settings(
            project_file=project,
            veneer_command_line=str(tmp_path / 'nowhere' / VENEER_EXE)))


def test_a_command_line_is_built_from_source_and_veneer(
        manage, project, install, tmp_path):
    source_dir, veneer_dir = install
    dest = str(tmp_path / 'built')

    exe = launch.veneer_command_line(settings(
        project_file=project, source_dir=source_dir, veneer_dir=veneer_dir,
        command_line_dir=dest))

    assert exe == os.path.join(dest, VENEER_EXE)
    [call] = manage.built
    # source_version=None: source_path is the install itself, not a parent
    # to search for a "Source <version>" folder in.
    assert call == dict(veneer_path=veneer_dir, source_version=None,
                        source_path=source_dir, dest=dest, force=True)


def test_a_built_command_line_is_reused(manage, project, install, tmp_path):
    """The copy is hundreds of MB - rebuilding it every session is not on."""
    source_dir, veneer_dir = install
    config = settings(project_file=project, source_dir=source_dir,
                      veneer_dir=veneer_dir,
                      command_line_dir=str(tmp_path / 'built'))

    launch.veneer_command_line(config)
    launch.veneer_command_line(config)

    assert len(manage.built) == 1


def test_a_different_veneer_rebuilds_the_command_line(
        manage, project, install, tmp_path):
    """Same directory, different ingredients: reusing the old build would
    run a Veneer that is not the one study.toml names."""
    source_dir, veneer_dir = install
    other_veneer = tmp_path / 'Veneer' / 'Source 5.42 newer'
    other_veneer.mkdir()
    config = settings(project_file=project, source_dir=source_dir,
                      veneer_dir=veneer_dir,
                      command_line_dir=str(tmp_path / 'built'))

    launch.veneer_command_line(config)
    launch.veneer_command_line(replace(config, veneer_dir=str(other_veneer)))

    assert len(manage.built) == 2


def test_the_build_records_what_it_was_built_from(
        manage, project, install, tmp_path):
    source_dir, veneer_dir = install
    dest = tmp_path / 'built'
    launch.veneer_command_line(settings(
        project_file=project, source_dir=source_dir, veneer_dir=veneer_dir,
        command_line_dir=str(dest)))

    stamp = json.loads((dest / launch.BUILD_STAMP).read_text())
    assert stamp == {'source_dir': source_dir, 'veneer_dir': veneer_dir}


def test_the_default_build_directory_is_per_user_and_per_source_version(
        manage, project, install, tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    source_dir, veneer_dir = install

    exe = launch.veneer_command_line(settings(
        project_file=project, source_dir=source_dir, veneer_dir=veneer_dir))

    assert exe == str(tmp_path / 'appdata' / 'fire_impacts' / 'veneer_cmd'
                      / 'Source 5.42.0.12345' / VENEER_EXE)


def test_source_dir_without_veneer_dir_names_the_missing_one(
        manage, project, install):
    source_dir, _ = install
    with pytest.raises(StudyConfigError, match='source.veneer_dir'):
        launch.veneer_command_line(
            settings(project_file=project, source_dir=source_dir))


def test_veneer_dir_without_source_dir_names_the_missing_one(
        manage, project, install):
    _, veneer_dir = install
    with pytest.raises(StudyConfigError, match='source.source_dir'):
        launch.veneer_command_line(
            settings(project_file=project, veneer_dir=veneer_dir))


def test_both_ways_of_getting_a_command_line_at_once_is_refused(
        manage, project, prebuilt, install):
    """Which one would win is not something a user should have to guess."""
    source_dir, veneer_dir = install
    with pytest.raises(StudyConfigError) as exc:
        launch.veneer_command_line(settings(
            project_file=project, veneer_command_line=prebuilt,
            source_dir=source_dir, veneer_dir=veneer_dir))
    assert 'source.veneer_command_line' in str(exc.value)
    assert 'source.source_dir' in str(exc.value)


def test_neither_way_of_getting_a_command_line_says_what_to_set(
        manage, project):
    with pytest.raises(StudyConfigError) as exc:
        launch.veneer_command_line(settings(project_file=project))
    message = str(exc.value)
    assert 'source.veneer_command_line' in message
    assert 'source.source_dir' in message
    assert 'source.veneer_dir' in message


def test_a_missing_source_install_names_the_setting(
        manage, project, install, tmp_path):
    _, veneer_dir = install
    with pytest.raises(StudyConfigError, match='source.source_dir'):
        launch.veneer_command_line(settings(
            project_file=project, source_dir=str(tmp_path / 'nowhere'),
            veneer_dir=veneer_dir))


# ---------------------------------------------------------------------------
# Starting the command line
# ---------------------------------------------------------------------------

def test_a_project_file_starts_a_command_line_on_it(
        manage, project, prebuilt, tmp_path):
    plugin = touch(tmp_path / 'plugins' / 'LoadDistributor.dll')

    session = launch.open_source(settings(
        project_file=project, veneer_command_line=prebuilt,
        plugins=[plugin], port=9876, detached=True))

    [call] = manage.started
    assert call['project_fn'] == project
    assert call['veneer_exe'] == prebuilt
    assert call['ports'] == 9876
    assert call['additional_plugins'] == [plugin]
    assert call['detached'] is True
    # v.model.* - including v.model.save - is IronPython, and needs it.
    assert call['script'] is True
    assert session.launched


def test_the_session_talks_to_the_port_actually_bound(
        manage, project, prebuilt):
    """start() moves along to a free port if the requested one is taken."""
    manage.bound_port = 9879

    session = launch.open_source(
        settings(project_file=project, veneer_command_line=prebuilt))

    assert session.port == 9879
    assert session.v.port == 9879


def test_closing_a_launched_session_stops_the_command_line_once(
        manage, project, prebuilt):
    session = launch.open_source(
        settings(project_file=project, veneer_command_line=prebuilt))

    session.close()
    session.close()

    assert manage.killed == [['process']]


def test_a_session_closes_itself_as_a_context_manager(
        manage, project, prebuilt):
    with launch.open_source(settings(
            project_file=project, veneer_command_line=prebuilt)):
        pass

    assert manage.killed == [['process']]


def test_a_missing_project_file_names_the_setting(
        manage, prebuilt, tmp_path):
    with pytest.raises(StudyConfigError, match='source.project_file'):
        launch.open_source(settings(
            project_file=str(tmp_path / 'nowhere.rsproj'),
            veneer_command_line=prebuilt))
    assert manage.started == []


def test_a_missing_plugin_names_the_setting_and_the_file(
        manage, project, prebuilt, tmp_path):
    missing = str(tmp_path / 'nowhere.dll')
    with pytest.raises(StudyConfigError) as exc:
        launch.open_source(settings(
            project_file=project, veneer_command_line=prebuilt,
            plugins=[missing]))
    assert 'source.plugins' in str(exc.value)
    assert missing in str(exc.value)
    assert manage.started == []


# ---------------------------------------------------------------------------
# Checking the plugins made it in
# ---------------------------------------------------------------------------

def test_plugins_reported_by_source_pass_the_check(caplog):
    """Passing has to mean the plugin was found, not that the check found
    no plugin list to look in and let everything through."""
    v = FakeVeneer(
        plugins=['C:\\Plugins\\FlowMatters.Source.LoadDistributor.dll'])
    veneer_config.check_plugins_loaded(
        v, ['D:/elsewhere/flowmatters.source.loaddistributor.dll'])
    assert 'cannot verify' not in caplog.text


def test_the_load_distributor_check_reads_the_same_field():
    v = FakeVeneer(plugins=['C:\\Plugins\\Other.dll'])
    with pytest.raises(RuntimeError, match='LoadDistributor'):
        veneer_config.check_load_distributor_plugin(v)


def test_a_plugin_source_did_not_load_is_reported_by_name():
    v = FakeVeneer(plugins=['C:\\Plugins\\Other.dll'])
    with pytest.raises(RuntimeError, match='LoadDistributor.dll'):
        veneer_config.check_plugins_loaded(
            v, ['C:/Plugins/LoadDistributor.dll'])


def test_an_older_veneer_that_lists_no_plugins_is_let_through(caplog):
    veneer_config.check_plugins_loaded(FakeVeneer(plugins=None), ['A.dll'])
    assert 'cannot verify' in caplog.text
