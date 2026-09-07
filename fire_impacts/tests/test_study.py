"""
Loading a project's study.toml.

The file is hand-edited by people who are not programmers, so the two
behaviours that matter most are that a typo is refused rather than
ignored - a silently dropped setting would let someone believe they had
configured a study they had not - and that a setting left out falls back
to its default, which is what lets an older file keep working against
newer templates.
"""

import os

import pytest

from fire_impacts.study import (
    CONFIG_NAME, StudyConfigError, StudySettings, load_study,
)

MINIMAL = """
[catchment]
name     = "Cat"
boundary = 'boundary.shp'
aridity  = 'aridity.tif'

[event]
name       = "2019_fire"
fire_start = "2019-01-15"
fire_end   = "2019-03-07"
"""


def write_study(directory, text=MINIMAL, files=('boundary.shp',
                                                'aridity.tif')):
    """Write a study.toml plus the input files it points at."""
    (directory / CONFIG_NAME).write_text(text, encoding='utf-8')
    for name in files:
        (directory / name).write_text('', encoding='utf-8')
    return directory


def test_a_minimal_file_loads(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.catchment.name == 'Cat'
    assert study.event.fire_start == '2019-01-15'


def test_settings_left_out_fall_back_to_defaults(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.ensemble.name == 'stochastic'
    assert study.ensemble.num_replicates == 10
    assert study.source.port == 9876


def test_clear_is_unset_rather_than_false_when_omitted(tmp_path):
    """Tri-state: PrepareData resolves unset to True, everything else
    treats it as 'do not clear'. A plain boolean could not tell an
    explicit false from an omission."""
    study = load_study(str(write_study(tmp_path)))
    assert study.project.clear is None


def test_an_explicit_false_is_distinguishable_from_an_omission(tmp_path):
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[project]\nclear = false\n')))
    assert study.project.clear is False


def test_a_misspelled_setting_is_refused_with_a_suggestion(tmp_path):
    text = MINIMAL.replace('fire_start', 'fire_strat')
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(write_study(tmp_path, text)))
    assert 'fire_strat' in str(exc.value)
    assert "Did you mean 'fire_start'" in str(exc.value)
    # Not just 'setting' - that word is also a substring of pytest's own
    # tmp_path directory name for this test, so it would pass regardless
    # of what the message actually says.
    assert 'Unknown setting' in str(exc.value)


def test_a_misspelled_section_is_refused(tmp_path):
    text = MINIMAL.replace('[event]', '[evnt]')
    with pytest.raises(StudyConfigError, match="Did you mean 'event'"):
        load_study(str(write_study(tmp_path, text)))


def test_a_missing_required_setting_names_the_file(tmp_path):
    text = MINIMAL.replace('name     = "Cat"', '')
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(write_study(tmp_path, text)))
    assert 'catchment.name' in str(exc.value)
    assert CONFIG_NAME in str(exc.value)


def test_a_missing_file_says_how_to_make_one(tmp_path):
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(tmp_path))
    assert CONFIG_NAME in str(exc.value)
    assert str(tmp_path) in str(exc.value)
    assert 'fire-impacts update' in str(exc.value)


def test_broken_toml_is_reported_as_a_config_error(tmp_path):
    with pytest.raises(StudyConfigError, match=CONFIG_NAME):
        load_study(str(write_study(tmp_path, '[catchment\nname = 1')))


def test_an_unreadable_file_is_reported_as_a_config_error(tmp_path):
    """A directory named study.toml, or a locked file, raises a bare
    OSError from open() - not just a TOMLDecodeError. Both must come
    back as a StudyConfigError, not an unhandled traceback."""
    (tmp_path / CONFIG_NAME).mkdir()
    with pytest.raises(StudyConfigError, match=CONFIG_NAME):
        load_study(str(tmp_path))


def test_recovery_breakpoints_load_as_a_list_of_numbers(tmp_path):
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\nrecovery_breakpoints = [0, 0.5, 1]\n')))
    assert study.event.recovery_breakpoints == [0.0, 0.5, 1.0]
    # [0, 0.5, 1] == [0.0, 0.5, 1.0] is true by numeric equality even if
    # the ints were never coerced to float - check the element types too,
    # since the digest downstream treats 0 and 0.0 as different values.
    assert [type(v) for v in study.event.recovery_breakpoints] == [
        float, float, float]


def test_which_settings_came_from_the_file_is_recorded(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert 'catchment.name' in study.provided
    assert 'ensemble.num_replicates' not in study.provided


def test_root_is_the_projects_absolute_directory(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.root == str(tmp_path.resolve())


def test_root_raises_on_a_settings_instance_not_loaded_from_a_file(tmp_path):
    """dataclasses.replace() rebuilds through __init__ and drops the
    attachment, so an instance that was never loaded must fail loudly
    rather than hand back a plausible-but-wrong os.getcwd()."""
    with pytest.raises(StudyConfigError):
        StudySettings().root
    with pytest.raises(StudyConfigError):
        StudySettings().provided


def test_a_stray_setting_outside_any_section_is_named(tmp_path):
    """A setting typed above its [section] - the most likely mistake a
    non-programmer makes - should say so, not "Unknown section [name]"."""
    text = 'stray = 1\n' + MINIMAL
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(write_study(tmp_path, text)))
    # Quoted, not just a bare substring check - pytest's own tmp_path
    # directory name for this test already contains "stray" (from
    # "test_a_stray_..."), so an unquoted check would pass regardless of
    # whether the message actually names the offending key.
    assert "'stray'" in str(exc.value)
    assert 'section' in str(exc.value)
    assert 'Unknown section' not in str(exc.value)


def test_paths_resolve_against_the_file_not_the_cwd(tmp_path, monkeypatch):
    """A notebook can be run from anywhere; '..\\test_data\\x.shp' has to
    mean the same thing regardless of where the kernel started."""
    project = tmp_path / 'proj'
    project.mkdir()
    write_study(project)

    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    study = load_study(str(project))
    assert os.path.isabs(study.catchment.boundary)
    assert study.catchment.boundary == str(project / 'boundary.shp')


def test_the_project_directory_is_resolved_too(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert os.path.isabs(study.project.directory)
    assert study.project.directory == str(tmp_path)


def test_a_missing_required_input_is_caught_at_load(tmp_path):
    """Better than a geopandas traceback forty cells later."""
    write_study(tmp_path, files=('aridity.tif',))
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(tmp_path))
    assert 'catchment.boundary' in str(exc.value)
    assert 'boundary.shp' in str(exc.value)


def test_an_optional_path_left_unset_is_not_checked(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.catchment.dem is None


def test_a_supplied_optional_path_must_still_exist(tmp_path):
    text = MINIMAL.replace(
        "aridity  = 'aridity.tif'",
        "aridity  = 'aridity.tif'\ndem = 'nope.tif'")
    write_study(tmp_path, text)
    with pytest.raises(StudyConfigError, match='catchment.dem'):
        load_study(str(tmp_path))


def test_the_key_comes_from_the_file_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv('TERN_API_KEY', 'from-env')
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[secrets]\ntern_api_key = "from-file"\n')))
    assert study.secret('tern_api_key') == 'from-file'


def test_the_key_falls_back_to_the_environment_variable(tmp_path,
                                                        monkeypatch):
    """Anyone who followed the old notebook instructions keeps working."""
    monkeypatch.setenv('TERN_API_KEY', 'from-env')
    study = load_study(str(write_study(tmp_path)))
    assert study.secret('tern_api_key') == 'from-env'


def test_a_missing_key_raises_only_when_it_is_asked_for(tmp_path,
                                                        monkeypatch):
    """A blank key must not stop someone running the Simulation notebook,
    which never touches TERN."""
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(tmp_path)))       # no raise
    with pytest.raises(StudyConfigError) as exc:
        study.secret('tern_api_key')
    assert 'tern_api_key' in str(exc.value)
    assert 'TERN_API_KEY' in str(exc.value)


def test_describe_says_where_each_value_came_from(tmp_path):
    text = MINIMAL + '\n[ensemble]\nnum_replicates = 4\n'
    study = load_study(str(write_study(tmp_path, text)))
    out = study.describe_text()
    assert 'catchment.name' in out

    # Not just 'study.toml' in out - the report's own header names the
    # config file ('Study settings from .../study.toml'), so that
    # substring is present unconditionally, regardless of what any
    # field's reported origin actually is. Isolate a setting the fixture
    # actually supplies, and check its own line.
    [provided_line] = [ln for ln in out.splitlines()
                        if 'ensemble.num_replicates' in ln]
    assert 'study.toml' in provided_line

    # Same technique for the opposite case: a setting the fixture leaves
    # unset, in the same [ensemble] section, must report 'default'.
    [default_line] = [ln for ln in out.splitlines()
                       if 'ensemble.inspect_replicate' in ln]
    assert 'default' in default_line

    assert 'parameters.json' in out          # points at the other file


def test_describe_never_prints_the_key(tmp_path, monkeypatch):
    """.ipynb files carry their output cells and are the artefact most
    likely to be emailed or committed."""
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[secrets]\ntern_api_key = "sekrit"\n')))
    out = study.describe_text()
    assert 'sekrit' not in out
    # Not just 'set' in out - 'settings' (the report's own header) already
    # contains that substring, so a check against the whole text would
    # pass regardless of what the secrets.tern_api_key line says. Find
    # that line specifically and check its value token.
    [line] = [ln for ln in out.splitlines() if 'tern_api_key' in ln]
    assert line.split()[1] == 'set'


def test_describe_reports_an_unset_key_as_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(tmp_path)))
    out = study.describe_text()
    [line] = [ln for ln in out.splitlines() if 'tern_api_key' in ln]
    assert 'not set' in line


def test_describe_reports_the_env_var_as_the_origin_when_it_supplies_the_key(
        tmp_path, monkeypatch):
    """Preserve the distinction: study.toml beats the env var, which beats
    the bare default, as the reported origin for a secret."""
    monkeypatch.setenv('TERN_API_KEY', 'from-env')
    study = load_study(str(write_study(tmp_path)))
    out = study.describe_text()
    [line] = [ln for ln in out.splitlines() if 'tern_api_key' in ln]
    assert 'TERN_API_KEY' in line


def test_describe_prints_to_stdout(tmp_path, capsys):
    study = load_study(str(write_study(tmp_path)))
    assert study.describe() is None
    out = capsys.readouterr().out
    assert 'catchment.name' in out


def test_secret_rejects_an_unknown_name(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    with pytest.raises(StudyConfigError) as exc:
        study.secret('tern_api_kee')
    assert 'Unknown secret' in str(exc.value)
    assert "'tern_api_kee'" in str(exc.value)


def test_a_whitespace_only_key_is_treated_as_not_set(tmp_path, monkeypatch):
    """A key pasted badly (leading/trailing whitespace only) must not be
    treated as present - that would surface downstream as a confusing
    TERN auth failure instead of this module's clear "no key available"
    message."""
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[secrets]\ntern_api_key = "   "\n')))
    with pytest.raises(StudyConfigError):
        study.secret('tern_api_key')

    out = study.describe_text()
    [line] = [ln for ln in out.splitlines() if 'tern_api_key' in ln]
    assert 'not set' in line
