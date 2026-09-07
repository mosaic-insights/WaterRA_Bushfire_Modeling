"""
The scaffold that `fire-impacts new` writes.

Its comments are generated from the same field metadata the loader
validates against, so they cannot describe settings the loader would
reject. The round-trip test below is what enforces that: a scaffold that
does not load is a scaffold whose comments have drifted from the code.
"""

import os
from dataclasses import fields

import pytest

import fire_impacts.study_scaffold as study_scaffold
from fire_impacts.study import (
    CONFIG_NAME, CatchmentSettings, StudyConfigError, StudySettings,
    check_study, load_study,
)
from fire_impacts.study_scaffold import (
    EXAMPLE_SEED, example_seed, scaffold_text, write_scaffold,
)


def test_every_setting_appears_in_the_scaffold():
    """Adding a field without documenting it should fail here."""
    text = scaffold_text()
    for group in fields(StudySettings):
        for entry in fields(getattr(StudySettings(), group.name)):
            assert entry.name in text, (
                f'{group.name}.{entry.name} is missing from the scaffold')


def _live_settings(text):
    """Dotted names of the settings a scaffold writes uncommented."""
    live = []
    group = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('[') and line.endswith(']'):
            group = line[1:-1]
        elif line and not line.startswith('#') and '=' in line:
            live.append(f'{group}.{line.split("=")[0].strip()}')
    return live


def _metadata(dotted):
    """The field metadata behind a dotted setting name."""
    group, _, entry = dotted.partition('.')
    by_name = {f.name: f for f in fields(getattr(StudySettings(), group))}
    return by_name[entry].metadata


def test_required_settings_are_uncommented_and_optional_ones_are_not():
    """The convention that answers 'what do I have to change?' at a glance.

    Run against the file `fire-impacts new` actually writes as well as the
    empty-seed form. Only the empty form was checked before, which is how
    the header came to claim something false of every file the tool has
    ever produced: seeding makes optional settings live too.
    """
    for label, seed in (('empty seed', None), ('shipped', example_seed())):
        text = scaffold_text(seed)
        lines = [ln.strip() for ln in text.splitlines()]
        live = _live_settings(text)

        assert 'catchment.name' in live, label          # required
        assert 'catchment.boundary' in live, label      # required
        assert 'ensemble.num_replicates' not in live, label
        assert any(ln.startswith('# num_replicates') for ln in lines), label

        # A live line is one the file has an answer for: the loader
        # requires it, the seed supplies it, or it is prompted for. A
        # live line that is none of those tells a user to supply
        # something nothing will ever ask them for.
        for dotted in live:
            meta = _metadata(dotted)
            assert (meta.get('required') or meta.get('prompt')
                    or dotted in (seed or {})), (
                f'{label}: {dotted} is written live but is neither '
                f'required, prompted, nor seeded')


def test_the_shipped_file_writes_optional_settings_live_too():
    """Exactly why the header cannot say "uncommented means required".

    `fire-impacts new` seeds two optional catchment settings and writes a
    blank TERN key, so three of the live lines in the file it ships are
    settings nobody has to supply. Naming them here means a fourth cannot
    arrive - as the subcatchment pair did - without someone re-reading
    the sentence at the top of the file.
    """
    live = _live_settings(scaffold_text(example_seed()))
    optional_but_live = sorted(
        dotted for dotted in live if not _metadata(dotted).get('required'))

    assert optional_but_live == [
        'catchment.subcatchment_id_field',
        'catchment.subcatchments',
        'secrets.tern_api_key',
    ], optional_but_live


def test_paths_are_written_as_toml_literal_strings():
    """Windows backslashes must not need escaping."""
    text = scaffold_text(EXAMPLE_SEED)
    for line in text.splitlines():
        if 'test_data' in line and not line.strip().startswith('#'):
            assert "'" in line, f'not a literal string: {line}'
            assert '"' not in line, f'double-quoted path: {line}'


def test_the_example_scaffold_loads(tmp_path):
    """Round trip: what `fire-impacts new` writes must be readable by
    load_study, or the two have drifted apart."""
    for name in ('EgSmallCatchment_7899.shp',
                 'AridityPT_EgSmallCatchment_7899.tif',
                 'Subcatchments_EgSmall_7899.shp'):
        (tmp_path / name).write_text('', encoding='utf-8')

    seed = dict(EXAMPLE_SEED)
    seed['catchment.boundary'] = 'EgSmallCatchment_7899.shp'
    seed['catchment.aridity'] = 'AridityPT_EgSmallCatchment_7899.tif'
    seed['catchment.subcatchments'] = 'Subcatchments_EgSmall_7899.shp'

    write_scaffold(str(tmp_path), seed)
    study = load_study(str(tmp_path))

    assert study.catchment.name == 'EgSmallCatchment_7899'
    assert study.event.name == '2019_fire'
    assert study.ensemble.name == 'stochastic'
    assert study.project.clear is None


def test_the_example_seeds_the_label_column_its_shapefile_actually_has():
    """The example's subcatchment shapefile identifies its polygons with
    an 'Id' column, not the 'SiteID' the schema defaults to. Seeding the
    shapefile without the label would register a column that is not
    there: get_subcatchments warns on every call and every consumer falls
    back to row numbers, which is a worse example than none at all."""
    seed = example_seed()

    assert seed['catchment.subcatchments'].endswith(
        'Subcatchments_EgSmall_7899.shp')
    assert seed['catchment.subcatchment_id_field'] == 'Id'
    assert CatchmentSettings().subcatchment_id_field == 'SiteID', (
        'the schema default is what a real study wants; only the '
        'bundled example overrides it')

    # Both are optional settings, so they are only in the generated file
    # at all because they are seeded - and a commented-out line would
    # leave the notebook registering no subcatchments.
    lines = [ln.strip() for ln in scaffold_text(seed).splitlines()]
    assert any(ln.startswith('subcatchments =') for ln in lines)
    assert any(ln.startswith('subcatchment_id_field =') for ln in lines)


def test_write_scaffold_refuses_to_overwrite(tmp_path):
    """study.toml is the user's file. Nothing replaces it."""
    (tmp_path / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')
    with pytest.raises(StudyConfigError, match='already'):
        write_scaffold(str(tmp_path))
    assert (tmp_path / CONFIG_NAME).read_text() == '# mine\n'


def test_example_seed_resolves_paths_when_test_data_is_found():
    """A project created outside the repository must still get a study.toml
    whose paths point at something real - otherwise 'a new project runs end
    to end' holds only for a project made in one particular directory.

    This worktree does have a test_data directory reachable from the
    package, so resolution must actually happen here: a no-op
    implementation would leave the relative literal in place, which is
    not absolute and does not exist relative to the current directory.
    """
    seed = example_seed()

    boundary = seed['catchment.boundary']
    assert os.path.isabs(boundary), boundary
    assert os.path.exists(boundary), boundary
    assert boundary.endswith('EgSmallCatchment_7899.shp'), boundary

    aridity = seed['catchment.aridity']
    assert os.path.isabs(aridity), aridity
    assert os.path.exists(aridity), aridity
    assert aridity.endswith('AridityPT_EgSmallCatchment_7899.tif'), aridity


def test_example_seed_keeps_the_relative_literal_when_test_data_is_missing(
        monkeypatch):
    """An installed wheel ships no test_data. Forcing the 'not found'
    branch (rather than relying on where this test happens to run) is
    what makes the fallback path itself provable."""
    monkeypatch.setattr(study_scaffold.os.path, 'isdir', lambda path: False)
    assert example_seed() == EXAMPLE_SEED


def test_a_seeded_value_that_could_not_be_recovered_is_flagged():
    """Migration: what can't be read off an existing project is left blank
    with a comment telling the user to fill it in - and, since boundary is
    a path setting, the blank value must still be a literal string, not a
    basic (double-quoted) one that would mangle a Windows path the user
    types into it."""
    text = scaffold_text({'catchment.name': 'RealCatchment',
                          'catchment.boundary': None})
    assert 'RealCatchment' in text
    assert 'could not be recovered' in text

    boundary_line = next(
        ln for ln in text.splitlines() if ln.strip().startswith('boundary'))
    assert "boundary = ''" in boundary_line, (
        f'path placeholder is not a literal string: {boundary_line!r}')
    assert '"' not in boundary_line, (
        f'double-quoted path placeholder: {boundary_line!r}')


def test_the_secret_setting_is_written_live_though_not_required():
    """secrets.tern_api_key is the setting most likely to strand someone:
    commented out with an empty value, nobody notices it needs filling in
    until the soil download fails. It is `prompt`ed, not `required` (a
    notebook that never touches TERN must still load), but it must still
    be live."""
    lines = [ln.strip() for ln in scaffold_text().splitlines()]
    assert any(ln.startswith('tern_api_key') for ln in lines)


def test_exactly_one_blank_line_follows_the_header():
    """Every other section is separated by a single blank line; the
    header must read the same way, not two."""
    text = scaffold_text()
    assert '\n\n\n' not in text, 'a double blank line survived'


def test_help_and_section_comments_stay_within_the_line_length_convention():
    """Long help strings (e.g. catchment.boundary's) must be wrapped, not
    emitted verbatim - this is the one artefact where line length is the
    feature, since the file is read by a person, not a machine."""
    text = scaffold_text()
    for line in text.splitlines():
        if line.startswith('#'):
            assert len(line) <= 79, f'comment line too long: {line!r}'


def test_a_multi_paragraph_group_docstring_keeps_every_paragraph():
    """ReportingSettings' docstring has two paragraphs. The second is the
    only place in the whole file that tells a user these thresholds do
    not feed the model - dropping it to show only the summary line would
    lose the one piece of information the section comment exists for."""
    lines = scaffold_text().splitlines()
    start = lines.index('[reporting]')
    end = start + 1
    while lines[end].startswith('#'):
        end += 1
    section_comment = lines[start + 1:end]

    assert any('exceedance maps are drawn at' in ln for ln in section_comment)
    assert any('not model calibration' in ln for ln in section_comment)
    assert '#' in section_comment, (
        'paragraphs are not separated by a bare "#" line')


def test_a_blank_line_separates_the_section_comment_from_its_first_setting():
    """Without it, "what this section is" and "what this setting is" run
    together as one block and the section framing gets lost in it."""
    lines = scaffold_text().splitlines()
    for group in fields(StudySettings):
        start = lines.index(f'[{group.name}]')
        i = start + 1
        assert lines[i].startswith('#'), (
            f'[{group.name}] has no section-docstring comment')
        while lines[i].startswith('#'):
            i += 1
        assert lines[i] == '', (
            f'no blank line after the [{group.name}] section comment')


def test_check_reports_a_missing_file(tmp_path):
    report = check_study(str(tmp_path))
    assert report['missing_file'] is True


def test_check_reports_a_typo_with_a_suggestion(tmp_path):
    (tmp_path / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert any("Did you mean 'name'" in u for u in report['unknown'])


def test_check_lists_settings_available_but_not_set(tmp_path):
    for name in ('b.shp', 'a.tif'):
        (tmp_path / name).write_text('', encoding='utf-8')
    (tmp_path / CONFIG_NAME).write_text(
        "[catchment]\nname='c'\nboundary='b.shp'\naridity='a.tif'\n"
        "[event]\nname='e'\nfire_start='2019-01-15'\n"
        "fire_end='2019-03-07'\n",
        encoding='utf-8')
    report = check_study(str(tmp_path))
    assert report['unknown'] == []
    assert report['required_missing'] == []
    assert 'source.port' in report['unset']


def test_check_reports_a_missing_required_setting(tmp_path):
    (tmp_path / CONFIG_NAME).write_text(
        '[catchment]\nname = "c"\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert 'catchment.boundary' in report['required_missing']


def test_check_survives_an_unreadable_file(tmp_path):
    """A directory named study.toml raises a bare OSError from open() -
    not just a TOMLDecodeError. check_study exists to survive exactly
    this, since fire-impacts status must report the problem, not crash
    on it."""
    (tmp_path / CONFIG_NAME).mkdir()
    report = check_study(str(tmp_path))  # must not raise
    assert report['missing_file'] is False
    assert any('could not be read' in u for u in report['unknown'])


def test_check_reports_a_stray_scalar_instead_of_a_section(tmp_path):
    """`catchment = "oops"` at top level is a plain value where a table
    belongs. Iterating it as if it were a dict of settings would walk its
    characters one at a time - assert that garbage is absent, not just
    that some message showed up."""
    (tmp_path / CONFIG_NAME).write_text(
        'catchment = "oops"\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert any(
        "'catchment' is not inside a [section]" in u
        for u in report['unknown'])
    assert not any(u.startswith('catchment.') for u in report['unknown'])


def test_check_reports_a_table_where_a_scalar_belongs(tmp_path):
    """`boundary = {x = 1}` under [catchment] flattens to
    'catchment.boundary.x', so the dotted name 'catchment.boundary' is
    still (correctly) reported as missing - but silently, unless an
    'unknown' entry also names what was actually found there."""
    (tmp_path / CONFIG_NAME).write_text(
        '[catchment]\nname = "c"\nboundary = {x = 1}\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert 'catchment.boundary' in report['required_missing']
    assert any(
        'catchment.boundary should be a plain value' in u
        for u in report['unknown'])
