"""
Generating a study.toml for a project that has none.

The file this writes is the first thing a user of the template notebooks
meets, so it has to explain itself: every setting carries the `help` text
from the schema, a setting the file has a value for is written live, and
one it does not is commented out with its default shown. Three things
give a setting a value: the loader requires it, a seed supplies it (the
bundled example's catchment, or what a migration could read off an
existing project), or it is `prompt`ed - a credential most projects need
but that must not block a notebook that never touches it, so it is
written live without the loader enforcing it.

Those comments are generated from the same field metadata that
``study.load_study`` validates against, which is the point of doing it this
way rather than keeping a hand-written template beside the code. A
hand-written one drifts - it ends up describing settings the loader would
reject - and nothing catches it, because nobody re-reads a comment block.

Writing lives here rather than in ``study.py`` so that reading a config and
producing one stay separable. The dependency runs one way: this module
imports from ``study``, never the reverse.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import inspect
import os
import textwrap
from dataclasses import fields

from .study import (
    CONFIG_NAME, StudyConfigError, StudySettings, config_path,
)

# Help text and group docstrings are wrapped to this width before the
# '# ' comment marker is added, so a generated line stays comfortably
# under the project's 79-column convention - the one artefact where
# that convention *is* the feature, since this file is read, not run.
_COMMENT_WIDTH = 72

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The bundled example, keyed by dotted setting name. `fire-impacts new`
# seeds a scaffold with these so a fresh project runs end to end before
# the user changes anything.
EXAMPLE_SEED = {
    'catchment.name': 'EgSmallCatchment_7899',
    'catchment.boundary': '../test_data/EgSmallCatchment_7899.shp',
    'catchment.aridity':
        '../test_data/AridityPT_EgSmallCatchment_7899.tif',
    'catchment.subcatchments':
        '../test_data/Subcatchments_EgSmall_7899.shp',
    # The example's subcatchment shapefile has no 'SiteID' column - its
    # identifying attribute is 'Id'. The schema default stays 'SiteID',
    # which is what a real study (and a Source model) wants; the example
    # is the exception and says so here, where a user can see it. Seeding
    # the shapefile without this would register a column that is not
    # there, warn on every read, and label every output by row number.
    'catchment.subcatchment_id_field': 'Id',
    'event.name': '2019_fire',
    'event.fire_start': '2019-01-15',
    'event.fire_end': '2019-03-07',
}

SCAFFOLD_HEADER = f"""\
# {CONFIG_NAME} - everything this study needs you to supply.
#
# Uncommented settings are the ones this file sets; commented-out
# settings are optional and unset, and their comment says what happens
# if you leave them out. Uncommented is not the same as required: the
# file also sets any optional setting it already has an answer for -
# what the bundled example uses, say - and the TERN API key, which the
# soil download needs but no other notebook does.
#
# The values here drive the bundled example catchment, so a new project
# runs end to end before you change anything.
#
# Calibration parameters are NOT set here - they live in
# parameters.json. See the PrepareData notebook."""

# Settings whose value is a filesystem path are written as TOML literal
# strings, so a Windows backslash needs no escaping.
_MISSING_NOTE = (
    '# could not be recovered from the project; please fill this in')


###############################################################################
def _toml_value(value, is_path):
    """
    Render a Python value as TOML.

    Parameters:
    - value: Value to render. Never None - see Notes.
    - is_path: True to use a literal (single-quoted) string.

    Returns:
    - TOML source for the value.
    --------------------------------------------------------------------
    Notes:
    - Raises on None rather than rendering the string "None": a scaffold
      line that looks like a real value is worse than a loud failure, and
      a field whose default is None should be marked `required` or
      `prompt`, or given an explicit `example`, so this is never reached.
    --------------------------------------------------------------------
    """
    if value is None:
        raise ValueError(
            'nothing to render for a None value; give the field an '
            "example, or mark it 'required' or 'prompt'")
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return '[' + ', '.join(_toml_value(v, False) for v in value) + ']'
    if is_path:
        return f"'{value}'"
    return f'"{value}"'


###############################################################################
def _wrapped_comment(text, width=_COMMENT_WIDTH):
    """
    Wrap prose into '#'-prefixed comment lines.

    Parameters:
    - text: Prose to wrap. A blank line within it starts a new paragraph;
      each paragraph is wrapped independently so an intentional break
      (e.g. between the sentences of `secrets.tern_api_key`'s help) is
      kept rather than being merged away.
    - width: Column to wrap prose at, before the '# ' marker is added.

    Returns:
    - List of comment lines, each already prefixed with '# '.
    --------------------------------------------------------------------
    Notes:
    - This is the one artefact where line length is the feature: a
      144-column help line is technically valid TOML but unreadable in
      an editor open to a normal-width pane.
    --------------------------------------------------------------------
    """
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph:
            continue
        lines.extend(f'# {ln}' for ln in textwrap.wrap(paragraph, width))
    return lines


###############################################################################
def _group_comment(group_type):
    """
    Return the group's whole docstring, as a section comment.

    Parameters:
    - group_type: The settings dataclass for the group, e.g.
      ReportingSettings.

    Returns:
    - List of comment lines, or an empty list if the class has no
      docstring. Multiple paragraphs are separated by a bare '#' line.
    --------------------------------------------------------------------
    Notes:
    - `ReportingSettings`' docstring carries exactly the framing this
      file wants ("Reporting choices, not model calibration...") - reading
      it off the class is what stops that explanation from having to be
      written twice and drifting between the two copies. Its second
      paragraph is the one that matters here (it is the only place that
      tells a user these thresholds do not feed the model), so the whole
      docstring is surfaced rather than only its summary line.
    --------------------------------------------------------------------
    """
    doc = inspect.getdoc(group_type)
    if not doc:
        return []

    lines = []
    for i, paragraph in enumerate(doc.split('\n\n')):
        if i:
            lines.append('#')
        collapsed = ' '.join(paragraph.split())
        lines.extend(_wrapped_comment(collapsed))
    return lines


###############################################################################
def scaffold_text(seed=None):
    """
    Generate a commented study.toml from the schema.

    Parameters:
    - seed: Optional dict of dotted setting name to value. A name mapped
      to None is written blank and flagged as needing attention.

    Returns:
    - The file contents, as a string.
    --------------------------------------------------------------------
    Notes:
    - Comments come from each field's `help` metadata - the same metadata
      the loader validates against - so they cannot describe a setting
      that no longer exists.
    - A setting this file has a value for is written uncommented; one it
      does not is commented out, showing the default it will fall back
      to. That convention is what lets a user see what the file actually
      sets by looking at which lines are live. Three things give a
      setting a value: `required` metadata, an entry in `seed`, and
      `prompt` metadata - the last for a setting most projects need but
      that must not block a notebook that never touches it, so it is
      written live without the loader enforcing it.
    - So a live line does NOT mean "you must supply this": with the
      bundled example seeded, `catchment.subcatchments` and
      `catchment.subcatchment_id_field` are live and optional. The
      header says as much; keep the two in step.
    --------------------------------------------------------------------
    """
    seed = seed or {}
    out = [SCAFFOLD_HEADER]

    for group in fields(StudySettings):
        blank = getattr(StudySettings(), group.name)

        out.append(f'\n[{group.name}]')
        group_comment = _group_comment(type(blank))
        out.extend(group_comment)
        if group_comment:
            # Separates "what this section is" from "what this setting
            # is" - without it the two comments read as one run-on block.
            out.append('')

        for entry in fields(blank):
            dotted = f'{group.name}.{entry.name}'
            meta = entry.metadata
            is_path = bool(meta.get('path'))

            out.extend(_wrapped_comment(meta.get('help', '')))

            # A setting is written live - not commented out - if the
            # loader requires it, or if it is merely `prompt`ed for: a
            # credential such as secrets.tern_api_key that most projects
            # need but that must not block a notebook that never uses it,
            # so it cannot be `required` without breaking that notebook.
            write_live = bool(meta.get('required') or meta.get('prompt'))
            seeded = dotted in seed

            if seeded and seed[dotted] is None:
                out.append(
                    f'{entry.name} = {_toml_value("", is_path)}'
                    f'  {_MISSING_NOTE}')
                continue

            if seeded:
                out.append(
                    f'{entry.name} = {_toml_value(seed[dotted], is_path)}')
                continue

            if write_live:
                # No seed, but the loader needs an answer (or the user
                # should be prompted for one): leave it live but empty,
                # so the loader's own error - or a blank credential -
                # tells the user what to fill in.
                out.append(f'{entry.name} = ""')
                continue

            shown = meta.get('example')
            if shown is None:
                shown = _toml_value(entry.default, is_path)
            out.append(f'# {entry.name} = {shown}')

    return '\n'.join(out) + '\n'


###############################################################################
def example_seed():
    """
    Return the example seed with its input paths resolved if possible.

    Returns:
    - Dict of dotted setting name to value, as EXAMPLE_SEED but with
      absolute paths when the repository's test_data can be found.
    --------------------------------------------------------------------
    Notes:
    - The relative paths in EXAMPLE_SEED only resolve for a project created
      one level below the repository root, which is where `examples/` sits.
      A project created anywhere else would get a study.toml whose very
      first load fails on a missing boundary file - so the promise that a
      new project runs before you change anything would hold only in the
      one place nobody actually works.
    - test_data is not package data, so an installed wheel has none. There
      the walk finds nothing and the relative literals stand, which is no
      worse than before.
    --------------------------------------------------------------------
    """
    seed = dict(EXAMPLE_SEED)

    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        here = os.path.dirname(here)
        candidate = os.path.join(here, 'test_data')
        if os.path.isdir(candidate):
            for key, value in list(seed.items()):
                if isinstance(value, str) and 'test_data' in value:
                    seed[key] = os.path.join(
                        candidate, os.path.basename(value))
            break

    return seed


###############################################################################
def write_scaffold(path, seed=None):
    """
    Write a commented study.toml into a project.

    Parameters:
    - path: Project directory.
    - seed: Optional dict of dotted setting name to value.

    Returns:
    - Path to the file written.
    --------------------------------------------------------------------
    Notes:
    - Refuses to overwrite. study.toml is the user's file: unlike the
      notebooks, nothing in this package ever replaces it.
    --------------------------------------------------------------------
    """
    fn = config_path(path)
    if os.path.exists(fn):
        raise StudyConfigError(
            f'{fn} already exists; it is never overwritten. Edit it, or '
            f'move it aside first.'
        )

    os.makedirs(path, exist_ok=True)
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(scaffold_text(seed))

    return fn
