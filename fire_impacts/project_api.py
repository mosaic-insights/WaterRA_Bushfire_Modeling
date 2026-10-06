"""
Jupyter/REPL-friendly equivalents of the `fire-impacts` command-line tool.

`fire-impacts new`, `fire-impacts update` and `fire-impacts status` are
Typer commands in `cli.py` that wrap these same functions for a terminal.
A user who starts and stays in Jupyter (or a plain Python REPL) can call
them directly instead:

    from fire_impacts import new_project, update_project, project_status

    new_project('./my-project')
    update_project('./my-project', dry_run=True)
    project_status('./my-project', verbose=True)

Output is written with `print()` rather than through `logging`, because a
fresh Jupyter kernel has no logging handler configured - a message sent
through `logger.info()` would be silently dropped there even though it is
visible in a terminal (where `fire_impacts.cli` configures the root
logger at import time). `print()` behaves the same in both places.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import os

from dataclasses import dataclass

from .pre import FireImpactsProject
from . import notebooks as nb
from .study import CONFIG_NAME, check_study, config_path
from .study_scaffold import example_seed, write_scaffold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class UpdateResult:
    """
    What `update_project` did.

    Attributes:
    - states: List of NotebookState, one per template notebook.
    - migrated_config: True if this project had no study.toml and one
      was written from what could be recovered off the project.
    - report: The check_study() report, or None. Populated only when a
      study.toml already existed before this call - the same condition
      `fire-impacts update` has always checked it under. A config
      written this same call (migrated_config=True) is not re-checked,
      matching today's behaviour.
    """

    states: list
    migrated_config: bool
    report: dict | None


@dataclass
class StatusResult:
    """
    What `project_status` found.

    Attributes:
    - states: List of NotebookState, one per template notebook.
    - report: The check_study() report.
    """

    states: list
    report: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


###############################################################################
def _summarise(states, dry_run):
    """
    Print a closing summary of what a refresh did.

    Parameters:
    - states: List of NotebookState returned by refresh_notebooks().
    - dry_run: Whether the refresh was only a preview.
    --------------------------------------------------------------------
    """
    counts = {
        'added': [s.name for s in states if s.action == nb.ACTION_INSTALL],
        'updated': [s.name for s in states
                    if s.action in (nb.ACTION_REPLACE, nb.ACTION_BACKUP)],
        'backed up': [s.name for s in states if s.action == nb.ACTION_BACKUP],
        }

    for label, names in counts.items():
        if not names:
            continue
        tense = 'would be' if dry_run else ('was' if len(names) == 1
                                            else 'were')
        print(f'{len(names)} {tense} {label}: {", ".join(names)}')

    if not any(counts.values()):
        print('Every notebook is already up to date.')
        return

    backups = {os.path.dirname(f) for s in states for f in s.backed_up}
    for folder in sorted(backups):
        print(f'Your previous notebooks are in {folder}')

    if not dry_run and any(s.action == nb.ACTION_INSTALL for s in states):
        if 'PrepareData' in [s.name for s in states]:
            print('Start with "PrepareData.ipynb" to prepare data.')


###############################################################################
def _seed_from_project(path):
    """
    Recover what a study.toml can be seeded with from an existing project.

    Parameters:
    - path: Project directory.

    Returns:
    - Dict of dotted setting name to value. A name mapped to None could
      not be recovered and is flagged in the generated file.
    --------------------------------------------------------------------
    Notes:
    - A project created before study.toml existed has real catchment,
      event and ensemble names on disk, so seeding from those beats
      seeding from the bundled example - the file comes out describing
      the study the user is actually doing.
    - Input paths and the API key are not recorded anywhere in a project,
      so they come back None and the file says so.
    - `FireImpactsProject(path, exist_ok=True)` falls back to creating a
      fresh settings.json when the existing one is missing or unreadable.
      This helper only reads, so it checks for settings.json itself and
      takes the empty seed rather than let that fallback write into a
      project it was only asked to inspect.
    --------------------------------------------------------------------
    """
    seed = {
        'catchment.boundary': None,
        'catchment.aridity': None,
        }

    if not os.path.exists(os.path.join(path, 'settings.json')):
        return seed

    try:
        project = FireImpactsProject(path, exist_ok=True)
        catchments = list(project.catchments)
    except Exception as e:                       # noqa: BLE001
        logger.debug('Could not read %s to seed a config: %s', path, e)
        return seed

    if not catchments:
        return seed

    catchment = catchments[0]
    seed['catchment.name'] = catchment

    for dotted, call in (
            ('event.name', lambda: project.events(catchment)),
            ('ensemble.name', lambda: project.ensembles(catchment)),
            ):
        try:
            found = list(call())
        except Exception as e:                   # noqa: BLE001
            logger.debug('Could not list %s: %s', dotted, e)
            continue
        if found:
            seed[dotted] = found[0]

    return seed


# ---------------------------------------------------------------------------
# new_project
# ---------------------------------------------------------------------------


###############################################################################
def new_project(path, notebooks=True):
    """
    Create a new fire-impacts project at the given path.

    Parameters:
    - path: Directory in which to create the project.
    - notebooks: If True, copy template notebooks into the project.

    Returns:
    - Path to the study.toml written.
    --------------------------------------------------------------------
    Notes:
    - Raises FileExistsError, uncaught, if `path` is already a project -
      the same thing FireImpactsProject(path) raises today. There is
      nothing to translate: `fire-impacts new` has never caught this
      either.
    --------------------------------------------------------------------
    """
    print(f'Creating a new project at {path}')
    FireImpactsProject(path)

    if notebooks:
        print('Adding template notebooks...')
        states = nb.refresh_notebooks(path)
        _summarise(states, dry_run=False)

    written = write_scaffold(path, example_seed())
    print(
        f'Wrote {os.path.basename(written)}. This is where you set '
        'the catchment, the fire and the input files for your study '
        '- it is the only file you need to edit.'
        )
    return written


# ---------------------------------------------------------------------------
# update_project
# ---------------------------------------------------------------------------


###############################################################################
def update_project(path, backup=True, only_new=False, dry_run=False):
    """
    Update an existing project to the latest template notebooks.

    Notebooks you have not edited are replaced outright. Notebooks you
    have edited are copied into a dated folder under `notebook_backups/`
    first, so nothing you have written is lost.

    Parameters:
    - path: Directory of the existing project to update.
    - backup: If False, edited notebooks are overwritten with no copy
      kept.
    - only_new: If True, add missing notebooks and change nothing else.
    - dry_run: If True, only report what would happen.

    Returns:
    - UpdateResult.

    Raises:
    - FileNotFoundError if `path` is not an existing project directory.
    --------------------------------------------------------------------
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'No such project directory: {path}. Use "fire-impacts new" '
            'to create one.'
            )

    print(f'Updating notebooks in {path}')
    if not backup and not dry_run:
        # Literal text unchanged from cli.py's logger.warning() call -
        # the "WARNING" marker there came from the logging formatter,
        # not the message, so print() carries no prefix either.
        print(
            'Backups are turned off: edits to these notebooks will be '
            'lost.'
            )

    states = nb.refresh_notebooks(
        path, backup=backup, only_new=only_new, dry_run=dry_run,
        )
    _summarise(states, dry_run=dry_run)

    migrated = False
    report = None

    if not dry_run and not os.path.exists(config_path(path)):
        # A project created before study.toml existed: its refreshed
        # notebooks will call load_study(), so it needs one. Seed it from
        # the project's own state rather than from the bundled example,
        # so the file describes the study actually in progress.
        migrated = True
        seed = _seed_from_project(path)
        write_scaffold(path, seed)
        print(
            f'This project had no {CONFIG_NAME}, so one has been written '
            'from what could be read off the project itself. Open it and '
            'fill in the settings marked "could not be recovered" before '
            'running the notebooks.'
            )
    elif not dry_run:
        # This is the static gap between the file and the full schema -
        # unrelated to anything this run changed - so the wording must
        # not imply otherwise. A config that has left the same 26
        # settings at their defaults since it was written is not news on
        # the hundredth update.
        report = check_study(path)
        if report['unset']:
            print(
                f'Your {CONFIG_NAME} leaves {len(report["unset"])} '
                f'optional setting(s) at their defaults. Run '
                f'"fire-impacts status {path}" to list them.'
                )

    return UpdateResult(
        states=states, migrated_config=migrated, report=report,
        )


# ---------------------------------------------------------------------------
# project_status
# ---------------------------------------------------------------------------


###############################################################################
def project_status(path, verbose=False):
    """
    Report how a project's notebooks compare with the latest templates.

    Parameters:
    - path: Directory of the project to inspect.
    - verbose: If True, name every unset optional setting instead of
      just counting them.

    Returns:
    - StatusResult.

    Raises:
    - FileNotFoundError if `path` is not an existing project directory.
    --------------------------------------------------------------------
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f'No such project directory: {path}')

    descriptions = {
        nb.ACTION_INSTALL: 'not in this project',
        nb.ACTION_CURRENT: 'up to date',
        nb.ACTION_REPLACE: 'out of date (unedited, safe to update)',
        nb.ACTION_BACKUP: 'out of date, and edited here '
                          '(will be backed up)',
        }

    states = nb.plan_update(path)
    for state in states:
        note = descriptions[state.action]
        if state.action == nb.ACTION_BACKUP and not state.recorded:
            # Without a manifest entry there is no way to tell an edit
            # from a template that has simply moved on. Say so, rather
            # than claiming to know.
            note = ('differs from the current template (added before '
                    'edits were tracked; will be backed up)')
        print(f'{state.name:<24} {note}')

    print('')
    report = check_study(path)

    if report['missing_file']:
        print(
            f'{CONFIG_NAME:<24} missing (run "fire-impacts update '
            f'{path}" to create one)'
            )
        return StatusResult(states=states, report=report)

    print(f'{CONFIG_NAME:<24} present')
    for problem in report['unknown']:
        print(f'  ! {problem}')
    for name in report['required_missing']:
        print(f'  ! {name} is required but not set')
    if report['unset']:
        if verbose:
            for name in report['unset']:
                print(f'  - {name} not set (uses the default)')
        else:
            print(
                f'  - {len(report["unset"])} optional setting(s) not set '
                f'(using defaults; use --verbose to list them)'
                )

    return StatusResult(states=states, report=report)
