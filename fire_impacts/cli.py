"""
Command-line interface for creating and updating fire-impacts projects.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import os

import typer

from . import notebooks as nb
from .pre import FireImpactsProject

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('fire-impacts-cli')

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


###############################################################################
def _summarise(states, dry_run):
    """
    Log a closing summary of what a refresh did.

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
        logger.info(
            '%d %s %s: %s', len(names), tense, label, ', '.join(names)
            )

    if not any(counts.values()):
        logger.info('Every notebook is already up to date.')
        return

    # Point at the backups, since that is where a user goes to recover
    # work the refresh moved aside:
    backups = {os.path.dirname(f) for s in states for f in s.backed_up}
    for folder in sorted(backups):
        logger.info('Your previous notebooks are in %s', folder)

    if not dry_run and any(s.action == nb.ACTION_INSTALL for s in states):
        if 'PrepareData' in [s.name for s in states]:
            logger.info('Start with "PrepareData.ipynb" to prepare data.')


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


###############################################################################
@app.command()
def new(path: str, notebooks: bool = True):
    """
    Create a new fire-impacts project at the given path.

    Parameters:
    - path: Directory in which to create the project.
    - notebooks: If True, copy template notebooks into the project.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    logger.info("Creating a new project at %s", path)
    FireImpactsProject(path)

    if notebooks:
        logger.info('Adding template notebooks...')
        states = nb.refresh_notebooks(path)
        _summarise(states, dry_run=False)


###############################################################################
@app.command()
def update(
    path: str,
    backup: bool = typer.Option(
        True,
        help='Keep a dated copy of any notebook you have edited before '
             'replacing it.',
        ),
    only_new: bool = typer.Option(
        False,
        help='Only add notebooks the project does not have yet; leave '
             'existing ones alone.',
        ),
    dry_run: bool = typer.Option(
        False,
        help='Report what would change without touching anything.',
        ),
    ):
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
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    if not os.path.isdir(path):
        raise typer.BadParameter(
            f'No such project directory: {path}. Use "fire-impacts new" '
            'to create one.'
            )

    logger.info('Updating notebooks in %s', path)
    if not backup and not dry_run:
        logger.warning(
            'Backups are turned off: edits to these notebooks will be '
            'lost.'
            )

    states = nb.refresh_notebooks(
        path, backup=backup, only_new=only_new, dry_run=dry_run,
        )
    _summarise(states, dry_run=dry_run)


###############################################################################
@app.command()
def status(path: str):
    """
    Report how a project's notebooks compare with the latest templates.

    Parameters:
    - path: Directory of the project to inspect.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    if not os.path.isdir(path):
        raise typer.BadParameter(f'No such project directory: {path}')

    descriptions = {
        nb.ACTION_INSTALL: 'not in this project',
        nb.ACTION_CURRENT: 'up to date',
        nb.ACTION_REPLACE: 'out of date (unedited, safe to update)',
        nb.ACTION_BACKUP: 'out of date, and edited here '
                          '(will be backed up)',
        }

    for state in nb.plan_update(path):
        note = descriptions[state.action]
        if state.action == nb.ACTION_BACKUP and not state.recorded:
            # Without a manifest entry there is no way to tell an edit
            # from a template that has simply moved on. Say so, rather
            # than claiming to know.
            note = ('differs from the current template (added before '
                    'edits were tracked; will be backed up)')
        typer.echo(f'{state.name:<24} {note}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
