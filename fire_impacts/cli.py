"""
Command-line interface for creating and updating fire-impacts projects.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import os

import typer

from . import netcheck
from . import notebooks as nb
from .pre import FireImpactsProject
from .study import CONFIG_NAME, check_study, config_path
from .study_scaffold import example_seed, write_scaffold

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

    # write_scaffold refuses to overwrite an existing study.toml, but
    # that can never happen here: FireImpactsProject(path) above already
    # raised FileExistsError if this path was an existing project.
    written = write_scaffold(path, example_seed())
    # Echoed rather than logged: this is the one line every new user
    # needs to see, and it must show up whether or not logging is
    # configured to print INFO messages.
    typer.echo(
        f'Wrote {os.path.basename(written)}. This is where you set '
        'the catchment, the fire and the input files for your study '
        '- it is the only file you need to edit.'
        )


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

    if not dry_run and not os.path.exists(config_path(path)):
        # A project created before study.toml existed: its refreshed
        # notebooks will call load_study(), so it needs one. Seed it from
        # the project's own state rather than from the bundled example,
        # so the file describes the study actually in progress.
        seed = _seed_from_project(path)
        write_scaffold(path, seed)
        # Echoed rather than logged, for the same reason as in new(): a
        # migrated project needs this seen, not just recorded.
        typer.echo(
            f'This project had no {CONFIG_NAME}, so one has been written '
            'from what could be read off the project itself. Open it and '
            'fill in the settings marked "could not be recovered" before '
            'running the notebooks.'
            )
    elif not dry_run:
        # This is the static gap between the file and the full schema -
        # unrelated to anything this run changed - so the wording must
        # not imply otherwise. Point at "status" rather than enumerate:
        # a config that has left the same 26 settings at their defaults
        # since it was written is not news on the hundredth update.
        report = check_study(path)
        if report['unset']:
            logger.info(
                'Your %s leaves %d optional setting(s) at their '
                'defaults. Run "fire-impacts status %s" to list them.',
                CONFIG_NAME, len(report['unset']), path
                )


###############################################################################
@app.command()
def status(
    path: str,
    verbose: bool = typer.Option(
        False, '--verbose', '-v',
        help='List optional settings left at their defaults, not just '
             'how many there are.',
        ),
    ):
    """
    Report how a project's notebooks compare with the latest templates.

    Parameters:
    - path: Directory of the project to inspect.
    - verbose: If True, name every unset optional setting instead of
      just counting them.
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

    typer.echo('')
    report = check_study(path)

    if report['missing_file']:
        typer.echo(
            f'{CONFIG_NAME:<24} missing (run "fire-impacts update '
            f'{path}" to create one)'
            )
        return

    typer.echo(f'{CONFIG_NAME:<24} present')
    # Real problems always print in full; only the "unset" list - which
    # is information, not a fault - collapses behind --verbose. Left
    # spelled out by default, a healthy project's report is nothing but
    # this list, and a user has to read every line to confirm that.
    for problem in report['unknown']:
        typer.echo(f'  ! {problem}')
    for name in report['required_missing']:
        typer.echo(f'  ! {name} is required but not set')
    if report['unset']:
        if verbose:
            for name in report['unset']:
                typer.echo(f'  - {name} not set (uses the default)')
        else:
            typer.echo(
                f'  - {len(report["unset"])} optional setting(s) not set '
                f'(using defaults; use --verbose to list them)'
                )


###############################################################################
@app.command('check-network')
def check_network(timeout: float = netcheck.DEFAULT_TIMEOUT):
    """
    Check that the remote data services are reachable, and diagnose TLS.

    Probes one endpoint per host through each of the three certificate
    trust stores this package uses, then prints the fix that matches
    whichever of them failed.  Intended for corporate networks that
    intercept TLS, where a download fails with
    CERTIFICATE_VERIFY_FAILED and the traceback says nothing about
    which part of the stack needs configuring.

    Parameters:
    - timeout: per-probe timeout in seconds.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    reasons = {
        netcheck.Outcome.OK: '',
        netcheck.Outcome.TLS: 'certificate not trusted',
        netcheck.Outcome.CONNECT: 'could not connect',
        netcheck.Outcome.TIMEOUT: 'timed out',
        netcheck.Outcome.HTTP: 'service returned an error',
        netcheck.Outcome.OTHER: 'failed',
        }

    typer.echo(
        'Checking the services fire-impacts downloads from.\n'
        'Each line shows the trust store used, because they are '
        'configured separately.\n'
        )

    results = netcheck.run_checks(timeout=timeout)

    for result in results:
        label = 'ok  ' if result.ok else 'FAIL'
        typer.echo(
            f'{label}  {result.probe.stack.value:<9}'
            f'{result.probe.host}'
            )
        typer.echo(f'      {"":<9}{result.probe.description}')
        if not result.ok:
            typer.echo(f'      {"":<9}-> {reasons[result.outcome]}')
            if result.detail:
                typer.echo(f'      {"":<9}   {result.detail}')

    paragraphs = netcheck.advice(results)
    if not paragraphs:
        typer.echo('\nAll services reachable; no action needed.')
        return

    for paragraph in paragraphs:
        typer.echo('\n' + paragraph)

    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
