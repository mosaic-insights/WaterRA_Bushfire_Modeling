"""
Command-line interface for creating and updating fire-impacts projects.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging

import typer

from . import netcheck
from . import project_api

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
    project_api.new_project(path, notebooks=notebooks)


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
    try:
        project_api.update_project(
            path, backup=backup, only_new=only_new, dry_run=dry_run,
            )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
    try:
        project_api.project_status(path, verbose=verbose)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


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
