"""
Command-line interface for creating and updating fire-impacts projects.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import glob
import logging
import os
import shutil

import jupytext
import typer

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
# App and constants
# ---------------------------------------------------------------------------

app = typer.Typer()

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'templates'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


###############################################################################
def _template_notebooks():
    """Discover all jupytext-percent template scripts in templates/."""
    paths = sorted(glob.glob(os.path.join(TEMPLATES_DIR, '*.py')))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


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
        copy_notebooks_to(path)


###############################################################################
def copy_notebooks_to(path: str, overwrite: bool = False):
    """
    Copy template notebooks and generate .ipynb files in a project.

    Parameters:
    - path: Destination directory for the notebook files.
    - overwrite: If True, replace existing notebooks; if False, skip
      them.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    logger.info("Adding template notebooks...")
    notebooks = _template_notebooks()

    for nb in notebooks:
        src = os.path.join(TEMPLATES_DIR, f'{nb}.py')
        dest = os.path.join(path, f'{nb}.py')

        # Copy the .py script, skipping if it already exists and
        # overwrite is not set:
        if os.path.exists(dest) and not overwrite:
            logger.warning(
                'Notebook script %s already exists, skipping.', dest
            )
        else:
            shutil.copy(src, dest)
            logger.info('Copied template: %s', dest)

        # Convert the .py script to a .ipynb notebook:
        nb_dest = os.path.join(path, f'{nb}.ipynb')
        if os.path.exists(nb_dest) and not overwrite:
            logger.warning(
                'Notebook %s already exists, skipping.', nb_dest
            )
        else:
            jupytext.write(
                jupytext.read(dest, fmt='py:percent'),
                nb_dest,
                fmt='ipynb',
            )

    logger.info("Template notebooks added.")
    if 'PrepareData' in notebooks:
        logger.info(
            'Start with "PrepareData.ipynb" to prepare data.'
        )


###############################################################################
@app.command()
def update(path: str, overwrite: bool = False):
    """
    Update an existing project to the latest notebook structure.

    Parameters:
    - path: Directory of the existing project to update.
    - overwrite: If True, replace existing notebooks; if False, skip
      them.
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    logger.info("Updating project at %s", path)
    copy_notebooks_to(path, overwrite=overwrite)
    logger.info("Project updated.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
