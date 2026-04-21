# my_package/cli.py
import glob
import shutil
import typer
from .pre import FireImpactsProject
import os
import jupytext
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fire-impacts-cli')

app = typer.Typer()

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')


def _template_notebooks():
    """Discover all jupytext-percent template scripts in templates/."""
    paths = sorted(glob.glob(os.path.join(TEMPLATES_DIR, '*.py')))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


@app.command()
def new(path: str, notebooks: bool = True):
    """Creates a new project at the specified path."""
    logger.info(f"Creating a new project at %s", path)
    proj = FireImpactsProject(path)

    if notebooks:
        copy_notebooks_to(path)

def copy_notebooks_to(path:str, overwrite:bool=False):
    logger.info("Adding template notebooks...")
    notebooks = _template_notebooks()
    for nb in notebooks:
        src = os.path.join(TEMPLATES_DIR, f'{nb}.py')
        dest = os.path.join(path, f'{nb}.py')
        if os.path.exists(dest) and not overwrite:
            logger.warning('Notebook script %s already exists, skipping.', dest)
        else:
            shutil.copy(src, dest)
            logger.info('Copied template: %s', dest)
        nb_dest = os.path.join(path, f'{nb}.ipynb')
        if os.path.exists(nb_dest) and not overwrite:
            logger.warning('Notebook %s already exists, skipping.', nb_dest)
        else:
            jupytext.write(jupytext.read(dest,fmt='py:percent'),
                            nb_dest,
                            fmt='ipynb')
    logger.info("Template notebooks added.")
    if 'PrepareData' in notebooks:
        logger.info('Start with "PrepareData.ipynb" to prepare data.')

@app.command()
def update(path:str,overwrite:bool=False):
    """Updates an existing project to the latest structure."""
    logger.info(f"Updating project at %s", path)
    copy_notebooks_to(path,overwrite=overwrite)
    logger.info("Project updated.")

if __name__ == "__main__":
    app()

