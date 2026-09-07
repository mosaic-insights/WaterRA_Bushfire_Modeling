"""
Installing and refreshing the template notebooks in a user's project.

A project gets its notebooks by copying the jupytext ``.py`` templates
shipped with the library and generating a paired ``.ipynb`` from each.
Once copied, the notebooks belong to the user: they are the working
copy, and they get edited.

Refreshing them therefore has to tell two things apart:

- the notebook is a **pristine copy** of a template — safe to replace
  outright, because nothing would be lost, and
- the notebook has been **edited** — replaceable only after the current
  version has been put somewhere safe.

To answer that without guessing, the fingerprints of the files written
at install time are recorded in a small manifest in the project
(``.fire_impacts_notebooks.json``). A file whose fingerprint still
matches its record has not been touched since it was installed.

Fingerprints deliberately ignore notebook *outputs*, execution counts
and kernel metadata, so that merely running a notebook does not make it
look edited. Only the cell contents count.

Projects created before the manifest existed have no record to compare
against. There, the fallback is to compare against the current
template: identical means pristine, anything else is treated as edited
and is backed up. That errs towards keeping a spare copy, which is the
harmless direction to err in.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import glob
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime

import jupytext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Record of what was installed, and what it looked like at the time.
MANIFEST_NAME = '.fire_impacts_notebooks.json'
MANIFEST_VERSION = 1

# Backups go into one timestamped folder per refresh, so a single
# `fire-impacts update` leaves a single, dated snapshot behind.
BACKUP_FOLDER_NAME = 'notebook_backups'
BACKUP_STAMP_FORMAT = '%Y%m%d-%H%M%S'

# Actions reported for each notebook by plan_update():
ACTION_INSTALL = 'install'      # not present in the project yet
ACTION_CURRENT = 'current'      # already matches the latest template
ACTION_REPLACE = 'replace'      # pristine but out of date; no backup needed
ACTION_BACKUP = 'backup'        # edited; back up before replacing
ACTION_SKIP = 'skip'            # left alone at the caller's request


# ---------------------------------------------------------------------------
# Locating the shipped templates
# ---------------------------------------------------------------------------


###############################################################################
def template_dir():
    """
    Return the directory holding the shipped template notebooks.

    Returns:
    - Path to the templates directory.
    --------------------------------------------------------------------
    Notes:
    - Inside the package, not beside it: the templates used to live in
      the repository's top-level `templates/`, which exists in a source
      checkout but is site-packages/ in an installed wheel - so every
      template was missing once installed, and `fire-impacts new` had
      nothing to copy. They ship as package data now (see
      `[tool.setuptools.package-data]` in pyproject.toml).
    --------------------------------------------------------------------
    """
    directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'templates'
        )
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f'Could not find the template notebooks in {directory}. '
            'This copy of fire_impacts looks incomplete - try '
            'reinstalling it.'
            )

    return directory


###############################################################################
def template_names():
    """
    Return the names of the available template notebooks.

    Returns:
    - Sorted list of template names, without file extensions.
    --------------------------------------------------------------------
    """
    paths = sorted(glob.glob(os.path.join(template_dir(), '*.py')))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


###############################################################################
def template_path(name):
    """
    Return the path to a template's jupytext script.

    Parameters:
    - name: Template name, without extension.

    Returns:
    - Full path to the template `.py` file.
    --------------------------------------------------------------------
    """
    return os.path.join(template_dir(), f'{name}.py')


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


###############################################################################
def _digest(text):
    """
    Return a stable hash of a block of text.

    Parameters:
    - text: Text to hash.

    Returns:
    - Hex SHA-256 digest.
    --------------------------------------------------------------------
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


###############################################################################
def _normalise(text):
    """
    Normalise text so that only meaningful differences survive.

    Parameters:
    - text: Text to normalise.

    Returns:
    - Text with uniform line endings, no trailing whitespace on any
      line, and no leading or trailing blank lines.
    --------------------------------------------------------------------
    Notes:
    - Line endings differ between platforms, and editors add or remove
      a trailing newline freely. Neither is an edit worth backing up
      for.
    --------------------------------------------------------------------
    """
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    return '\n'.join(line.rstrip() for line in lines).strip('\n')


###############################################################################
def _fingerprint_script_text(text):
    """
    Fingerprint the contents of a jupytext `.py` script.

    Parameters:
    - text: Script contents.

    Returns:
    - Hex digest of the normalised text.
    --------------------------------------------------------------------
    """
    return _digest(_normalise(text))


###############################################################################
def _fingerprint_notebook_json(notebook):
    """
    Fingerprint the cell contents of a parsed notebook.

    Parameters:
    - notebook: Notebook as a dict (parsed `.ipynb` JSON).

    Returns:
    - Hex digest covering each cell's type and source, in order.
    --------------------------------------------------------------------
    Notes:
    - Outputs, execution counts and metadata are deliberately excluded,
      so running a notebook does not make it look edited.
    --------------------------------------------------------------------
    """
    parts = []
    for cell in notebook.get('cells', []):
        source = cell.get('source', '')
        # nbformat stores source either as one string or as a list of
        # lines, depending on who wrote the file:
        if isinstance(source, list):
            source = ''.join(source)
        parts.append(cell.get('cell_type', 'code'))
        parts.append(_normalise(source))

    return _digest('\n'.join(parts))


###############################################################################
def fingerprint_file(path):
    """
    Fingerprint a notebook script or notebook on disk.

    Parameters:
    - path: Path to a `.py` script or `.ipynb` notebook.

    Returns:
    - Hex digest, or None if the file does not exist.
    --------------------------------------------------------------------
    Notes:
    - A notebook that cannot be parsed as JSON is hashed as raw text
      instead. That will not match any recorded fingerprint, so a
      damaged notebook is treated as edited and gets backed up.
    --------------------------------------------------------------------
    """
    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    if path.endswith('.ipynb'):
        try:
            return _fingerprint_notebook_json(json.loads(text))
        except (ValueError, AttributeError):
            logger.warning(
                'Could not parse %s as a notebook; treating its raw '
                'contents as the fingerprint.', path
                )

    return _fingerprint_script_text(text)


###############################################################################
def template_fingerprints(name):
    """
    Fingerprint a template as it would be once installed.

    Parameters:
    - name: Template name, without extension.

    Returns:
    - Dict with 'py' and 'ipynb' fingerprints.
    --------------------------------------------------------------------
    Notes:
    - The `.ipynb` is generated in memory from the template script, the
      same way install_notebook() generates it on disk, so the two
      fingerprints agree.
    --------------------------------------------------------------------
    """
    src = template_path(name)
    with open(src, 'r', encoding='utf-8') as f:
        script = f.read()

    notebook = jupytext.reads(script, fmt='py:percent')
    generated = json.loads(jupytext.writes(notebook, fmt='ipynb'))

    return {
        'py': _fingerprint_script_text(script),
        'ipynb': _fingerprint_notebook_json(generated),
        }


# ---------------------------------------------------------------------------
# The manifest of what was installed
# ---------------------------------------------------------------------------


###############################################################################
def manifest_path(path):
    """
    Return the path to a project's notebook manifest.

    Parameters:
    - path: Project directory.

    Returns:
    - Full path to the manifest file, which need not exist.
    --------------------------------------------------------------------
    """
    return os.path.join(path, MANIFEST_NAME)


###############################################################################
def read_manifest(path):
    """
    Read the record of notebooks installed into a project.

    Parameters:
    - path: Project directory.

    Returns:
    - Dict mapping notebook name to its recorded fingerprints. Empty if
      the project has no manifest, or if it cannot be read.
    --------------------------------------------------------------------
    """
    fn = manifest_path(path)
    if not os.path.exists(fn):
        return {}

    try:
        with open(fn, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except (ValueError, OSError) as e:
        # A missing or unreadable manifest is not fatal: it just means
        # falling back to comparing against the current template.
        logger.warning('Could not read %s (%s); ignoring it.', fn, e)
        return {}

    return manifest.get('notebooks', {})


###############################################################################
def _write_manifest(path, notebooks):
    """
    Write the record of notebooks installed into a project.

    Parameters:
    - path: Project directory.
    - notebooks: Dict mapping notebook name to recorded fingerprints.
    --------------------------------------------------------------------
    """
    fn = manifest_path(path)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(
            {'version': MANIFEST_VERSION, 'notebooks': notebooks},
            f,
            indent=2,
            sort_keys=True,
            )
        f.write('\n')


###############################################################################
def _record_installed(path, name, fingerprints, when):
    """
    Add one notebook to a project's manifest.

    Parameters:
    - path: Project directory.
    - name: Notebook name, without extension.
    - fingerprints: Dict with 'py' and 'ipynb' fingerprints.
    - when: Timestamp string to record against the installation.
    --------------------------------------------------------------------
    """
    notebooks = read_manifest(path)
    notebooks[name] = dict(fingerprints, installed=when)
    _write_manifest(path, notebooks)


# ---------------------------------------------------------------------------
# Working out what needs doing
# ---------------------------------------------------------------------------


@dataclass
class NotebookState:
    """
    What one template looks like in a project, and what to do about it.
    """

    name: str
    action: str
    edited: bool
    present: bool
    py_path: str
    ipynb_path: str
    recorded: bool
    backed_up: list = field(default_factory=list)

    ###########################################################################
    @property
    def existing_files(self):
        """
        Return the notebook files that are actually present on disk.

        Returns:
        - List of existing paths, `.py` first.
        --------------------------------------------------------------------
        """
        return [p for p in (self.py_path, self.ipynb_path)
                if os.path.exists(p)]


###############################################################################
def _is_edited(py_path, ipynb_path, record, template):
    """
    Decide whether the user has changed a notebook since it was installed.

    Parameters:
    - py_path: Path to the project's copy of the script.
    - ipynb_path: Path to the project's copy of the notebook.
    - record: The manifest entry for this notebook, or None.
    - template: Fingerprints of the current template.

    Returns:
    - True if either file differs from what it is compared against.
    --------------------------------------------------------------------
    Notes:
    - With a manifest entry, the comparison is against the files as
      installed, which answers the question exactly. Without one, the
      comparison is against the current template, which cannot separate
      a user's edit from a change to the template — so anything that
      differs is called edited.
    --------------------------------------------------------------------
    """
    baseline = record if record else template

    for path, key in ((py_path, 'py'), (ipynb_path, 'ipynb')):
        if not os.path.exists(path):
            continue
        expected = baseline.get(key)
        if expected is None or fingerprint_file(path) != expected:
            return True

    return False


###############################################################################
def plan_update(path, names=None):
    """
    Work out what refreshing a project's notebooks would do.

    Parameters:
    - path: Project directory.
    - names: Optional list of template names; defaults to all of them.

    Returns:
    - List of NotebookState, one per template, in template order.
    --------------------------------------------------------------------
    """
    manifest = read_manifest(path)
    states = []

    for name in (names if names is not None else template_names()):
        py_path = os.path.join(path, f'{name}.py')
        ipynb_path = os.path.join(path, f'{name}.ipynb')
        present = os.path.exists(py_path) or os.path.exists(ipynb_path)

        template = template_fingerprints(name)
        record = manifest.get(name)
        edited = present and _is_edited(
            py_path, ipynb_path, record, template
            )

        if not present:
            action = ACTION_INSTALL
        elif edited:
            action = ACTION_BACKUP
        elif (fingerprint_file(py_path) == template['py']
                and fingerprint_file(ipynb_path) == template['ipynb']):
            # Unedited, and both files already match the latest
            # template: there is nothing to copy.
            action = ACTION_CURRENT
        else:
            action = ACTION_REPLACE

        states.append(NotebookState(
            name=name,
            action=action,
            edited=edited,
            present=present,
            py_path=py_path,
            ipynb_path=ipynb_path,
            recorded=record is not None,
            ))

    return states


# ---------------------------------------------------------------------------
# Doing it
# ---------------------------------------------------------------------------


###############################################################################
def backup_dir(path, stamp=None):
    """
    Return the folder that a backup taken now would be written to.

    Parameters:
    - path: Project directory.
    - stamp: Optional timestamp string; defaults to the current time.

    Returns:
    - Path to a timestamped folder under the project's backup folder.
      The folder is not created.
    --------------------------------------------------------------------
    """
    if stamp is None:
        stamp = datetime.now().strftime(BACKUP_STAMP_FORMAT)
    return os.path.join(path, BACKUP_FOLDER_NAME, stamp)


###############################################################################
def back_up_notebook(state, destination):
    """
    Copy a notebook's files into a backup folder.

    Parameters:
    - state: NotebookState for the notebook to back up.
    - destination: Folder to copy the files into; created if needed.

    Returns:
    - List of paths written.
    --------------------------------------------------------------------
    """
    files = state.existing_files
    if not files:
        return []

    os.makedirs(destination, exist_ok=True)
    written = []
    for src in files:
        dest = os.path.join(destination, os.path.basename(src))
        # copy2 rather than copy, so the backup keeps the modification
        # time the user last edited the notebook at:
        shutil.copy2(src, dest)
        written.append(dest)

    return written


###############################################################################
def install_notebook(path, name):
    """
    Copy one template into a project, replacing what is there.

    Parameters:
    - path: Project directory.
    - name: Template name, without extension.

    Returns:
    - Dict with the 'py' and 'ipynb' fingerprints of the files written.
    --------------------------------------------------------------------
    """
    src = template_path(name)
    py_path = os.path.join(path, f'{name}.py')
    ipynb_path = os.path.join(path, f'{name}.ipynb')

    shutil.copy(src, py_path)
    jupytext.write(
        jupytext.read(py_path, fmt='py:percent'), ipynb_path, fmt='ipynb'
        )

    return {
        'py': fingerprint_file(py_path),
        'ipynb': fingerprint_file(ipynb_path),
        }


###############################################################################
def refresh_notebooks(
    path,
    names=None,
    backup=True,
    only_new=False,
    dry_run=False,
    ):
    """
    Bring a project's notebooks up to date with the shipped templates.

    Parameters:
    - path: Project directory. Created if it does not exist.
    - names: Optional list of template names; defaults to all of them.
    - backup: If True, copy an edited notebook into a timestamped
      backup folder before replacing it.
    - only_new: If True, install templates the project does not have and
      leave every existing notebook untouched.
    - dry_run: If True, report what would happen and change nothing.

    Returns:
    - List of NotebookState describing what was done (or would be), with
      `action` set to what actually happened and `backed_up` listing any
      files copied aside.
    --------------------------------------------------------------------
    Notes:
    - Backups from one call all land in the same timestamped folder.
    --------------------------------------------------------------------
    """
    if not dry_run:
        os.makedirs(path, exist_ok=True)

    states = plan_update(path, names=names)
    when = datetime.now()
    destination = backup_dir(path, when.strftime(BACKUP_STAMP_FORMAT))

    for state in states:
        # Nothing to do: the project already has this template, exactly
        # as it is shipped today.
        if state.action == ACTION_CURRENT:
            logger.info('%s is already up to date.', state.name)
            continue

        # Asked for new notebooks only, and this one is not new.
        if only_new and state.action != ACTION_INSTALL:
            logger.info(
                '%s already exists; leaving it alone.', state.name
                )
            state.action = ACTION_SKIP
            continue

        if state.action == ACTION_BACKUP and not backup:
            # Explicitly asked for, but worth saying out loud: this is
            # the one path that discards the user's edits.
            logger.warning(
                '%s has been edited and backups are turned off; '
                'those changes will be lost.', state.name
                )

        if dry_run:
            logger.info('Would %s.', _describe(state, backup, destination))
            continue

        if state.action == ACTION_BACKUP and backup:
            state.backed_up = back_up_notebook(state, destination)
            logger.info('Backed up %s to %s.', state.name, destination)

        install_notebook(path, state.name)
        _record_installed(
            path,
            state.name,
            template_fingerprints(state.name),
            when.isoformat(timespec='seconds'),
            )
        logger.info(
            '%s %s.',
            'Added' if state.action == ACTION_INSTALL else 'Updated',
            state.name,
            )

    return states


###############################################################################
def _describe(state, backup, destination):
    """
    Describe, in words, what refreshing one notebook would do.

    Parameters:
    - state: NotebookState to describe.
    - backup: Whether backups are being taken.
    - destination: Folder backups would be written to.

    Returns:
    - Phrase completing a sentence beginning "Would ...".
    --------------------------------------------------------------------
    """
    if state.action == ACTION_INSTALL:
        return f'add {state.name}'
    if state.action == ACTION_BACKUP and backup:
        return (
            f'back up {state.name} to {destination}, then update it'
            )
    if state.action == ACTION_BACKUP:
        return f'overwrite {state.name}, discarding its edits'
    return f'update {state.name} (unedited since it was added)'
