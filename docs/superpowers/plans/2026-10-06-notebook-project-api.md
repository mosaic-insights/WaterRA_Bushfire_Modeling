# Notebook-friendly project API, and a README quickstart — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Jupyter/REPL users a plain-Python equivalent of `fire-impacts new`/`update`/`status` (`new_project`/`update_project`/`project_status`, importable from `fire_impacts`), and add a short README quickstart that shows the install → create project → edit `study.toml` → open notebook path in one place, with the terminal and Python forms shown side by side.

**Architecture:** A new `fire_impacts/project_api.py` module holds the actual logic, ported unchanged in behaviour from `fire_impacts/cli.py`'s `new`/`update`/`status` commands and their `_summarise`/`_seed_from_project` helpers — `logger.info`/`typer.echo` becomes `print()`, `typer.BadParameter` becomes a plain `FileNotFoundError`, and each function returns a result (a path, or a small dataclass) instead of only printing one. `cli.py` is then cut down to three thin Typer wrappers that call into `project_api.py` and translate `FileNotFoundError` back into `typer.BadParameter` for the CLI's exit-code behaviour. The three functions are re-exported from `fire_impacts/__init__.py`.

**Tech Stack:** Python 3.11+, `typer` (CLI only), `pytest` (`fire_impacts/tests/`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-notebook-project-api-design.md` — read this first; it has the full rationale and was reviewed twice against the current code.

---

## Before you start

Read these files in full — the plan below quotes and moves code from them, and getting the current behaviour wrong anywhere is exactly the mistake this plan exists to avoid:

- `fire_impacts/cli.py` — the code being moved and thinned.
- `fire_impacts/notebooks.py` — `refresh_notebooks`, `plan_update`, `NotebookState`, the `ACTION_*` constants. Not modified, only called.
- `fire_impacts/study.py` — `load_study`, `check_study`, `config_path`, `CONFIG_NAME`. Not modified, only called.
- `fire_impacts/study_scaffold.py` — `write_scaffold`, `example_seed`. Not modified, only called.
- `fire_impacts/tests/test_study_cli.py` — the existing CLI test suite. It must keep passing, **unmodified**, after every task below.

Run the existing test suite once before touching anything, so you have a known-good baseline to compare against:

```
pytest fire_impacts/tests/test_study_cli.py -v
```

Expected: all 10 tests pass (confirmed while writing this plan: `10
passed in 133.52s`). This repo's import chain is slow on first
collection (geospatial dependencies) — give it a couple of minutes if it
seems to hang; it is faster on later runs once imports are cached by the
interpreter's bytecode cache.

---

### Task 1: `project_api.py` — `new_project`

**Files:**
- Create: `fire_impacts/project_api.py`
- Test: `fire_impacts/tests/test_project_api.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `fire_impacts/tests/test_project_api.py`:

```python
"""
`new_project`, `update_project` and `project_status` are the Jupyter/REPL
equivalent of `fire-impacts new`/`update`/`status` - the same logic,
called directly as Python instead of through the CLI. These tests mirror
`test_study_cli.py`'s scenarios one-for-one, calling the functions
instead of invoking `typer`'s CliRunner.
"""

import os

from fire_impacts.project_api import new_project
from fire_impacts.study import CONFIG_NAME


def test_new_writes_a_study_config(tmp_path):
    project = tmp_path / 'proj'
    written = new_project(str(project), notebooks=False)
    assert (project / CONFIG_NAME).exists()
    assert os.path.basename(written) == CONFIG_NAME


def test_new_seeds_the_config_with_the_example(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'EgSmallCatchment_7899' in text
    assert '2019_fire' in text


def test_new_project_prints_where_the_config_was_written(tmp_path, capsys):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    captured = capsys.readouterr()
    assert CONFIG_NAME in captured.out


def test_a_new_project_has_notebooks_that_can_find_their_config(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project))

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        assert (project / f'{name}.py').exists()
    assert (project / CONFIG_NAME).exists()

    from fire_impacts.study import check_study
    report = check_study(str(project))
    assert report['unknown'] == []
    assert report['required_missing'] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: FAIL (or ERROR) — `ModuleNotFoundError: No module named 'fire_impacts.project_api'`.

- [ ] **Step 3: Create `project_api.py` with `new_project`**

Create `fire_impacts/project_api.py`. This step writes the module header,
the `_summarise` helper (ported from `cli.py`'s `_summarise`, `logger.info`
→ `print`), and `new_project` only — `_seed_from_project`,
`update_project`, `project_status` and the two result dataclasses are
added in later tasks, but write the imports and docstring for the whole
module now so later tasks only add to it:

```python
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

from .pre import FireImpactsProject
from . import notebooks as nb
from .study_scaffold import example_seed, write_scaffold

logger = logging.getLogger(__name__)


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/project_api.py fire_impacts/tests/test_project_api.py
git commit -m "Add new_project, a notebook-callable equivalent of 'fire-impacts new'"
```

---

### Task 2: `project_api.py` — `update_project`

**Files:**
- Modify: `fire_impacts/project_api.py`
- Test: `fire_impacts/tests/test_project_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/tests/test_project_api.py` (add `json` and `pytest`
to the imports at the top of the file):

```python
import json

import pytest

from fire_impacts.project_api import new_project, update_project
```

```python
def test_update_leaves_an_existing_config_alone(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    (project / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')

    result = update_project(str(project))

    assert (project / CONFIG_NAME).read_text(encoding='utf-8') == '# mine\n'
    assert result.migrated_config is False


def test_update_creates_a_config_for_a_project_that_predates_it(tmp_path):
    """Such a project has notebooks but no config, and after the update its
    notebooks call load_study()."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    result = update_project(str(project))

    assert (project / CONFIG_NAME).exists()
    assert result.migrated_config is True
    # Not re-checked the same call it was written in - matches
    # cli.update's own behaviour (see project_api.update_project).
    assert result.report is None


def test_update_report_reflects_unset_settings_of_an_existing_config(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)

    result = update_project(str(project))

    assert result.migrated_config is False
    assert result.report is not None
    assert isinstance(result.report['unset'], list)


def test_update_raises_for_a_missing_directory(tmp_path):
    missing = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError, match='fire-impacts new'):
        update_project(str(missing))


def test_update_recovers_names_from_an_existing_catchment(tmp_path):
    """The seeding path that actually matters: a project created before
    study.toml existed, with a real catchment, event and ensemble on
    disk, migrated for the first time."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    (project / 'settings.json').write_text(
        json.dumps({'catchments': ['Avon']}), encoding='utf-8')
    events_dir = project / 'Catchments' / 'Avon' / 'Events' / '2019_fire'
    events_dir.mkdir(parents=True)
    ensembles_dir = (
        project / 'Catchments' / 'Avon' / 'Ensembles' / 'stochastic_v2')
    ensembles_dir.mkdir(parents=True)

    update_project(str(project))

    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'name = "Avon"' in text
    assert 'name = "2019_fire"' in text
    assert 'name = "stochastic_v2"' in text
    assert text.count('could not be recovered from the project') == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: the 5 new tests FAIL/ERROR with `ImportError` (`update_project`
does not exist yet); the 4 tests from Task 1 still PASS.

- [ ] **Step 3: Add `_seed_from_project`, `UpdateResult` and `update_project`**

Add to the imports at the top of `fire_impacts/project_api.py`:

```python
from dataclasses import dataclass

from .study import CONFIG_NAME, check_study, config_path
```

Add a `UpdateResult` dataclass (place it after the module-level `logger`
line, before `_summarise`):

```python
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
```

Add `_seed_from_project` (ported from `cli.py`'s `_seed_from_project`,
unchanged) after `_summarise`:

```python
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
```

Add `update_project` after `new_project`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/project_api.py fire_impacts/tests/test_project_api.py
git commit -m "Add update_project, a notebook-callable equivalent of 'fire-impacts update'"
```

---

### Task 3: `project_api.py` — `project_status`

**Files:**
- Modify: `fire_impacts/project_api.py`
- Test: `fire_impacts/tests/test_project_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/tests/test_project_api.py`:

```python
from fire_impacts.project_api import project_status
```

```python
def test_status_reports_a_typo_in_the_config(tmp_path, capsys):
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    (project / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')

    result = project_status(str(project))
    captured = capsys.readouterr()

    assert 'naem' in captured.out
    assert "Did you mean 'name'" in captured.out
    assert any('naem' in u for u in result.report['unknown'])


def test_status_reports_a_missing_config(tmp_path, capsys):
    """Both branches of the report name the file, so looking only for
    CONFIG_NAME would pass against a report that never set missing_file
    at all. The report has to say which branch it took, and name the
    command that fixes it."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)
    os.remove(project / CONFIG_NAME)

    result = project_status(str(project))
    captured = capsys.readouterr()

    assert CONFIG_NAME in captured.out
    assert 'missing' in captured.out
    assert 'fire-impacts update' in captured.out
    assert result.report['missing_file'] is True


def test_status_does_not_call_a_config_that_is_there_missing(tmp_path, capsys):
    """The other half. Without it, a status that printed 'missing'
    unconditionally would still satisfy the test above."""
    project = tmp_path / 'proj'
    new_project(str(project), notebooks=False)

    project_status(str(project))
    captured = capsys.readouterr()

    assert f'{CONFIG_NAME:<24} present' in captured.out
    assert 'missing' not in captured.out


def test_status_raises_for_a_missing_directory(tmp_path):
    missing = tmp_path / 'does-not-exist'
    with pytest.raises(FileNotFoundError):
        project_status(str(missing))


def test_status_returns_the_notebook_states_too(tmp_path):
    project = tmp_path / 'proj'
    new_project(str(project))

    result = project_status(str(project))

    names = {s.name for s in result.states}
    assert names == {'PrepareData', 'Simulation', 'SimulationEnsemble',
                      'SourceIntegration'}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: the 5 new tests FAIL/ERROR with `ImportError`; the 9 tests from
Tasks 1–2 still PASS.

- [ ] **Step 3: Add `StatusResult` and `project_status`**

Add a `StatusResult` dataclass next to `UpdateResult`:

```python
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
```

Add `project_status` after `update_project`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/project_api.py fire_impacts/tests/test_project_api.py
git commit -m "Add project_status, a notebook-callable equivalent of 'fire-impacts status'"
```

---

### Task 4: Thin `cli.py` down to wrap `project_api`

**Files:**
- Modify: `fire_impacts/cli.py`
- Test: `fire_impacts/tests/test_study_cli.py` (must pass unmodified)

- [ ] **Step 1: Rewrite `cli.py`**

Replace the whole file with:

```python
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
```

This removes `_summarise` and `_seed_from_project` from `cli.py` entirely
(they now live in `project_api.py`), along with the now-unused imports
(`os`, `from .pre import FireImpactsProject`, `from . import notebooks as
nb`, `from .study import CONFIG_NAME, check_study, config_path`, `from
.study_scaffold import example_seed, write_scaffold`). The
`check-network` command is untouched byte-for-byte.

- [ ] **Step 2: Run the existing CLI tests**

Run: `pytest fire_impacts/tests/test_study_cli.py -v`
Expected: PASS, all 10 tests, unmodified from before this plan started
(confirmed as the baseline before Task 1: `10 passed in 133.52s`).
This is the regression check for the whole refactor — if anything here
fails, the bug is almost certainly a mismatch between what moved to
`project_api.py` and what `cli.py` used to do; compare against `git show
HEAD:fire_impacts/cli.py` (the version before this task) rather than
guessing.

- [ ] **Step 3: Run the full project_api test suite too**

Run: `pytest fire_impacts/tests/test_project_api.py -v`
Expected: PASS, all 14 tests (unaffected by this task, but cheap to
confirm).

- [ ] **Step 4: Commit**

```bash
git add fire_impacts/cli.py
git commit -m "Thin cli.py down to wrap project_api"
```

---

### Task 5: Export from the package top level

**Files:**
- Modify: `fire_impacts/__init__.py`
- Test: `fire_impacts/tests/test_project_api.py`

- [ ] **Step 1: Write the failing test**

Append to `fire_impacts/tests/test_project_api.py`:

```python
def test_the_functions_are_importable_from_the_package_top_level():
    import fire_impacts
    assert fire_impacts.new_project is new_project
    from fire_impacts.project_api import update_project, project_status
    assert fire_impacts.update_project is update_project
    assert fire_impacts.project_status is project_status
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest fire_impacts/tests/test_project_api.py -k top_level -v`
Expected: FAIL — `AttributeError: module 'fire_impacts' has no attribute
'new_project'`.

- [ ] **Step 3: Add the export**

Read `fire_impacts/__init__.py` (it is 14 lines; the only line of code is
`from .pre import *`). Change it to:

```python
from .pre import *
from .project_api import new_project, update_project, project_status
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest fire_impacts/tests/test_project_api.py -k top_level -v`
Expected: PASS.

- [ ] **Step 5: Run the whole package's fast test suite**

Run: `pytest fire_impacts/tests/ -v`
Expected: PASS — this directory holds `test_study_cli.py` and
`test_project_api.py`, the two files this plan touches, alongside the
package's other top-level tests (unaffected by this change). It is
faster than the full suite because it skips the slower geospatial tests
that live under `fire_impacts/pre/tests/`, `fire_impacts/sim/tests/` etc.
Task 7 below runs everything.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/__init__.py fire_impacts/tests/test_project_api.py
git commit -m "Export new_project, update_project, project_status from fire_impacts"
```

---

### Task 6: README quickstart

**Files:**
- Modify: `README.md`

No automated test — this is documentation. Verify by reading it back as
a brand-new user would.

- [ ] **Step 1: Read the current README's opening**

Read `README.md` lines 1–100. The quickstart goes immediately after the
three-paragraph introduction (ends "...is being tested on, Australian
conditions.") and before the existing `## Installation` heading.

- [ ] **Step 2: Insert the quickstart section**

Insert this new section (verify the two anchor links resolve to the
existing headings `## Configuring a study` and `### Starting a project,
and keeping its notebooks current` further down the file — GitHub/most
Markdown renderers slugify headings to lower-case, hyphen-joined text,
which is what these links already assume):

```markdown
## Quickstart

The fastest path from nothing installed to a running example, in four
steps. Everything here is explained in more depth further down — this
section exists so you don't have to read the rest of the document
first.

**1. Install.** From the repository folder:

```
pip install -e .
```

See [Installation](#installation) below if you don't yet have a
scientific Python environment to install into.

**2. Create a project.** This creates a project folder, copies in the
starter notebooks, and writes a `study.toml` pre-filled with a bundled
example catchment — so the project runs before you change anything.
From a terminal:

```
fire-impacts new ./my-project
```

Prefer to stay in Jupyter or a Python REPL? Every `fire-impacts`
command has a plain-Python equivalent:

```python
from fire_impacts import new_project
new_project('./my-project')
```

**3. Point it at your own study.** Open `my-project/study.toml` and
edit the settings it sets live (uncommented) — your catchment boundary,
its name, and your fire's dates. See [Configuring a
study](#configuring-a-study) below for what each setting means and
which ones are required.

**4. Run it.** Open `my-project/PrepareData.ipynb` in Jupyter and run it
top to bottom. See [Starting a project, and keeping its notebooks
current](#starting-a-project-and-keeping-its-notebooks-current) below
for the other three notebooks, and for pulling in newer templates later
with `fire-impacts update` / `update_project(...)`.
```

- [ ] **Step 3: Read it back**

Read the edited `README.md` from the top through the end of the new
section, plus the two sections it links to, and confirm:
- The two links' anchor text matches the actual heading slugs.
- Nothing in the new section restates content from "Configuring a
  study" or "Starting a project" rather than linking to it (the spec's
  explicit constraint — see "The shape of the solution, 4. README
  quickstart" in the design doc).
- The code fences render as intended (a shell block, then a separate
  Python block) if you have a Markdown previewer available.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add a README quickstart, with the Python/Jupyter equivalent of each CLI step"
```

---

### Task 7: Final full-suite check

**Files:** none (verification only)

- [ ] **Step 1: Run the fast test suite**

Run: `pytest fire_impacts/tests/ -v`
Expected: PASS.

- [ ] **Step 2: Run the full project test suite**

Run: `pytest -v`
Expected: PASS (network- and Veneer-marked tests are deselected by
`addopts` in `pyproject.toml`; this may take a few minutes on first
import).

- [ ] **Step 3: Confirm nothing else references the moved helpers**

Run: `grep -rn "_summarise\|_seed_from_project" fire_impacts/ --include=*.py`
Expected: both names appear only in `fire_impacts/project_api.py` and
`fire_impacts/tests/test_project_api.py` (if referenced there), never in
`fire_impacts/cli.py`.

- [ ] **Step 4: Update the design spec's status line**

Edit `docs/superpowers/specs/2026-10-06-notebook-project-api-design.md`
line 3 from:

```
**Status:** design agreed, not yet implemented
```

to:

```
**Status:** implemented
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-10-06-notebook-project-api-design.md
git commit -m "Mark the notebook-project-api design as implemented"
```
