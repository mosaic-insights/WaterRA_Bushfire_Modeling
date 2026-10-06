# A Jupyter/REPL-friendly project API, and a quickstart

**Status:** implemented
**Date:** 2026-10-06

## The problem

`fire-impacts new`, `fire-impacts update` and `fire-impacts status` are the
only way to create a project, pull in newer template notebooks, or check a
project's `study.toml` against the current schema. They are Typer commands in
`fire_impacts/cli.py`, invoked from a terminal.

Some users are comfortable in a terminal; others start and stay in Jupyter
(or a plain Python REPL) and would rather not leave it just to run three
commands whose logic is already plain Python underneath (`notebooks.py`'s
`refresh_notebooks`, `study_scaffold.py`'s `write_scaffold`/`example_seed`,
`study.py`'s `check_study`). `cli.py` currently calls those functions and
glues the results into `typer.echo` / `logger.info` output and
`typer.BadParameter` errors — there is no single supported entry point a
notebook cell can call and get the same report.

Separately, the README documents installation and getting-started end to
end, but the path a brand-new user actually needs — install, create a
project, edit `study.toml`, open `PrepareData.ipynb` — is spread across
~600 lines interleaved with reference material (the data model, calibration
parameters, the low-level interface). There is no single short section that
walks straight through it.

## The shape of the solution

### 1. `fire_impacts/project_api.py`

Three functions, each a plain-Python equivalent of one CLI command, callable
from a notebook cell with no logging configuration required:

```python
def new_project(path, notebooks=True): ...
def update_project(path, backup=True, only_new=False, dry_run=False): ...
def project_status(path, verbose=False): ...
```

- User-facing output is written with `print()`, not `logging` — a fresh
  Jupyter kernel has no handler configured, so anything written through
  `logger.info()` is invisible there even though it shows up in a terminal
  (where `cli.py` calls `logging.basicConfig()` at import time). `print()`
  behaves the same in both places.
- Each function also **returns** its result rather than only printing it,
  so the last expression in a notebook cell is useful on its own:
  - `new_project` returns the path to the `study.toml` written.
  - `update_project` returns a small result object (or dict) with: the
    list of `NotebookState` from `refresh_notebooks` (what it returns
    today), whether a `study.toml` was migrated/written for a
    pre-`study.toml` project, and — matching `cli.update`'s own logic
    exactly (`cli.py:237-264`) — a `check_study()` report **only when
    a config already existed** (the `elif not dry_run:` branch). A
    freshly migrated config's `check_study()` report is not computed
    here either, the same as today; this is moved logic, not a new
    check, so the migration branch's result carries the
    migrated-config flag but no report. A notebook user can still tell,
    from the return value alone, whether a config was just created and
    — when one already existed — what it leaves unset.
  - `project_status` returns both halves of what it prints: the list of
    `NotebookState` from `plan_update()` and the `check_study()` report
    dict — `cli.status` already computes and prints both; only the
    config-report half was captured in the first draft of this design.
- `update_project(path)` / `project_status(path)` on a directory that does
  not exist raise a plain `FileNotFoundError`, each carrying that
  command's own existing message (`cli.update`'s "...Use 'fire-impacts
  new' to create one." and `cli.status`'s "No such project directory:
  {path}" are different messages today, and stay different), not a
  `typer`-specific exception — so each behaves the way any other Python
  function does when called directly.
  `new_project(path)` on a directory that is already a project raises
  whatever `FireImpactsProject(path)` raises today (`FileExistsError`,
  uncaught) — this change leaves that path untouched rather than
  translating it, since `cli.new` itself has no existence check or
  translation for it today.
- These functions contain the actual logic (what is today split between
  `cli.new`/`cli.update`/`cli.status` and the `cli._summarise` /
  `cli._seed_from_project` helpers). That logic moves here unchanged in
  behaviour, just re-expressed with `print()` in place of
  `typer.echo`/`logger.info`, and plain exceptions in place of
  `typer.BadParameter`.

### 2. `cli.py` becomes a thin wrapper

`update` and `status` parse arguments, call the matching `project_api`
function, and catch `FileNotFoundError` to re-raise as `typer.BadParameter`
(preserving today's CLI exit-code behaviour for a missing project
directory). `new` has no such check today and gets none added — it calls
`new_project` directly, and a `FileExistsError` from an already-existing
project surfaces exactly as it does today (an uncaught traceback).
`check-network` is untouched — it is a diagnostic, not project setup, and
is out of scope here.

### 3. Top-level export

`fire_impacts/__init__.py` gains:

```python
from .project_api import new_project, update_project, project_status
```

so a notebook does `from fire_impacts import new_project` rather than
reaching into a submodule.

### 4. README quickstart

A new "Quickstart" section near the top of `README.md`, before the existing
"Installation" detail, walking through exactly four steps — install, create
a project, edit `study.toml`, open `PrepareData.ipynb` — showing the
terminal command and its `project_api` equivalent side by side at the one
step that has both:

```
fire-impacts new ./my-project
```
```python
from fire_impacts import new_project
new_project('./my-project')
```

The existing "Configuring a study" and "Starting a project, and keeping
its notebooks current" sections stay exactly where they are and keep
their current content — the quickstart does not re-explain the
uncommented/commented `study.toml` convention, the backup behaviour of
`update`, or anything else those sections already cover in depth. It
links to them by heading (`[Configuring a study](#configuring-a-study)`,
`[Starting a project...](#starting-a-project-and-keeping-its-notebooks-current)`)
at the point each quickstart step would otherwise need to go into more
detail, with wording such as "see Configuring a study below for what
goes in this file" rather than restating it. The quickstart's own step
order (install → create → configure → open notebook) differs from the
current document order (Configuring a study precedes Starting a
project) — that is fine, since the quickstart is a self-contained
condensed path and the two linked sections do not need to match its
order, only to not be contradicted by it.

## What this does not change

- No change to `notebooks.py`, `study.py`, `study_scaffold.py` or
  `netcheck.py` — `project_api.py` calls their existing public functions
  as-is. One consequence: `refresh_notebooks()` itself still narrates
  through its own `logger` (per-notebook added/updated/backed-up lines,
  and the warning that an edited notebook is being overwritten with
  backups off). That logger output is invisible in a fresh notebook
  kernel the same way `cli.py`'s used to be. This change does not fix
  that — `update_project`'s own `print()`-based summary (ported from
  `cli._summarise`) covers the same ground in aggregate (counts and
  names, and the backups-off warning at the `cli.py` level), so nothing
  safety-relevant is silently lost, but the per-notebook detail some
  users might want from `refresh_notebooks`'s own logging is only
  visible with `logging.basicConfig()` configured, in a notebook as in
  a terminal. Worth a follow-up if it turns out to matter in practice.
- No change to the CLI's observable behaviour or its existing tests
  (`test_study_cli.py`), which exercise it through `typer.testing.CliRunner`
  and assert on `result.output` / `result.exit_code`.
- `check-network` gets no notebook equivalent in this change — it is a
  separate concern (diagnosing TLS interception) from project setup, and
  nothing in the request asked for one.

## Testing

A new `test_project_api.py` mirrors the scenarios already covered in
`test_study_cli.py` (config written/seeded/left alone, drift reported,
migration of a pre-`study.toml` project), calling `new_project` /
`update_project` / `project_status` directly instead of through
`CliRunner`, and additionally asserts on each function's return value
(the migrated-config flag and `check_study()` report from
`update_project`; the combined notebook-states-and-report from
`project_status`). `test_study_cli.py` is expected to keep passing
unmodified: verified line-by-line against the current file, none of its
assertions depend on `logger`-originated text (they all check
`typer.echo`'d strings or filesystem state), so moving that output to
`print()` does not change anything `CliRunner` captures.
