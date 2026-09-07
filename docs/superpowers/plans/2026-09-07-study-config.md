# Study Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every value a user must supply out of the four template notebooks and into a single `study.toml` per project, read by a schema-driven `fire_impacts.study` module.

**Architecture:** Frozen dataclasses carrying `field(metadata=...)` describe the settings. That metadata is the single source of truth for three consumers — the loader's validation, the generated commented scaffold, and the drift report shown by `fire-impacts status`. Validation/coercion machinery already exists in `params.py` and is lifted into a shared internal `_schema.py` rather than duplicated. Templates gain a "Settings for this study" block that unpacks named constants; their bodies reference those names.

**Tech Stack:** Python 3.11+, `tomllib` (standard library, read-only), `dataclasses`, `difflib`, `pytest`, `typer` (existing CLI), `jupytext` (templates are `py:percent` scripts).

**Spec:** `docs/superpowers/specs/2026-09-07-study-config-design.md`

## Global Constraints

- **Python floor is 3.11** (`pyproject.toml:16`, `requires-python = ">=3.11"`). `tomllib` is standard library from 3.11 — do not add a TOML dependency.
- **No TOML *writing* library.** The scaffold is generated as text from the schema. `tomllib` is read-only and that is sufficient; do not add `tomlkit` or `toml`.
- **Windows is the primary platform.** Every path in generated TOML uses **literal strings (single quotes)** so backslashes need no escaping. Never emit a double-quoted Windows path.
- **`study.toml` is the user's file.** It is never fingerprinted, backed up, overwritten or recorded in `.fire_impacts_notebooks.json`.
- **House style** (follow `params.py` and `notebooks.py`): module docstring explaining *why*; `###...###` banner comments above public functions; `Parameters:` / `Returns:` / `Notes:` docstring sections; British spelling in prose.
- **Templates are `py:percent` jupytext scripts.** Cells are separated by `# %%` and markdown cells by `# %% [markdown]` with `#`-prefixed lines. Editing a template means editing the `.py` under `fire_impacts/templates/` only — the `.ipynb` is generated on install.
- **The example must keep working.** After every template change, the scaffolded `study.toml` values must still describe the bundled `test_data` example.

---

### Task 1: Lift the shared schema helpers into `_schema.py`

Six private helpers in `params.py` are exactly the machinery `study.py` needs. Move them verbatim into a new internal module, with one additive change each to `_from_dict` (its error text hardcodes the word "parameter") and `_coerce` (it has no `list` branch, which `recovery_breakpoints` needs).

**Files:**
- Create: `fire_impacts/_schema.py`
- Modify: `fire_impacts/params.py` (remove the six definitions; import them back)
- Test: `fire_impacts/tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `_schema.did_you_mean(name: str, options) -> str`
  - `_schema.to_dict(obj) -> Any`
  - `_schema.hints(cls) -> dict`
  - `_schema.coerce(value, annotation, path: str)`
  - `_schema.deep_merge(base: dict, overlay: dict) -> dict`
  - `_schema.from_dict(cls, data: dict, *, path: str, noun: str = 'parameter')`

  Public (no leading underscore) inside a private module. `params.py` keeps its old private names as aliases so nothing referencing them breaks.

- [ ] **Step 1: Write the failing test**

Create `fire_impacts/tests/test_schema.py`:

```python
"""
The shared schema helpers behave the same after being lifted out of
params.py, and handle the list annotation that study settings need.
"""

from dataclasses import dataclass, field

import pytest

from fire_impacts import _schema


@dataclass(frozen=True)
class Inner:
    a: int = 1
    b: float = 2.0


@dataclass(frozen=True)
class Outer:
    inner: Inner = Inner()
    name: str = 'x'
    items: list | None = None


def test_unknown_key_raises_with_a_suggestion():
    with pytest.raises(ValueError, match="Did you mean 'name'"):
        _schema.from_dict(Outer, {'nme': 'y'}, path='')


def test_the_noun_in_the_error_is_configurable():
    with pytest.raises(ValueError, match='Unknown setting'):
        _schema.from_dict(Outer, {'nope': 1}, path='', noun='setting')


def test_the_default_noun_is_parameter():
    with pytest.raises(ValueError, match='Unknown parameter'):
        _schema.from_dict(Outer, {'nope': 1}, path='')


def test_a_partial_nested_override_keeps_the_other_defaults():
    result = _schema.from_dict(Outer, {'inner': {'a': 5}}, path='')
    assert result.inner.a == 5
    assert result.inner.b == 2.0


def test_an_int_for_a_float_field_normalises():
    result = _schema.from_dict(Outer, {'inner': {'b': 3}}, path='')
    assert isinstance(result.inner.b, float)


def test_a_string_where_a_number_belongs_is_rejected():
    with pytest.raises(ValueError, match='must be a whole number'):
        _schema.from_dict(Outer, {'inner': {'a': 'five'}}, path='')


def test_a_list_field_accepts_a_list():
    result = _schema.from_dict(Outer, {'items': [0, 1, 2]}, path='')
    assert result.items == [0, 1, 2]


def test_a_list_field_rejects_a_scalar():
    with pytest.raises(ValueError, match='must be a list'):
        _schema.from_dict(Outer, {'items': 3}, path='')


def test_params_still_exposes_the_old_private_names():
    """params.py re-exports them, so anything importing them keeps working."""
    from fire_impacts import params
    assert params._did_you_mean('nme', ['name']) == " Did you mean 'name'?"
    assert params._deep_merge({'a': {'b': 1}}, {'a': {'c': 2}}) == {
        'a': {'b': 1, 'c': 2}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest fire_impacts/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fire_impacts._schema'`

- [ ] **Step 3: Create `fire_impacts/_schema.py`**

Move the bodies of `_to_dict`, `_did_you_mean`, `_from_dict`, `_hints`, `_coerce`, `_deep_merge` out of `params.py` verbatim, renaming them without the leading underscore, and applying the two additive changes below.

**Move the module-level cache too.** `_hints` reads `_TYPE_HINTS: dict = {}`, declared at `params.py:969`. Move that declaration into `_schema.py` alongside `hints()`. Leaving it behind in `params.py` gives you either a `NameError` or — worse, because it fails silently — two independent caches. It is not re-exported; nothing outside `hints()` touches it.

The module needs this header:

```python
"""
Schema machinery shared by the calibration parameters and the study
settings.

Both subsystems do the same three things to a dict that came out of a
file a user hand-edited: reject keys that are not in the schema (with a
suggestion, because a silently ignored typo lets someone believe they
configured something they did not), coerce values to the annotated type,
and merge a sparse override over a set of defaults.

These started life private in ``params.py``. They live here so that
``study.py`` can reuse them rather than grow a second, subtly different
copy of the same validation - two implementations of "did you spell this
right" would eventually disagree, and the disagreement would show up as
one file being stricter than the other for no reason a user could see.
"""

from __future__ import annotations

import difflib
import numbers
import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints
```

**Additive change 1 — `from_dict` takes a `noun`.** Thread it through the recursive call so nested groups report the same noun:

```python
def from_dict(cls, data: dict, *, path: str, noun: str = 'parameter'):
    """
    Build a (possibly nested) dataclass from a dict, rejecting unknown keys.

    Parameters:
    - cls: Dataclass to build.
    - data: Dict of values, which may be sparse.
    - path: Dotted prefix used in error messages.
    - noun: What to call an entry in error messages - 'parameter' for
      calibration parameters, 'setting' for study settings.

    Returns:
    - An instance of cls.
    --------------------------------------------------------------------
    """
    if not isinstance(data, dict):
        raise ValueError(
            f'{path or noun + "s"}: expected an object, got '
            f'{type(data).__name__}.'
        )
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            where = f'{path}{key}' if path else key
            raise ValueError(
                f'Unknown {noun} {where!r}.'
                f'{did_you_mean(key, known)} '
                f'Valid names here: {sorted(known)}.'
            )
        default = known[key].default
        if is_dataclass(default) and not isinstance(default, type):
            if not isinstance(value, dict):
                raise ValueError(
                    f'{path}{key}: expected an object, got '
                    f'{type(value).__name__}.'
                )
            kwargs[key] = from_dict(
                type(default),
                deep_merge(to_dict(default), value),
                path=f'{path}{key}.',
                noun=noun,
            )
        else:
            kwargs[key] = coerce(
                value, hints(cls).get(key), f'{path}{key}',
            )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f'{path or noun + "s"}: {exc}') from exc
    except ValueError as exc:
        message = str(exc)
        prefix = path or ''
        if prefix and not message.startswith(prefix):
            message = f'{prefix}{message}'
        raise ValueError(message) from None
```

Note the one behavioural subtlety: the original used the literal
`'parameters'` in the two "expected an object" / TypeError messages. `noun + "s"`
reproduces that exactly for the default noun.

**Additive change 2 — `coerce` gains a `list` branch.** Insert it immediately
before the final `return value`, so every annotation `params.py` already
handles keeps its existing path:

```python
    if annotation is list or get_origin(annotation) is list:
        if not isinstance(value, list):
            raise ValueError(f'{path} must be a list, got {value!r}.')
        member = (get_args(annotation) or (None,))[0]
        return [coerce(v, member, f'{path}[{i}]')
                for i, v in enumerate(value)]

    return value
```

- [ ] **Step 4: Rewire `params.py`**

Delete the six moved definitions from `params.py`. Immediately after its
existing `from .const import (...)` block, add:

```python
# These were private to this module until study.py needed the same
# validation. They live in _schema now; the old names are kept because
# they are referenced throughout this file.
from ._schema import (
    coerce as _coerce,
    deep_merge as _deep_merge,
    did_you_mean as _did_you_mean,
    from_dict as _from_dict,
    hints as _hints,
    to_dict as _to_dict,
)
```

Then remove any imports in `params.py` that are now unused. Check each of
`difflib`, `numbers`, `types`, `get_args`, `get_origin`, `get_type_hints`,
`Union`, `is_dataclass`, `Any` — remove only those with no remaining use.
Run `python -m pyflakes fire_impacts/params.py` (or grep) to confirm.

- [ ] **Step 5: Run the new tests**

Run: `pytest fire_impacts/tests/test_schema.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Run the existing parameter tests — this is the evidence the move was clean**

Run: `pytest fire_impacts/tests/test_params.py fire_impacts/tests/test_params_persistence.py -v`
Expected: PASS, with no changes to those files. If any test fails, the move
was not pure — fix `_schema.py` rather than the test.

- [ ] **Step 7: Run the whole suite**

Run: `pytest fire_impacts/tests/ -q`
Expected: PASS (no new failures versus the pre-change baseline)

- [ ] **Step 8: Commit**

```bash
git add fire_impacts/_schema.py fire_impacts/params.py fire_impacts/tests/test_schema.py
git commit -m "Share the schema validation between parameters and settings

study.toml needs the same unknown-key rejection, type coercion and
sparse merge that parameters.json already has. Rather than grow a second
copy that would eventually disagree with the first, the helpers move to
an internal _schema module and params.py imports them back under their
old private names.

Two additions the study settings need: from_dict takes the noun to use
in its error messages, and coerce learns about lists."
```

---

### Task 2: The study settings schema and `load_study`

**Files:**
- Create: `fire_impacts/study.py`
- Test: `fire_impacts/tests/test_study.py`

**Interfaces:**
- Consumes: `_schema.from_dict`, `_schema.did_you_mean` from Task 1.
- Produces:
  - `StudySettings` with fields `project`, `catchment`, `event`, `ensemble`, `secrets`, `source`
  - `ProjectSettings(directory: str, clear: bool | None)`
  - `CatchmentSettings(name, boundary, aridity, dem, subcatchments)`
  - `EventSettings(name, fire_start, fire_end, recovery_breakpoints)`
  - `EnsembleSettings(name, num_replicates, inspect_replicate, mean_annual_rainfall, average_temperature)`
  - `SecretSettings(tern_api_key)`
  - `SourceSettings(port, constituent, functional_unit)`
  - `load_study(path: str = '.') -> StudySettings`
  - `StudyConfigError(Exception)`
  - `CONFIG_NAME = 'study.toml'`
  - `study.root` (str) and `study.provided` (frozenset of dotted keys present in the file)

- [ ] **Step 1: Write the failing test**

Create `fire_impacts/tests/test_study.py`:

```python
"""
Loading a project's study.toml.

The file is hand-edited by people who are not programmers, so the two
behaviours that matter most are that a typo is refused rather than
ignored - a silently dropped setting would let someone believe they had
configured a study they had not - and that a setting left out falls back
to its default, which is what lets an older file keep working against
newer templates.
"""

import pytest

from fire_impacts.study import (
    CONFIG_NAME, StudyConfigError, load_study,
)

MINIMAL = """
[catchment]
name     = "Cat"
boundary = 'boundary.shp'
aridity  = 'aridity.tif'

[event]
name       = "2019_fire"
fire_start = "2019-01-15"
fire_end   = "2019-03-07"
"""


def write_study(directory, text=MINIMAL, files=('boundary.shp',
                                                'aridity.tif')):
    """Write a study.toml plus the input files it points at."""
    (directory / CONFIG_NAME).write_text(text, encoding='utf-8')
    for name in files:
        (directory / name).write_text('', encoding='utf-8')
    return directory


def test_a_minimal_file_loads(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.catchment.name == 'Cat'
    assert study.event.fire_start == '2019-01-15'


def test_settings_left_out_fall_back_to_defaults(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.ensemble.name == 'stochastic'
    assert study.ensemble.num_replicates == 10
    assert study.source.port == 9876


def test_clear_is_unset_rather_than_false_when_omitted(tmp_path):
    """Tri-state: PrepareData resolves unset to True, everything else
    treats it as 'do not clear'. A plain boolean could not tell an
    explicit false from an omission."""
    study = load_study(str(write_study(tmp_path)))
    assert study.project.clear is None


def test_an_explicit_false_is_distinguishable_from_an_omission(tmp_path):
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[project]\nclear = false\n')))
    assert study.project.clear is False


def test_a_misspelled_setting_is_refused_with_a_suggestion(tmp_path):
    text = MINIMAL.replace('fire_start', 'fire_strat')
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(write_study(tmp_path, text)))
    assert 'fire_strat' in str(exc.value)
    assert "Did you mean 'fire_start'" in str(exc.value)
    assert 'setting' in str(exc.value)


def test_a_misspelled_section_is_refused(tmp_path):
    text = MINIMAL.replace('[event]', '[evnt]')
    with pytest.raises(StudyConfigError, match="Did you mean 'event'"):
        load_study(str(write_study(tmp_path, text)))


def test_a_missing_required_setting_names_the_file(tmp_path):
    text = MINIMAL.replace('name     = "Cat"', '')
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(write_study(tmp_path, text)))
    assert 'catchment.name' in str(exc.value)
    assert CONFIG_NAME in str(exc.value)


def test_a_missing_file_says_how_to_make_one(tmp_path):
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(tmp_path))
    assert CONFIG_NAME in str(exc.value)
    assert str(tmp_path) in str(exc.value)
    assert 'fire-impacts update' in str(exc.value)


def test_broken_toml_is_reported_as_a_config_error(tmp_path):
    with pytest.raises(StudyConfigError, match=CONFIG_NAME):
        load_study(str(write_study(tmp_path, '[catchment\nname = 1')))


def test_recovery_breakpoints_load_as_a_list_of_numbers(tmp_path):
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\nrecovery_breakpoints = [0, 0.5, 1]\n')))
    assert study.event.recovery_breakpoints == [0.0, 0.5, 1.0]


def test_which_settings_came_from_the_file_is_recorded(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert 'catchment.name' in study.provided
    assert 'ensemble.num_replicates' not in study.provided
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest fire_impacts/tests/test_study.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fire_impacts.study'`

- [ ] **Step 3: Write `fire_impacts/study.py`**

```python
"""
The settings a user supplies for one study.

The template notebooks are runnable except for a handful of values that
depend on the catchment being studied: where the boundary coverage is,
what the fire was called, when it burned. Those values used to sit in
the body of each notebook, which made them hard to find and - because
`fire-impacts update` replaces notebooks - hard to carry forward to a
newer template. They live in a single ``study.toml`` per project now,
and the notebooks read them.

The schema below is the single source of truth for three separate
consumers: this module's validation, the commented scaffold written by
`fire-impacts new`, and the drift report shown by `fire-impacts status`.
Keeping them on one definition is what stops a scaffold's explanatory
comments from describing settings the loader no longer accepts.

This is *not* ``parameters.json``. That file holds sparse calibration
overrides resolved through five scopes, and feeds a digest used to
detect that derived layers were built with different values than a run
resolves. A shapefile path has no calibration range and no scope, and
putting one in that digest would mean renaming a directory invalidated
every built layer. The two files answer different questions and stay
separate.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass

from ._schema import did_you_mean, from_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_NAME = 'study.toml'

# The environment variable consulted when secrets.tern_api_key is blank,
# so anyone who followed the previous notebook instructions keeps working.
TERN_ENV_VAR = 'TERN_API_KEY'


###############################################################################
class StudyConfigError(Exception):
    """A study.toml is missing, unreadable, or does not match the schema."""


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectSettings:
    """Where the project's data lives, and whether to start it afresh."""

    directory: str = field(default='.', metadata={
        'path': True,
        'help': 'project data root, relative to this file'})
    clear: bool | None = field(default=None, metadata={
        'example': 'false',
        'help': 'Leave this out and PrepareData starts from scratch, '
                'wiping existing project data. Set it false once the '
                'project holds work worth keeping.'})


@dataclass(frozen=True)
class CatchmentSettings:
    """The study area and the input rasters describing it."""

    name: str | None = field(default=None, metadata={
        'required': True,
        'help': 'name for this catchment within the project'})
    boundary: str | None = field(default=None, metadata={
        'required': True, 'path': True, 'must_exist': True,
        'help': 'boundary coverage (.shp/.geojson). Must carry a CRS - '
                'it becomes the CRS for everything else in the catchment.'})
    aridity: str | None = field(default=None, metadata={
        'required': True, 'path': True, 'must_exist': True,
        'help': 'aridity raster covering the catchment'})
    dem: str | None = field(default=None, metadata={
        'path': True, 'must_exist': True, 'example': "''",
        'help': 'default: download the GA 1 arc-second national DEM'})
    subcatchments: str | None = field(default=None, metadata={
        'path': True, 'must_exist': True, 'example': "''",
        'help': 'default: no subcatchment reporting'})
    subcatchment_id_field: str = field(default='SiteID', metadata={
        'example': "'SiteID'",
        'help': 'attribute naming each subcatchment. Must match the '
                'subcatchment names in your Source model.'})
    cell_size_m: float = field(default=30.0, metadata={
        'example': '30',
        'help': 'DEM cell size in metres, used to convert per-cell results '
                'to t/ha. Change it if your DEM is not 30 m.'})


@dataclass(frozen=True)
class EventSettings:
    """The fire being modelled."""

    name: str | None = field(default=None, metadata={
        'required': True,
        'help': 'name for this fire; becomes the Events/<name> directory'})
    fire_start: str | None = field(default=None, metadata={
        'required': True,
        'help': 'date the fire started, YYYY-MM-DD'})
    fire_end: str | None = field(default=None, metadata={
        'required': True,
        'help': 'date the fire ended, YYYY-MM-DD'})
    recovery_breakpoints: list | None = field(default=None, metadata={
        'example': '[0, 1, 2, 3]',
        'help': 'default: const.DEFAULT_RECOVERY_BREAKPOINTS'})


@dataclass(frozen=True)
class EnsembleSettings:
    """The stochastic rainfall realisation driving the simulations."""

    name: str = field(default='stochastic', metadata={
        'help': 'names this climate realisation'})
    num_replicates: int = field(default=10, metadata={
        'example': '10',
        'help': 'replicates drawn from pyraingen'})
    inspect_replicate: int = field(default=9, metadata={
        'example': '9',
        'help': 'which one the single-run Simulation notebook plots'})
    mean_annual_rainfall: float | None = field(default=None, metadata={
        'example': '600',
        'help': 'mm; default: estimated from catchment lat/lon'})
    average_temperature: float | None = field(default=None, metadata={
        'example': '20',
        'help': 'degrees C; default: estimated from catchment lat/lon'})
    n_workers: int = field(default=10, metadata={
        'example': '10',
        'help': 'replicates run in parallel; cap this for your machine'})


@dataclass(frozen=True)
class ReportingSettings:
    """Thresholds the ensemble notebook's exceedance maps are drawn at.

    Reporting choices, not model calibration: they change what a map
    shows, never what the model computes. That is why they are here and
    not in parameters.json.
    """

    erosion_threshold_t_ha: float = field(default=0.5, metadata={
        'example': '0.5',
        'help': 'catchment erosion exceedance threshold, t/ha'})
    delivered_threshold_kg_ha: float = field(default=500.0, metadata={
        'example': '500',
        'help': 'delivered-load exceedance threshold, kg/ha'})


@dataclass(frozen=True)
class SecretSettings:
    """Credentials for the data services the preprocessing downloads from."""

    tern_api_key: str = field(default='', metadata={
        'secret': True,
        'help': 'Your TERN API key, needed to download soil data.\n'
                'Free - see the "Soils" section of PrepareData for how to '
                'get one.\n'
                f'If left blank, the {TERN_ENV_VAR} environment variable '
                'is used instead.'})


@dataclass(frozen=True)
class SourceIntegrationSettings:
    """Connecting to a running eWater Source instance via Veneer."""

    port: int = field(default=9876, metadata={
        'example': '9876',
        'help': 'Veneer port for the running Source instance'})
    constituent: str | None = field(default=None, metadata={
        'example': "'TSS'",
        'help': 'default: auto-detected'})
    functional_unit: str | None = field(default=None, metadata={
        'example': "'Forested'",
        'help': 'default: auto-detected'})
    replicate: int = field(default=0, metadata={
        'example': '0',
        'help': 'which ensemble replicate to push into Source. Distinct '
                'from ensemble.inspect_replicate, which only chooses what '
                'the Simulation notebook plots.'})
    timestep: str = field(default='D', metadata={
        'example': "'D'",
        'help': "must match your Source model: 'D' for daily, 'h' hourly"})
    date_format: str = field(default='%d/%m/%Y', metadata={
        'example': "'%d/%m/%Y'",
        'help': 'how your Source install writes run-period dates'})
    output_dir: str = field(default='source_inputs', metadata={
        'example': "'source_inputs'",
        'help': 'where the generated CSVs are written, inside the project'})
    load_attenuation: float = field(default=10.0, metadata={
        'example': '10.0',
        'help': 'Load Distributor attenuation'})
    maximum_concentration: float = field(default=1000.0, metadata={
        'example': '1000.0',
        'help': 'Load Distributor concentration cap, mg/L'})
    tss_data_source: str = field(default='fire_tss', metadata={
        'example': "'fire_tss'",
        'help': 'name of the Source data source holding sediment load'})
    rainfall_data_source: str = field(default='stochastic_rain', metadata={
        'example': "'stochastic_rain'",
        'help': 'name of the Source data source holding rainfall'})


@dataclass(frozen=True)
class StudySettings:
    """Everything one study needs a user to supply."""

    project: ProjectSettings = ProjectSettings()
    catchment: CatchmentSettings = CatchmentSettings()
    event: EventSettings = EventSettings()
    ensemble: EnsembleSettings = EnsembleSettings()
    reporting: ReportingSettings = ReportingSettings()
    secrets: SecretSettings = SecretSettings()
    source: SourceIntegrationSettings = SourceIntegrationSettings()

    ###########################################################################
    @property
    def root(self):
        """
        Return the directory the study.toml was read from.

        Returns:
        - Absolute path, as a string.
        --------------------------------------------------------------------
        """
        return getattr(self, '_root', os.getcwd())

    ###########################################################################
    @property
    def provided(self):
        """
        Return the dotted names of settings the file actually set.

        Returns:
        - frozenset of names such as 'catchment.name'.
        --------------------------------------------------------------------
        Notes:
        - Distinguishing a value somebody chose from one that defaulted is
          what makes describe() worth reading.
        --------------------------------------------------------------------
        """
        return getattr(self, '_provided', frozenset())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


###############################################################################
def config_path(path):
    """
    Return the path a project's study.toml would live at.

    Parameters:
    - path: Project directory.

    Returns:
    - Full path to the config file, which need not exist.
    --------------------------------------------------------------------
    """
    return os.path.join(path, CONFIG_NAME)


###############################################################################
def _flatten_provided(data, prefix=''):
    """
    Return the dotted names of every leaf present in a parsed file.

    Parameters:
    - data: Parsed TOML, as nested dicts.
    - prefix: Dotted prefix for recursion.

    Returns:
    - Set of dotted names.
    --------------------------------------------------------------------
    """
    names = set()
    for key, value in data.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            names |= _flatten_provided(value, f'{name}.')
        else:
            names.add(name)
    return names


###############################################################################
def _check_required(settings, path):
    """
    Raise if a setting marked required was not supplied.

    Parameters:
    - settings: The StudySettings just built.
    - path: Path to the config file, for the error message.
    --------------------------------------------------------------------
    """
    missing = []
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        for entry in fields(value):
            if not entry.metadata.get('required'):
                continue
            if getattr(value, entry.name) in (None, ''):
                missing.append(f'{group.name}.{entry.name}')

    if missing:
        raise StudyConfigError(
            f'{path} is missing {len(missing)} required setting(s): '
            f'{", ".join(missing)}. Every setting that is not commented '
            f'out in the file is one you need to supply.'
        )


###############################################################################
def load_study(path='.'):
    """
    Read and validate a project's study.toml.

    Parameters:
    - path: Project directory holding the config file.

    Returns:
    - A StudySettings.
    --------------------------------------------------------------------
    Notes:
    - A setting left out of the file falls back to its default, silently.
      That is what lets a study.toml written against an older set of
      templates keep working after an update.
    - A setting that is *misspelled*, by contrast, raises. Ignoring it
      would let someone believe they had configured a study they had not.
    --------------------------------------------------------------------
    """
    fn = config_path(path)
    if not os.path.exists(fn):
        raise StudyConfigError(
            f'No {CONFIG_NAME} in {os.path.abspath(path)}. Run '
            f'"fire-impacts update {path}" to create one, or copy it '
            f'from another project.'
        )

    try:
        with open(fn, 'rb') as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise StudyConfigError(f'{fn} could not be read: {exc}') from None

    known = {f.name for f in fields(StudySettings)}
    for section in data:
        if section not in known:
            raise StudyConfigError(
                f'Unknown section [{section}] in {fn}.'
                f'{did_you_mean(section, known)} '
                f'Valid sections: {sorted(known)}.'
            )

    try:
        settings = from_dict(
            StudySettings, data, path='', noun='setting')
    except ValueError as exc:
        raise StudyConfigError(f'{fn}: {exc}') from None

    object.__setattr__(settings, '_root', os.path.abspath(path))
    object.__setattr__(
        settings, '_provided', frozenset(_flatten_provided(data)))

    _check_required(settings, fn)

    return settings
```

Two implementation notes for the engineer:

- `StudySettings` uses **frozen dataclass instances as defaults**
  (`project: ProjectSettings = ProjectSettings()`), not `default_factory`.
  `_schema.from_dict` detects a nested group by `is_dataclass(field.default)`,
  so a factory would break the sparse-merge behaviour. Frozen dataclass
  instances are hashable and immutable, so this is safe and is what
  `params.py` already does.
- `root` and `provided` are attached with `object.__setattr__` rather than
  declared as fields. A declared field would appear in `fields(StudySettings)`
  and so become a settable key in the TOML file, which it must not be.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/study.py fire_impacts/tests/test_study.py
git commit -m "Read a project's study.toml into a validated schema

One file per project holds the values the template notebooks need a user
to supply. A setting left out falls back to its default, so an older file
keeps working against newer templates; a misspelled one raises with a
suggestion, because a silently ignored typo would let someone believe
they had configured a study they had not."
```

---

### Task 3: Resolve paths against the file, and check the required ones exist

**Files:**
- Modify: `fire_impacts/study.py`
- Test: `fire_impacts/tests/test_study.py`

**Interfaces:**
- Consumes: `load_study`, the `path` / `must_exist` field metadata from Task 2.
- Produces: every `path`-marked setting comes back as an absolute path string. `load_study` raises `StudyConfigError` for a `must_exist` path that is absent.

- [ ] **Step 1: Write the failing test**

Append to `fire_impacts/tests/test_study.py`:

```python
import os


def test_paths_resolve_against_the_file_not_the_cwd(tmp_path, monkeypatch):
    """A notebook can be run from anywhere; '..\\test_data\\x.shp' has to
    mean the same thing regardless of where the kernel started."""
    project = tmp_path / 'proj'
    project.mkdir()
    write_study(project)

    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    study = load_study(str(project))
    assert os.path.isabs(study.catchment.boundary)
    assert study.catchment.boundary == str(project / 'boundary.shp')


def test_the_project_directory_is_resolved_too(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert os.path.isabs(study.project.directory)
    assert study.project.directory == str(tmp_path)


def test_a_missing_required_input_is_caught_at_load(tmp_path):
    """Better than a geopandas traceback forty cells later."""
    write_study(tmp_path, files=('aridity.tif',))
    with pytest.raises(StudyConfigError) as exc:
        load_study(str(tmp_path))
    assert 'catchment.boundary' in str(exc.value)
    assert 'boundary.shp' in str(exc.value)


def test_an_optional_path_left_unset_is_not_checked(tmp_path):
    study = load_study(str(write_study(tmp_path)))
    assert study.catchment.dem is None


def test_a_supplied_optional_path_must_still_exist(tmp_path):
    text = MINIMAL.replace(
        "aridity  = 'aridity.tif'",
        "aridity  = 'aridity.tif'\ndem = 'nope.tif'")
    write_study(tmp_path, text)
    with pytest.raises(StudyConfigError, match='catchment.dem'):
        load_study(str(tmp_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest fire_impacts/tests/test_study.py -k "path or directory or input" -v`
Expected: FAIL — paths come back as the literal relative strings

- [ ] **Step 3: Add resolution and checking to `study.py`**

Add these two functions before `load_study`:

```python
###############################################################################
def _resolve_paths(settings, root):
    """
    Rewrite every path setting as an absolute path.

    Parameters:
    - settings: The StudySettings just built.
    - root: Directory the config file was read from.

    Returns:
    - A new StudySettings with paths resolved.
    --------------------------------------------------------------------
    Notes:
    - Relative to the config file, not the working directory: a notebook
      may be run from anywhere, and '..\\test_data\\x.shp' has to mean the
      same thing regardless.
    --------------------------------------------------------------------
    """
    from dataclasses import replace

    groups = {}
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        changes = {}
        for entry in fields(value):
            if not entry.metadata.get('path'):
                continue
            raw = getattr(value, entry.name)
            if raw in (None, ''):
                continue
            changes[entry.name] = os.path.abspath(os.path.join(root, raw))
        groups[group.name] = replace(value, **changes) if changes else value

    return replace(settings, **groups)


###############################################################################
def _check_paths_exist(settings, path):
    """
    Raise if an input file named in the config is not there.

    Parameters:
    - settings: The StudySettings, with paths already resolved.
    - path: Path to the config file, for the error message.
    --------------------------------------------------------------------
    Notes:
    - Caught here rather than left to fail deep inside a read, so the
      message names the setting to fix instead of arriving as a geopandas
      traceback many cells into a notebook run.
    --------------------------------------------------------------------
    """
    missing = []
    for group in fields(StudySettings):
        value = getattr(settings, group.name)
        for entry in fields(value):
            if not entry.metadata.get('must_exist'):
                continue
            resolved = getattr(value, entry.name)
            if resolved and not os.path.exists(resolved):
                missing.append(
                    f'{group.name}.{entry.name}: no such file {resolved}')

    if missing:
        raise StudyConfigError(
            f'Input files named in {path} are missing:\n  '
            + '\n  '.join(missing)
        )
```

Then in `load_study`, replace the tail (from `object.__setattr__` onwards) with:

```python
    root = os.path.abspath(path)
    settings = _resolve_paths(settings, root)
    object.__setattr__(settings, '_root', root)
    object.__setattr__(
        settings, '_provided', frozenset(_flatten_provided(data)))

    _check_required(settings, fn)
    _check_paths_exist(settings, fn)

    return settings
```

Note the ordering: `_resolve_paths` returns a **new** instance via
`dataclasses.replace`, so `_root` and `_provided` must be attached
*after* it, not before.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/study.py fire_impacts/tests/test_study.py
git commit -m "Resolve study paths against the config file, and check they exist

A notebook gets run from wherever the kernel happened to start, so a
relative path in study.toml has to mean the same thing regardless. A
required input that is not there is now caught at load, naming the
setting to fix, rather than surfacing as a read error many cells later."
```

---

### Task 4: `secret()` and `describe()`

**Files:**
- Modify: `fire_impacts/study.py`
- Test: `fire_impacts/tests/test_study.py`

**Interfaces:**
- Consumes: `StudySettings` from Task 2, `provided` from Task 2.
- Produces:
  - `StudySettings.secret(name: str) -> str` — raises `StudyConfigError` when neither the setting nor the environment variable is set
  - `StudySettings.describe() -> None` — prints; returns nothing
  - `StudySettings.describe_text() -> str` — the same content as a string, so it can be tested

- [ ] **Step 1: Write the failing test**

Append to `fire_impacts/tests/test_study.py`:

```python
def test_the_key_comes_from_the_file_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv('TERN_API_KEY', 'from-env')
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[secrets]\ntern_api_key = "from-file"\n')))
    assert study.secret('tern_api_key') == 'from-file'


def test_the_key_falls_back_to_the_environment_variable(tmp_path,
                                                        monkeypatch):
    """Anyone who followed the old notebook instructions keeps working."""
    monkeypatch.setenv('TERN_API_KEY', 'from-env')
    study = load_study(str(write_study(tmp_path)))
    assert study.secret('tern_api_key') == 'from-env'


def test_a_missing_key_raises_only_when_it_is_asked_for(tmp_path,
                                                        monkeypatch):
    """A blank key must not stop someone running the Simulation notebook,
    which never touches TERN."""
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(tmp_path)))       # no raise
    with pytest.raises(StudyConfigError) as exc:
        study.secret('tern_api_key')
    assert 'tern_api_key' in str(exc.value)
    assert 'TERN_API_KEY' in str(exc.value)


def test_describe_says_where_each_value_came_from(tmp_path):
    text = MINIMAL + '\n[ensemble]\nnum_replicates = 4\n'
    study = load_study(str(write_study(tmp_path, text)))
    out = study.describe_text()
    assert 'catchment.name' in out
    assert 'study.toml' in out
    assert 'default' in out
    assert 'parameters.json' in out          # points at the other file


def test_describe_never_prints_the_key(tmp_path):
    """.ipynb files carry their output cells and are the artefact most
    likely to be emailed or committed."""
    study = load_study(str(write_study(
        tmp_path, MINIMAL + '\n[secrets]\ntern_api_key = "sekrit"\n')))
    out = study.describe_text()
    assert 'sekrit' not in out
    assert 'set' in out


def test_describe_reports_an_unset_key_as_not_set(tmp_path, monkeypatch):
    monkeypatch.delenv('TERN_API_KEY', raising=False)
    study = load_study(str(write_study(tmp_path)))
    assert 'not set' in study.describe_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest fire_impacts/tests/test_study.py -k "key or describe" -v`
Expected: FAIL — `AttributeError: 'StudySettings' object has no attribute 'secret'`

- [ ] **Step 3: Add the methods to `StudySettings`**

```python
    ###########################################################################
    def secret(self, name):
        """
        Return a credential, from the config file or the environment.

        Parameters:
        - name: Setting name within [secrets], e.g. 'tern_api_key'.

        Returns:
        - The credential, as a string.
        --------------------------------------------------------------------
        Notes:
        - Raises here rather than at load time: only the soil download
          needs the TERN key, and a blank one must not stop someone
          running a notebook that never touches TERN.
        --------------------------------------------------------------------
        """
        known = {f.name for f in fields(SecretSettings)}
        if name not in known:
            raise StudyConfigError(
                f'Unknown secret {name!r}.{did_you_mean(name, known)}')

        value = getattr(self.secrets, name) or os.environ.get(TERN_ENV_VAR, '')
        if not value:
            raise StudyConfigError(
                f'No {name} available. Set it in the [secrets] section of '
                f'{CONFIG_NAME}, or set the {TERN_ENV_VAR} environment '
                f'variable. See the "Soils" section of the PrepareData '
                f'notebook for how to obtain one.'
            )
        return value

    ###########################################################################
    def describe_text(self):
        """
        Return a readable summary of the settings and their origins.

        Returns:
        - Multi-line string.
        --------------------------------------------------------------------
        Notes:
        - Secrets are reported as 'set'/'not set', never printed. Notebook
          output cells are saved into the .ipynb and get shared.
        --------------------------------------------------------------------
        """
        lines = [f'Study settings from {os.path.join(self.root, CONFIG_NAME)}',
                 '']
        for group in fields(StudySettings):
            value = getattr(self, group.name)
            for entry in fields(value):
                dotted = f'{group.name}.{entry.name}'
                origin = ('study.toml' if dotted in self.provided
                          else 'default')
                if entry.metadata.get('secret'):
                    shown = 'set' if (getattr(value, entry.name)
                                      or os.environ.get(TERN_ENV_VAR)
                                      ) else 'not set'
                    origin = ('study.toml' if dotted in self.provided
                              else f'{TERN_ENV_VAR}' if os.environ.get(
                                  TERN_ENV_VAR) else 'default')
                else:
                    shown = getattr(value, entry.name)
                lines.append(f'  {dotted:<34} {str(shown):<44} ({origin})')

        lines += [
            '',
            'Calibration parameters are separate, and live in '
            'parameters.json;',
            'see the "Calibration parameters" section of the PrepareData '
            'notebook.',
        ]
        return '\n'.join(lines)

    ###########################################################################
    def describe(self):
        """
        Print a readable summary of the settings and their origins.
        --------------------------------------------------------------------
        """
        print(self.describe_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/study.py fire_impacts/tests/test_study.py
git commit -m "Report the settings, and fetch the TERN key without printing it

describe() is the first thing a notebook shows, so it says where each
value came from - a value somebody chose reads differently from one that
merely defaulted. It reports the API key as set/not set rather than
printing it: .ipynb files carry their output cells and are the artefact
most likely to be emailed around."
```

---

### Task 5: Generate the commented scaffold from the schema

**Files:**
- Modify: `fire_impacts/study.py`
- Test: `fire_impacts/tests/test_study_scaffold.py`

**Interfaces:**
- Consumes: the schema and its `help` / `required` / `example` / `secret` metadata from Task 2.
- Produces:
  - `scaffold_text(seed: dict | None = None) -> str`
  - `write_scaffold(path: str, seed: dict | None = None) -> str` (returns the path written; raises `StudyConfigError` if the file already exists)
  - `EXAMPLE_SEED: dict` — the bundled example's values, keyed by dotted name

- [ ] **Step 1: Write the failing test**

Create `fire_impacts/tests/test_study_scaffold.py`:

```python
"""
The scaffold that `fire-impacts new` writes.

Its comments are generated from the same field metadata the loader
validates against, so they cannot describe settings the loader would
reject. The round-trip test below is what enforces that: a scaffold that
does not load is a scaffold whose comments have drifted from the code.
"""

import os
from dataclasses import fields

import pytest

from fire_impacts.study import (
    CONFIG_NAME, EXAMPLE_SEED, StudyConfigError, StudySettings,
    load_study, scaffold_text, write_scaffold,
)


def test_every_setting_appears_in_the_scaffold():
    """Adding a field without documenting it should fail here."""
    text = scaffold_text()
    for group in fields(StudySettings):
        for entry in fields(getattr(StudySettings(), group.name)):
            assert entry.name in text, (
                f'{group.name}.{entry.name} is missing from the scaffold')


def test_required_settings_are_uncommented_and_optional_ones_are_not():
    """The convention that answers 'what must I change?' at a glance."""
    text = scaffold_text()
    lines = [ln.strip() for ln in text.splitlines()]

    assert any(ln.startswith('name') for ln in lines)          # required
    assert any(ln.startswith('boundary') for ln in lines)      # required
    assert not any(ln.startswith('num_replicates') for ln in lines)
    assert any(ln.startswith('# num_replicates') for ln in lines)


def test_paths_are_written_as_toml_literal_strings():
    """Windows backslashes must not need escaping."""
    text = scaffold_text(EXAMPLE_SEED)
    for line in text.splitlines():
        if 'test_data' in line and not line.strip().startswith('#'):
            assert "'" in line, f'not a literal string: {line}'
            assert '"' not in line, f'double-quoted path: {line}'


def test_the_example_scaffold_loads(tmp_path):
    """Round trip: what `fire-impacts new` writes must be readable by
    load_study, or the two have drifted apart."""
    for name in ('EgSmallCatchment_7899.shp',
                 'AridityPT_EgSmallCatchment_7899.tif'):
        (tmp_path / name).write_text('', encoding='utf-8')

    seed = dict(EXAMPLE_SEED)
    seed['catchment.boundary'] = 'EgSmallCatchment_7899.shp'
    seed['catchment.aridity'] = 'AridityPT_EgSmallCatchment_7899.tif'

    write_scaffold(str(tmp_path), seed)
    study = load_study(str(tmp_path))

    assert study.catchment.name == 'EgSmallCatchment_7899'
    assert study.event.name == '2019_fire'
    assert study.ensemble.name == 'stochastic'
    assert study.project.clear is None


def test_write_scaffold_refuses_to_overwrite(tmp_path):
    """study.toml is the user's file. Nothing replaces it."""
    (tmp_path / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')
    with pytest.raises(StudyConfigError, match='already'):
        write_scaffold(str(tmp_path))
    assert (tmp_path / CONFIG_NAME).read_text() == '# mine\n'


def test_a_seeded_value_that_could_not_be_recovered_is_flagged(tmp_path):
    """Migration: what can't be read off an existing project is left blank
    with a comment telling the user to fill it in."""
    text = scaffold_text({'catchment.name': 'RealCatchment',
                          'catchment.boundary': None})
    assert 'RealCatchment' in text
    assert 'could not be recovered' in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest fire_impacts/tests/test_study_scaffold.py -v`
Expected: FAIL — `ImportError: cannot import name 'scaffold_text'`

- [ ] **Step 3: Add the generator to `study.py`**

```python
# The bundled example, keyed by dotted setting name. `fire-impacts new`
# seeds a scaffold with these so a fresh project runs end to end before
# the user changes anything.
EXAMPLE_SEED = {
    'catchment.name': 'EgSmallCatchment_7899',
    'catchment.boundary': r'..\test_data\EgSmallCatchment_7899.shp',
    'catchment.aridity':
        r'..\test_data\AridityPT_EgSmallCatchment_7899.tif',
    'event.name': '2019_fire',
    'event.fire_start': '2019-01-15',
    'event.fire_end': '2019-03-07',
}

SCAFFOLD_HEADER = f"""\
# {CONFIG_NAME} - everything this study needs you to supply.
#
# Uncommented settings are required. Commented-out settings are
# optional; the comment shows what happens if you leave them out.
# Values shown here drive the bundled example catchment, so a new
# project runs end to end before you change anything.
#
# Calibration parameters are NOT set here - they live in
# parameters.json. See the PrepareData notebook.
"""

# Settings whose value is a filesystem path are written as TOML literal
# strings, so a Windows backslash needs no escaping.
_MISSING_NOTE = '# could not be recovered from the project; please fill this in'


###############################################################################
def _toml_value(value, is_path):
    """
    Render a Python value as TOML.

    Parameters:
    - value: Value to render.
    - is_path: True to use a literal (single-quoted) string.

    Returns:
    - TOML source for the value.
    --------------------------------------------------------------------
    """
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
    - Required settings are written uncommented, optional ones commented
      out. That convention is what lets a user see what they must supply
      by looking at which lines are live.
    --------------------------------------------------------------------
    """
    seed = seed or {}
    out = [SCAFFOLD_HEADER]

    for group in fields(StudySettings):
        out.append(f'\n[{group.name}]')
        blank = getattr(StudySettings(), group.name)

        for entry in fields(blank):
            dotted = f'{group.name}.{entry.name}'
            meta = entry.metadata
            is_path = bool(meta.get('path'))

            for line in meta.get('help', '').split('\n'):
                if line:
                    out.append(f'# {line}')

            required = bool(meta.get('required'))
            seeded = dotted in seed

            if seeded and seed[dotted] is None:
                out.append(f'{entry.name} = ""  {_MISSING_NOTE}')
                continue

            if seeded:
                out.append(
                    f'{entry.name} = {_toml_value(seed[dotted], is_path)}')
                continue

            if required:
                # No seed and required: leave it live but empty, so the
                # loader's own error tells the user what to fill in.
                out.append(f'{entry.name} = ""')
                continue

            shown = meta.get('example')
            if shown is None:
                shown = _toml_value(entry.default, is_path)
            out.append(f'# {entry.name} = {shown}')

    return '\n'.join(out) + '\n'


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study_scaffold.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Eyeball the generated file**

Run:

```bash
python -c "from fire_impacts.study import scaffold_text, EXAMPLE_SEED; print(scaffold_text(EXAMPLE_SEED))"
```

Expected: the file shown in the "The file" section of the spec. Confirm by
eye that required settings are live, optional ones commented, and every
path is single-quoted.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/study.py fire_impacts/tests/test_study_scaffold.py
git commit -m "Generate study.toml from the schema that validates it

The scaffold's explanatory comments come from the same field metadata the
loader checks against, so they cannot drift into describing settings that
no longer exist. A round-trip test - write the scaffold, load it back -
is what holds that true.

Required settings are written live and optional ones commented out, so
'what do I have to change?' is answered by looking at the file."
```

---

### Task 6: The drift report

**Files:**
- Modify: `fire_impacts/study.py`
- Test: `fire_impacts/tests/test_study_scaffold.py`

**Interfaces:**
- Consumes: the schema from Task 2, `load_study` from Task 2.
- Produces: `check_study(path: str) -> dict` with keys `missing_file` (bool), `unknown` (list of `str` messages), `required_missing` (list of dotted names), `unset` (list of dotted names available but not set).

- [ ] **Step 1: Write the failing test**

Append to `fire_impacts/tests/test_study_scaffold.py`:

```python
from fire_impacts.study import check_study


def test_check_reports_a_missing_file(tmp_path):
    report = check_study(str(tmp_path))
    assert report['missing_file'] is True


def test_check_reports_a_typo_with_a_suggestion(tmp_path):
    (tmp_path / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert any("Did you mean 'name'" in u for u in report['unknown'])


def test_check_lists_settings_available_but_not_set(tmp_path):
    for name in ('b.shp', 'a.tif'):
        (tmp_path / name).write_text('', encoding='utf-8')
    (tmp_path / CONFIG_NAME).write_text(
        "[catchment]\nname='c'\nboundary='b.shp'\naridity='a.tif'\n"
        "[event]\nname='e'\nfire_start='2019-01-15'\n"
        "fire_end='2019-03-07'\n",
        encoding='utf-8')
    report = check_study(str(tmp_path))
    assert report['unknown'] == []
    assert report['required_missing'] == []
    assert 'source.port' in report['unset']


def test_check_reports_a_missing_required_setting(tmp_path):
    (tmp_path / CONFIG_NAME).write_text(
        '[catchment]\nname = "c"\n', encoding='utf-8')
    report = check_study(str(tmp_path))
    assert 'catchment.boundary' in report['required_missing']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest fire_impacts/tests/test_study_scaffold.py -k check -v`
Expected: FAIL — `ImportError: cannot import name 'check_study'`

- [ ] **Step 3: Add `check_study` to `study.py`**

It must not raise on a bad file — `fire-impacts status` reports problems
rather than failing:

```python
###############################################################################
def check_study(path):
    """
    Report how a project's study.toml compares with the current schema.

    Parameters:
    - path: Project directory.

    Returns:
    - Dict with keys 'missing_file', 'unknown', 'required_missing' and
      'unset'.
    --------------------------------------------------------------------
    Notes:
    - Reports rather than raises: `fire-impacts status` exists to tell a
      user what is wrong, so it must survive a file that will not load.
    - A setting in 'unset' is information, not a problem. It is how a
      user finds out that newer templates understand something their
      file does not mention.
    --------------------------------------------------------------------
    """
    report = {
        'missing_file': False,
        'unknown': [],
        'required_missing': [],
        'unset': [],
        }

    fn = config_path(path)
    if not os.path.exists(fn):
        report['missing_file'] = True
        return report

    try:
        with open(fn, 'rb') as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        report['unknown'].append(f'{CONFIG_NAME} could not be read: {exc}')
        return report

    provided = _flatten_provided(data)
    blank = StudySettings()

    known_sections = {f.name for f in fields(StudySettings)}
    for section in data:
        if section not in known_sections:
            report['unknown'].append(
                f'[{section}] is not a known section.'
                f'{did_you_mean(section, known_sections)}')

    for group in fields(StudySettings):
        value = getattr(blank, group.name)
        names = {f.name for f in fields(value)}

        for key in data.get(group.name, {}):
            if key not in names:
                report['unknown'].append(
                    f'{group.name}.{key} is not a known setting.'
                    f'{did_you_mean(key, names)}')

        for entry in fields(value):
            dotted = f'{group.name}.{entry.name}'
            if dotted in provided:
                continue
            if entry.metadata.get('required'):
                report['required_missing'].append(dotted)
            else:
                report['unset'].append(dotted)

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study_scaffold.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/study.py fire_impacts/tests/test_study_scaffold.py
git commit -m "Report how a study.toml compares with the current schema

Reports rather than raises: `fire-impacts status` exists to say what is
wrong, so it has to survive a file that will not load. An unset setting
is information rather than a fault - it is how someone finds out their
file predates a setting the newer templates understand."
```

---

### Task 7: Wire the config into the CLI

**Files:**
- Modify: `fire_impacts/cli.py`
- Test: `fire_impacts/tests/test_study_cli.py`

**Interfaces:**
- Consumes: `write_scaffold`, `check_study`, `EXAMPLE_SEED`, `CONFIG_NAME` from Tasks 5 and 6; `nb.refresh_notebooks`, `nb.plan_update` from the existing `notebooks.py`.
- Produces: `_seed_from_project(path) -> dict` in `cli.py`, used only by `update`.

**Verify before writing `_seed_from_project`:** confirm the exact signatures
of `FireImpactsProject.catchments`, `.events(...)` and `.ensembles(...)` in
`fire_impacts/pre/project.py` — whether `events` and `ensembles` require a
catchment argument. Adjust the calls below to match what is actually there;
do not guess.

- [ ] **Step 1: Write the failing test**

Create `fire_impacts/tests/test_study_cli.py`:

```python
"""
`fire-impacts new` and `update` look after study.toml.

The rule that matters: the file is created when it is absent and never
touched when it is present. It is the one file in a project that holds
the user's own decisions in a form this package does not replace.
"""

import os

from typer.testing import CliRunner

from fire_impacts.cli import app
from fire_impacts.study import CONFIG_NAME

runner = CliRunner()


def test_new_writes_a_study_config(tmp_path):
    project = tmp_path / 'proj'
    result = runner.invoke(app, ['new', str(project), '--no-notebooks'])
    assert result.exit_code == 0, result.output
    assert (project / CONFIG_NAME).exists()
    assert CONFIG_NAME in result.output


def test_new_seeds_the_config_with_the_example(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    text = (project / CONFIG_NAME).read_text(encoding='utf-8')
    assert 'EgSmallCatchment_7899' in text
    assert '2019_fire' in text


def test_update_leaves_an_existing_config_alone(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    (project / CONFIG_NAME).write_text('# mine\n', encoding='utf-8')

    result = runner.invoke(app, ['update', str(project)])
    assert result.exit_code == 0, result.output
    assert (project / CONFIG_NAME).read_text(encoding='utf-8') == '# mine\n'


def test_update_creates_a_config_for_a_project_that_predates_it(tmp_path):
    """Such a project has notebooks but no config, and after the update its
    notebooks call load_study()."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    os.remove(project / CONFIG_NAME)

    result = runner.invoke(app, ['update', str(project)])
    assert (project / CONFIG_NAME).exists()
    assert CONFIG_NAME in result.output


def test_status_reports_a_typo_in_the_config(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    (project / CONFIG_NAME).write_text(
        '[catchment]\nnaem = "x"\n', encoding='utf-8')

    result = runner.invoke(app, ['status', str(project)])
    assert 'naem' in result.output
    assert "Did you mean 'name'" in result.output


def test_status_reports_a_missing_config(tmp_path):
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project), '--no-notebooks'])
    os.remove(project / CONFIG_NAME)

    result = runner.invoke(app, ['status', str(project)])
    assert CONFIG_NAME in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest fire_impacts/tests/test_study_cli.py -v`
Expected: FAIL — no `study.toml` is written

- [ ] **Step 3: Add the seeding helper to `cli.py`**

Add after `_summarise`, adjusting the API calls to match what Step 0's
verification found:

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
    --------------------------------------------------------------------
    """
    seed = {
        'catchment.boundary': None,
        'catchment.aridity': None,
        }

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

Add `from .study import CONFIG_NAME, check_study, config_path, write_scaffold, EXAMPLE_SEED, StudyConfigError`
to the imports at the top of `cli.py`.

- [ ] **Step 4: Write the config in `new`**

At the end of `new()`, after the notebook block:

```python
    try:
        written = write_scaffold(path, EXAMPLE_SEED)
        logger.info(
            'Wrote %s. This is where you set the catchment, the fire and '
            'the input files for your study - it is the only file you '
            'need to edit.', os.path.basename(written)
            )
    except StudyConfigError as e:
        logger.info('Left the existing configuration alone: %s', e)
```

- [ ] **Step 5: Write it in `update` only when absent, and report drift**

In `update()`, after the `refresh_notebooks` / `_summarise` calls:

```python
    if not dry_run and not os.path.exists(config_path(path)):
        # A project created before study.toml existed: its refreshed
        # notebooks will call load_study(), so it needs one. Seed it from
        # the project's own state rather than from the bundled example,
        # so the file describes the study actually in progress.
        seed = _seed_from_project(path)
        write_scaffold(path, seed)
        logger.warning(
            'This project had no %s, so one has been written from what '
            'could be read off the project itself. Open it and fill in '
            'the settings marked "could not be recovered" before running '
            'the notebooks.', CONFIG_NAME
            )
    elif not dry_run:
        report = check_study(path)
        if report['unset']:
            logger.info(
                'The updated notebooks understand %d setting(s) your %s '
                'does not mention: %s. All are optional and will use '
                'their defaults.',
                len(report['unset']), CONFIG_NAME,
                ', '.join(report['unset'])
                )
```

- [ ] **Step 6: Add the config section to `status`**

At the end of `status()`:

```python
    typer.echo('')
    report = check_study(path)

    if report['missing_file']:
        typer.echo(
            f'{CONFIG_NAME:<24} missing (run "fire-impacts update '
            f'{path}" to create one)'
            )
        return

    typer.echo(f'{CONFIG_NAME:<24} present')
    for problem in report['unknown']:
        typer.echo(f'  ! {problem}')
    for name in report['required_missing']:
        typer.echo(f'  ! {name} is required but not set')
    for name in report['unset']:
        typer.echo(f'  - {name} not set (uses the default)')
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest fire_impacts/tests/test_study_cli.py -v`
Expected: PASS (6 tests)

- [ ] **Step 8: Run the notebook tests, which share this CLI**

Run: `pytest fire_impacts/tests/test_notebooks.py -v`
Expected: PASS, unchanged

- [ ] **Step 9: Commit**

```bash
git add fire_impacts/cli.py fire_impacts/tests/test_study_cli.py
git commit -m "Create study.toml on new, and only when absent on update

A project created before the config existed gets one written on its next
update, because its refreshed notebooks will call load_study(). That file
is seeded from the project's own catchment, event and ensemble names
rather than from the bundled example, so it describes the study actually
in progress; what a project does not record - input paths, the API key -
is written blank and flagged.

An existing study.toml is never touched. It is the one file here that
holds the user's own decisions in a form this package does not replace."
```

---

### Task 8: Teach the template checker to follow attribute chains

Do this **before** editing any template, so the checker is in place to catch mistakes in Tasks 9-11.

**Files:**
- Modify: `fire_impacts/tests/test_template_currency.py`

**Interfaces:**
- Consumes: `StudySettings` from Task 2.
- Produces: nothing other tasks import; `_attribute_uses` now yields multi-hop chains.

- [ ] **Step 1: Write the failing test**

Add to `fire_impacts/tests/test_template_currency.py`:

```python
def test_the_checker_follows_a_chain_of_attributes(tmp_path, probes):
    """`study.catchment.name` must have BOTH hops checked. Before this,
    _attribute_uses required node.value to be an ast.Name, so only the
    first hop of any chain was verified and the rest went unchecked -
    which is exactly the shape every settings block uses."""
    script = tmp_path / 'Chained.py'
    script.write_text(
        'study = None\n'
        'x = study.catchment.no_such_setting\n',
        encoding='utf-8')

    found = [(name, attr) for _, name, attr, _, _
             in _attribute_uses(script)]
    assert ('study', 'catchment') in found
    assert any(attr == 'no_such_setting' for _, attr in found)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest fire_impacts/tests/test_template_currency.py::test_the_checker_follows_a_chain_of_attributes -v`
Expected: FAIL — only `('study', 'catchment')` is found; the second hop is missing

- [ ] **Step 3: Rewrite `_attribute_uses` to walk chains**

Replace the existing function with:

```python
def _attribute_uses(path):
    """Yield (lineno, receiver_name, attribute, owner, was_called) for
    every attribute access whose receiver can be resolved.

    Chains are followed hop by hop: in `record.parameters.delivery.max_sdr`
    the checker resolves `record.parameters`, then looks `delivery` up on
    whatever that returned, and so on. Requiring the receiver to be a bare
    ast.Name - as this did originally - checked only the first hop of a
    chain and silently passed the rest, which is the shape every settings
    block and much of the provenance walkthrough uses.
    """
    tree = ast.parse(path.read_text())
    called = {id(node.func) for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    modules = _module_receivers(tree)

    def root_owner(node):
        """Resolve the object a bare name refers to, or None."""
        if not isinstance(node, ast.Name):
            return None
        return modules.get(node.id) or RECEIVER_TYPES.get(node.id)

    def chain(node):
        """Return the list of attribute names from a bare name outwards,
        plus the bare name node, or None if the chain is not rooted in a
        resolvable name."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current)
            current = current.value
        if not isinstance(current, ast.Name):
            return None, None
        return current, list(reversed(parts))

    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if id(node) in seen:
            continue

        name_node, parts = chain(node)
        if name_node is None:
            continue
        owner = root_owner(name_node)
        if owner is None:
            continue

        # Walk outwards, carrying the resolved owner from hop to hop.
        # An unresolvable hop stops the walk rather than guessing: better
        # a gap than a false failure that trains people to ignore this.
        for part in parts:
            seen.add(id(part))
            yield (part.lineno, name_node.id, part.attr, owner,
                   id(part) in called)
            owner = _next_owner(owner, part.attr)
            if owner is None:
                break


def _next_owner(owner, attr):
    """Return the object an attribute yields, for the next hop, or None."""
    target = _PROBES.get(owner, owner) if _PROBES else owner
    value = getattr(target, attr, None)
    if value is None or callable(value):
        return None
    return type(value) if not is_dataclass_instance(value) else value


def is_dataclass_instance(obj):
    import dataclasses
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)
```

The chain walk needs the probe instances, which are currently built in a
fixture. Promote them to a module-level `_PROBES` dict populated by the
fixture so `_next_owner` can reach them:

```python
_PROBES = {}


@pytest.fixture(scope='module')
def probes(tmp_path_factory):
    """Real instances to resolve attributes against. (docstring unchanged)"""
    root = tmp_path_factory.mktemp('currency')
    project = FireImpactsProject(str(root / 'proj'), exist_ok=False)
    project.catchments.append('probe')
    _PROBES.update({
        FireImpactsProject: project,
        RunContext: RunContext(
            project=project, catchment='probe', event='e', ensemble='n'),
        ParameterRecord: resolve_parameters([]),
        RunProvenance: RunProvenance(
            run={}, parameters=resolve_parameters([]), inputs={},
            section='Results'),
        StudySettings: StudySettings(),
    })
    return _PROBES
```

- [ ] **Step 4: Register the study receiver**

Add the import and the entry:

```python
from fire_impacts.study import StudySettings
```

and in `RECEIVER_TYPES`:

```python
    'study': StudySettings,
```

- [ ] **Step 5: Run the checker tests**

Run: `pytest fire_impacts/tests/test_template_currency.py -v`
Expected: PASS. If the newly-checked later hops flag genuine drift in an
existing template (for example in the `record.parameters.*` or `prov.*`
walkthrough), that is a real find — fix the template, do not weaken the
checker.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/tests/test_template_currency.py
git commit -m "Check every hop of an attribute chain in a template

The checker required a receiver to be a bare name, so in
record.parameters.delivery.max_sdr only the first hop was verified. Every
line of the settings block the templates are about to gain has that
shape, so the gap had to close before the templates change.

StudySettings joins the receiver map, which is what ties a template's
settings block to the schema that validates study.toml."
```

---

### Task 9: Rewrite the PrepareData template

**Files:**
- Modify: `fire_impacts/templates/PrepareData.py`

**Interfaces:**
- Consumes: `load_study` from Task 2, the checker from Task 8.
- Produces: the settings-block convention the other three templates copy.

- [ ] **Step 1: Add the settings block**

Replace the existing `import os` cell (the first `# %%` cell) with a markdown
cell and a code cell:

```python
# %% [markdown]
# ## Settings for this study
#
# Every value below comes from `study.toml`, in this project folder.
# **That file is the only one you need to edit** to point this notebook at
# your own catchment and fire — the cells further down refer to the names
# defined here.
#
# Running the cell prints each setting and where it came from, so you can
# check the notebook is about to do what you expect.
#
# > Calibration parameters are a separate matter and live in
# > `parameters.json`; see the *Calibration parameters* section below.

# %%
import os

from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR = study.project.directory
CATCHMENT   = study.catchment.name
BOUNDARY    = study.catchment.boundary
DEM         = study.catchment.dem          # None -> download national DEM
ARIDITY     = study.catchment.aridity
EVENT       = study.event.name
FIRE_START  = study.event.fire_start
FIRE_END    = study.event.fire_end
BREAKPOINTS = study.event.recovery_breakpoints

# PrepareData builds a project from scratch, so it clears by default.
# Set clear = false in study.toml to keep data already in the project.
CLEAR = True if study.project.clear is None else study.project.clear
```

- [ ] **Step 2: Replace each hard-coded value in the body**

Make exactly these substitutions, leaving all other prose and cells intact:

| Find | Replace with |
|---|---|
| `proj = FireImpactsProject('.',clear=True)` | `proj = FireImpactsProject(PROJECT_DIR, clear=CLEAR)` |
| `example_catchment_name = 'EgSmallCatchment_7899'` | *(delete the line)* |
| `proj.add_catchment(f'..\\test_data\\{example_catchment_name}.shp')` | `proj.add_catchment(BOUNDARY)` |
| `optional_DEM_filename = '..\\test_data\\example_dem.tif'` | *(delete the line and its two comment lines)* |
| `topography.extract_catchment_dems(prep_ctx, None)` | `topography.extract_catchment_dems(prep_ctx, DEM)` |
| `proj.plot_headwaters(example_catchment_name)` | `proj.plot_headwaters(CATCHMENT)` |
| `fire_start_date = '2019-01-15'  # ...` | *(delete the line)* |
| `fire_end_date = '2019-03-07'    # ...` | *(delete the line)* |
| `ctx = RunContext.solo_event(proj, event='2019_fire')` | `ctx = RunContext.solo_event(proj, event=EVENT)` |
| `fire_start_date=fire_start_date,` | `fire_start_date=FIRE_START,` |
| `fire_end_date=fire_end_date,` | `fire_end_date=FIRE_END,` |
| `API_KEY = os.environ.get('TERN_API_KEY')` | `API_KEY = study.secret('tern_api_key')` |
| `ARIDITY=r'..\\test_data\\AridityPT_EgSmallCatchment_7899.tif'` | *(delete the line; `ARIDITY` is set in the block)* |
| `proj.set_catchment_parameter_overrides(\n#     example_catchment_name, ...` | `proj.set_catchment_parameter_overrides(\n#     CATCHMENT, ...` |
| `proj.plot_headwaters(example_catchment_name, colour_col='dNBR_mean', table=summary)` | `proj.plot_headwaters(CATCHMENT, colour_col='dNBR_mean', table=summary)` |

And in the `compute_adjusted_k_c` cell, replace the commented-out override
with a live argument:

```python
rusle.compute_adjusted_k_c(
    ctx,
    recovery_breakpoints=BREAKPOINTS,   # None -> the package default
)
```

- [ ] **Step 3: Update the prose that names the example**

Three passages assume the bundled example. Rewrite them to describe the
configured catchment instead:

- In *Catchment areas*, replace "Here, we add a small example catchment,
  from the `test_data` directory, but you can use your own." with:
  "Here we add the catchment named in `study.toml`. A freshly created
  project points at the small example catchment in `test_data`; change
  `catchment.boundary` to use your own."
- In *Digital Elevation Model (DEM)*, replace the "> **Option**: If you have
  a specific DEM..." note with: "> **Option**: set `catchment.dem` in
  `study.toml` to use your own DEM. Leave it out and the national DEM is
  downloaded. Make sure any DEM you supply covers the entire catchment."
- In *Fire Severity*, replace "> **Note**: The fire start and end dates here
  correspond to an actual fire that took place in this area." with:
  "> **Note**: the dates come from `event.fire_start` and `event.fire_end`
  in `study.toml`. Those shipped with a new project describe a real fire in
  the example catchment."
- In *Soils*, replace steps 3-5 of the TERN instructions (the Windows
  environment-variable walkthrough) with: "3. Paste the key into the
  `[secrets]` section of `study.toml`, as `tern_api_key`. If you would
  rather keep it out of that file, leave the setting blank and set a
  `TERN_API_KEY` environment variable instead — the notebook checks both."

- [ ] **Step 4: Verify the template still parses and every name resolves**

Run: `pytest fire_impacts/tests/test_template_currency.py -v`
Expected: PASS. A failure naming `study.<something>` means the block
references a setting the schema does not have — fix the template.

- [ ] **Step 5: Verify no example literals are left**

Run:

```bash
grep -n "EgSmallCatchment\|2019_fire\|2019-01-15\|2019-03-07\|test_data\|TERN_API_KEY\|example_catchment_name\|optional_DEM" fire_impacts/templates/PrepareData.py
```

Expected: matches only inside prose/markdown lines (those beginning `#`),
never in a code line. Any code-line match is a value that should have moved
to `study.toml`.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/templates/PrepareData.py
git commit -m "Drive PrepareData from study.toml

Every value a user has to supply now arrives through one block at the top
of the notebook, and the prose no longer assumes the bundled example.

This also fixes a live defect: optional_DEM_filename was assigned and then
never used - extract_catchment_dems() received None - so the notebook
downloaded the national DEM regardless of what was put there. The setting
now reaches the call."
```

---

### Task 10: Rewrite the Simulation template

**Files:**
- Modify: `fire_impacts/templates/Simulation.py`

**Interfaces:**
- Consumes: `load_study` from Task 2, the block convention from Task 9.

- [ ] **Step 1: Add the settings block**

After the existing logging/import cell, insert:

```python
# %% [markdown]
# ## Settings for this study
#
# These come from `study.toml`, the same file the PrepareData notebook
# read. Anything PrepareData already worked out — the fire dates, the
# recovery windows — is read back from the project rather than repeated
# here, so there is only ever one copy of it.

# %%
from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR = study.project.directory
CATCHMENT   = study.catchment.name
EVENT       = study.event.name
ENSEMBLE    = study.ensemble.name
N_REPLICATES = study.ensemble.num_replicates
REPLICATE    = study.ensemble.inspect_replicate
SUBCATCHMENTS = study.catchment.subcatchments
MEAN_ANNUAL_RAINFALL = study.ensemble.mean_annual_rainfall
AVERAGE_TEMPERATURE  = study.ensemble.average_temperature
```

- [ ] **Step 2: Replace each hard-coded value in the body**

| Find | Replace with |
|---|---|
| `proj = FireImpactsProject('.', exist_ok=True)` | `proj = FireImpactsProject(PROJECT_DIR, exist_ok=True)` |
| `catchment_name = proj.catchments[0]` | *(delete; `CATCHMENT` is set in the block)* |
| `proj.events(catchment_name)` | `proj.events(CATCHMENT)` |
| `proj.ensembles(catchment_name)` | `proj.ensembles(CATCHMENT)` |
| `ctx = RunContext.solo_run(\n    proj, event='2019_fire', ensemble='stochastic',\n)` | `ctx = RunContext.solo_run(\n    proj, event=EVENT, ensemble=ENSEMBLE, catchment=CATCHMENT,\n)` |
| `subcatch_path = '..\\test_data\\Subcatchments_EgSmall_7899.shp'` | *(delete the line)* |
| `#proj.add_subcatchments(catchment_name, subcatch_path)` | `if SUBCATCHMENTS:\n    proj.add_subcatchments(CATCHMENT, SUBCATCHMENTS)` |
| `num_replicates=10,` | `num_replicates=N_REPLICATES,` |
| `    # mean_annual_rainfall=600,   # mm  — optional` | `    mean_annual_rainfall=MEAN_ANNUAL_RAINFALL,   # None -> estimated` |
| `    # average_temperature=20,     # °C  — optional` | `    average_temperature=AVERAGE_TEMPERATURE,     # None -> estimated` |
| `rain_seq = rainfall_30min.rainfall[:,9].to_pandas()` | `rain_seq = rainfall_30min.rainfall[:, REPLICATE].to_pandas()` |
| `rain_intensity_seq = rainfall.rainfall[:,9].to_pandas()` | `rain_intensity_seq = rainfall.rainfall[:, REPLICATE].to_pandas()` |

Then replace every remaining `catchment_name` with `CATCHMENT` — it appears
in the `plot_subcatchments` and `plot_headwaters` calls throughout the
results sections. Check with:

```bash
grep -n "catchment_name" fire_impacts/templates/Simulation.py
```

Expected after the edit: no matches.

- [ ] **Step 3: Update the prose that assumed one catchment**

Replace "> **Note:** You can capture multiple study areas / catchments within
a single `FireImpactsProject` directory structure. Here we will assume you
have one, so we will take the first (presumed only)" with:
"> **Note:** a `FireImpactsProject` can hold several catchments. This
notebook works on the one named by `catchment.name` in `study.toml`."

**`None` is the intended contract for the optional climate statistics.**
Pass `MEAN_ANNUAL_RAINFALL` and `AVERAGE_TEMPERATURE` through to
`get_rainfall_replicates` as written above. If the current implementation
in `fire_impacts/stochastic/rainfall.py` does not accept `None` — if
omitting the argument is the only way to get the lat/lon estimate — then
**fix the library, not the template**: make `None` mean the same as
omitted, which is what the parameter already documents ("if omitted, the
backend service estimates them from the catchment's lat/lon"). A settings
file cannot express "omit this argument", so `None` has to carry that
meaning for the config to work at all.

If you make that change, add a test beside the existing rainfall tests:

```python
def test_none_climate_statistics_mean_estimate_from_the_catchment():
    """A study.toml cannot express 'omit this argument', so None has to
    mean what omitting it means."""
    # Assert that get_rainfall_replicates(..., mean_annual_rainfall=None)
    # sends the same request as get_rainfall_replicates(...) with the
    # argument left out. Mock the backend call and compare the payloads.
```

Commit that fix separately from the template change.

- [ ] **Step 4: Verify**

Run: `pytest fire_impacts/tests/test_template_currency.py -v`
Expected: PASS

Run:

```bash
grep -n "'2019_fire'\|'stochastic'\|test_data\|\[:,9\]\|\[:, 9\]" fire_impacts/templates/Simulation.py
```

Expected: no matches in code lines.

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/templates/Simulation.py
git commit -m "Drive Simulation from study.toml

The event and ensemble names no longer have to be typed into this
notebook to match what PrepareData used, and the replicate the examples
plot is a setting rather than a literal 9 in two places.

The fire dates stay where they were: read back from the project through
ctx.simulation_period(), not repeated in the config."
```

---

### Task 11: Rewrite the SimulationEnsemble and SourceIntegration templates

These two take the same short block, so they are one task.

**Files:**
- Modify: `fire_impacts/templates/SimulationEnsemble.py`
- Modify: `fire_impacts/templates/SourceIntegration.py`

**Interfaces:**
- Consumes: `load_study` from Task 2, the block convention from Task 9.

- [ ] **Step 1: Add the block to `SimulationEnsemble.py`**

After the logging/import cells:

```python
# %% [markdown]
# ## Settings for this study
#
# From `study.toml` — the same file the other notebooks read.

# %%
from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR  = study.project.directory
CATCHMENT    = study.catchment.name
EVENT        = study.event.name
ENSEMBLE     = study.ensemble.name
N_REPLICATES = study.ensemble.num_replicates
N_WORKERS    = study.ensemble.n_workers
SUBCATCHMENT_ID = study.catchment.subcatchment_id_field

# Per-cell results are reported per hectare, so the model needs to know how
# big a cell is. This follows the DEM: change catchment.cell_size_m if yours
# is not the 30 m of the national DEM.
CELL_AREA_HA = study.catchment.cell_size_m ** 2 / 10_000

EROSION_THRESHOLD_T_HA    = study.reporting.erosion_threshold_t_ha
DELIVERED_THRESHOLD_KG_HA = study.reporting.delivered_threshold_kg_ha
```

Then substitute:

| Find | Replace with |
|---|---|
| `proj = FireImpactsProject('.', exist_ok=True)` | `proj = FireImpactsProject(PROJECT_DIR, exist_ok=True)` |
| `CATCHMENT = proj.catchments[0]` | *(delete; set in the block)* |
| `    proj, event='2019_fire', ensemble='stochastic',` | `    proj, event=EVENT, ensemble=ENSEMBLE,` |
| `CELL_AREA_HA = 30 * 30 / 10_000` and its comment (line ~202) | *(delete; set in the block)* |
| `THRESHOLD_T_HA = 0.5` (line ~221) | *(delete; use `EROSION_THRESHOLD_T_HA` below)* |
| every later use of `THRESHOLD_T_HA` | `EROSION_THRESHOLD_T_HA` |
| `n_workers=min(N_REPLICATES, 10)` — **three sites**, lines ~172, ~189, ~272 | `n_workers=min(N_REPLICATES, N_WORKERS)` |
| the literal `500` as an exceedance threshold — **three sites**, lines ~388, ~398, ~407 | `DELIVERED_THRESHOLD_KG_HA` |
| the `500` appearing in **three plot titles**, lines ~390, ~400, ~410 | interpolate: `f'... {DELIVERED_THRESHOLD_KG_HA:g} kg/ha ...'` |
| `'SiteID'` (line ~322) | `SUBCATCHMENT_ID` |

Check each threshold site individually — `500` is a common enough literal
that a blind replace-all would be wrong. Confirm with:

```bash
grep -n "500\|THRESHOLD_T_HA\|SiteID\|n_workers\|CELL_AREA_HA" fire_impacts/templates/SimulationEnsemble.py
```

Expected afterwards: no bare literals, and `CELL_AREA_HA`, `THRESHOLD`-named
constants and `SiteID` appearing only in the settings block.

`N_REPLICATES` is already the name this template uses, so the existing
`num_replicates=N_REPLICATES` call needs no change — but **delete the
line that assigns `N_REPLICATES` a literal**, wherever it appears. Find it
with:

```bash
grep -n "N_REPLICATES *=" fire_impacts/templates/SimulationEnsemble.py
```

Expected afterwards: exactly one match, the one in the settings block.

Replace the prose "A `FireImpactsProject` can host multiple study
catchments; in this ..." with "This notebook works on the catchment named
by `catchment.name` in `study.toml`."

- [ ] **Step 2: Add the block to `SourceIntegration.py`**

After the import cells:

```python
# %% [markdown]
# ## Settings for this study
#
# From `study.toml`. `constituent` and `functional_unit` are optional —
# left unset, the notebook auto-detects them from the running Source
# model and shows you what it picked.

# %%
from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR = study.project.directory
CATCHMENT   = study.catchment.name
EVENT       = study.event.name
ENSEMBLE    = study.ensemble.name
SUBCATCHMENT_ID = study.catchment.subcatchment_id_field

PORT        = study.source.port
REPLICATE   = study.source.replicate
TIMESTEP    = study.source.timestep
DATE_FORMAT = study.source.date_format
OUTPUT_DIR  = study.source.output_dir
LOAD_ATTENUATION      = study.source.load_attenuation
MAXIMUM_CONCENTRATION = study.source.maximum_concentration

# Named once each. The notebook creates these data sources and then reads
# them back, so the two references have to agree.
TSS_SOURCE      = study.source.tss_data_source
RAINFALL_SOURCE = study.source.rainfall_data_source
```

Then substitute:

| Find | Replace with |
|---|---|
| `proj = FireImpactsProject('.', exist_ok=True)` | `proj = FireImpactsProject(PROJECT_DIR, exist_ok=True)` |
| `CATCHMENT = proj.catchments[0]` | *(delete; set in the block)* |
| `    proj, event='2019_fire', ensemble='stochastic',` | `    proj, event=EVENT, ensemble=ENSEMBLE,` |
| `PORT = 9876` | *(delete; set in the block)* |
| `REPLICATE = 0` (line ~214) | *(delete; set in the block)* |
| `CONSTITUENT = detect_constituent(v)` | `CONSTITUENT = study.source.constituent or detect_constituent(v)` |
| `FUNCTIONAL_UNIT = detect_functional_unit(v)` | `FUNCTIONAL_UNIT = study.source.functional_unit or detect_functional_unit(v)` |
| `load_ensemble_combined(ctx, freq='D')` (line ~131) | `load_ensemble_combined(ctx, freq=TIMESTEP)` |
| `load_attenuation=10.0` (line ~201) | `load_attenuation=LOAD_ATTENUATION` |
| `maximum_concentration=1000.0` (line ~202) | `maximum_concentration=MAXIMUM_CONCENTRATION` |
| `'fire_tss'` — lines ~227, ~233, ~293 | `TSS_SOURCE` |
| `'fire_tss.csv'` (line ~268) | `f'{TSS_SOURCE}.csv'` |
| `'stochastic_rain'` — lines ~228, ~236 | `RAINFALL_SOURCE` |
| **`'rainfall'` as a data-source name — lines ~302, ~327** | `RAINFALL_SOURCE` (see the defect note below) |
| `'rainfall.csv'` (line ~269) | `f'{RAINFALL_SOURCE}.csv'` |
| `write_replicate_csvs(0)` (line ~284) | `write_replicate_csvs(REPLICATE)` |
| `'%d/%m/%Y'` — lines ~243-244, ~344-345 | `DATE_FORMAT` |
| `'source_inputs'` (lines ~265-266) | `OUTPUT_DIR` |
| `'SiteID'`, if present | `SUBCATCHMENT_ID` |

**A defect this fixes, worth understanding before you edit.** The notebook
creates a rainfall data source called `'stochastic_rain'` (line ~236) but
Part A rewires from one called `'rainfall'` (lines ~302, ~327) — the same
notebook disagreeing with itself about the name of a thing it just made.
Routing both through `RAINFALL_SOURCE` makes them agree by construction.
Read both passages before replacing, and if they turn out to be two
genuinely different data sources rather than one misnamed, stop and say so
rather than merging them.

Also fix the saved-project filenames so they follow the configured
replicate:

| Find | Replace with |
|---|---|
| `f'{CATCHMENT}_with_fire_inputs_rep{REPLICATE:02d}.rsproj'` | *(unchanged — it already reads `REPLICATE`, which is now a setting)* |

And replace the commented-out override cell:

```python
# %%
# Override if the auto-detection picked the wrong option:
# CONSTITUENT = 'TSS'
# FUNCTIONAL_UNIT = 'Forested'
```

with:

```python
# %%
# Wrong pick? Set constituent / functional_unit in the [source] section
# of study.toml and re-run the cell above.
```

Update the prose above it: "`detect_constituent` and
`detect_functional_unit` pick the most likely candidates (`TSS`,
`Forested` etc.). Set `source.constituent` or `source.functional_unit` in
`study.toml` to override them."

- [ ] **Step 3: Verify both**

Run: `pytest fire_impacts/tests/test_template_currency.py -v`
Expected: PASS

Run:

```bash
grep -n "'2019_fire'\|'stochastic'\|9876\|proj.catchments\[0\]" fire_impacts/templates/SimulationEnsemble.py fire_impacts/templates/SourceIntegration.py
```

Expected: no matches in code lines.

- [ ] **Step 4: Commit**

```bash
git add fire_impacts/templates/SimulationEnsemble.py fire_impacts/templates/SourceIntegration.py
git commit -m "Drive the ensemble and Source notebooks from study.toml

The event and ensemble names were repeated in all four notebooks; this is
the last of them. The Veneer port, constituent and functional unit become
settings, with auto-detection still the default when they are left unset."
```

---

### Task 12: End-to-end check and documentation

**Files:**
- Modify: `README.md`
- Test: `fire_impacts/tests/test_study_cli.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing test**

Append to `fire_impacts/tests/test_study_cli.py`:

```python
def test_a_new_project_has_notebooks_that_can_find_their_config(tmp_path):
    """The whole point: `fire-impacts new` produces a project where the
    notebooks' first cell works."""
    project = tmp_path / 'proj'
    result = runner.invoke(app, ['new', str(project)])
    assert result.exit_code == 0, result.output

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        assert (project / f'{name}.py').exists()
    assert (project / CONFIG_NAME).exists()

    from fire_impacts.study import check_study
    report = check_study(str(project))
    assert report['unknown'] == []
    assert report['required_missing'] == []


def test_every_template_loads_the_study(tmp_path):
    """A template that forgot its settings block would silently keep its
    hard-coded example values."""
    project = tmp_path / 'proj'
    runner.invoke(app, ['new', str(project)])

    for name in ('PrepareData', 'Simulation', 'SimulationEnsemble',
                 'SourceIntegration'):
        text = (project / f'{name}.py').read_text(encoding='utf-8')
        assert 'from fire_impacts.study import load_study' in text, name
        assert "load_study('.')" in text, name
```

- [ ] **Step 2: Run the test**

Run: `pytest fire_impacts/tests/test_study_cli.py -v`
Expected: PASS. If `required_missing` is non-empty, the scaffold's
`EXAMPLE_SEED` does not cover every required setting — add the missing
entries to `EXAMPLE_SEED` in `study.py`.

- [ ] **Step 3: Run the whole suite**

Run: `pytest fire_impacts/tests/ -q`
Expected: PASS

- [ ] **Step 4: Document it in the README**

Add a section after the installation instructions and before the notebook
walkthrough:

```markdown
## Configuring a study

Each project holds one `study.toml` describing the study: which catchment,
which fire, where the input files are. It is written for you by
`fire-impacts new`, pre-filled with the bundled example so a new project
runs end to end before you change anything.

It is the only file you need to edit to point the notebooks at your own
data. The four notebooks read it, so a value like the event name is set
once rather than in each of them.

The convention in the file: **a setting that is not commented out is one
you must supply; a commented-out setting is optional, and the comment
says what happens if you leave it out.**

`fire-impacts status <project>` reports what your file sets, anything it
sets that is not a real setting (with a suggestion), and any settings the
current notebooks understand that it does not mention.

`fire-impacts update` never overwrites `study.toml`. If a project made
before the file existed does not have one, `update` writes it, filling in
the catchment, event and ensemble names it can read off the project and
flagging what it cannot.

> Calibration parameters are a separate matter and live in
> `parameters.json` — see [Calibration parameters](#calibration-parameters).
```

Check the anchor `#calibration-parameters` matches an existing heading in
`README.md`; adjust to whatever that heading actually is.

- [ ] **Step 5: Commit**

```bash
git add README.md fire_impacts/tests/test_study_cli.py
git commit -m "Document study.toml, and check a new project hangs together

The end-to-end test is the one that would have caught a template keeping
its hard-coded example values after everything else moved."
```

---

## Notes for the executor

- **The fact-check is done, and this plan incorporates it.** Six claims the
  plan rests on were verified against the code. Four confirmed. Two
  corrections are already folded in: `_hints` reads a module-level
  `_TYPE_HINTS` cache that must move with it (Task 1 Step 3), and the
  settings inventory was incomplete — `SiteID`, the DEM cell area, the
  reporting thresholds, the worker cap, the Source replicate, timestep,
  date format and Load Distributor calibration were all missed on the first
  reading and are now in the schema (Task 2) and the template edits (Task
  11). Confirmed incidentally: `examples/PrepareData.py` and
  `examples/Simulation.py` are independent stale copies, generated by
  nothing and collected by no test, so leaving them alone is safe.
- **Two defects to fix in passing, both found by that check.**
  `SourceIntegration.py` names its rainfall data source `'stochastic_rain'`
  in one place and `'rainfall'` in another (Task 11), and
  `PrepareData.py:310` writes the aridity path as a raw string with doubled
  backslashes (Task 9 deletes that line). Neither survives the move to
  settings.
- **Verify before you assume, in two places.** Task 7 Step 3 depends on the
  real signatures of `project.events()` / `project.ensembles()` — the
  fact-check found both take an *optional* catchment and that `catchments`
  is a plain list attribute, not a method, so the plan's calls are right,
  but confirm before relying on it. Task 10 Step 3 depends on whether
  `get_rainfall_replicates` accepts `None` for the optional climate
  statistics; that one was **not** checked, so check it rather than guess.
- **`fire_impacts/tests/test_template_currency.py` is the safety net for
  Tasks 9-11.** Run it after every template edit, not just at the end.
- **The `examples/` directory** holds a separate copy of two notebooks. It is
  outside this plan's scope; leave it alone unless the fact-check reports
  that it is generated from the templates, in which case regenerate it.
