# RUSLE Timestep Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a RUSLE run stop computing a full raster per timestep, by having the generator yield a lazy `ScaledGrid` that recorders reduce as a scalar and only turn into a grid at `finalize()`.

**Architecture:** `RUSLE(t, cell) = R(t) * klscp(cell) * cell_area` is separable. `generate_rusle` builds its six static "unit" layers once per recovery segment and yields `ScaledGrid(scale, unit)` per timestep. Scale-aware recorders accumulate `scale`; recorders that know nothing about the type materialise it transparently through a duck-array facade, so every task below leaves a working system.

**Tech Stack:** Python 3.13, numpy 2.4, pandas 2.3, xarray, rasterio, geopandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-rusle-timestep-collapse-design.md` — read it before Task 1. The plan argues from it.

## Global Constraints

- Line length 79 characters. The repo has no linter configured; check with the loop in "Verifying" below.
- Docstrings follow the existing house style in `fire_impacts/sim/rusle.py`: `Parameters:` / `Returns:` lists, and a `Notes:` block fenced by lines of hyphens.
- **Two tests fail on this Windows workstation on a clean checkout** and are not your problem: `fire_impacts/tests/test_integration_pipeline.py::test_default_outputs_are_unchanged` (path separators) and `fire_impacts/tests/test_notebooks.py::TestDetectingEdits::test_line_endings_are_not_an_edit` (CRLF). A full-suite run showing exactly these two and nothing else is a pass. Task 8 deliberately changes the first one's golden data.
- Numerical target: collapsed and eager paths agree to **1e-6 relative** with **identical NaN patterns**. Never assert bit-equality between the two paths.
- `unit >= 0` and `scale >= 0` are what make the `max` collapse valid. Never remove the guards in Task 6.
- Baseline for "did I make it faster": the default recorder set currently takes ~3.0 min per 5-year replicate on the WaterNSW case study.

## Verifying

Full suite: `python -m pytest fire_impacts -q`
Line length:
```bash
python -c "
import io,sys
for f in sys.argv[1:]:
    for i,l in enumerate(io.open(f,encoding='utf-8'),1):
        if len(l.rstrip(chr(10)))>79: print(f'{f}:{i}')
" fire_impacts/sim/scaled_grid.py fire_impacts/sim/recorders.py fire_impacts/sim/rusle.py
```

## File Structure

| File | Responsibility |
|---|---|
| `fire_impacts/sim/scaled_grid.py` | **Create.** `ScaledGrid` and `MaterialisationCounter`. No project imports — pure numpy. |
| `fire_impacts/sim/recorders.py` | **Create.** The four `record_*` builders, `default_rusle_recorders`, and the period helpers moved out of `rusle.py`. Imports `scaled_grid`, never `rusle`. |
| `fire_impacts/sim/rusle.py` | **Modify.** Keeps the model, parameter grids, runners and save helpers. Re-exports every name moved to `recorders.py`, including the private ones. |
| `fire_impacts/const.py` | **Modify.** Gains `MODEL_TIMESTEP`, so `rusle.py` and `recorders.py` share it without a cycle. |
| `fire_impacts/sim/tests/test_scaled_grid.py` | **Create.** The value type's contract. |
| `fire_impacts/sim/tests/test_dry_timesteps.py` | **Rewrite** in Task 3 — it pins the contract `ScaledGrid` replaces. |
| `fire_impacts/sim/tests/test_recorders.py` | **Modify.** Add scale-path equivalence tests. |
| `fire_impacts/sim/tests/test_subcatchment_timeseries.py` | **Modify.** Add scale-path, straddling and weight-matrix tests. |
| `fire_impacts/tests/test_integration_pipeline.py` | **Modify** in Task 8. Differential test, and regenerated golden hashes. |

---

### Task 1: Move the recorders into their own module

A pure refactor with no behaviour change, done first so later tasks edit a focused file. `rusle.py` is 2032 lines; this removes about 661.

**Files:**
- Create: `fire_impacts/sim/recorders.py`
- Modify: `fire_impacts/sim/rusle.py`
- Modify: `fire_impacts/const.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fire_impacts.sim.recorders` exporting `record_subcatchment_timeseries`, `record_grid_transform`, `record_timestep_grid`, `record_multi_period_grid`, `default_rusle_recorders`, `_compute_periods`, `_calendar_floor`, `_spatial_coords_from_transform`, `_PERIOD_OFFSETS`. `fire_impacts.const.MODEL_TIMESTEP` (a `pd.Timedelta` of 30 minutes).

- [ ] **Step 1: Record the current test baseline**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: `2 failed, 887 passed` — the two Windows failures named in Global Constraints. Write the number down; it must not change in this task.

- [ ] **Step 2: Add the shared constant**

In `fire_impacts/const.py`, append:

```python
# The simulation's fixed model timestep. Lives here so both rusle.py and
# recorders.py can reach it without importing each other.
MODEL_TIMESTEP = pd.Timedelta(minutes=30)
```

Add `import pandas as pd` at the top of `const.py` if it is not already there.

- [ ] **Step 3: Create recorders.py by moving code verbatim**

Cut these from `fire_impacts/sim/rusle.py` and paste them into a new
`fire_impacts/sim/recorders.py`, in this order, **unchanged**:

`_spatial_coords_from_transform`, `_PERIOD_OFFSETS`, `_calendar_floor`,
`_compute_periods`, `record_subcatchment_timeseries`,
`record_grid_transform`, `record_timestep_grid`,
`record_multi_period_grid`, `default_rusle_recorders`.

Give the new file this header:

```python
"""
Recorders: the closures that turn per-timestep simulation output into the
grids and time series a run actually keeps.

Each recorder is a callable taking (timestep, **data) with .reset() and
.finalize() attached. They are built fresh per run, so a replicate running
in parallel gets its own state.
"""

import logging

import numpy as np
import pandas as pd
import rasterio.features

from fire_impacts import const as c

logger = logging.getLogger(__name__)
```

Replace `_MODEL_TIMESTEP` inside `default_rusle_recorders` with
`c.MODEL_TIMESTEP`. This module must not import `rusle`.

- [ ] **Step 4: Re-export from rusle.py**

Near the top of `fire_impacts/sim/rusle.py`, after its existing imports:

```python
# Recorders moved to their own module; re-exported so existing imports
# (including the private helpers the tests reach for) keep working.
from fire_impacts.sim.recorders import (  # noqa: F401
    _calendar_floor,
    _compute_periods,
    _PERIOD_OFFSETS,
    _spatial_coords_from_transform,
    default_rusle_recorders,
    record_grid_transform,
    record_multi_period_grid,
    record_subcatchment_timeseries,
    record_timestep_grid,
)
```

Point `rusle.py`'s own `_MODEL_TIMESTEP` / `_MODEL_TIMESTEP_HOURS` at
`c.MODEL_TIMESTEP` rather than redefining the duration.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: identical to Step 1 — `2 failed, 887 passed`. Any other failure
means the move was not verbatim. In particular `test_periods.py` imports
`_calendar_floor` and `_compute_periods`, and `test_recorders.py` imports
`_spatial_coords_from_transform`, both from `fire_impacts.sim.rusle`.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/sim/recorders.py fire_impacts/sim/rusle.py fire_impacts/const.py
git commit -m "Move the recorders out of rusle.py"
```

---

### Task 2: The ScaledGrid value type

**Files:**
- Create: `fire_impacts/sim/scaled_grid.py`
- Test: `fire_impacts/sim/tests/test_scaled_grid.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MaterialisationCounter()` with a mutable `.count` int.
  - `ScaledGrid(scale, unit, rain_index=None, counter=None)` where `scale` is a 1-D float64 ndarray over rain cells, `unit` a 2-D ndarray, `rain_index` `None` or a 2-D int ndarray. Attributes `.scale`, `.unit`, `.rain_index`. Methods `.materialise()`, `.copy()`, `.reshape()`, `.ravel()`, `.astype()`. Properties `.shape`, `.ndim`, `.dtype`, `.flags`.

- [ ] **Step 1: Write the failing tests**

Create `fire_impacts/sim/tests/test_scaled_grid.py`:

```python
"""
The lazy grid the simulation hands to recorders: a static per-cell layer
times a per-timestep scale, materialised only when something insists.
"""

import numpy as np
import pytest

from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid


def unit_grid():
    u = np.arange(6, dtype=np.float32).reshape(2, 3)
    u[0, 0] = np.nan          # a cell outside the catchment
    return u


def scaled(value=2.0):
    return ScaledGrid(np.array([value]), unit_grid())


class TestMaterialise:

    def test_uniform_scale_multiplies_the_layer(self):
        assert np.allclose(
            scaled(2.0).materialise(), unit_grid() * 2.0, equal_nan=True)

    def test_a_scale_per_rain_cell_is_spread_over_the_grid(self):
        rain_index = np.array([[0, 0, 1], [0, 1, 1]])
        sg = ScaledGrid(np.array([2.0, 10.0]), unit_grid(), rain_index)
        expected = np.array([[np.nan, 2.0, 20.0], [6.0, 40.0, 50.0]])

        assert np.allclose(sg.materialise(), expected, equal_nan=True)

    def test_the_nan_mask_of_the_layer_survives(self):
        assert np.isnan(scaled().materialise()[0, 0])


class TestArrayFacade:
    """A recorder that knows nothing about the type must still work."""

    @pytest.mark.parametrize('operation', [
        lambda sg: np.asarray(sg),
        lambda sg: sg * 2.0,
        lambda sg: 2.0 * sg,
        lambda sg: sg * np.ones((2, 3)),
        lambda sg: np.ones((2, 3)) * sg,
        lambda sg: sg + sg,
        lambda sg: np.maximum(np.zeros((2, 3)), sg),
        lambda sg: np.where(np.ones((2, 3), dtype=bool), sg, 0),
        lambda sg: np.stack([sg, sg]),
        lambda sg: sg.reshape(-1),
        lambda sg: sg.ravel(),
        lambda sg: sg.astype(np.float32),
        lambda sg: sg[1],
    ])
    def test_numpy_operations_pass_through(self, operation):
        assert operation(scaled()) is not None

    def test_nansum_ignores_the_masked_cell(self):
        # 2 * (1 + 2 + 3 + 4 + 5), with cell [0, 0] masked out.
        assert np.nansum(scaled(2.0)) == pytest.approx(30.0)

    def test_reports_the_layer_shape_and_length(self):
        assert scaled().shape == (2, 3)
        assert scaled().ndim == 2
        assert len(scaled()) == 2

    def test_dtype_combines_scale_and_layer(self):
        assert scaled().dtype == np.float64

    def test_flags_are_readable(self):
        assert scaled().flags.writeable


class TestImmutability:

    def test_copy_gives_a_writable_array(self):
        c = scaled(2.0).copy()
        c += 1.0

        assert c[0, 1] == pytest.approx(3.0)

    def test_copying_does_not_disturb_the_original(self):
        sg = scaled(2.0)
        sg.copy()[1, 1] = 99.0

        assert sg.materialise()[1, 1] == pytest.approx(8.0)

    def test_writing_in_place_raises_rather_than_crashing(self):
        # NDArrayOperatorsMixin routes += through __array_ufunc__ with
        # out=(self,). Without a guard numpy writes into a temporary from
        # __array__ and segfaults the interpreter, so this must raise.
        sg = scaled()
        with pytest.raises(TypeError, match='immutable'):
            np.add(sg, 1.0, out=(sg,))


class TestMaterialisationCounter:

    def test_counts_every_materialisation(self):
        counter = MaterialisationCounter()
        sg = ScaledGrid(np.array([2.0]), unit_grid(), counter=counter)
        sg.materialise()
        np.asarray(sg)

        assert counter.count == 2

    def test_a_grid_without_a_counter_still_materialises(self):
        assert scaled().materialise() is not None

    def test_counters_are_independent(self):
        a, b = MaterialisationCounter(), MaterialisationCounter()
        ScaledGrid(np.array([1.0]), unit_grid(), counter=a).materialise()

        assert (a.count, b.count) == (1, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_scaled_grid.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'fire_impacts.sim.scaled_grid'`.

- [ ] **Step 3: Write the implementation**

Create `fire_impacts/sim/scaled_grid.py`:

```python
"""
A grid the simulation has not bothered to compute yet.

RUSLE is separable — erosion in a cell is the timestep's erosivity times a
static per-cell layer — so a simulation that materialised a full raster
every timestep was doing work that a recorder could do once, at the end,
from an accumulated scale. ScaledGrid is that deferral made explicit.

It is deliberately array-like: a recorder that knows nothing about the type
can multiply it, index it and reduce it exactly as before, and simply pays
for the materialisation.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class MaterialisationCounter:
    """
    A tally of how many grids a single run has been forced to compute.

    One per run rather than a module global: replicates run concurrently on
    threads by default, and a shared tally would mix them together.
    """

    __slots__ = ('count',)

    def __init__(self):
        self.count = 0


class ScaledGrid(NDArrayOperatorsMixin):
    """
    A static per-cell layer times a per-timestep scale.

    Parameters:
    - scale: 1-D float array with one entry per rainfall cell. Length 1
      when rainfall is spatially uniform.
    - unit: 2-D per-cell layer, constant for a whole recovery segment.
    - rain_index: None when rainfall is uniform; otherwise a 2-D integer
      array giving each grid cell's rainfall cell.
    - counter: optional MaterialisationCounter to tally against.
    ------------------------------------------------------------------------
    Notes:
    - Immutable. `copy()` returns a materialised, writable ndarray, which
      is what a caller asking for a copy actually wants.
    - Writing in place raises TypeError. Allowing it would segfault: the
      operator mixin routes `+=` through __array_ufunc__ with out=(self,),
      and numpy would write into a temporary built by __array__.
    ------------------------------------------------------------------------
    """

    __slots__ = ('scale', 'unit', 'rain_index', 'counter')

    def __init__(self, scale, unit, rain_index=None, counter=None):
        self.scale = scale
        self.unit = unit
        self.rain_index = rain_index
        self.counter = counter

    def materialise(self):
        """Compute the full grid, and tally that we had to."""
        if self.counter is not None:
            self.counter.count += 1
        if self.rain_index is None:
            return self.scale[0] * self.unit
        return self.scale[self.rain_index] * self.unit

    # -- numpy interoperability --------------------------------------------

    def __array__(self, dtype=None, copy=None):
        out = self.materialise()
        return out if dtype is None else out.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for target in kwargs.get('out', ()):
            if isinstance(target, ScaledGrid):
                raise TypeError(
                    'ScaledGrid is immutable; call .copy() for a '
                    'writable array'
                )
        inputs = tuple(
            i.materialise() if isinstance(i, ScaledGrid) else i
            for i in inputs
        )
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, key):
        return self.materialise()[key]

    def __len__(self):
        return len(self.unit)

    # -- the parts of the ndarray surface recorders actually touch ---------

    @property
    def shape(self):
        return self.unit.shape

    @property
    def ndim(self):
        return self.unit.ndim

    @property
    def dtype(self):
        return np.result_type(self.scale, self.unit)

    @property
    def flags(self):
        return self.materialise().flags

    def copy(self):
        return self.materialise()

    def reshape(self, *args, **kwargs):
        return self.materialise().reshape(*args, **kwargs)

    def ravel(self):
        return self.materialise().ravel()

    def astype(self, dtype):
        return self.materialise().astype(dtype)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_scaled_grid.py -q`
Expected: all pass, **and the process exits 0**. A segfault here means the
`out=` guard is missing; check the exit code explicitly with
`echo "exit=$?"`.

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/sim/scaled_grid.py fire_impacts/sim/tests/test_scaled_grid.py
git commit -m "Add ScaledGrid, a grid the simulation defers computing"
```

---

### Task 3: generate_rusle yields ScaledGrid

After this task everything still works and nothing is faster yet — the
existing recorders materialise transparently. That is the point: it isolates
the generator change from the recorder changes.

**Files:**
- Modify: `fire_impacts/sim/rusle.py` — `generate_rusle`
- Rewrite: `fire_impacts/sim/tests/test_dry_timesteps.py`

**Interfaces:**
- Consumes: `ScaledGrid`, `MaterialisationCounter` from Task 2.
- Produces: `generate_rusle(rainfall, klscp, sdr, dnbr, cell_area_ha, erosion=None, materialise_grids=False, counter=None)`. Yields `(timestep, data)` where the six grid keys are `ScaledGrid` unless `materialise_grids=True`, and `data['dry']` is a bool.

- [ ] **Step 1: Rewrite test_dry_timesteps.py**

Replace `fire_impacts/sim/tests/test_dry_timesteps.py` entirely. Keep the
module docstring's framing but specify the new contract:

```python
"""
What generate_rusle yields per timestep.

Erosion in a cell is the timestep's erosivity times a static per-cell
layer, so the generator yields that pair rather than the product: one set
of layers per recovery segment, and a scale per timestep. Around three
quarters of timesteps are dry, and for those the scale is simply zero.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts.sim.rusle import generate_rusle, rainfall_erosivity
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid

SHAPE = (2, 3)
CELL_AREA_HA = 0.09

GRID_KEYS = (
    'RUSLE', 'delivered',
    'RUSLE_below_threshold', 'RUSLE_above_threshold',
    'delivered_below_threshold', 'delivered_above_threshold',
)


@pytest.fixture()
def grids():
    klscp = np.full(SHAPE, 2.0, dtype=np.float32)
    sdr = np.full(SHAPE, 0.5, dtype=np.float32)
    dnbr = np.array([[0.0, 0.0, 0.0], [900.0, 900.0, 900.0]],
                    dtype=np.float32)
    return klscp, sdr, dnbr


def run(rain_mm, grids, **kwargs):
    rain = pd.Series(
        rain_mm,
        index=pd.date_range('2020-01-01', periods=len(rain_mm), freq='30min'),
    )
    return list(generate_rusle(rain, *grids, CELL_AREA_HA, **kwargs))


class TestWhatIsYielded:

    def test_grids_are_deferred(self, grids):
        _, data = run([5.0], grids)[0]
        assert all(isinstance(data[key], ScaledGrid) for key in GRID_KEYS)

    def test_the_layers_are_shared_across_timesteps(self, grids):
        steps = run([5.0, 0.0, 7.0], grids)
        units = [data['RUSLE'].unit for _, data in steps]
        assert units[0] is units[1] is units[2]

    def test_each_timestep_has_its_own_scale(self, grids):
        steps = run([5.0, 7.0], grids)
        assert steps[0][1]['RUSLE'].scale != steps[1][1]['RUSLE'].scale

    def test_rainfall_is_uniform_so_there_is_one_rain_cell(self, grids):
        _, data = run([5.0], grids)[0]
        assert data['RUSLE'].rain_index is None
        assert data['RUSLE'].scale.shape == (1,)


class TestDryTimesteps:

    def test_timesteps_are_flagged_dry_when_there_is_no_rain(self, grids):
        steps = run([0.0, 5.0, 0.0], grids)
        assert [d['dry'] for _, d in steps] == [True, False, True]

    def test_a_dry_timestep_scales_the_layers_by_zero(self, grids):
        _, data = run([0.0], grids)[0]
        assert data['RUSLE'].scale[0] == 0.0

    def test_a_dry_timestep_erodes_nothing(self, grids):
        _, data = run([0.0], grids)[0]
        for key in GRID_KEYS:
            assert np.array_equal(np.asarray(data[key]), np.zeros(SHAPE))

    def test_a_dry_timestep_reports_no_intensity_or_erosivity(self, grids):
        _, data = run([0.0], grids)[0]
        assert (data['total_rain'], data['intensity'],
                data['erosivity']) == (0.0, 0.0, 0.0)


class TestValuesMatchTheModel:

    def test_erosion_is_erosivity_by_klscp_by_cell_area(self, grids):
        klscp, sdr, _ = grids
        _, data = run([5.0], grids)[0]
        intensity, R = rainfall_erosivity(5.0)

        assert data['intensity'] == intensity
        assert data['erosivity'] == R
        assert np.allclose(data['RUSLE'], R * klscp * CELL_AREA_HA)
        assert np.allclose(data['delivered'],
                           np.asarray(data['RUSLE']) * sdr)

    def test_severity_thresholds_split_the_grid(self, grids):
        _, data = run([5.0], grids)[0]
        below = np.asarray(data['RUSLE_below_threshold'])
        above = np.asarray(data['RUSLE_above_threshold'])

        assert np.allclose(above[0], 0.0)
        assert np.allclose(below[1], 0.0)
        assert np.allclose(below + above, np.asarray(data['RUSLE']))


class TestEagerMode:

    def test_materialise_grids_yields_plain_arrays(self, grids):
        _, data = run([5.0], grids, materialise_grids=True)[0]
        assert all(isinstance(data[key], np.ndarray) for key in GRID_KEYS)

    def test_eager_and_deferred_agree(self, grids):
        _, lazy = run([5.0], grids)[0]
        _, eager = run([5.0], grids, materialise_grids=True)[0]
        for key in GRID_KEYS:
            assert np.allclose(np.asarray(lazy[key]), eager[key])


class TestCounter:

    def test_deferred_grids_carry_the_run_counter(self, grids):
        counter = MaterialisationCounter()
        _, data = run([5.0], grids, counter=counter)[0]
        assert counter.count == 0

        np.asarray(data['RUSLE'])
        assert counter.count == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_dry_timesteps.py -q`
Expected: failures in `TestWhatIsYielded`, `TestEagerMode` and
`TestCounter` — `isinstance(..., ScaledGrid)` is False, and
`generate_rusle` has no `materialise_grids` or `counter` parameter
(`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Rewrite the generator's body**

In `fire_impacts/sim/rusle.py`, add the import:

```python
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid
```

`MaterialisationCounter` is unused until Task 7, but importing it here
means `rusle.MaterialisationCounter` resolves, which Tasks 7 and 8 rely
on.

Extend the signature to
`generate_rusle(rainfall, klscp, sdr, dnbr, cell_area_ha, erosion=None, materialise_grids=False, counter=None)`.

Replace the body from the severity-mask precompute down to the end of the
loop with:

```python
    # The static half of the model. Erosion in a cell is this layer times
    # the timestep's erosivity, so it is built once per segment rather
    # than rebuilt every timestep.
    below = dnbr < erosion.dnbr_severity_threshold
    above = dnbr >= erosion.dnbr_severity_threshold
    unit_rusle = klscp * cell_area_ha
    unit_delivered = unit_rusle * sdr
    units = {
        'RUSLE': unit_rusle,
        'delivered': unit_delivered,
        'RUSLE_below_threshold': np.where(below, unit_rusle, 0),
        'RUSLE_above_threshold': np.where(above, unit_rusle, 0),
        'delivered_below_threshold': np.where(below, unit_delivered, 0),
        'delivered_above_threshold': np.where(above, unit_delivered, 0),
    }

    total_timesteps = len(rainfall.index)
    start_time = time.time()
    last_log_time = start_time
    iteration_count = 0

    for timestep, delta_v_r in zip(rainfall.index, rainfall.values):
        iteration_count += 1

        if delta_v_r == 0:
            intensity = 0.0
            R = 0.0
        else:
            intensity, R = rainfall_erosivity(
                delta_v_r, rate=erosion.kinetic_energy_coefficient)

        # One scale array shared by all six grids of this timestep. It is
        # an array, not a float, so that spatially varying rainfall widens
        # it without any accumulator needing a second code path.
        scale = np.array([R], dtype=np.float64)
        result = {
            'total_rain': delta_v_r,
            'intensity': intensity,
            'erosivity': R,
            'dry': bool(delta_v_r == 0),
        }
        for key, unit in units.items():
            if materialise_grids:
                result[key] = scale[0] * unit
            else:
                result[key] = ScaledGrid(scale, unit, counter=counter)

        current_time = time.time()
        if current_time - last_log_time >= LOG_INTERVAL_SECONDS:
            _log_progress(iteration_count, total_timesteps, start_time,
                          timestep)
            last_log_time = current_time

        yield (timestep, result)
```

Extract the existing progress-logging block verbatim into a module-level
`_log_progress(iteration_count, total_timesteps, start_time, timestep)`
helper — it is unchanged, only moved, and the loop body is long enough
already.

Update the docstring: the six grid keys are now "`ScaledGrid`, or a plain
float array when `materialise_grids` is set", document the two new
parameters, and delete the note about a shared read-only dry buffer.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_dry_timesteps.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: still `2 failed, 887 passed` plus the new tests. The unchanged
recorders now materialise transparently, so nothing else moves. If
`test_integration_pipeline.py::test_default_outputs_are_unchanged` grows a
*different* failure message than the path-separator one, stop — the model's
numbers have moved and they should not have yet.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/sim/rusle.py fire_impacts/sim/tests/test_dry_timesteps.py
git commit -m "Yield deferred grids from generate_rusle"
```

---

### Task 4: Teach record_multi_period_grid to accumulate scales

**Files:**
- Modify: `fire_impacts/sim/recorders.py` — `record_multi_period_grid`
- Test: `fire_impacts/sim/tests/test_recorders.py`

**Interfaces:**
- Consumes: `ScaledGrid` from Task 2; the `dry` flag from Task 3.
- Produces: `record_multi_period_grid(variable, fn, periods)` unchanged in
  signature. Accepts `ScaledGrid` or ndarray per timestep.

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/sim/tests/test_recorders.py`. Add
`from fire_impacts.sim.scaled_grid import ScaledGrid` to its imports.

```python
class TestMultiPeriodGridDeferredInput:
    """
    The recorder accumulates the scale and touches a grid once, at the
    end. Every result must match what it would have produced from the
    materialised grids.
    """

    @staticmethod
    def deferred(value, unit):
        return ScaledGrid(np.array([float(value)]), unit)

    def test_sum_matches_the_materialised_path(self):
        unit = grid(2.0)
        lazy = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        eager = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        for stamp, scale in [(TS('2019-03-01'), 1.0), (TS('2019-09-01'), 3.0)]:
            lazy(stamp, RUSLE=self.deferred(scale, unit), transform=TRANSFORM)
            eager(stamp, RUSLE=scale * unit, transform=TRANSFORM)

        assert np.allclose(lazy.finalize().values, eager.finalize().values)
        assert np.allclose(lazy.finalize().values, 8.0)

    def test_max_matches_the_materialised_path(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(5.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(2.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 10.0)

    def test_mean_divides_by_the_period_timestep_count(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(1.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(3.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 4.0)

    def test_a_new_layer_starts_a_new_accumulator(self):
        # A recovery boundary hands the recorder a different unit layer.
        # Both contributions must reach the sum.
        first, second = grid(2.0), grid(10.0)
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(1.0, first),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(1.0, second),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 12.0)

    def test_max_across_two_layers_takes_the_larger_product(self):
        first, second = grid(2.0), grid(10.0)
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(5.0, first),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(1.0, second),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 10.0)

    def test_dry_timesteps_contribute_nothing_but_still_count(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(4.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(0.0, unit),
            transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 4.0)

    def test_a_dry_period_still_finalises_to_plain_zeros(self):
        # A dry timestep registers no layer at all, so this period is
        # indistinguishable from one that recorded nothing - and keeps
        # today's NaN-free zeros rather than inheriting a layer's mask.
        unit = grid(2.0)
        unit[0, 0] = np.nan
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(0.0, unit),
            transform=TRANSFORM, dry=True)
        result = rec.finalize()

        assert result.shape == (2, 3)
        assert not np.isnan(result.values).any()
        assert np.allclose(result.values, 0.0)

    def test_a_period_with_no_timesteps_does_not_divide_by_zero(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', TWO_PERIODS)
        rec(TS('2019-06-01'), RUSLE=self.deferred(4.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().isel(time=1).values, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_recorders.py -q -k DeferredInput`
Expected: several failures. `test_a_new_layer_starts_a_new_accumulator`
gives 4.0 instead of 12.0 — the current code materialises and accumulates
grids, which happens to be right here, so the ones that genuinely fail are
those where the deferred path must differ. If a test passes at this step,
confirm why before moving on: the facade makes the old code work, just
slowly.

- [ ] **Step 3: Rewrite the recorder**

Replace the state and body of `record_multi_period_grid` in
`fire_impacts/sim/recorders.py`:

```python
    # Per period: an eager grid accumulator, plus one (unit, rain_index,
    # scale) entry for each distinct layer the period has seen. There is
    # one layer per recovery segment, so the list stays tiny.
    grids = [None] * len(periods)
    scales = [[] for _ in periods]
    counts = [0] * len(periods)
    captured_transform = [None]
    captured_shape = [None]

    def _accumulate_scale(index, data):
        for entry in scales[index]:
            if entry[0] is data.unit:
                if fn == 'max':
                    np.maximum(entry[2], data.scale, out=entry[2])
                else:
                    entry[2] += data.scale
                return
        scales[index].append(
            [data.unit, data.rain_index, data.scale.astype(np.float64)])

    def _accumulate_grid(index, data):
        if grids[index] is None:
            grids[index] = data.copy()
        elif fn == 'max':
            np.maximum(grids[index], data, out=grids[index])
        else:
            grids[index] += data

    def recorder(timestep, **kwargs):
        data = kwargs[variable]
        dry = kwargs.get('dry', False)
        if captured_transform[0] is None and 'transform' in kwargs:
            captured_transform[0] = kwargs['transform']
        if captured_shape[0] is None:
            captured_shape[0] = data.shape
        for i, (ps, pe) in enumerate(periods):
            if timestep < ps or timestep > pe:
                continue
            counts[i] += 1
            # A dry timestep erodes nothing. Registering its layer anyway
            # would widen the finalised NaN mask to the union of every
            # segment's, rather than only those that contributed.
            if dry:
                continue
            if isinstance(data, ScaledGrid):
                _accumulate_scale(i, data)
            else:
                _accumulate_grid(i, data)

    def reset():
        for i in range(len(periods)):
            grids[i] = None
            scales[i] = []
            counts[i] = 0
        captured_transform[0] = None
        captured_shape[0] = None
```

Add a module-level helper and rewrite `finalize`'s per-period combination:

```python
def _spread(scale, rain_index, unit):
    """Turn an accumulated scale back into a grid."""
    if rain_index is None:
        return scale[0] * unit
    return scale[rain_index] * unit
```

```python
    def _combine(index):
        """Fold this period's accumulators into one grid, or None."""
        parts = []
        if grids[index] is not None:
            parts.append(grids[index])
        parts.extend(
            _spread(scale, rain_index, unit)
            for unit, rain_index, scale in scales[index]
        )
        if not parts:
            return None
        out = parts[0].copy()
        for part in parts[1:]:
            if fn == 'max':
                np.maximum(out, part, out=out)
            else:
                out += part
        if fn == 'mean' and counts[index] > 0:
            out = out / counts[index]
        return out
```

`finalize` then builds `arrays = [_combine(i) for i in range(len(periods))]`,
takes `shape` from the first non-None entry falling back to
`captured_shape[0]`, returns `None` if both are absent, and substitutes
`np.zeros(shape, dtype=np.float32)` for each `None` — exactly as today.
Import `ScaledGrid` at the top of `recorders.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_recorders.py -q`
Expected: all pass, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/sim/recorders.py fire_impacts/sim/tests/test_recorders.py
git commit -m "Accumulate scales, not grids, in the period recorder"
```

---

### Task 5: Teach record_subcatchment_timeseries to accumulate scales

**Files:**
- Modify: `fire_impacts/sim/recorders.py` — `record_subcatchment_timeseries`
- Test: `fire_impacts/sim/tests/test_subcatchment_timeseries.py`

**Interfaces:**
- Consumes: `ScaledGrid`, `_spread` from Task 4.
- Produces: `record_subcatchment_timeseries(ctx, variable_name, fn='sum', label_field=None, agg_count=1)` unchanged in signature.

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/sim/tests/test_subcatchment_timeseries.py`, adding
`from fire_impacts.sim.scaled_grid import ScaledGrid` to its imports.

```python
def deferred(value, unit):
    return ScaledGrid(np.array([float(value)]), unit)


def layer(value=1.0):
    return np.full(SHAPE, value, dtype=np.float32)


class TestDeferredInput:

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_matches_the_materialised_path(self, fn):
        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        lazy, eager = recorder(fn=fn, agg_count=2), recorder(fn=fn,
                                                            agg_count=2)
        t = stamps(2)
        for stamp, scale in zip(t, [1.0, 3.0]):
            lazy(stamp, catchment='Catchment', transform=TRANSFORM,
                 RUSLE=deferred(scale, unit))
            eager(stamp, catchment='Catchment', transform=TRANSFORM,
                  RUSLE=scale * unit)

        assert np.allclose(lazy.finalize().to_numpy(),
                           eager.finalize().to_numpy())

    def test_cells_outside_the_catchment_stay_excluded(self):
        unit = layer(2.0)
        unit[:, 0] = np.nan
        rec = recorder(agg_count=1)
        feed_deferred(rec, [(stamps(1)[0], 3.0, unit)])
        row = rec.finalize()

        # 4 cells per zone survive on the left, 8 on the right.
        assert row['left'].iloc[0] == pytest.approx(24.0)
        assert row['right'].iloc[0] == pytest.approx(48.0)

    def test_dry_timesteps_add_nothing_but_advance_the_window(self):
        unit = layer(1.0)
        rec = recorder(agg_count=2)
        t = stamps(4)
        rec(t[0], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(0.0, unit), dry=True)
        rec(t[1], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(2.0, unit))
        rec(t[2], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(2.0, unit))
        rec(t[3], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(0.0, unit), dry=True)
        result = rec.finalize()

        assert list(result.index) == [t[1], t[3]]
        assert list(result['left']) == pytest.approx([16.0, 16.0])

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_an_all_dry_window_records_zero_not_nan(self, fn):
        # W is NaN for a zone whose cells are all NaN, so multiplying a
        # zero scale through it would put a silent NaN in the column.
        unit = layer(1.0)
        unit[:] = np.nan
        rec = recorder(fn=fn, agg_count=2)
        t = stamps(2)
        for stamp in t:
            rec(stamp, catchment='Catchment', transform=TRANSFORM,
                RUSLE=deferred(0.0, unit), dry=True)

        assert rec.finalize()['left'].iloc[0] == pytest.approx(0.0)


class TestWindowStraddlingARecoveryBoundary:

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_a_window_spanning_two_layers_is_exact(self, fn):
        first = np.arange(16, dtype=np.float32).reshape(SHAPE)
        second = first * 10.0
        lazy, eager = recorder(fn=fn, agg_count=2), recorder(fn=fn,
                                                            agg_count=2)
        t = stamps(2)
        lazy(t[0], catchment='Catchment', transform=TRANSFORM,
             RUSLE=deferred(2.0, first))
        lazy(t[1], catchment='Catchment', transform=TRANSFORM,
             RUSLE=deferred(3.0, second))
        eager(t[0], catchment='Catchment', transform=TRANSFORM,
              RUSLE=2.0 * first)
        eager(t[1], catchment='Catchment', transform=TRANSFORM,
              RUSLE=3.0 * second)

        assert np.allclose(lazy.finalize().to_numpy(),
                           eager.finalize().to_numpy())


class TestSpatiallyVaryingRainfall:
    """
    The seam for rainfall coarser than the DEM. Nothing produces this yet;
    these tests are what stop the weight matrix rotting before it does.
    """

    RAIN_INDEX = np.array([[0, 0, 1, 1]] * 4)

    def brute_force(self, scales, unit, fn):
        """Zonal aggregation the slow, obviously-correct way."""
        total = None
        for scale in scales:
            step = scale[self.RAIN_INDEX] * unit
            total = step if total is None else total + step
        flat = total.reshape(-1)
        agg = {'sum': np.nansum, 'mean': np.nanmean, 'max': np.nanmax}[fn]
        return [agg(flat[z]) for z in zone_positions()]

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_matches_a_brute_force_loop(self, fn):
        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        scales = [np.array([2.0, 10.0]), np.array([1.0, 4.0])]
        rec = recorder(fn=fn, agg_count=2)
        for stamp, scale in zip(stamps(2), scales):
            rec(stamp, catchment='Catchment', transform=TRANSFORM,
                RUSLE=ScaledGrid(scale, unit, self.RAIN_INDEX))

        assert np.allclose(rec.finalize().to_numpy()[0],
                           self.brute_force(scales, unit, fn))
```

Add two helpers near the top of the file:

```python
def feed_deferred(rec, samples):
    """Push (timestep, scale, unit) triples through as deferred grids."""
    for timestep, scale, unit in samples:
        rec(timestep, catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(scale, unit))


def zone_positions():
    """Flat cell positions of each subcatchment, for brute-force checks."""
    import rasterio.features
    return [
        np.flatnonzero(~np.isnan(rasterio.features.rasterize(
            [g], transform=TRANSFORM, fill=np.nan, dtype=np.float32,
            out_shape=SHAPE)))
        for g in SUBCATCHMENTS.geometry
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_subcatchment_timeseries.py -q -k "Deferred or Straddling or Spatially"`
Expected: `TestSpatiallyVaryingRainfall` fails — the current code
materialises a full grid and reduces it, which is correct but proves
nothing about the weight matrix, so check the *count* of failures rather
than assuming. `test_an_all_dry_window_records_zero_not_nan` should pass
already (the existing short-circuit), and must keep passing.

- [ ] **Step 3: Rewrite the recorder**

In `fire_impacts/sim/recorders.py`, add a module-level weight builder:

```python
def _zonal_weights(unit, rain_index, zone_positions):
    """
    Reduce a static layer to per-(rain cell, subcatchment) weights.

    Parameters:
    - unit: the static per-cell layer.
    - rain_index: None for uniform rainfall, else each cell's rain cell.
    - zone_positions: flat cell positions of each subcatchment.

    Returns:
    - Tuple of (sums, counts, maxima). `sums` has shape
      (n_rain_cells, n_zones); `counts` and `maxima` have one entry per
      zone, and `maxima` is only meaningful for uniform rainfall.
    ------------------------------------------------------------------------
    Notes:
    - A zonal sum and mean are linear in the scale vector, so they fold
      into a matrix product. A maximum does not, which is why the caller
      falls back to materialising when there is more than one rain cell.
    ------------------------------------------------------------------------
    """
    flat = unit.reshape(-1)
    n_rain = 1 if rain_index is None else int(rain_index.max()) + 1
    n_zones = len(zone_positions)
    sums = np.zeros((n_rain, n_zones))
    counts = np.zeros(n_zones)
    maxima = np.full(n_zones, np.nan)

    flat_rain = None if rain_index is None else rain_index.reshape(-1)
    for z, positions in enumerate(zone_positions):
        values = flat[positions]
        keep = ~np.isnan(values)
        counts[z] = keep.sum()
        if counts[z]:
            maxima[z] = values[keep].max()
        if rain_index is None:
            sums[0, z] = values[keep].sum()
        else:
            sums[:, z] = np.bincount(
                flat_rain[positions][keep], weights=values[keep],
                minlength=n_rain)
    return sums, counts, maxima
```

Replace the recorder's accumulation state with a `window` list of
`[unit, rain_index, scale]` entries plus the existing eager `intermediate`,
and a `weights` list of `(unit, (sums, counts, maxima))` pairs matched by
identity:

```python
        def _weights_for(unit, rain_index):
            """Zonal weights for this layer, built once per segment."""
            for cached_unit, cached in weights:
                if cached_unit is unit:
                    return cached
            cached = _zonal_weights(unit, rain_index, zone_indices)
            weights.append((unit, cached))
            return cached
```

On each call:

```python
        intermediate_count += 1
        if not dry:
            if isinstance(data, ScaledGrid):
                for entry in window:
                    if entry[0] is data.unit:
                        entry[2] += data.scale
                        break
                else:
                    window.append([data.unit, data.rain_index,
                                   data.scale.astype(np.float64)])
            elif intermediate is None:
                intermediate = data.copy()
            else:
                intermediate += data
```

At flush, choose the path. `_collapsed_row` returns `None` when it
cannot collapse, so the materialise branch is a fall-through rather than
an `else`:

```python
        grouped = None
        if intermediate is None and not window:
            # Every timestep in this window was dry.
            grouped = [0.0] * len(zone_indices)
        elif intermediate is None and len(window) == 1:
            grouped = _collapsed_row(*window[0])

        if grouped is None:
            # A window spanning a recovery boundary, eager input, or a
            # maximum over more than one rain cell. Materialising is
            # exact for every aggregation function, and at a boundary it
            # can only happen once.
            total = intermediate
            for unit, rain_index, scale in window:
                part = _spread(scale, rain_index, unit)
                total = part if total is None else total + part
            flat = np.asarray(total).reshape(-1)
            grouped = [agg(flat[positions]) for positions in zone_indices]
```

where `_collapsed_row` is a closure over `fn` and the weight cache:

```python
        def _collapsed_row(unit, rain_index, scale):
            sums, counts, maxima = _weights_for(unit, rain_index)
            if fn == 'max':
                if rain_index is None:
                    return list(scale[0] * maxima)
                return None          # caller must materialise instead
            totals = scale @ sums
            if fn == 'mean':
                with np.errstate(invalid='ignore', divide='ignore'):
                    return list(np.where(counts > 0, totals / counts,
                                         np.nan))
            return list(totals)
```

When `_collapsed_row` returns `None`, fall through to the materialise
branch. Keep the existing `agg()` closure for that branch, and keep
`zone_indices` exactly as it is — the weight builder consumes it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_subcatchment_timeseries.py -q`
Expected: all pass, including every pre-existing test in the file.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: the two known Windows failures and nothing else.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/sim/recorders.py fire_impacts/sim/tests/test_subcatchment_timeseries.py
git commit -m "Reduce subcatchment rows from zonal weights, not full grids"
```

---

### Task 6: Guard the assumptions the collapse rests on

**Files:**
- Modify: `fire_impacts/sim/rusle.py` — `generate_rusle`
- Test: `fire_impacts/sim/tests/test_dry_timesteps.py`

**Interfaces:**
- Consumes: `generate_rusle` from Task 3.
- Produces: no signature change. `generate_rusle` degrades to eager output for a segment whose layers or rainfall are negative.

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/sim/tests/test_dry_timesteps.py`:

```python
class TestNonNegativityGuards:
    """
    Deferring a maximum is only valid because a layer times a scale is
    monotone in the scale, which needs both to be non-negative. Rather
    than trust that, check it and fall back.
    """

    def test_a_negative_layer_falls_back_to_eager_grids(self, grids,
                                                        caplog):
        klscp, sdr, dnbr = grids
        klscp = klscp.copy()
        klscp[0, 0] = -1.0
        with caplog.at_level('WARNING'):
            steps = run([5.0], (klscp, sdr, dnbr))

        _, data = steps[0]
        assert isinstance(data['RUSLE'], np.ndarray)
        assert 'negative' in caplog.text.lower()

    def test_negative_rainfall_falls_back_to_eager_grids(self, grids,
                                                         caplog):
        with caplog.at_level('WARNING'):
            steps = run([-1.0, 5.0], grids)

        assert isinstance(steps[1][1]['RUSLE'], np.ndarray)
        assert 'negative' in caplog.text.lower()

    def test_the_fallback_still_produces_the_right_numbers(self, grids):
        klscp, sdr, dnbr = grids
        klscp = klscp.copy()
        klscp[0, 0] = -1.0
        _, data = run([5.0], (klscp, sdr, dnbr))[0]
        _, R = rainfall_erosivity(5.0)

        assert np.allclose(data['RUSLE'], R * klscp * CELL_AREA_HA)

    def test_ordinary_layers_stay_deferred(self, grids):
        _, data = run([5.0], grids)[0]
        assert isinstance(data['RUSLE'], ScaledGrid)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_dry_timesteps.py -q -k NonNegativity`
Expected: the first three fail — grids are still `ScaledGrid` and no
warning is logged.

- [ ] **Step 3: Add the guards**

In `generate_rusle`, immediately after building `units` and before the
loop:

```python
    # Deferring a maximum assumes a layer times a scale is monotone in
    # the scale, which holds only while both are non-negative. A noisy
    # DEM can produce a negative LS cell, and that must not silently
    # produce a wrong maximum - nor stop a run that works today.
    negative_layer = any(
        np.nanmin(unit) < 0 for unit in units.values() if unit.size
    )
    negative_rain = bool(np.nanmin(rainfall.values) < 0) \
        if len(rainfall.values) else False
    if negative_layer or negative_rain:
        logger.warning(
            'Negative %s in this segment, so its grids cannot be '
            'deferred; computing them in full instead. Results are '
            'unaffected, but this segment will be slower.',
            'erosion layer' if negative_layer else 'rainfall',
        )
        materialise_grids = True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_dry_timesteps.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add fire_impacts/sim/rusle.py fire_impacts/sim/tests/test_dry_timesteps.py
git commit -m "Check the assumptions that make deferring a maximum valid"
```

---

### Task 7: Thread the counter through the run and warn

**Files:**
- Modify: `fire_impacts/sim/rusle.py` — `run_usle_simulation`
- Test: `fire_impacts/sim/tests/test_materialisation_warning.py` (create)

**Interfaces:**
- Consumes: `MaterialisationCounter` from Task 2, `generate_rusle`'s `counter` and `materialise_grids` from Task 3.
- Produces: `run_usle_simulation(..., materialise_grids: bool = False)`.

- [ ] **Step 1: Write the failing tests**

Create `fire_impacts/sim/tests/test_materialisation_warning.py`:

```python
"""
Telling the user when a run is paying full price.

A recorder that forces a full grid every timestep costs roughly a hundred
times a deferred one. That is legitimate for some recorders, but it should
never be a surprise.
"""

import numpy as np
import pandas as pd

from fire_impacts.sim.rusle import _warn_about_materialising
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid


def greedy(timestep, **data):
    """A recorder that insists on a real grid."""
    return float(np.nansum(data['RUSLE']))


def frugal(timestep, **data):
    """A recorder that only looks at the scale."""
    return data['RUSLE'].scale[0]


def deferred(counter):
    return ScaledGrid(np.array([2.0]), np.ones((2, 3)), counter=counter)


class TestNamingTheCulprit:

    def test_names_the_recorder_that_materialises(self, caplog):
        counter = MaterialisationCounter()
        with caplog.at_level('WARNING'):
            _warn_about_materialising(
                {'greedy': greedy, 'frugal': frugal},
                pd.Timestamp('2020-01-01'),
                {'RUSLE': deferred(counter)},
                counter,
            )

        assert 'greedy' in caplog.text
        assert 'frugal' not in caplog.text

    def test_says_nothing_when_every_recorder_defers(self, caplog):
        counter = MaterialisationCounter()
        with caplog.at_level('WARNING'):
            _warn_about_materialising(
                {'frugal': frugal},
                pd.Timestamp('2020-01-01'),
                {'RUSLE': deferred(counter)},
                counter,
            )

        assert caplog.text == ''


class TestCounterIsolation:

    def test_each_run_gets_its_own_counter(self):
        # Replicates run concurrently on threads by default, so a shared
        # tally would blame the wrong recorder.
        a, b = MaterialisationCounter(), MaterialisationCounter()
        np.asarray(deferred(a))

        assert (a.count, b.count) == (1, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/sim/tests/test_materialisation_warning.py -q`
Expected: `ImportError: cannot import name '_warn_about_materialising'`.

- [ ] **Step 3: Implement the probe and thread the counter**

Add to `fire_impacts/sim/rusle.py`:

```python
def _warn_about_materialising(recorders, timestep, data, counter):
    """
    Log which recorders force a full grid, using one probe timestep.

    Parameters:
    - recorders: the run's recorder dict.
    - timestep: the timestep to probe with - the first wet one.
    - data: that timestep's data dict.
    - counter: the run's MaterialisationCounter.
    ------------------------------------------------------------------------
    Notes:
    - Called once per run. The recorders are invoked here and must not be
      invoked again for this timestep.
    ------------------------------------------------------------------------
    """
    greedy = []
    for name, recorder in recorders.items():
        before = counter.count
        recorder(timestep, **data)
        if counter.count > before:
            greedy.append(name)
    if greedy:
        logger.warning(
            'Recorder(s) %s materialise a full grid every timestep; this '
            'run will be substantially slower than a collapsed one.',
            ', '.join(sorted(greedy)),
        )
```

In `run_usle_simulation`, add the `materialise_grids: bool = False`
parameter, create `counter = MaterialisationCounter()` before the segment
loop, pass `materialise_grids=materialise_grids, counter=counter` to
`generate_rusle`, and replace the inner recorder loop with:

```python
        probed = materialise_grids
        for timestep, data in generate_rusle(...):
            if not probed and not data['dry']:
                _warn_about_materialising(recorders, timestep, data,
                                          counter)
                probed = True
                continue
            for recorder in recorders.values():
                recorder(timestep, **data, catchment=ctx.catchment,
                         transform=transform)
```

Note the `continue`: the probe already fed that timestep to every recorder.
Add `catchment` and `transform` to the dict passed into
`_warn_about_materialising` so the probe call is identical to a normal one.
Document `materialise_grids` in the docstring as the escape hatch that
forces the pre-collapse code path, for debugging and for the differential
test.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest fire_impacts/sim/tests/test_materialisation_warning.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: the two known Windows failures and nothing else. If a
`record_timestep_grid` test now emits the warning, that is correct
behaviour, not a failure.

- [ ] **Step 6: Commit**

```bash
git add fire_impacts/sim/rusle.py fire_impacts/sim/tests/test_materialisation_warning.py
git commit -m "Warn when a recorder forces a grid every timestep"
```

---

### Task 8: Prove the two paths agree, and regenerate the goldens

**Files:**
- Modify: `fire_impacts/tests/test_integration_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing further.

- [ ] **Step 1: Write the failing tests**

Append to `fire_impacts/tests/test_integration_pipeline.py`:

```python
def _rich_recorders():
    """
    A recorder set wide enough to be worth diffing.

    The pipeline fixture's own set is one variable, one function and one
    period, which would exercise almost none of the collapsed paths.
    """
    return simr.default_rusle_recorders(
        grid_variables=('RUSLE', 'delivered', 'RUSLE_above_threshold'),
        grid_fns=('sum', 'max', 'mean'),
        grid_timesteps=('total', 'yearly'),
        timeseries_variables=('RUSLE',),
        timeseries_fn='sum',
        timeseries_timestep='24h',
    )


@pytest.fixture(scope='module')
def both_ways(pipeline):
    """
    The same fire-adjusted run through both engines.

    save_rasters and save_timeseries are off because the module-scoped
    pipeline fixture has already written provenance for this run
    directory, and check_run_not_overwritten would reject a second write.
    """
    run, rain = pipeline['run'], pipeline['rain']
    factory = _rich_recorders()
    start, end = rain.index[0], rain.index[-1]
    out = {}
    for label, eager in (('collapsed', False), ('eager', True)):
        out[label] = simr.run_usle_simulation(
            run, rain,
            recorders=factory(run, start, end),
            save_rasters=False, save_timeseries=False,
            materialise_grids=eager,
        )
    return out['collapsed'], out['eager']


def test_the_collapsed_and_eager_engines_agree(both_ways):
    """The keystone: one engine must not drift from the other."""
    collapsed, eager = both_ways
    assert set(collapsed) == set(eager)

    for key in collapsed:
        if key == 'params':
            continue
        got, want = collapsed[key], eager[key]
        if got is None:
            assert want is None
            continue
        got = np.asarray(getattr(got, 'values', got), dtype=float)
        want = np.asarray(getattr(want, 'values', want), dtype=float)
        assert np.array_equal(np.isnan(got), np.isnan(want)), key
        scale = np.nanmax(np.abs(want)) or 1.0
        assert np.nanmax(np.abs(got - want)) / scale < 1e-6, key


def test_the_collapsed_engine_materialises_nothing_in_the_loop(pipeline):
    """
    The assertion with teeth. Correctness tests would still pass if a
    future change quietly fell back to per-timestep grid work; this one
    would not.
    """
    run, rain = pipeline['run'], pipeline['rain']
    factory = _rich_recorders()
    counter = simr.MaterialisationCounter()

    klscp, sdr, dnbr, cell_area_ha, transform = simr._rusle_parameter_grids(
        run, use_fire_adjusted=False)
    recorders = factory(run, rain.index[0], rain.index[-1])
    for recorder in recorders.values():
        recorder.reset()

    for timestep, data in simr.generate_rusle(
            rain, klscp, sdr, dnbr, cell_area_ha, counter=counter):
        for recorder in recorders.values():
            recorder(timestep, **data, catchment=run.catchment,
                     transform=transform)

    assert counter.count == 0


def test_params_and_transform_survive_the_collapse(both_ways):
    """Downstream code unpacks results['params'] as five plain values."""
    collapsed, eager = both_ways
    klscp, sdr, dnbr, cell_area_ha, transform = collapsed['params']

    assert klscp.ndim == 2 and sdr.ndim == 2 and dnbr.ndim == 2
    assert float(cell_area_ha) > 0
    assert transform == eager['params'][4]
    assert collapsed['the_transform'] == eager['the_transform']


def test_each_run_gets_its_own_materialisation_counter(pipeline,
                                                       monkeypatch):
    """
    Replicates run concurrently on threads by default, so a counter
    shared between runs would mix their tallies and blame the wrong
    recorder.
    """
    made = []
    real = simr.MaterialisationCounter

    def spy():
        counter = real()
        made.append(counter)
        return counter

    monkeypatch.setattr(simr, 'MaterialisationCounter', spy)
    run, rain = pipeline['run'], pipeline['rain']
    factory = _rich_recorders()
    for _ in range(2):
        simr.run_usle_simulation(
            run, rain,
            recorders=factory(run, rain.index[0], rain.index[-1]),
            save_rasters=False, save_timeseries=False)

    assert len(made) == 2
    assert made[0] is not made[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest fire_impacts/tests/test_integration_pipeline.py -q -k "agree or materialises or survive or counter"`
Expected: `test_..._agree` fails with `TypeError: run_usle_simulation()
got an unexpected keyword argument 'materialise_grids'` if Task 7 is
incomplete; otherwise `test_..._materialises_nothing_in_the_loop` is the
one that must fail before the collapse works.

The `pipeline` fixture returns its RunContext under the key `'run'` and
its rainfall Series under `'rain'` — verified, not assumed. It is
module-scoped and has already written provenance for that run directory,
which is why `save_rasters=False, save_timeseries=False` is not optional
here.

- [ ] **Step 3: Make them pass**

No production change should be needed. If
`test_..._materialises_nothing` fails with a non-zero count, find the
recorder responsible by running with `-o log_cli=true --log-cli-level=WARNING`
and reading the Task 7 warning, then fix that recorder's scale path. If
`test_..._agree` fails, the eager and collapsed paths genuinely disagree —
that is a real bug in Tasks 4 to 6, not a test to loosen. Do not raise the
1e-6 tolerance to make it pass.

- [ ] **Step 4: Regenerate the golden hashes**

`GOLDEN_PREP_HASHES` includes two *simulation* rasters despite its name.
Their contents shift by ~1e-7, so the hashes must be regenerated
deliberately.

Run: `python -m pytest fire_impacts/tests/test_integration_pipeline.py::test_default_outputs_are_unchanged -q`

On Windows this fails on path separators regardless — read the two
`Runs/.../RUSLE_sum_total.tif` entries out of the assertion diff, confirm
**only those two changed** and every preprocessing hash is untouched, and
update just those two literals at `test_integration_pipeline.py:92-93`. If
any preprocessing hash moved, stop: this change must not touch
preprocessing.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest fire_impacts -q 2>&1 | tail -3`
Expected: the two known Windows failures and nothing else.

- [ ] **Step 6: Measure the result**

Confirm the change did what it was for, against the ~3.0 min per 5-year
replicate baseline in Global Constraints. Use the benchmark harness under
`.../scratchpad/bench.py` if it survives, or time a single replicate in the
working project at `D:\Geospatial\waterra\case-studies\waternsw\modelling`.
Report the number; do not assert on it in a test.

- [ ] **Step 7: Commit**

```bash
git add fire_impacts/tests/test_integration_pipeline.py
git commit -m "Assert the collapsed and eager engines agree

Also regenerates the two simulation entries in GOLDEN_PREP_HASHES. Sums
are now (sum of scale) * layer rather than sum of (scale * layer), which
moves results by about 1e-7 relative - float32 noise. Every preprocessing
hash is unchanged."
```

---

## Notes for the executor

- **Every task leaves a working system.** That is deliberate: after Task 3 the
  recorders still materialise and the run is no faster, and that is a correct
  intermediate state, not a half-finished one.
- **Never assert bit-equality between the eager and collapsed paths.** The
  collapse changes the order of floating-point accumulation. 1e-6 relative
  with matching NaN patterns is the contract.
- **If a test passes before you have written the implementation**, work out
  why before continuing. The `ScaledGrid` facade makes a lot of old code work
  unchanged, so a green test can mean "the slow path handled it", which is not
  what the test was for.
