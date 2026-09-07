# Collapsing the RUSLE timestep loop

**Status:** design agreed, not yet implemented
**Date:** 2026-09-07

## The problem

A single-replicate RUSLE run takes minutes. Profiling the current code on the
WaterNSW case study (772x449 grid, 33 subcatchments, 5 years of 30-minute
rainfall, default recorder set) attributes it as:

| | share | per 5-year replicate |
|---|---|---|
| `generate_rusle` (the per-cell maths) | 85.6% | 2.92 min |
| `erosion_daily_time_series` | 6.5% | 0.22 min |
| `RUSLE_sum_yearly` + `RUSLE_max_yearly` | 7.6% | 0.26 min |
| `the_transform` | 0.1% | — |

Almost all of it is the per-timestep grid arithmetic, not the recorders. The
work is avoidable, because RUSLE is separable:

```
RUSLE(t, cell) = R(t) * klscp(cell) * cell_area
```

`R` depends only on rainfall; `klscp` depends only on the cell. For a fixed set
of layers, every recorder output is therefore a reduction over `R` times a
static grid:

```
sum over period   = (sum of R over period) * klscp * area
max over period   = (max of R over period) * klscp * area      [klscp >= 0]
zonal row         = (sum of R over window) * zonal_sum(klscp * area)
```

A spike confirmed this against the real recorders: agreement of 8e-08 to
1.2e-07 relative, identical NaN patterns, identical timeseries index. Both
figures here are measured over 8,000 timesteps and scaled to a full 5-year
replicate on the same basis as the table above: **3.49 min to 0.6 s**.

RUSLE is stateless and linear in `R` per cell. The only non-linearity seen in
practice is a rainfall threshold (for example, no erosion below 12.7 mm/day),
which is a function of rainfall alone and so applies to `R` before any grid is
involved. Separability is unaffected by it.

## Why now

Spatial rainfall support is imminent. Under spatial rainfall `R` becomes a
vector over rain cells, and the collapse does not break, it generalises. The
subcatchment timeseries becomes a matrix product `R_windowed @ W`, where
`W[r, z]` sums `klscp * area` over the cells of subcatchment `z` falling in rain
cell `r`. This was verified against a brute-force per-timestep loop at 772x449
with 5 km rainfall: relative difference 1.9e-16, 1609x faster.

Uniform rainfall is the one-rain-cell case of the same code. Doing this before
spatial rainfall lands means one implementation rather than two.

## Approach: lazy scaled grids

Three approaches were considered.

**A. Declarative recorder specs plus a separate analytic engine.** Fastest
(~0.6 s), but the RUSLE formula would exist in two places, the generator and
the analytic engine, and they can drift. A single unrecognised recorder puts
the whole run back on the slow path.

**B. Lazy scaled grids through the existing generator and recorders.** Chosen.
The formula stays in one place; degradation is per-recorder rather than
all-or-nothing; it is the natural home for the spatial-rainfall
generalisation. Slightly slower than A, since the per-timestep Python loop
remains.

**C. Restructure recorders into declarative descriptions.** Conceptually
cleanest but breaks the closure protocol the docstrings advertise, for no gain
over B here.

B was chosen because in a research model the drift risk in A is the one that
bites: a formula change that silently fails to reach the fast path produces
plausible wrong numbers. The difference between 0.6 s and an estimated 1.5-2 s
is irrelevant beside that.

## The value type

```python
class ScaledGrid:
    """A grid that is a static per-cell layer times a per-timestep scale."""
    __slots__ = ('scale', 'unit', 'rain_index')
```

- `scale` — erosivity for this timestep, **always** a 1-D float64 array over
  rain cells, of length 1 when rainfall is uniform. Always an array so that no
  accumulator needs a scalar/vector branch. The six `ScaledGrid` objects of one
  timestep share a single scale array.
- `unit` — the static per-cell layer for this recovery segment, shared by every
  timestep in the segment. Object **identity** is what tells a recorder that
  the layers are unchanged since the last timestep.
- `rain_index` — `None` for uniform rainfall, otherwise a 2-D int array mapping
  each DEM cell to a rain cell.

`materialise()` returns `scale[0] * unit` when `rain_index is None` (the fast
path; fancy-indexing a full grid through an all-zeros map would be slower),
otherwise `scale[rain_index] * unit`.

### The facade must be a proper duck array, not a short list of methods

A first cut exposing only `__array__`, `shape`, `dtype`, `ndim`, `copy`,
`reshape`, `ravel` and `astype` was tested and is **not sufficient**. All of
these raise `TypeError` or `AttributeError` under it: `sg * 2.0`, `2.0 * sg`,
`sg + sg`, `sg[1]`, `len(sg)`, `sg.flags.writeable`. Even `sg * ndarray` only
works by accident, via `ndarray.__rmul__` coercing through `__array__`.

`ScaledGrid` therefore subclasses `numpy.lib.mixins.NDArrayOperatorsMixin` and
implements `__array_ufunc__` (materialise every `ScaledGrid` input, then
delegate to the ufunc), plus `__getitem__`, `__len__` and `flags`. With that,
all sixteen operations exercised by the current recorders and tests pass.

**`__array_ufunc__` must reject a `ScaledGrid` in `out=`.** Without that guard
`sg += 1.0` **segfaults the interpreter**: `NDArrayOperatorsMixin` routes
in-place operators through `__array_ufunc__` with `out=(self,)`, and numpy then
writes into a temporary produced by `__array__`. Raising
`TypeError('ScaledGrid is immutable; call .copy() for a writable array')` when
any output is a `ScaledGrid` turns a crash into a clear message. This is
verified behaviour, not a precaution.

`copy()` returns a **materialised, mutable ndarray**, because a caller asking
for a copy wants something it can write into.

The type is immutable, so the shared read-only dry buffer introduced in 8da1bc2
becomes unnecessary: a dry timestep is `scale = [0.0]`. The `dry` flag is
retained, because recorders still use it to skip work.

`fire_impacts/sim/tests/test_dry_timesteps.py` specifies the *current* contract
directly: six grid keys being the same object, `+=` raising `ValueError`,
`.flags.writeable` on a wet grid. It is rewritten as part of this change, to
specify the `ScaledGrid` contract instead. That is a decision, not a
discovery.

## What generate_rusle yields

The six static layers are built **once per segment**, at the top of the
generator, rather than six fresh grids per timestep:

```
unit_RUSLE                    = klscp * cell_area_ha
unit_delivered                = unit_RUSLE * sdr
unit_RUSLE_below/above        = where(dnbr </>= threshold, unit_RUSLE, 0)
unit_delivered_below/above    = where(dnbr </>= threshold, unit_delivered, 0)
```

Each timestep then yields `ScaledGrid(R, unit_X)` for each of the six grid
keys, alongside today's `total_rain`, `intensity`, `erosivity` and `dry`.

The formula appears **once on the live path**. (`generate_rusle_for_feature`,
`rusle.py:1313-1415`, holds a second copy, reached from the deprecated
`gridded_total_rusle` and `calculate_lumped_rusle`; the latter is still live via
`lumped_daily_rusle`. It is left untouched by this change.) Avoiding a *third*
copy is the point of choosing B.

`generate_rusle` takes its own `materialise_grids` parameter, which
`run_usle_simulation` threads through: it is a public documented generator
called directly from tests, and its docstring promise of a "float32 numpy
array" needs rewriting either way.

## Recorder changes

### record_multi_period_grid

Accumulates scales rather than grids. Because `unit` changes between recovery
segments, each period holds a short list of `(unit, accumulated_scale)` pairs,
one entry per segment that period touches, keyed by `unit` identity.

**A dry timestep increments the period's count but registers no
`(unit, scale)` pair.** This matters beyond efficiency: registering one would
make the finalised NaN mask the union of every segment's mask rather than only
those that actually contributed, and the per-recovery C/K/SDR rasters are
separate files that need not share a nodata footprint.

At `finalize()`:

- `sum` — sum over `u` of `(scale_sum_u * unit_u)`
- `max` — elementwise max over `u` of `(scale_max_u * unit_u)`
- `mean` — as `sum`, divided by the period's timestep count (dry included, as
  today), **retaining the existing `count > 0` guard**. A period with zero
  timesteps is reachable: `_compute_periods` bins the whole `(start, end)`
  range while `_recovery_run_segments` drops rainfall falling outside every
  recovery window.
- a period with **no pairs at all**, whether it saw no timesteps or only dry
  ones, finalises to `np.zeros(shape)` exactly as today.

That last rule means this design introduces **no behaviour change** for empty
or entirely-dry periods, and `test_period_with_no_data_becomes_zeros` and
`test_a_period_of_only_dry_timesteps_finalises_to_zeros` continue to pass
unmodified.

Grid operations drop from O(timesteps) to O(periods x segments).

### record_subcatchment_timeseries

Precomputes, per distinct `unit`, the zonal reduction of that layer as a matrix
`W[u]` of shape `(n_rain_cells, n_zones)`, which is `(1, n_zones)` today. A row
is then `accumulated_scale @ W[u]`.

`W[u]` is a zonal **sum** for `fn='sum'` and `fn='mean'`, and both stay linear
in the scale vector for any number of rain cells. `fn='max'` does **not**:
`nanmax over zone of (scale[rain_index] * unit)` cannot be written as
`scale @ W[:, z]` for any fixed `W`. So `max` takes the collapsed path only when
`n_rain_cells == 1` — where it reduces to `scale_sum * nanmax_zone(unit)`, valid
because `unit >= 0` — and the materialise path otherwise. Test-plan item 4 is
parametrised over all three functions so this cannot rot silently once spatial
rainfall lands.

As in the period recorder, **a dry timestep advances the window counter but
registers no `(unit, scale)` pair.**

The recorder holds `(unit, scale_sum)` pairs for the current aggregation
window:

- **no pairs** (every timestep dry) — short-circuit to `0.0` per zone, as today,
  rather than computing `0 * W[u]`. For `mean` and `max`, `W[u]` is `NaN` for a
  zone whose cells are all NaN, and `0 * NaN` would put a silent NaN into a
  timeseries column that then propagates through `ensemble.py`.
- **one pair** (the overwhelming majority) — collapsed path
- **more than one pair** — materialise the sum over `u` of `(scale_u * unit_u)`
  into a single grid and run the existing zonal reduction on it

The second branch is exactly correct for `sum`, `mean` and `max` without
special-casing, and can only occur on windows that span a recovery-window
boundary: at most `n_segments - 1` windows out of thousands. `max` therefore
needs no special treatment, and no warning is required for this case.

### record_timestep_grid and record_grid_transform

Unchanged. `record_timestep_grid` materialises every timestep by definition; it
is already documented as memory-heavy and diagnostic-only.

### Both accumulation paths are retained

Every scale-aware recorder keeps its existing grid-accumulation path as well,
for plain-ndarray input. This is not duplicated model formula, it is two
trivial accumulators, and it is what makes the escape hatch below a one-line
switch. Both paths are exercised on every test run.

## Warning on per-timestep materialisation

`ScaledGrid` counts materialisations against a **per-run counter object**
threaded onto each `ScaledGrid`, never a module global or class attribute:
`run_rusle_all_replicates` defaults to `scheduler='threads'` and runs N
replicates concurrently in one process, so a shared counter would mix deltas
across replicates, naming innocent recorders and masking real ones.

On the first wet timestep, `run_usle_simulation` checks the counter delta around
each recorder call, which names the culprit for the price of one integer
compare, and logs once:

> Recorder 'X' materialises a full grid every timestep; this run will be
> substantially slower than a collapsed one.

`record_timestep_grid` will legitimately trip this, and the message remains the
honest thing to say about such a run.

## Escape hatch

`run_usle_simulation(..., materialise_grids: bool = False)`. When true,
`generate_rusle` yields plain ndarrays exactly as today, and the recorders'
retained grid paths handle them.

Its primary purpose is the differential test. Secondarily, when a result looks
wrong it separates a wrong collapse from a wrong model.

## Assumptions, checked rather than trusted

The collapse is valid only because `unit >= 0` and `scale >= 0`. That is what
makes `max_t(scale * unit)` equal `(max_t scale) * unit`. A negative P factor,
or a negative value in a rainfall series, would silently produce wrong maxima.

Therefore: one `nanmin` check per segment on each unit layer, and one
vectorised non-negativity check on the segment's rainfall. Both are a single
pass per segment.

**On failure the run warns and falls back to the eager path for that segment**
rather than raising. A catchment with a single negative LS cell — plausible from
a noisy DEM — produces a number today, and this change should not be the thing
that stops it running. The check exists only to protect the `max` collapse, so
degrading to eager is both sufficient and safe.

## Spatial rainfall: what is and is not built

**Built now:** `scale` as a 1-D array throughout; `rain_index` on `ScaledGrid`;
`W` construction via `np.bincount` for `sum` and `mean`, written generally and
**tested generally** with a synthetic multi-rain-cell map, so the seam is
exercised before anything depends on it. `max` with more than one rain cell
takes the materialise path, rather than a speculative max-product matrix.

**Not built:** any rainfall reader, rain-grid alignment, or plumbing. Spatial
rainfall then becomes a data-source change rather than an engine change.

## File organisation

`fire_impacts/sim/rusle.py` is 2032 lines and the recorders are what this
change rewrites. Split out:

- `fire_impacts/sim/scaled_grid.py` — the value type, ~80 lines
- `fire_impacts/sim/recorders.py` — `record_subcatchment_timeseries`,
  `record_grid_transform`, `record_timestep_grid`, `record_multi_period_grid`,
  `default_rusle_recorders`, `_compute_periods`, `_calendar_floor`,
  `_spatial_coords_from_transform`, `_PERIOD_OFFSETS`, ~700 lines

about 661 lines in total, leaving `rusle.py` at roughly 1370 lines of model,
parameter grids and runners.

The re-exports from `rusle.py` must include the **private** names, because the
tests import them by name: `test_periods.py` imports `_calendar_floor` and
`_compute_periods`, and `test_recorders.py` imports
`_spatial_coords_from_transform`, both from `fire_impacts.sim.rusle`.

Dependencies stay one-way: `rusle.py` imports `recorders.py`, never the
reverse. Any constant both need, such as `_MODEL_TIMESTEP`, moves to
`const.py`.

One hazard to note for later: monkeypatching a re-exported name on the `rusle`
module patches the `rusle` binding while code living in `recorders.py` resolves
its own. `test_save_grid_results.py` patches `save_catchment_raster`, which
stays in `rusle.py`, so it is unaffected — but the pattern is a trap.

## Numerical consequences

Sums become `(sum of scale) * unit` rather than `sum of (scale * unit)`, so
results shift at approximately **1e-7 relative**: float32 noise, matching the
measured spike. Accumulation is float64 throughout. Output raster dtype is
unaffected, because `save_catchment_raster` derives it from the template
raster's metadata and casts on write.

**`GOLDEN_PREP_HASHES` must be regenerated as an explicit step.**
`test_integration_pipeline.py` hashes every `.tif` in the fixture project
bit-exactly, and despite its name that dict includes two *simulation* outputs:
`Runs/2019_fire/historical/Results/RUSLE_sum_total.tif` and its
`Results_baseline` counterpart. A ~1e-7 shift changes those hashes. This is the
project's guard against an equation having changed, so regenerating it must be
a deliberate, called-out step in the commit rather than a surprise.

**No behaviour change for empty or all-dry periods** — see the
`record_multi_period_grid` rules above; the existing zeros behaviour is
retained.

**A second raster writer exists.** `_save_grid_results` goes through
`save_catchment_raster`, which casts to the template dtype. But
`fire_impacts/sim/results.py` `_write_dataarray_as_geotiff` calls
`da.rio.to_raster` with no template, so the array's own dtype reaches disk.
Accumulation is already float64 on that path today, because erosivity is
float64 whenever the rainfall series is, so this design does not change it — but
the asymmetry is worth knowing before anyone tunes output size.

## Test plan

1. `ScaledGrid` — materialise (uniform and vector scale); the full duck-array
   surface including `sg * 2.0`, `sg[i]`, `len`, `flags`; `copy()` mutability;
   and that an in-place op raises `TypeError` rather than segfaulting.
2. `record_multi_period_grid` — scale path equals grid path for sum, max and
   mean; across two distinct units; for all-dry periods; for a period with zero
   timesteps (the `mean` denominator guard).
3. `record_subcatchment_timeseries` — scale path equals grid path; a straddling
   window materialises and is exact for all three functions; an all-dry window
   yields `0.0` and not `NaN`; row cadence and index unchanged.
4. `W` construction verified against a brute-force loop using a synthetic
   multi-rain-cell map, **parametrised over `sum`, `mean` and `max`**, so the
   `max` restriction is pinned rather than assumed.
5. **Zero materialisations during the timestep loop** for the default recorder
   set. This is the assertion with teeth: a behavioural check that the collapse
   actually happened, without the flakiness of a timing test, and what stops a
   future change quietly falling back to per-timestep grid work while every
   correctness test still passes.
6. The warning fires and names the offending recorder, and the counter is not
   shared between replicates — a two-replicate
   `run_rusle_all_replicates(scheduler='threads')` smoke test.
7. The non-negativity check fires on a negative unit layer and degrades to
   eager rather than raising.
8. `results['params']` and `the_transform` are unchanged; downstream code
   unpacks the former as five plain arrays.
9. Differential test: `run_usle_simulation` twice, eager and collapsed,
   asserting every recorder result agrees within tolerance with identical NaN
   patterns. It must build its **own richer recorder factory** rather than
   reuse the `pipeline` fixture's, which is a single variable, a single `sum`
   and a single `total` period — it would never exercise multi-period times
   multi-segment, `max` or `mean`, the `delivered` and `*_threshold` variables
   (whose NaN masks differ, being 0 outside the catchment rather than NaN), or
   `record_timestep_grid`. It must also pass `save_rasters=False,
   save_timeseries=False`, or `check_run_not_overwritten` will reject the
   second run against the provenance the fixture already wrote.

## Out of scope

- Debris flow, which does not use `generate_rusle`.
- Parallelism. At an estimated 1.5-2 s per replicate the question is moot for
  RUSLE, so `run_rusle_all_replicates` is left as it is.
- The dead `pd.DataFrame` branch in `aggregate_rainfall_data`, which
  `flatten_pyraingen_rainfall` makes unreachable.
