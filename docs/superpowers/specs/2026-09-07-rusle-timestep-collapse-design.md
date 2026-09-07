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

A spike confirmed this against the real recorders on 8,000 timesteps: agreement
of 8e-08 to 1.2e-07 relative, identical NaN patterns, identical timeseries
index, and 3.49 min to 0.6 s.

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

The facade for callers that know nothing about the type: `__array__`, `shape`,
`dtype`, `ndim`, `copy`, `reshape`, `ravel`, `astype`. Verified interop:
`np.asarray(sg)`, `sg * arr`, `arr * sg`, `np.nansum`, `np.maximum`,
`np.stack`.

`copy()` returns a **materialised, mutable ndarray**, because a caller asking
for a copy wants something it can write into.

The type is immutable, so the shared read-only dry buffer introduced in 8da1bc2
becomes unnecessary: a dry timestep is `scale = [0.0]`. The `dry` flag is
retained, because recorders still use it to skip work.

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

The RUSLE formula appears exactly once in the codebase. That is the point of
choosing B.

## Recorder changes

### record_multi_period_grid

Accumulates scales rather than grids. Because `unit` changes between recovery
segments, each period holds a short list of `(unit, accumulated_scale)` pairs,
one entry per segment that period touches, keyed by `unit` identity.

At `finalize()`:

- `sum` — sum over `u` of `(scale_sum_u * unit_u)`
- `max` — elementwise max over `u` of `(scale_max_u * unit_u)`
- `mean` — the sum, divided by the count of timesteps in the period (dry
  timesteps included, as today)

Grid operations drop from O(timesteps) to O(periods x segments).

### record_subcatchment_timeseries

Precomputes, per distinct `unit`, the zonal reduction of that layer as a matrix
`W[u]` of shape `(n_rain_cells, n_zones)`, which is `(1, n_zones)` today. A row
is then `accumulated_scale @ W[u]`.

The recorder holds `(unit, scale_sum)` pairs for the current aggregation
window:

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

`ScaledGrid` counts materialisations. On the first wet timestep,
`run_usle_simulation` checks the counter delta around each recorder call, which
names the culprit for the price of one integer compare, and logs once:

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
pass per segment, and they convert an invisible wrong answer into a loud one.

## Spatial rainfall: what is and is not built

**Built now:** `scale` as a 1-D array throughout; `rain_index` on `ScaledGrid`;
`W` construction via `np.bincount`, written generally and **tested generally**
with a synthetic multi-rain-cell map, so the seam is exercised before anything
depends on it.

**Not built:** any rainfall reader, rain-grid alignment, or plumbing. Spatial
rainfall then becomes a data-source change rather than an engine change.

## File organisation

`fire_impacts/sim/rusle.py` is 1978 lines and the recorders are what this
change rewrites. Split out:

- `fire_impacts/sim/scaled_grid.py` — the value type, ~80 lines
- `fire_impacts/sim/recorders.py` — `record_subcatchment_timeseries`,
  `record_grid_transform`, `record_timestep_grid`, `record_multi_period_grid`,
  `default_rusle_recorders`, `_compute_periods`, `_calendar_floor`,
  `_spatial_coords_from_transform`, `_PERIOD_OFFSETS`, ~700 lines

with re-exports from `rusle.py` so existing imports keep working, leaving
`rusle.py` at ~1200 lines of model, parameter grids and runners.

Dependencies stay one-way: `rusle.py` imports `recorders.py`, never the
reverse. Any constant both need, such as `_MODEL_TIMESTEP`, moves to
`const.py`.

## Numerical consequences

Sums become `(sum of scale) * unit` rather than `sum of (scale * unit)`, so
results shift at approximately **1e-7 relative**: float32 noise, matching the
measured spike. Accumulation is float64 throughout. Output raster dtype is
unaffected, because `save_catchment_raster` derives it from the template
raster's metadata and casts on write.

**One deliberate behaviour change.** A period containing no rain at all
currently finalises to plain zeros with no NaN outside the catchment, because
`finalize` fabricates `np.zeros(shape)` having never seen a grid. Under this
design it finalises to `0 * unit`, carrying the same NaN mask as every other
period. This is the more correct answer and is taken deliberately.

## Test plan

1. `ScaledGrid` — materialise (uniform and vector scale), numpy interop,
   `copy()` mutability, `shape` / `dtype` / `reshape` / `ravel`.
2. `record_multi_period_grid` — scale path equals grid path for sum, max and
   mean; across two distinct units; and for all-dry periods.
3. `record_subcatchment_timeseries` — scale path equals grid path; a straddling
   window materialises and is exact for all three functions; row cadence and
   index unchanged.
4. `W` construction verified against a brute-force loop using a synthetic
   multi-rain-cell map.
5. **Materialisation count** for the default recorder set over a long run is
   bounded by `periods * segments + straddling windows + finalize`. This is a
   behavioural assertion that the collapse actually happened, without the
   flakiness of a timing test, and is what stops a future change quietly
   falling back to per-timestep grid work while every correctness test still
   passes.
6. The warning fires and names the offending recorder.
7. Differential test on the existing `pipeline` fixture in
   `test_integration_pipeline.py`: `run_usle_simulation` twice, eager and
   collapsed, asserting every recorder result agrees within tolerance with
   identical NaN patterns.

## Out of scope

- Debris flow, which does not use `generate_rusle`.
- Parallelism. At an estimated 1.5-2 s per replicate the question is moot for
  RUSLE, so `run_rusle_all_replicates` is left as it is.
- The dead `pd.DataFrame` branch in `aggregate_rainfall_data`, which
  `flatten_pyraingen_rainfall` makes unreachable.
