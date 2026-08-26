### Title: Debris flow — year length (365 vs 365.25) and leap days in the driving rainfall

**Created by**: joelrahman
**Created on**: 2026-08-26
**Status**: parked — needs a decision on rainfall provenance before it can be settled

---

## Summary

`sim/debris.py` partitions the post-fire period into simulation years with
`pd.Timedelta(days=365)` (`const.DAYS_PER_SIM_YEAR`). Elsewhere —
`context.EventDefinition.absolute_window` and `simulation_period` — the same
years→days conversion uses `365.25`. The two disagree, and it is not obvious
which is correct, because the answer depends on whether the driving rainfall
contains leap days at all.

This is a small effect, deliberately parked rather than guessed at.

## Why it is not just a typo

Neither value is unambiguously right:

- **365 is exact three years in four.** For a single post-fire year, which is
  what the debris-flow windows are, it is the correct length in the common
  case. Using 365.25 makes every window a quarter-day too long, and after two
  years the second window is offset by half a day.
- **365.25 is the better long-run average.** For recovery breakpoints spanning
  several years, which is what `EventDefinition` handles, drift matters more
  than any individual year being exact.

So the current split may in fact be defensible: 365 for a discrete
single-year window, 365.25 for a multi-year offset. What it is not, at
present, is *documented as a decision* — it reads as an inconsistency.

## The complication: does the rainfall have leap days?

The correct year length depends on the calendar the rainfall actually uses,
which differs by source:

| Source | Leap days? | Implication |
|---|---|---|
| **pyraingen** stochastic replicates (the usual driver) | **Unknown — needs checking.** Stochastic generators often emit fixed 365-day years. | If there are no 29 Februaries, a 365-day window is exactly one year of data and 365.25 would drift a day every four years for no reason. |
| Historical observations | Yes | A 365-day window steps back one day every leap year relative to the calendar date. |

If the two sources differ, a single constant is wrong for one of them, and
the model would need to derive the year length from the rainfall index rather
than assume it.

## What to check

1. **Does pyraingen output include 29 February?** Inspect a generated
   replicate spanning a leap year — count timestamps per calendar year, or
   look for `02-29` in the index. `fire_impacts/stochastic/rainfall/` is the
   entry point.
2. **What does `aggregate_rainfall_data` do to a leap day** when converting
   between 12-minute and 30-minute resolutions?
3. If the sources differ, decide whether the year length should be derived
   per-run from the rainfall index instead of being a constant.

## Scope of the impact

Small, and bounded:

- It shifts the boundary between the Year 1 and Year 2 debris windows by up
  to a day or so. Only rainfall events falling within that boundary window
  change which year's `I12_crit` threshold they are tested against.
- It does not affect the RUSLE erosion path, which segments on recovery
  breakpoints rather than on whole years.
- It has no effect at all if the rainfall contains no leap days and the
  period is under four years — the common case today.

## Related, already resolved

- The debris year windows previously started at `rainfall.index[0]` rather
  than the fire end date, which offset them from the recovery clock the rest
  of the model uses whenever the rainfall did not begin exactly at the fire
  end. **Fixed** — they now start at `ctx.fire_end_date`.
- The lookup's `years` bins (0.434 and 1.434, i.e. roughly 5 and 17 months)
  are treated as representative of their whole year, making the threshold
  piecewise-constant in time since fire. **Documented** in `const.py` and in
  the `debris_flow` / `calc_I12_crit_columns` docstrings; not a defect.

## Suggested resolution when picked up

Keep `DAYS_PER_SIM_YEAR = 365` for the discrete debris windows and `365.25`
for multi-year recovery offsets, but state that split explicitly as a
decision — *unless* checking (1) shows the two rainfall sources use different
calendars, in which case derive the year length from the rainfall index and
drop the constant.
