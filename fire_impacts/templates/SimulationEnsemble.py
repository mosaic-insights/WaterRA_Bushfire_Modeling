# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fire impacts simulations — stochastic ensemble
#
# This notebook demonstrates the fire impacts simulation modules using a
# full ensemble of stochastic rainfall replicates.  Both the RUSLE
# erosion module and the debris-flow module are run across every
# replicate in parallel, and the library then provides ready-made
# analytics and visualisations to communicate the resulting *risk*
# rather than any single point prediction.
#
# It is assumed that you have a `FireImpactsProject` populated with all
# pre-processed data for the catchment.

# %%
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

import matplotlib.pyplot as plt

from fire_impacts import FireImpactsProject
from fire_impacts.context import RunContext
from fire_impacts.sim import (
    aggregate_rainfall_data,
    convert_rainfall_depth_to_intensity,
    default_rusle_recorders,
    run_rusle_all_replicates,
    run_debris_flow_all_replicates,
    postprocess_debris_flow,
    exceedance_probability,
    plot_exceedance,
    plot_ensemble_statistics_panel,
    plot_catchment_exceedance_curve,
    plot_ensemble_daily_ribbon,
    combine_rusle_and_debris_subcatchment,
    rusle_subcatchment_ensemble,
    debris_subcatchment_ensemble,
    plot_subcatchment_ensemble,
    save_ensemble_run,
)
from fire_impacts.stochastic.rainfall import get_rainfall_replicates

# %% [markdown]
# ## Settings for this study
#
# From `study.toml` — the same file the other notebooks read. Everything
# the ensemble needs you to choose is in there; the cells below refer to
# the names defined here.
#
# Running the cell prints each setting and where it came from, so you can
# check the notebook is about to do what you expect.

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
SUBCATCHMENTS = study.catchment.subcatchments
SUBCATCHMENT_ID_FIELD = study.catchment.subcatchment_id_field

# Optional climate statistics for the rainfall generator. Left out of
# study.toml they arrive as None, and the backend estimates them from the
# catchment's lat/lon. The Simulation notebook reads the same two, so both
# notebooks ask pyraingen for the same rainfall.
MEAN_ANNUAL_RAINFALL = study.ensemble.mean_annual_rainfall
AVERAGE_TEMPERATURE  = study.ensemble.average_temperature

# Per-cell results are reported per hectare, so the model needs to know how
# big a cell is. This follows the DEM: change catchment.cell_size_m if yours
# is not the 30 m of the national DEM this project downloads by default.
# Nothing checks it against the raster, and a wrong value is silent — it
# rescales every t/ha figure below, and with them the erosion threshold the
# exceedance map is drawn at.
CELL_AREA_HA = study.catchment.cell_size_m ** 2 / 10_000

# Where the exceedance maps put their line. Reporting choices, not model
# calibration: they change what a map shows, never what the model computes.
EROSION_THRESHOLD_T_HA    = study.reporting.erosion_threshold_t_ha
DELIVERED_THRESHOLD_KG_HA = study.reporting.delivered_threshold_kg_ha

# %% [markdown]
# ## Load project

# %%
proj = FireImpactsProject(PROJECT_DIR, exist_ok=True)
proj.catchments

# %% [markdown]
# > **Note:** a `FireImpactsProject` can hold several catchments. This
# > notebook works on the one named by `catchment.name` in `study.toml`.

# %%
# A run binds to one (catchment, event, ensemble) combination via a
# RunContext. The event must match a directory produced by PrepareData;
# the ensemble names this climate realisation. All three are named in
# study.toml.
# Pass label='...' to name this run's output directory, so several
# parameter variants of one (event, ensemble) can sit side by side.
# It defaults to the ensemble name.
ctx = RunContext.solo_run(
    proj, event=EVENT, ensemble=ENSEMBLE,
    catchment=CATCHMENT,
)

# %% [markdown]
# ## Rainfall data
#
# Both the erosion (RUSLE) and debris-flow modules require sub-daily
# rainfall.  RUSLE uses 30-minute data, debris flow uses 12-minute data.
#
# The library generates stochastic sub-daily rainfall replicates via
# [pyraingen](https://github.com/crdykman/pyraingen).  You can install
# pyraingen locally and calibrate it to available sub-daily rainfall
# observations, or call the remote pyraingen API which uses publicly
# available climate statistics — this template uses the remote API.
# The library infers location and elevation from the catchment
# boundary and DEM.  Mean annual rainfall and average temperature are
# optional: the backend service estimates them from lat/lon when not
# supplied.  Set `ensemble.mean_annual_rainfall` and
# `ensemble.average_temperature` in `study.toml` when you have
# site-specific values.
#
# The same set of replicates feeds both simulations.

# %%
# The simulation period spans the recovery windows recorded in the
# event definition (fire end date -> end of the last window), so it
# isn't hard-coded here.
rain_data_start, rain_data_end = ctx.simulation_period()

# %%
# `num_years` is inferred from start/end (one API year per calendar
# year spanned).  `mean_annual_rainfall` and `average_temperature` are
# optional: leave them out of study.toml and they arrive here as None,
# which tells the backend service to estimate them from the catchment's
# lat/lon.
# The generated rainfall is cached under Ensembles/<ensemble>/ and reused
# on repeat runs — identical rainfall, no repeat API call. Pass
# regenerate=True to force a fresh draw.
# Note what that means for the two climate statistics: only the window and
# the replicate count decide whether the cache is reused, so changing
# mean_annual_rainfall or average_temperature after a run has no effect
# until you pass regenerate=True. You get the cached rainfall, silently.
replicates = get_rainfall_replicates(
    ctx,
    start=rain_data_start,
    end=rain_data_end,
    num_replicates=N_REPLICATES,
    mean_annual_rainfall=MEAN_ANNUAL_RAINFALL,   # None -> estimated
    average_temperature=AVERAGE_TEMPERATURE,     # None -> estimated
)
rainfall_ds = replicates
rainfall_ds

# %% [markdown]
# ### 30-minute rainfall for RUSLE

# %%
rainfall_30min = aggregate_rainfall_data(rainfall_ds, rain_data_start, rain_data_end)
rainfall_30min

# %% [markdown]
# ### 12-minute rainfall intensity for debris flow

# %%
rainfall_intensity = convert_rainfall_depth_to_intensity(rainfall_ds)
rainfall_12min = aggregate_rainfall_data(
    rainfall_intensity, rain_data_start, rain_data_end, time_res='12min',
)
rainfall_12min

# %% [markdown]
# ## Erosion — ensemble RUSLE simulation
#
# `run_rusle_all_replicates` runs every replicate in parallel. Recovery
# windows are applied internally by run_usle_simulation, so each replicate
# yields a single continuous result. The grid recorders use
# `grid_timesteps=('total',)` — one whole-period grid:
#
# * Total erosion grid over the window (`RUSLE_sum_total`)
# * Peak 30-min erosion grid over the window (`RUSLE_max_total`)
# * Daily subcatchment-level erosion timeseries (`erosion_daily_time_series`)

# %%
recorder_factory = default_rusle_recorders(
    include_timeseries=True,
    grid_timesteps=('total',),
)

# %%
# Run every replicate in parallel. Recovery windows are applied internally
# by run_usle_simulation, so the result is the standard
# {replicate: {catchment: recorder-results}}.
rusle_results = run_rusle_all_replicates(
    ctx,
    rainfall_30min,
    n_workers=min(N_REPLICATES, N_WORKERS),
    recorder_factory=recorder_factory,
)

# %% [markdown]
# ### Baseline (no-fire) ensemble
#
# Re-run the ensemble using the **unadjusted** C and K factors to
# produce a pre-fire baseline for each replicate. Differencing
# ``rusle_results`` against ``baseline_results`` isolates the
# fire-attributable component of the erosion response under the same
# stochastic rainfall.

# %%
baseline_results = run_rusle_all_replicates(
    ctx,
    rainfall_30min,
    n_workers=min(N_REPLICATES, N_WORKERS),
    recorder_factory=recorder_factory,
    use_fire_adjusted=False,
)

# %% [markdown]
# ### Ensemble statistics (median / P90 / IQR)
#
# Publication-quality three-panel map of the selected recovery window's
# total erosion, with a shared colour scale clipped to the 99th percentile
# to avoid extreme outliers dominating.

# %%
# CELL_AREA_HA follows catchment.cell_size_m, set at the top of the
# notebook — the per-hectare figures below are only right if it matches
# the DEM this project was built from.
plot_ensemble_statistics_panel(
    rusle_results,
    'RUSLE_sum_total',
    catchment=CATCHMENT,
    time=None,
    project=proj,
    cell_area_ha=CELL_AREA_HA,
    units='t / ha',
)
plt.show()

# %% [markdown]
# ### Exceedance probability map
#
# For each grid cell, what fraction of replicates exceed a policy
# threshold?  The threshold is `reporting.erosion_threshold_t_ha` in
# `study.toml`, in tonnes per hectare; the recorded grids are per cell,
# so it is converted using the cell area set at the top.

# %%
THRESHOLD_PER_CELL = EROSION_THRESHOLD_T_HA * CELL_AREA_HA

prob = exceedance_probability(
    rusle_results, 'RUSLE_sum_total', THRESHOLD_PER_CELL,
    catchment=CATCHMENT, time=None,
)
ax = plot_exceedance(prob, project=proj, catchment=CATCHMENT)
ax.set_title(
    f'P(erosion > {EROSION_THRESHOLD_T_HA:g} t/ha)  '
    f'(n={N_REPLICATES} replicates)'
)
plt.show()

# %% [markdown]
# ### Catchment-lumped exceedance curve (AEP)

# %%
plot_catchment_exceedance_curve(
    rusle_results,
    'RUSLE_sum_total',
    catchment=CATCHMENT,
    time=None,
    scale=1e-3,
    value_units='thousand tonnes',
)
plt.show()

# %% [markdown]
# ### Ensemble daily timeseries ribbon

# %%
plot_ensemble_daily_ribbon(
    rusle_results,
    catchment=CATCHMENT,
    timeseries_key='erosion_daily_time_series',
)
plt.show()

# %% [markdown]
# ## Debris flow — ensemble simulation
#
# `run_debris_flow_all_replicates` parallelises `debris_flow` across
# replicates and converts the per-headwater event counts into mass
# timeseries (kg).  `postprocess_debris_flow` then allocates each
# headwater to user-defined subcatchments (spatial overlay is computed
# once and reused across replicates).

# %%
debris_results = run_debris_flow_all_replicates(
    ctx,
    rainfall_12min,
    n_workers=min(N_REPLICATES, N_WORKERS),
)

# %%
debris_mass_per_replicate = {
    rep: res[CATCHMENT][1] for rep, res in debris_results.items()
}

# %% [markdown]
# Post-process the debris-flow results to subcatchment scale.  The
# spatial overlay between headwaters and subcatchments is computed
# once and reused across every replicate.  We keep the result at the
# native 12-minute resolution so we can later aggregate freely to
# any coarser resolution without losing information.

# %%
debris_post = postprocess_debris_flow(
    ctx, debris_mass_per_replicate, save=False,
)
sc_debris_12min = debris_post['aggregated']

# %% [markdown]
# ## Combined RUSLE + debris-flow load at the subcatchment scale
#
# RUSLE subcatchment outputs are in tonnes; debris-flow outputs are in
# kilograms.  `combine_rusle_and_debris_subcatchment` rescales RUSLE to
# kg, sums the two, aggregates to a requested temporal resolution, and
# relabels the columns using a string attribute from the subcatchment
# coverage — the one named by `catchment.subcatchment_id_field` in
# `study.toml`.
#
# The available temporal resolutions cover the common use cases:
#
# * `freq='total'` — a single row summing the entire simulation.
# * `freq='YS'` — annual totals (typical for reporting).
# * `freq='MS'` — monthly totals.
# * `freq='D'`  — daily loads (commonly linked to downstream sediment
#   transport models).
# * `freq='h'`  — hourly loads; requires RUSLE to have been recorded at
#   hourly resolution (``default_rusle_recorders(timeseries_timestep='1h')``)
#   to avoid artificial smoothing of the erosion signal.

# %% [markdown]
# The label field is normally captured when the subcatchment coverage is
# registered — the *Simulation* notebook does that with
# ``add_subcatchments(..., label_field=...)`` — and is then read back from
# the project's ``settings.json``. If it was missed, the cell below
# registers `catchment.subcatchment_id_field` now, and it is persisted for
# every future session against this project.
#
# `catchment.subcatchments` is optional: leave that setting out and this
# cell does nothing, exactly as the equivalent cell in *Simulation* does.
#
# A field already registered is left alone — the persisted value wins, so
# editing `catchment.subcatchment_id_field` in `study.toml` does not
# change it from here. Re-run the *Simulation* notebook's subcatchment
# cell to change it.
#
# > This notebook needs a subcatchment coverage to have been registered
# > already. If you have not run *Simulation*, set
# > `catchment.subcatchments` in `study.toml` and run its subcatchment
# > cell first; without it the combined-load cells below have nothing to
# > aggregate to.

# %%
if SUBCATCHMENTS and proj.subcatchment_label_field(CATCHMENT) is None:
    proj.set_subcatchment_label_field(CATCHMENT, SUBCATCHMENT_ID_FIELD)

# %%
def combine_at(freq):
    return combine_rusle_and_debris_subcatchment(
        rusle_results,
        sc_debris_12min,
        project=proj,
        catchment=CATCHMENT,
        freq=freq,
    )

combined_total  = combine_at('total')
combined_annual = combine_at('YS')
combined_daily  = combine_at('D')

# %%
combined_annual[next(iter(combined_annual))]

# %% [markdown]
# ### Ensemble mean across replicates
#
# A simple arithmetic mean over the replicate dimension for whichever
# resolution you want to report.

# %%
def ensemble_mean(per_replicate):
    return sum(per_replicate.values()) / len(per_replicate)

mean_annual_kg = ensemble_mean(combined_annual)
mean_annual_kg

# %% [markdown]
# ### Subcatchment choropleth maps
#
# `plot_subcatchment_ensemble` turns any ``{replicate: wide DataFrame}``
# dict into a choropleth of the subcatchment coverage.  It accepts any
# reduction (`'mean'`, `'median'`, `('quantile', q)`,
# `('exceedance', threshold)`, or a callable) and can normalise per
# replicate *before* the reduction — so
# ``reduction=('exceedance', 500)`` with ``normalise_by='area_ha'``
# correctly answers *"probability that per-hectare load exceeds
# 500 kg/ha"*.
#
# The same function works on the combined load, RUSLE only, or debris
# only: `rusle_subcatchment_ensemble` and `debris_subcatchment_ensemble`
# produce the same ``{replicate: DataFrame}`` shape as
# `combine_rusle_and_debris_subcatchment` so the plotter treats them
# interchangeably.
#
# Exceedance plots default to a locked `[0, 1]` colour scale so maps
# for different thresholds or years are directly comparable; pass
# `vmin=` / `vmax=` to override.
#
# The three exceedance maps below are all drawn at
# `reporting.delivered_threshold_kg_ha` from `study.toml`, so changing
# that setting moves all of them together — titles included.

# %%
# Ensemble mean, year 1 (data-derived colour scale):
plot_subcatchment_ensemble(
    combined_annual, project=proj, catchment=CATCHMENT,
    time=0, reduction='mean', normalise_by='area_ha', units='kg',
    title='Year 1 mean load (kg/ha)',
)
plt.show()

# P(year 1 combined load > the reporting threshold):
plot_subcatchment_ensemble(
    combined_annual, project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', DELIVERED_THRESHOLD_KG_HA),
    normalise_by='area_ha',
    cmap='RdYlGn_r',
    title=f'P(Year 1 combined load > {DELIVERED_THRESHOLD_KG_HA:g} kg/ha)',
)
plt.show()

# Same threshold against RUSLE only. These two need `project=` as well as
# `catchment=`: that is how they find the registered label field, and
# without it their columns come back as raw subcatchment IDs, which the
# plotter cannot match to the labelled coverage.
plot_subcatchment_ensemble(
    rusle_subcatchment_ensemble(
        rusle_results, project=proj, catchment=CATCHMENT),
    project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', DELIVERED_THRESHOLD_KG_HA),
    normalise_by='area_ha',
    cmap='RdYlGn_r',
    title=f'P(Year 1 RUSLE load > {DELIVERED_THRESHOLD_KG_HA:g} kg/ha)',
)
plt.show()

# ...and against debris flow only:
plot_subcatchment_ensemble(
    debris_subcatchment_ensemble(
        sc_debris_12min, project=proj, catchment=CATCHMENT),
    project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', DELIVERED_THRESHOLD_KG_HA),
    normalise_by='area_ha',
    cmap='RdYlGn_r',
    title=f'P(Year 1 debris load > {DELIVERED_THRESHOLD_KG_HA:g} kg/ha)',
)
plt.show()

# %% [markdown]
# ## Save the ensemble run for downstream modelling
#
# The combined loads and the driving rainfall are the two inputs a
# broader sediment-transport model needs.  `save_ensemble_run` writes:
#
# * rainfall to ``Catchments/<c>/Ensembles/<ensemble>/`` (climate-only,
#   shareable across events), and
# * everything else to ``Catchments/<c>/Runs/<event>/<ensemble>/``.
#
#
# The same RunContext used for the simulation drives the save —
# rainfall lands under Ensembles/<ensemble>/ (climate-only, shareable
# across events) while run outputs land under Runs/<event>/<ensemble>/.
#
# > The *SourceIntegration* notebook reads the loads back at the
# > resolution named by `source.timestep` in `study.toml`. That has to be
# > one of the frequencies saved below — daily (`'D'`) as it stands. If
# > you set `source.timestep` to something else, add it here as well, or
# > SourceIntegration will not find anything to load.

# %%
save_ensemble_run(
    ctx,
    rainfall_ds=rainfall_ds,
    rusle_results=rusle_results,
    debris_results=debris_results,
    combined_by_freq={
        'total': combined_total,
        'YS':    combined_annual,
        'D':     combined_daily,
    },
    include_rusle_grids=False,   # opt in when you need raw grids
    include_raw_debris=False,    # opt in for per-headwater debris series
    # extra_manifest={                       # add any custom metadata
    #     'mean_annual_rainfall_mm': 600,    # to record alongside the
    #     'average_temperature_c': 20,       # ensemble run
    # },
)
