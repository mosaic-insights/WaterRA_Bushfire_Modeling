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
# ## Load project

# %%
proj = FireImpactsProject('.', exist_ok=True)
proj.catchments

# %% [markdown]
# A `FireImpactsProject` can host multiple study catchments; in this
# example we assume just one and take the first.

# %%
CATCHMENT = proj.catchments[0]
CATCHMENT

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
# supplied.  Pass them explicitly when you have site-specific values.
#
# The same set of replicates feeds both simulations.

# %%
# The simulation period spans the recovery windows recorded in the
# run-context (fire end date -> end of the last window), so it isn't
# hard-coded here.
rain_data_start, rain_data_end = proj.get_simulation_period(CATCHMENT)

N_REPLICATES = 10

# %%
# `num_years` is inferred from start/end (one API year per calendar
# year spanned).  Uncomment the climate kwargs to override the
# backend-estimated values.
replicates = get_rainfall_replicates(
    proj,
    catchment=CATCHMENT,
    start=rain_data_start,
    end=rain_data_end,
    num_replicates=N_REPLICATES,
    # mean_annual_rainfall=600,   # mm  — optional
    # average_temperature=20,     # °C  — optional
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
    proj,
    rainfall_30min,
    n_workers=min(N_REPLICATES, 10),
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
    proj,
    rainfall_30min,
    n_workers=min(N_REPLICATES, 10),
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
CELL_AREA_HA = 30 * 30 / 10_000  # nominal 30 m cell
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
# threshold?

# %%
THRESHOLD_T_HA = 0.5
THRESHOLD_PER_CELL = THRESHOLD_T_HA * CELL_AREA_HA

prob = exceedance_probability(
    rusle_results, 'RUSLE_sum_total', THRESHOLD_PER_CELL,
    catchment=CATCHMENT, time=None,
)
ax = plot_exceedance(prob, project=proj, catchment=CATCHMENT)
ax.set_title(
    f'P(erosion > {THRESHOLD_T_HA} t/ha)  (n={N_REPLICATES} replicates)'
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
    proj,
    rainfall_12min,
    n_workers=min(N_REPLICATES, 10),
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
    proj, CATCHMENT, debris_mass_per_replicate, save=False,
)
sc_debris_12min = debris_post['aggregated']

# %% [markdown]
# ## Combined RUSLE + debris-flow load at the subcatchment scale
#
# RUSLE subcatchment outputs are in tonnes; debris-flow outputs are in
# kilograms.  `combine_rusle_and_debris_subcatchment` rescales RUSLE to
# kg, sums the two, aggregates to a requested temporal resolution, and
# relabels the columns using a string attribute from the subcatchment
# shapefile (``SiteID`` by default).
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
# The preferred string label field (e.g. ``'SiteID'``) is typically
# captured by ``add_subcatchments(..., label_field='SiteID')`` at
# project setup and is picked up automatically from ``settings.json``.
# If it was missed, register it now — it will be persisted for reuse
# across all future sessions against this project.

# %%
if proj.subcatchment_label_field(CATCHMENT) is None:
    proj.set_subcatchment_label_field(CATCHMENT, 'SiteID')

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

# %%
# Ensemble mean, year 1 (data-derived colour scale):
plot_subcatchment_ensemble(
    combined_annual, project=proj, catchment=CATCHMENT,
    time=0, reduction='mean', normalise_by='area_ha', units='kg',
    title='Year 1 mean load (kg/ha)',
)
plt.show()

# P(year 1 combined load > 500 kg/ha):
plot_subcatchment_ensemble(
    combined_annual, project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', 500), normalise_by='area_ha',
    cmap='RdYlGn_r',
    title='P(Year 1 combined load > 500 kg/ha)',
)
plt.show()

# Same threshold against RUSLE only:
plot_subcatchment_ensemble(
    rusle_subcatchment_ensemble(rusle_results, catchment=CATCHMENT),
    project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', 500), normalise_by='area_ha',
    cmap='RdYlGn_r',
    title='P(Year 1 RUSLE load > 500 kg/ha)',
)
plt.show()

# ...and against debris flow only:
plot_subcatchment_ensemble(
    debris_subcatchment_ensemble(sc_debris_12min, catchment=CATCHMENT),
    project=proj, catchment=CATCHMENT,
    time=0, reduction=('exceedance', 500), normalise_by='area_ha',
    cmap='RdYlGn_r',
    title='P(Year 1 debris load > 500 kg/ha)',
)
plt.show()

# %% [markdown]
# ## Save the ensemble run for downstream modelling
#
# The combined loads and the driving rainfall are the two inputs a
# broader sediment-transport model needs.  `save_ensemble_run` writes
# both (plus optional debris-flow raw outputs) into a library-managed
# directory under the project::
#
#     Catchments/<catchment>/Events/<event>/Ensemble/<ensemble>/
#
# `event` and `ensemble` both default to ``'default'``.  Use distinct
# ensemble names to compare scenarios against the same fire event —
# e.g. ``ensemble='current_climate'`` vs
# ``ensemble='future_climate_2050'``.

# %%
EVENT = 'default'
ENSEMBLE = 'default'

save_ensemble_run(
    proj, CATCHMENT,
    rainfall_ds=rainfall_ds,
    rusle_results=rusle_results,
    debris_results=debris_results,
    combined_by_freq={
        'total': combined_total,
        'YS':    combined_annual,
        'D':     combined_daily,
    },
    event=EVENT,
    ensemble=ENSEMBLE,
    include_rusle_grids=False,   # opt in when you need raw grids
    include_raw_debris=True,
    # extra_manifest={                       # add any custom metadata
    #     'mean_annual_rainfall_mm': 600,    # to record alongside the
    #     'average_temperature_c': 20,       # ensemble run
    # },
)
