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
# # Fire impacts — integration with eWater Source
#
# This notebook drives an **eWater Source** catchment model with the
# sediment loads and rainfall produced by the fire impacts ensemble.
# Subcatchment-scale TSS loads (combined RUSLE + debris-flow) are pushed
# into Source as time-series inputs to the **Load Distributor** plugin,
# and the matching stochastic rainfall is assigned to the
# rainfall-runoff model.
#
# The notebook has two parts:
#
# * **Part B — single replicate.**  The simplest end-to-end flow: pick
#   one replicate from the ensemble, load its data into Source, run,
#   save.  Good for validating the wiring against the model before
#   running the full ensemble.
# * **Part A — full ensemble via ReloadOnRun CSVs.**  Create the two
#   Source data sources once, each backed by a CSV on disk with
#   `ReloadOnRun=True`.  Each iteration overwrites the CSVs and
#   re-runs Source — the idiomatic Source pattern for swapping inputs
#   between runs.
#
# ## Prerequisites
#
# 1. You have already run `SimulationEnsemble.py` (or equivalent) against
#    this `FireImpactsProject`. The combined loads live under
#    `Catchments/<catchment>/Runs/<event>/<ensemble>/` and the driving
#    rainfall under `Catchments/<catchment>/Ensembles/<ensemble>/`; the
#    RunContext built below resolves both.
# 2. Your Source project is open in Source with the
#    [**Load Distributor**](https://github.com/flowmatters/source-loaddistributor)
#    plugin loaded, and Veneer is running (default port `9876`).
# 3. The Source project has a **constituent** you want to use for the
#    fire-derived sediment load — typically `TSS`.  If one is not
#    already defined, create it in Source before running this notebook.
#    The notebook will auto-detect a likely candidate but you can
#    override.
# 4. Subcatchment names in Source match the `SiteID` labels used in the
#    ensemble output columns (set at project setup via
#    `add_subcatchments(..., label_field='SiteID')`).

# %%
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
from pathlib import Path

import pandas as pd

from fire_impacts import FireImpactsProject
from fire_impacts.context import RunContext
from fire_impacts.sim import (
    list_events,
    list_ensembles,
    list_runs,
    load_ensemble_combined,
    load_ensemble_rainfall,
)
from fire_impacts.source import (
    connect_to_veneer,
    detect_constituent,
    detect_functional_unit,
    configure_load_distributor_model,
    create_veneer_data_sources,
    assign_fire_sediment_timeseries,
    assign_rainfall_timeseries,
    run_model_simulation,
    save_model,
)

# %% [markdown]
# ## Load project and choose an ensemble run
#
# The project may host multiple catchments, events and ensembles.  The
# helpers below list what is available so you can pick one.

# %%
proj = FireImpactsProject('.', exist_ok=True)
CATCHMENT = proj.catchments[0]
CATCHMENT

# %%
list_events(proj, CATCHMENT)

# %%
# Ensembles are siblings of events (the same rainfall realisation can
# drive multiple fires), so list_ensembles takes no event argument.
list_ensembles(proj, CATCHMENT)

# %%
# list_runs returns the (event, ensemble) tuples that have already
# been executed and have outputs on disk.
list_runs(proj, CATCHMENT)

# %%
# Pick one combination and build a run-level RunContext.
ctx = RunContext.solo_run(
    proj, event='2019_fire', ensemble='stochastic',
    catchment=CATCHMENT,
)

# %% [markdown]
# ## Load ensemble outputs
#
# The ensemble run produced per-replicate combined TSS loads
# (RUSLE + debris flow, in **kg**) at several temporal resolutions and
# the matching stochastic rainfall.  Source is most commonly run on a
# daily timestep, so we use `freq='D'` here.  Switch to `'h'` if your
# Source model runs hourly and the ensemble was saved at hourly
# resolution (requires `default_rusle_recorders(timeseries_timestep='1h')`
# when generating the ensemble).

# %%
combined_daily = load_ensemble_combined(ctx, freq='D')
list(combined_daily)[:5], next(iter(combined_daily.values())).shape

# %% [markdown]
# Rainfall comes back as an xarray `Dataset` with `simulation x day x
# subday` dimensions.  Flatten to a `time x simulation` DataFrame and
# aggregate to daily totals in mm so it matches the daily loads.

# %%
rainfall_ds = load_ensemble_rainfall(ctx)
rainfall_ds

# %%
from fire_impacts.sim import aggregate_rainfall_data
from fire_impacts.sim.rainfall import convert_rainfall_to_dataframe

rain_start = str(pd.to_datetime(rainfall_ds['time'].values[0]).date())
rain_end = str(pd.to_datetime(rainfall_ds['time'].values[-1]).date())

rainfall_daily_ds = aggregate_rainfall_data(
    rainfall_ds, rain_start, rain_end, time_res='D',
)
rainfall_daily = convert_rainfall_to_dataframe(rainfall_daily_ds)
rainfall_daily.head()

# %% [markdown]
# ## Connect to Source (Veneer)

# %%
PORT = 9876
v = connect_to_veneer(port=PORT)
v.scenario_info()

# %% [markdown]
# ## Pick the constituent and functional unit
#
# `detect_constituent` and `detect_functional_unit` pick the most
# likely candidates (`TSS`, `Forested` etc.).  Override by assigning
# the variables directly below if the defaults are wrong.

# %%
CONSTITUENT = detect_constituent(v)
FUNCTIONAL_UNIT = detect_functional_unit(v)
CONSTITUENT, FUNCTIONAL_UNIT

# %%
# Override if the auto-detection picked the wrong option:
# CONSTITUENT = 'TSS'
# FUNCTIONAL_UNIT = 'Forested'

# %% [markdown]
# ## Configure the Load Distributor model
#
# Switches every subcatchment/functional-unit combination for the chosen
# constituent onto the `DistributeLoadModel`, and sets attenuation and
# concentration cap parameters.

# %%
configure_load_distributor_model(
    v,
    constituent=CONSTITUENT,
    load_attenuation=10.0,
    maximum_concentration=1000.0,
)

# %% [markdown]
# # Part B — single replicate
#
# Pick one replicate, push its loads and rainfall into Source as
# in-memory data sources, wire them up and run.  This is the quickest
# way to confirm the model is wired up correctly end-to-end before
# looping over all replicates.

# %%
REPLICATE = 0

tss_single = combined_daily[REPLICATE]
rain_single = rainfall_daily[[REPLICATE]].rename(columns={REPLICATE: 'rainfall'})
tss_single.head(), rain_single.head()

# %% [markdown]
# Units note: `combined_daily` is in **kg/day**; `aggregate_rainfall_data`
# returns rainfall depth in **mm/day** after the daily resample.

# %%
create_veneer_data_sources(
    v, tss_single, rain_single,
    tss_source_name='fire_tss',
    rainfall_source_name='stochastic_rain',
)

# %%
assign_fire_sediment_timeseries(
    v, tss_source_name='fire_tss',
    constituent=CONSTITUENT, functional_unit=FUNCTIONAL_UNIT,
)
assign_rainfall_timeseries(v, rainfall_source_name='stochastic_rain')

# %% [markdown]
# Run the Source simulation over the period of the data.  The Source
# run-period dates are in `dd/mm/yyyy` format.

# %%
start = tss_single.index[0].strftime('%d/%m/%Y')
end = tss_single.index[-1].strftime('%d/%m/%Y')

sim_results = run_model_simulation(v, start_date=start, end_date=end)
sim_results['Status']

# %%
save_model(v, f'{CATCHMENT}_with_fire_inputs_rep{REPLICATE:02d}.rsproj')

# %% [markdown]
# # Part A — full ensemble via ReloadOnRun CSVs
#
# For the full ensemble we create the two Source data sources **once**,
# each pointing at a CSV file on disk with `ReloadOnRun=True`.  Each
# iteration of the loop overwrites the two CSVs and re-runs Source, so
# Source re-reads the inputs from disk at the start of every run.
#
# This avoids round-tripping large time-series payloads through Veneer
# on every replicate and is the idiomatic way Source users manage
# scenario inputs.

# %%
source_inputs_dir = Path(ctx.ensemble_path()) / 'source_inputs'
source_inputs_dir.mkdir(parents=True, exist_ok=True)

tss_csv = source_inputs_dir / 'fire_tss.csv'
rain_csv = source_inputs_dir / 'rainfall.csv'

# %% [markdown]
# Seed the two CSVs with the first replicate's data so the data sources
# can be created with valid content.  The loop below will overwrite
# them per-replicate.

# %%
def write_replicate_csvs(rep: int):
    """Overwrite the two on-disk CSVs with data for a given replicate."""
    loads = combined_daily[rep]
    rain = rainfall_daily[[rep]].rename(columns={rep: 'rainfall'})
    loads.to_csv(tss_csv)
    rain.to_csv(rain_csv)

write_replicate_csvs(0)

# %% [markdown]
# ### Recreate the data sources backed by the on-disk CSVs
#
# Delete any existing data sources from Part B and recreate them with
# `reload_on_run=True` so Source re-reads the CSV at every run.

# %%
for name in ('fire_tss', 'stochastic_rain'):
    try:
        v.delete_data_source(name)
    except Exception as e:
        logging.info(f'(No existing data source {name} to remove: {e})')

# %%
# Source derives the data-source name from the CSV filename stem when
# no inline data is provided — so `fire_tss.csv` becomes data source
# `fire_tss`, and `rainfall.csv` becomes `rainfall`.
v.create_data_source(
    str(tss_csv), units='kg/d', reload_on_run=True,
)
v.create_data_source(
    str(rain_csv), units='mm/d', reload_on_run=True,
)

# Verify the names Source registered — if these don't match what you
# expect, adjust the `tss_source_name` / `rainfall_source_name`
# arguments below accordingly.
[ds['Name'] for ds in v.data_sources()]

# %% [markdown]
# Source derives the data-source name from the CSV filename stem
# (`fire_tss`, `rainfall`).  Re-wire Source's Load Distributor inputs
# and rainfall inputs to point at those names.  This only needs to be
# done once — the assignments persist across runs; only the CSV content
# changes.

# %%
assign_fire_sediment_timeseries(
    v, tss_source_name='fire_tss',
    constituent=CONSTITUENT, functional_unit=FUNCTIONAL_UNIT,
)
assign_rainfall_timeseries(v, rainfall_source_name='rainfall')

# %% [markdown]
# ### Loop over replicates
#
# For each replicate: overwrite the CSVs, run Source, and keep a note
# of the returned run URL.  Source persists its own results in the
# running scenario — visualisation and extraction are covered in a
# follow-up notebook.

# %%
replicate_ids = sorted(combined_daily)
source_runs = {}

for rep in replicate_ids:
    logging.info(f'Replicate {rep:02d}: writing CSVs and running Source')
    write_replicate_csvs(rep)
    start = combined_daily[rep].index[0].strftime('%d/%m/%Y')
    end = combined_daily[rep].index[-1].strftime('%d/%m/%Y')
    result = run_model_simulation(v, start_date=start, end_date=end)
    source_runs[rep] = result
    logging.info(f'Replicate {rep:02d}: status={result.get("Status")}')

# %%
{rep: r.get('Status') for rep, r in source_runs.items()}

# %% [markdown]
# ## Save the configured Source project
#
# Save the project with the Load Distributor wiring and ReloadOnRun
# data sources in place, so it can be re-opened and re-run without
# having to repeat the configuration steps.

# %%
save_model(v, f'{CATCHMENT}_with_fire_inputs_ensemble.rsproj')

# %% [markdown]
# ## Next steps
#
# Visualising the per-replicate Source outputs (flow and constituent
# loads at gauges of interest) is the natural follow-on — to be added
# in a subsequent template.
