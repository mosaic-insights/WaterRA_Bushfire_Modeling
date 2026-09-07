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
#    plugin loaded, and Veneer is running on `source.port` — 9876 unless
#    you change it in `study.toml`.
# 3. The Source project has a **constituent** you want to use for the
#    fire-derived sediment load — typically `TSS`.  If one is not
#    already defined, create it in Source before running this notebook.
#    The notebook auto-detects a likely candidate; name it in
#    `source.constituent` if the guess is wrong.
# 4. Subcatchment names in Source match the labels used in the ensemble
#    output columns — the attribute named by
#    `catchment.subcatchment_id_field` in `study.toml`. That setting is
#    what ties the two models together, so it is worth checking before
#    you run anything here.

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
    load_ensemble_manifest,
    load_ensemble_rainfall,
)
from fire_impacts.source import (
    connect_to_veneer,
    check_load_distributor_plugin,
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
# ## Settings for this study
#
# From `study.toml`, the same file the other notebooks read. The
# `[source]` section is the one that matters here: it says which Source
# instance to talk to, which replicate to push, and what the two data
# sources this notebook creates are called.
#
# `constituent` and `functional_unit` are optional — left unset, the
# notebook auto-detects them from the running Source model and shows you
# what it picked.
#
# Running the cell prints each setting and where it came from, so you can
# check the notebook is about to do what you expect.

# %%
from fire_impacts.study import load_study

study = load_study('.')
study.describe()

PROJECT_DIR = study.project.directory
CATCHMENT   = study.catchment.name
EVENT       = study.event.name
ENSEMBLE    = study.ensemble.name
SUBCATCHMENT_ID_FIELD = study.catchment.subcatchment_id_field

PORT        = study.source.port
SOURCE_REPLICATE = study.source.replicate
TIMESTEP    = study.source.timestep
DATE_FORMAT = study.source.date_format
OUTPUT_DIR  = study.source.output_dir
LOAD_ATTENUATION      = study.source.load_attenuation
MAXIMUM_CONCENTRATION = study.source.maximum_concentration

# Named once each. The notebook creates these data sources and then reads
# them back, so the two references have to agree.
TSS_SOURCE      = study.source.tss_data_source
RAINFALL_SOURCE = study.source.rainfall_data_source

# Source labels its data sources with a unit word ('kg/day'), while
# source.timestep is a pandas frequency ('D'). Translate, and refuse a
# frequency we have no word for rather than labelling it wrongly: Source
# reads these labels, so a wrong one is a silent scaling error.
_UNIT_WORDS = {'D': 'day', 'h': 'hour'}
if TIMESTEP not in _UNIT_WORDS:
    raise ValueError(
        f'source.timestep = {TIMESTEP!r} in study.toml has no units word; '
        f'use one of {sorted(_UNIT_WORDS)}.')
TIMESTEP_UNITS = _UNIT_WORDS[TIMESTEP]

# %% [markdown]
# ## Load project and choose an ensemble run
#
# The project may host multiple catchments, events and ensembles.  The
# helpers below list what is available; the one this notebook uses is
# named by `catchment.name`, `event.name` and `ensemble.name` in
# `study.toml`.

# %%
proj = FireImpactsProject(PROJECT_DIR, exist_ok=True)
proj.catchments

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
# Build a run-level RunContext for the combination named in study.toml.
ctx = RunContext.solo_run(
    proj, event=EVENT, ensemble=ENSEMBLE,
    catchment=CATCHMENT,
)

# %% [markdown]
# ## Load ensemble outputs
#
# The ensemble run produced per-replicate combined TSS loads
# (RUSLE + debris flow, in **kg**) at several temporal resolutions and
# the matching stochastic rainfall.  `source.timestep` in `study.toml`
# picks the resolution, and it has to match your Source model: `'D'`, the
# default, for a daily model, `'h'` for an hourly one.
#
# Whichever you choose, the *SimulationEnsemble* notebook must have saved
# that resolution — it saves `'total'`, `'YS'` and `'D'` as it stands —
# and an hourly Source model also needs the ensemble to have been run
# with `default_rusle_recorders(timeseries_timestep='1h')`, or the
# erosion signal is smoothed on the way in.
#
# The variables below are named `..._daily` after the common case; they
# hold whatever resolution `source.timestep` names.

# %%
combined_daily = load_ensemble_combined(ctx, freq=TIMESTEP)
list(combined_daily)[:5], next(iter(combined_daily.values())).shape

# %%
# The ensemble run numbers its replicates, and source.replicate picks one
# of them to push into Source. Checked here rather than left to fail as a
# bare KeyError in Part B, which says nothing about study.toml.
if SOURCE_REPLICATE not in combined_daily:
    raise ValueError(
        f'source.replicate is {SOURCE_REPLICATE}, but this run holds '
        f'replicates {sorted(combined_daily)}. '
        f'Lower source.replicate in study.toml, '
        f'or re-run SimulationEnsemble with more replicates. Note that '
        f'source.replicate is the one pushed into Source; '
        f'ensemble.inspect_replicate is a different setting, and only '
        f'chooses what the Simulation notebook plots.')

# %%
# One column per subcatchment. These are the names Source has to know the
# same subcatchments by — worth reading before you go further, because a
# mismatch here leaves loads unassigned rather than failing.
#
# The label field is read back from the run's manifest rather than from
# study.toml: what labelled these columns is whatever was registered when
# the ensemble was saved, which is not necessarily what
# `catchment.subcatchment_id_field` says today.
manifest = load_ensemble_manifest(ctx)
labelled_by = manifest.get('subcatchment_label_field')

if labelled_by is None:
    # No label field was registered when the ensemble was saved, so the
    # columns are the project's raw subcatchment IDs rather than names a
    # Source model is likely to share.
    logging.warning(
        'This run recorded no subcatchment label field, so the columns '
        'below are raw subcatchment IDs. Set catchment.subcatchments and '
        'catchment.subcatchment_id_field in study.toml and re-run '
        'SimulationEnsemble to label them.')
elif labelled_by != SUBCATCHMENT_ID_FIELD:
    logging.warning(
        f'These loads were labelled by {labelled_by!r}, but '
        f'catchment.subcatchment_id_field is now '
        f'{SUBCATCHMENT_ID_FIELD!r}. Re-run SimulationEnsemble if you '
        f'meant to relabel them.')

print(f'Load columns are labelled by {labelled_by!r}:')
list(next(iter(combined_daily.values())).columns)

# %% [markdown]
# Rainfall comes back as an xarray `Dataset` with `simulation x day x
# subday` dimensions.  Flatten to a `time x simulation` DataFrame and
# aggregate to totals in mm at the same `source.timestep` as the loads,
# so the two line up.

# %%
rainfall_ds = load_ensemble_rainfall(ctx)
rainfall_ds

# %%
from fire_impacts.sim import aggregate_rainfall_data
from fire_impacts.sim.rainfall import convert_rainfall_to_dataframe

rain_start = str(pd.to_datetime(rainfall_ds['time'].values[0]).date())
rain_end = str(pd.to_datetime(rainfall_ds['time'].values[-1]).date())

rainfall_daily_ds = aggregate_rainfall_data(
    rainfall_ds, rain_start, rain_end, time_res=TIMESTEP,
)
rainfall_daily = convert_rainfall_to_dataframe(rainfall_daily_ds)
rainfall_daily.head()

# %% [markdown]
# ## Connect to Source (Veneer)
#
# Source must already be open with your project loaded and Veneer running
# on `source.port` — 9876 unless you changed it in `study.toml`.

# %%
v = connect_to_veneer(port=PORT)
v.scenario_info()

# %% [markdown]
# Confirm the [Load Distributor](https://github.com/flowmatters/source-loaddistributor)
# plugin is loaded.  Raises if the running Source instance reports a plugin
# list without it; warns (but does not fail) on older Veneer releases that
# don't expose a `Plugins` field in `scenario_info()`.

# %%
check_load_distributor_plugin(v)

# %% [markdown]
# ## Pick the constituent and functional unit
#
# `detect_constituent` and `detect_functional_unit` pick the most
# likely candidates (`TSS`, `Forested` etc.) out of the running Source
# model. Set `source.constituent` or `source.functional_unit` in
# `study.toml` to override them; leave either one out and it is
# auto-detected. The cell prints what it ended up with either way.

# %%
CONSTITUENT = study.source.constituent or detect_constituent(v)
FUNCTIONAL_UNIT = study.source.functional_unit or detect_functional_unit(v)
CONSTITUENT, FUNCTIONAL_UNIT

# %% [markdown]
# Wrong pick? Set `constituent` / `functional_unit` in the `[source]`
# section of `study.toml` and re-run the cell above.

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
    load_attenuation=LOAD_ATTENUATION,
    maximum_concentration=MAXIMUM_CONCENTRATION,
)

# %% [markdown]
# # Part B — single replicate
#
# Push one replicate's loads and rainfall into Source as in-memory data
# sources, wire them up and run.  This is the quickest way to confirm the
# model is wired up correctly end-to-end before looping over all
# replicates.  `source.replicate` in `study.toml` chooses which one.

# %%
# The 'rainfall' below is the *column* name inside the data source, which
# Source's runoff models look for by that name. It is not the name of the
# data source itself — that is RAINFALL_SOURCE.
tss_single = combined_daily[SOURCE_REPLICATE]
rain_single = rainfall_daily[[SOURCE_REPLICATE]].rename(
    columns={SOURCE_REPLICATE: 'rainfall'})
tss_single.head(), rain_single.head()

# %% [markdown]
# Units note: the loads are **kg** per `source.timestep`, and
# `aggregate_rainfall_data` returns rainfall depth in **mm** over that
# same interval, so the two are on the same footing.
#
# Source is told as much: the data sources below are labelled
# `kg/{TIMESTEP_UNITS}` and `mm/{TIMESTEP_UNITS}` — `kg/day` and `mm/day`
# by default — and Part A labels its CSV-backed sources the same way.
# Source reads those labels, so they have to follow `source.timestep`
# rather than assume daily.

# %%
create_veneer_data_sources(
    v, tss_single, rain_single,
    tss_source_name=TSS_SOURCE,
    rainfall_source_name=RAINFALL_SOURCE,
    timestep=TIMESTEP_UNITS,
)

# %%
assign_fire_sediment_timeseries(
    v, tss_source_name=TSS_SOURCE,
    constituent=CONSTITUENT, functional_unit=FUNCTIONAL_UNIT,
)
assign_rainfall_timeseries(v, rainfall_source_name=RAINFALL_SOURCE)

# %% [markdown]
# Run the Source simulation over the period of the data.  Source expects
# the run-period dates in the format your install writes them — `dd/mm/yyyy`
# unless you change `source.date_format` in `study.toml`.

# %%
start = tss_single.index[0].strftime(DATE_FORMAT)
end = tss_single.index[-1].strftime(DATE_FORMAT)

sim_results = run_model_simulation(v, start_date=start, end_date=end)
sim_results['Status']

# %%
save_model(v, f'{CATCHMENT}_with_fire_inputs_rep{SOURCE_REPLICATE:02d}.rsproj')

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
source_inputs_dir = Path(ctx.ensemble_path()) / OUTPUT_DIR
source_inputs_dir.mkdir(parents=True, exist_ok=True)

# Source names a file-backed data source after the CSV's filename stem,
# so the files are named after the two data sources — that is what makes
# the names Source registers agree with the ones assigned below.
tss_csv = source_inputs_dir / f'{TSS_SOURCE}.csv'
rain_csv = source_inputs_dir / f'{RAINFALL_SOURCE}.csv'

# %% [markdown]
# Seed the two CSVs with one replicate's data so the data sources can be
# created with valid content.  The loop below will overwrite them
# per-replicate.

# %%
def write_replicate_csvs(rep: int):
    """Overwrite the two on-disk CSVs with data for a given replicate."""
    loads = combined_daily[rep]
    rain = rainfall_daily[[rep]].rename(columns={rep: 'rainfall'})
    loads.to_csv(tss_csv)
    rain.to_csv(rain_csv)

write_replicate_csvs(SOURCE_REPLICATE)

# %% [markdown]
# ### Recreate the data sources backed by the on-disk CSVs
#
# Delete the in-memory data sources Part B created and recreate them
# under the same names with `reload_on_run=True`, so Source re-reads the
# CSV at every run.

# %%
for name in (TSS_SOURCE, RAINFALL_SOURCE):
    try:
        v.delete_data_source(name)
    except Exception as e:
        logging.info(f'(No existing data source {name} to remove: {e})')

# %%
# No inline data this time, so Source takes each data source's name from
# the CSV filename stem — which is why the files were named after
# `source.tss_data_source` and `source.rainfall_data_source` above.
v.create_data_source(
    str(tss_csv), units=f'kg/{TIMESTEP_UNITS}', reload_on_run=True,
)
v.create_data_source(
    str(rain_csv), units=f'mm/{TIMESTEP_UNITS}', reload_on_run=True,
)

# Confirm the names Source registered: these should be TSS_SOURCE and
# RAINFALL_SOURCE. If they are not, your Source version derives a name
# from a filename differently. Changing the study.toml settings will not
# help — they name the CSVs as well, so Source would just derive a new
# name from the new stem. Pass the names Source actually reports to the
# `tss_source_name` / `rainfall_source_name` arguments in the cell below
# instead.
[ds['Name'] for ds in v.data_sources()]

# %% [markdown]
# Re-wire Source's Load Distributor inputs and rainfall inputs to point at
# those two names.  This only needs to be done once — the assignments
# persist across runs; only the CSV content changes.

# %%
assign_fire_sediment_timeseries(
    v, tss_source_name=TSS_SOURCE,
    constituent=CONSTITUENT, functional_unit=FUNCTIONAL_UNIT,
)
assign_rainfall_timeseries(v, rainfall_source_name=RAINFALL_SOURCE)

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
    start = combined_daily[rep].index[0].strftime(DATE_FORMAT)
    end = combined_daily[rep].index[-1].strftime(DATE_FORMAT)
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
