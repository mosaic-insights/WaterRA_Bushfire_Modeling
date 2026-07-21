"""
End-to-end integration test of the multi-event RunContext pipeline.

Drives the real example catchment (test_data/) through the whole
preprocessing -> simulation chain on a single catchment and single event,
then asserts that every artefact lands at the scope the unified-runcontext
design requires:

- catchment scope: fire-independent base layers (C/K/LS factor, baseline SDR)
- event scope:      per-recovery fire-adjusted layers (+ fire data, event.json)
- run scope:        simulation results

The remote severity path (DEA STAC + Land Cover mask) is replaced by
generate_synthetic_fire, and the pyraingen ensemble by a synthetic rainfall
Series, so the whole test runs offline. Marked ``slow`` (pysheds topography +
SDR + two simulations); deselect with ``-m 'not slow'``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fire_impacts
from fire_impacts import const as c
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre import topography, soil, rusle, synthetic_fire
from fire_impacts.sim import rusle as simr
from fire_impacts.context import RunContext

pytestmark = pytest.mark.slow

CATCHMENT_FILE = 'EgSmallCatchment_7899.shp'
SUBCATCHMENT_FILE = 'Subcatchments_EgSmall_7899.shp'
DEM_FILE = 'DEM_10m_EgSmallCatchment_7899.tif'
ARIDITY_FILE = 'AridityPT_EgSmallCatchment_7899.tif'
# Coarse C/K rasters (native ~5x5) — deliberately coarser than the DEM grid,
# which is what exercises the baseline-SDR alignment fix (see the regression
# test below).
C_FACTOR_FILE = 'Soil_RusleFactor_C_EgSmallCatchment_7899.tif'
K_FACTOR_FILE = 'Soil_RusleFactor_K_EgSmallCatchment_7899.tif'

EVENT = '2019_fire'
ENSEMBLE = 'historical'
FIRE_START = '2019-01-15'
FIRE_END = '2019-03-07'
# Small recovery windows keep the simulated rainfall span short (fast) while
# still crossing a breakpoint, so the run is genuinely segmented into two
# recovery windows.
BREAKPOINTS = [0, 0.05, 0.1]


def _data(name):
    return str(Path(fire_impacts.__file__).parent.parent / 'test_data' / name)


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    """Run the full prep -> sim pipeline once and return everything the
    assertions need. Module-scoped because the topography/SDR/simulation
    work is the expensive part."""
    root = tmp_path_factory.mktemp('project')
    proj = FireImpactsProject(str(root), clear=True)
    proj.add_catchment(_data(CATCHMENT_FILE), subcatchment_id_cols=['Id'])
    catchment = proj.catchments[0]

    # Register the provided subcatchment coverage.
    import geopandas as gpd
    sc_dir = Path(proj.catchment_path(catchment, 'Subcatchments'))
    sc_dir.mkdir(parents=True, exist_ok=True)
    gpd.read_file(_data(SUBCATCHMENT_FILE)).to_file(
        str(sc_dir / 'Subcatchments.shp'))

    prep = RunContext.solo_catchment(proj)
    ev = RunContext.solo_event(proj, event=EVENT)
    run = RunContext.solo_run(proj, event=EVENT, ensemble=ENSEMBLE)
    # Rainfall prep would create the ensemble dir; we inject synthetic
    # rainfall instead, so create it so run.validate() passes.
    Path(run.ensemble_path()).mkdir(parents=True, exist_ok=True)

    # --- Preprocessing ---------------------------------------------------
    topography.extract_catchment_dems(prep, _data(DEM_FILE))
    topography.extract_headwaters(prep)
    soil.extract_aridity_data(prep, aridity_raster=_data(ARIDITY_FILE))

    Path(ev.event_path(c.FIRE_SEVERITY_FOLDER_NAME)).mkdir(
        parents=True, exist_ok=True)
    synthetic_fire.generate_synthetic_fire(ev, random_seed=0)
    # generate_synthetic_fire does not write FireMeta.csv (the remote
    # severity path does); write it as severity.py would, since the recovery
    # accessors read the dates back from it.
    fire_meta = pd.DataFrame(
        {'Value': [pd.to_datetime(FIRE_START), pd.to_datetime(FIRE_END)]},
        index=['start_date', 'end_date'])
    fire_meta.index.name = 'Key'
    fire_meta.to_csv(
        ev.event_path(c.FIRE_SEVERITY_FOLDER_NAME, 'FireMeta.csv'),
        date_format='%Y-%m-%d')

    rusle.compute_adjusted_k_c(
        ev, c_factor_fn=_data(C_FACTOR_FILE), k_factor_fn=_data(K_FACTOR_FILE),
        recovery_breakpoints=BREAKPOINTS)

    # --- Simulation ------------------------------------------------------
    sim_start, sim_end = ev.simulation_period()
    idx = pd.date_range(sim_start, sim_end, freq='30min')
    rain = pd.Series(0.0, index=idx)
    rng = np.random.default_rng(0)
    spikes = rng.choice(len(idx), size=min(120, len(idx)), replace=False)
    rain.iloc[spikes] = rng.uniform(2, 20, size=len(spikes))
    rain.attrs['units'] = 'mm'

    factory = simr.default_rusle_recorders(
        grid_variables=('RUSLE',), grid_fns=('sum',),
        grid_timesteps=('total',), include_timeseries=True)
    fire_res = simr.run_usle_simulation(
        run, rain, recorders=factory(run, sim_start, sim_end),
        save_rasters=True, save_timeseries=True, use_fire_adjusted=True)
    base_res = simr.run_usle_simulation(
        run, rain, recorders=factory(run, sim_start, sim_end),
        save_rasters=True, save_timeseries=True, use_fire_adjusted=False)

    return {
        'proj': proj, 'catchment': catchment,
        'prep': prep, 'ev': ev, 'run': run,
        'rain': rain, 'fire_res': fire_res, 'base_res': base_res,
    }


# --- Layer scoping -------------------------------------------------------

def test_catchment_scope_layers_are_fire_independent(pipeline):
    """Base C/K/LS factor and the baseline SDR live at catchment scope."""
    prep = pipeline['prep']
    for name in ('C_factor.tif', 'K_factor.tif', 'LS_factor.tif'):
        assert Path(prep.catchment_path('Erodibility', name)).exists(), name
    assert Path(
        prep.catchment_path('Delivery', 'SDR_baseline.tif')).exists()


def test_event_scope_has_per_recovery_layers(pipeline):
    """Each recovery window gets its own adjusted C/K and SDR at event scope,
    alongside the fire data and the event definition."""
    ev = pipeline['ev']
    for t in BREAKPOINTS[:-1]:  # window-start times
        suf = c.recovery_time_suffix(t)
        for name in (f'C_factor_adjusted_{suf}.tif',
                     f'K_factor_adjusted_{suf}.tif'):
            assert Path(ev.event_path('Erodibility', name)).exists(), name
        assert Path(ev.event_path('Delivery', f'SDR_{suf}.tif')).exists(), suf
    assert Path(ev.event_path('FireSeverity', 'masked_dNBR.tif')).exists()
    assert Path(ev.event_path('FireSeverity', 'FireMeta.csv')).exists()
    assert Path(ev.event_path(c.EVENT_DEFINITION_NAME)).exists()


def test_no_scope_leaks(pipeline):
    """Fire-adjusted layers never leak to catchment scope, and the baseline
    SDR is never written at event scope."""
    prep, ev = pipeline['prep'], pipeline['ev']
    assert not Path(
        prep.catchment_path('Erodibility', 'C_factor_adjusted_t0.tif')
    ).exists()
    assert not Path(ev.event_path('Delivery', 'SDR_baseline.tif')).exists()


def test_results_land_at_run_scope(pipeline):
    """Fire-adjusted and baseline results go under Runs/<event>/<ensemble>/,
    not at catchment scope."""
    run, prep = pipeline['run'], pipeline['prep']
    ts = c.RUSLE_OP_TIMESERIES_NAME + '.csv'
    assert Path(run.run_path(c.RESULTS_FOLDER_NAME, ts)).exists()
    assert Path(run.run_path(c.RESULTS_BASELINE_FOLDER_NAME, ts)).exists()
    assert not Path(prep.catchment_path(c.RESULTS_FOLDER_NAME, ts)).exists()


# --- Behaviour -----------------------------------------------------------

def test_run_is_segmented_by_recovery_window(pipeline):
    """A fire-adjusted run splits into one segment per recovery window, each
    reading its own recovery-time layers."""
    run, rain = pipeline['run'], pipeline['rain']
    segments = simr._recovery_run_segments(
        run, rain, use_fire_adjusted=True)
    times = [t for t, _ in segments]
    assert times == BREAKPOINTS[:-1]
    # every segment carries some rainfall (windows span the sim period)
    assert all(len(seg) > 0 for _, seg in segments)


def test_baseline_run_is_a_single_segment(pipeline):
    """The baseline run is one whole-period segment (no fire adjustment)."""
    run, rain = pipeline['run'], pipeline['rain']
    segments = simr._recovery_run_segments(
        run, rain, use_fire_adjusted=False)
    assert len(segments) == 1
    assert segments[0][0] is None


def test_fire_erosion_exceeds_baseline(pipeline):
    """The fire raises the C factor, so fire-adjusted erosion must exceed the
    unburnt baseline over the same rainfall."""
    col = c.RUSLE_OP_TIMESERIES_NAME
    fire = pipeline['fire_res']['erosion_daily_time_series']
    base = pipeline['base_res']['erosion_daily_time_series']
    fire_total = pd.DataFrame(fire).sum().sum()
    base_total = pd.DataFrame(base).sum().sum()
    assert base_total > 0
    assert fire_total > base_total


def test_manifest_records_recovery_breakpoints(pipeline):
    """save_ensemble_run resolves the event's breakpoints into the manifest,
    replacing the per-run recovery_time / interval fields."""
    import json
    from fire_impacts.sim.results import save_ensemble_run, MANIFEST_NAME
    root = save_ensemble_run(pipeline['run'])
    manifest = json.loads((Path(root) / MANIFEST_NAME).read_text())
    assert manifest['recovery_breakpoints'] == BREAKPOINTS
    assert manifest['event'] == EVENT
    assert 'recovery_time' not in manifest
    assert 'recovery_interval_years' not in manifest


# --- Regression ----------------------------------------------------------

def test_baseline_sdr_survives_a_coarse_c_factor(pipeline):
    """Regression: the baseline SDR is derived from the base C factor, which
    here is coarser than the DEM grid. compute_sediment_delivery_ratio must
    align it to the DEM before the downslope BFS indexes it — a raw read
    raised IndexError. Reaching this point (fixture built) already proves it
    no longer crashes; assert the output exists and is finite."""
    import rasterio as rio
    prep = pipeline['prep']
    sdr_path = prep.catchment_path('Delivery', 'SDR_baseline.tif')
    assert Path(sdr_path).exists()
    with rio.open(sdr_path) as src:
        data = src.read(1)
    assert np.isfinite(data).any()
