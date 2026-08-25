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
import rasterio

import fire_impacts
from fire_impacts import const as c
from fire_impacts.pre.project import FireImpactsProject
from fire_impacts.pre import topography, soil, rusle, synthetic_fire
from fire_impacts.sim import rusle as simr
from fire_impacts.context import RunContext
from fire_impacts.params import ModelParameters

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

# Content hashes of every raster the preprocessing pipeline writes at default
# parameters. Regenerate deliberately (see test_default_outputs_are_unchanged)
# only when a default or an equation is intentionally changed.
#
# Last regenerated when dNBR scaling was unified (const.DNBR_SCALE): the
# synthetic severity path had been writing the conventional 0-1000 scale
# while the real path wrote the stored band-ratio difference, so every
# fire-adjusted layer built from synthetic dNBR had saturated at c_peak.
GOLDEN_PREP_HASHES = {
    "Catchments/EgSmallCatchment_7899/Delivery/Cth_baseline.tif": "6fd676cdfe9ea85a",
    "Catchments/EgSmallCatchment_7899/Delivery/Ddn_baseline.tif": "649f4b38c5852f8b",
    "Catchments/EgSmallCatchment_7899/Delivery/Distance_to_stream.tif": "5b18c6f980bb8463",
    "Catchments/EgSmallCatchment_7899/Delivery/Dup_baseline.tif": "8eb4eb527f909267",
    "Catchments/EgSmallCatchment_7899/Delivery/IC_baseline.tif": "7629543efd170838",
    "Catchments/EgSmallCatchment_7899/Delivery/SDR_baseline.tif": "9dd5e31f594b8dd3",
    "Catchments/EgSmallCatchment_7899/Delivery/Sth.tif": "7aa6577d6d77a64b",
    "Catchments/EgSmallCatchment_7899/Delivery/Streams.tif": "241054773ccceb30",
    "Catchments/EgSmallCatchment_7899/Erodibility/C_factor.tif": "265597f6cbc0d122",
    "Catchments/EgSmallCatchment_7899/Erodibility/K_factor.tif": "0fa357811da0b7c1",
    "Catchments/EgSmallCatchment_7899/Erodibility/LS_factor.tif": "9b9c0b6651b8950e",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Cth_t0.tif": "f45629277d65c511",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Cth_t0_05.tif": "c2ca2f9ea56e98a3",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Ddn_t0.tif": "47c783fd4236dad5",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Ddn_t0_05.tif": "5c5e776ef2ace911",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Distance_to_stream.tif": "5b18c6f980bb8463",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Dup_t0.tif": "ca28b39f720f9912",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Dup_t0_05.tif": "73bb0d8b159c6f21",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/IC_t0.tif": "a385723fa64c982d",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/IC_t0_05.tif": "47ff7021080cb547",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/SDR_t0.tif": "08bb02bfe0649d9f",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/SDR_t0_05.tif": "559ec731246244a1",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Sth.tif": "7aa6577d6d77a64b",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Delivery/Streams.tif": "241054773ccceb30",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Erodibility/C_factor_adjusted_t0.tif": "9815aedfcf7bde1b",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Erodibility/C_factor_adjusted_t0_05.tif": "f0f5bd9c9b299ebe",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Erodibility/K_factor_adjusted_t0.tif": "01d46cbd2f4ab83c",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/Erodibility/K_factor_adjusted_t0_05.tif": "c8000d1b77e806f5",
    "Catchments/EgSmallCatchment_7899/Events/2019_fire/FireSeverity/masked_dNBR.tif": "febf6b8df6a5e38b",
    "Catchments/EgSmallCatchment_7899/Runs/2019_fire/historical/Results/RUSLE_sum_total.tif": "12a5a0fc30973c2a",
    "Catchments/EgSmallCatchment_7899/Runs/2019_fire/historical/Results_baseline/RUSLE_sum_total.tif": "50270151e3192181",
    "Catchments/EgSmallCatchment_7899/Soils/Aridity.tif": "d304bc9ba572d2a0",
    "Catchments/EgSmallCatchment_7899/Topography/DEM.tif": "357d6a5f877a8625",
    "Catchments/EgSmallCatchment_7899/Topography/Flow_accumulation.tif": "950d350d781f1175",
    "Catchments/EgSmallCatchment_7899/Topography/Flow_direction.tif": "9ef73c48072e9e7d",
    "Catchments/EgSmallCatchment_7899/Topography/Headwaters.tif": "4243d5597f140f66",
    "Catchments/EgSmallCatchment_7899/Topography/Slope.tif": "971539b361b302d2",
    "Catchments/EgSmallCatchment_7899/Topography/Stream_Network.tif": "a1d6216a9cb001e0"
}


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

    # Register the provided subcatchment coverage through the library so it
    # lands where get_subcatchments looks (<catchment>_subcatchments.shp) and
    # the RUSLE aggregation writes its subcatchment summary.
    proj.add_subcatchments(catchment, _data(SUBCATCHMENT_FILE), id_cols=['Id'])

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


# --- Calibration parameters ----------------------------------------------

def test_provenance_is_written_at_both_scopes(pipeline):
    """compute_adjusted_k_c writes catchment-scope layers (base C/K, LS,
    baseline SDR) and event-scope ones, so both trees get a record."""
    for ctx, scope in ((pipeline['prep'], 'catchment'),
                       (pipeline['ev'], 'event')):
        record = ctx.read_provenance(scope=scope)
        assert record is not None, scope
        assert record.digest().startswith('sha256:')
        # Nothing was overridden in the fixture, so everything is default.
        assert set(record.sources.values()) == {'default'}


def test_provenance_records_a_non_default_value(pipeline):
    """With all defaults this would pass even if write_provenance ignored
    its argument entirely, so drive it with something overridden."""
    ev = pipeline['ev']
    record = ev.parameters(fire_adjustment__c_peak=0.42)
    ev.write_provenance(record, scope='event')
    try:
        read_back = ev.read_provenance(scope='event')
        assert read_back.parameters.fire_adjustment.c_peak == 0.42
        assert read_back.sources['fire_adjustment.c_peak'] == 'call'
        assert read_back.digest() == record.digest()
    finally:
        ev.write_provenance(ev.parameters(), scope='event')
    assert ev.read_provenance(
        scope='event').parameters == ModelParameters()


def _prep_raster_hashes(pipeline):
    """Hash every raster the preprocessing pipeline wrote, NaN-normalised."""
    import hashlib
    root = Path(pipeline['proj'].project_path)
    out = {}
    for tif in sorted(root.rglob('*.tif')):
        # Probe outputs written by other tests must not perturb the set.
        if 'probe' in tif.name:
            continue
        with rasterio.open(tif) as src:
            arr = src.read(1)
        out[str(tif.relative_to(root))] = hashlib.sha256(
            np.nan_to_num(arr, nan=-9e9).astype('float64').tobytes()
        ).hexdigest()[:16]
    return out


def test_default_outputs_are_unchanged(pipeline):
    """Phase 2 replaced twelve hard-coded literals with resolved parameters.
    At default values the outputs must be identical, and must stay that way:
    without this, any of those defaults can drift and the suite stays green.

    A mismatch here means either a default moved (update GOLDEN_PREP_HASHES
    deliberately) or an equation changed (that is the bug this catches).
    """
    got = _prep_raster_hashes(pipeline)
    assert got == GOLDEN_PREP_HASHES


@pytest.mark.parametrize('field,value,layer,check', [
    ('delivery__max_sdr', 0.5, 'Delivery/SDR_{s}.tif',
     lambda a: np.nanmax(a) <= 0.5),
    ('delivery__min_c_factor', 0.5, 'Delivery/Cth_{s}.tif',
     lambda a: np.nanmin(a) >= 0.5),
    ('delivery__stream_area_threshold_m2', 1e12, 'Delivery/Streams.tif',
     lambda a: not np.any(a > 0)),
])
def test_each_delivery_field_moves_its_own_output(
        pipeline, field, value, layer, check):
    """One field at a time, so a swapped pair (min_slope/max_slope,
    ic0/k) cannot pass. Writes to a probe suffix, so the fixture's
    rasters are untouched and no restore is needed."""
    ev = pipeline['ev']
    suffix = f'probe_{field}'
    rusle.compute_sediment_delivery_ratio(
        ev, output_suffix=suffix, params=ev.parameters(**{field: value}),
    )
    path = ev.catchment_path(layer.format(s=suffix))
    with rasterio.open(path) as src:
        assert check(src.read(1)), f'{field} did not reach {layer}'


def test_the_slope_clamp_is_not_inverted(pipeline):
    """min_slope and max_slope are both used in one nested np.where; a
    swap would still produce a plausible raster."""
    ev = pipeline['ev']
    rusle.compute_sediment_delivery_ratio(
        ev, output_suffix='probe_slope',
        params=ev.parameters(delivery__min_slope=0.4,
                             delivery__max_slope=0.6),
    )
    with rasterio.open(ev.catchment_path('Delivery', 'Sth.tif')) as src:
        sth = src.read(1)
    finite = sth[np.isfinite(sth)]
    assert finite.min() >= 0.4 and finite.max() <= 0.6


def test_the_ls_slope_length_cap_reaches_the_ls_factor(pipeline):
    """compute_lsi writes LS_factor.tif with no suffix option, so this
    overrides, captures the returned array, then recomputes at defaults to
    restore — and verifies the restore by hash.

    Catches the mutation a source-inspection test cannot: reading the
    parameter into a local and then overwriting that local with the old
    literal, which is invisible while the literal equals the default.
    """
    import hashlib
    prep, ev = pipeline['prep'], pipeline['ev']
    ls_path = prep.catchment_path('Erodibility', 'LS_factor.tif')

    def _ls_hash():
        with rasterio.open(ls_path) as src:
            return hashlib.sha256(
                np.nan_to_num(src.read(1), nan=-9e9).astype('float64').tobytes()
            ).hexdigest()

    before = _ls_hash()
    *_, overridden = rusle.compute_lsi(
        prep, params=ev.parameters(topography__max_slope_length_m=20.0))
    overridden = np.array(overridden, copy=True)
    *_, restored = rusle.compute_lsi(prep)

    assert not np.allclose(
        np.nan_to_num(overridden), np.nan_to_num(restored)), \
        'max_slope_length_m did not reach the LS factor'
    assert _ls_hash() == before, 'fixture LS_factor was not restored'


def test_a_catchment_layer_override_reaches_the_raster(pipeline):
    """The call layer is covered above; this proves the *persisted*
    catchment layer reaches the producer too. Writes to a probe suffix and
    restores the override, verifying the restore rather than assuming it."""
    proj, catchment = pipeline['proj'], pipeline['catchment']
    ev = pipeline['ev']
    proj.set_catchment_parameter_overrides(
        catchment, {'delivery': {'max_sdr': 0.5}})
    try:
        rusle.compute_sediment_delivery_ratio(
            ev, output_suffix='probe_catchment_layer')
        with rasterio.open(ev.catchment_path(
                'Delivery', 'SDR_probe_catchment_layer.tif')) as src:
            assert float(np.nanmax(src.read(1))) <= 0.5
    finally:
        proj.set_catchment_parameter_overrides(catchment, {})
    # The restore is verified, not assumed.
    assert proj.catchment_parameter_overrides(catchment) == {}
    assert ev.parameters().parameters.delivery.max_sdr == 0.8


def test_a_deprecated_kwarg_still_works_and_warns(pipeline):
    ev = pipeline['ev']
    with pytest.warns(DeprecationWarning, match='delivery__max_sdr'):
        rusle.compute_sediment_delivery_ratio(
            ev, max_sdr=0.5, output_suffix='deprecated_kwarg_check',
        )
    with rasterio.open(
            ev.catchment_path('Delivery',
                              'SDR_deprecated_kwarg_check.tif')) as src:
        assert float(np.nanmax(src.read(1))) <= 0.5


# --- dNBR scale ----------------------------------------------------------

def test_stored_dnbr_is_on_the_stored_scale(pipeline):
    """generate_synthetic_fire samples reference rasters published on the
    conventional 0-1000 scale; it must convert before writing, or the
    synthetic and real severity paths disagree by 1000x."""
    ev = pipeline['ev']
    with rasterio.open(
            ev.event_path(c.FIRE_SEVERITY_FOLDER_NAME, 'masked_dNBR.tif')) as src:
        dnbr = src.read(1)
    finite = dnbr[np.isfinite(dnbr)]
    assert finite.size
    assert finite.max() < 10, (
        'masked_dNBR.tif holds conventional-scale values; it should store '
        'the raw band-ratio difference (see const.DNBR_SCALE)'
    )


def test_the_severity_split_actually_fires(pipeline):
    """The regression this whole change exists for: dNBR was compared on
    the stored scale against a 400 threshold, so `dnbr >= 400` was never
    true and every high-severity output was identically zero while the
    low-severity ones carried the entire total."""
    from fire_impacts.sim.rusle import _rusle_parameter_grids
    ev = pipeline['ev']
    _, _, dnbr, _, _ = _rusle_parameter_grids(
        ev, recovery_time=BREAKPOINTS[0])
    finite = dnbr[np.isfinite(dnbr)]
    assert finite.max() > c.DEFAULT_DNBR_SEVERITY_THRESHOLD, (
        'no cell reaches the severity threshold — the grids are being read '
        'on the wrong scale'
    )
    high = (finite >= c.DEFAULT_DNBR_SEVERITY_THRESHOLD).sum()
    assert 0 < high < finite.size, (
        'the severity split is degenerate: every cell fell on one side'
    )


def test_the_adjusted_c_factor_is_not_saturated_everywhere(pipeline):
    """With dNBR 1000x too large, CdNBR exceeded the saturation threshold
    in every cell and the adjusted C factor collapsed to c_peak, losing
    the whole severity gradient."""
    ev = pipeline['ev']
    with rasterio.open(ev.event_path(
            'Erodibility', 'C_factor_adjusted_t0.tif')) as src:
        c_adj = src.read(1)
    finite = c_adj[np.isfinite(c_adj)]
    assert len(np.unique(np.round(finite, 6))) > 100, (
        'adjusted C factor has collapsed to a handful of values — dNBR is '
        'saturating the interpolation'
    )


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


def test_rusle_subcatchment_plot_finds_run_scoped_summary(pipeline):
    """Regression: plotting a 'RUSLE_*' subcatchment column must route to the
    run-scoped rusle_subcatchment_summary. The recorder keys are 'RUSLE_*'
    (not the legacy 'erosion_*'), which previously missed plot_subcatchments'
    auto-detection and fell through to the soil/slope summary — a file that
    doesn't exist, so the plot silently rendered blank. With allow_basic=False
    this raises unless the column is routed correctly."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    proj, run, catchment = (
        pipeline['proj'], pipeline['run'], pipeline['catchment'])

    summary = run.run_path(
        c.RESULTS_FOLDER_NAME, c.RUSLE_SC_SUMMARY_NAME + '.csv')
    assert Path(summary).exists(), 'the run did not write the RUSLE summary'

    # Colouring by a RUSLE column must find the run-scoped summary and not
    # raise (allow_basic=False turns a missing table into an error).
    proj.plot_subcatchments(
        catchment=catchment, data_type=c.RESULTS_FOLDER_NAME,
        colour_col='RUSLE_sum_total', ctx=run)
    plt.close('all')


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
