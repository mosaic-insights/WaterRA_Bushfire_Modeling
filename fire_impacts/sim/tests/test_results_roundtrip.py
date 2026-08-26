"""
Persisting an ensemble run and reading it back.

Everything here runs against tmp_path with a stub project - real parquet
and JSON, no network. The round trip is the contract a downstream
catchment model depends on. A run is a (event, ensemble) pair, so its
outputs land under Runs/<event>/<ensemble>/ and the context routes both
the save and the reload.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fire_impacts.context import RunContext
from fire_impacts.sim.results import (
    MANIFEST_NAME,
    _safe_key,
    list_ensembles,
    load_ensemble_combined,
    load_ensemble_manifest,
    save_ensemble_run,
)


CATCHMENT = 'Eg'


class StubProject:
    """Minimal FireImpactsProject stand-in rooted at a tmp directory."""

    catchments = [CATCHMENT]

    def __init__(self, root):
        self.root = root

    def catchment_path(self, catchment, *args):
        return str(self.root.joinpath('Catchments', catchment, *args))

    def event_path(self, catchment, *args, event):
        return str(self.root.joinpath(
            'Catchments', catchment, 'Events', event, *args))

    def ensemble_path(self, catchment, *args, ensemble):
        return str(self.root.joinpath(
            'Catchments', catchment, 'Ensembles', ensemble, *args))

    def run_path(self, catchment, *args, event, ensemble, label=None):
        return str(self.root.joinpath(
            'Catchments', catchment, 'Runs', event,
            label or ensemble, *args))

    def subcatchment_label_field(self, catchment):
        return 'SiteID'

    def events(self, catchment=None):
        base = self.root / 'Catchments' / (catchment or 'c') / 'Events'
        return sorted(p.name for p in base.iterdir()) if base.exists() else []

    def ensembles(self, catchment=None):
        base = self.root / 'Catchments' / (catchment or 'c') / 'Ensembles'
        return sorted(p.name for p in base.iterdir()) if base.exists() else []


@pytest.fixture()
def proj(tmp_path):
    return StubProject(tmp_path)


def run_ctx(proj, *, event='default', ensemble='default'):
    """A run-level RunContext bound to the stub project. Constructed
    directly (not via solo_run) so no on-disk catchment registration is
    needed."""
    return RunContext(
        project=proj, catchment=CATCHMENT, event=event, ensemble=ensemble)


def subcatchment_frame(seed=0, periods=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.uniform(0, 10, (periods, 3)),
        index=pd.date_range('2019-01-01', periods=periods, freq='D'),
        columns=['SC_1', 'SC_2', 'SC_3'],
    )


@pytest.fixture()
def combined():
    """Two replicates of daily subcatchment loads."""
    return {'D': {0: subcatchment_frame(0), 1: subcatchment_frame(1)}}


class TestRoundTrip:

    def test_saves_and_reloads_combined_frames(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        assert sorted(loaded) == [0, 1]
        for rep, original in combined['D'].items():
            # check_freq=False: parquet stores the timestamps but not the
            # DatetimeIndex freq attribute. See
            # test_index_freq_is_not_preserved.
            pd.testing.assert_frame_equal(
                loaded[rep], original, check_freq=False)

    def test_index_freq_is_not_preserved(self, proj, combined):
        # Harmless in itself - the timestamps are intact and resampling
        # re-infers - but anything downstream reading index.freq gets
        # None after a save/load rather than the original offset.
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)
        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        assert combined['D'][0].index.freq is not None
        assert loaded[0].index.freq is None
        assert loaded[0].index.equals(combined['D'][0].index)

    def test_column_labels_survive(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)
        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        assert list(loaded[0].columns) == ['SC_1', 'SC_2', 'SC_3']

    def test_numeric_columns_become_strings(self, proj):
        # Parquet needs string column names; the saver coerces so that a
        # caller who skipped SiteID labelling still round-trips.
        frame = subcatchment_frame()
        frame.columns = [1, 2, 3]
        save_ensemble_run(run_ctx(proj), combined_by_freq={'D': {0: frame}})

        loaded = load_ensemble_combined(run_ctx(proj), freq='D')
        assert list(loaded[0].columns) == ['1', '2', '3']

    def test_datetime_index_survives(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)
        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        assert isinstance(loaded[0].index, pd.DatetimeIndex)
        assert loaded[0].index[0] == pd.Timestamp('2019-01-01')

    def test_multiple_frequencies_are_kept_apart(self, proj):
        by_freq = {
            'D': {0: subcatchment_frame(0)},
            'YS': {0: subcatchment_frame(1, periods=1)},
        }
        save_ensemble_run(run_ctx(proj), combined_by_freq=by_freq)

        daily = load_ensemble_combined(run_ctx(proj), freq='D')
        yearly = load_ensemble_combined(run_ctx(proj), freq='YS')

        assert len(daily[0]) == 3
        assert len(yearly[0]) == 1

    def test_replicate_indices_are_integers(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)
        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        assert all(isinstance(k, int) for k in loaded)

    def test_double_digit_replicates_reload_in_order(self, proj):
        by_freq = {'D': {i: subcatchment_frame(i) for i in range(12)}}
        save_ensemble_run(run_ctx(proj), combined_by_freq=by_freq)

        loaded = load_ensemble_combined(run_ctx(proj), freq='D')

        # Directories are zero-padded ('00'..'11') so sorted() order
        # matches numeric order rather than putting '10' after '1'.
        assert list(loaded) == list(range(12))


class TestManifest:

    def test_records_the_run_shape(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        manifest = load_ensemble_manifest(run_ctx(proj))

        assert manifest['catchment'] == CATCHMENT
        assert manifest['event'] == 'default'
        assert manifest['ensemble'] == 'default'
        assert manifest['replicates'] == [0, 1]
        assert manifest['n_replicates'] == 2
        assert manifest['combined_frequencies'] == ['D']

    def test_records_which_artefacts_exist(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        artefacts = load_ensemble_manifest(run_ctx(proj))['artefacts']

        assert artefacts['combined'] is True
        assert artefacts['rainfall'] is False
        assert artefacts['rusle_timeseries'] is False
        assert artefacts['debris_flow_raw'] is False

    def test_picks_up_the_projects_label_field(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        manifest = load_ensemble_manifest(run_ctx(proj))
        assert manifest['subcatchment_label_field'] == 'SiteID'

    def test_extra_manifest_entries_are_merged(self, proj, combined):
        save_ensemble_run(
            run_ctx(proj), combined_by_freq=combined,
            extra_manifest={'climate_scenario': 'RCP8.5', 'seed': 42},
        )

        manifest = load_ensemble_manifest(run_ctx(proj))
        assert manifest['climate_scenario'] == 'RCP8.5'
        assert manifest['seed'] == 42

    def test_recovery_breakpoints_default_to_none_without_an_event(
            self, proj, combined):
        # The breakpoints come from the event's event.json/FireMeta; with
        # no event prepared, the manifest records None rather than failing.
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        manifest = load_ensemble_manifest(run_ctx(proj))
        assert manifest['recovery_breakpoints'] is None

    def test_manifest_is_valid_json_on_disk(self, proj, combined):
        root = save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        with open(Path(root) / MANIFEST_NAME) as f:
            assert json.load(f)['catchment'] == CATCHMENT

    def test_missing_manifest_raises(self, proj):
        with pytest.raises(FileNotFoundError, match='No run manifest'):
            load_ensemble_manifest(run_ctx(proj))


class TestRunRouting:
    """A run is a (event, ensemble) pair; each lands in its own directory
    and reloads independently."""

    def test_ensembles_are_kept_apart(self, proj):
        for name, seed in (('current', 0), ('future', 1)):
            save_ensemble_run(
                run_ctx(proj, ensemble=name),
                combined_by_freq={'D': {0: subcatchment_frame(seed)}})

        cur = load_ensemble_combined(run_ctx(proj, ensemble='current'), freq='D')
        fut = load_ensemble_combined(run_ctx(proj, ensemble='future'), freq='D')

        pd.testing.assert_frame_equal(
            cur[0], subcatchment_frame(0), check_freq=False)
        pd.testing.assert_frame_equal(
            fut[0], subcatchment_frame(1), check_freq=False)

    def test_events_are_kept_apart(self, proj):
        save_ensemble_run(
            run_ctx(proj, event='fire2019'),
            combined_by_freq={'D': {0: subcatchment_frame(2)}})

        # The default event has no run, so its reload fails.
        with pytest.raises(FileNotFoundError):
            load_ensemble_combined(run_ctx(proj, event='default'), freq='D')

        loaded = load_ensemble_combined(
            run_ctx(proj, event='fire2019'), freq='D')
        assert list(loaded) == [0]


class TestListEnsembles:

    def test_lists_prepared_ensemble_directories(self, proj):
        for name in ('current', 'future'):
            Path(proj.ensemble_path(CATCHMENT, ensemble=name)).mkdir(
                parents=True)

        assert list_ensembles(proj, CATCHMENT) == ['current', 'future']

    def test_listing_an_unknown_catchment_is_empty(self, proj):
        assert list_ensembles(proj, 'NoSuchCatchment') == []


class TestErrors:

    def test_unknown_frequency_raises(self, proj, combined):
        save_ensemble_run(run_ctx(proj), combined_by_freq=combined)

        with pytest.raises(FileNotFoundError, match='No combined parquet'):
            load_ensemble_combined(run_ctx(proj), freq='h')

    def test_missing_replicates_folder_raises(self, proj):
        save_ensemble_run(run_ctx(proj))

        with pytest.raises(FileNotFoundError, match='No replicates folder'):
            load_ensemble_combined(run_ctx(proj), freq='D')

    def test_an_empty_run_still_writes_a_manifest(self, proj):
        save_ensemble_run(run_ctx(proj))

        manifest = load_ensemble_manifest(run_ctx(proj))
        assert manifest['replicates'] == []
        assert manifest['n_replicates'] == 0


class TestSafeKey:

    @pytest.mark.parametrize('key,expected', [
        ('D', 'D'),
        ('YS', 'YS'),
        ('total', 'total'),
        ('30min', '30min'),
        ('a/b', 'a_b'),
        ('a b', 'a_b'),
    ])
    def test_sanitises_filename_keys(self, key, expected):
        assert _safe_key(key) == expected

    def test_frequency_keys_survive_the_round_trip(self, proj):
        # '30min' contains no unsafe characters, but the load path
        # re-derives the name through _safe_key - so it must agree.
        save_ensemble_run(
            run_ctx(proj),
            combined_by_freq={'30min': {0: subcatchment_frame()}},
        )

        loaded = load_ensemble_combined(run_ctx(proj), freq='30min')
        assert list(loaded) == [0]
