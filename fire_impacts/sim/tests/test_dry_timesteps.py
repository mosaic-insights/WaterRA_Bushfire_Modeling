"""
What generate_rusle yields per timestep.

Erosion in a cell is the timestep's erosivity times a static per-cell
layer, so the generator yields that pair rather than the product: one set
of layers per recovery segment, and a scale per timestep. Around three
quarters of timesteps are dry, and for those the scale is simply zero.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts.sim.rusle import (
    generate_rusle, rainfall_erosivity, record_multi_period_grid,
)
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid

SHAPE = (2, 3)
CELL_AREA_HA = 0.09

GRID_KEYS = (
    'RUSLE', 'delivered',
    'RUSLE_below_threshold', 'RUSLE_above_threshold',
    'delivered_below_threshold', 'delivered_above_threshold',
)


@pytest.fixture()
def grids():
    klscp = np.full(SHAPE, 2.0, dtype=np.float32)
    # A cell outside the catchment mask, as klscp_masked would carry in
    # production (klscp * mask, mask is NaN outside the boundary). This
    # is the only NaN cell in the fixture, so tests below can key off it.
    klscp[1, 2] = np.nan
    sdr = np.full(SHAPE, 0.5, dtype=np.float32)
    dnbr = np.array([[0.0, 0.0, 0.0], [900.0, 900.0, 900.0]],
                    dtype=np.float32)
    return klscp, sdr, dnbr


def run(rain_mm, grids, **kwargs):
    rain = pd.Series(
        rain_mm,
        index=pd.date_range('2020-01-01', periods=len(rain_mm), freq='30min'),
    )
    return list(generate_rusle(rain, *grids, CELL_AREA_HA, **kwargs))


class TestWhatIsYielded:

    def test_grids_are_deferred(self, grids):
        _, data = run([5.0], grids)[0]
        assert all(isinstance(data[key], ScaledGrid) for key in GRID_KEYS)

    def test_the_layers_are_shared_across_timesteps(self, grids):
        steps = run([5.0, 0.0, 7.0], grids)
        units = [data['RUSLE'].unit for _, data in steps]
        assert units[0] is units[1] is units[2]

    def test_each_timestep_has_its_own_scale(self, grids):
        steps = run([5.0, 7.0], grids)
        assert steps[0][1]['RUSLE'].scale != steps[1][1]['RUSLE'].scale

    def test_rainfall_is_uniform_so_there_is_one_rain_cell(self, grids):
        _, data = run([5.0], grids)[0]
        assert data['RUSLE'].rain_index is None
        assert data['RUSLE'].scale.shape == (1,)

    def test_all_six_grids_share_one_scale_array(self, grids):
        _, data = run([5.0], grids)[0]
        assert all(
            data[key].scale is data['RUSLE'].scale for key in GRID_KEYS
        )

    def test_the_scale_is_float64(self, grids):
        _, data = run([5.0], grids)[0]
        assert data['RUSLE'].scale.dtype == np.float64


class TestDryTimesteps:

    def test_timesteps_are_flagged_dry_when_there_is_no_rain(self, grids):
        steps = run([0.0, 5.0, 0.0], grids)
        assert [d['dry'] for _, d in steps] == [True, False, True]

    def test_a_dry_timestep_scales_the_layers_by_zero(self, grids):
        _, data = run([0.0], grids)[0]
        assert data['RUSLE'].scale[0] == 0.0

    def test_a_dry_timestep_erodes_nothing(self, grids):
        # Every finite cell is exactly zero. A dry timestep's grid is
        # 0.0 * unit, so a cell where unit is NaN stays NaN rather than
        # becoming zero (see test_a_dry_timestep_is_nan_where_the_layer_
        # is_nan below); nan_to_num folds that into 0 here so this test
        # only asserts what it says: nothing erodes anywhere finite.
        _, data = run([0.0], grids)[0]
        for key in GRID_KEYS:
            arr = np.nan_to_num(np.asarray(data[key]))
            assert np.array_equal(arr, np.zeros(SHAPE))

    def test_a_dry_timestep_is_nan_where_the_layer_is_nan(self, grids):
        klscp, sdr, _ = grids
        nan_cell = np.isnan(klscp)
        _, data = run([0.0], grids)[0]

        for key in ('RUSLE', 'delivered'):
            arr = np.asarray(data[key])
            assert np.all(np.isnan(arr[nan_cell]))
            assert np.array_equal(
                arr[~nan_cell], np.zeros((~nan_cell).sum()))

    def test_a_dry_timestep_reports_no_intensity_or_erosivity(self, grids):
        _, data = run([0.0], grids)[0]
        assert (data['total_rain'], data['intensity'],
                data['erosivity']) == (0.0, 0.0, 0.0)


class TestValuesMatchTheModel:

    def test_erosion_is_erosivity_by_klscp_by_cell_area(self, grids):
        klscp, sdr, _ = grids
        _, data = run([5.0], grids)[0]
        intensity, R = rainfall_erosivity(5.0)

        assert data['intensity'] == intensity
        assert data['erosivity'] == R
        # Exact equality, not allclose: this pins the arithmetic (dtype
        # included) rather than merely checking it is close. equal_nan
        # is needed only because the fixture carries one NaN cell
        # (outside the catchment mask); every other cell must match bit
        # for bit.
        assert np.array_equal(
            np.asarray(data['RUSLE']), R * klscp * CELL_AREA_HA,
            equal_nan=True)
        assert np.array_equal(
            np.asarray(data['delivered']),
            np.asarray(data['RUSLE']) * sdr, equal_nan=True)

    def test_severity_thresholds_split_the_grid(self, grids):
        _, data = run([5.0], grids)[0]
        below = np.asarray(data['RUSLE_below_threshold'])
        above = np.asarray(data['RUSLE_above_threshold'])

        assert np.allclose(above[0], 0.0)
        assert np.allclose(below[1], 0.0)
        assert np.allclose(below + above, np.asarray(data['RUSLE']),
                           equal_nan=True)


class TestEagerMode:

    def test_materialise_grids_yields_plain_arrays(self, grids):
        _, data = run([5.0], grids, materialise_grids=True)[0]
        assert all(isinstance(data[key], np.ndarray) for key in GRID_KEYS)

    def test_eager_and_deferred_agree(self, grids):
        _, lazy = run([5.0], grids)[0]
        _, eager = run([5.0], grids, materialise_grids=True)[0]
        for key in GRID_KEYS:
            assert np.allclose(np.asarray(lazy[key]), eager[key],
                               equal_nan=True)


class TestCounter:

    def test_deferred_grids_carry_the_run_counter(self, grids):
        counter = MaterialisationCounter()
        _, data = run([5.0], grids, counter=counter)[0]
        assert counter.count == 0

        np.asarray(data['RUSLE'])
        assert counter.count == 1


class TestNonNegativityGuards:
    """
    Deferring a maximum is only valid because a layer times a scale is
    monotone in the scale, which needs both to be non-negative. Rather
    than trust that, check it and fall back.
    """

    def test_a_negative_layer_falls_back_to_eager_grids(self, grids,
                                                        caplog):
        klscp, sdr, dnbr = grids
        klscp = klscp.copy()
        klscp[0, 0] = -1.0
        with caplog.at_level('WARNING'):
            steps = run([5.0], (klscp, sdr, dnbr))

        _, data = steps[0]
        assert isinstance(data['RUSLE'], np.ndarray)
        # Not just "negative" - the message must name the layer, not
        # the rainfall, or a later edit could swap the two conditions
        # without any test noticing.
        assert 'negative erosion layer' in caplog.text.lower()

    def test_negative_rainfall_falls_back_to_eager_grids(self, grids,
                                                         caplog):
        with caplog.at_level('WARNING'):
            steps = run([-1.0, 5.0], grids)

        assert isinstance(steps[1][1]['RUSLE'], np.ndarray)
        assert 'negative rainfall' in caplog.text.lower()

    def test_the_fallback_still_produces_the_right_numbers(self, grids):
        klscp, sdr, dnbr = grids
        klscp = klscp.copy()
        klscp[0, 0] = -1.0
        _, data = run([5.0], (klscp, sdr, dnbr))[0]
        _, R = rainfall_erosivity(5.0)

        # equal_nan via allclose's `equal_nan` kwarg: the fixture's
        # klscp carries one NaN cell (outside the catchment mask,
        # unrelated to the negative cell under test here), so a plain
        # allclose would report a mismatch where both sides are NaN.
        assert np.allclose(data['RUSLE'], R * klscp * CELL_AREA_HA,
                           equal_nan=True)

    def test_ordinary_layers_stay_deferred(self, grids):
        _, data = run([5.0], grids)[0]
        assert isinstance(data['RUSLE'], ScaledGrid)

    def test_an_empty_rainfall_series_does_not_raise(self, grids):
        # The guard's negative_rain check short-circuits on
        # len(rainfall.values) before calling np.nanmin, specifically
        # so an empty series (a segment with no timesteps at all)
        # cannot raise. That branch is otherwise unexercised.
        assert run([], grids) == []

    def test_the_true_multi_timestep_max_survives_a_deferred_recorder(
            self, grids):
        """
        A single timestep can't tell a guarded run apart from an
        unguarded one: ScaledGrid materialises scale * unit correctly
        regardless of sign when there is only one scale value (see
        test_the_fallback_still_produces_the_right_numbers above,
        which passes with or without the guard). The wrong number this
        guard actually exists to prevent only appears once a recorder
        defers a maximum ACROSS two or more timesteps that share one
        unit layer - so this test drives generate_rusle's output
        through record_multi_period_grid(fn='max') and checks the
        finalised grid against a maximum computed independently, by
        hand, timestep by timestep - not by restating the guard or the
        recorder's own arithmetic.
        """
        klscp = np.array([[2.0, -2.0]], dtype=np.float32)
        sdr = np.ones_like(klscp)
        dnbr = np.zeros_like(klscp)
        rain_mm = [5.0, 20.0]

        steps = run(rain_mm, (klscp, sdr, dnbr))
        period = [(steps[0][0], steps[-1][0])]
        recorder = record_multi_period_grid('RUSLE', 'max', period)
        for timestep, data in steps:
            recorder(timestep, **data)
        result = np.asarray(recorder.finalize())

        expected = None
        for rain in rain_mm:
            _, R = rainfall_erosivity(rain)
            step_grid = R * klscp * CELL_AREA_HA
            expected = step_grid if expected is None \
                else np.maximum(expected, step_grid)

        assert np.allclose(result, expected)
