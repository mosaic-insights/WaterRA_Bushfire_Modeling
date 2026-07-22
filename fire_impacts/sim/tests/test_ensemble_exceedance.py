"""
Per-pixel exceedance probability across an ensemble.

The subtle part is NaN handling: catchment grids are NaN outside the
boundary, and those pixels must be excluded from the denominator rather
than counted as non-exceedances - otherwise probability is diluted by
pixels that were never in the catchment.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fire_impacts.sim.ensemble import exceedance_probability


CATCHMENT = 'Eg'
KEY = 'RUSLE_sum_yearly'


def replicate(grid, catchment=CATCHMENT, key=KEY):
    return {catchment: {key: np.asarray(grid, dtype=float)}}


def ensemble(*grids):
    """Build the {replicate_index: {catchment: {key: grid}}} shape."""
    return {i: replicate(g) for i, g in enumerate(grids)}


class TestBasicCounting:

    def test_all_replicates_exceed(self):
        result = exceedance_probability(
            ensemble([[10.0]], [[20.0]], [[30.0]]), KEY, threshold=5.0)

        assert np.allclose(result.values, 1.0)

    def test_no_replicate_exceeds(self):
        result = exceedance_probability(
            ensemble([[1.0]], [[2.0]], [[3.0]]), KEY, threshold=5.0)

        assert np.allclose(result.values, 0.0)

    def test_fraction_of_replicates_exceeding(self):
        result = exceedance_probability(
            ensemble([[10.0]], [[10.0]], [[1.0]], [[1.0]]),
            KEY, threshold=5.0)

        assert result.values[0, 0] == pytest.approx(0.5)

    def test_threshold_is_strict(self):
        # P(X > threshold), so a value exactly on the threshold does not
        # count as an exceedance.
        result = exceedance_probability(
            ensemble([[5.0]], [[5.0]]), KEY, threshold=5.0)

        assert np.allclose(result.values, 0.0)

    def test_computed_per_pixel(self):
        result = exceedance_probability(
            ensemble(
                [[10.0, 1.0], [10.0, 10.0]],
                [[10.0, 1.0], [1.0, 10.0]],
            ),
            KEY, threshold=5.0)

        assert np.allclose(result.values, [[1.0, 0.0], [0.5, 1.0]])

    def test_accepts_a_plain_list_of_replicates(self):
        as_list = [replicate([[10.0]]), replicate([[1.0]])]
        result = exceedance_probability(as_list, KEY, threshold=5.0)

        assert result.values[0, 0] == pytest.approx(0.5)


class TestNaNHandling:

    def test_nan_pixels_leave_the_denominator(self):
        # Two replicates have data at this pixel, one is NaN. One of the
        # two exceeds, so the probability is 1/2 - not 1/3.
        result = exceedance_probability(
            ensemble([[10.0]], [[1.0]], [[np.nan]]), KEY, threshold=5.0)

        assert result.values[0, 0] == pytest.approx(0.5)

    def test_a_pixel_nan_everywhere_is_nan(self):
        # Outside the catchment boundary: no information, not zero.
        result = exceedance_probability(
            ensemble([[np.nan]], [[np.nan]]), KEY, threshold=5.0)

        assert np.isnan(result.values[0, 0])

    def test_nan_does_not_count_as_an_exceedance(self):
        result = exceedance_probability(
            ensemble([[np.nan]], [[1.0]]), KEY, threshold=5.0)

        assert result.values[0, 0] == pytest.approx(0.0)

    def test_nan_masks_differ_between_replicates(self):
        result = exceedance_probability(
            ensemble(
                [[10.0, np.nan]],
                [[np.nan, 10.0]],
                [[1.0, 1.0]],
            ),
            KEY, threshold=5.0)

        # Left pixel: 1 of 2 valid exceed. Right pixel: 1 of 2 valid.
        assert np.allclose(result.values, [[0.5, 0.5]])

    def test_probabilities_stay_within_zero_and_one(self):
        rng = np.random.default_rng(0)
        grids = []
        for _ in range(5):
            g = rng.uniform(0, 10, (4, 4))
            g[rng.random((4, 4)) < 0.3] = np.nan
            grids.append(g)

        result = exceedance_probability(ensemble(*grids), KEY, threshold=5.0)
        valid = result.values[~np.isnan(result.values)]

        assert np.all((valid >= 0.0) & (valid <= 1.0))


class TestGridTypes:

    def test_two_dimensional_dataarrays(self):
        grids = [
            xr.DataArray(np.array([[10.0]]), dims=['northing', 'easting']),
            xr.DataArray(np.array([[1.0]]), dims=['northing', 'easting']),
        ]
        results = [{CATCHMENT: {KEY: g}} for g in grids]

        result = exceedance_probability(results, KEY, threshold=5.0)
        assert result.values[0, 0] == pytest.approx(0.5)

    def test_three_dimensional_dataarray_needs_a_time_when_ambiguous(self):
        times = [pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01')]
        da = xr.DataArray(
            np.zeros((2, 1, 1)),
            dims=['time', 'northing', 'easting'],
            coords={'time': times},
        )
        results = [{CATCHMENT: {KEY: da}}]

        with pytest.raises(ValueError, match='time steps'):
            exceedance_probability(results, KEY, threshold=5.0)

    def test_time_selects_a_slice_positionally(self):
        times = [pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01')]

        def da(values):
            return xr.DataArray(
                np.array(values).reshape(2, 1, 1),
                dims=['time', 'northing', 'easting'],
                coords={'time': times},
            )

        results = [
            {CATCHMENT: {KEY: da([10.0, 1.0])}},
            {CATCHMENT: {KEY: da([10.0, 1.0])}},
        ]

        assert exceedance_probability(
            results, KEY, threshold=5.0, time=0).values[0, 0] == \
            pytest.approx(1.0)
        assert exceedance_probability(
            results, KEY, threshold=5.0, time=1).values[0, 0] == \
            pytest.approx(0.0)

    def test_time_selects_a_slice_by_label(self):
        times = [pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01')]
        da = xr.DataArray(
            np.array([10.0, 1.0]).reshape(2, 1, 1),
            dims=['time', 'northing', 'easting'],
            coords={'time': times},
        )
        results = [{CATCHMENT: {KEY: da}}]

        result = exceedance_probability(
            results, KEY, threshold=5.0, time=pd.Timestamp('2019-01-01'))
        assert result.values[0, 0] == pytest.approx(1.0)

    def test_single_time_step_needs_no_selector(self):
        da = xr.DataArray(
            np.array([10.0]).reshape(1, 1, 1),
            dims=['time', 'northing', 'easting'],
            coords={'time': [pd.Timestamp('2019-01-01')]},
        )
        results = [{CATCHMENT: {KEY: da}}]

        assert exceedance_probability(
            results, KEY, threshold=5.0).values[0, 0] == pytest.approx(1.0)


class TestResultShape:

    def test_records_provenance_in_attrs(self):
        result = exceedance_probability(
            ensemble([[10.0]], [[1.0]]), KEY, threshold=5.0)

        assert result.attrs['result_key'] == KEY
        assert result.attrs['threshold'] == 5.0
        assert result.attrs['n_replicates'] == 2

    def test_dims_and_dtype(self):
        result = exceedance_probability(
            ensemble([[10.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]),
            KEY, threshold=5.0)

        assert result.dims == ('y', 'x')
        assert result.shape == (2, 2)
        assert result.dtype == np.float32


class TestErrors:

    def test_empty_ensemble_raises(self):
        with pytest.raises(ValueError, match='No replicates'):
            exceedance_probability({}, KEY, threshold=5.0)

    def test_unknown_catchment_raises(self):
        with pytest.raises(KeyError):
            exceedance_probability(
                ensemble([[1.0]]), KEY, threshold=5.0, catchment='Nope')

    def test_ambiguous_catchment_raises(self):
        results = [{
            'A': {KEY: np.array([[1.0]])},
            'B': {KEY: np.array([[1.0]])},
        }]
        with pytest.raises(ValueError, match='Multiple catchments'):
            exceedance_probability(results, KEY, threshold=5.0)

    def test_named_catchment_is_selected(self):
        results = [{
            'A': {KEY: np.array([[10.0]])},
            'B': {KEY: np.array([[1.0]])},
        }]

        assert exceedance_probability(
            results, KEY, threshold=5.0, catchment='A').values[0, 0] == \
            pytest.approx(1.0)

    def test_none_result_raises(self):
        with pytest.raises(ValueError, match='is None'):
            exceedance_probability(
                [{CATCHMENT: {KEY: None}}], KEY, threshold=5.0)
