"""
Aggregating stochastic rainfall to the model timestep.

aggregate_rainfall_data resamples a replicate set from pyraingen's
6-minute output to whatever the simulation needs - 30 minutes for
erosion, 12 for debris flow. The dataset is small, so the cost should
track the amount of data, not the number of output bins.
"""

import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fire_impacts.sim.rainfall import aggregate_rainfall_data


def replicates(n_steps, n_replicates=3, units='mm', freq='6min',
               start='2001-12-31 12:06'):
    """A pyraingen-shaped dataset already flattened to (replicate, time)."""
    rng = np.random.default_rng(0)
    values = rng.random((n_replicates, n_steps)) * 2.0
    values[values < 1.5] = 0.0            # mostly dry, as real rainfall is
    time_index = pd.date_range(start, periods=n_steps, freq=freq)
    da = xr.DataArray(
        values,
        dims=['replicate', 'time'],
        coords={
            'replicate': [f'Simulation_{i}' for i in range(n_replicates)],
            'time': time_index,
        },
        attrs={'units': units},
    )
    return xr.Dataset({'rainfall': da})


class TestMatchesXarrayResample:
    """
    xarray's own resample is the reference implementation - it is just
    far too slow to keep using. Every result has to stay identical to it.
    """

    def test_depth_sums_match(self):
        ds = replicates(4000)
        expected = ds.resample(time='30min').sum()
        result = aggregate_rainfall_data(ds, time_res='30min')

        xr.testing.assert_allclose(result, expected)

    def test_intensity_means_match(self):
        ds = replicates(4000, units='mm/h')
        expected = ds.resample(time='12min').mean()
        result = aggregate_rainfall_data(ds, time_res='12min')

        xr.testing.assert_allclose(result, expected)

    def test_daily_aggregation_matches(self):
        ds = replicates(4000)
        expected = ds.resample(time='D').sum()
        result = aggregate_rainfall_data(ds, time_res='D')

        xr.testing.assert_allclose(result, expected)

    def test_gaps_in_the_series_still_produce_empty_bins(self):
        ds = replicates(400).isel(time=np.r_[0:100, 300:400])
        expected = ds.resample(time='30min').sum()
        result = aggregate_rainfall_data(ds, time_res='30min')

        xr.testing.assert_allclose(result, expected)


    def test_a_gap_in_the_record_stays_missing_rather_than_zero(self):
        ds = replicates(400).isel(time=np.r_[0:100, 300:400])
        result = aggregate_rainfall_data(ds, time_res='30min')['rainfall']

        # The dropped stretch is missing rainfall, not an absence of it.
        assert np.isnan(result.values).any()

    def test_a_bin_holding_only_missing_values_still_sums_to_zero(self):
        # Distinct from a gap: these timestamps exist, so xarray sums
        # them - skipping the NaNs - to zero.
        # Start on a bin boundary so indices 5-9 are exactly the
        # second half-hour and nothing else.
        ds = replicates(600, start='2020-01-01')
        ds['rainfall'][:, 5:10] = np.nan
        expected = ds.resample(time='30min').sum()
        result = aggregate_rainfall_data(ds, time_res='30min')

        assert result['rainfall'].values[1, 0] == 0.0
        xr.testing.assert_allclose(result, expected)


class TestOutputShape:

    def test_time_leads_the_remaining_dimensions(self):
        # Callers index the result as rainfall[:, replicate_idx], so
        # time has to come first - which is also where xarray's own
        # resample puts the dimension it grouped on.
        ds = replicates(4000)
        result = aggregate_rainfall_data(ds, time_res='30min')

        assert result['rainfall'].dims == ('time', 'replicate')
        assert list(result['replicate'].values) == list(ds['replicate'].values)

    def test_extra_dimensions_keep_their_relative_order(self):
        # Spatial rainfall will add a dimension beside replicate; it has
        # to land where xarray would have put it.
        ds = replicates(600)
        ds = ds.expand_dims(station=[0, 1]).transpose(
            'replicate', 'time', 'station')
        expected = ds.resample(time='30min').sum()
        result = aggregate_rainfall_data(ds, time_res='30min')

        assert result['rainfall'].dims == expected['rainfall'].dims
        xr.testing.assert_allclose(result, expected)

    def test_units_attribute_survives(self):
        result = aggregate_rainfall_data(replicates(600), time_res='30min')
        assert result['rainfall'].attrs['units'] == 'mm'

    def test_unrecognised_units_are_rejected(self):
        ds = replicates(600, units='inches')
        with pytest.raises(ValueError, match='Unrecognized rainfall units'):
            aggregate_rainfall_data(ds, time_res='30min')


class TestCost:

    def test_cost_does_not_scale_with_the_number_of_output_bins(self):
        # xarray's groupby-reduce costs ~3.4 ms per output bin regardless
        # of how much data lands in it, which is what made a 35 MB
        # dataset take minutes. 40,000 6-minute steps is 8,000 half-hour
        # bins - comfortably over a minute the slow way.
        ds = replicates(40_000)

        started = time.perf_counter()
        aggregate_rainfall_data(ds, time_res='30min')
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, f'took {elapsed:.1f}s'
