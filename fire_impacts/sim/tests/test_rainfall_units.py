"""
Rainfall depth/intensity conversion.

Everything downstream of rainfall is scaled by these, so a units error
here multiplies every erosion number in the model rather than perturbing
it - which makes it the sort of bug that looks plausible in the output.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts.sim.rainfall import (
    depth_to_intensity,
    get_stamps_per_hour,
    intensity_to_depth,
)


def series(freq, values=None, periods=8, col='depth_mm'):
    index = pd.date_range('2019-01-01', periods=periods, freq=freq)
    if values is None:
        values = np.arange(periods, dtype=float)
    return pd.DataFrame({col: values}, index=index)


class TestStampsPerHour:

    @pytest.mark.parametrize('freq,expected', [
        ('30min', 2.0),
        ('12min', 5.0),
        ('h', 1.0),
        ('15min', 4.0),
        ('5min', 12.0),
        ('6min', 10.0),
    ])
    def test_uniform_frequencies(self, freq, expected):
        assert get_stamps_per_hour(series(freq)) == pytest.approx(expected)

    def test_daily_data_is_sub_unity(self):
        assert get_stamps_per_hour(series('D')) == pytest.approx(1 / 24)

    def test_uses_the_median_so_a_gap_does_not_skew_it(self):
        # One missing hour in an otherwise 30-minute series.
        index = pd.DatetimeIndex([
            '2019-01-01 00:00', '2019-01-01 00:30', '2019-01-01 01:00',
            '2019-01-01 02:00',  # gap
            '2019-01-01 02:30', '2019-01-01 03:00',
        ])
        ts = pd.DataFrame({'depth_mm': 1.0}, index=index)

        assert get_stamps_per_hour(ts) == pytest.approx(2.0)

    def test_duplicate_timestamps_are_ignored(self):
        index = pd.DatetimeIndex([
            '2019-01-01 00:00', '2019-01-01 00:00',
            '2019-01-01 00:30', '2019-01-01 01:00',
        ])
        ts = pd.DataFrame({'depth_mm': 1.0}, index=index)

        assert get_stamps_per_hour(ts) == pytest.approx(2.0)

    def test_unsorted_index_is_sorted_first(self):
        ordered = series('30min', periods=4)
        shuffled = ordered.iloc[[2, 0, 3, 1]]

        assert get_stamps_per_hour(shuffled) == \
            pytest.approx(get_stamps_per_hour(ordered))

    def test_two_timestamps_are_enough(self):
        ts = pd.DataFrame(
            {'depth_mm': [1.0, 2.0]},
            index=pd.DatetimeIndex(['2019-01-01 00:00', '2019-01-01 00:30']),
        )
        assert get_stamps_per_hour(ts) == pytest.approx(2.0)

    def test_all_duplicate_timestamps_raise(self):
        ts = pd.DataFrame(
            {'depth_mm': [1.0, 2.0]},
            index=pd.DatetimeIndex(['2019-01-01', '2019-01-01']),
        )
        with pytest.raises(ValueError, match='duplicates'):
            get_stamps_per_hour(ts)

    def test_single_timestamp_raises(self):
        ts = pd.DataFrame(
            {'depth_mm': [1.0]}, index=pd.DatetimeIndex(['2019-01-01']))

        with pytest.raises(ValueError, match='duplicates'):
            get_stamps_per_hour(ts)


class TestDepthToIntensity:

    def test_half_hourly_depth_doubles(self):
        # 5 mm falling in half an hour is 10 mm/h.
        ts = series('30min', values=[5.0] * 4, periods=4)
        assert np.allclose(depth_to_intensity(ts, 'depth_mm'), 10.0)

    def test_twelve_minute_depth_scales_by_five(self):
        ts = series('12min', values=[2.0] * 5, periods=5)
        assert np.allclose(depth_to_intensity(ts, 'depth_mm'), 10.0)

    def test_hourly_depth_is_unchanged(self):
        ts = series('h', values=[3.0] * 4, periods=4)
        assert np.allclose(depth_to_intensity(ts, 'depth_mm'), 3.0)

    def test_preserves_the_index(self):
        ts = series('30min')
        assert depth_to_intensity(ts, 'depth_mm').index.equals(ts.index)


class TestIntensityToDepth:

    def test_half_hourly_intensity_halves(self):
        ts = series('30min', values=[10.0] * 4, periods=4, col='intensity')
        assert np.allclose(intensity_to_depth(ts, 'intensity'), 5.0)

    def test_hourly_intensity_is_unchanged(self):
        ts = series('h', values=[3.0] * 4, periods=4, col='intensity')
        assert np.allclose(intensity_to_depth(ts, 'intensity'), 3.0)


class TestRoundTrip:

    @pytest.mark.parametrize('freq', ['5min', '12min', '30min', 'h'])
    def test_depth_survives_a_round_trip(self, freq):
        depth = series(freq, values=[0.0, 1.5, 7.25, 0.5], periods=4)

        intensity = depth_to_intensity(depth, 'depth_mm')
        recovered = intensity_to_depth(
            intensity.to_frame('intensity'), 'intensity')

        assert np.allclose(recovered, depth['depth_mm'])

    def test_total_depth_is_conserved(self):
        # The physical invariant: converting to intensity and back must
        # not change how much rain fell.
        depth = series('12min', values=[1.0, 2.0, 3.0, 4.0, 5.0], periods=5)

        intensity = depth_to_intensity(depth, 'depth_mm')
        recovered = intensity_to_depth(
            intensity.to_frame('intensity'), 'intensity')

        assert recovered.sum() == pytest.approx(depth['depth_mm'].sum())

    def test_intensity_and_depth_are_exact_inverses(self):
        rng = np.random.default_rng(0)
        values = rng.uniform(0, 20, 24)
        frame = series('30min', values=values, periods=24, col='intensity')

        depth = intensity_to_depth(frame, 'intensity')
        recovered = depth_to_intensity(depth.to_frame('depth_mm'), 'depth_mm')

        assert np.allclose(recovered, frame['intensity'])
