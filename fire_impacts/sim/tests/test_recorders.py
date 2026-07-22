"""
Grid recorder closures: how per-timestep RUSLE grids get accumulated into
the arrays that eventually become output rasters.
"""

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from fire_impacts.sim.rusle import (
    _spatial_coords_from_transform,
    record_multi_period_grid,
    record_timestep_grid,
)


TS = pd.Timestamp

# 10 m cells, origin at (1000, 2000), northing decreasing down the rows
TRANSFORM = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)

TWO_PERIODS = [
    (TS('2019-01-01'), TS('2020-01-01') - pd.Timedelta(seconds=1)),
    (TS('2020-01-01'), TS('2021-01-01')),
]
ONE_PERIOD = [(TS('2019-01-01'), TS('2020-01-01'))]


def grid(value, shape=(2, 3)):
    return np.full(shape, value, dtype=np.float32)


def feed(recorder, samples, transform=TRANSFORM):
    """Push (timestep, grid) pairs through a recorder."""
    for timestep, data in samples:
        recorder(timestep, RUSLE=data, transform=transform)


class TestMultiPeriodGrid:

    def test_sums_within_each_period(self):
        rec = record_multi_period_grid('RUSLE', 'sum', TWO_PERIODS)
        feed(rec, [
            (TS('2019-03-01'), grid(1.0)),
            (TS('2019-09-01'), grid(2.0)),
            (TS('2020-06-01'), grid(4.0)),
        ])
        result = rec.finalize()

        assert np.allclose(result.sel(time=TWO_PERIODS[0][0]).values, 3.0)
        assert np.allclose(result.sel(time=TWO_PERIODS[1][0]).values, 4.0)

    def test_max_within_each_period(self):
        rec = record_multi_period_grid('RUSLE', 'max', TWO_PERIODS)
        feed(rec, [
            (TS('2019-03-01'), grid(5.0)),
            (TS('2019-09-01'), grid(2.0)),
            (TS('2020-06-01'), grid(4.0)),
        ])
        result = rec.finalize()

        assert np.allclose(result.isel(time=0).values, 5.0)
        assert np.allclose(result.isel(time=1).values, 4.0)

    def test_mean_divides_by_that_periods_own_count(self):
        rec = record_multi_period_grid('RUSLE', 'mean', TWO_PERIODS)
        feed(rec, [
            (TS('2019-03-01'), grid(1.0)),
            (TS('2019-09-01'), grid(3.0)),
            # A single sample in the second period: the mean must be 10,
            # not diluted by the two samples from the first period.
            (TS('2020-06-01'), grid(10.0)),
        ])
        result = rec.finalize()

        assert np.allclose(result.isel(time=0).values, 2.0)
        assert np.allclose(result.isel(time=1).values, 10.0)

    def test_timesteps_outside_every_period_are_ignored(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [
            (TS('2018-06-01'), grid(99.0)),   # before
            (TS('2019-06-01'), grid(1.0)),    # inside
            (TS('2021-06-01'), grid(99.0)),   # after
        ])
        assert np.allclose(rec.finalize().values, 1.0)

    def test_period_bounds_are_inclusive(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [
            (ONE_PERIOD[0][0], grid(1.0)),
            (ONE_PERIOD[0][1], grid(2.0)),
        ])
        assert np.allclose(rec.finalize().values, 3.0)

    def test_single_period_finalises_to_two_dimensions(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [(TS('2019-06-01'), grid(1.0))])
        result = rec.finalize()

        assert result.dims == ('northing', 'easting')
        assert result.shape == (2, 3)

    def test_multiple_periods_add_a_time_dimension(self):
        rec = record_multi_period_grid('RUSLE', 'sum', TWO_PERIODS)
        feed(rec, [(TS('2019-06-01'), grid(1.0))])
        result = rec.finalize()

        assert result.dims == ('time', 'northing', 'easting')
        assert result.shape == (2, 2, 3)
        assert list(result['time'].values) == [
            np.datetime64(ps) for ps, _ in TWO_PERIODS
        ]

    def test_period_with_no_data_becomes_zeros(self):
        # Pinning current behaviour: an unrecorded period is
        # indistinguishable downstream from one that genuinely eroded
        # nothing.
        rec = record_multi_period_grid('RUSLE', 'sum', TWO_PERIODS)
        feed(rec, [(TS('2019-06-01'), grid(1.0))])
        result = rec.finalize()

        assert np.allclose(result.isel(time=1).values, 0.0)

    def test_finalize_returns_none_when_nothing_recorded(self):
        rec = record_multi_period_grid('RUSLE', 'sum', TWO_PERIODS)
        assert rec.finalize() is None

    def test_does_not_mutate_the_caller_grid(self):
        # The recorder accumulates in place, so it must copy the first
        # grid it sees - otherwise it corrupts the simulation's buffer.
        first = grid(1.0)
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [
            (TS('2019-03-01'), first),
            (TS('2019-09-01'), grid(2.0)),
        ])

        assert np.allclose(first, 1.0)
        assert np.allclose(rec.finalize().values, 3.0)

    def test_reset_discards_accumulated_state(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [(TS('2019-03-01'), grid(5.0))])
        rec.reset()

        assert rec.finalize() is None
        feed(rec, [(TS('2019-03-01'), grid(1.0))])
        assert np.allclose(rec.finalize().values, 1.0)

    def test_georeferences_from_the_transform(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [(TS('2019-06-01'), grid(1.0))])
        result = rec.finalize()

        assert np.allclose(result['easting'].values, [1005.0, 1015.0, 1025.0])
        assert np.allclose(result['northing'].values, [1995.0, 1985.0])

    def test_survives_a_missing_transform(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-06-01'), RUSLE=grid(1.0))
        result = rec.finalize()

        assert result.shape == (2, 3)
        assert 'easting' not in result.coords


class TestTimestepGrid:

    def test_stacks_one_slice_per_timestep(self):
        rec = record_timestep_grid('RUSLE')
        stamps = [TS('2019-01-01 00:00'), TS('2019-01-01 00:30'),
                  TS('2019-01-01 01:00')]
        feed(rec, [(t, grid(float(i))) for i, t in enumerate(stamps)])
        result = rec.finalize()

        assert result.dims == ('time', 'northing', 'easting')
        assert result.shape == (3, 2, 3)
        assert list(result['time'].values) == [np.datetime64(t) for t in stamps]
        assert np.allclose(result.isel(time=2).values, 2.0)

    def test_does_not_mutate_the_caller_grid(self):
        data = grid(1.0)
        rec = record_timestep_grid('RUSLE')
        feed(rec, [(TS('2019-01-01'), data)])
        data += 5.0

        assert np.allclose(rec.finalize().isel(time=0).values, 1.0)

    def test_finalize_returns_none_when_nothing_recorded(self):
        assert record_timestep_grid('RUSLE').finalize() is None

    def test_reset_discards_accumulated_state(self):
        rec = record_timestep_grid('RUSLE')
        feed(rec, [(TS('2019-01-01'), grid(1.0))])
        rec.reset()
        assert rec.finalize() is None


class TestSpatialCoordsFromTransform:

    def test_returns_cell_centres(self):
        coords = _spatial_coords_from_transform(TRANSFORM, (2, 3))

        # Half a cell in from the raster origin, not the corner.
        assert np.allclose(coords['easting'], [1005.0, 1015.0, 1025.0])
        assert np.allclose(coords['northing'], [1995.0, 1985.0])

    def test_northing_descends_with_a_north_up_transform(self):
        coords = _spatial_coords_from_transform(TRANSFORM, (4, 1))
        assert np.all(np.diff(coords['northing']) < 0)

    def test_no_transform_gives_no_coords(self):
        assert _spatial_coords_from_transform(None, (2, 3)) == {}
