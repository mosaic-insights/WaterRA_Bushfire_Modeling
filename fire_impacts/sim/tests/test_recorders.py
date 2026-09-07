"""
Grid recorder closures: how per-timestep RUSLE grids get accumulated into
the arrays that eventually become output rasters.
"""

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from fire_impacts.sim.recorders import _spread
from fire_impacts.sim.rusle import (
    _spatial_coords_from_transform,
    record_multi_period_grid,
    record_timestep_grid,
)
from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid


TS = pd.Timestamp

# 10 m cells, origin at (1000, 2000), northing decreasing down the rows
TRANSFORM = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)

TWO_PERIODS = [
    (TS('2019-01-01'), TS('2020-01-01') - pd.Timedelta(seconds=1)),
    (TS('2020-01-01'), TS('2021-01-01')),
]
ONE_PERIOD = [(TS('2019-01-01'), TS('2020-01-01'))]

SHAPE = (2, 3)


def grid(value, shape=SHAPE):
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


class TestMultiPeriodGridDryTimesteps:
    """
    A timestep flagged dry erodes nothing, so the simulation hands every
    recorder one shared read-only zeros grid rather than six fresh ones.
    The recorder must not read or write it - but a dry timestep is still
    a timestep, so it has to keep counting towards a mean.
    """

    def test_dry_timesteps_do_not_contribute_to_a_sum(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=grid(1.0), transform=TRANSFORM, dry=False)
        rec(TS('2019-09-01'), RUSLE=grid(99.0), transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 1.0)

    def test_dry_timesteps_do_not_contribute_to_a_max(self):
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=grid(1.0), transform=TRANSFORM, dry=False)
        rec(TS('2019-09-01'), RUSLE=grid(99.0), transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 1.0)

    def test_dry_timesteps_still_count_towards_a_mean(self):
        # Two timesteps in the period, only one of them wet: the mean
        # over the period is 4/2, not 4/1.
        rec = record_multi_period_grid('RUSLE', 'mean', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=grid(4.0), transform=TRANSFORM, dry=False)
        rec(TS('2019-09-01'), RUSLE=grid(99.0), transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 2.0)

    def test_a_period_of_only_dry_timesteps_finalises_to_zeros(self):
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=grid(99.0), transform=TRANSFORM, dry=True)
        result = rec.finalize()

        assert result.shape == (2, 3)
        assert np.allclose(result.values, 0.0)

    def test_a_read_only_dry_grid_is_never_written_to(self):
        read_only = grid(0.0)
        read_only.flags.writeable = False
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)

        rec(TS('2019-03-01'), RUSLE=read_only, transform=TRANSFORM, dry=True)
        rec(TS('2019-06-01'), RUSLE=grid(1.0), transform=TRANSFORM, dry=False)
        rec(TS('2019-09-01'), RUSLE=read_only, transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 1.0)

    def test_precision_does_not_depend_on_a_dry_first_timestep(self):
        # The accumulator takes its dtype from the first grid it keeps.
        # Seeding it from a dry timestep used to pin it to the float32
        # zeros grid and quietly round every later float64 addition -
        # so the precision of a five-year sum depended on whether it
        # happened to start raining.
        wet = np.full(SHAPE, 1.0, dtype=np.float64)

        dry_first = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        dry_first(TS('2019-02-01'), RUSLE=grid(0.0), transform=TRANSFORM,
                  dry=True)
        dry_first(TS('2019-03-01'), RUSLE=wet, transform=TRANSFORM)

        wet_first = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        wet_first(TS('2019-03-01'), RUSLE=wet, transform=TRANSFORM)

        assert dry_first.finalize().dtype == wet_first.finalize().dtype

    def test_timesteps_are_wet_when_no_flag_is_given(self):
        # Callers that predate the flag - and every existing test in
        # this file - must keep accumulating exactly as before.
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        feed(rec, [(TS('2019-03-01'), grid(2.0))])

        assert np.allclose(rec.finalize().values, 2.0)


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


class TestMultiPeriodGridDeferredInput:
    """
    The recorder accumulates the scale and touches a grid once, at the
    end. Every result must match what it would have produced from the
    materialised grids.
    """

    @staticmethod
    def deferred(value, unit):
        return ScaledGrid(np.array([float(value)]), unit)

    def test_sum_matches_the_materialised_path(self):
        unit = grid(2.0)
        lazy = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        eager = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        for stamp, scale in [(TS('2019-03-01'), 1.0),
                              (TS('2019-09-01'), 3.0)]:
            lazy(stamp, RUSLE=self.deferred(scale, unit),
                 transform=TRANSFORM)
            eager(stamp, RUSLE=scale * unit, transform=TRANSFORM)

        assert np.allclose(lazy.finalize().values, eager.finalize().values)
        assert np.allclose(lazy.finalize().values, 8.0)

    def test_max_matches_the_materialised_path(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(5.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(2.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 10.0)

    def test_mean_divides_by_the_period_timestep_count(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(1.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(3.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 4.0)

    def test_a_new_layer_starts_a_new_accumulator(self):
        # A recovery boundary hands the recorder a different unit layer.
        # Both contributions must reach the sum.
        first, second = grid(2.0), grid(10.0)
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(1.0, first),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(1.0, second),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 12.0)

    def test_max_across_two_layers_takes_the_larger_product(self):
        first, second = grid(2.0), grid(10.0)
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(5.0, first),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(1.0, second),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, 10.0)

    def test_max_across_two_layers_is_elementwise_not_scalar(self):
        # Uniform layers can't tell an elementwise np.maximum apart from
        # a bug that compares the two layers' scales as plain numbers
        # and picks one whole grid. These two layers cross over, so
        # only a genuine per-cell max gives the right answer.
        first = np.array([[1.0, 10.0]])
        second = np.array([[10.0, 1.0]])
        rec = record_multi_period_grid('RUSLE', 'max', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(5.0, first),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(1.0, second),
            transform=TRANSFORM)

        # first * 5 = [[5, 50]]; second * 1 = [[10, 1]]; elementwise
        # max = [[10, 50]] - neither whole input grid on its own.
        assert np.allclose(rec.finalize().values, [[10.0, 50.0]])

    def test_dry_timesteps_contribute_nothing_but_still_count(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(4.0, unit),
            transform=TRANSFORM)
        rec(TS('2019-09-01'), RUSLE=self.deferred(0.0, unit),
            transform=TRANSFORM, dry=True)

        assert np.allclose(rec.finalize().values, 4.0)

    def test_a_dry_period_still_finalises_to_plain_zeros(self):
        # A dry timestep registers no layer at all, so this period is
        # indistinguishable from one that recorded nothing - and keeps
        # today's NaN-free zeros rather than inheriting a layer's mask.
        unit = grid(2.0)
        unit[0, 0] = np.nan
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'), RUSLE=self.deferred(0.0, unit),
            transform=TRANSFORM, dry=True)
        result = rec.finalize()

        assert result.shape == (2, 3)
        assert not np.isnan(result.values).any()
        assert np.allclose(result.values, 0.0)

    def test_a_period_with_no_timesteps_does_not_divide_by_zero(self):
        unit = grid(2.0)
        rec = record_multi_period_grid('RUSLE', 'mean', TWO_PERIODS)
        rec(TS('2019-06-01'), RUSLE=self.deferred(4.0, unit),
            transform=TRANSFORM)

        assert np.allclose(rec.finalize().isel(time=1).values, 0.0)

    @pytest.mark.parametrize('fn, expected', [
        ('sum', 8.0),
        ('max', 6.0),
        ('mean', 4.0),
    ])
    def test_the_recorder_never_materialises_a_shared_layer(
        self, fn, expected,
    ):
        # A value check alone can't tell "accumulate the scale" apart
        # from "materialise a grid every timestep and combine those" -
        # ScaledGrid duck-types through the old eager path and gives
        # the same numbers either way. The counter is the only thing
        # that can see the difference: it only advances when something
        # actually calls .materialise() (directly, or via np.asarray,
        # .copy(), or an arithmetic ufunc), and none of those should
        # happen before finalize() for a run that never goes dry.
        unit = grid(2.0)
        counter = MaterialisationCounter()
        rec = record_multi_period_grid('RUSLE', fn, ONE_PERIOD)
        for stamp, scale in [(TS('2019-03-01'), 1.0),
                              (TS('2019-09-01'), 3.0)]:
            data = ScaledGrid(
                np.array([scale]), unit, counter=counter)
            rec(stamp, RUSLE=data, transform=TRANSFORM)

        assert counter.count == 0

        assert np.allclose(rec.finalize().values, expected)

    def test_two_recorders_do_not_corrupt_the_shared_scale(self):
        # All six ScaledGrids of one timestep - and every recorder that
        # sees them - share a single scale array. Correctness depends
        # entirely on the seeding site copying it (via .astype(), which
        # copies by default) rather than aliasing it (as
        # np.asarray(..., dtype=...) would when the dtype already
        # matches). An aliasing bug would make one recorder's += mutate
        # the generator's shared array, which every other recorder -
        # and every other grid key of that timestep - is still reading.
        unit, scale = grid(2.0), np.array([3.0])
        a = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        b = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        for rec in (a, b):
            rec(TS('2019-03-01'), RUSLE=ScaledGrid(scale, unit),
                transform=TRANSFORM)
            rec(TS('2019-09-01'), RUSLE=ScaledGrid(scale, unit),
                transform=TRANSFORM)

        assert scale[0] == 3.0
        assert np.allclose(a.finalize().values, 12.0)
        assert np.allclose(b.finalize().values, 12.0)

    def test_a_mismatched_rain_index_does_not_blend_into_an_entry(self):
        # Two ScaledGrids can share a unit object while carrying
        # different rain_index maps - a producer bug, but one the
        # recorder must not amplify into silently wrong output by
        # treating them as the same accumulator and later spreading
        # the blended scale through whichever index happened to be
        # captured first.
        unit = grid(2.0)
        index_a = np.zeros(SHAPE, dtype=int)
        index_b = np.ones(SHAPE, dtype=int)
        rec = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        rec(TS('2019-03-01'),
            RUSLE=ScaledGrid(np.array([1.0, 99.0]), unit, index_a),
            transform=TRANSFORM)
        rec(TS('2019-09-01'),
            RUSLE=ScaledGrid(np.array([99.0, 3.0]), unit, index_b),
            transform=TRANSFORM)
        eager = record_multi_period_grid('RUSLE', 'sum', ONE_PERIOD)
        eager(TS('2019-03-01'),
              RUSLE=np.asarray(
                  ScaledGrid(np.array([1.0, 99.0]), unit, index_a)),
              transform=TRANSFORM)
        eager(TS('2019-09-01'),
              RUSLE=np.asarray(
                  ScaledGrid(np.array([99.0, 3.0]), unit, index_b)),
              transform=TRANSFORM)

        assert np.allclose(rec.finalize().values, eager.finalize().values)


class TestSpread:
    """
    _spread turns one accumulated (scale, rain_index, unit) triple back
    into a grid. Task 5 reuses it once rainfall varies in space, via
    the rain_index branch this exercises directly.
    """

    def test_uniform_rainfall_broadcasts_the_single_scale(self):
        unit = np.array([[1.0, 2.0], [3.0, 4.0]])
        scale = np.array([5.0])

        result = _spread(scale, None, unit)

        assert np.allclose(result, [[5.0, 10.0], [15.0, 20.0]])

    def test_spatially_varying_rainfall_indexes_the_scale_per_cell(self):
        unit = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Two rainfall cells: the left column reads scale[0], the
        # right column reads scale[1].
        rain_index = np.array([[0, 1], [0, 1]])
        scale = np.array([10.0, 100.0])

        result = _spread(scale, rain_index, unit)

        assert np.allclose(result, [[10.0, 200.0], [30.0, 400.0]])
