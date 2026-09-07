"""
Spatial aggregation of per-timestep grids to a subcatchment time series.

The recorder sees every model timestep and emits one row per agg_count
of them. Most of those timesteps are dry - the simulation hands it a
shared read-only zeros grid for those - so the interesting cases are
what a dry timestep does to a row, and what the recorder is allowed to
do to the array it was given.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

from fire_impacts.sim.rusle import record_subcatchment_timeseries
from fire_impacts.sim.scaled_grid import ScaledGrid


TS = pd.Timestamp

# 4x4 grid of 10 m cells, origin at (0, 40), northing decreasing down
# the rows. Two subcatchments split it down the middle: 'left' is
# columns 0-1, 'right' is columns 2-3.
SHAPE = (4, 4)
TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 40.0)

SUBCATCHMENTS = gpd.GeoDataFrame(
    {'sc_ID': ['left', 'right']},
    geometry=[box(0, 0, 20, 40), box(20, 0, 40, 40)],
)


class FakeProject:
    """Only the three things record_subcatchment_timeseries asks for."""

    def get_subcatchments(self, catchment):
        return SUBCATCHMENTS

    def subcatchment_label_field(self, catchment):
        return 'sc_ID'

    def catchment_boundary(self, catchment):
        raise AssertionError('subcatchments are available')


class FakeCtx:
    project = FakeProject()
    catchment = 'Catchment'


def grid(value):
    return np.full(SHAPE, value, dtype=np.float32)


def read_only(value):
    g = grid(value)
    g.flags.writeable = False
    return g


def recorder(fn='sum', agg_count=1):
    rec = record_subcatchment_timeseries(
        FakeCtx(), 'RUSLE', fn=fn, agg_count=agg_count,
    )
    rec.reset()
    return rec


def feed(rec, samples):
    """Push (timestep, grid) or (timestep, grid, dry) tuples through."""
    for sample in samples:
        timestep, data = sample[0], sample[1]
        dry = sample[2] if len(sample) > 2 else False
        rec(timestep, catchment='Catchment', transform=TRANSFORM,
            RUSLE=data, dry=dry)


def stamps(n):
    return list(pd.date_range('2020-01-01', periods=n, freq='30min'))


def feed_deferred(rec, samples):
    """Push (timestep, scale, unit) triples through as deferred grids."""
    for timestep, scale, unit in samples:
        rec(timestep, catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(scale, unit))


def zone_positions():
    """Flat cell positions of each subcatchment, for brute-force checks."""
    import rasterio.features
    return [
        np.flatnonzero(~np.isnan(rasterio.features.rasterize(
            [g], transform=TRANSFORM, fill=np.nan, dtype=np.float32,
            out_shape=SHAPE)))
        for g in SUBCATCHMENTS.geometry
    ]


class TestZoneAggregation:

    def test_each_zone_sums_only_its_own_cells(self):
        data = np.arange(16, dtype=np.float32).reshape(SHAPE)
        rec = recorder()
        feed(rec, [(stamps(1)[0], data)])
        row = rec.finalize()

        # Columns 0-1 hold 0,1,4,5,8,9,12,13; columns 2-3 the rest.
        assert row['left'].iloc[0] == pytest.approx(52.0)
        assert row['right'].iloc[0] == pytest.approx(68.0)

    def test_cells_outside_the_catchment_are_ignored(self):
        data = grid(2.0)
        data[:, 0] = np.nan          # a whole column masked out
        rec = recorder()
        feed(rec, [(stamps(1)[0], data)])
        row = rec.finalize()

        assert row['left'].iloc[0] == pytest.approx(8.0)   # 4 cells left
        assert row['right'].iloc[0] == pytest.approx(16.0)

    def test_mean_averages_over_that_zones_cells_only(self):
        data = np.arange(16, dtype=np.float32).reshape(SHAPE)
        rec = recorder(fn='mean')
        feed(rec, [(stamps(1)[0], data)])

        assert rec.finalize()['left'].iloc[0] == pytest.approx(52.0 / 8)

    def test_max_is_taken_within_the_zone(self):
        data = np.arange(16, dtype=np.float32).reshape(SHAPE)
        rec = recorder(fn='max')
        feed(rec, [(stamps(1)[0], data)])
        result = rec.finalize()

        assert result['left'].iloc[0] == pytest.approx(13.0)
        assert result['right'].iloc[0] == pytest.approx(15.0)


class TestAggregationWindow:

    def test_a_row_is_emitted_once_per_agg_count_timesteps(self):
        rec = recorder(agg_count=2)
        t = stamps(4)
        feed(rec, [(t[0], grid(1.0)), (t[1], grid(1.0)),
                   (t[2], grid(1.0)), (t[3], grid(1.0))])
        result = rec.finalize()

        assert len(result) == 2
        assert list(result.index) == [t[1], t[3]]

    def test_values_accumulate_across_the_window(self):
        rec = recorder(agg_count=2)
        t = stamps(2)
        feed(rec, [(t[0], grid(1.0)), (t[1], grid(3.0))])

        # 8 cells per zone, 1 + 3 per cell.
        assert rec.finalize()['left'].iloc[0] == pytest.approx(32.0)


class TestDryTimesteps:

    def test_a_dry_timestep_adds_nothing_to_the_row(self):
        rec = recorder(agg_count=2)
        t = stamps(2)
        feed(rec, [(t[0], grid(1.0)), (t[1], read_only(99.0), True)])

        assert rec.finalize()['left'].iloc[0] == pytest.approx(8.0)

    def test_a_window_of_only_dry_timesteps_records_zeros(self):
        rec = recorder(agg_count=2)
        t = stamps(2)
        feed(rec, [(t[0], read_only(99.0), True),
                   (t[1], read_only(99.0), True)])
        result = rec.finalize()

        assert len(result) == 1
        assert result['left'].iloc[0] == pytest.approx(0.0)
        assert result['right'].iloc[0] == pytest.approx(0.0)

    def test_dry_timesteps_still_advance_the_window(self):
        rec = recorder(agg_count=2)
        t = stamps(4)
        feed(rec, [(t[0], read_only(0.0), True), (t[1], grid(1.0)),
                   (t[2], grid(1.0)), (t[3], read_only(0.0), True)])
        result = rec.finalize()

        assert list(result.index) == [t[1], t[3]]
        assert list(result['left']) == pytest.approx([8.0, 8.0])


class TestCallerGrid:

    def test_does_not_mutate_the_grid_it_was_given(self):
        # The recorder accumulates in place across the window, so it has
        # to copy the first grid of each window - the simulation reuses
        # and hands out that buffer elsewhere.
        first = grid(1.0)
        rec = recorder(agg_count=2)
        t = stamps(2)
        feed(rec, [(t[0], first), (t[1], grid(2.0))])

        assert np.allclose(first, 1.0)
        assert rec.finalize()['left'].iloc[0] == pytest.approx(24.0)


def deferred(value, unit):
    return ScaledGrid(np.array([float(value)]), unit)


def layer(value=1.0):
    return np.full(SHAPE, value, dtype=np.float32)


class TestDeferredInput:

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_matches_the_materialised_path(self, fn):
        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        lazy, eager = recorder(fn=fn, agg_count=2), recorder(fn=fn,
                                                            agg_count=2)
        t = stamps(2)
        for stamp, scale in zip(t, [1.0, 3.0]):
            lazy(stamp, catchment='Catchment', transform=TRANSFORM,
                 RUSLE=deferred(scale, unit))
            eager(stamp, catchment='Catchment', transform=TRANSFORM,
                  RUSLE=scale * unit)

        assert np.allclose(lazy.finalize().to_numpy(),
                           eager.finalize().to_numpy())

    def test_cells_outside_the_catchment_stay_excluded(self):
        unit = layer(2.0)
        unit[:, 0] = np.nan
        rec = recorder(agg_count=1)
        feed_deferred(rec, [(stamps(1)[0], 3.0, unit)])
        row = rec.finalize()

        # 4 cells per zone survive on the left, 8 on the right.
        assert row['left'].iloc[0] == pytest.approx(24.0)
        assert row['right'].iloc[0] == pytest.approx(48.0)

    def test_dry_timesteps_add_nothing_but_advance_the_window(self):
        unit = layer(1.0)
        rec = recorder(agg_count=2)
        t = stamps(4)
        rec(t[0], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(0.0, unit), dry=True)
        rec(t[1], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(2.0, unit))
        rec(t[2], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(2.0, unit))
        rec(t[3], catchment='Catchment', transform=TRANSFORM,
            RUSLE=deferred(0.0, unit), dry=True)
        result = rec.finalize()

        assert list(result.index) == [t[1], t[3]]
        assert list(result['left']) == pytest.approx([16.0, 16.0])

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_an_all_dry_window_records_zero_not_nan(self, fn):
        # W is NaN for a zone whose cells are all NaN, so multiplying a
        # zero scale through it would put a silent NaN in the column.
        unit = layer(1.0)
        unit[:] = np.nan
        rec = recorder(fn=fn, agg_count=2)
        t = stamps(2)
        for stamp in t:
            rec(stamp, catchment='Catchment', transform=TRANSFORM,
                RUSLE=deferred(0.0, unit), dry=True)

        assert rec.finalize()['left'].iloc[0] == pytest.approx(0.0)

    def test_mean_ignores_nan_cells_within_the_zone(self):
        # A collapsed mean divides by the zone's valid-cell count, not
        # by len(positions) - an implementation that forgot to exclude
        # the masked column would divide by 8 instead of 4 and return
        # 3.0 rather than 6.0.
        unit = layer(2.0)
        unit[:, 0] = np.nan
        rec = recorder(fn='mean', agg_count=1)
        feed_deferred(rec, [(stamps(1)[0], 3.0, unit)])

        assert rec.finalize()['left'].iloc[0] == pytest.approx(6.0)

    def test_max_ignores_nan_cells_within_the_zone(self):
        # An implementation that took max() over the raw zone values
        # instead of values[keep] would return NaN here, since the
        # masked column's NaN would win the comparison.
        unit = layer(2.0)
        unit[:, 0] = np.nan
        rec = recorder(fn='max', agg_count=1)
        feed_deferred(rec, [(stamps(1)[0], 3.0, unit)])

        assert rec.finalize()['left'].iloc[0] == pytest.approx(6.0)

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_a_shared_unit_is_never_materialised_mid_window(self, fn):
        # A value check alone can't tell "accumulate the scale" apart
        # from "materialise a grid every timestep and combine those" -
        # ScaledGrid duck-types through the old eager path and gives the
        # same numbers either way. The counter is the only thing that
        # can see the difference. agg_count=4 means the flush - and any
        # collapse-vs-materialise decision - happens on the fourth call,
        # inside the loop; asserting immediately after the loop already
        # covers that flush, and asserting again after finalize()
        # confirms finalize() forces no further materialisation either,
        # since the window is empty by then.
        from fire_impacts.sim.scaled_grid import MaterialisationCounter

        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        counter = MaterialisationCounter()
        rec = recorder(fn=fn, agg_count=4)
        t = stamps(4)
        for stamp, scale in zip(t, [1.0, 2.0, 3.0, 4.0]):
            rec(stamp, catchment='Catchment', transform=TRANSFORM,
                RUSLE=ScaledGrid(np.array([scale]), unit,
                                  counter=counter))

        assert counter.count == 0
        rec.finalize()
        assert counter.count == 0


class TestSharedScaleNonMutation:

    def test_two_recorders_do_not_corrupt_the_shared_scale(self):
        # Every recorder in a run - and every one of a timestep's six
        # grid keys - holds the SAME scale array object. That is only
        # safe because the seeding site here copies it (.astype(),
        # which copies by default) rather than aliasing it the way
        # np.asarray(..., dtype=...) would when the dtype already
        # matches. An aliasing bug would let one recorder's += mutate
        # the shared array in place, corrupting whatever else is still
        # reading it.
        unit, scale = layer(2.0), np.array([3.0])
        a, b = recorder(agg_count=2), recorder(agg_count=2)
        t = stamps(2)
        for rec in (a, b):
            rec(t[0], catchment='Catchment', transform=TRANSFORM,
                RUSLE=ScaledGrid(scale, unit))
            rec(t[1], catchment='Catchment', transform=TRANSFORM,
                RUSLE=ScaledGrid(scale, unit))

        assert scale[0] == 3.0
        # 8 cells per zone; accumulated scale (3.0 + 3.0) * unit (2.0)
        # = 12.0 per cell, summed over 8 cells in the 'left' zone.
        assert a.finalize()['left'].iloc[0] == pytest.approx(96.0)
        assert b.finalize()['left'].iloc[0] == pytest.approx(96.0)


class TestWindowStraddlingARecoveryBoundary:

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_a_window_spanning_two_layers_is_exact(self, fn):
        first = np.arange(16, dtype=np.float32).reshape(SHAPE)
        second = first * 10.0
        lazy, eager = recorder(fn=fn, agg_count=2), recorder(fn=fn,
                                                            agg_count=2)
        t = stamps(2)
        lazy(t[0], catchment='Catchment', transform=TRANSFORM,
             RUSLE=deferred(2.0, first))
        lazy(t[1], catchment='Catchment', transform=TRANSFORM,
             RUSLE=deferred(3.0, second))
        eager(t[0], catchment='Catchment', transform=TRANSFORM,
              RUSLE=2.0 * first)
        eager(t[1], catchment='Catchment', transform=TRANSFORM,
              RUSLE=3.0 * second)

        assert np.allclose(lazy.finalize().to_numpy(),
                           eager.finalize().to_numpy())


class TestSpatiallyVaryingRainfall:
    """
    The seam for rainfall coarser than the DEM. Nothing produces this yet;
    these tests are what stop the weight matrix rotting before it does.
    """

    RAIN_INDEX = np.array([[0, 0, 1, 1]] * 4)

    def brute_force(self, scales, unit, fn, rain_index=None):
        """Zonal aggregation the slow, obviously-correct way."""
        if rain_index is None:
            rain_index = self.RAIN_INDEX
        total = None
        for scale in scales:
            step = scale[rain_index] * unit
            total = step if total is None else total + step
        flat = total.reshape(-1)
        agg = {'sum': np.nansum, 'mean': np.nanmean, 'max': np.nanmax}[fn]
        return [agg(flat[z]) for z in zone_positions()]

    @pytest.mark.parametrize('fn', ['sum', 'mean', 'max'])
    def test_matches_a_brute_force_loop(self, fn):
        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        scales = [np.array([2.0, 10.0]), np.array([1.0, 4.0])]
        rec = recorder(fn=fn, agg_count=2)
        for stamp, scale in zip(stamps(2), scales):
            rec(stamp, catchment='Catchment', transform=TRANSFORM,
                RUSLE=ScaledGrid(scale, unit, self.RAIN_INDEX))

        assert np.allclose(rec.finalize().to_numpy()[0],
                           self.brute_force(scales, unit, fn))

    def test_a_shared_unit_with_a_different_rain_index_is_not_cached(self):
        # The zonal-weights cache is keyed on identity of BOTH unit and
        # rain_index. Two windows sharing a unit object but carrying
        # different rain_index maps must each get their own weights -
        # reusing the first window's weights for the second would
        # silently mix up which rain cell feeds which zone.
        unit = np.arange(16, dtype=np.float32).reshape(SHAPE)
        other_index = np.array([[1, 1, 0, 0]] * 4)
        scale = np.array([2.0, 10.0])
        rec = recorder(fn='sum', agg_count=1)
        t = stamps(2)
        rec(t[0], catchment='Catchment', transform=TRANSFORM,
            RUSLE=ScaledGrid(scale, unit, self.RAIN_INDEX))
        rec(t[1], catchment='Catchment', transform=TRANSFORM,
            RUSLE=ScaledGrid(scale, unit, other_index))
        result = rec.finalize()

        expected_first = self.brute_force(
            [scale], unit, 'sum', self.RAIN_INDEX)
        expected_second = self.brute_force(
            [scale], unit, 'sum', other_index)

        assert np.allclose(result.to_numpy()[0], expected_first)
        assert np.allclose(result.to_numpy()[1], expected_second)


class TestInvalidAggregationFunction:

    def test_an_unrecognised_fn_raises_at_construction(self):
        with pytest.raises(ValueError):
            recorder(fn='median')
