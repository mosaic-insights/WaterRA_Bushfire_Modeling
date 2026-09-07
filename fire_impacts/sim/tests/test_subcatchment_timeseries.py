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
