"""
Dispatch of recorder results to GeoTIFFs.

save_catchment_raster is stubbed out throughout - what matters here is
which results get written, and under what names, not the raster I/O.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fire_impacts.sim import rusle
from fire_impacts.sim.rusle import _MAX_GRID_SLICES_TO_DISK, _save_grid_results


TS = pd.Timestamp
META = {'width': 3, 'height': 2}


@pytest.fixture()
def writes(monkeypatch):
    """Capture save_catchment_raster calls instead of writing rasters."""
    calls = []

    def fake_save(project, catchment_name, file_name, section, data, meta):
        calls.append({'name': file_name, 'section': section, 'data': data})

    monkeypatch.setattr(rusle, 'save_catchment_raster', fake_save)
    return calls


def grid(value=1.0, shape=(2, 3)):
    return np.full(shape, value, dtype=np.float32)


def da_2d(value=1.0):
    return xr.DataArray(grid(value), dims=['northing', 'easting'])


def da_3d(times, value=1.0):
    data = np.stack([grid(value) for _ in times])
    return xr.DataArray(
        data, dims=['time', 'northing', 'easting'], coords={'time': list(times)})


def save(results, writes=None):
    return _save_grid_results(None, 'Catchment', 'Results', results, META)


class TestDispatch:

    def test_writes_a_two_dimensional_numpy_grid(self, writes):
        saved = save({'erosion_total': grid()})

        assert saved == ['erosion_total']
        assert [w['name'] for w in writes] == ['erosion_total']
        assert writes[0]['section'] == 'Results'

    def test_writes_a_two_dimensional_dataarray(self, writes):
        saved = save({'RUSLE_sum_total': da_2d()})

        assert saved == ['RUSLE_sum_total']
        assert [w['name'] for w in writes] == ['RUSLE_sum_total']

    def test_writes_one_raster_per_time_slice(self, writes):
        times = [TS('2019-01-01'), TS('2020-01-01'), TS('2021-01-01')]
        saved = save({'RUSLE_sum_yearly': da_3d(times)})

        assert saved == [
            'RUSLE_sum_yearly_20190101',
            'RUSLE_sum_yearly_20200101',
            'RUSLE_sum_yearly_20210101',
        ]
        assert [w['name'] for w in writes] == saved

    def test_time_slices_are_written_as_two_dimensional_arrays(self, writes):
        save({'RUSLE_sum_yearly': da_3d([TS('2019-01-01'), TS('2020-01-01')])})

        for w in writes:
            assert w['data'].ndim == 2
            assert w['data'].shape == (2, 3)

    @pytest.mark.parametrize('key,value', [
        ('params', {'anything': 1}),
        ('empty_recorder', None),
        ('RUSLE_timeseries', pd.DataFrame({'a': [1, 2]})),
        ('transform', (10.0, 0.0, 1000.0)),
        ('a_scalar', 42),
    ])
    def test_skips_non_grid_results(self, writes, key, value):
        assert save({key: value}) == []
        assert writes == []

    def test_skips_a_three_dimensional_numpy_array(self, writes):
        # Only DataArrays carry the time coords needed to label slices.
        assert save({'raw_stack': np.zeros((3, 2, 3))}) == []
        assert writes == []

    def test_mixed_results_write_only_the_grids(self, writes):
        saved = save({
            'params': {'x': 1},
            'RUSLE_timeseries': pd.DataFrame({'a': [1]}),
            'erosion_total': grid(),
            'RUSLE_sum_total': da_2d(),
            'nothing': None,
        })
        assert sorted(saved) == ['RUSLE_sum_total', 'erosion_total']


class TestSliceCap:

    def test_writes_up_to_the_cap(self, writes):
        times = pd.date_range('2019-01-01', periods=_MAX_GRID_SLICES_TO_DISK)
        saved = save({'daily': da_3d(times)})

        assert len(saved) == _MAX_GRID_SLICES_TO_DISK

    def test_skips_beyond_the_cap(self, writes):
        times = pd.date_range('2019-01-01', periods=_MAX_GRID_SLICES_TO_DISK + 1)
        saved = save({'daily': da_3d(times)})

        # Kept in memory only, rather than writing 501 rasters.
        assert saved == []
        assert writes == []

    def test_capped_recorder_does_not_block_the_others(self, writes):
        times = pd.date_range('2019-01-01', periods=_MAX_GRID_SLICES_TO_DISK + 1)
        saved = save({'daily': da_3d(times), 'erosion_total': grid()})

        assert saved == ['erosion_total']


class TestSliceNaming:
    """
    record_timestep_grid is sub-daily (one slice per 30 min model
    timestep), so a date-only label collided and every slice after the
    first silently overwrote its predecessor.
    """

    def test_sub_daily_slices_get_unique_names(self, writes):
        times = pd.date_range('2019-01-01', periods=48, freq='30min')
        saved = save({'RUSLE': da_3d(times)})

        assert len(saved) == 48
        assert len(set(saved)) == 48, 'slice names collided - rasters overwritten'

    def test_sub_daily_names_carry_the_time(self, writes):
        times = pd.date_range('2019-01-01', periods=3, freq='30min')
        saved = save({'RUSLE': da_3d(times)})

        assert saved == [
            'RUSLE_20190101_0000',
            'RUSLE_20190101_0030',
            'RUSLE_20190101_0100',
        ]

    def test_daily_and_coarser_names_are_unchanged(self, writes):
        # Period grids must keep their existing on-disk names.
        times = [TS('2019-01-01'), TS('2020-01-01')]
        saved = save({'RUSLE_sum_yearly': da_3d(times)})

        assert saved == ['RUSLE_sum_yearly_20190101', 'RUSLE_sum_yearly_20200101']

    def test_every_slice_is_written_once(self, writes):
        times = pd.date_range('2019-01-01', periods=48, freq='30min')
        save({'RUSLE': da_3d(times)})

        names = [w['name'] for w in writes]
        assert len(names) == len(set(names))
