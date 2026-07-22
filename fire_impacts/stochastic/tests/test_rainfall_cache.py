"""
Caching of stochastic rainfall in the ensemble folder.

get_rainfall_replicates persists its (slow, remote) output to
Ensembles/<ensemble>/rainfall.nc and reuses it on repeat runs, so repeat
simulations don't re-hit the pyraingen API and are driven by identical
rainfall. The API call and the DEM/boundary reads are stubbed here — what
matters is *when* a fresh request is made versus when the cache is reused.
"""
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('geopandas')
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from fire_impacts import const as c
from fire_impacts.context import RunContext
from fire_impacts.stochastic.rainfall import replicates as R


START, END = '2019-03-07', '2020-03-06'


class StubProject:
    """Minimal project: only the paths and the catchment boundary that
    get_rainfall_replicates touches."""

    catchments = ['C']

    def __init__(self, root):
        self.root = str(root)

    def catchment_path(self, catchment, *args):
        return os.path.join(self.root, 'Catchments', catchment, *args)

    def ensemble_path(self, catchment, *args, ensemble):
        return os.path.join(
            self.root, 'Catchments', catchment, 'Ensembles', ensemble, *args)

    def catchment_boundary(self, catchment):
        return gpd.GeoDataFrame(
            {'geometry': [box(149.0, -35.0, 149.1, -34.9)]}, crs='EPSG:4326')


@pytest.fixture()
def stub_api(monkeypatch):
    """Replace the pyraingen call and the DEM read; count API hits."""
    calls = {'n': 0, 'num_years': None}

    def fake_get_replicates(lat, lon, elev, annual_rain, mean_temp,
                            num_years, num_sims, **kw):
        calls['n'] += 1
        calls['num_years'] = num_years
        # Daily series anchored ~Dec 31 of the prior year, spanning
        # num_years — mirrors pyraingen's shape (replicate, time).
        periods = int(num_years * 366)
        idx = pd.date_range('2018-12-31', periods=periods, freq='D')
        data = np.zeros((num_sims, periods), dtype=float)
        da = xr.DataArray(
            data, dims=['replicate', 'time'],
            coords={'replicate': list(range(num_sims)), 'time': idx})
        da.attrs['units'] = 'mm'
        return xr.Dataset({'rainfall': da})

    monkeypatch.setattr(R, 'get_replicates', fake_get_replicates)
    monkeypatch.setattr(R, 'read_raster', lambda *a, **k: (np.full((2, 2), 100.0), {}))
    return calls


@pytest.fixture()
def run_ctx(tmp_path):
    return RunContext(
        project=StubProject(tmp_path), catchment='C',
        event='2019_fire', ensemble='stochastic')


def _get(ctx, **kw):
    kw.setdefault('start', START)
    kw.setdefault('end', END)
    kw.setdefault('num_replicates', 5)
    return R.get_rainfall_replicates(ctx, **kw)


def test_first_call_generates_and_caches(run_ctx, stub_api):
    ds = _get(run_ctx)

    assert stub_api['n'] == 1
    assert os.path.exists(
        os.path.join(run_ctx.ensemble_path(), c.RAINFALL_NAME))
    assert ds.sizes['replicate'] == 5
    times = ds['time'].to_index()
    assert times.min() >= pd.Timestamp(START)
    assert times.max() <= pd.Timestamp(END)


def test_second_call_reuses_the_cache(run_ctx, stub_api):
    _get(run_ctx)
    ds = _get(run_ctx)

    assert stub_api['n'] == 1, 'the API must not be hit a second time'
    assert ds.sizes['replicate'] == 5


def test_reused_rainfall_is_identical(run_ctx, stub_api):
    a = _get(run_ctx)
    b = _get(run_ctx)
    xr.testing.assert_identical(a, b)


def test_regenerate_forces_a_fresh_request(run_ctx, stub_api):
    _get(run_ctx)
    _get(run_ctx, regenerate=True)

    assert stub_api['n'] == 2


def test_no_ensemble_context_is_never_cached(tmp_path, stub_api):
    ctx = RunContext(project=StubProject(tmp_path), catchment='C')

    _get(ctx)
    _get(ctx)

    assert stub_api['n'] == 2, 'without an ensemble there is nowhere to cache'


def test_cache_is_regenerated_for_a_window_it_cannot_cover(run_ctx, stub_api):
    _get(run_ctx)  # caches ~3 years from 2018-12-31
    # A window running years past the cached span cannot be served.
    _get(run_ctx, end='2024-03-06')

    assert stub_api['n'] == 2


def test_more_replicates_than_cached_regenerates(run_ctx, stub_api):
    _get(run_ctx, num_replicates=5)
    _get(run_ctx, num_replicates=12)

    assert stub_api['n'] == 2


def test_fewer_replicates_reuses_and_trims(run_ctx, stub_api):
    _get(run_ctx, num_replicates=10)
    ds = _get(run_ctx, num_replicates=3)

    assert stub_api['n'] == 1
    assert ds.sizes['replicate'] == 3
