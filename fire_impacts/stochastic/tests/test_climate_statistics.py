"""
None means 'let the service estimate it' for the optional climate stats.

``mean_annual_rainfall`` and ``average_temperature`` are optional inputs
to the pyraingen request: leave them out and the backend estimates them
from the catchment's lat/lon. The Simulation notebook now reads them from
``study.toml``, and a settings file cannot express "omit this argument" —
an unset setting arrives as None. So None has to carry the same meaning
as omitting the argument, or a study that does not know its own mean
annual rainfall could not be configured at all.

The request is intercepted at ``requests.get`` rather than at
``get_replicates``, so what is compared is the query the API would
actually receive.
"""

import inspect

import numpy as np
import pytest

pytest.importorskip('geopandas')
import geopandas as gpd
from shapely.geometry import box

from fire_impacts.context import RunContext
from fire_impacts.stochastic.rainfall import replicates as R

START, END = '2019-03-07', '2020-03-06'


class StubProject:
    """Only the paths and boundary get_rainfall_replicates touches."""

    catchments = ['C']

    def __init__(self, root):
        self.root = str(root)

    def catchment_path(self, catchment, *args):
        import os
        return os.path.join(self.root, 'Catchments', catchment, *args)

    def ensemble_path(self, catchment, *args, ensemble):
        import os
        return os.path.join(
            self.root, 'Catchments', catchment, 'Ensembles', ensemble, *args)

    def catchment_boundary(self, catchment):
        return gpd.GeoDataFrame(
            {'geometry': [box(149.0, -35.0, 149.1, -34.9)]}, crs='EPSG:4326')


class FakeResponse:
    status_code = 200
    text = ''

    def __init__(self, num_sims, num_years):
        self._num_sims = num_sims
        self._periods = int(num_years * 366)

    def json(self):
        return {
            'indexes': [{
                'start': '2018-12-31',
                'length': self._periods,
                'step': 86400,
            }],
            'timeseries': [
                {'values': [[0.0, self._periods]], 'scale': 1.0}
                for _ in range(self._num_sims)
            ],
        }


@pytest.fixture()
def captured_params(monkeypatch):
    """Record the query parameters of every outgoing API request."""
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(dict(params or {}))
        return FakeResponse(params['count'], params['length'])

    monkeypatch.setattr(R.requests, 'get', fake_get)
    monkeypatch.setattr(
        R, 'read_raster', lambda *a, **k: (np.full((2, 2), 100.0), {}))
    return seen


@pytest.fixture()
def ctx(tmp_path):
    # No ensemble: nothing is cached, so each call really does hit the API.
    return RunContext(project=StubProject(tmp_path), catchment='C')


def test_none_climate_statistics_mean_estimate_from_the_catchment(
        ctx, captured_params):
    """A study.toml cannot express 'omit this argument', so None has to
    mean what omitting it means.

    Not written as "call it twice and compare the two requests": both
    parameters default to None, so the two calls would pass identical
    arguments and the comparison could not fail whatever the
    implementation did. The equivalence is established by the two
    signature assertions - which a changed default really would break -
    and the content of the request is asserted directly.
    """
    R.get_rainfall_replicates(
        ctx, start=START, end=END, num_replicates=3,
        mean_annual_rainfall=None, average_temperature=None,
    )

    sent, = captured_params
    # Absent, not present-and-None: a None in the query string would
    # reach the service as the literal text 'None'.
    assert 'mean_annual_rainfall' not in sent
    assert 'mean_temperature' not in sent
    # The keys are dropped, not the request. Without this, dropping every
    # parameter would pass.
    assert set(sent) == {
        'latitude', 'longitude', 'elevation', 'length', 'count'}

    # ...and leaving the arguments out lands on exactly the path just
    # exercised, because that is what they default to.
    parameters = inspect.signature(R.get_rainfall_replicates).parameters
    assert parameters['mean_annual_rainfall'].default is None
    assert parameters['average_temperature'].default is None


def test_supplied_climate_statistics_are_sent(ctx, captured_params):
    """The other half of the contract: a value the user did set has to
    reach the service, or the assertion above would pass on a function
    that ignored both arguments."""
    R.get_rainfall_replicates(
        ctx, start=START, end=END, num_replicates=3,
        mean_annual_rainfall=600, average_temperature=20,
    )

    sent, = captured_params
    assert sent['mean_annual_rainfall'] == 600
    assert sent['mean_temperature'] == 20


def test_the_two_statistics_are_independent(ctx, captured_params):
    """One set and one unset is the common case — a study that knows its
    rainfall but not its temperature. Setting one must not smuggle the
    other in."""
    R.get_rainfall_replicates(
        ctx, start=START, end=END, num_replicates=3,
        mean_annual_rainfall=600, average_temperature=None,
    )

    sent, = captured_params
    assert sent['mean_annual_rainfall'] == 600
    assert 'mean_temperature' not in sent
