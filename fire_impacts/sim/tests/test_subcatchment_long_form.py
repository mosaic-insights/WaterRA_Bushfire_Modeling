"""
Long-form frames built for the subcatchment choropleths.

A project need not register a subcatchment label field: without one the
wide frames are keyed by the raw subcatchment ID, and the long form must
still carry that ID column exactly once - plot_catchment_polygons()
merges on it, and pandas refuses a merge key that labels two columns.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from fire_impacts.sim.ensemble import (
    reduce_ensemble_subcatchments,
    subcatchment_series_to_long,
)


CATCHMENT = 'Eg'


class FakeProject:

    subcatchment_id = 'sc_ID'

    def __init__(self, label_field=None):
        self._label_field = label_field
        self._subs = gpd.GeoDataFrame(
            {
                'sc_ID': [0, 1],
                'SiteID': ['upper', 'lower'],
            },
            geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
            crs='EPSG:3577',
        )

    def get_subcatchments(self, catchment):
        return self._subs

    def subcatchment_label_field(self, catchment):
        return self._label_field


def wide(columns, values):
    return pd.DataFrame([values], columns=columns)


class TestWithoutLabelField:

    def test_series_to_long_has_one_id_column(self):
        long = subcatchment_series_to_long(
            wide([0, 1], [10.0, 20.0]),
            project=FakeProject(), catchment=CATCHMENT, time=0)

        assert list(long.columns) == ['sc_ID', 'value']
        assert long['value'].tolist() == [10.0, 20.0]

    def test_ensemble_reduction_has_one_id_column(self):
        ensemble = {
            0: wide([0, 1], [10.0, 20.0]),
            1: wide([0, 1], [30.0, 40.0]),
        }
        long = reduce_ensemble_subcatchments(
            ensemble, project=FakeProject(), catchment=CATCHMENT,
            time=0, normalise_by='area_ha')

        assert list(long.columns) == ['sc_ID', 'value_mean_per_ha']
        # Each subcatchment is 1 ha.
        assert long['value_mean_per_ha'].tolist() == [20.0, 30.0]


class TestWithLabelField:

    def test_carries_both_id_and_label(self):
        long = subcatchment_series_to_long(
            wide(['upper', 'lower'], [10.0, 20.0]),
            project=FakeProject('SiteID'), catchment=CATCHMENT, time=0)

        assert list(long.columns) == ['sc_ID', 'SiteID', 'value']
        assert long['sc_ID'].tolist() == [0, 1]
