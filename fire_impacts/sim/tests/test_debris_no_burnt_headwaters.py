"""
A fire that leaves no headwater above the debris-flow dNBR threshold.

That is a legitimate outcome - the fire missed the catchment, or burnt it
too lightly to trigger debris flow - so it must give zero debris flow
with a warning, not an error. It used to fail deep in debris_flow() with
"max() iterable argument is empty", after logging that every headwater
had been "excluded upstream" when the dNBR filter had removed them.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from fire_impacts import const as c
from fire_impacts.context import RunContext
from fire_impacts.pre.tests.util import *  # noqa: F401,F403 - fixtures
from fire_impacts.sim import debris


ID = 'hw_ID'
THRESHOLD = 100.0


def condition(ids, dnbr):
    return pd.DataFrame({ID: ids, c.DNBR_MEAN: dnbr})


def topography(ids):
    return pd.DataFrame({ID: ids, 'Area_m2': 1.0})


class TestSelectingBurntHeadwaters:

    def test_none_burnt_gives_an_empty_table(self):
        joined = debris._join_burnt_headwaters(
            condition([1, 2], [10.0, 20.0]), topography([1, 2]),
            ID, THRESHOLD)

        assert joined.empty

    def test_none_burnt_warns_with_the_threshold(self, caplog):
        with caplog.at_level(logging.WARNING, logger=debris.__name__):
            debris._join_burnt_headwaters(
                condition([1, 2], [10.0, 20.0]), topography([1, 2]),
                ID, THRESHOLD)

        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'No headwaters' in warnings[0]
        assert '100' in warnings[0]
        assert '20' in warnings[0]  # the highest mean dNBR found

    def test_dnbr_removals_are_not_reported_as_upstream(self, caplog):
        # Headwater 3 has no condition data (excluded upstream); 1 and 2
        # fall to the dNBR filter. Only 3 is an upstream exclusion.
        with caplog.at_level(logging.INFO, logger=debris.__name__):
            debris._join_burnt_headwaters(
                condition([1, 2], [10.0, 20.0]), topography([1, 2, 3]),
                ID, THRESHOLD)

        messages = [r.getMessage() for r in caplog.records]
        assert any('2 headwaters removed' in m for m in messages)
        assert any('1 of 3 headwaters already excluded upstream' in m
                   for m in messages)

    def test_some_burnt_keeps_those_and_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=debris.__name__):
            joined = debris._join_burnt_headwaters(
                condition([1, 2], [10.0, 200.0]), topography([1, 2]),
                ID, THRESHOLD)

        assert joined[ID].tolist() == [2]
        assert 'Area_m2' in joined.columns
        assert not [r for r in caplog.records
                    if r.levelno == logging.WARNING]

    def test_condition_without_topography_still_raises(self):
        with pytest.raises(ValueError, match='missing from Headwaters.csv'):
            debris._join_burnt_headwaters(
                condition([1, 9], [200.0, 200.0]), topography([1]),
                ID, THRESHOLD)


@pytest.fixture
def run_ctx(get_project):
    proj = get_project()
    run = RunContext.solo_run(proj, event='fire', ensemble='ens')
    Path(run.ensemble_path()).mkdir(parents=True, exist_ok=True)
    severity = Path(run.event_path(c.FIRE_SEVERITY_FOLDER_NAME))
    severity.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        {'Value': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-10')]},
        index=['start_date', 'end_date'])
    meta.index.name = 'Key'
    meta.to_csv(severity / 'FireMeta.csv', date_format='%Y-%m-%d')
    return run


def no_headwaters():
    """The shape prep_debris_flow_simulation returns with none burnt."""
    return pd.DataFrame({
        debris.HW_ID: pd.Series(dtype='int64'),
        c.I12_CRIT_Y + '1': pd.Series(dtype='float64'),
        c.I12_CRIT_Y + '2': pd.Series(dtype='float64'),
        debris.DEBRIS_MASS_FIELD: pd.Series(dtype='float64'),
    })


def rain():
    idx = pd.date_range('2020-01-10', periods=24 * 5, freq='12min')
    series = pd.Series(50.0, index=idx)
    series.attrs['units'] = 'mm/h'
    return series


class TestDebrisFlowWithNoHeadwaters:

    def test_runs_and_returns_no_events(self, run_ctx):
        summary, event_ts = debris.debris_flow(
            run_ctx, rain(), save=False,
            save_daily_catchment_timeseries=True,
            prepared=no_headwaters())

        assert summary.empty
        assert event_ts.shape == (len(rain()), 0)

    def test_mass_timeseries_is_empty_not_an_error(self, run_ctx):
        summary, event_ts = debris.debris_flow(
            run_ctx, rain(), save=False,
            save_daily_catchment_timeseries=False,
            prepared=no_headwaters())

        mass = debris.event_ts_to_mass(summary, event_ts)
        assert mass.shape == (len(rain()), 0)

    def test_aggregates_to_zero_load_per_subcatchment(self):
        # Allocated headwaters with no series still give their
        # subcatchment a column, so the combined load is RUSLE plus zero.
        mass = pd.DataFrame(index=rain().index)
        allocations = pd.DataFrame({
            debris.SC_ID: [0, 1], debris.HW_ID: [1, 2],
            'area_fraction': [1.0, 1.0]})

        aggregated = debris.aggregate_debris_to_subcatchments(
            mass, allocations)

        assert aggregated.columns.tolist() == [0, 1]
        assert (aggregated.values == 0).all()


class TestMappingWithNoHeadwaters:
    """The Simulation notebook maps the debris tables straight after
    debris_flow(); with none burnt those tables have no rows."""

    def test_an_empty_table_maps_as_plain_shapes(self, get_project,
                                                 get_file):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        proj = get_project()
        catchment = proj.catchments[0]
        proj.add_subcatchments(
            catchment, str(get_file('Subcatchments_EgSmall_7899.shp')),
            id_cols=['Id'])
        polygons = proj.get_subcatchments(catchment)
        id_col = proj.subcatchment_id
        # As read back from a CSV with a header and no rows.
        empty = pd.DataFrame({
            id_col: pd.Series(dtype=object),
            'Year1_num_events': pd.Series(dtype=object),
        })

        proj.plot_catchment_polygons(
            catchment=catchment, polygons=polygons,
            colour_col='Year1_num_events',
            vis_params=proj.get_vis_params('Year1_num_events'),
            title='no debris flow', non_geo_data=empty, id_col=id_col)
        plt.close('all')
