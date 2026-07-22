"""
scoped_path resolves data folders at the right scope for the multi-event
layout: catchment sections stay put, but FireSeverity resolves per event
and Results / Results_baseline / DebrisFlow resolve per run — but only when
a RunContext carrying the relevant binding is supplied.
"""
import pytest

from fire_impacts import const as c
from fire_impacts.context import RunContext
from .util import CATCHMENT, get_file, get_project  # noqa: F401


@pytest.fixture()
def proj(get_project):
    return get_project()


def _tail(path, *parts):
    """True if path ends with the given components in order."""
    return path.replace('\\', '/').endswith('/'.join(parts))


class TestNoContext:
    """Without a context everything stays at catchment scope (unchanged
    pre-multi-event behaviour)."""

    def test_catchment_section(self, proj):
        p = proj.scoped_path(CATCHMENT, 'Topography', 'DEM.tif')
        assert _tail(p, CATCHMENT, 'Topography', 'DEM.tif')
        assert 'Events' not in p and 'Runs' not in p

    def test_run_section_without_ctx_stays_at_catchment(self, proj):
        p = proj.scoped_path(CATCHMENT, 'Results', 'RUSLE_sum_total.tif')
        assert _tail(p, CATCHMENT, 'Results', 'RUSLE_sum_total.tif')
        assert 'Runs' not in p

    def test_none_section_is_the_catchment_root(self, proj):
        assert proj.scoped_path(CATCHMENT, None) == proj.catchment_path(
            CATCHMENT)


class TestEventScope:

    def test_fire_severity_resolves_under_the_event(self, proj):
        ctx = RunContext.solo_event(proj, event='2019_fire')
        p = proj.scoped_path(CATCHMENT, 'FireSeverity', 'dNBR.tif', ctx=ctx)
        assert _tail(p, 'Events', '2019_fire', 'FireSeverity', 'dNBR.tif')

    def test_catchment_section_ignores_the_event(self, proj):
        ctx = RunContext.solo_event(proj, event='2019_fire')
        p = proj.scoped_path(CATCHMENT, 'Topography', 'DEM.tif', ctx=ctx)
        assert _tail(p, CATCHMENT, 'Topography', 'DEM.tif')
        assert 'Events' not in p


class TestRunScope:

    def test_results_resolve_under_the_run(self, proj):
        ctx = RunContext.solo_run(
            proj, event='2019_fire', ensemble='historical')
        p = proj.scoped_path(CATCHMENT, 'Results', 'RUSLE_sum_total.tif',
                             ctx=ctx)
        assert _tail(
            p, 'Runs', '2019_fire', 'historical', 'Results',
            'RUSLE_sum_total.tif')

    def test_baseline_results_resolve_under_the_run(self, proj):
        ctx = RunContext.solo_run(
            proj, event='2019_fire', ensemble='historical')
        p = proj.scoped_path(CATCHMENT, c.RESULTS_BASELINE_FOLDER_NAME,
                             ctx=ctx)
        assert _tail(
            p, 'Runs', '2019_fire', 'historical',
            c.RESULTS_BASELINE_FOLDER_NAME)

    def test_debris_flow_resolves_under_the_run(self, proj):
        ctx = RunContext.solo_run(
            proj, event='2019_fire', ensemble='historical')
        p = proj.scoped_path(CATCHMENT, 'DebrisFlow',
                             'DebrisFlowData_subcatchments.csv', ctx=ctx)
        assert _tail(
            p, 'Runs', '2019_fire', 'historical', 'DebrisFlow',
            'DebrisFlowData_subcatchments.csv')

    def test_run_section_needs_an_ensemble(self, proj):
        # An event-only context can't reach a run folder, so a run-scoped
        # section falls back to catchment scope rather than raising.
        ctx = RunContext.solo_event(proj, event='2019_fire')
        p = proj.scoped_path(CATCHMENT, 'Results', ctx=ctx)
        assert _tail(p, CATCHMENT, 'Results')
        assert 'Runs' not in p

    def test_event_section_under_a_run_context(self, proj):
        # A run context still resolves FireSeverity at event scope.
        ctx = RunContext.solo_run(
            proj, event='2019_fire', ensemble='historical')
        p = proj.scoped_path(CATCHMENT, 'FireSeverity', 'dNBR.tif', ctx=ctx)
        assert _tail(p, 'Events', '2019_fire', 'FireSeverity', 'dNBR.tif')
        assert 'Runs' not in p
