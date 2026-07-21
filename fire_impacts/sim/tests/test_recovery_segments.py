"""
Splitting a rainfall series into the per-recovery-window segments a
continuous fire-adjusted run processes.

Each segment must pick up the C/K/SDR layers for its own recovery time, so
the mapping from window to segment is what decides which erodibility a
given day of rain is modelled against.
"""

import pandas as pd
import pytest

from fire_impacts.run_context import EventRunContext
from fire_impacts.sim.rusle import _recovery_run_segments


TS = pd.Timestamp
FIRE_END = TS('2019-01-01')
CATCHMENT = 'Eg'


class FakeProject:
    """Minimal stand-in for FireImpactsProject."""

    def __init__(self, tmp_path, ctx):
        self.root = tmp_path
        self.ctx = ctx
        (tmp_path / 'Erodibility').mkdir(parents=True, exist_ok=True)

    def get_run_context(self, catchment):
        return self.ctx

    def catchment_path(self, catchment, section, file_name=None):
        path = self.root / section
        return str(path / file_name) if file_name else str(path)

    def add_layers(self, *suffixes):
        """Create the empty C-factor rasters the splitter checks for."""
        for suffix in suffixes:
            (self.root / 'Erodibility'
             / f'C_factor_adjusted_{suffix}.tif').touch()


@pytest.fixture()
def rainfall():
    # Three years of daily rain starting at the fire end date
    index = pd.date_range(FIRE_END, periods=3 * 365, freq='D')
    return pd.DataFrame({'rainfall': 1.0}, index=index)


@pytest.fixture()
def make_project(tmp_path):
    def _(breakpoints=(0, 1, 2), fire_end_date=FIRE_END, layers=None):
        ctx = EventRunContext(
            fire_start_date=FIRE_END,
            fire_end_date=fire_end_date,
            recovery_breakpoints=list(breakpoints),
        )
        proj = FakeProject(tmp_path, ctx)
        if layers is None:
            layers = ['t%d' % b for b in breakpoints[:-1]]
        proj.add_layers(*layers)
        return proj
    return _


class TestBaseline:

    def test_baseline_is_one_unsplit_segment(self, make_project, rainfall):
        proj = make_project()
        segments = _recovery_run_segments(
            proj, CATCHMENT, rainfall, use_fire_adjusted=False)

        assert len(segments) == 1
        recovery_time, segment = segments[0]
        assert recovery_time is None
        assert segment is rainfall

    def test_baseline_ignores_a_missing_run_context(self, tmp_path, rainfall):
        class NoContext:
            def get_run_context(self, catchment):
                raise AssertionError('baseline must not read the run-context')

        segments = _recovery_run_segments(
            NoContext(), CATCHMENT, rainfall, use_fire_adjusted=False)
        assert segments == [(None, rainfall)]


class TestFireAdjusted:

    def test_one_segment_per_recovery_window(self, make_project, rainfall):
        proj = make_project(breakpoints=(0, 1, 2))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        assert [rt for rt, _ in segments] == [0, 1]

    def test_segments_are_chronological(self, make_project, rainfall):
        proj = make_project(breakpoints=(0, 1, 2))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        first, second = (seg for _, seg in segments)
        assert first.index.max() < second.index.min()

    def test_segments_do_not_overlap_or_lose_rain(self, make_project, rainfall):
        proj = make_project(breakpoints=(0, 1, 2))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        covered = pd.DatetimeIndex([])
        for _, segment in segments:
            assert covered.intersection(segment.index).empty
            covered = covered.append(segment.index)

        # Two one-year windows off a 3-year series: everything inside the
        # windows is covered exactly once, the third year falls outside.
        window_end = FIRE_END + pd.DateOffset(days=int(2 * 365.25))
        expected = rainfall.index[rainfall.index < window_end]
        assert covered.equals(expected)

    def test_window_is_half_open(self, make_project, rainfall):
        # [start, end) - the boundary day belongs to the later window.
        proj = make_project(breakpoints=(0, 1, 2))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)
        boundary = FIRE_END + pd.DateOffset(days=365)

        assert boundary not in segments[0][1].index
        assert boundary in segments[1][1].index

    def test_fractional_recovery_times(self, make_project, rainfall):
        proj = make_project(
            breakpoints=(0, 0.5, 1), layers=['t0', 't0_5'])
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        assert [rt for rt, _ in segments] == [0, 0.5]

    def test_windows_without_rainfall_are_skipped(self, make_project, rainfall):
        # The series only spans 3 years, so the 4th window is empty.
        proj = make_project(breakpoints=(0, 1, 2, 3, 4))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        assert [rt for rt, _ in segments] == [0, 1, 2]

    def test_all_segments_are_non_empty(self, make_project, rainfall):
        proj = make_project(breakpoints=(0, 1, 2, 3, 4))
        segments = _recovery_run_segments(proj, CATCHMENT, rainfall, True)

        assert all(not segment.empty for _, segment in segments)


class TestErrors:

    def test_missing_fire_end_date(self, make_project, rainfall):
        proj = make_project(fire_end_date=None)

        with pytest.raises(ValueError, match='No fire end date'):
            _recovery_run_segments(proj, CATCHMENT, rainfall, True)

    def test_missing_adjusted_layer(self, make_project, rainfall):
        # Window T=1 has rainfall but no C_factor_adjusted_t1.tif.
        proj = make_project(breakpoints=(0, 1, 2), layers=['t0'])

        with pytest.raises(FileNotFoundError, match='recovery T=1'):
            _recovery_run_segments(proj, CATCHMENT, rainfall, True)

    def test_rainfall_outside_every_window(self, make_project):
        proj = make_project(breakpoints=(0, 1))
        away = pd.DataFrame(
            {'rainfall': 1.0},
            index=pd.date_range('2025-01-01', periods=10, freq='D'),
        )

        with pytest.raises(ValueError, match='No rainfall overlaps'):
            _recovery_run_segments(proj, CATCHMENT, away, True)

    def test_empty_rainfall(self, make_project):
        proj = make_project(breakpoints=(0, 1))
        empty = pd.DataFrame(
            {'rainfall': []}, index=pd.DatetimeIndex([], name=None))

        with pytest.raises(ValueError, match='No rainfall overlaps'):
            _recovery_run_segments(proj, CATCHMENT, empty, True)
