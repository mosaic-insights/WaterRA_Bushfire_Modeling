"""
Period boundary arithmetic used to bin gridded RUSLE output.

These decide which timesteps land in which output raster, so an off-by-one
here silently double counts or drops erosion at a period boundary.
"""

import pandas as pd
import pytest

from fire_impacts.sim.rusle import _calendar_floor, _compute_periods


TS = pd.Timestamp


class TestCalendarFloor:

    @pytest.mark.parametrize('granularity,expected', [
        ('yearly', '2019-01-01'),
        ('quarterly', '2019-04-01'),
        ('monthly', '2019-06-01'),
        # 2019-06-15 is a Saturday, so the containing week starts Mon 10th
        ('weekly', '2019-06-10'),
        ('daily', '2019-06-15'),
    ])
    def test_floors_to_period_start(self, granularity, expected):
        assert _calendar_floor(TS('2019-06-15 13:45'), granularity) == TS(expected)

    @pytest.mark.parametrize('month,expected_month', [
        (1, 1), (2, 1), (3, 1),
        (4, 4), (5, 4), (6, 4),
        (7, 7), (8, 7), (9, 7),
        (10, 10), (11, 10), (12, 10),
    ])
    def test_every_month_maps_to_its_quarter(self, month, expected_month):
        floored = _calendar_floor(TS(year=2019, month=month, day=20), 'quarterly')
        assert floored == TS(year=2019, month=expected_month, day=1)

    def test_already_on_boundary_is_unchanged(self):
        assert _calendar_floor(TS('2019-01-01'), 'yearly') == TS('2019-01-01')

    def test_accepts_a_string(self):
        assert _calendar_floor('2019-06-15', 'monthly') == TS('2019-06-01')

    def test_rejects_unknown_granularity(self):
        # 'total' has no calendar boundary - _compute_periods handles it
        # before ever reaching here.
        with pytest.raises(ValueError, match='total'):
            _calendar_floor(TS('2019-06-15'), 'total')


class TestComputePeriods:

    def test_total_is_a_single_undivided_span(self):
        periods = _compute_periods(TS('2019-06-15'), TS('2021-03-01'), 'total')
        assert periods == [(TS('2019-06-15'), TS('2021-03-01'))]

    def test_calendar_origin_snaps_first_period_back(self):
        periods = _compute_periods(
            TS('2019-06-15'), TS('2021-03-01'), 'yearly', origin='calendar')

        starts = [ps for ps, _ in periods]
        assert starts == [TS('2019-01-01'), TS('2020-01-01'), TS('2021-01-01')]
        # The first bin is deliberately partial: it is labelled by its
        # calendar boundary, which precedes the simulation start.
        assert starts[0] < TS('2019-06-15')

    def test_fire_origin_starts_at_the_fire(self):
        periods = _compute_periods(
            TS('2019-06-15'), TS('2021-03-01'), 'yearly', origin='fire')

        starts = [ps for ps, _ in periods]
        assert starts == [TS('2019-06-15'), TS('2020-06-15')]

    def test_final_period_is_clipped_to_end(self):
        periods = _compute_periods(
            TS('2019-06-15'), TS('2021-03-01'), 'yearly', origin='fire')
        assert periods[-1][1] == TS('2021-03-01')

    def test_non_final_ends_back_off_one_second(self):
        periods = _compute_periods(
            TS('2019-01-01'), TS('2021-03-01'), 'yearly', origin='fire')

        assert periods[0][1] == TS('2020-01-01') - pd.Timedelta(seconds=1)
        assert periods[1][1] == TS('2021-01-01') - pd.Timedelta(seconds=1)

    def test_periods_neither_overlap_nor_leave_gaps(self):
        periods = _compute_periods(
            TS('2019-02-10'), TS('2020-08-01'), 'monthly')

        for (_, prev_end), (next_start, _) in zip(periods, periods[1:]):
            # Adjacent bins abut exactly: the 1 s back-off means a
            # timestep on the boundary belongs to the later bin only.
            assert prev_end < next_start
            assert next_start - prev_end == pd.Timedelta(seconds=1)

    def test_boundary_timestep_belongs_to_exactly_one_period(self):
        periods = _compute_periods(
            TS('2019-01-01'), TS('2021-01-01'), 'yearly', origin='fire')
        boundary = TS('2020-01-01')

        containing = [
            i for i, (ps, pe) in enumerate(periods)
            if ps <= boundary <= pe
        ]
        assert len(containing) == 1

    def test_period_covers_the_whole_span(self):
        start, end = TS('2019-02-10'), TS('2020-08-01')
        periods = _compute_periods(start, end, 'monthly')

        assert periods[0][0] <= start
        assert periods[-1][1] == end

    @pytest.mark.parametrize('granularity,expected_count', [
        ('yearly', 1),
        ('quarterly', 4),
        ('monthly', 12),
    ])
    def test_period_count_matches_granularity(self, granularity, expected_count):
        periods = _compute_periods(
            TS('2019-01-01'), TS('2020-01-01'), granularity)
        assert len(periods) == expected_count

    def test_empty_span_produces_no_periods(self):
        # A zero-length run records nothing; the recorder then finalises
        # to None and no raster is written.
        assert _compute_periods(TS('2019-01-01'), TS('2019-01-01'), 'yearly') == []

    def test_rejects_unknown_granularity(self):
        with pytest.raises(ValueError, match='Unsupported grid_timestep'):
            _compute_periods(TS('2019-01-01'), TS('2020-01-01'), 'fortnightly')

    def test_rejects_unknown_origin(self):
        with pytest.raises(ValueError, match='origin must be'):
            _compute_periods(
                TS('2019-01-01'), TS('2020-01-01'), 'yearly', origin='epoch')
