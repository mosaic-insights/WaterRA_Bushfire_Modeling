"""
EventRunContext: the persisted description of a fire event and the
recovery windows modelled for it.

The class itself does no I/O - FireImpactsProject reads and writes it as
JSON - so to_dict/from_dict is the storage contract, and the absolute
window arithmetic is what maps years-since-fire onto real rainfall dates.
"""

import dataclasses
import json

import pandas as pd
import pytest

from fire_impacts import const as c
from fire_impacts.run_context import EventRunContext


TS = pd.Timestamp
FIRE_START = TS('2019-12-30')
FIRE_END = TS('2020-01-15')


def ctx(**kwargs):
    kwargs.setdefault('fire_start_date', FIRE_START)
    kwargs.setdefault('fire_end_date', FIRE_END)
    return EventRunContext(**kwargs)


class TestDefaults:

    def test_dates_default_to_none(self):
        blank = EventRunContext()
        assert blank.fire_start_date is None
        assert blank.fire_end_date is None

    def test_breakpoints_default_to_the_package_default(self):
        assert EventRunContext().recovery_breakpoints \
            == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_each_instance_gets_its_own_breakpoint_list(self):
        # A shared mutable default would leak edits between contexts.
        a, b = EventRunContext(), EventRunContext()
        assert a.recovery_breakpoints is not b.recovery_breakpoints
        assert a.recovery_breakpoints is not c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx().fire_end_date = TS('2021-01-01')


class TestRecoveryStructure:

    def test_recovery_times_are_the_window_starts(self):
        assert ctx(recovery_breakpoints=[0, 1, 2]).recovery_times() == [0, 1]

    def test_windows_pairs_the_breakpoints(self):
        assert ctx(recovery_breakpoints=[0, 1, 2]).windows() == [(0, 1), (1, 2)]

    def test_window_for_finds_a_window_by_its_start(self):
        assert ctx(recovery_breakpoints=[0, 0.5, 1]).window_for(0.5) == (0.5, 1)

    def test_window_for_rejects_a_non_start(self):
        # 1 is the final breakpoint, so it closes a window rather than
        # starting one - there is no layer modelled at T=1 here.
        with pytest.raises(ValueError, match='not a window start'):
            ctx(recovery_breakpoints=[0, 0.5, 1]).window_for(1)

    def test_window_for_rejects_an_unknown_time(self):
        with pytest.raises(ValueError, match='not a window start'):
            ctx(recovery_breakpoints=[0, 1, 2]).window_for(0.5)

    def test_invalid_breakpoints_raise_when_used(self):
        # Construction is permissive; validation happens in recovery_windows.
        bad = ctx(recovery_breakpoints=[2, 1])
        with pytest.raises(ValueError, match='strictly increasing'):
            bad.windows()


class TestAbsoluteWindow:

    def test_measures_from_the_fire_end_date(self):
        start, end = ctx(recovery_breakpoints=[0, 1, 2]).absolute_window(0)

        assert start == FIRE_END
        assert end == FIRE_END + pd.DateOffset(days=365)

    def test_half_year_window(self):
        start, end = ctx(recovery_breakpoints=[0, 0.5, 1]).absolute_window(0.5)

        # int(0.5 * 365.25) == 182, int(1 * 365.25) == 365
        assert start == FIRE_END + pd.DateOffset(days=182)
        assert end == FIRE_END + pd.DateOffset(days=365)

    def test_consecutive_windows_abut_exactly(self):
        # No gap and no overlap: each window's end is the next one's start,
        # so no day of rainfall is modelled twice or skipped.
        context = ctx()
        windows = [
            context.absolute_window(rt) for rt in context.recovery_times()
        ]
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            assert prev_end == next_start

    def test_windows_are_ordered_and_non_empty(self):
        context = ctx()
        for recovery_time in context.recovery_times():
            start, end = context.absolute_window(recovery_time)
            assert start < end

    def test_rejects_a_non_window_start(self):
        with pytest.raises(ValueError, match='not a window start'):
            ctx(recovery_breakpoints=[0, 1]).absolute_window(0.5)

    def test_needs_a_fire_end_date(self):
        no_end = EventRunContext(
            fire_start_date=FIRE_START,
            fire_end_date=None,
            recovery_breakpoints=[0, 1],
        )
        with pytest.raises(ValueError, match='no fire_end_date'):
            no_end.absolute_window(0)


class TestSimulationPeriod:

    def test_spans_first_to_last_breakpoint(self):
        start, end = ctx(recovery_breakpoints=[0, 1, 2]).simulation_period()

        assert start == FIRE_END
        assert end == FIRE_END + pd.DateOffset(days=730)

    def test_covers_every_recovery_window(self):
        context = ctx()
        sim_start, sim_end = context.simulation_period()

        for recovery_time in context.recovery_times():
            start, end = context.absolute_window(recovery_time)
            assert sim_start <= start
            assert end <= sim_end

    def test_non_zero_first_breakpoint_delays_the_start(self):
        start, _ = ctx(recovery_breakpoints=[1, 2]).simulation_period()
        assert start == FIRE_END + pd.DateOffset(days=365)

    def test_needs_a_fire_end_date(self):
        with pytest.raises(ValueError, match='no fire_end_date'):
            EventRunContext(recovery_breakpoints=[0, 1]).simulation_period()


class TestSerialisation:

    def test_round_trips(self):
        original = ctx(recovery_breakpoints=[0, 0.5, 1])
        restored = EventRunContext.from_dict(original.to_dict())

        assert restored == original

    def test_round_trips_through_json(self):
        # This is how FireImpactsProject actually persists it.
        original = ctx()
        restored = EventRunContext.from_dict(
            json.loads(json.dumps(original.to_dict())))

        assert restored == original

    def test_to_dict_is_json_serialisable(self):
        json.dumps(ctx().to_dict())

    def test_dates_serialise_as_iso_strings(self):
        data = ctx().to_dict()

        assert data['fire_start_date'] == '2019-12-30'
        assert data['fire_end_date'] == '2020-01-15'

    def test_none_dates_survive(self):
        restored = EventRunContext.from_dict(EventRunContext().to_dict())

        assert restored.fire_start_date is None
        assert restored.fire_end_date is None

    def test_from_dict_parses_date_strings(self):
        restored = EventRunContext.from_dict({
            'fire_start_date': '2019-12-30',
            'fire_end_date': '2020-01-15',
            'recovery_breakpoints': [0, 1],
        })

        assert restored.fire_end_date == FIRE_END
        assert isinstance(restored.fire_end_date, pd.Timestamp)

    def test_time_of_day_is_dropped(self):
        # KNOWN SHARP EDGE, pinned rather than endorsed: _to_iso formats
        # as %Y-%m-%d, so a context built with a timestamp does not
        # survive a save/load unchanged. Harmless while fire dates are
        # whole days, which is what fire-severity preprocessing produces.
        precise = ctx(fire_end_date=TS('2020-01-15 13:45'))
        restored = EventRunContext.from_dict(precise.to_dict())

        assert restored.fire_end_date == TS('2020-01-15')
        assert restored != precise

    @pytest.mark.parametrize('breakpoints', [None, []])
    def test_falsy_breakpoints_fall_back_to_the_default(self, breakpoints):
        # from_dict tests truthiness, so an explicitly empty list is
        # silently replaced rather than kept or rejected.
        restored = EventRunContext.from_dict({
            'fire_end_date': '2020-01-15',
            'recovery_breakpoints': breakpoints,
        })

        assert restored.recovery_breakpoints == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_missing_keys_fall_back_to_defaults(self):
        restored = EventRunContext.from_dict({})

        assert restored.fire_start_date is None
        assert restored.fire_end_date is None
        assert restored.recovery_breakpoints == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_breakpoints_are_copied_not_aliased(self):
        breakpoints = [0, 1, 2]
        restored = EventRunContext.from_dict(
            {'recovery_breakpoints': breakpoints})
        breakpoints.append(3)

        assert restored.recovery_breakpoints == [0, 1, 2]

    def test_to_dict_does_not_alias_the_context(self):
        context = ctx(recovery_breakpoints=[0, 1])
        data = context.to_dict()
        data['recovery_breakpoints'].append(2)

        assert context.recovery_breakpoints == [0, 1]

    def test_reloaded_breakpoints_name_the_same_layers(self):
        # The C/K/SDR layers are written from the breakpoints passed to
        # compute_adjusted_k_c and read back from here, so a round trip
        # must not change the suffixes those names are built from.
        restored = EventRunContext.from_dict(
            json.loads(json.dumps(ctx().to_dict())))

        assert [c.recovery_time_suffix(b)
                for b in restored.recovery_breakpoints] \
            == [c.recovery_time_suffix(b)
                for b in c.DEFAULT_RECOVERY_BREAKPOINTS]

    def test_float_breakpoints_reload_to_the_same_layers(self):
        # Whichever numeric type the JSON happens to hold, the layer
        # names must match those written from the int-valued defaults.
        floats = [float(b) for b in c.DEFAULT_RECOVERY_BREAKPOINTS]
        restored = EventRunContext.from_dict(json.loads(json.dumps({
            'fire_end_date': '2020-01-15',
            'recovery_breakpoints': floats,
        })))

        assert [c.recovery_time_suffix(b)
                for b in restored.recovery_breakpoints] \
            == [c.recovery_time_suffix(b)
                for b in c.DEFAULT_RECOVERY_BREAKPOINTS]
