"""
EventDefinition: one fire event's recovery-window setup.

It carries the fire dates (loaded from the event's FireMeta.csv) plus the
recovery breakpoints, and the absolute-window arithmetic maps
years-since-fire onto real rainfall dates. Only the breakpoints are
persisted to event.json (the dates stay in FireMeta), so to_dict/from_dict
is a breakpoints-only contract.
"""

import dataclasses
import json

import pandas as pd
import pytest

from fire_impacts import const as c
from fire_impacts.context import EventDefinition


TS = pd.Timestamp
FIRE_START = TS('2019-12-30')
FIRE_END = TS('2020-01-15')


def definition(**kwargs):
    kwargs.setdefault('fire_start_date', FIRE_START)
    kwargs.setdefault('fire_end_date', FIRE_END)
    return EventDefinition(**kwargs)


class TestDefaults:

    def test_dates_default_to_none(self):
        blank = EventDefinition()
        assert blank.fire_start_date is None
        assert blank.fire_end_date is None

    def test_breakpoints_default_to_the_package_default(self):
        assert EventDefinition().recovery_breakpoints \
            == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_each_instance_gets_its_own_breakpoint_list(self):
        # A shared mutable default would leak edits between definitions.
        a, b = EventDefinition(), EventDefinition()
        assert a.recovery_breakpoints is not b.recovery_breakpoints
        assert a.recovery_breakpoints is not c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            definition().fire_end_date = TS('2021-01-01')


class TestRecoveryStructure:

    def test_recovery_times_are_the_window_starts(self):
        assert definition(
            recovery_breakpoints=[0, 1, 2]).recovery_times() == [0, 1]

    def test_windows_pairs_the_breakpoints(self):
        assert definition(
            recovery_breakpoints=[0, 1, 2]).windows() == [(0, 1), (1, 2)]

    def test_window_for_finds_a_window_by_its_start(self):
        assert definition(
            recovery_breakpoints=[0, 0.5, 1]).window_for(0.5) == (0.5, 1)

    def test_window_for_rejects_a_non_start(self):
        # 1 is the final breakpoint, so it closes a window rather than
        # starting one - there is no layer modelled at T=1 here.
        with pytest.raises(ValueError, match='not a window start'):
            definition(recovery_breakpoints=[0, 0.5, 1]).window_for(1)

    def test_window_for_rejects_an_unknown_time(self):
        with pytest.raises(ValueError, match='not a window start'):
            definition(recovery_breakpoints=[0, 1, 2]).window_for(0.5)

    def test_invalid_breakpoints_raise_when_used(self):
        # Construction is permissive; validation happens in recovery_windows.
        bad = definition(recovery_breakpoints=[2, 1])
        with pytest.raises(ValueError, match='strictly increasing'):
            bad.windows()


class TestAbsoluteWindow:

    def test_measures_from_the_fire_end_date(self):
        start, end = definition(
            recovery_breakpoints=[0, 1, 2]).absolute_window(0)

        assert start == FIRE_END
        assert end == FIRE_END + pd.DateOffset(days=365)

    def test_half_year_window(self):
        start, end = definition(
            recovery_breakpoints=[0, 0.5, 1]).absolute_window(0.5)

        # int(0.5 * 365.25) == 182, int(1 * 365.25) == 365
        assert start == FIRE_END + pd.DateOffset(days=182)
        assert end == FIRE_END + pd.DateOffset(days=365)

    def test_consecutive_windows_abut_exactly(self):
        # No gap and no overlap: each window's end is the next one's start,
        # so no day of rainfall is modelled twice or skipped.
        d = definition()
        windows = [d.absolute_window(rt) for rt in d.recovery_times()]
        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            assert prev_end == next_start

    def test_windows_are_ordered_and_non_empty(self):
        d = definition()
        for recovery_time in d.recovery_times():
            start, end = d.absolute_window(recovery_time)
            assert start < end

    def test_rejects_a_non_window_start(self):
        with pytest.raises(ValueError, match='not a window start'):
            definition(recovery_breakpoints=[0, 1]).absolute_window(0.5)

    def test_needs_a_fire_end_date(self):
        no_end = EventDefinition(
            fire_start_date=FIRE_START,
            fire_end_date=None,
            recovery_breakpoints=[0, 1],
        )
        with pytest.raises(ValueError, match='no fire_end_date'):
            no_end.absolute_window(0)


class TestSimulationPeriod:

    def test_spans_first_to_last_breakpoint(self):
        start, end = definition(
            recovery_breakpoints=[0, 1, 2]).simulation_period()

        assert start == FIRE_END
        assert end == FIRE_END + pd.DateOffset(days=730)

    def test_covers_every_recovery_window(self):
        d = definition()
        sim_start, sim_end = d.simulation_period()

        for recovery_time in d.recovery_times():
            start, end = d.absolute_window(recovery_time)
            assert sim_start <= start
            assert end <= sim_end

    def test_non_zero_first_breakpoint_delays_the_start(self):
        start, _ = definition(recovery_breakpoints=[1, 2]).simulation_period()
        assert start == FIRE_END + pd.DateOffset(days=365)

    def test_needs_a_fire_end_date(self):
        with pytest.raises(ValueError, match='no fire_end_date'):
            EventDefinition(recovery_breakpoints=[0, 1]).simulation_period()


class TestSerialisation:
    """to_dict/from_dict round-trips only the recovery breakpoints;
    event.json never stores the fire dates (they live in FireMeta.csv)."""

    def test_to_dict_holds_only_the_breakpoints(self):
        data = definition(recovery_breakpoints=[0, 0.5, 1]).to_dict()

        assert data == {'recovery_breakpoints': [0, 0.5, 1]}
        assert 'fire_start_date' not in data
        assert 'fire_end_date' not in data

    def test_to_dict_is_json_serialisable(self):
        json.dumps(definition().to_dict())

    def test_breakpoints_round_trip(self):
        original = definition(recovery_breakpoints=[0, 0.5, 1])
        restored = EventDefinition.from_dict(original.to_dict())

        assert restored.recovery_breakpoints == [0, 0.5, 1]

    def test_round_trips_through_json(self):
        # This is how the event definition is actually persisted.
        original = definition(recovery_breakpoints=[0, 1, 2])
        restored = EventDefinition.from_dict(
            json.loads(json.dumps(original.to_dict())))

        assert restored.recovery_breakpoints == [0, 1, 2]

    def test_from_dict_takes_the_dates_from_the_caller(self):
        # The dates are supplied by RunContext (which reads FireMeta), not
        # carried in the dict.
        restored = EventDefinition.from_dict(
            {'recovery_breakpoints': [0, 1]},
            fire_start_date=FIRE_START,
            fire_end_date=FIRE_END,
        )

        assert restored.fire_start_date == FIRE_START
        assert restored.fire_end_date == FIRE_END

    def test_from_dict_defaults_the_dates_to_none(self):
        restored = EventDefinition.from_dict({'recovery_breakpoints': [0, 1]})

        assert restored.fire_start_date is None
        assert restored.fire_end_date is None

    @pytest.mark.parametrize('breakpoints', [None, []])
    def test_falsy_breakpoints_fall_back_to_the_default(self, breakpoints):
        # from_dict tests truthiness, so an explicitly empty list is
        # silently replaced rather than kept or rejected.
        restored = EventDefinition.from_dict(
            {'recovery_breakpoints': breakpoints})

        assert restored.recovery_breakpoints == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_missing_key_falls_back_to_the_default(self):
        restored = EventDefinition.from_dict({})

        assert restored.recovery_breakpoints == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_breakpoints_are_copied_not_aliased(self):
        breakpoints = [0, 1, 2]
        restored = EventDefinition.from_dict(
            {'recovery_breakpoints': breakpoints})
        breakpoints.append(3)

        assert restored.recovery_breakpoints == [0, 1, 2]

    def test_to_dict_does_not_alias_the_definition(self):
        d = definition(recovery_breakpoints=[0, 1])
        data = d.to_dict()
        data['recovery_breakpoints'].append(2)

        assert d.recovery_breakpoints == [0, 1]

    def test_reloaded_breakpoints_name_the_same_layers(self):
        # The C/K/SDR layers are written from the breakpoints passed to
        # compute_adjusted_k_c and read back from here, so a round trip
        # must not change the suffixes those names are built from.
        restored = EventDefinition.from_dict(
            json.loads(json.dumps(definition().to_dict())))

        assert [c.recovery_time_suffix(b)
                for b in restored.recovery_breakpoints] \
            == [c.recovery_time_suffix(b)
                for b in c.DEFAULT_RECOVERY_BREAKPOINTS]

    def test_float_breakpoints_reload_to_the_same_layers(self):
        # Whichever numeric type the JSON happens to hold, the layer names
        # must match those written from the int-valued defaults.
        floats = [float(b) for b in c.DEFAULT_RECOVERY_BREAKPOINTS]
        restored = EventDefinition.from_dict(
            json.loads(json.dumps({'recovery_breakpoints': floats})))

        assert [c.recovery_time_suffix(b)
                for b in restored.recovery_breakpoints] \
            == [c.recovery_time_suffix(b)
                for b in c.DEFAULT_RECOVERY_BREAKPOINTS]
