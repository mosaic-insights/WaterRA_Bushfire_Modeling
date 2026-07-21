"""
Recovery breakpoint algebra and the filename suffixes derived from it.

recovery_time_suffix decides on-disk names for the fire-adjusted C/K/SDR
layers and the ensemble folders, so its output is a storage contract, not
just a formatting detail.
"""

import pytest

from fire_impacts import const as c


class TestRecoveryTimeSuffix:

    @pytest.mark.parametrize('recovery_time,expected', [
        (0, 't0'),
        (0.0, 't0'),
        (0.5, 't0_5'),
        (1, 't1'),
        (1.5, 't1_5'),
        (2.5, 't2_5'),
    ])
    def test_documented_examples(self, recovery_time, expected):
        assert c.recovery_time_suffix(recovery_time) == expected

    def test_suffix_has_no_dot(self):
        # A '.' would read as a file extension in the middle of a name.
        for breakpoint in c.DEFAULT_RECOVERY_BREAKPOINTS:
            assert '.' not in c.recovery_time_suffix(breakpoint)

    def test_default_breakpoints_give_distinct_suffixes(self):
        suffixes = [
            c.recovery_time_suffix(b) for b in c.DEFAULT_RECOVERY_BREAKPOINTS
        ]
        assert len(set(suffixes)) == len(suffixes)

    @pytest.mark.parametrize('whole_number', [0, 1, 2, 3])
    def test_int_and_float_agree_for_whole_numbers(self, whole_number):
        # compute_adjusted_k_c writes the layers from its own breakpoints
        # argument while _recovery_run_segments reads them back from the
        # persisted run-context. If those two disagreed on int vs float,
        # the read would look for a layer that exists under a near-miss
        # name and raise FileNotFoundError.
        assert c.recovery_time_suffix(whole_number) \
            == c.recovery_time_suffix(float(whole_number))

    def test_whole_number_floats_normalise_to_the_integer_form(self):
        assert c.recovery_time_suffix(0.0) == 't0'
        assert c.recovery_time_suffix(1.0) == 't1'
        assert c.recovery_time_suffix(2.0) == 't2'

    def test_int_and_float_breakpoint_lists_name_the_same_layers(self):
        as_ints = [0, 1, 2]
        as_floats = [0.0, 1.0, 2.0]

        assert [c.recovery_time_suffix(b) for b in as_ints] \
            == [c.recovery_time_suffix(b) for b in as_floats]

    def test_numpy_scalars_normalise_the_same_way(self):
        # np.arange / np.linspace are natural ways to build breakpoints.
        np = pytest.importorskip('numpy')

        assert c.recovery_time_suffix(np.float64(0.0)) == 't0'
        assert c.recovery_time_suffix(np.float64(0.5)) == 't0_5'
        assert c.recovery_time_suffix(np.int64(2)) == 't2'

    def test_numpy_breakpoints_match_the_defaults(self):
        np = pytest.importorskip('numpy')
        generated = np.arange(0, 3.5, 0.5)

        assert [c.recovery_time_suffix(b) for b in generated] \
            == [c.recovery_time_suffix(b)
                for b in c.DEFAULT_RECOVERY_BREAKPOINTS]


class TestRecoveryWindows:

    def test_n_plus_one_breakpoints_give_n_windows(self):
        assert c.recovery_windows([0, 1, 2, 3]) == [(0, 1), (1, 2), (2, 3)]

    def test_minimum_case_is_a_single_window(self):
        assert c.recovery_windows([0, 1]) == [(0, 1)]

    def test_windows_are_contiguous(self):
        windows = c.recovery_windows(c.DEFAULT_RECOVERY_BREAKPOINTS)

        for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
            assert prev_end == next_start

    def test_window_starts_are_the_modelled_recovery_times(self):
        breakpoints = c.DEFAULT_RECOVERY_BREAKPOINTS
        starts = [start for start, _ in c.recovery_windows(breakpoints)]

        assert starts == breakpoints[:-1]

    def test_accepts_any_iterable(self):
        assert c.recovery_windows((0, 1, 2)) == [(0, 1), (1, 2)]

    def test_does_not_mutate_the_input(self):
        breakpoints = [0, 1, 2]
        c.recovery_windows(breakpoints)
        assert breakpoints == [0, 1, 2]

    @pytest.mark.parametrize('breakpoints', [[], [0]])
    def test_needs_at_least_two_breakpoints(self, breakpoints):
        with pytest.raises(ValueError, match='at least two values'):
            c.recovery_windows(breakpoints)

    def test_rejects_repeated_breakpoints(self):
        # A zero-length window would model a recovery time no rain lands in.
        with pytest.raises(ValueError, match='strictly increasing'):
            c.recovery_windows([0, 1, 1, 2])

    def test_rejects_decreasing_breakpoints(self):
        with pytest.raises(ValueError, match='strictly increasing'):
            c.recovery_windows([0, 2, 1])

    def test_rejects_unsorted_breakpoints(self):
        with pytest.raises(ValueError, match='strictly increasing'):
            c.recovery_windows([1, 0, 2])


class TestBreakpointsFromTimesAndInterval:

    def test_closes_the_final_window(self):
        assert c.breakpoints_from_times_and_interval([0, 0.5, 1], 0.5) \
            == [0, 0.5, 1, 1.5]

    def test_single_time_gives_one_window(self):
        breakpoints = c.breakpoints_from_times_and_interval([0], 0.5)
        assert c.recovery_windows(breakpoints) == [(0, 0.5)]

    def test_round_trips_back_to_the_original_times(self):
        times = [0, 0.5, 1, 1.5]
        breakpoints = c.breakpoints_from_times_and_interval(times, 0.5)
        starts = [start for start, _ in c.recovery_windows(breakpoints)]

        assert starts == times

    def test_reproduces_the_package_defaults(self):
        breakpoints = c.breakpoints_from_times_and_interval(
            c.DEFAULT_RECOVERY_TIMES, c.DEFAULT_RECOVERY_INTERVAL_YEARS)

        assert breakpoints == c.DEFAULT_RECOVERY_BREAKPOINTS

    def test_does_not_mutate_the_input(self):
        times = [0, 0.5]
        c.breakpoints_from_times_and_interval(times, 0.5)
        assert times == [0, 0.5]

    def test_rejects_empty_times(self):
        with pytest.raises(ValueError, match='empty'):
            c.breakpoints_from_times_and_interval([], 0.5)


class TestDefaults:

    def test_default_breakpoints_are_valid(self):
        # Guards against someone editing the constant into a bad state.
        windows = c.recovery_windows(c.DEFAULT_RECOVERY_BREAKPOINTS)
        assert len(windows) == len(c.DEFAULT_RECOVERY_BREAKPOINTS) - 1

    def test_deprecated_times_track_the_breakpoints(self):
        assert c.DEFAULT_RECOVERY_TIMES == c.DEFAULT_RECOVERY_BREAKPOINTS[:-1]
