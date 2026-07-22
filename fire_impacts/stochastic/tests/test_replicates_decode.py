"""
Decoding of the stochastic rainfall API payload.

These run against canned response dicts - no HTTP. Between them,
decode_rle and hg_to_data_frame turn the wire format into the rainfall
every replicate is driven by, so a decode error is a silent change to the
model's input.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts.stochastic.rainfall.replicates import (
    decode_rle,
    hg_to_data_frame,
)


def response(timeseries, start='2019-01-01T00:00:00', step=1800, length=4):
    """Build a minimal API response dict."""
    return {
        'indexes': [{'start': start, 'length': length, 'step': step}],
        'timeseries': timeseries,
    }


class TestDecodeRle:

    def test_bare_scalars_decode_one_for_one(self):
        assert list(decode_rle([1, 2, 3])) == [1, 2, 3]

    def test_pairs_expand_to_their_count(self):
        assert list(decode_rle([[0, 3]])) == [0, 0, 0]

    def test_mixed_scalars_and_pairs(self):
        assert list(decode_rle([1, [0, 3], 2])) == [1, 0, 0, 0, 2]

    def test_long_zero_run(self):
        # The encoding exists because dry periods dominate the series.
        decoded = decode_rle([[0, 1000], 5])
        assert len(decoded) == 1001
        assert decoded[-1] == 5
        assert decoded[:-1].sum() == 0

    def test_count_of_one_is_a_single_value(self):
        assert list(decode_rle([[7, 1]])) == [7]

    def test_count_of_zero_contributes_nothing(self):
        assert list(decode_rle([[7, 0], 1])) == [1]

    def test_empty_input_gives_an_empty_array(self):
        assert len(decode_rle([])) == 0

    def test_returns_a_numpy_array(self):
        assert isinstance(decode_rle([1, 2]), np.ndarray)

    def test_preserves_floats(self):
        decoded = decode_rle([[0.5, 2], 1.25])
        assert np.allclose(decoded, [0.5, 0.5, 1.25])

    def test_a_three_element_list_is_treated_as_a_scalar(self):
        # Only 2-element lists are run-length pairs; anything else is
        # taken as a value in its own right.
        decoded = decode_rle([[1, 2, 3]])
        assert len(decoded) == 1


class TestHgToDataFrame:

    def test_builds_the_index_from_start_length_and_step(self):
        df = hg_to_data_frame(response([{'values': [1, 2, 3, 4]}]))

        assert df.index[0] == pd.Timestamp('2019-01-01 00:00')
        assert len(df.index) == 4
        assert df.index.freq is not None or (df.index[1] - df.index[0]) \
            == pd.Timedelta(minutes=30)

    @pytest.mark.parametrize('step,expected', [
        (1800, pd.Timedelta(minutes=30)),
        (720, pd.Timedelta(minutes=12)),
        (3600, pd.Timedelta(hours=1)),
    ])
    def test_step_is_in_seconds(self, step, expected):
        df = hg_to_data_frame(
            response([{'values': [1, 2, 3, 4]}], step=step))

        assert df.index[1] - df.index[0] == expected

    def test_one_column_per_simulation(self):
        df = hg_to_data_frame(response([
            {'values': [1, 2, 3, 4]},
            {'values': [5, 6, 7, 8]},
            {'values': [9, 10, 11, 12]},
        ]))

        assert list(df.columns) == \
            ['Simulation_0', 'Simulation_1', 'Simulation_2']
        assert df.shape == (4, 3)

    def test_values_land_in_the_right_column(self):
        df = hg_to_data_frame(response([
            {'values': [1, 2, 3, 4]},
            {'values': [5, 6, 7, 8]},
        ]))

        assert list(df['Simulation_0']) == [1, 2, 3, 4]
        assert list(df['Simulation_1']) == [5, 6, 7, 8]

    def test_scale_divides_the_decoded_values(self):
        # The API sends integers with a scale factor to save bandwidth.
        df = hg_to_data_frame(response([
            {'values': [10, 20, 30, 40], 'scale': 10.0},
        ]))

        assert np.allclose(df['Simulation_0'], [1.0, 2.0, 3.0, 4.0])

    def test_missing_scale_defaults_to_one(self):
        df = hg_to_data_frame(response([{'values': [10, 20, 30, 40]}]))

        assert np.allclose(df['Simulation_0'], [10, 20, 30, 40])

    def test_per_simulation_scales_are_independent(self):
        df = hg_to_data_frame(response([
            {'values': [10, 20, 30, 40], 'scale': 10.0},
            {'values': [10, 20, 30, 40], 'scale': 2.0},
        ]))

        assert np.allclose(df['Simulation_0'], [1, 2, 3, 4])
        assert np.allclose(df['Simulation_1'], [5, 10, 15, 20])

    def test_run_length_encoded_values_are_expanded(self):
        df = hg_to_data_frame(response([{'values': [[0, 3], 5]}]))

        assert list(df['Simulation_0']) == [0, 0, 0, 5]

    def test_decoded_length_matches_the_declared_index(self):
        # A mismatch between the RLE payload and the index metadata is
        # the failure mode worth catching early.
        df = hg_to_data_frame(response(
            [{'values': [[1, 48]]}], step=1800, length=48))

        assert len(df) == 48
        assert df.index[-1] == pd.Timestamp('2019-01-01 23:30')
