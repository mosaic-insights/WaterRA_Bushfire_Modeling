"""
What generate_rusle yields for a timestep with no rainfall.

Around three quarters of 30-minute timesteps in a typical replicate are
dry and can produce no erosion. They still have to be yielded, so that
recorders keep their cadence, but nothing about them is worth computing
twice: the grids are all zero, they are all the same zeros, and no
recorder has any business writing into them.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts.sim.rusle import generate_rusle, rainfall_erosivity


SHAPE = (2, 3)
CELL_AREA_HA = 0.09

GRID_KEYS = (
    'RUSLE',
    'delivered',
    'RUSLE_below_threshold',
    'RUSLE_above_threshold',
    'delivered_below_threshold',
    'delivered_above_threshold',
)


@pytest.fixture()
def grids():
    klscp = np.full(SHAPE, 2.0, dtype=np.float32)
    sdr = np.full(SHAPE, 0.5, dtype=np.float32)
    # Half the cells below the default severity threshold, half above.
    dnbr = np.array([[0.0, 0.0, 0.0], [900.0, 900.0, 900.0]],
                    dtype=np.float32)
    return klscp, sdr, dnbr


def run(rain_mm, grids):
    """Feed a list of per-timestep depths through generate_rusle."""
    rain = pd.Series(
        rain_mm,
        index=pd.date_range('2020-01-01', periods=len(rain_mm), freq='30min'),
    )
    return list(generate_rusle(rain, *grids, CELL_AREA_HA))


class TestDryFlag:

    def test_timesteps_are_flagged_dry_when_there_is_no_rain(self, grids):
        steps = run([0.0, 5.0, 0.0], grids)
        assert [data['dry'] for _, data in steps] == [True, False, True]


class TestDryGridBuffer:

    def test_every_dry_grid_is_the_same_buffer(self, grids):
        _, data = run([0.0], grids)[0]
        first = data['RUSLE']
        assert all(data[key] is first for key in GRID_KEYS)

    def test_consecutive_dry_timesteps_share_one_buffer(self, grids):
        steps = run([0.0, 5.0, 0.0], grids)
        assert steps[0][1]['RUSLE'] is steps[2][1]['RUSLE']

    def test_the_dry_buffer_is_read_only(self, grids):
        # It is shared, so a recorder that accumulates into it in place
        # would corrupt every other timestep. Make that raise rather
        # than silently produce wrong numbers.
        _, data = run([0.0], grids)[0]
        with pytest.raises(ValueError):
            data['RUSLE'] += 1.0

    def test_wet_timesteps_get_their_own_writable_grids(self, grids):
        steps = run([5.0, 5.0], grids)
        first, second = steps[0][1]['RUSLE'], steps[1][1]['RUSLE']

        assert first is not second
        assert first.flags.writeable


class TestDryValues:

    def test_a_dry_timestep_erodes_nothing(self, grids):
        _, data = run([0.0], grids)[0]
        for key in GRID_KEYS:
            assert np.array_equal(data[key], np.zeros(SHAPE, dtype=np.float32))

    def test_a_dry_timestep_reports_no_intensity_or_erosivity(self, grids):
        _, data = run([0.0], grids)[0]
        assert data['total_rain'] == 0.0
        assert data['intensity'] == 0.0
        assert data['erosivity'] == 0.0


class TestWetValuesAreUnchanged:

    def test_erosion_is_erosivity_by_klscp_by_cell_area(self, grids):
        klscp, sdr, _ = grids
        _, data = run([5.0], grids)[0]
        intensity, R = rainfall_erosivity(5.0)

        assert data['intensity'] == intensity
        assert data['erosivity'] == R
        assert np.allclose(data['RUSLE'], R * klscp * CELL_AREA_HA)
        assert np.allclose(data['delivered'], data['RUSLE'] * sdr)

    def test_severity_thresholds_split_the_grid(self, grids):
        _, data = run([5.0], grids)[0]
        # dnbr row 0 is below the threshold, row 1 above.
        assert np.allclose(data['RUSLE_below_threshold'][1], 0.0)
        assert np.allclose(data['RUSLE_above_threshold'][0], 0.0)
        assert np.allclose(
            data['RUSLE_below_threshold'] + data['RUSLE_above_threshold'],
            data['RUSLE'],
        )
