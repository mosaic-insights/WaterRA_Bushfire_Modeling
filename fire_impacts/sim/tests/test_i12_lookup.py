"""
The I12 critical-intensity lookup: slope binning and table selection.

The lookup maps (aridity, dNBR, years since fire, slope) onto a critical
12-minute rainfall intensity — it is the debris-flow triggering model, so
how a headwater is placed on its grid decides whether anything triggers
at all.
"""

import numpy as np
import pandas as pd
import pytest

from fire_impacts import const as c
from fire_impacts.params import DebrisFlowParams
from fire_impacts.sim.debris import _clip_to_lookup_bins
from fire_impacts.util import load_package_data


@pytest.fixture(scope='module')
def lookup():
    return load_package_data(c.DEFAULT_I12_LOOKUP)


class TestSlopeIsAGradient:
    """The lookup's slope column holds dimensionless gradients, so degrees
    convert with tan — not by dividing by 100, which is what the code did
    and which put every headwater in a flatter bin than it belonged to."""

    @pytest.mark.parametrize('degrees,expected_bin', [
        (5, 0.1), (10, 0.2), (15, 0.3), (20, 0.4),
        (26, 0.5), (30, 0.6), (35, 0.7), (40, 0.8), (45, 1.0),
    ])
    def test_degrees_map_to_the_gradient_bin(self, degrees, expected_bin,
                                             lookup):
        gradient = pd.Series([np.tan(np.radians(degrees))])
        got = _clip_to_lookup_bins(
            gradient, lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert float(got.iloc[0]) == pytest.approx(expected_bin)

    def test_the_old_conversion_gave_a_flatter_bin(self):
        # Pinned as a regression: /100 and tan diverge across the whole
        # usable range, so this is not a rounding nicety.
        for degrees in (15, 20, 26, 30, 35, 40, 45):
            old = round(degrees / 100, 1)
            new = round(float(np.tan(np.radians(degrees))), 1)
            assert new > old, degrees

    def test_a_flatter_bin_means_a_higher_critical_intensity(self, lookup):
        # Which is why the bug under-triggered: too-flat a bin demands more
        # rain before a debris flow is counted.
        year1 = lookup[lookup[c.HF_YEARS_THRESH] < 1]
        sub = year1[(year1[c.HF_ARID_IDX_THRESH] == 1.0)
                    & (year1[c.HF_DNBR_THRESH] == 500)]
        sub = sub.sort_values(c.HF_GRADIENT_THRESH)
        intensities = sub[c.HF_I12_CRIT].tolist()
        assert intensities == sorted(intensities, reverse=True)


class TestClippingToBins:
    """The join is on exact bin values, so an out-of-range value matches
    nothing, leaves I12_crit NaN, and the headwater is skipped entirely."""

    def test_gradients_above_45_degrees_exceed_the_table(self, lookup):
        # tan(50 degrees) is 1.19, past the top bin of 1.0. This is the
        # edge the gradient fix introduces: /100 could never exceed 0.9.
        assert np.tan(np.radians(50)) > lookup[c.HF_GRADIENT_THRESH].max()

    def test_steep_slopes_clip_to_the_top_bin(self, lookup):
        got = _clip_to_lookup_bins(
            pd.Series([np.tan(np.radians(d)) for d in (50, 60, 70)]),
            lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert list(got) == [1.0, 1.0, 1.0]

    def test_shallow_slopes_clip_to_the_bottom_bin(self, lookup):
        got = _clip_to_lookup_bins(
            pd.Series([0.0, 0.01]),
            lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert list(got) == [0.1, 0.1]

    def test_clipped_values_all_match_a_real_bin(self, lookup):
        bins = set(lookup[c.HF_GRADIENT_THRESH].round(1))
        got = _clip_to_lookup_bins(
            pd.Series([np.tan(np.radians(d)) for d in range(1, 80)]),
            lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert set(got.round(1)) <= bins

    def test_out_of_range_values_are_reported(self, lookup, caplog):
        with caplog.at_level('WARNING'):
            _clip_to_lookup_bins(
                pd.Series([np.tan(np.radians(60))]),
                lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert 'outside the lookup range' in caplog.text
        assert 'dropped from the debris-flow results' in caplog.text

    def test_in_range_values_are_silent(self, lookup, caplog):
        with caplog.at_level('WARNING'):
            _clip_to_lookup_bins(
                pd.Series([0.5]),
                lookup[c.HF_GRADIENT_THRESH], 'slope gradient')
        assert 'outside the lookup range' not in caplog.text


class TestCalcI12CritColumns:
    """End-to-end through the join, which is what actually decides the
    threshold each headwater is given."""

    @staticmethod
    def _headwaters(slope_degrees):
        return pd.DataFrame({
            c.HW_ID: [1],
            c.ARID_MEAN: [1.0],
            c.DNBR_MEAN: [500.0],
            c.SLOPE_DEG_MEAN: [float(slope_degrees)],
        })

    def _i12(self, lookup, slope_degrees):
        from fire_impacts.sim.debris import calc_I12_crit_columns
        out = calc_I12_crit_columns(
            self._headwaters(slope_degrees), lookup, c.HW_ID)
        return float(out[c.I12_CRIT_Y + '1'].iloc[0])

    def _expected(self, lookup, gradient_bin):
        year1 = lookup[lookup[c.HF_YEARS_THRESH] < 1]
        row = year1[(year1[c.HF_ARID_IDX_THRESH] == 1.0)
                    & (year1[c.HF_DNBR_THRESH] == 500)
                    & (year1[c.HF_GRADIENT_THRESH] == gradient_bin)]
        return float(row[c.HF_I12_CRIT].iloc[0])

    @pytest.mark.parametrize('degrees,gradient_bin', [
        (15, 0.3), (26, 0.5), (35, 0.7), (40, 0.8),
    ])
    def test_the_threshold_comes_from_the_gradient_bin(
            self, lookup, degrees, gradient_bin):
        assert self._i12(lookup, degrees) == pytest.approx(
            self._expected(lookup, gradient_bin), abs=0.05)

    @pytest.mark.parametrize('degrees,wrong_bin', [
        (26, 0.3), (35, 0.3), (40, 0.4),
    ])
    def test_the_threshold_is_not_the_old_degrees_over_100_bin(
            self, lookup, degrees, wrong_bin):
        # The regression itself: dividing degrees by 100 selected a
        # flatter bin and therefore a higher critical intensity, so
        # debris flows were under-triggered by 1.2-1.4x.
        wrong = self._expected(lookup, wrong_bin)
        assert self._i12(lookup, degrees) != pytest.approx(wrong, abs=0.05)
        assert self._i12(lookup, degrees) < wrong

    def test_a_very_steep_headwater_still_gets_a_threshold(self, lookup):
        # Clipped to the top bin rather than dropped for want of a match.
        assert not np.isnan(self._i12(lookup, 60))
        assert self._i12(lookup, 60) == pytest.approx(
            self._expected(lookup, 1.0), abs=0.05)


class TestTableSelection:
    """The table is overridable, though no tooling is provided to build
    an alternative and the packaged default is expected to stand."""

    def test_the_default_names_the_packaged_table(self):
        assert DebrisFlowParams().i12_lookup == c.DEFAULT_I12_LOOKUP

    def test_a_bare_filename_resolves_against_the_package(self):
        assert len(load_package_data(c.DEFAULT_I12_LOOKUP)) > 0

    def test_a_path_is_used_as_given(self, tmp_path, lookup):
        alternate = tmp_path / 'my_lookup.csv'
        trimmed = lookup.head(10)
        trimmed.to_csv(alternate, index=False)
        got = load_package_data(str(alternate))
        assert len(got) == 10

    def test_a_missing_path_fails_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='Lookup table not found'):
            load_package_data(str(tmp_path / 'nope.csv'))

    def test_an_empty_table_name_is_rejected(self):
        with pytest.raises(ValueError, match='i12_lookup'):
            DebrisFlowParams(i12_lookup='')

    def test_the_table_name_survives_a_round_trip(self):
        from fire_impacts.params import ModelParameters
        params = ModelParameters().replace(
            debris__i12_lookup='/data/custom.csv')
        assert ModelParameters.from_dict(
            params.to_dict()).debris.i12_lookup == '/data/custom.csv'
