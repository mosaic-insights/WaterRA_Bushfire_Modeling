"""
dNBR scale convention: stored as the raw band-ratio difference, thresholded
on the conventional 0-1000 scale.

The bug these pin: two producers wrote masked_dNBR.tif on *different*
scales (the real severity path the stored one, generate_synthetic_fire the
conventional one), and consumers disagreed too — pre/rusle.py and
summary_stats multiplied by 1000, sim/rusle.py did not. So both paths were
wrong, in opposite directions, and no consumer could tell which convention
it had been handed.

Design notes: design-notes/calibration-parameters-proposal.md §3.5
"""

import numpy as np
import pytest

from fire_impacts import const as c
from fire_impacts.params import ErosionParams, FireAdjustmentParams
from fire_impacts.pre.util import (
    from_dnbr_scale, to_dnbr_scale,
)


class TestScaleConstant:

    def test_the_factor_is_1000(self):
        assert c.DNBR_SCALE == 1000

    def test_conversions_round_trip(self):
        values = np.array([0.0, 0.25, 0.65, 1.2])
        assert np.allclose(from_dnbr_scale(to_dnbr_scale(values)), values)

    def test_to_scale_lands_in_the_conventional_range(self):
        # A typical stored high-severity value is ~0.66; on the
        # conventional scale that is ~660, above the 400 split.
        assert to_dnbr_scale(0.66) == pytest.approx(660.0)

    def test_from_scale_lands_in_the_stored_range(self):
        assert from_dnbr_scale(660.0) == pytest.approx(0.66)


class TestThresholdsShareTheScale:
    """Every dNBR threshold is quoted on the conventional scale, and the
    parameter defaults must agree with the constants they came from."""

    def test_severity_threshold(self):
        assert ErosionParams().dnbr_severity_threshold == \
            c.DEFAULT_DNBR_SEVERITY_THRESHOLD

    def test_saturation(self):
        assert FireAdjustmentParams().dnbr_saturation == \
            c.DEFAULT_DNBR_SATURATION

    def test_thresholds_are_on_the_conventional_scale_not_the_stored_one(self):
        # The failure mode being guarded: a threshold silently drifting to
        # the stored scale (0.4 instead of 400) is not a type error and not
        # a range error — it just makes the branch unreachable.
        for threshold in (
            c.DEFAULT_DNBR_SEVERITY_THRESHOLD,
            c.DEFAULT_DNBR_SATURATION,
            c.DEFAULT_DEBRIS_DNBR_THRESHOLD,
        ):
            assert threshold > 1, (
                f'{threshold} looks like a stored-scale value; dNBR '
                f'thresholds are on the 0-{c.DNBR_SCALE} scale'
            )
            assert threshold <= c.DNBR_SCALE

    def test_a_stored_scale_dnbr_never_reaches_the_severity_threshold(self):
        # Restates the original defect as an assertion: raw stored values
        # are ~[0, 1], so comparing them directly against the threshold can
        # never be true. Anything reading dNBR must scale first.
        stored_maximum = 1.2  # generous; real dNBR rarely exceeds ~1.1
        assert stored_maximum < c.DEFAULT_DNBR_SEVERITY_THRESHOLD
        assert to_dnbr_scale(stored_maximum) > \
            c.DEFAULT_DNBR_SEVERITY_THRESHOLD


class TestReadersApplyTheScale:

    def test_aligned_like_reader_scales(self, tmp_path, monkeypatch):
        from fire_impacts.pre import util
        sentinel = np.array([[0.4, 0.66]])
        monkeypatch.setattr(util, 'read_aligned_like',
                            lambda *a, **k: sentinel)
        got = util.read_dnbr_aligned_like('ignored', None)
        assert np.allclose(got, [[400.0, 660.0]])

    def test_aligned_reader_scales(self, tmp_path, monkeypatch):
        from fire_impacts.pre import util
        sentinel = np.array([[0.4, 0.66]])
        monkeypatch.setattr(util, 'read_aligned', lambda *a, **k: sentinel)
        got = util.read_dnbr_aligned('ignored', None, None, None)
        assert np.allclose(got, [[400.0, 660.0]])


class TestProducersAgree:
    """Both producers of masked_dNBR.tif must write the stored convention."""

    def test_synthetic_fire_converts_reference_values(self):
        # The reference rasters are published on the conventional scale
        # (measured max ~1143 and ~1477), so the synthetic path must divide
        # before writing or it is 1000x out from the real path.
        import inspect
        from fire_impacts.pre import synthetic_fire
        src = inspect.getsource(synthetic_fire.generate_synthetic_fire)
        assert 'from_dnbr_scale' in src

    def test_format_dnbr_uses_the_shared_factor(self):
        from fire_impacts.pre.project import format_dNBR
        import pandas as pd
        got = format_dNBR(pd.Series([0.0, 0.4, -0.2, 0.66]))
        assert list(got) == [0.0, 400.0, 0.0, pytest.approx(660.0)]
