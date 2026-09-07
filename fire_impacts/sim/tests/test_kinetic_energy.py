"""
Unit kinetic energy: e_r = 0.29 * [1 - 0.72 * exp(-k * i_r)].

The three coefficients have different standing, which is the point of
these tests: 0.29 and 0.72 fix the ends of the curve and belong to its
form, while k selects a published version of the model and is the only
one exposed as a parameter.

  0.05   Brown & Foster (1987), used in RUSLE (Renard et al. 1997)
  0.082  McGregor et al. (1995), adopted by USDA-ARS (2013) in RUSLE2

Reference: Yin, Nearing, Borrelli & Xue (2017), "Rainfall Erosivity: An
Overview of Methodologies and Applications", Vadose Zone Journal,
equations [2] and [3].
"""

import numpy as np
import pytest

from fire_impacts import const as c
from fire_impacts.params import ErosionParams
from fire_impacts.sim import rusle as simr
from fire_impacts.sim.rusle import rainfall_erosivity, unit_kinetic_energy


def unit_energy(intensity, k):
    """Alias for the module under test — these assertions are about the
    real implementation, not a reimplementation of it."""
    return unit_kinetic_energy(intensity, rate=k)


class TestPublishedCoefficients:

    def test_the_two_published_rate_constants(self):
        assert c.DEFAULT_KE_RATE_RUSLE == 0.05      # Brown & Foster (1987)
        assert c.DEFAULT_KE_RATE_RUSLE2 == 0.082    # McGregor et al. (1995)

    def test_the_default_is_the_rusle2_value(self):
        assert ErosionParams().kinetic_energy_coefficient == \
            c.DEFAULT_KE_RATE_RUSLE2
        assert simr.EMPIRICAL_COEFFICIENT == c.DEFAULT_KE_RATE_RUSLE2

    def test_0082_is_not_a_transcription_of_the_drizzle_floor(self):
        # 0.29 * (1 - 0.72) = 0.0812, which is close enough to 0.082 to
        # look like a coefficient dropped into the wrong slot. It is not:
        # 0.082 is the published RUSLE2 rate constant. The near-collision
        # is a coincidence, and this test exists so nobody "corrects" it.
        drizzle_floor = c.KE_ASYMPTOTE * (1 - c.KE_FLOOR_FRACTION)
        assert drizzle_floor == pytest.approx(0.0812, abs=1e-4)
        assert c.DEFAULT_KE_RATE_RUSLE2 != pytest.approx(drizzle_floor,
                                                         abs=1e-5)


class TestCurveShape:
    """0.29 and 0.72 fix the ends; k only changes the approach."""

    def test_the_asymptote_is_independent_of_the_rate_constant(self):
        for k in (c.DEFAULT_KE_RATE_RUSLE, c.DEFAULT_KE_RATE_RUSLE2):
            assert unit_energy(1e6, k) == pytest.approx(c.KE_ASYMPTOTE)

    def test_the_drizzle_floor_is_independent_of_the_rate_constant(self):
        # exp(0) == 1 whatever k is, so both versions start together.
        floor = c.KE_ASYMPTOTE * (1 - c.KE_FLOOR_FRACTION)
        for k in (c.DEFAULT_KE_RATE_RUSLE, c.DEFAULT_KE_RATE_RUSLE2):
            assert unit_energy(0.0, k) == pytest.approx(floor)

    def test_energy_increases_monotonically_with_intensity(self):
        intensities = np.linspace(0, 200, 400)
        energy = unit_energy(intensities, c.DEFAULT_KE_RATE_RUSLE2)
        assert np.all(np.diff(energy) > 0)

    def test_energy_never_exceeds_the_asymptote(self):
        # Approached from below, and reached exactly once the exponential
        # underflows the float64 mantissa (around i = 450 mm/h) — far
        # above any real rainfall, so equality there is fine.
        intensities = np.linspace(0, 1000, 1000)
        assert unit_energy(
            intensities, c.DEFAULT_KE_RATE_RUSLE2).max() <= c.KE_ASYMPTOTE
        assert unit_energy(200.0, c.DEFAULT_KE_RATE_RUSLE2) < c.KE_ASYMPTOTE


class TestVersionDifference:
    """RUSLE2 gives more unit energy than RUSLE at every finite intensity,
    which is the direction Australian evidence supports (Yu 1999)."""

    def test_rusle2_is_always_the_higher_of_the_two(self):
        intensities = np.linspace(0.1, 200, 500)
        rusle = unit_energy(intensities, c.DEFAULT_KE_RATE_RUSLE)
        rusle2 = unit_energy(intensities, c.DEFAULT_KE_RATE_RUSLE2)
        assert np.all(rusle2 > rusle)

    def test_the_gap_peaks_in_the_erosive_mid_range(self):
        # The two versions share both ends, so they can only differ in the
        # middle — and the peak sits where most erosive rainfall lands,
        # not out at extremes where it would hardly matter.
        intensities = np.linspace(0.1, 200, 2000)
        ratio = (unit_energy(intensities, c.DEFAULT_KE_RATE_RUSLE2)
                 / unit_energy(intensities, c.DEFAULT_KE_RATE_RUSLE))
        peak_at = intensities[np.argmax(ratio)]
        assert 5 < peak_at < 20
        assert ratio.max() == pytest.approx(1.21, abs=0.02)


class TestErosivityHelper:
    """rainfall_erosivity bundles the depth -> intensity -> energy -> R
    chain that was previously written out twice in sim/rusle.py."""

    def test_intensity_is_depth_over_the_timestep(self):
        # 30-minute timestep, so 5 mm in a step is 10 mm/h.
        intensity, _ = rainfall_erosivity(5.0)
        assert intensity == pytest.approx(10.0)

    def test_erosivity_is_energy_times_intensity(self):
        depth = 5.0
        intensity, erosivity = rainfall_erosivity(depth)
        expected = unit_kinetic_energy(intensity) * depth * intensity
        assert erosivity == expected

    def test_zero_depth_gives_zero_erosivity(self):
        intensity, erosivity = rainfall_erosivity(0.0)
        assert (intensity, erosivity) == (0.0, 0.0)

    def test_the_timestep_is_derived_not_hard_coded(self):
        from fire_impacts.sim.rusle import (
            _MODEL_TIMESTEP, _MODEL_TIMESTEP_HOURS)
        assert _MODEL_TIMESTEP_HOURS == \
            _MODEL_TIMESTEP.total_seconds() / 3600.0

    def test_a_different_timestep_changes_the_intensity(self):
        # 12-minute rainfall (the debris-flow resolution) is 0.2 h.
        intensity, _ = rainfall_erosivity(5.0, timestep_hours=0.2)
        assert intensity == pytest.approx(25.0)

    def test_the_rate_constant_can_be_switched_to_rusle(self):
        _, rusle2 = rainfall_erosivity(5.0)
        _, rusle = rainfall_erosivity(5.0, rate=c.DEFAULT_KE_RATE_RUSLE)
        assert rusle < rusle2
