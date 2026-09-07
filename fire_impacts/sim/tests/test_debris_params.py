"""
Debris-flow erosion parameters reaching the pixel-level calculation.

net_erosion and compute_net_erosion take plain values and a parameter
group rather than a RunContext, so the transport physics stays testable
without a project on disk.
"""

import dataclasses

import numpy as np
import pytest

from fire_impacts import const as c
from fire_impacts.params import DebrisDepthParams, DebrisFlowParams
from fire_impacts.sim.debris import compute_net_erosion, net_erosion


@pytest.fixture
def grids():
    """Four pixels chosen so each zone boundary is straddled.

    1e3 is hillslope (<= 1.3e4); 1e5 and 5e6 are both channelised under
    the defaults but separate once the channel threshold drops below
    5e6; 1e8 is already beyond the default channel threshold. Without a
    pixel between a lowered threshold and the default one, moving that
    threshold changes nothing and the test would pass vacuously.
    """
    return dict(
        flow_acc_area=np.array([[1.0e3, 1.0e5, 5.0e6, 1.0e8]]),
        clay_frac_0_05=np.full((1, 4), 0.20),
        clay_frac_05_15=np.full((1, 4), 0.35),
        gradient_arr=np.full((1, 4), 0.5),
        pixel_area=100.0,
    )


class TestDefaultsPreserveBehaviour:

    def test_the_group_defaults_match_the_package_constants(self):
        debris = DebrisFlowParams()
        assert debris.hillslope_area_m2 == c.HILLSLOPE_AREA
        assert debris.channelised_flow_threshold_m2 == \
            c.CHANNELISED_FLOW_THRESHOLD
        assert debris.sediment_bulk_density == c.SEDIMENT_BULK_DENSITY
        assert debris.rock_bulk_density == c.ROCK_BULK_DENSITY
        assert dataclasses.asdict(debris.hillslope) == c.HILLSLOPE_PARAMETERS
        assert dataclasses.asdict(debris.channel) == c.CHANNEL_PARAMETERS

    def test_passing_the_defaults_explicitly_changes_nothing(self, grids):
        implicit = compute_net_erosion(**grids)
        explicit = compute_net_erosion(**grids, debris=DebrisFlowParams())
        for a, b in zip(implicit, explicit):
            assert np.array_equal(a, b, equal_nan=True)

    def test_net_erosion_densities_default_to_the_package_values(self):
        common = dict(
            threshold_met=np.array([[True]]), ae=4.5e-4, be=0.36,
            ad=1.35e-4, bd=0.36, rock=0.12,
            clay_fraction=np.array([[0.2]]),
            flow_area=np.array([[1e4]]),
            gradient_arr=np.array([[0.5]]), pixel_area=100.0,
        )
        implicit = net_erosion(**common)
        explicit = net_erosion(
            **common,
            sediment_bulk_density=c.SEDIMENT_BULK_DENSITY,
            rock_bulk_density=c.ROCK_BULK_DENSITY,
        )
        for a, b in zip(implicit, explicit):
            assert np.array_equal(a, b)


class TestParametersReachTheCalculation:

    def _total(self, grids, **overrides):
        debris = DebrisFlowParams(**overrides) if overrides else None
        return float(np.nansum(compute_net_erosion(**grids, debris=debris)[0]))

    def test_sediment_bulk_density_moves_the_mass(self, grids):
        assert self._total(grids, sediment_bulk_density=2540.0) != \
            self._total(grids)

    def test_rock_bulk_density_moves_the_mass(self, grids):
        assert self._total(grids, rock_bulk_density=1110.0) != \
            self._total(grids)

    def test_the_hillslope_threshold_moves_the_zone_boundary(self, grids):
        # Raising it puts the middle pixel in the hillslope regime rather
        # than the channel one, which use different coefficients.
        assert self._total(grids, hillslope_area_m2=1.0e6) != \
            self._total(grids)

    @pytest.mark.parametrize('hillslope_area_m2', [1.3e4, 1.0e5, 1.0e6])
    def test_the_two_zones_tile_without_a_gap(self, grids,
                                              hillslope_area_m2):
        """The hillslope threshold is the upper bound of one regime and
        the lower bound of the other, so it must be the same value in
        both. Asserting only that a total 'differs' cannot catch the two
        drifting apart — that opens a band of pixels belonging to neither
        regime, which silently contribute nothing.
        """
        debris = DebrisFlowParams(hillslope_area_m2=hillslope_area_m2)
        total, _, _ = compute_net_erosion(**grids, debris=debris)
        area = grids['flow_acc_area']
        # Every pixel inside the modelled range must be in exactly one
        # regime, so none of them can come out at zero.
        inside = area <= debris.channelised_flow_threshold_m2
        assert inside.any()
        assert np.all(total[inside] != 0), (
            f'pixels {area[inside & (total == 0)]} m2 fall in neither '
            f'regime'
        )

    def test_the_channel_threshold_excludes_large_areas(self, grids):
        # Dropping it below the largest pixel takes that pixel out of the
        # channelised zone entirely.
        assert self._total(grids, channelised_flow_threshold_m2=1.0e6) != \
            self._total(grids)

    def test_the_hillslope_depth_coefficients_reach_the_result(self, grids):
        steeper = DebrisDepthParams(
            ae=9.0e-4, be=0.36, ad=1.35e-4, bd=0.36, rock=0.12)
        assert self._total(grids, hillslope=steeper) != self._total(grids)

    def test_the_channel_depth_coefficients_reach_the_result(self, grids):
        steeper = DebrisDepthParams(
            ae=8.2e-4, be=0.52, ad=3.7e-7, bd=1.06, rock=0.45)
        assert self._total(grids, channel=steeper) != self._total(grids)

    def test_the_rock_fraction_splits_sediment_from_rock(self, grids):
        all_rock = DebrisDepthParams(
            ae=4.5e-4, be=0.36, ad=1.35e-4, bd=0.36, rock=1.0)
        _, clay, sediment = compute_net_erosion(
            **grids, debris=DebrisFlowParams(hillslope=all_rock))
        # With no sediment fraction there is no clay in the hillslope zone.
        assert np.nansum(sediment[:, :1]) == 0
        assert np.nansum(clay[:, :1]) == 0


class TestSimulationHorizon:

    def test_the_default_horizon_matches_the_lookup(self):
        assert DebrisFlowParams().num_sim_years == c.NUM_SIM_YEARS == 2

    def test_a_zero_horizon_is_rejected(self):
        with pytest.raises(ValueError, match='num_sim_years'):
            DebrisFlowParams(num_sim_years=0)

    def test_a_longer_horizon_is_allowed_by_the_parameter(self):
        # Not rejected here: the bound comes from the lookup table, which
        # is itself now selectable, so a table covering more years would
        # make this legitimate. debris_flow raises if the table cannot
        # supply the thresholds.
        assert DebrisFlowParams(num_sim_years=3).num_sim_years == 3


class TestYearWindows:
    """Debris year windows are measured from the fire end date."""

    def test_the_year_length_constant_is_365(self):
        # Deliberately not 365.25 — see issues/debris-flow-year-length.md.
        assert c.DAYS_PER_SIM_YEAR == 365

    def test_the_windows_key_off_the_fire_end_not_the_rainfall(self):
        """Reading the source, because running debris_flow end to end
        needs a full prepared catchment. The distinction matters whenever
        the rainfall series does not begin exactly at the fire end."""
        import inspect
        from fire_impacts.sim.debris import debris_flow
        src = inspect.getsource(debris_flow)
        assert 't0 = pd.Timestamp(ctx.fire_end_date)' in src
        assert 't0 = rainfall.index[0]' not in src

    def test_the_representative_year_assumption_is_recorded(self):
        # The lookup tabulates one time per year (0.434, 1.434); those are
        # applied across the whole year rather than interpolated. Pinned
        # so the assumption is not quietly changed.
        from fire_impacts.sim.debris import calc_I12_crit_columns
        assert 'representative of their whole year' in \
            calc_I12_crit_columns.__doc__

    def test_the_packaged_lookup_has_one_bin_per_simulated_year(self):
        from fire_impacts.util import load_package_data
        lookup = load_package_data(c.DEFAULT_I12_LOOKUP)
        assert len(lookup[c.HF_YEARS_THRESH].unique()) == \
            DebrisFlowParams().num_sim_years


class TestReplicateDispatchSignatures:
    """The replicate path has no end-to-end test (it needs a prepared
    catchment and a dask scheduler), so its plumbing is checked by
    signature. A NameError here would only surface in production."""

    @pytest.mark.parametrize('fn_name', [
        'prep_debris_flow_simulation',
        '_prepare_debris_flow_per_catchment',
        'debris_flow',
        'run_debris_flow_all_replicates',
    ])
    def test_every_entry_point_accepts_params(self, fn_name):
        import inspect
        from fire_impacts.sim import debris
        fn = getattr(debris, fn_name)
        assert 'params' in inspect.signature(fn).parameters, fn_name

    @pytest.mark.parametrize('fn_name', [
        'prep_debris_flow_simulation',
        '_prepare_debris_flow_per_catchment',
        'debris_flow',
        'run_debris_flow_all_replicates',
    ])
    def test_the_deprecated_kwarg_uses_the_sentinel(self, fn_name):
        import inspect
        from fire_impacts.sim import debris
        sig = inspect.signature(getattr(debris, fn_name))
        assert sig.parameters['dnbr_threshold'].default is c.UNSET, fn_name
