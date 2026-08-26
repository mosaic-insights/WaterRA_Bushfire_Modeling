"""
ModelParameters: the calibration parameter set and its resolution.

Covers the four things phase 1 has to get right — construction-time
validation, strict unknown-key rejection, layered merge precedence with
provenance, and a stable digest — plus the JSON round trip that persistence
depends on.

Design notes: design-notes/calibration-parameters-proposal.md
"""

import dataclasses
import json
from typing import Union

from fire_impacts import const as c

import pytest

from fire_impacts.params import (
    DebrisDepthParams,
    sparse_overrides,
    DebrisFlowParams,
    DeliveryParams,
    ErosionParams,
    FireAdjustmentParams,
    ModelParameters,
    ParameterRecord,
    SCOPES,
    SeverityParams,
    TopographyParams,
    check_scope,
    resolve_parameters,
    scope_of,
)


# Every value phase 2 lifted out of a hard-coded literal in pre/rusle.py is
# here. Pinning the whole vector — rather than a hand-picked few — is what
# stops a default drifting silently: each of these changes every raster the
# preprocessing pipeline produces, and without this snapshot the suite stays
# green when they move. Changing one must be a deliberate edit to this
# literal, reviewed alongside the change.
EXPECTED_DEFAULTS = {
    "fire_adjustment": {
        "c_peak": 0.35,
        "k_fire": 0.081,
        "c_recovery_scale": 0.4,
        "k_recovery_scale": 1.0,
        "dnbr_saturation": 400.0,
        "default_c_factor": 0.01
    },
    "delivery": {
        "max_sdr": 0.8,
        "ic0": 0.5,
        "k": 1.0,
        "stream_area_threshold_m2": 13000.0,
        "min_slope": 0.005,
        "max_slope": 1.0,
        "min_c_factor": 0.001
    },
    "topography": {
        "headwater_threshold_m2": 20000.0,
        "max_slope_length_m": 141.0
    },
    "erosion": {
        "support_practice_factor": 1.0,
        "dnbr_severity_threshold": 400.0,
        "kinetic_energy_coefficient": 0.082
    },
    "debris": {
        "hillslope": {
            "ae": 0.00045,
            "be": 0.36,
            "ad": 0.000135,
            "bd": 0.36,
            "rock": 0.12
        },
        "channel": {
            "ae": 0.00041,
            "be": 0.52,
            "ad": 3.7e-07,
            "bd": 1.06,
            "rock": 0.45
        },
        "hillslope_area_m2": 13000.0,
        "channelised_flow_threshold_m2": 14000000.0,
        "sediment_bulk_density": 1270.0,
        "rock_bulk_density": 2220.0,
        "dnbr_threshold": 100.0,
        "num_sim_years": 2,
        "i12_lookup": "HFlookup_b30pt27.csv"
    },
    "severity": {
        "max_cloud_cover": 20.0,
        "resolution_m": 20.0,
        "pre_fire_window_days": 90,
        "post_fire_window_days": 90,
        "bbox_buffer_km": 10.0,
        "force_sensor": None,
        "natural_veg_code": 112
    }
}

# The digest of EXPECTED_DEFAULTS. Pinned separately because it also proves
# the digest is stable across processes, platforms and Python versions —
# comparing two instances in one process cannot show that.
EXPECTED_DEFAULT_DIGEST = 'sha256:33ffb05d11382e2f0f7a238076bc65e4c78c36af762aa1d7721d700152f4e852'


class TestDefaults:

    def test_the_whole_default_vector_is_pinned(self):
        assert ModelParameters().to_dict() == EXPECTED_DEFAULTS

    def test_the_default_digest_is_pinned(self):
        # Also pins cross-process/platform digest stability, which
        # comparing two in-process instances cannot show.
        assert ModelParameters().digest() == EXPECTED_DEFAULT_DIGEST

    def test_the_pinned_values_are_the_pre_phase2_hard_coded_literals(self):
        # Spelled out so the mapping from the old locals is greppable.
        fa = ModelParameters().fire_adjustment
        assert fa.c_recovery_scale == 0.4      # was x_c
        assert fa.k_recovery_scale == 1.0      # was x_k
        assert fa.k_fire == 0.081              # was Kfire
        assert fa.c_peak == 0.35               # was Cpeak
        assert fa.dnbr_saturation == 400.0     # was the inline 400
        assert fa.default_c_factor == 0.01     # was the inline 0.01
        d = ModelParameters().delivery
        assert (d.max_sdr, d.ic0, d.k) == (0.8, 0.5, 1.0)
        assert d.min_slope == 0.005            # was the inline slope clamp
        assert d.max_slope == 1.0
        assert d.min_c_factor == 0.001         # was the inline C floor
        assert d.stream_area_threshold_m2 == 1.3e4
        assert ModelParameters().topography.max_slope_length_m == 141.0
        assert ModelParameters().topography.headwater_threshold_m2 == \
            float(c.DEFAULT_HW_THRESHOLD)

    def test_groups_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            ModelParameters().delivery.max_sdr = 0.9

    def test_nested_debris_regimes_carry_the_const_values(self):
        debris = ModelParameters().debris
        assert debris.hillslope.ae == 4.5e-4
        assert debris.hillslope.ad == pytest.approx(0.3 * 4.5e-4)
        assert debris.channel.bd == 1.06


class TestValidation:
    """__post_init__ rejects values that would fail silently downstream."""

    def test_zero_k_recovery_scale_is_rejected(self):
        # The motivating case: x_k = 0 does not raise or produce inf, it
        # silently reverts K to the unburnt baseline for t > 0 and gives
        # NaN at t = 0.
        with pytest.raises(ValueError, match='k_recovery_scale must be > 0'):
            FireAdjustmentParams(k_recovery_scale=0)

    def test_negative_c_recovery_scale_is_rejected(self):
        with pytest.raises(ValueError, match='c_recovery_scale must be > 0'):
            FireAdjustmentParams(c_recovery_scale=-0.4)

    @pytest.mark.parametrize('value', [0, -0.1, 1.5])
    def test_max_sdr_must_be_a_ratio(self, value):
        with pytest.raises(ValueError, match='max_sdr'):
            DeliveryParams(max_sdr=value)

    def test_zero_k_is_rejected_as_a_divisor(self):
        with pytest.raises(ValueError, match='k must be non-zero'):
            DeliveryParams(k=0)

    def test_slope_clamp_must_be_ordered(self):
        with pytest.raises(ValueError, match='min_slope .* must be <'):
            DeliveryParams(min_slope=2.0, max_slope=1.0)

    def test_debris_zone_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match='must be <'):
            DebrisFlowParams(
                hillslope_area_m2=1e8,
                channelised_flow_threshold_m2=1.4e7,
            )

    def test_rock_fraction_must_be_a_fraction(self):
        with pytest.raises(ValueError, match='rock must be a fraction'):
            DebrisDepthParams(ae=1e-4, be=0.3, ad=1e-5, bd=0.3, rock=1.4)

    def test_support_practice_factor_must_be_a_ratio(self):
        with pytest.raises(ValueError, match='support_practice_factor'):
            ErosionParams(support_practice_factor=1.5)

    def test_force_sensor_is_constrained(self):
        with pytest.raises(ValueError, match='force_sensor'):
            SeverityParams(force_sensor='modis')

    def test_validation_runs_when_built_from_a_dict(self):
        with pytest.raises(ValueError, match='max_sdr'):
            ModelParameters.from_dict({'delivery': {'max_sdr': 2.0}})

    def test_validation_runs_on_replace(self):
        # replace() rebuilds through from_dict, so every __post_init__
        # re-runs. What matters is that it never uses object.__setattr__,
        # which would bypass validation on a frozen dataclass.
        with pytest.raises(ValueError, match='max_sdr'):
            ModelParameters().replace(delivery__max_sdr=2.0)


class TestUnknownKeys:
    """A typo must raise, not silently do nothing."""

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown parameter 'delivery.mx_sdr'"):
            ModelParameters.from_dict({'delivery': {'mx_sdr': 0.9}})

    def test_unknown_group_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown parameter 'deliverry'"):
            ModelParameters.from_dict({'deliverry': {'max_sdr': 0.9}})

    def test_the_error_suggests_the_closest_valid_name(self):
        with pytest.raises(ValueError, match="Did you mean 'max_sdr'"):
            ModelParameters.from_dict({'delivery': {'mx_sdr': 0.9}})

    def test_a_field_under_the_wrong_group_is_rejected(self):
        # max_sdr is real, but not in fire_adjustment.
        with pytest.raises(ValueError, match='Unknown parameter'):
            ModelParameters.from_dict({'fire_adjustment': {'max_sdr': 0.9}})

    def test_unknown_override_key_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown parameter'):
            ModelParameters().replace(delivery__max_sdrr=0.9)

    def test_a_non_object_group_is_rejected(self):
        with pytest.raises(ValueError, match='expected an object'):
            ModelParameters.from_dict({'delivery': 0.8})


class TestSerialisation:

    def test_round_trips_through_a_dict(self):
        params = ModelParameters().replace(
            delivery__max_sdr=0.9, fire_adjustment__c_peak=0.4,
        )
        assert ModelParameters.from_dict(params.to_dict()) == params

    def test_round_trips_through_json(self):
        params = ModelParameters().replace(debris__num_sim_years=3)
        rebuilt = ModelParameters.from_dict(
            json.loads(json.dumps(params.to_dict()))
        )
        assert rebuilt == params

    def test_a_partial_dict_keeps_defaults_for_everything_else(self):
        params = ModelParameters.from_dict({'delivery': {'max_sdr': 0.9}})
        assert params.delivery.max_sdr == 0.9
        assert params.delivery.ic0 == DeliveryParams().ic0
        assert params.fire_adjustment == FireAdjustmentParams()

    def test_an_empty_dict_gives_the_defaults(self):
        assert ModelParameters.from_dict({}) == ModelParameters()

    def test_nested_groups_round_trip(self):
        params = ModelParameters.from_dict(
            {'debris': {'channel': {'rock': 0.5}}}
        )
        assert params.debris.channel.rock == 0.5
        assert params.debris.hillslope == DebrisFlowParams().hillslope


class TestReplace:

    def test_dotted_and_underscore_forms_agree(self):
        by_underscore = ModelParameters().replace(delivery__max_sdr=0.9)
        by_dot = ModelParameters().replace(**{'delivery.max_sdr': 0.9})
        assert by_underscore == by_dot

    def test_replace_returns_a_new_instance(self):
        original = ModelParameters()
        updated = original.replace(delivery__max_sdr=0.9)
        assert original.delivery.max_sdr == 0.8
        assert updated is not original

    def test_replace_with_nothing_is_identity(self):
        params = ModelParameters()
        assert params.replace() is params

    def test_reaches_nested_groups_at_any_depth(self):
        # Regression: replace() used to hard-code exactly two path
        # segments, so the whole DebrisDepthParams tree — the debris knobs
        # most likely to be regionally refitted — could be set from JSON
        # but never from a call site.
        params = ModelParameters().replace(debris__hillslope__ae=5e-4)
        assert params.debris.hillslope.ae == 5e-4
        assert params.debris.channel == DebrisFlowParams().channel

    def test_dotted_nested_form_works_too(self):
        params = ModelParameters().replace(**{'debris.channel.rock': 0.5})
        assert params.debris.channel.rock == 0.5

    def test_a_bare_field_name_is_rejected(self):
        with pytest.raises(ValueError, match='must name a group'):
            ModelParameters().replace(max_sdr=0.9)


class TestTypeCoercion:
    """JSON has no int/float distinction, and a hand-edited file can hold
    anything. Coerce to the annotated type, or reject."""

    def test_an_int_for_a_float_field_normalises(self):
        params = ModelParameters.from_dict({'delivery': {'max_sdr': 1}})
        assert isinstance(params.delivery.max_sdr, float)

    def test_int_and_float_spellings_share_a_digest(self):
        # Otherwise an int/float difference in a hand-edited parameters.json
        # would spuriously flag every derived layer stale.
        as_int = ModelParameters.from_dict({'delivery': {'max_sdr': 1}})
        as_float = ModelParameters.from_dict({'delivery': {'max_sdr': 1.0}})
        assert as_int.digest() == as_float.digest()

    def test_a_string_for_a_number_is_rejected_as_a_parameter_error(self):
        # Previously leaked a bare TypeError from the __post_init__
        # comparison.
        with pytest.raises(ValueError, match='must be a number'):
            ModelParameters.from_dict({'delivery': {'max_sdr': '0.9'}})

    def test_a_fractional_count_is_rejected(self):
        with pytest.raises(ValueError, match='must be a whole number'):
            ModelParameters.from_dict({'debris': {'num_sim_years': 2.7}})

    def test_a_whole_float_count_is_accepted(self):
        params = ModelParameters.from_dict({'debris': {'num_sim_years': 2.0}})
        assert params.debris.num_sim_years == 2

    def test_a_bool_is_not_a_number(self):
        with pytest.raises(ValueError, match='must be a number'):
            ModelParameters.from_dict({'delivery': {'max_sdr': True}})

    def test_an_optional_field_accepts_none(self):
        params = ModelParameters.from_dict({'severity': {'force_sensor': None}})
        assert params.severity.force_sensor is None

    def test_an_optional_field_accepts_its_type(self):
        params = ModelParameters.from_dict(
            {'severity': {'force_sensor': 'landsat'}})
        assert params.severity.force_sensor == 'landsat'

    def test_pep604_unions_are_coerced(self):
        # get_origin() gives types.UnionType for `X | None` on 3.10-3.13 and
        # typing.Union for typing.Optional; both must be matched or the
        # branch is dead on the pinned runtime.
        import types as _types
        from typing import Optional, get_origin
        from fire_impacts.params import _coerce
        for annotation in (str | None, Optional[str]):
            assert get_origin(annotation) in (Union, _types.UnionType)
            assert _coerce('landsat', annotation, 'x') == 'landsat'
            assert _coerce(None, annotation, 'x') is None
            with pytest.raises(ValueError):
                _coerce(5, annotation, 'x')

    def test_numpy_scalars_are_accepted(self):
        np = pytest.importorskip('numpy')
        for value in (np.float32(0.5), np.float64(0.5)):
            assert ModelParameters.from_dict(
                {'delivery': {'max_sdr': value}}).delivery.max_sdr == 0.5
        for value in (np.int32(3), np.int64(3)):
            assert ModelParameters.from_dict(
                {'debris': {'num_sim_years': value}}).debris.num_sim_years == 3

    def test_an_out_of_range_number_is_a_parameter_error(self):
        # float(10**400) raises OverflowError, which would escape the
        # ValueError contract and bypass the path prefix.
        with pytest.raises(ValueError, match='out of range'):
            ModelParameters.from_dict({'delivery': {'max_sdr': 10 ** 400}})

    def test_an_optional_field_rejects_a_wrong_type(self):
        with pytest.raises(ValueError, match='force_sensor'):
            ModelParameters.from_dict({'severity': {'force_sensor': 5}})


class TestDigest:

    def test_is_stable_across_instances(self):
        assert ModelParameters().digest() == ModelParameters().digest()

    def test_changes_when_a_value_changes(self):
        assert (
            ModelParameters().digest()
            != ModelParameters().replace(delivery__max_sdr=0.9).digest()
        )

    def test_survives_a_json_round_trip(self):
        params = ModelParameters().replace(delivery__ic0=0.6)
        rebuilt = ModelParameters.from_dict(
            json.loads(json.dumps(params.to_dict()))
        )
        assert rebuilt.digest() == params.digest()

    def test_group_digest_only_covers_the_named_groups(self):
        base = ModelParameters()
        # Changing erosion must not disturb the delivery-group digest, or
        # every simulation would look like it invalidated the SDR layers.
        changed = base.replace(erosion__support_practice_factor=0.9)
        assert changed.group_digest('delivery') == base.group_digest('delivery')
        assert changed.digest() != base.digest()

    def test_group_digest_needs_at_least_one_path(self):
        # With none it hashes {}, identical for every parameter set, so a
        # staleness check would always report "unchanged".
        with pytest.raises(ValueError, match='at least one path'):
            ModelParameters().group_digest()

    def test_group_digest_rejects_an_unknown_path(self):
        with pytest.raises(ValueError, match='Unknown parameter'):
            ModelParameters().group_digest('nope')

    def test_group_digest_accepts_a_single_leaf(self):
        # The groups do not line up with the producers: topography holds
        # the headwater threshold (builds Headwaters.*) beside the LS
        # slope-length cap (builds LS_factor.tif). Digesting the whole
        # group would flag the LS factor stale whenever the headwater
        # threshold moved.
        params = ModelParameters()
        moved = params.replace(topography__headwater_threshold_m2=50000)
        leaf = 'topography.max_slope_length_m'
        assert moved.group_digest(leaf) == params.group_digest(leaf)
        assert moved.group_digest('topography') != \
            params.group_digest('topography')

    def test_subset_returns_only_the_named_paths(self):
        got = ModelParameters().subset('delivery.max_sdr',
                                       'erosion')
        assert set(got) == {'delivery', 'erosion'}
        assert set(got['delivery']) == {'max_sdr'}


class TestResolution:
    """Layered merge, most specific wins, with the origin recorded."""

    def test_no_layers_gives_the_defaults_all_marked_default(self):
        record = resolve_parameters([])
        assert record.parameters == ModelParameters()
        assert set(record.sources.values()) == {'default'}

    def test_a_later_layer_wins(self):
        record = resolve_parameters([
            ('project', {'delivery': {'max_sdr': 0.9}}),
            ('event', {'delivery': {'max_sdr': 0.7}}),
        ])
        assert record.parameters.delivery.max_sdr == 0.7
        assert record.sources['delivery.max_sdr'] == 'event'

    def test_layers_merge_rather_than_replace(self):
        record = resolve_parameters([
            ('project', {'delivery': {'max_sdr': 0.9}}),
            ('event', {'delivery': {'ic0': 0.6}}),
        ])
        assert record.parameters.delivery.max_sdr == 0.9
        assert record.parameters.delivery.ic0 == 0.6
        assert record.sources['delivery.max_sdr'] == 'project'
        assert record.sources['delivery.ic0'] == 'event'

    def test_untouched_values_are_marked_default(self):
        record = resolve_parameters([('project', {'delivery': {'ic0': 0.6}})])
        assert record.sources['delivery.max_sdr'] == 'default'
        assert record.sources['fire_adjustment.c_peak'] == 'default'

    def test_empty_layers_are_skipped(self):
        record = resolve_parameters([
            ('project', {}),
            ('event', None),
            ('call', {'delivery': {'k': 2.0}}),
        ])
        assert record.parameters.delivery.k == 2.0
        assert record.sources['delivery.k'] == 'call'

    def test_every_leaf_has_a_source(self):
        record = resolve_parameters([('project', {'delivery': {'ic0': 0.6}})])
        leaves = _leaf_paths(record.parameters.to_dict())
        assert set(record.sources) == leaves

    def test_validation_still_applies_after_merging(self):
        with pytest.raises(ValueError, match='max_sdr'):
            resolve_parameters([('project', {'delivery': {'max_sdr': 3.0}})])

    def test_unknown_keys_are_rejected_at_merge_time(self):
        with pytest.raises(ValueError, match='Unknown parameter'):
            resolve_parameters([('project', {'delivery': {'nope': 1}})])


class TestParameterRecord:

    def test_to_dict_carries_values_sources_and_digest(self):
        record = resolve_parameters([('project', {'delivery': {'ic0': 0.6}})])
        data = record.to_dict()
        assert data['values']['delivery']['ic0'] == 0.6
        assert data['sources']['delivery.ic0'] == 'project'
        assert data['digest'] == record.parameters.digest()
        assert data['resolved_at']
        assert 'fire_impacts_version' in data

    def test_round_trips(self):
        record = resolve_parameters([('event', {'delivery': {'ic0': 0.6}})])
        rebuilt = ParameterRecord.from_dict(record.to_dict())
        assert rebuilt.parameters == record.parameters
        assert rebuilt.sources == record.sources
        assert rebuilt.digest() == record.digest()

    def test_is_json_serialisable(self):
        record = resolve_parameters([('event', {'debris': {'num_sim_years': 3}})])
        assert json.loads(json.dumps(record.to_dict()))['values'][
            'debris']['num_sim_years'] == 3

    def test_two_identical_resolutions_compare_equal(self):
        # resolved_at differs between them; comparing it makes == useless.
        assert resolve_parameters([]) == resolve_parameters([])

    def test_sources_for_answers_what_nobody_chose(self):
        record = resolve_parameters([('project', {'delivery': {'ic0': 0.6}})])
        assert record.sources_for('project') == ['delivery.ic0']
        assert 'delivery.max_sdr' in record.sources_for('default')


class TestSparseOverrides:
    """An override file records choices, not the whole parameter set."""

    def test_defaults_reduce_to_nothing(self):
        assert sparse_overrides(ModelParameters()) == {}

    def test_only_changed_leaves_survive(self):
        got = sparse_overrides(ModelParameters().replace(delivery__ic0=0.6))
        assert got == {'delivery': {'ic0': 0.6}}

    def test_nested_changes_survive(self):
        got = sparse_overrides(
            ModelParameters().replace(debris__hillslope__ae=5e-4))
        assert got == {'debris': {'hillslope': {'ae': 5e-4}}}

    def test_round_trips_back_to_the_same_parameters(self):
        original = ModelParameters().replace(
            delivery__ic0=0.6, debris__channel__rock=0.5)
        assert ModelParameters.from_dict(
            sparse_overrides(original)) == original


class TestRestrictedToScope:
    """A record written beside outputs at a broader scope must not carry
    values that could not have influenced them."""

    def test_narrower_scoped_values_drop_to_default(self):
        record = resolve_parameters([
            ('catchment', {'delivery': {'max_sdr': 0.5}}),
            ('event', {'fire_adjustment': {'c_peak': 0.5}}),
        ])
        restricted = record.restricted_to_scope('catchment')
        assert restricted.parameters.delivery.max_sdr == 0.5
        assert restricted.parameters.fire_adjustment.c_peak == \
            FireAdjustmentParams().c_peak

    def test_digest_is_stable_across_event_scoped_changes(self):
        # Otherwise the catchment record's digest flips on every event
        # switch, giving a false staleness positive each time.
        def catchment_digest(c_peak):
            return resolve_parameters([
                ('catchment', {'delivery': {'max_sdr': 0.5}}),
                ('event', {'fire_adjustment': {'c_peak': c_peak}}),
            ]).restricted_to_scope('catchment').digest()
        assert catchment_digest(0.5) == catchment_digest(0.9)

    def test_digest_still_moves_on_a_catchment_scoped_change(self):
        def catchment_digest(max_sdr):
            return resolve_parameters([
                ('catchment', {'delivery': {'max_sdr': max_sdr}}),
            ]).restricted_to_scope('catchment').digest()
        assert catchment_digest(0.5) != catchment_digest(0.6)

    def test_rejects_an_unknown_scope(self):
        with pytest.raises(ValueError, match='Unknown scope'):
            resolve_parameters([]).restricted_to_scope('nope')


class TestScope:
    """Each parameter declares the scope of the output it controls."""

    def test_check_scope_rejects_an_unknown_group_even_when_empty(self):
        # _flatten yields no leaves for an empty group, so the group names
        # have to be validated separately.
        with pytest.raises(ValueError, match='Unknown parameter'):
            check_scope({'deliverry': {}}, 'event')

    def test_group_scopes(self):
        assert TopographyParams.__scope__ == 'catchment'
        assert DeliveryParams.__scope__ == 'catchment'
        assert FireAdjustmentParams.__scope__ == 'event'
        assert SeverityParams.__scope__ == 'event'
        assert ErosionParams.__scope__ == 'run'
        assert DebrisFlowParams.__scope__ == 'run'

    def test_a_field_can_narrow_its_groups_scope(self):
        # fire_adjustment is event-scoped, but default_c_factor writes the
        # catchment-level C_factor.tif every event shares.
        assert scope_of('fire_adjustment.c_peak') == 'event'
        assert scope_of('fire_adjustment.default_c_factor') == 'catchment'

    def test_nested_groups_inherit_their_parents_scope(self):
        assert scope_of('debris.hillslope.ae') == 'run'

    def test_a_group_path_returns_the_group_scope(self):
        assert scope_of('delivery') == 'catchment'

    def test_unknown_path_raises(self):
        with pytest.raises(ValueError, match='Unknown parameter'):
            scope_of('delivery.nope')

    def test_check_scope_rejects_a_narrower_layer(self):
        with pytest.raises(ValueError, match='catchment-scoped'):
            check_scope({'delivery': {'max_sdr': 0.9}}, 'event')

    def test_check_scope_allows_a_broader_layer(self):
        check_scope({'delivery': {'max_sdr': 0.9}}, 'catchment')
        check_scope({'delivery': {'max_sdr': 0.9}}, 'project')

    def test_check_scope_reports_every_offender(self):
        with pytest.raises(ValueError, match='2 parameter'):
            check_scope(
                {'delivery': {'max_sdr': 0.9},
                 'topography': {'max_slope_length_m': 200.0}},
                'event',
            )

    def test_check_scope_rejects_an_unknown_layer(self):
        with pytest.raises(ValueError, match='Unknown scope'):
            check_scope({}, 'nonsense')

    def test_every_leaf_has_a_resolvable_scope(self):
        for path in _leaf_paths(ModelParameters().to_dict()):
            assert scope_of(path) in SCOPES


def _leaf_paths(data, prefix=''):
    """Flatten a nested dict to the set of its dotted leaf paths."""
    paths = set()
    for key, value in data.items():
        full = f'{prefix}{key}'
        if isinstance(value, dict):
            paths |= _leaf_paths(value, prefix=f'{full}.')
        else:
            paths.add(full)
    return paths
