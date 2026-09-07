"""
Input bindings: the tagged union and its resolution rules.

A binding says where a gridded input comes from, as distinct from a
parameter, which says what coefficient the model uses. These cover the
union itself; the resolver that writes rasters is covered end to end in
test_integration_pipeline.py, where a real DEM is available.
"""

import json

import pytest

from fire_impacts import const as c
from fire_impacts.bindings import (
    DNBR_UNITS,
    Constant,
    Derived,
    FromFile,
    InputBindings,
    SyntheticFire,
    binding_from_dict,
)


class TestUnitsAreRequired:
    """dNBR is stored on a different scale from the one it is quoted on,
    and the two differ by 1000x. That mismatch already shipped once, via
    two producers writing incompatible rasters — so a binding has to say
    which scale it means, and there is no default."""

    def test_a_constant_without_units_is_rejected(self):
        with pytest.raises(ValueError, match='Incomplete'):
            binding_from_dict({'source': 'constant', 'value': 300})

    def test_a_file_without_units_is_rejected(self):
        with pytest.raises(ValueError, match='Incomplete'):
            binding_from_dict({'source': 'file', 'path': '/tmp/x.tif'})

    def test_empty_units_are_rejected_by_the_variant(self):
        with pytest.raises(ValueError, match='needs units'):
            Constant(value=300, units='')

    def test_unknown_units_are_rejected_against_the_input(self):
        # Units are validated where the input is known: a Constant does
        # not know whether it is painting a dNBR or a cover factor.
        with pytest.raises(ValueError, match='Unknown units'):
            InputBindings(dnbr=Constant(value=300, units='metres'))

    def test_conventional_units_convert_to_the_stored_scale(self):
        assert Constant(value=300, units='dnbr_x1000').to_stored_scale() \
            == pytest.approx(0.3)

    def test_stored_units_pass_through(self):
        assert Constant(value=0.3, units='dnbr').to_stored_scale() \
            == pytest.approx(0.3)

    def test_the_unit_factors_match_the_package_scale(self):
        assert DNBR_UNITS['dnbr'] == 1.0
        assert DNBR_UNITS['dnbr_x1000'] == pytest.approx(1 / c.DNBR_SCALE)

    def test_a_file_binding_reports_its_scale_factor(self):
        assert FromFile(path='x.tif', units='dnbr_x1000') \
            .scale_to_stored() == pytest.approx(1 / c.DNBR_SCALE)


class TestPerInputRules:
    """Units and permitted variants depend on the input, which a binding
    cannot know by itself."""

    def test_a_cover_factor_is_dimensionless(self):
        bound = InputBindings(
            c_factor=Constant(value=0.01, units='dimensionless'))
        assert bound.c_factor.to_stored_scale('c_factor') == 0.01

    def test_dnbr_units_are_refused_for_the_cover_factor(self):
        with pytest.raises(ValueError, match='Unknown units'):
            InputBindings(
                c_factor=Constant(value=0.01, units='dnbr_x1000'))

    def test_dimensionless_is_refused_for_dnbr(self):
        with pytest.raises(ValueError, match='Unknown units'):
            InputBindings(dnbr=Constant(value=0.3, units='dimensionless'))

    def test_a_synthetic_cover_factor_is_refused(self):
        # There is no reference distribution to sample a cover factor
        # from; accepting one would write dNBR values into a C-factor file.
        with pytest.raises(ValueError, match='not defined for'):
            InputBindings(c_factor=SyntheticFire())

    def test_a_synthetic_dnbr_is_allowed(self):
        assert isinstance(
            InputBindings(dnbr=SyntheticFire()).dnbr, SyntheticFire)

    @pytest.mark.parametrize('input_name', ['dnbr', 'c_factor'])
    def test_every_input_accepts_derived_constant_and_file(self, input_name):
        from fire_impacts.bindings import UNITS_BY_INPUT
        units = sorted(UNITS_BY_INPUT[input_name])[0]
        for binding in (Derived(),
                        Constant(value=0.1, units=units),
                        FromFile(path='x.tif', units=units)):
            InputBindings(**{input_name: binding})


class TestDomains:

    def test_the_default_domain_is_the_catchment(self):
        assert Constant(value=0.3, units='dnbr').domain == 'catchment'

    @pytest.mark.parametrize('domain', [
        'catchment', 'dem_valid', 'mask:FireSeverity/masked_dNBR.tif'])
    def test_valid_domains_are_accepted(self, domain):
        assert Constant(value=0.3, units='dnbr', domain=domain).domain \
            == domain

    def test_an_unknown_domain_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown domain'):
            Constant(value=0.3, units='dnbr', domain='everywhere')

    def test_a_mask_domain_must_name_its_layer(self):
        with pytest.raises(ValueError, match='must name the layer'):
            Constant(value=0.3, units='dnbr', domain='mask')


class TestTaggedUnion:

    def test_an_unknown_source_is_rejected_with_a_suggestion(self):
        with pytest.raises(ValueError, match="Did you mean 'constant'"):
            binding_from_dict({'source': 'konstant'})

    def test_a_missing_source_is_rejected(self):
        with pytest.raises(ValueError, match='needs a "source" key'):
            binding_from_dict({'value': 300})

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValueError, match='Unknown field'):
            binding_from_dict(
                {'source': 'derived', 'severity': 'high'})

    def test_an_unknown_input_name_is_rejected(self):
        # Only dNBR is bindable; a typo must not silently do nothing.
        with pytest.raises(ValueError, match='Unknown input'):
            InputBindings.from_dict({'dnbrr': {'source': 'derived'}})

    def test_none_resolves_to_derived(self):
        assert isinstance(binding_from_dict(None), Derived)

    def test_a_non_object_is_rejected(self):
        with pytest.raises(ValueError, match='must be an object'):
            binding_from_dict(300)

    @pytest.mark.parametrize('binding', [
        Derived(),
        Constant(value=300, units='dnbr_x1000', domain='dem_valid'),
        FromFile(path='/data/dnbr.tif', units='dnbr'),
        SyntheticFire(severity='high', random_seed=7),
    ])
    def test_every_variant_round_trips(self, binding):
        assert binding_from_dict(binding.to_dict()) == binding

    def test_the_set_round_trips_through_json(self):
        bindings = InputBindings(
            dnbr=Constant(value=300, units='dnbr_x1000'))
        rebuilt = InputBindings.from_dict(
            json.loads(json.dumps(bindings.to_dict())))
        assert rebuilt == bindings


class TestDefaults:

    def test_the_default_is_the_derived_pipeline(self):
        assert isinstance(InputBindings().dnbr, Derived)
        assert InputBindings().is_default()

    def test_a_substituted_input_is_not_default(self):
        assert not InputBindings(
            dnbr=Constant(value=0.3, units='dnbr')).is_default()

    def test_each_input_declares_the_scope_of_its_output(self):
        from fire_impacts.bindings import SCOPE_BY_INPUT
        # masked_dNBR.tif is per event; C_factor.tif is built once per
        # catchment and shared by every fire in it.
        assert SCOPE_BY_INPUT['dnbr'] == 'event'
        assert SCOPE_BY_INPUT['c_factor'] == 'catchment'

    def test_every_bindable_input_declares_a_scope(self):
        from dataclasses import fields
        from fire_impacts.bindings import SCOPE_BY_INPUT
        assert {f.name for f in fields(InputBindings)} == set(SCOPE_BY_INPUT)


class TestScopeEnforcement:
    """A binding may only be set at a layer at least as broad as the
    output it produces — the same rule the parameter groups follow."""

    def test_a_catchment_scoped_input_is_refused_at_event_scope(self):
        from fire_impacts.bindings import check_binding_scope
        with pytest.raises(ValueError, match='catchment-scoped'):
            check_binding_scope(
                {'c_factor': {'source': 'derived'}}, 'event')

    def test_the_error_names_where_it_belongs(self):
        from fire_impacts.bindings import check_binding_scope
        with pytest.raises(ValueError, match='catchment-level'):
            check_binding_scope(
                {'c_factor': {'source': 'derived'}}, 'event')

    def test_an_event_scoped_input_is_allowed_at_event_scope(self):
        from fire_impacts.bindings import check_binding_scope
        check_binding_scope({'dnbr': {'source': 'derived'}}, 'event')

    def test_broader_layers_accept_everything(self):
        from fire_impacts.bindings import check_binding_scope
        both = {'dnbr': {'source': 'derived'},
                'c_factor': {'source': 'derived'}}
        check_binding_scope(both, 'catchment')
        check_binding_scope(both, 'project')

    def test_an_unknown_input_is_rejected(self):
        from fire_impacts.bindings import check_binding_scope
        with pytest.raises(ValueError, match='Unknown input'):
            check_binding_scope({'nope': {'source': 'derived'}}, 'project')


class TestSyntheticSeed:

    def test_a_pinned_seed_is_kept(self):
        assert SyntheticFire(random_seed=7).random_seed == 7

    def test_an_unpinned_seed_is_none_until_resolved(self):
        # The resolver draws one and records the effective value; storing
        # None in the record would leave it unable to describe the draw
        # that actually produced the raster beside it.
        assert SyntheticFire().random_seed is None

    def test_a_non_integer_seed_is_rejected(self):
        with pytest.raises(ValueError, match='random_seed'):
            SyntheticFire(random_seed='seven')
