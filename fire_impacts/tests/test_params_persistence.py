"""
Parameter persistence: parameters.json, event.json, and the RunContext merge.

Project-scope overrides live in a separate <project>/parameters.json rather
than settings.json, because settings.json is machine-written — _settings()
only serialises its four known attributes and _write() rewrites the file
whenever a catchment is added, so a key added there would be dropped. The
first test class pins that reasoning.

Design notes: design-notes/calibration-parameters-proposal.md
"""

import json
import os

import pytest

from fire_impacts import const as c
from fire_impacts.context import RunContext
from fire_impacts.params import DeliveryParams, ModelParameters
from fire_impacts.pre.project import FireImpactsProject


@pytest.fixture
def project(tmp_path):
    """A bare project with one registered catchment, no data."""
    proj = FireImpactsProject(str(tmp_path / 'proj'), exist_ok=False)
    # Register a catchment without going through add_catchment(), which
    # wants a real shapefile. These tests only exercise paths and JSON.
    proj.catchments.append('TestCatchment')
    proj._write()
    return proj


@pytest.fixture
def ctx(project):
    return RunContext(project=project, catchment='TestCatchment', event='fire1')


class TestProjectParameterStore:

    def test_absent_file_means_no_overrides(self, project):
        assert project.parameter_overrides() == {}

    def test_writes_and_reads_back_a_sparse_dict(self, project):
        project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        assert project.parameter_overrides() == {'delivery': {'max_sdr': 0.9}}

    def test_writes_to_parameters_json_not_settings_json(self, project):
        project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        params_path = os.path.join(
            project.project_path, c.PARAMETERS_FILE_NAME)
        assert os.path.exists(params_path)
        with open(os.path.join(project.project_path, 'settings.json')) as f:
            assert 'parameters' not in json.load(f)

    def test_survives_a_settings_rewrite(self, project):
        # The whole reason for a separate file: _write() rewrites
        # settings.json wholesale, so anything stored there would be lost.
        project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        project._write()
        assert project.parameter_overrides() == {'delivery': {'max_sdr': 0.9}}

    def test_survives_a_project_reload(self, project):
        project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        reloaded = FireImpactsProject(project.project_path, exist_ok=True)
        assert reloaded.parameter_overrides() == {'delivery': {'max_sdr': 0.9}}

    def test_a_full_parameter_set_is_reduced_to_what_changed(self, project):
        # Writing the whole set would make every package default an
        # explicit user setting: sources would read back as
        # chosen-everywhere, and a later release that fixes a default would
        # be silently overridden by the frozen file.
        project.set_parameter_overrides(
            ModelParameters().replace(delivery__ic0=0.6))
        assert project.parameter_overrides() == {'delivery': {'ic0': 0.6}}

    def test_writing_the_bare_defaults_writes_nothing(self, project):
        project.set_parameter_overrides(ModelParameters())
        assert project.parameter_overrides() == {}

    def test_a_full_set_does_not_pin_every_value_as_project_scoped(self, ctx):
        ctx.project.set_parameter_overrides(
            ModelParameters().replace(delivery__ic0=0.6))
        record = ctx.parameters()
        assert record.sources['delivery.ic0'] == 'project'
        assert record.sources['delivery.max_sdr'] == 'default'

    def test_an_event_scope_full_set_is_accepted_when_sparse(self, ctx):
        # Previously impossible: the full instance carried all ten
        # catchment-scoped leaves and always tripped the scope check.
        ctx.set_event_parameter_overrides(
            ModelParameters().replace(fire_adjustment__c_peak=0.4))
        assert ctx.event_parameter_overrides() == {
            'fire_adjustment': {'c_peak': 0.4}}

    def test_an_event_scope_full_set_still_refuses_out_of_scope_changes(
            self, ctx):
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.set_event_parameter_overrides(
                ModelParameters().replace(delivery__max_sdr=0.5))

    def test_invalid_overrides_raise_before_writing(self, project):
        with pytest.raises(ValueError, match='max_sdr'):
            project.set_parameter_overrides({'delivery': {'max_sdr': 5.0}})
        assert not os.path.exists(
            os.path.join(project.project_path, c.PARAMETERS_FILE_NAME))

    def test_unknown_keys_raise_before_writing(self, project):
        with pytest.raises(ValueError, match='Unknown parameter'):
            project.set_parameter_overrides({'delivery': {'nope': 1}})


class TestEventParameterStore:

    def test_absent_event_json_means_no_overrides(self, ctx):
        assert ctx.event_parameter_overrides() == {}

    def test_writes_and_reads_back(self, ctx):
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        assert ctx.event_parameter_overrides() == {'fire_adjustment': {'c_peak': 0.4}}

    def test_lands_in_the_events_event_json(self, ctx):
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        with open(ctx.event_path(c.EVENT_DEFINITION_NAME)) as f:
            assert json.load(f)['parameters'] == {
                'fire_adjustment': {'c_peak': 0.4}}

    def test_invalid_overrides_raise(self, ctx):
        with pytest.raises(ValueError, match='max_sdr'):
            ctx.set_event_parameter_overrides({'delivery': {'max_sdr': 5.0}})

    def test_requires_an_event(self, project):
        catchment_only = RunContext(
            project=project, catchment='TestCatchment')
        with pytest.raises(ValueError, match='no event'):
            catchment_only.set_event_parameter_overrides(
                {'fire_adjustment': {'c_peak': 0.4}})


class TestEventJsonCoexistence:
    """event.json holds several concerns; each writer must preserve the rest."""

    @pytest.fixture(autouse=True)
    def no_fire_meta(self, monkeypatch):
        """Stub the fire dates, which normally come from FireMeta.csv.

        These tests exercise event.json only; writing a FireSeverity
        folder just to satisfy the date lookup would obscure that.
        """
        monkeypatch.setattr(
            RunContext, 'fire_start_date', property(lambda self: None))
        monkeypatch.setattr(
            RunContext, 'fire_end_date', property(lambda self: None))

    def test_setting_parameters_preserves_recovery_breakpoints(self, ctx):
        ctx.set_recovery_breakpoints([0, 1, 2])
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        with open(ctx.event_path(c.EVENT_DEFINITION_NAME)) as f:
            data = json.load(f)
        assert data['recovery_breakpoints'] == [0, 1, 2]
        assert data['parameters'] == {'fire_adjustment': {'c_peak': 0.4}}

    def test_setting_breakpoints_preserves_parameters(self, ctx):
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        ctx.set_recovery_breakpoints([0, 1, 2])
        with open(ctx.event_path(c.EVENT_DEFINITION_NAME)) as f:
            data = json.load(f)
        assert data['parameters'] == {'fire_adjustment': {'c_peak': 0.4}}
        assert data['recovery_breakpoints'] == [0, 1, 2]

    def test_event_definition_ignores_the_parameters_key(self, ctx):
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        # from_dict only looks at recovery_breakpoints; the extra key must
        # not upset it.
        definition = ctx.event_definition()
        assert definition.recovery_breakpoints == list(
            c.DEFAULT_RECOVERY_BREAKPOINTS)


class TestCatchmentParameterStore:

    def test_absent_file_means_no_overrides(self, ctx):
        assert ctx.catchment_parameter_overrides() == {}

    def test_writes_and_reads_back(self, ctx):
        ctx.set_catchment_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        assert ctx.catchment_parameter_overrides() == {
            'delivery': {'max_sdr': 0.9}}

    def test_lands_under_the_catchment(self, ctx):
        ctx.set_catchment_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        assert os.path.exists(
            ctx.catchment_path(c.PARAMETERS_FILE_NAME))

    def test_is_separate_from_the_project_file(self, ctx):
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        ctx.set_catchment_parameter_overrides({'delivery': {'max_sdr': 0.7}})
        assert ctx.project.parameter_overrides() == {
            'delivery': {'max_sdr': 0.9}}
        assert ctx.catchment_parameter_overrides() == {
            'delivery': {'max_sdr': 0.7}}

    def test_rejects_an_unregistered_catchment(self, project):
        with pytest.raises(ValueError, match='not registered'):
            project.set_catchment_parameter_overrides(
                'Nope', {'delivery': {'max_sdr': 0.9}})

    def test_accepts_catchment_scoped_groups(self, ctx):
        # The whole point: topography and delivery belong here.
        ctx.set_catchment_parameter_overrides(
            {'topography': {'headwater_threshold_m2': 50000}})
        assert ctx.catchment_parameter_overrides()[
            'topography']['headwater_threshold_m2'] == 50000


class TestScopeEnforcement:
    """A catchment-scoped value in an event file is either ignored or
    corrupting, so it is refused at the point of writing."""

    def test_event_file_rejects_a_catchment_scoped_group(self, ctx):
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.set_event_parameter_overrides({'delivery': {'max_sdr': 0.9}})

    def test_event_file_rejects_topography(self, ctx):
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.set_event_parameter_overrides(
                {'topography': {'headwater_threshold_m2': 50000}})

    def test_event_file_rejects_the_catchment_scoped_field_of_a_mixed_group(
            self, ctx):
        # fire_adjustment is event-scoped, but default_c_factor writes the
        # catchment-level C_factor.tif that every event shares.
        with pytest.raises(ValueError, match='default_c_factor'):
            ctx.set_event_parameter_overrides(
                {'fire_adjustment': {'default_c_factor': 0.02}})

    def test_event_file_accepts_the_rest_of_that_group(self, ctx):
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        assert ctx.event_parameter_overrides()['fire_adjustment']['c_peak'] == 0.4

    def test_event_file_accepts_run_scoped_groups(self, ctx):
        ctx.set_event_parameter_overrides(
            {'erosion': {'support_practice_factor': 0.8}})
        assert ctx.event_parameter_overrides()['erosion'][
            'support_practice_factor'] == 0.8

    def test_the_error_names_where_to_put_it_instead(self, ctx):
        with pytest.raises(ValueError, match='catchment-level parameters'):
            ctx.set_event_parameter_overrides({'delivery': {'max_sdr': 0.9}})

    def test_catchment_file_accepts_everything(self, ctx):
        ctx.set_catchment_parameter_overrides({
            'delivery': {'max_sdr': 0.9},
            'topography': {'max_slope_length_m': 200.0},
            'fire_adjustment': {'c_peak': 0.4, 'default_c_factor': 0.02},
            'erosion': {'support_practice_factor': 0.8},
        })
        assert ctx.catchment_parameter_overrides()['delivery']['max_sdr'] == 0.9

    def test_a_hand_edited_event_file_is_refused_on_read(self, ctx):
        # The setters are not the only way a value gets into these files —
        # both are documented as hand-editable. A catchment-scoped value in
        # an event file would otherwise rewrite the SDR_baseline.tif and
        # LS_factor.tif that every other event in the catchment shares.
        import json
        path = ctx.event_path(c.EVENT_DEFINITION_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'parameters': {'delivery': {'max_sdr': 0.42}}}, f)
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.parameters()

    def test_the_read_error_names_the_offending_file(self, ctx):
        import json
        path = ctx.event_path(c.EVENT_DEFINITION_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'parameters': {'topography':
                                      {'max_slope_length_m': 7.0}}}, f)
        with pytest.raises(ValueError, match=c.EVENT_DEFINITION_NAME):
            ctx.parameters()

    def test_a_valid_hand_edited_event_file_still_resolves(self, ctx):
        import json
        path = ctx.event_path(c.EVENT_DEFINITION_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'parameters': {'fire_adjustment': {'c_peak': 0.4}}}, f)
        assert ctx.parameters().parameters.fire_adjustment.c_peak == 0.4

    def test_call_overrides_are_not_scope_restricted(self, ctx):
        # Deliberate: a call override is explicit, transient, and recorded
        # as 'call'. Only the persisted layers are enforced.
        record = ctx.parameters(delivery__max_sdr=0.9)
        assert record.parameters.delivery.max_sdr == 0.9
        assert record.sources['delivery.max_sdr'] == 'call'


class TestRunContextResolution:

    def test_defaults_when_nothing_is_configured(self, ctx):
        record = ctx.parameters()
        assert record.parameters == ModelParameters()
        assert set(record.sources.values()) == {'default'}

    def test_project_layer_is_picked_up(self, ctx):
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        record = ctx.parameters()
        assert record.parameters.delivery.max_sdr == 0.9
        assert record.sources['delivery.max_sdr'] == 'project'

    def test_event_layer_beats_catchment(self, ctx):
        ctx.set_catchment_parameter_overrides(
            {'fire_adjustment': {'c_peak': 0.5}})
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        record = ctx.parameters()
        assert record.parameters.fire_adjustment.c_peak == 0.4
        assert record.sources['fire_adjustment.c_peak'] == 'event'

    def test_call_layer_beats_event(self, ctx):
        ctx.project.set_parameter_overrides({'fire_adjustment': {'c_peak': 0.5}})
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        record = ctx.parameters(fire_adjustment__c_peak=0.3)
        assert record.parameters.fire_adjustment.c_peak == 0.3
        assert record.sources['fire_adjustment.c_peak'] == 'call'

    def test_layers_merge_across_different_fields(self, ctx):
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        record = ctx.parameters(erosion__support_practice_factor=0.8)
        assert record.parameters.delivery.max_sdr == 0.9
        assert record.parameters.fire_adjustment.c_peak == 0.4
        assert record.parameters.erosion.support_practice_factor == 0.8
        assert record.sources['delivery.max_sdr'] == 'project'
        assert record.sources['fire_adjustment.c_peak'] == 'event'
        assert record.sources['erosion.support_practice_factor'] == 'call'

    def test_an_explicit_call_override_equal_to_the_default_still_wins(self, ctx):
        # Regression: the call layer used to be built by diffing against
        # the package defaults, so reverting one knob to its default in a
        # calibrated project silently resolved to the project value —
        # exactly the failure this system exists to prevent.
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        record = ctx.parameters(delivery__max_sdr=DeliveryParams().max_sdr)
        assert record.parameters.delivery.max_sdr == DeliveryParams().max_sdr
        assert record.sources['delivery.max_sdr'] == 'call'

    def test_a_nested_group_is_reachable_from_a_call_override(self, ctx):
        record = ctx.parameters(debris__hillslope__ae=5e-4)
        assert record.parameters.debris.hillslope.ae == 5e-4
        assert record.sources['debris.hillslope.ae'] == 'call'

    def test_an_override_must_name_a_group(self, ctx):
        with pytest.raises(ValueError, match='must name a group'):
            ctx.parameters(max_sdr=0.9)

    def test_catchment_layer_beats_project(self, ctx):
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        ctx.set_catchment_parameter_overrides({'delivery': {'max_sdr': 0.7}})
        record = ctx.parameters()
        assert record.parameters.delivery.max_sdr == 0.7
        assert record.sources['delivery.max_sdr'] == 'catchment'

    def test_catchment_layer_applies_to_a_catchment_only_context(self, project):
        # The layer these parameters actually control (LS_factor, Headwaters)
        # is built from a catchment-only context, so the override has to
        # reach it there — this is the case the event layer could not serve.
        project.set_catchment_parameter_overrides(
            'TestCatchment', {'topography': {'headwater_threshold_m2': 50000}})
        catchment_only = RunContext(
            project=project, catchment='TestCatchment')
        record = catchment_only.parameters()
        assert record.parameters.topography.headwater_threshold_m2 == 50000
        assert record.sources[
            'topography.headwater_threshold_m2'] == 'catchment'

    def test_all_five_layers_compose(self, ctx):
        ctx.project.set_parameter_overrides({'severity': {'max_cloud_cover': 30.0}})
        ctx.set_catchment_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        ctx.set_event_parameter_overrides({'fire_adjustment': {'c_peak': 0.4}})
        record = ctx.parameters(erosion__support_practice_factor=0.8)
        assert record.sources['severity.max_cloud_cover'] == 'project'
        assert record.sources['delivery.max_sdr'] == 'catchment'
        assert record.sources['fire_adjustment.c_peak'] == 'event'
        assert record.sources['erosion.support_practice_factor'] == 'call'
        assert record.sources['debris.dnbr_threshold'] == 'default'

    def test_catchment_only_context_skips_the_event_layer(self, project):
        project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        catchment_only = RunContext(
            project=project, catchment='TestCatchment')
        record = catchment_only.parameters()
        assert record.parameters.delivery.max_sdr == 0.9
        assert record.sources['delivery.max_sdr'] == 'project'

    def test_bad_call_override_raises(self, ctx):
        with pytest.raises(ValueError, match='Unknown parameter'):
            ctx.parameters(delivery__nope=1)

    def test_params_and_an_override_together_are_refused(self, ctx):
        # Which one wins would be ambiguous, so it raises rather than
        # silently picking one.
        with pytest.raises(ValueError, match='ambiguous'):
            ctx._resolved_params(ctx.parameters(), delivery__max_sdr=0.5)

    def test_a_bare_model_parameters_is_attributed_to_the_call(self, ctx):
        record = ctx._resolved_params(
            ModelParameters().replace(delivery__max_sdr=0.5))
        assert record.parameters.delivery.max_sdr == 0.5
        assert set(record.sources.values()) == {'call'}

    def test_resolved_params_rejects_a_wrong_type(self, ctx):
        with pytest.raises(TypeError, match='ParameterRecord'):
            ctx._resolved_params({'delivery': {'max_sdr': 0.5}})

    def test_digest_reflects_the_resolved_values(self, ctx):
        before = ctx.parameters().digest()
        ctx.project.set_parameter_overrides({'delivery': {'max_sdr': 0.9}})
        assert ctx.parameters().digest() != before


class TestBindingsShareTheFile:
    """parameters.json carries parameter groups at the top level and input
    bindings under a "bindings" key. Each writer must preserve the other,
    and each reader must ignore the other."""

    def test_writing_parameters_keeps_the_bindings(self, ctx):
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        ctx.project.set_catchment_parameter_overrides(
            ctx.catchment, {'delivery': {'max_sdr': 0.7}})
        assert ctx.catchment_binding_overrides()['c_factor']['value'] == 0.02

    def test_writing_bindings_keeps_the_parameters(self, ctx):
        ctx.project.set_catchment_parameter_overrides(
            ctx.catchment, {'delivery': {'max_sdr': 0.7}})
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        assert ctx.project.catchment_parameter_overrides(
            ctx.catchment) == {'delivery': {'max_sdr': 0.7}}

    def test_the_parameter_reader_ignores_the_bindings_key(self, ctx):
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        # Would otherwise read 'bindings' as an unknown parameter group.
        assert ctx.parameters().parameters.delivery.max_sdr == 0.8

    def test_both_survive_a_reload(self, ctx):
        from fire_impacts.pre.project import FireImpactsProject
        ctx.project.set_catchment_parameter_overrides(
            ctx.catchment, {'delivery': {'max_sdr': 0.7}})
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        reloaded = FireImpactsProject(ctx.project.project_path, exist_ok=True)
        assert reloaded.catchment_parameter_overrides(ctx.catchment) == {
            'delivery': {'max_sdr': 0.7}}
        assert reloaded.catchment_binding_overrides(
            ctx.catchment)['c_factor']['value'] == 0.02


class TestBindingScope:
    """A binding may only be set at a layer at least as broad as the
    output it produces — the same rule the parameter groups follow."""

    def test_a_c_factor_binding_is_refused_at_event_scope(self, ctx):
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.set_event_binding_overrides({'c_factor': {
                'source': 'constant', 'value': 0.02,
                'units': 'dimensionless'}})

    def test_a_dnbr_binding_is_accepted_at_event_scope(self, ctx):
        ctx.set_event_binding_overrides({'dnbr': {
            'source': 'synthetic', 'severity': 'high'}})
        assert ctx.bindings().dnbr.severity == 'high'

    def test_a_c_factor_binding_is_accepted_at_catchment_scope(self, ctx):
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        assert ctx.bindings().c_factor.value == 0.02

    def test_a_hand_edited_event_file_is_refused_on_read(self, ctx):
        # The setters are not the only way in: the files are documented
        # as hand-editable.
        import json
        path = ctx.event_path(c.EVENT_DEFINITION_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'bindings': {'c_factor': {
                'source': 'constant', 'value': 0.02,
                'units': 'dimensionless'}}}, f)
        with pytest.raises(ValueError, match='catchment-scoped'):
            ctx.bindings()

    def test_the_read_error_names_the_offending_file(self, ctx):
        import json
        path = ctx.event_path(c.EVENT_DEFINITION_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'bindings': {'c_factor': {
                'source': 'constant', 'value': 0.02,
                'units': 'dimensionless'}}}, f)
        with pytest.raises(ValueError, match=c.EVENT_DEFINITION_NAME):
            ctx.bindings()

    def test_layers_merge_across_scopes(self, ctx):
        ctx.set_catchment_binding_overrides({'c_factor': {
            'source': 'constant', 'value': 0.02,
            'units': 'dimensionless'}})
        ctx.set_event_binding_overrides({'dnbr': {
            'source': 'synthetic', 'severity': 'high'}})
        bindings = ctx.bindings()
        assert bindings.c_factor.value == 0.02
        assert bindings.dnbr.severity == 'high'
