"""
Severity acquisition parameters.

These do not change any equation, but they change the imagery the dNBR is
built from — and every downstream layer is built on that dNBR. They were
partly exposed as keyword arguments and partly buried as literals (the
+/-90 day windows, the 10 km search buffer), and none of them were
recorded anywhere.

calculate_fire_severity itself needs the DEA STAC catalogue, so these
tests cover the parameter resolution and the defaults rather than a live
acquisition.
"""

import inspect

import pytest

from fire_impacts import const as c
from fire_impacts.params import SeverityParams
from fire_impacts.pre import mask_dnbr as mdnbr
from fire_impacts.pre import severity


class TestBuriedDefaultsAreNowParameters:
    """The windows and buffer used to be literals inside the function
    body, so they could not be changed or recorded."""

    def test_the_imagery_windows_are_parameters(self):
        params = SeverityParams()
        assert params.pre_fire_window_days == 90
        assert params.post_fire_window_days == 90

    def test_the_search_buffer_is_a_parameter(self):
        assert SeverityParams().bbox_buffer_km == 10.0

    def test_the_literals_are_gone_from_the_function_body(self):
        src = inspect.getsource(severity.calculate_fire_severity)
        assert 'date_rel(fire_start_date, -90)' not in src
        assert 'catchment_bounds(catchment, 10)' not in src
        assert 'sev.pre_fire_window_days' in src
        assert 'sev.bbox_buffer_km' in src

    def test_the_windows_can_be_set_independently(self):
        params = SeverityParams(
            pre_fire_window_days=30, post_fire_window_days=180)
        assert params.pre_fire_window_days == 30
        assert params.post_fire_window_days == 180

    @pytest.mark.parametrize('field', [
        'pre_fire_window_days', 'post_fire_window_days'])
    def test_a_non_positive_window_is_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            SeverityParams(**{field: 0})


class TestDeprecatedKwargs:

    @pytest.mark.parametrize('name', [
        'max_cloud_cover', 'resolution_input', 'force_sensor'])
    def test_severity_kwargs_use_the_sentinel(self, name):
        sig = inspect.signature(severity.calculate_fire_severity)
        assert sig.parameters[name].default is c.UNSET, name

    def test_mask_dnbr_natural_code_uses_the_sentinel(self):
        sig = inspect.signature(mdnbr.mask_dnbr)
        assert sig.parameters['natural_code'].default is c.UNSET

    def test_both_accept_params(self):
        for fn in (severity.calculate_fire_severity, mdnbr.mask_dnbr):
            assert 'params' in inspect.signature(fn).parameters, fn.__name__


class TestResolution:
    """The resolved values must reach the local names the body uses."""

    def test_severity_group_is_event_scoped(self):
        # FireSeverity outputs are per event, so these may vary per fire.
        assert SeverityParams.__scope__ == 'event'

    def test_defaults_match_the_previous_signature_defaults(self):
        # Behaviour preservation: the old signature had 20 and 20.
        assert SeverityParams().max_cloud_cover == 20
        assert SeverityParams().resolution_m == 20
        assert SeverityParams().force_sensor is None

    def test_natural_veg_code_default_is_unchanged(self):
        assert SeverityParams().natural_veg_code == 112

    def test_force_sensor_is_constrained_to_known_sensors(self):
        for sensor in (None, 'landsat', 'sentinel'):
            assert SeverityParams(force_sensor=sensor).force_sensor == sensor
        with pytest.raises(ValueError, match='force_sensor'):
            SeverityParams(force_sensor='modis')

    def test_cloud_cover_is_a_percentage(self):
        with pytest.raises(ValueError, match='max_cloud_cover'):
            SeverityParams(max_cloud_cover=150)
