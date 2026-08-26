"""
Per-layer provenance tags and the staleness check built on them.

A provenance.json records what a *step* resolved; a tag records what a
*file* was built from. The tag is the better authority because it is
per-file — it survives partial rebuilds, orphaned layers from a shortened
breakpoint list, and the raster being copied out of the project.
"""

import json

import numpy as np
import pytest
import rasterio

from fire_impacts.params import ModelParameters, resolve_parameters
from fire_impacts.pre.util import write_raster
from fire_impacts.provenance import (
    DIGEST_TAG,
    PARAMS_TAG,
    VERSION_TAG,
    check_layer_freshness,
    parameter_tags,
    read_parameter_tags,
)


@pytest.fixture
def raster(tmp_path):
    """A tiny raster factory; returns write(path_name, tags) -> path."""
    def _write(name, tags=None):
        path = tmp_path / f'{name}.tif'
        meta = {
            'driver': 'GTiff', 'height': 2, 'width': 2, 'count': 1,
            'dtype': 'float32', 'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_origin(0, 0, 1, 1),
        }
        write_raster(str(path), np.ones((2, 2), dtype='float32'), meta,
                     tags=tags)
        return str(path)
    return _write


class TestTagging:

    def test_tags_round_trip(self, raster):
        record = resolve_parameters([('project', {'delivery':
                                                  {'max_sdr': 0.9}})])
        path = raster('sdr', parameter_tags(record, 'delivery'))
        got = read_parameter_tags(path)
        assert got['digest'] == record.parameters.group_digest('delivery')
        assert got['values']['delivery']['max_sdr'] == 0.9
        assert got['version'] == record.version

    def test_only_the_consumed_paths_are_stamped(self, raster):
        record = resolve_parameters([])
        path = raster('ls', parameter_tags(
            record, 'topography.max_slope_length_m'))
        values = read_parameter_tags(path)['values']
        assert set(values) == {'topography'}
        assert set(values['topography']) == {'max_slope_length_m'}

    def test_an_untagged_raster_reads_back_as_none(self, raster):
        assert read_parameter_tags(raster('plain')) is None

    def test_the_tag_names_are_namespaced(self, raster):
        record = resolve_parameters([])
        path = raster('x', parameter_tags(record, 'delivery'))
        with rasterio.open(path) as src:
            tags = src.tags()
        for tag in (PARAMS_TAG, DIGEST_TAG, VERSION_TAG):
            assert tag in tags
            assert tag.startswith('FIRE_IMPACTS_')

    def test_the_stamped_values_are_canonical_json(self, raster):
        record = resolve_parameters([])
        path = raster('x', parameter_tags(record, 'delivery'))
        with rasterio.open(path) as src:
            raw = src.tags()[PARAMS_TAG]
        # Sorted keys and no whitespace, so the tag is stable across runs.
        assert raw == json.dumps(json.loads(raw), sort_keys=True,
                                 separators=(',', ':'))


class TestFreshness:

    def test_matching_parameters_pass(self, raster):
        record = resolve_parameters([])
        path = raster('sdr', parameter_tags(record, 'delivery'))
        assert check_layer_freshness(path, record, 'delivery') is True

    def test_a_changed_parameter_raises(self, raster):
        built = resolve_parameters([])
        path = raster('sdr', parameter_tags(built, 'delivery'))
        changed = resolve_parameters(
            [('project', {'delivery': {'max_sdr': 0.5}})])
        with pytest.raises(ValueError, match='built with different'):
            check_layer_freshness(path, changed, 'delivery')

    def test_the_message_names_the_differing_value(self, raster):
        """Not two digests — those tell a reader nothing. The digest is
        only the fast 'are these identical' test."""
        built = resolve_parameters([])
        path = raster('sdr', parameter_tags(built, 'delivery'))
        changed = resolve_parameters(
            [('project', {'delivery': {'max_sdr': 0.5}})])
        with pytest.raises(ValueError) as excinfo:
            check_layer_freshness(path, changed, 'delivery')
        message = str(excinfo.value)
        assert 'delivery.max_sdr' in message
        assert 'built with 0.8' in message
        assert 'now 0.5' in message
        assert 'sha256' not in message

    def test_strict_false_warns_and_returns_false(self, raster, caplog):
        built = resolve_parameters([])
        path = raster('sdr', parameter_tags(built, 'delivery'))
        changed = resolve_parameters(
            [('project', {'delivery': {'max_sdr': 0.5}})])
        with caplog.at_level('WARNING'):
            fresh = check_layer_freshness(
                path, changed, 'delivery', strict=False)
        assert fresh is False
        assert 'built with different' in caplog.text

    def test_an_unrelated_change_does_not_flag_the_layer(self, raster):
        """The whole point of naming consumed paths: a layer must not go
        stale because something it never depended on moved."""
        built = resolve_parameters([])
        path = raster('sdr', parameter_tags(built, 'delivery'))
        unrelated = resolve_parameters(
            [('project', {'erosion': {'support_practice_factor': 0.5}})])
        assert check_layer_freshness(path, unrelated, 'delivery') is True

    def test_a_sibling_leaf_does_not_flag_the_layer(self, raster):
        """topography holds two leaves building two different files."""
        built = resolve_parameters([])
        path = raster('ls', parameter_tags(
            built, 'topography.max_slope_length_m'))
        moved = resolve_parameters([
            ('project', {'topography': {'headwater_threshold_m2': 50000}})])
        assert check_layer_freshness(
            path, moved, 'topography.max_slope_length_m') is True

    def test_an_untagged_layer_is_not_reported_as_stale(self, raster,
                                                        caplog):
        """Layers written before tagging existed would otherwise make
        every existing project fail on upgrade."""
        with caplog.at_level('INFO'):
            fresh = check_layer_freshness(
                raster('old'), resolve_parameters([]), 'delivery')
        assert fresh is True
        assert 'predates layer tagging' in caplog.text


class TestKnownLimits:

    def test_the_digest_covers_parameters_only(self, raster):
        """Stated so nobody trusts the check further than it goes: a
        re-extracted DEM or a hand-edited C factor changes nothing here.
        Staleness is one edge of a dependency graph, not the whole of it.
        """
        import fire_impacts.provenance as prov
        assert 'parameters only' in prov.__doc__

    def test_a_tampered_record_is_rejected_on_read(self):
        """ParameterRecord.from_dict used to discard the stored digest and
        recompute it, so a record whose digest did not describe its own
        values read back as self-consistent."""
        from fire_impacts.params import ParameterRecord
        data = resolve_parameters([]).to_dict()
        data['digest'] = 'sha256:' + '0' * 64
        with pytest.raises(ValueError, match='inconsistent'):
            ParameterRecord.from_dict(data)

    def test_verification_can_be_waived(self):
        from fire_impacts.params import ParameterRecord
        data = resolve_parameters([]).to_dict()
        data['digest'] = 'sha256:' + '0' * 64
        record = ParameterRecord.from_dict(data, verify=False)
        assert record.parameters == ModelParameters()
