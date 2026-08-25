"""
Phase 2: pre/rusle.py consuming resolved calibration parameters.

The contract has three parts:
  1. at default parameters the outputs are unchanged (bit-identical),
  2. an override at any layer actually reaches the raster,
  3. what was used is recorded beside the outputs.

Design notes: design-notes/calibration-parameters-proposal.md
"""

import warnings

import numpy as np
import pytest

from fire_impacts import const as c
from fire_impacts.params import DeliveryParams
from fire_impacts.pre import rusle


class TestDeprecatedKwargs:
    """The legacy SDR kwargs default to the same numbers as the dataclass,
    so they need a sentinel to be distinguishable from 'not supplied'."""

    def test_default_constants_track_the_dataclass(self):
        assert rusle.DEFAULT_MAX_SDR == DeliveryParams().max_sdr
        assert rusle.DEFAULT_IC0 == DeliveryParams().ic0
        assert rusle.DEFAULT_K == DeliveryParams().k

    def test_unsupplied_kwargs_are_the_sentinel(self):
        import inspect
        sig = inspect.signature(rusle.compute_sediment_delivery_ratio)
        for name in ('max_sdr', 'ic0', 'k'):
            assert sig.parameters[name].default is c.UNSET, name

    def test_a_supplied_kwarg_becomes_a_call_override(self):
        from fire_impacts.params import deprecated_overrides
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            got = deprecated_overrides({
                'delivery.max_sdr': 0.9, 'delivery.ic0': c.UNSET,
            })
        # Flat dotted keys: the result is splatted into resolve_parameters,
        # which nests it. Returning it pre-nested double-nests.
        assert got == {'delivery.max_sdr': 0.9}

    def test_a_kwarg_equal_to_the_default_is_still_honoured(self):
        # The reason the sentinel exists: 0.8 is also the package default,
        # so a plain default could not tell this from "not supplied", and a
        # lower layer would silently win.
        from fire_impacts.params import deprecated_overrides
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            got = deprecated_overrides({'delivery.max_sdr': 0.8})
        assert got == {'delivery.max_sdr': 0.8}

    def test_supplying_one_warns(self):
        from fire_impacts.params import deprecated_overrides
        with pytest.warns(DeprecationWarning, match='delivery__max_sdr'):
            deprecated_overrides({'delivery.max_sdr': 0.9})

    def test_not_supplying_any_does_not_warn(self):
        from fire_impacts.params import deprecated_overrides
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            assert deprecated_overrides({'delivery.max_sdr': c.UNSET}) == {}


class TestAridityGuard:
    """AI divides the C/K recovery exponent; invalid values fail silently,
    so they are counted and reported."""

    def test_negative_and_zero_aridity_are_reported(self, caplog):
        ai = np.array([[0.5, 0.0, -0.3], [np.nan, 1.0, 2.0]])
        nodata = np.zeros(ai.shape, dtype=bool)
        with caplog.at_level('WARNING'):
            rusle._warn_on_invalid_aridity(ai, nodata)
        assert '1 NaN, 1 zero and 1 negative' in caplog.text

    def test_valid_aridity_is_silent(self, caplog):
        ai = np.array([[0.5, 1.0], [1.5, 2.0]])
        with caplog.at_level('WARNING'):
            rusle._warn_on_invalid_aridity(ai, np.zeros(ai.shape, bool))
        assert 'Aridity has' not in caplog.text

    def test_cells_outside_the_valid_domain_are_ignored(self, caplog):
        # Ocean and water bodies carry negative aridity in the source grids;
        # outside the DEM domain they are harmless.
        ai = np.array([[0.5, -9.0], [1.0, -9.0]])
        nodata = np.array([[False, True], [False, True]])
        with caplog.at_level('WARNING'):
            rusle._warn_on_invalid_aridity(ai, nodata)
        assert 'Aridity has' not in caplog.text

    def test_an_all_nodata_grid_does_not_divide_by_zero(self, caplog):
        ai = np.array([[np.nan, np.nan]])
        with caplog.at_level('WARNING'):
            rusle._warn_on_invalid_aridity(ai, np.ones(ai.shape, bool))
        assert 'Aridity has' not in caplog.text


# NOTE: the behavioural counterpart of these unit tests — proving each
# parameter actually moves the raster it controls — lives in
# tests/test_integration_pipeline.py, where a real DEM is available.
# Asserting on inspect.getsource() was tried and removed: it passed every
# realistic mis-wiring (a parameter read then overwritten on the next line,
# min_slope/max_slope swapped, ic0/k swapped) and failed a pure rename of a
# local, so it constrained the spelling of the code and nothing about the
# model.
