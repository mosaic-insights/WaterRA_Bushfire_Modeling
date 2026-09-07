"""
The lazy grid the simulation hands to recorders: a static per-cell layer
times a per-timestep scale, materialised only when something insists.
"""

import numpy as np
import pytest

from fire_impacts.sim.scaled_grid import MaterialisationCounter, ScaledGrid


def unit_grid():
    u = np.arange(6, dtype=np.float32).reshape(2, 3)
    u[0, 0] = np.nan          # a cell outside the catchment
    return u


def scaled(value=2.0):
    return ScaledGrid(np.array([value]), unit_grid())


class TestMaterialise:

    def test_uniform_scale_multiplies_the_layer(self):
        assert np.allclose(
            scaled(2.0).materialise(), unit_grid() * 2.0, equal_nan=True)

    def test_a_scale_per_rain_cell_is_spread_over_the_grid(self):
        rain_index = np.array([[0, 0, 1], [0, 1, 1]])
        sg = ScaledGrid(np.array([2.0, 10.0]), unit_grid(), rain_index)
        expected = np.array([[np.nan, 2.0, 20.0], [6.0, 40.0, 50.0]])

        assert np.allclose(sg.materialise(), expected, equal_nan=True)

    def test_the_nan_mask_of_the_layer_survives(self):
        assert np.isnan(scaled().materialise()[0, 0])


class TestArrayFacade:
    """A recorder that knows nothing about the type must still work."""

    # scaled() materialises to unit_grid() * 2.0 -
    # [[nan, 2, 4], [6, 8, 10]] - so each case's expected value is
    # computed straight from that, never restated as a magic number.
    _MATERIALISED = unit_grid() * 2.0

    @pytest.mark.parametrize('operation, expected', [
        (lambda sg: np.asarray(sg), _MATERIALISED),
        (lambda sg: sg * 2.0, _MATERIALISED * 2.0),
        (lambda sg: 2.0 * sg, _MATERIALISED * 2.0),
        (lambda sg: sg * np.ones((2, 3)), _MATERIALISED),
        (lambda sg: np.ones((2, 3)) * sg, _MATERIALISED),
        (lambda sg: sg + sg, _MATERIALISED * 2.0),
        (lambda sg: np.maximum(np.zeros((2, 3)), sg), _MATERIALISED),
        (lambda sg: np.where(np.ones((2, 3), dtype=bool), sg, 0),
         _MATERIALISED),
        (lambda sg: np.stack([sg, sg]),
         np.stack([_MATERIALISED, _MATERIALISED])),
        (lambda sg: sg.reshape(-1), _MATERIALISED.reshape(-1)),
        (lambda sg: sg.ravel(), _MATERIALISED.ravel()),
        (lambda sg: sg.astype(np.float32),
         _MATERIALISED.astype(np.float32)),
        (lambda sg: sg[1], _MATERIALISED[1]),
    ])
    def test_numpy_operations_pass_through(self, operation, expected):
        # `sg * 2.0` returning the unscaled layer would pass a bare
        # `is not None` check; comparing against the expected value
        # catches that.
        assert np.allclose(operation(scaled()), expected, equal_nan=True)

    def test_nansum_ignores_the_masked_cell(self):
        # 2 * (1 + 2 + 3 + 4 + 5), with cell [0, 0] masked out.
        assert np.nansum(scaled(2.0)) == pytest.approx(30.0)

    def test_reports_the_layer_shape_and_length(self):
        assert scaled().shape == (2, 3)
        assert scaled().ndim == 2
        assert len(scaled()) == 2

    def test_dtype_combines_scale_and_layer(self):
        assert scaled().dtype == np.float64

    def test_flags_are_readable(self):
        assert scaled().flags.writeable


class TestImmutability:

    def test_copy_gives_a_writable_array(self):
        c = scaled(2.0).copy()
        c += 1.0

        assert c[0, 1] == pytest.approx(3.0)

    def test_copying_does_not_disturb_the_original(self):
        sg = scaled(2.0)
        sg.copy()[1, 1] = 99.0

        assert sg.materialise()[1, 1] == pytest.approx(8.0)

    def test_writing_in_place_raises_rather_than_crashing(self):
        # NDArrayOperatorsMixin routes += through __array_ufunc__ with
        # out=(self,). Without a guard numpy writes into a temporary from
        # __array__ and segfaults the interpreter, so this must raise.
        sg = scaled()
        with pytest.raises(TypeError, match='immutable'):
            np.add(sg, 1.0, out=(sg,))


class TestTruthValue:

    def test_bool_raises_like_a_real_array_would(self):
        # NDArrayOperatorsMixin supplies __len__ but not __bool__, so
        # without one of our own `if grid:` would silently take the
        # truthy branch for any grid with rows instead of raising -
        # exactly the mistake this error exists to catch.
        with pytest.raises(ValueError, match='ambiguous'):
            bool(scaled())


class TestMaterialisationCounter:

    def test_counts_every_materialisation(self):
        counter = MaterialisationCounter()
        sg = ScaledGrid(np.array([2.0]), unit_grid(), counter=counter)
        sg.materialise()
        np.asarray(sg)

        assert counter.count == 2

    def test_a_grid_without_a_counter_still_materialises(self):
        assert scaled().materialise() is not None

    def test_counters_are_independent(self):
        a, b = MaterialisationCounter(), MaterialisationCounter()
        ScaledGrid(np.array([1.0]), unit_grid(), counter=a).materialise()

        assert (a.count, b.count) == (1, 0)
