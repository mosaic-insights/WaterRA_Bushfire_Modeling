"""
A grid the simulation has not bothered to compute yet.

RUSLE is separable — erosion in a cell is the timestep's erosivity times a
static per-cell layer — so a simulation that materialised a full raster
every timestep was doing work that a recorder could do once, at the end,
from an accumulated scale. ScaledGrid is that deferral made explicit.

It is deliberately array-like: a recorder that knows nothing about the type
can multiply it, index it and reduce it exactly as before, and simply pays
for the materialisation.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class MaterialisationCounter:
    """
    A tally of how many grids a single run has been forced to compute.

    One per run rather than a module global: replicates run concurrently on
    threads by default, and a shared tally would mix them together.
    """

    __slots__ = ('count',)

    def __init__(self):
        self.count = 0


class ScaledGrid(NDArrayOperatorsMixin):
    """
    A static per-cell layer times a per-timestep scale.

    Parameters:
    - scale: 1-D float array with one entry per rainfall cell. Length 1
      when rainfall is spatially uniform.
    - unit: 2-D per-cell layer, constant for a whole recovery segment.
    - rain_index: None when rainfall is uniform; otherwise a 2-D integer
      array giving each grid cell's rainfall cell.
    - counter: optional MaterialisationCounter to tally against.
    ------------------------------------------------------------------------
    Notes:
    - Immutable. `copy()` returns a materialised, writable ndarray, which
      is what a caller asking for a copy actually wants.
    - Writing in place raises TypeError. Allowing it would segfault: the
      operator mixin routes `+=` through __array_ufunc__ with out=(self,),
      and numpy would write into a temporary built by __array__.
    ------------------------------------------------------------------------
    """

    __slots__ = ('scale', 'unit', 'rain_index', 'counter')

    def __init__(self, scale, unit, rain_index=None, counter=None):
        self.scale = scale
        self.unit = unit
        self.rain_index = rain_index
        self.counter = counter

    def materialise(self):
        """Compute the full grid, and tally that we had to."""
        if self.counter is not None:
            self.counter.count += 1
        if self.rain_index is None:
            return self.scale[0] * self.unit
        return self.scale[self.rain_index] * self.unit

    # -- numpy interoperability --------------------------------------------

    def __array__(self, dtype=None, copy=None):
        out = self.materialise()
        return out if dtype is None else out.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for target in kwargs.get('out', ()):
            if isinstance(target, ScaledGrid):
                raise TypeError(
                    'ScaledGrid is immutable; call .copy() for a '
                    'writable array'
                )
        inputs = tuple(
            i.materialise() if isinstance(i, ScaledGrid) else i
            for i in inputs
        )
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, key):
        return self.materialise()[key]

    def __len__(self):
        return len(self.unit)

    def __bool__(self):
        # NDArrayOperatorsMixin supplies no __bool__, so without this
        # Python would fall back to __len__ and `if grid:` would be True
        # for any grid with rows - exactly the ambiguous-truth-value
        # mistake a real ndarray raises on. Deferring to the
        # materialised array's own __bool__ reproduces that behaviour
        # (and its message) exactly, since it then *is* a real ndarray.
        return bool(self.materialise())

    # -- the parts of the ndarray surface recorders actually touch ---------

    @property
    def shape(self):
        return self.unit.shape

    @property
    def ndim(self):
        return self.unit.ndim

    @property
    def dtype(self):
        return np.result_type(self.scale, self.unit)

    @property
    def flags(self):
        return self.materialise().flags

    def copy(self):
        return self.materialise()

    def reshape(self, *args, **kwargs):
        return self.materialise().reshape(*args, **kwargs)

    def ravel(self):
        return self.materialise().ravel()

    def astype(self, dtype):
        return self.materialise().astype(dtype)
