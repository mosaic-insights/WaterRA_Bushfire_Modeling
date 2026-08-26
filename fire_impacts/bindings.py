"""
Input bindings: where a gridded model input comes from.

A *parameter* answers "what coefficient does the model use". A *binding*
answers "where does this input come from" — the normal pipeline, a
user-supplied raster, a uniform scalar, or a synthetic draw. The two are
kept apart because their validity rules differ: a coefficient has a
range, a binding has a domain, a unit and a provenance.

The rule is **materialise, don't branch**. A binding is resolved once, at
preprocessing time, by writing a real raster to the standard path.
Everything downstream keeps reading that path and needs no knowledge of
where it came from. Teaching every read site to accept a scalar-or-array
would touch dozens of call sites, break the ``read_aligned_like``
contract (which needs a grid to align *to*), lose the ability to open the
input in QGIS, and silently change every zonal statistic.

Resolution differs from parameters in one way that matters: resolving a
binding has a *side effect*, so there is no call-site layer. A binding
override at simulation time would either rewrite a preprocessing artefact
mid-run or be silently ignored. Bindings therefore resolve through
project -> catchment -> event only, and are applied at exactly one point.

Design notes: ``design-notes/calibration-parameters-proposal.md`` section 5.
"""

from __future__ import annotations

import difflib
from dataclasses import MISSING as _MISSING
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from .const import DNBR_SCALE

__all__ = [
    'Derived',
    'Constant',
    'FromFile',
    'SyntheticFire',
    'InputBindings',
    'DNBR_UNITS',
    'DOMAINS',
    'SCOPE_BY_INPUT',
    'binding_from_dict',
    'check_binding_scope',
]

# Accepted units for a dNBR binding. dNBR is *stored* as the raw
# band-ratio difference but *quoted* on the 0-1000 scale, and the two
# differ by 1000x — the exact confusion that had two producers writing
# incompatible rasters. A binding must therefore say which it means;
# there is no safe default.
DNBR_UNITS = {
    # The stored representation: pre-fire NBR minus post-fire NBR, ~[0, 1].
    'dnbr': 1.0,
    # The conventional scale used by every threshold and lookup table.
    'dnbr_x1000': 1.0 / DNBR_SCALE,
}

# The RUSLE cover factor is a dimensionless ratio, so there is only one
# unit — but it is still required, so that every binding reads the same
# way and adding a second scale later cannot silently reinterpret files
# written under the first.
C_FACTOR_UNITS = {'dimensionless': 1.0}

#: Units accepted for each bindable input.
UNITS_BY_INPUT = {
    'dnbr': DNBR_UNITS,
    'c_factor': C_FACTOR_UNITS,
}

#: The scope of the output each input produces — and therefore the
#: narrowest layer it may be set at. C_factor.tif is built once per
#: catchment and shared by every fire in it, so an event-level c_factor
#: binding would rewrite a layer its siblings depend on; masked_dNBR.tif
#: is per event and may vary freely. Same reasoning as the parameter
#: groups' __scope__, and enforced the same way.
SCOPE_BY_INPUT = {
    'dnbr': 'event',
    'c_factor': 'catchment',
}

#: Which variants make sense for each input. A synthetic draw is defined
#: only for fire severity: there is no reference distribution to sample a
#: cover factor from, and silently accepting one would produce a raster
#: of dNBR values in a C-factor file.
VARIANTS_BY_INPUT = {
    'dnbr': ('derived', 'constant', 'file', 'synthetic'),
    'c_factor': ('derived', 'constant', 'file'),
}

# Where a Constant is painted.
DOMAINS = (
    # Inside the catchment boundary. Claims lakes, rock and urban area
    # burned too, which inflates catchment totals — see the warning in
    # the resolver.
    'catchment',
    # Every cell with valid DEM data.
    'dem_valid',
    # Borrow the valid-cell footprint of an existing raster, named as
    # 'mask:<section>/<file>' — e.g. 'mask:FireSeverity/masked_dNBR.tif'
    # to reuse the natural-vegetation mask a previous run produced. That
    # layer has to exist already; the resolver fails clearly if it does
    # not rather than silently falling back to the whole catchment.
    'mask',
)


# ---------------------------------------------------------------------------
# Binding variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Derived:
    """Produce the input through the normal pipeline. The default."""

    SOURCE = 'derived'

    def to_dict(self) -> dict:
        return {'source': self.SOURCE}


@dataclass(frozen=True)
class Constant:
    """
    Paint a uniform value over a domain.

    ``units`` is required: see :data:`DNBR_UNITS`. ``domain`` decides
    which cells are filled — the default fills the catchment, which for
    dNBR asserts that water bodies and bare rock burned as well.
    """

    SOURCE = 'constant'

    value: float
    units: str
    domain: str = 'catchment'

    def __post_init__(self):
        if not self.units:
            raise ValueError(
                'A constant binding needs units. They are required '
                'because an input may be stored on a different scale '
                'from the one it is quoted on — dNBR is, by a factor of '
                f'{DNBR_SCALE} — and nothing about a bare number says '
                'which is meant. Validated against the input it is '
                'attached to.'
            )
        base = self.domain.split(':', 1)[0]
        if base not in DOMAINS:
            raise ValueError(
                f'Unknown domain {self.domain!r}. Valid domains: '
                f'{list(DOMAINS)} (use "mask:<section>/<file>" to borrow '
                f'an existing layer\'s valid cells).'
            )
        if base == 'mask' and ':' not in self.domain:
            raise ValueError(
                'A mask domain must name the layer to borrow, as '
                '"mask:<section>/<file>" — for example '
                '"mask:FireSeverity/masked_dNBR.tif".'
            )

    def to_dict(self) -> dict:
        return {
            'source': self.SOURCE, 'value': self.value,
            'units': self.units, 'domain': self.domain,
        }

    def to_stored_scale(self, input_name: str = 'dnbr') -> float:
        """Return the value converted to the input's stored scale."""
        return self.value * UNITS_BY_INPUT[input_name][self.units]


@dataclass(frozen=True)
class FromFile:
    """
    Use a raster the user supplies.

    ``units`` is required for the same reason as :class:`Constant`, and
    matters more here: a user-supplied dNBR raster on the 0-1000 scale is
    at least as likely as a mis-scaled scalar, and nothing about the file
    reveals which it is.
    """

    SOURCE = 'file'

    path: str
    units: str

    def __post_init__(self):
        if not self.path:
            raise ValueError('A file binding needs a path.')
        if not self.units:
            raise ValueError(
                'A file binding needs units: nothing about a raster says '
                'which scale its values are on.'
            )

    def to_dict(self) -> dict:
        return {'source': self.SOURCE, 'path': self.path,
                'units': self.units}

    def scale_to_stored(self, input_name: str = 'dnbr') -> float:
        """Factor converting this file's values to the stored scale."""
        return UNITS_BY_INPUT[input_name][self.units]


@dataclass(frozen=True)
class SyntheticFire:
    """
    Sample from the empirical dNBR distribution of a reference fire.

    ``random_seed`` may be left None, in which case the resolver draws
    one and records the *effective* seed. Recording None would make the
    provenance record undescriptive of the raster it sits beside — the
    draw could never be reproduced.
    """

    SOURCE = 'synthetic'

    severity: str = 'medium'
    random_seed: int | None = None
    reference_url: str | None = None

    def __post_init__(self):
        if self.random_seed is not None and not isinstance(
                self.random_seed, int):
            raise ValueError(
                f'random_seed must be an integer or None, got '
                f'{self.random_seed!r}.'
            )

    def to_dict(self) -> dict:
        return {
            'source': self.SOURCE, 'severity': self.severity,
            'random_seed': self.random_seed,
            'reference_url': self.reference_url,
        }


_VARIANTS = {
    cls.SOURCE: cls for cls in (Derived, Constant, FromFile, SyntheticFire)
}


def check_binding_scope(data: dict, layer: str) -> None:
    """
    Raise if any binding in an override dict may not be set at ``layer``.

    Mirrors :func:`fire_impacts.params.check_scope`. A binding may only
    be set at a layer at least as broad as the scope of the output it
    produces: a c_factor binding in an event file would have that event
    rewrite the catchment's shared C_factor.tif, which is the cross-event
    corruption the parameter scopes already prevent.
    """
    from .params import SCOPES

    if layer not in SCOPES:
        raise ValueError(
            f'Unknown scope {layer!r}. Valid scopes: {list(SCOPES)}.')
    limit = SCOPES.index(layer)
    offenders = []
    for name in (data or {}):
        if name not in SCOPE_BY_INPUT:
            raise ValueError(
                f'Unknown input {name!r}. Bindable inputs: '
                f'{sorted(SCOPE_BY_INPUT)}.'
            )
        scope = SCOPE_BY_INPUT[name]
        if SCOPES.index(scope) < limit:
            offenders.append((name, scope))
    if offenders:
        detail = '; '.join(
            f'{name!r} is {scope}-scoped' for name, scope in offenders)
        first = offenders[0]
        raise ValueError(
            f'Cannot bind {len(offenders)} input(s) at {layer} scope: '
            f'{detail}. The layer each produces is written at that scope, '
            f'so a {layer}-level binding would overwrite a file its '
            f'siblings share. Bind {first[0]!r} in the {first[1]}-level '
            f'parameters.json instead.'
        )


def validate_binding(binding, input_name: str) -> None:
    """
    Check a binding against the input it is attached to.

    Units and permitted variants depend on the input, which the binding
    itself cannot know — a Constant does not know whether it is painting
    a dNBR or a cover factor. Validating here rather than in the
    variant's __post_init__ keeps the check early without hard-coding
    one input's vocabulary into the union.
    """
    if input_name not in UNITS_BY_INPUT:
        raise ValueError(
            f'Unknown input {input_name!r}. Bindable inputs: '
            f'{sorted(UNITS_BY_INPUT)}.'
        )
    allowed = VARIANTS_BY_INPUT[input_name]
    if binding.SOURCE not in allowed:
        raise ValueError(
            f'A {binding.SOURCE!r} binding is not defined for '
            f'{input_name!r}. Valid sources for it: {list(allowed)}.'
        )
    units = getattr(binding, 'units', None)
    if units is not None and units not in UNITS_BY_INPUT[input_name]:
        raise ValueError(
            f'Unknown units {units!r} for {input_name!r}. Valid units: '
            f'{sorted(UNITS_BY_INPUT[input_name])}.'
        )


def binding_from_dict(data: Any):
    """
    Rebuild a binding from its tagged-union dict.

    Unknown source tags and unknown fields raise rather than being
    ignored, for the same reason parameters do: a binding that silently
    fell back to 'derived' would let a user believe they had substituted
    an input when they had not.
    """
    if is_dataclass(data) and not isinstance(data, type):
        return data
    if data is None:
        return Derived()
    if not isinstance(data, dict):
        raise ValueError(
            f'A binding must be an object with a "source" key, got '
            f'{type(data).__name__}.'
        )
    source = data.get('source')
    if source is None:
        raise ValueError(
            f'A binding needs a "source" key. Valid sources: '
            f'{sorted(_VARIANTS)}.'
        )
    if source not in _VARIANTS:
        close = difflib.get_close_matches(str(source), _VARIANTS, n=1)
        hint = f' Did you mean {close[0]!r}?' if close else ''
        raise ValueError(
            f'Unknown binding source {source!r}.{hint} Valid sources: '
            f'{sorted(_VARIANTS)}.'
        )
    cls = _VARIANTS[source]
    known = {f.name for f in fields(cls)}
    supplied = {k: v for k, v in data.items() if k != 'source'}
    unknown = set(supplied) - known
    if unknown:
        raise ValueError(
            f'Unknown field(s) {sorted(unknown)} for a {source!r} '
            f'binding. Valid fields: {sorted(known)}.'
        )
    try:
        return cls(**supplied)
    except TypeError as exc:
        # A missing required field (units, path) would otherwise surface
        # as a bare constructor TypeError naming __init__.
        raise ValueError(
            f'Incomplete {source!r} binding: {exc}. Required fields: '
            f'{sorted(f.name for f in fields(cls) if f.default is _MISSING)}.'
        ) from None


# ---------------------------------------------------------------------------
# The binding set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputBindings:
    """
    How each substitutable model input is supplied.

    Only dNBR is bindable today. The tree is deliberately not populated
    with inputs that have no resolver: a declared binding nothing
    consumes is the same silent failure as a declared parameter nothing
    reads.
    """

    dnbr: Any = Derived()
    c_factor: Any = Derived()

    def __post_init__(self):
        for spec in fields(self):
            validate_binding(getattr(self, spec.name), spec.name)

    def to_dict(self) -> dict:
        return {name: getattr(self, name).to_dict()
                for name in (f.name for f in fields(self))}

    @classmethod
    def from_dict(cls, data: dict) -> 'InputBindings':
        data = data or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f'Unknown input(s) {sorted(unknown)}. Bindable inputs: '
                f'{sorted(known)}.'
            )
        return cls(**{
            name: binding_from_dict(value) for name, value in data.items()
        })

    def is_default(self) -> bool:
        """True when every input uses the normal pipeline."""
        return all(isinstance(getattr(self, f.name), Derived)
                   for f in fields(self))
