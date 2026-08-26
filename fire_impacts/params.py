"""
Calibration parameters: the tunable coefficients of the fire-impacts models.

Every value here is something a user may legitimately change for their
catchment or region. Unit conversions and the fixed coefficients of published
equations (the McCool slope factors, the Brown & Foster kinetic-energy form)
are *not* parameters and stay in :mod:`fire_impacts.const`.

Parameters resolve through five layers, most specific winning:

1. the dataclass defaults below,
2. ``<project>/parameters.json``,
3. ``Catchments/<c>/parameters.json``,
4. ``Catchments/<c>/Events/<event>/event.json`` (the ``parameters`` key),
5. a call-site override.

Each group declares the scope of the output it controls, and every persisted
layer is checked against it on read and on write, so a catchment-scoped value
cannot be set per event where it would rewrite layers its siblings share.

:meth:`fire_impacts.context.RunContext.parameters` performs that merge and
returns a :class:`ParameterRecord` carrying the resolved values, where each
one came from, and a digest for staleness detection.

Design notes: ``design-notes/calibration-parameters-proposal.md``.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import numbers
import types
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Union, get_args, get_origin, get_type_hints

from .const import (
    DEFAULT_DNBR_SATURATION, DEFAULT_DNBR_SEVERITY_THRESHOLD,
    DEFAULT_DEBRIS_DNBR_THRESHOLD, DEFAULT_I12_LOOKUP,
    DEFAULT_KE_RATE_RUSLE2, UNSET,
)

__all__ = [
    'FireAdjustmentParams',
    'DeliveryParams',
    'TopographyParams',
    'ErosionParams',
    'DebrisDepthParams',
    'DebrisFlowParams',
    'SeverityParams',
    'ModelParameters',
    'ParameterRecord',
    'resolve_parameters',
    'nest_overrides',
    'sparse_overrides',
    'SCOPES',
    'scope_of',
    'check_scope',
    'as_record',
    'deprecated_overrides',
]

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
# A parameter may only be overridden at a layer at least as broad as the scope
# of the output it controls. Setting a catchment-scoped value per event is
# either a no-op (the catchment-level producer never sees the event layer) or a
# corruption (an event-scoped producer rewrites a catchment-level file that a
# sibling event's outputs were derived from). Both are silent, so the scope is
# declared on the data and enforced when overrides are persisted.
#
# Broadest to narrowest. A leaf declared at scope X is settable at any layer L
# with SCOPES.index(L) <= SCOPES.index(X).
SCOPES = ('project', 'catchment', 'event', 'run')

# Group-level default, overridable per field via
# field(metadata={'scope': ...}) where a group spans scopes.
_DEFAULT_SCOPE = 'event'


def _article(word: str) -> str:
    """Return 'an' or 'a' to suit the next word."""
    return 'an' if word[:1].lower() in 'aeiou' else 'a'


def _require(condition: bool, message: str) -> None:
    """Raise ValueError(message) unless condition holds."""
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------
# Each group maps onto the pipeline stage that consumes it, so a group
# corresponds one-to-one with the outputs it produces.

@dataclass(frozen=True)
class FireAdjustmentParams:
    """
    Fire-adjusted C and K factors.

    Consumed by ``pre.rusle.compute_adjusted_k_c``; produces
    ``Events/<event>/Erodibility/{C,K}_factor_adjusted_<t>.tif``.

    C and K recover exponentially towards their unburnt baselines with a time
    constant of ``<scale> * AI``, where AI is the aridity index. The scales
    must be positive: zero or negative removes or inverts the fire signal
    (see the note on aridity validity in compute_adjusted_k_c).
    """

    __scope__ = 'event'

    c_peak: float = 0.35        # was Cpeak
    k_fire: float = 0.081       # was Kfire
    c_recovery_scale: float = 0.4   # was x_c
    k_recovery_scale: float = 1.0   # was x_k
    dnbr_saturation: float = float(DEFAULT_DNBR_SATURATION)
    # Writes the *catchment*-scoped C_factor.tif, not an event layer, so it
    # cannot vary per event: compute_adjusted_k_c runs per event but this
    # output is shared, and a per-event value would have the last event to
    # run silently rewrite the base layer every other event's adjusted
    # factors and baseline SDR were derived from.
    default_c_factor: float = field(
        default=0.01, metadata={'scope': 'catchment'})

    def __post_init__(self):
        _require(
            self.c_recovery_scale > 0,
            f'c_recovery_scale must be > 0, got {self.c_recovery_scale}. '
            'It scales the aridity index in the C-factor recovery time '
            'constant; zero or negative removes the fire signal entirely.',
        )
        _require(
            self.k_recovery_scale > 0,
            f'k_recovery_scale must be > 0, got {self.k_recovery_scale}. '
            'It scales the aridity index in the K-factor recovery time '
            'constant; zero or negative removes the fire signal entirely.',
        )
        _require(
            self.dnbr_saturation > 0,
            f'dnbr_saturation must be > 0, got {self.dnbr_saturation}.',
        )
        _require(
            self.default_c_factor > 0,
            f'default_c_factor must be > 0, got {self.default_c_factor}. '
            'A zero C factor gives zero erosion everywhere.',
        )
        _require(
            self.c_peak > 0,
            f'c_peak must be > 0, got {self.c_peak}.',
        )
        _require(
            self.k_fire > 0,
            f'k_fire must be > 0, got {self.k_fire}.',
        )


@dataclass(frozen=True)
class DeliveryParams:
    """
    Sediment delivery ratio and the connectivity index behind it.

    Consumed by ``pre.rusle.compute_sediment_delivery_ratio``; produces
    ``Delivery/SDR_<suffix>.tif`` and the IC/Dup/Ddn intermediates.

    Catchment-scoped, because the same fields drive both the per-event
    ``Events/<e>/Delivery/SDR_<t>.tif`` *and* the catchment-scoped
    ``Delivery/SDR_baseline.tif``. Narrowing this group to event scope
    requires moving the baseline SDR under the event first.
    """

    __scope__ = 'catchment'

    max_sdr: float = 0.8
    ic0: float = 0.5
    k: float = 1.0
    stream_area_threshold_m2: float = 1.3e4
    min_slope: float = 0.005
    max_slope: float = 1.0
    min_c_factor: float = 0.001

    def __post_init__(self):
        _require(
            0 < self.max_sdr <= 1,
            f'max_sdr must be greater than 0 and at most 1, got '
            f'{self.max_sdr}. It is the ceiling of a delivery *ratio*.',
        )
        _require(
            self.k != 0,
            'k must be non-zero: it is the divisor in the IC-SDR logistic '
            'and zero gives a division by zero.',
        )
        _require(
            self.min_slope < self.max_slope,
            f'min_slope ({self.min_slope}) must be < max_slope '
            f'({self.max_slope}).',
        )
        _require(
            self.min_slope > 0,
            f'min_slope must be > 0, got {self.min_slope}. It is a divisor '
            'in the downslope connectivity component.',
        )
        _require(
            self.min_c_factor > 0,
            f'min_c_factor must be > 0, got {self.min_c_factor}. It is a '
            'divisor in the downslope connectivity component.',
        )
        _require(
            self.stream_area_threshold_m2 > 0,
            'stream_area_threshold_m2 must be > 0, got '
            f'{self.stream_area_threshold_m2}.',
        )


@dataclass(frozen=True)
class TopographyParams:
    """
    Terrain-derived layers.

    ``headwater_threshold_m2`` is consumed by
    ``pre.topography.extract_headwaters``; ``max_slope_length_m`` caps the
    specific catchment area in the LS factor (``pre.rusle.compute_lsi``) to
    avoid overestimation in heterogeneous landscapes.
    """

    __scope__ = 'catchment'

    headwater_threshold_m2: float = 20000.0
    max_slope_length_m: float = 141.0

    def __post_init__(self):
        _require(
            self.headwater_threshold_m2 > 0,
            'headwater_threshold_m2 must be > 0, got '
            f'{self.headwater_threshold_m2}.',
        )
        _require(
            self.max_slope_length_m > 0,
            'max_slope_length_m must be > 0, got '
            f'{self.max_slope_length_m}.',
        )


@dataclass(frozen=True)
class ErosionParams:
    """
    RUSLE erosion simulation.

    Consumed by ``sim.rusle``; produces the run's ``Results/`` outputs.

    ``dnbr_severity_threshold`` splits low- from high-severity cells for
    reporting. Like every dNBR threshold in the package it is on the
    conventional 0-1000 scale — see ``const.DNBR_SCALE``, and read dNBR
    through ``pre.util.read_dnbr_*`` so the comparison is on that scale.

    ``kinetic_energy_coefficient`` is the rate constant *k* in the unit
    kinetic-energy relation ``0.29 * [1 - 0.72 * exp(-k * i)]``. It governs
    only how fast energy climbs from the drizzle floor to the asymptote;
    the other two coefficients fix those ends and are not parameters.

    The default 0.082 is the RUSLE2 value (McGregor et al. 1995, adopted
    by USDA-ARS 2013), not a transcription of Brown & Foster's 0.05 — see
    ``const.DEFAULT_KE_RATE_RUSLE2``. Setting it to
    ``const.DEFAULT_KE_RATE_RUSLE`` selects the older RUSLE formulation,
    which yields roughly 20% less unit energy around 10 mm/h.
    """

    __scope__ = 'run'

    support_practice_factor: float = 1.0
    dnbr_severity_threshold: float = float(DEFAULT_DNBR_SEVERITY_THRESHOLD)
    kinetic_energy_coefficient: float = DEFAULT_KE_RATE_RUSLE2

    def __post_init__(self):
        _require(
            0 < self.support_practice_factor <= 1,
            'support_practice_factor (RUSLE P) must be greater than 0 and '
            f'at most 1, got {self.support_practice_factor}. 1.0 means no '
            'support practice.',
        )
        _require(
            self.dnbr_severity_threshold >= 0,
            'dnbr_severity_threshold must be >= 0, got '
            f'{self.dnbr_severity_threshold}.',
        )
        _require(
            self.kinetic_energy_coefficient > 0,
            'kinetic_energy_coefficient must be > 0, got '
            f'{self.kinetic_energy_coefficient}.',
        )


@dataclass(frozen=True)
class DebrisDepthParams:
    """
    Erosion/deposition depth regression for one flow regime.

    Depths are ``ae * (gradient * area) ** be`` for erosion and
    ``ad * (gradient * area) ** bd`` for deposition; ``rock`` is the rock
    fraction of the mobilised material.
    """

    ae: float
    be: float
    ad: float
    bd: float
    rock: float

    def __post_init__(self):
        _require(self.ae > 0, f'ae must be > 0, got {self.ae}.')
        _require(self.ad > 0, f'ad must be > 0, got {self.ad}.')
        _require(
            0 <= self.rock <= 1,
            f'rock must be a fraction in [0, 1], got {self.rock}.',
        )


@dataclass(frozen=True)
class DebrisFlowParams:
    """
    Debris flow transport through headwater catchments.

    Consumed by ``sim.debris``; produces the run's ``DebrisFlow/`` outputs.

    ``num_sim_years`` is coupled to the Year1/Year2 structure of the I12
    critical-intensity lookup table, so raising it above 2 has no effect
    until that table is generalised.

    ``i12_lookup`` names the hydrogeomorphic-hazard table that maps
    (aridity, dNBR, years since fire, slope gradient) onto a critical
    12-minute rainfall intensity — the debris-flow triggering model
    itself. A bare filename resolves against the packaged tables; any
    path containing a separator is used as given, so an alternative table
    can be supplied without repackaging. It is a filename rather than a
    DataFrame because parameters have to survive a JSON round trip and
    appear in the provenance digest; callers holding a DataFrame can pass
    it straight to ``debris_flow_load``.
    """

    __scope__ = 'run'

    hillslope: DebrisDepthParams = DebrisDepthParams(
        ae=4.5e-4, be=0.36, ad=0.3 * 4.5e-4, bd=0.36, rock=0.12,
    )
    channel: DebrisDepthParams = DebrisDepthParams(
        ae=4.1e-4, be=0.52, ad=3.7e-7, bd=1.06, rock=0.45,
    )
    hillslope_area_m2: float = 1.3e4
    channelised_flow_threshold_m2: float = 1.4e7
    sediment_bulk_density: float = 1270.0
    rock_bulk_density: float = 2220.0
    dnbr_threshold: float = float(DEFAULT_DEBRIS_DNBR_THRESHOLD)
    num_sim_years: int = 2
    i12_lookup: str = DEFAULT_I12_LOOKUP

    def __post_init__(self):
        _require(
            self.hillslope_area_m2 < self.channelised_flow_threshold_m2,
            f'hillslope_area_m2 ({self.hillslope_area_m2}) must be < '
            f'channelised_flow_threshold_m2 '
            f'({self.channelised_flow_threshold_m2}); they bound the '
            'hillslope and channelised zones in order.',
        )
        _require(
            self.sediment_bulk_density > 0,
            'sediment_bulk_density must be > 0, got '
            f'{self.sediment_bulk_density}.',
        )
        _require(
            self.rock_bulk_density > 0,
            f'rock_bulk_density must be > 0, got {self.rock_bulk_density}.',
        )
        _require(
            self.num_sim_years >= 1,
            f'num_sim_years must be >= 1, got {self.num_sim_years}.',
        )
        _require(
            bool(self.i12_lookup),
            'i12_lookup must name a lookup table; it is the debris-flow '
            'triggering model and has no meaningful empty value.',
        )


@dataclass(frozen=True)
class SeverityParams:
    """
    Fire-severity imagery acquisition.

    Not calibration in the modelling sense — these do not change the
    equations — but they change the dNBR the whole pipeline is built on, so
    they are recorded for provenance and exposed for reproducibility.
    """

    __scope__ = 'event'

    max_cloud_cover: float = 20.0
    resolution_m: float = 20.0
    pre_fire_window_days: int = 90
    post_fire_window_days: int = 90
    bbox_buffer_km: float = 10.0
    force_sensor: str | None = None
    natural_veg_code: int = 112

    def __post_init__(self):
        _require(
            0 <= self.max_cloud_cover <= 100,
            'max_cloud_cover is a percentage in [0, 100], got '
            f'{self.max_cloud_cover}.',
        )
        _require(
            self.resolution_m > 0,
            f'resolution_m must be > 0, got {self.resolution_m}.',
        )
        _require(
            self.pre_fire_window_days > 0,
            'pre_fire_window_days must be > 0, got '
            f'{self.pre_fire_window_days}.',
        )
        _require(
            self.post_fire_window_days > 0,
            'post_fire_window_days must be > 0, got '
            f'{self.post_fire_window_days}.',
        )
        _require(
            self.bbox_buffer_km >= 0,
            f'bbox_buffer_km must be >= 0, got {self.bbox_buffer_km}.',
        )
        _require(
            self.force_sensor in (None, 'landsat', 'sentinel'),
            "force_sensor must be None, 'landsat' or 'sentinel', got "
            f'{self.force_sensor!r}.',
        )


@dataclass(frozen=True)
class ModelParameters:
    """The full parameter set: every group, fully resolved."""

    fire_adjustment: FireAdjustmentParams = FireAdjustmentParams()
    delivery: DeliveryParams = DeliveryParams()
    topography: TopographyParams = TopographyParams()
    erosion: ErosionParams = ErosionParams()
    debris: DebrisFlowParams = DebrisFlowParams()
    severity: SeverityParams = SeverityParams()

    # -- Serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """Return a nested, JSON-serialisable dict of every value."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelParameters':
        """
        Build from a nested dict, rejecting unknown keys.

        Partial dicts are fine — anything absent keeps its default. Unknown
        group or field names raise ValueError rather than being ignored: a
        typo that silently did nothing would be a worse failure than the
        hard-coded values this replaces.
        """
        return _from_dict(cls, data or {}, path='')

    # -- Overrides -------------------------------------------------------

    def replace(self, **overrides) -> 'ModelParameters':
        """
        Return a new ModelParameters with dotted overrides applied.

        Accepts ``group__field=value`` (double underscore, keyword-safe) or
        ``{'group.field': value}`` via ``**{...}``, to any depth — nested
        groups are reachable as ``debris__hillslope__ae=5e-4``. Rebuilds
        through the constructors so every ``__post_init__`` re-runs; this
        must never use ``object.__setattr__``, which would bypass
        validation.
        """
        if not overrides:
            return self
        merged = _deep_merge(self.to_dict(), nest_overrides(overrides))
        return type(self).from_dict(merged)

    # -- Identity --------------------------------------------------------

    def digest(self) -> str:
        """
        Return a stable ``sha256:...`` digest over the resolved values.

        Used to detect that derived layers were built with different
        parameters than the run now resolves.
        """
        return _digest(self.to_dict())

    def group_digest(self, *paths: str) -> str:
        """
        Return a digest over a subset of the values.

        Each path may name a whole group ('delivery') or a single leaf
        ('topography.max_slope_length_m'). Leaf paths matter because the
        groups do not line up with the producers: `topography` holds
        `headwater_threshold_m2`, which builds Headwaters.*, alongside
        `max_slope_length_m`, which builds LS_factor.tif. Digesting the
        whole group would flag the LS factor stale whenever the headwater
        threshold moved.
        """
        if not paths:
            raise ValueError(
                'group_digest() needs at least one path: with none it '
                'hashes an empty dict, which is the same value for every '
                'parameter set and would make a staleness check always '
                'report "unchanged".'
            )
        return _digest(self.subset(*paths))

    def subset(self, *paths: str) -> dict:
        """Return a nested dict of just the named groups and leaves."""
        flat = _flatten(self.to_dict())
        selected = {}
        for path in paths:
            scope_of(path)          # validates the name, raises if unknown
            matched = {
                key: value for key, value in flat.items()
                if key == path or key.startswith(f'{path}.')
            }
            if not matched:
                raise ValueError(
                    f'Parameter path {path!r} names a group rather than a '
                    f'value, but matched nothing. Known: {sorted(flat)}'
                )
            selected.update(matched)
        return nest_overrides(selected)


# ---------------------------------------------------------------------------
# Resolution and provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterRecord:
    """
    A resolved parameter set plus where each value came from.

    This is what gets written into ``event.json``, the run's
    ``parameters.json``, and the ensemble ``manifest.json``. ``sources`` maps
    a dotted path to the layer that supplied it ('default', 'project',
    'event' or 'call'), which is what makes the record readable months later:
    it distinguishes a deliberate 0.5 from a defaulted one.
    """

    parameters: ModelParameters
    sources: dict
    # Excluded from equality: two records of the same resolution differ only
    # by when they were taken, and comparing those makes == useless.
    version: str = field(compare=False, default='unknown')
    resolved_at: str = field(compare=False, default='')

    def digest(self) -> str:
        """The digest of the resolved values.

        A method, not a property, so that ``record.digest()`` and
        ``record.parameters.digest()`` are spelled the same way — one
        attribute hop apart, they were easy to confuse.
        """
        return self.parameters.digest()

    def to_dict(self) -> dict:
        """Return the JSON-serialisable provenance record."""
        return {
            'fire_impacts_version': self.version,
            'resolved_at': self.resolved_at,
            'values': self.parameters.to_dict(),
            'sources': dict(sorted(self.sources.items())),
            'digest': self.digest(),
        }

    @classmethod
    def from_dict(cls, data: dict, *, verify: bool = True
                  ) -> 'ParameterRecord':
        """
        Rebuild a record written by :meth:`to_dict`.

        The stored digest is checked against the stored values unless
        ``verify=False``. Previously it was discarded and silently
        recomputed, which meant a record whose digest did not describe
        its own values — a stale one left behind by a partial write, or a
        hand-edited one — read back as self-consistent, and any staleness
        check built on it was trusting a number nothing verified.
        """
        record = cls(
            parameters=ModelParameters.from_dict(data.get('values', {})),
            sources=dict(data.get('sources', {})),
            version=data.get('fire_impacts_version', 'unknown'),
            resolved_at=data.get('resolved_at', ''),
        )
        stored = data.get('digest')
        if verify and stored is not None and stored != record.digest():
            raise ValueError(
                f'Parameter record is inconsistent: its stored digest '
                f'{stored} does not match a digest of its own values '
                f'({record.digest()}). The file was probably written '
                f'partially or edited by hand. Pass verify=False to read '
                f'it anyway.'
            )
        return record

    def restricted_to_scope(self, scope: str) -> 'ParameterRecord':
        """
        Return a copy carrying only leaves settable at ``scope`` or broader.

        Used when recording provenance beside outputs written at a scope
        broader than the step that produced them. ``compute_adjusted_k_c``
        runs per event but also writes catchment-level layers (the base C/K
        factors, the LS factor, the baseline SDR); recording its *full*
        resolution there would overwrite that file on every event and make
        its digest flip on purely event-scoped changes, giving false
        staleness positives on every event switch.

        Values outside the scope are dropped back to the package default,
        so the digest depends only on what could have influenced the
        outputs at that scope.
        """
        if scope not in SCOPES:
            raise ValueError(
                f'Unknown scope {scope!r}. Valid scopes: {list(SCOPES)}.')
        limit = SCOPES.index(scope)
        keep = {
            path: value
            for path, value in _flatten(self.parameters.to_dict()).items()
            if SCOPES.index(scope_of(path)) <= limit
        }
        return ParameterRecord(
            parameters=ModelParameters.from_dict(nest_overrides(keep)),
            sources={
                path: self.sources.get(path, 'default') for path in keep
            },
            version=self.version,
            resolved_at=self.resolved_at,
        )

    def sources_for(self, layer: str) -> list:
        """Return the dotted paths that came from a given layer, sorted.

        ``record.sources_for('default')`` answers "what did nobody choose?"
        """
        return sorted(k for k, v in self.sources.items() if v == layer)


def as_record(params, *, layer: str = 'call') -> ParameterRecord:
    """
    Normalise whatever a caller passed as ``params`` into a ParameterRecord.

    Model functions need a record, not a bare :class:`ModelParameters`: they
    write a provenance record, and a bare parameter set carries no ``sources``
    to write. Passing one is still allowed as a convenience — every leaf is
    then attributed to ``layer`` ('call' by default), because an explicitly
    constructed parameter set is by definition the caller's choice.

    Parameters:
    - params: a ParameterRecord (returned unchanged) or a ModelParameters.
    - layer: the source to attribute a bare ModelParameters to.

    Returns:
    - A ParameterRecord.
    """
    if isinstance(params, ParameterRecord):
        return params
    if isinstance(params, ModelParameters):
        return ParameterRecord(
            parameters=params,
            sources={path: layer for path in _flatten(params.to_dict())},
            version=_package_version(),
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
    raise TypeError(
        'params must be a ParameterRecord (from ctx.parameters()) or a '
        f'ModelParameters, got {type(params).__name__}.'
    )


def deprecated_overrides(mapping: dict) -> dict:
    """
    Turn supplied deprecated kwargs into a call-layer override dict.

    Parameters:
    - mapping: ``{'delivery.max_sdr': <value or const.UNSET>, ...}``, i.e.
      the dotted path each legacy kwarg corresponds to, mapped to whatever
      the caller passed.

    Returns:
    - A flat ``{dotted_path: value}`` dict of only the entries actually
      supplied, ready to splat into
      :meth:`~fire_impacts.context.RunContext.resolve_parameters`. Entries
      left as ``const.UNSET`` are omitted, so a legacy kwarg whose value
      happens to equal the package default is still honoured as an
      explicit choice.

    Emits a DeprecationWarning naming the replacement for each one supplied.
    """
    supplied = {
        path: value for path, value in mapping.items() if value is not UNSET
    }
    for path in sorted(supplied):
        warnings.warn(
            f'Passing this value as a keyword argument is deprecated; use '
            f'the parameter system instead — e.g. '
            f'ctx.parameters({path.replace(".", "__")}=...), or set it in '
            f'a parameters.json. See the Calibration parameters section of '
            f'the README.',
            DeprecationWarning,
            stacklevel=3,
        )
    return supplied


def resolve_parameters(layers, *, version: str = None) -> ParameterRecord:
    """
    Merge parameter layers in increasing precedence and record the origin.

    Parameters:
    - layers: sequence of ``(source_name, dict)`` pairs, lowest precedence
      first. A None or empty dict is skipped, so callers can pass optional
      layers without filtering. Typically::

          [('project', project_dict), ('event', event_dict),
           ('call', override_dict)]

    - version: fire_impacts version string recorded in the result. Defaults
      to the installed package version.

    Returns:
    - A ParameterRecord. Any leaf no layer supplied is marked 'default'.
    """
    merged: dict = {}
    origins: dict = {}
    for name, data in layers:
        if not data:
            continue
        flat = _flatten(data)
        merged = _deep_merge(merged, data)
        for path in flat:
            origins[path] = name

    parameters = ModelParameters.from_dict(merged)

    # Anything no layer touched came from the dataclass defaults.
    sources = {
        path: origins.get(path, 'default')
        for path in _flatten(parameters.to_dict())
    }
    return ParameterRecord(
        parameters=parameters,
        sources=sources,
        version=version or _package_version(),
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _package_version() -> str:
    """Return the installed fire_impacts version, or 'unknown'."""
    try:
        from importlib.metadata import version
        return version('fire_impacts')
    except Exception:
        return 'unknown'


def _to_dict(obj: Any) -> Any:
    """Recursively convert nested frozen dataclasses to plain dicts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _did_you_mean(name: str, options) -> str:
    """Return a ' Did you mean X?' suffix, or '' if nothing is close."""
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f' Did you mean {close[0]!r}?' if close else ''


def _from_dict(cls, data: dict, *, path: str):
    """
    Build a (possibly nested) dataclass from a dict, rejecting unknown keys.

    Nested dataclass fields recurse; everything else is passed through to the
    constructor so __post_init__ validation runs.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f'{path or "parameters"}: expected an object, got '
            f'{type(data).__name__}.'
        )
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in known:
            where = f'{path}{key}' if path else key
            raise ValueError(
                f'Unknown parameter {where!r}.'
                f'{_did_you_mean(key, known)} '
                f'Valid names here: {sorted(known)}.'
            )
        default = known[key].default
        if is_dataclass(default) and not isinstance(default, type):
            # Merge onto the default instance so a partial override of a
            # nested group works even when that group's own fields have no
            # defaults (DebrisDepthParams requires all five).
            if not isinstance(value, dict):
                raise ValueError(
                    f'{path}{key}: expected an object, got '
                    f'{type(value).__name__}.'
                )
            kwargs[key] = _from_dict(
                type(default),
                _deep_merge(_to_dict(default), value),
                path=f'{path}{key}.',
            )
        else:
            kwargs[key] = _coerce(
                value, _hints(cls).get(key), f'{path}{key}',
            )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        # A nested group with no default (DebrisDepthParams) needs every
        # field; surface that as a parameter error rather than a TypeError.
        raise ValueError(f'{path or "parameters"}: {exc}') from exc
    except ValueError as exc:
        # __post_init__ range errors name the field but not the group, and
        # a user editing one of three parameters.json files needs the full
        # path to know what to change.
        message = str(exc)
        prefix = path or ''
        if prefix and not message.startswith(prefix):
            message = f'{prefix}{message}'
        raise ValueError(message) from None


def scope_of(path: str) -> str:
    """
    Return the scope of a dotted parameter path, e.g. 'delivery.max_sdr'.

    Falls back to the group's ``__scope__``; a field may narrow or broaden
    it with ``field(metadata={'scope': ...})``. A path naming a group with
    no field ('delivery') returns the group's scope.

    Raises ValueError for an unknown group or field.
    """
    parts = [p for p in path.split('.') if p]
    if not parts:
        raise ValueError('Empty parameter path.')

    cls = ModelParameters
    scope = _DEFAULT_SCOPE
    for depth, part in enumerate(parts):
        known = {f.name: f for f in fields(cls)}
        if part not in known:
            raise ValueError(
                f'Unknown parameter {".".join(parts[:depth + 1])!r}.'
                f'{_did_you_mean(part, known)} '
                f'Valid names here: {sorted(known)}.'
            )
        spec = known[part]
        default = spec.default
        nested = is_dataclass(default) and not isinstance(default, type)
        if nested:
            scope = getattr(type(default), '__scope__', scope)
            cls = type(default)
        else:
            scope = spec.metadata.get('scope', scope)
    return scope


def check_scope(data: dict, layer: str) -> None:
    """
    Raise if any leaf in an override dict may not be set at ``layer``.

    ``layer`` is one of :data:`SCOPES`. A leaf declared at scope X is
    settable at any layer at least as broad as X, so an event file may not
    carry a catchment-scoped value. Enforced when overrides are persisted;
    call-site overrides are deliberately unrestricted (they are explicit,
    transient, and recorded as 'call').
    """
    if layer not in SCOPES:
        raise ValueError(
            f'Unknown scope {layer!r}. Valid scopes: {list(SCOPES)}.')
    limit = SCOPES.index(layer)
    offenders = []
    # Validate the group names themselves: _flatten yields no leaves for an
    # empty group, so {'delivery': {}} would otherwise slip through where
    # {'delivery': {...}} is refused.
    for group in (data or {}):
        scope_of(group)
    for path in _flatten(data or {}):
        scope = scope_of(path)
        if SCOPES.index(scope) < limit:
            offenders.append((path, scope))
    if offenders:
        detail = '; '.join(
            f'{path!r} is {scope}-scoped' for path, scope in offenders
        )
        wrong = offenders[0]
        raise ValueError(
            f'Cannot set {len(offenders)} parameter(s) at {layer} scope: '
            f'{detail}. The output each controls is written at that scope, '
            f'so {_article(layer)} {layer}-level value would either be '
            f'ignored or would '
            f'overwrite a file its siblings share. Set {wrong[0]!r} in the '
            f'{wrong[1]}-level parameters.json instead.'
        )


_TYPE_HINTS: dict = {}


def _hints(cls) -> dict:
    """Return (and cache) a dataclass's resolved type hints.

    ``from __future__ import annotations`` makes ``field.type`` a string,
    so the annotations have to be resolved before they can be used to
    coerce.
    """
    if cls not in _TYPE_HINTS:
        _TYPE_HINTS[cls] = get_type_hints(cls)
    return _TYPE_HINTS[cls]


def _coerce(value, annotation, path: str):
    """
    Coerce a JSON value to a field's annotated type, or raise.

    JSON has no int/float distinction, so a hand-edited ``1`` for a float
    field must normalise to ``1.0`` — otherwise it is a different value to
    the digest and would spuriously flag every derived layer stale. A
    string where a number belongs is rejected here rather than leaking a
    comparison TypeError out of __post_init__.
    """
    if annotation is None:
        return value

    # get_origin() returns types.UnionType for `X | None` on Python
    # 3.10-3.13 and typing.Union for typing.Optional[X]; 3.14 unified them.
    # Match both, or this branch is dead on the pinned runtime.
    if get_origin(annotation) in (Union, types.UnionType):
        args = get_args(annotation)
        if value is None and type(None) in args:
            return None
        for member in (a for a in args if a is not type(None)):
            try:
                return _coerce(value, member, path)
            except ValueError:
                continue
        raise ValueError(
            f'{path}: expected one of '
            f'{[getattr(a, "__name__", a) for a in args]}, got '
            f'{type(value).__name__} ({value!r}).'
        )

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f'{path} must be true or false, got {value!r}.')

    if annotation is int:
        # bool is a subclass of int; a JSON true is not a count. numbers.*
        # rather than int/float so numpy and pandas scalars (np.int64 out of
        # a DataFrame lookup) are accepted rather than rejected with a
        # confusing "must be a whole number".
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{path} must be a whole number, got {value!r}.')
        try:
            as_float = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f'{path}: {value!r} is out of range.') from exc
        if not as_float.is_integer():
            raise ValueError(f'{path} must be a whole number, got {value!r}.')
        return int(as_float)

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{path} must be a number, got {value!r}.')
        try:
            return float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f'{path}: {value!r} is out of range.') from exc

    if annotation is str:
        if isinstance(value, str):
            return value
        raise ValueError(f'{path} must be a string, got {value!r}.')

    return value


def sparse_overrides(params: 'ModelParameters') -> dict:
    """
    Reduce a full ModelParameters to only what differs from the defaults.

    An override file should record choices, not the whole parameter set:
    writing every field makes each package default an explicit user
    setting, so ``sources`` reads back as chosen-everywhere and a later
    release that fixes a default is silently overridden by the frozen file.
    """
    defaults = _flatten(ModelParameters().to_dict())
    changed = {
        path: value
        for path, value in _flatten(params.to_dict()).items()
        if value != defaults.get(path)
    }
    return nest_overrides(changed) if changed else {}


def nest_overrides(overrides: dict) -> dict:
    """
    Turn dotted override keys into the nested dict the merge layers use.

    Accepts ``group__field`` and ``group.field`` interchangeably, to any
    depth (``debris__hillslope__ae``). The result carries exactly the keys
    the caller named and the values they gave — deliberately *not* a diff
    against the defaults, so that explicitly passing a value equal to the
    package default still counts as a choice and still overrides a lower
    layer.

    Raises ValueError for a key that names no group (a bare field name).
    """
    nested: dict = {}
    for key, value in overrides.items():
        parts = [p for p in key.replace('__', '.').split('.') if p]
        if len(parts) < 2:
            raise ValueError(
                f'Override key {key!r} must name a group and a field, '
                "e.g. 'delivery__max_sdr' or 'delivery.max_sdr'."
            )
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(
                    f'Override key {key!r} conflicts with another override '
                    'that sets one of its parent groups to a value.'
                )
        leaf = parts[-1]
        if isinstance(cursor.get(leaf), dict):
            raise ValueError(
                f'Override key {key!r} conflicts with another override that '
                f'sets a field inside {leaf!r}.'
            )
        cursor[leaf] = value
    return nested


def _flatten(data: dict, prefix: str = '') -> dict:
    """Flatten a nested dict to {'group.field': value}."""
    flat = {}
    for key, value in data.items():
        full = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f'{full}.'))
        else:
            flat[full] = value
    return flat


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Return a new dict with overlay merged recursively over base."""
    merged = dict(base)
    for key, value in overlay.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _digest(values: dict) -> str:
    """Return a stable sha256 digest over a nested value dict."""
    canonical = json.dumps(values, sort_keys=True, separators=(',', ':'))
    return 'sha256:' + hashlib.sha256(canonical.encode()).hexdigest()
