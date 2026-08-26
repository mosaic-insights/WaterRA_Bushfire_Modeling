"""
Run-context object bundling project + catchment + event + ensemble, and
the per-event definition (fire dates + recovery windows) it resolves.

A FireImpactsProject organises data along three dimensions inside each
catchment: events (fires), ensembles (climate realisations), and the
cartesian-product runs that combine them. Every public preprocessing or
simulation function operates on exactly one fully-specified combination,
encoded as a :class:`RunContext`.

Bulk operations are expressed as loops over contexts produced by
:meth:`RunContext.enumerate_events` (for event-level prep) or
:meth:`RunContext.enumerate_runs` (for simulation, which requires both
an event and an ensemble).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd

from . import const

if TYPE_CHECKING:
    from .pre.project import FireImpactsProject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventDefinition:
    """
    Immutable description of one fire event's recovery modelling setup.

    Attributes:
    - fire_start_date / fire_end_date: pandas Timestamps, loaded from the
      event's FireMeta.csv (written by fire-severity preprocessing, which
      remains the source of truth for the dates).
    - recovery_breakpoints: monotonically increasing years-since-fire
      boundaries. n+1 breakpoints define n contiguous recovery windows;
      window i is [b_i, b_{i+1}) and is modelled at recovery time b_i.

    Only the breakpoints are persisted (to Events/<event>/event.json);
    the dates are carried in memory so the derived-window helpers below
    can resolve absolute dates without a second lookup.
    """

    fire_start_date: object = None
    fire_end_date: object = None
    recovery_breakpoints: list = field(
        default_factory=lambda: list(const.DEFAULT_RECOVERY_BREAKPOINTS)
    )

    # -- Derived recovery structure ------------------------------------------

    def recovery_times(self):
        """Return the window-start recovery times (breakpoints[:-1])."""
        return list(self.recovery_breakpoints[:-1])

    def windows(self):
        """Return [(start, end), ...] window pairs in years since fire."""
        return const.recovery_windows(self.recovery_breakpoints)

    def window_for(self, recovery_time):
        """Return the (start, end) years for the window starting at
        recovery_time. Raises ValueError if recovery_time is not a
        window start."""
        for start, end in self.windows():
            if start == recovery_time:
                return (start, end)
        raise ValueError(
            f'recovery_time {recovery_time} is not a window start in '
            f'breakpoints {self.recovery_breakpoints!r}.'
        )

    def absolute_window(self, recovery_time):
        """Return the (start, end) pandas Timestamps of the window
        starting at recovery_time, measured from fire_end_date."""
        start_years, end_years = self.window_for(recovery_time)
        base = self._base_date('an absolute recovery window')
        return (
            base + pd.DateOffset(days=int(start_years * 365.25)),
            base + pd.DateOffset(days=int(end_years * 365.25)),
        )

    def simulation_period(self):
        """Return the (start, end) pandas Timestamps spanning every
        recovery window, measured from fire_end_date.

        The start is the first breakpoint (usually the fire end date
        itself) and the end is the last breakpoint — i.e. the rainfall
        span the recovery series needs.
        """
        base = self._base_date('a simulation period')
        breakpoints = self.recovery_breakpoints
        return (
            base + pd.DateOffset(days=int(breakpoints[0] * 365.25)),
            base + pd.DateOffset(days=int(breakpoints[-1] * 365.25)),
        )

    def _base_date(self, what):
        """Return fire_end_date as a Timestamp, or raise if unset."""
        if self.fire_end_date is None:
            raise ValueError(
                f'event definition has no fire_end_date; cannot resolve '
                f'{what}.'
            )
        return pd.Timestamp(self.fire_end_date)

    # -- Serialisation -------------------------------------------------------

    def to_dict(self):
        """Return the JSON-serialisable dict written to event.json.

        Only the recovery breakpoints are persisted — the fire dates live
        in the event's FireMeta.csv.
        """
        return {'recovery_breakpoints': list(self.recovery_breakpoints)}

    @classmethod
    def from_dict(cls, data, *, fire_start_date=None, fire_end_date=None):
        """Build an EventDefinition from an event.json dict, taking the
        fire dates from the caller (which reads them from FireMeta.csv).
        Missing breakpoints fall back to the package default."""
        breakpoints = data.get('recovery_breakpoints')
        return cls(
            fire_start_date=fire_start_date,
            fire_end_date=fire_end_date,
            recovery_breakpoints=(
                list(breakpoints) if breakpoints
                else list(const.DEFAULT_RECOVERY_BREAKPOINTS)
            ),
        )


@dataclass(frozen=True)
class RunContext:
    """A (catchment, event, ensemble) tuple at varying degrees of binding.

    Three levels of binding:
    - **catchment-only** (event=None, ensemble=None): static preprocessing
      that does not depend on a fire (DEM, soil, base topography).
    - **event-level** (ensemble=None): per-event preprocessing (fire
      severity, fire-adjusted erodibility, SDR).
    - **run-level** (both set): simulation runs and post-processing.

    Methods that require event or ensemble raise if those fields are
    None, so the context's binding level is enforced at the call site.
    """

    project: 'FireImpactsProject'
    catchment: str
    event: str | None = None
    ensemble: str | None = None
    # Names this run's output directory. Defaults to the ensemble name,
    # so a project that never labels a run has the paths it always had.
    # A label lets several parameter variants of one (event, ensemble)
    # sit side by side without a further level of nesting; the ensemble
    # is recorded in run.json rather than encoded in the path.
    label: str | None = None

    # -- Constructors --------------------------------------------------------

    @classmethod
    def solo_catchment(
        cls,
        project: 'FireImpactsProject',
        *,
        catchment: str | None = None,
    ) -> 'RunContext':
        """Convenience for static catchment-level preprocessing.

        Returns a catchment-only context (no event, no ensemble), used
        for DEM, soil, and base topography preparation. If ``catchment``
        is None and the project has exactly one catchment, that catchment
        is used; otherwise ``catchment`` must be supplied.
        """
        catchment = _resolve_solo_catchment(project, catchment)
        _assert_catchment_registered(project, catchment)
        return cls(project=project, catchment=catchment)

    @classmethod
    def solo_event(
        cls,
        project: 'FireImpactsProject',
        *,
        event: str,
        catchment: str | None = None,
    ) -> 'RunContext':
        """Convenience for the single-catchment case.

        Returns the unique event context for ``event``. If ``catchment``
        is None and the project has exactly one catchment, that catchment
        is used; otherwise ``catchment`` must be supplied.
        """
        catchment = _resolve_solo_catchment(project, catchment)
        _assert_catchment_registered(project, catchment)
        return cls(project=project, catchment=catchment, event=event)

    @classmethod
    def solo_run(
        cls,
        project: 'FireImpactsProject',
        *,
        event: str,
        ensemble: str,
        catchment: str | None = None,
        label: str | None = None,
    ) -> 'RunContext':
        """Convenience for the single-catchment case.

        Returns the unique (event, ensemble) run context. If ``catchment``
        is None and the project has exactly one catchment, that catchment
        is used; otherwise ``catchment`` must be supplied.

        ``label`` names the output directory, defaulting to the ensemble
        name. Give one to keep several parameter variants of the same
        (event, ensemble) side by side.
        """
        catchment = _resolve_solo_catchment(project, catchment)
        _assert_catchment_registered(project, catchment)
        return cls(
            project=project, catchment=catchment,
            event=event, ensemble=ensemble, label=label,
        )

    @classmethod
    def enumerate_catchments(
        cls,
        project: 'FireImpactsProject',
    ) -> Iterator['RunContext']:
        """Yield one catchment-only context per registered catchment.

        For static preprocessing operations that do not depend on a fire
        event (DEM extraction, soil download, base topography).
        """
        for c in project.catchments:
            yield cls(project=project, catchment=c)

    # -- Enumeration ---------------------------------------------------------

    @classmethod
    def enumerate_events(
        cls,
        project: 'FireImpactsProject',
        *,
        catchment: str | None = None,
        event: str | None = None,
    ) -> Iterator['RunContext']:
        """Yield one context per existing (catchment, event) directory.

        Filters are wildcard-style — pass None on any axis to include all
        matches present on disk. Returned contexts have ``ensemble=None``;
        use :meth:`enumerate_runs` for run-level iteration.
        """
        for c in _resolve_catchments(project, catchment):
            events_dir = Path(project.catchment_path(c)) / 'Events'
            if not events_dir.exists():
                continue
            for ev_dir in sorted(events_dir.iterdir()):
                if not ev_dir.is_dir():
                    continue
                if event is not None and ev_dir.name != event:
                    continue
                yield cls(
                    project=project, catchment=c,
                    event=ev_dir.name, ensemble=None,
                )

    @classmethod
    def enumerate_runs(
        cls,
        project: 'FireImpactsProject',
        *,
        catchment: str | None = None,
        event: str | None = None,
        ensemble: str | None = None,
    ) -> Iterator['RunContext']:
        """Yield one context per (catchment, event, ensemble) where both
        sides of the cartesian product exist on disk.

        Specifically: for each catchment, takes the cross product of
        directories under ``Events/`` and ``Ensembles/``. A context is
        only yielded when both the event and ensemble are prepared,
        regardless of whether ``Runs/<event>/<ensemble>/`` has been
        materialised yet — so this represents *runnable* combinations,
        not necessarily *executed* ones.

        Use :func:`fire_impacts.sim.results.list_runs` to discover
        combinations that have already produced output.
        """
        for c in _resolve_catchments(project, catchment):
            events_dir = Path(project.catchment_path(c)) / 'Events'
            ensembles_dir = Path(project.catchment_path(c)) / 'Ensembles'
            if not events_dir.exists() or not ensembles_dir.exists():
                continue
            events = sorted(
                d.name for d in events_dir.iterdir() if d.is_dir()
            )
            ensembles = sorted(
                d.name for d in ensembles_dir.iterdir() if d.is_dir()
            )
            for ev in events:
                if event is not None and ev != event:
                    continue
                for ens in ensembles:
                    if ensemble is not None and ens != ensemble:
                        continue
                    yield cls(
                        project=project, catchment=c,
                        event=ev, ensemble=ens,
                    )

    # -- Validation ----------------------------------------------------------

    def validate(
        self,
        *,
        require_event_dir: bool = True,
        require_ensemble_dir: bool = True,
    ) -> None:
        """Check the context against the filesystem.

        Always checks that the catchment is registered with the project.
        By default also checks that ``Events/<event>/`` and (if
        ``ensemble`` is set) ``Ensembles/<ensemble>/`` exist on disk.
        Disable the directory checks when the calling function is the
        one creating those directories. Event-dir checks are skipped
        automatically when this is a catchment-only context.
        """
        _assert_catchment_registered(self.project, self.catchment)
        if require_event_dir and self.event is not None:
            event_dir = Path(self.project.event_path(
                self.catchment, event=self.event,
            ))
            if not event_dir.exists():
                raise FileNotFoundError(
                    f"Event directory does not exist: {event_dir}. "
                    f"Run the prep notebook for event {self.event!r} "
                    f"first."
                )
        if require_ensemble_dir and self.ensemble is not None:
            ens_dir = Path(self.project.ensemble_path(
                self.catchment, ensemble=self.ensemble,
            ))
            if not ens_dir.exists():
                raise FileNotFoundError(
                    f"Ensemble directory does not exist: {ens_dir}. "
                    f"Prepare rainfall for ensemble {self.ensemble!r} "
                    f"first."
                )

    # -- Path accessors ------------------------------------------------------

    def catchment_path(self, *args) -> str:
        """Resolve a path under this context's catchment folder."""
        return self.project.catchment_path(self.catchment, *args)

    def event_path(self, *args) -> str:
        """Resolve a path under this context's event folder.

        Raises if this context has no event (i.e. it's a catchment-only
        context — use solo_event / enumerate_events to bind an event).
        """
        if self.event is None:
            raise ValueError(
                'This context has no event; event_path() requires an '
                'event-level context.'
            )
        return self.project.event_path(
            self.catchment, *args, event=self.event,
        )

    def ensemble_path(self, *args) -> str:
        """Resolve a path under this context's ensemble folder.

        Raises if this context has no ensemble (use an
        :meth:`enumerate_runs` context instead of an
        :meth:`enumerate_events` one).
        """
        if self.ensemble is None:
            raise ValueError(
                'This context has no ensemble; ensemble_path() '
                'requires a run-level context.'
            )
        return self.project.ensemble_path(
            self.catchment, *args, ensemble=self.ensemble,
        )

    @property
    def run_label(self) -> str:
        """The name of this run's output directory.

        The label when one is set, otherwise the ensemble name.
        """
        return self.label or self.ensemble

    def run_path(self, *args) -> str:
        """Resolve a path under this context's run folder.

        Named by :attr:`run_label`. Raises if this context has no
        ensemble.
        """
        if self.ensemble is None:
            raise ValueError(
                'This context has no ensemble; run_path() requires a '
                'run-level context.'
            )
        return self.project.run_path(
            self.catchment, *args,
            event=self.event, ensemble=self.ensemble, label=self.label,
        )

    # -- Event definition (fire dates + recovery breakpoints) ----------------

    @property
    def fire_start_date(self):
        """The event's fire start date, as a pandas Timestamp.

        Read from the event-scoped FireMeta.csv written by
        :func:`calculate_fire_severity`. Raises if this context has no
        event, or if fire severity has not been run for it.
        """
        return self._fire_meta_date('start_date')

    @property
    def fire_end_date(self):
        """The event's fire end date, as a pandas Timestamp.

        Read from the event-scoped FireMeta.csv written by
        :func:`calculate_fire_severity`. Raises if this context has no
        event, or if fire severity has not been run for it.
        """
        return self._fire_meta_date('end_date')

    def event_definition(self) -> EventDefinition:
        """Return this event's :class:`EventDefinition`.

        Combines the fire dates from the event's FireMeta.csv with the
        recovery breakpoints persisted in ``Events/<event>/event.json``.
        If no event.json exists yet, the package default breakpoints are
        used and a warning is logged.
        """
        path = self._event_definition_path()
        if os.path.exists(path):
            data = self._read_event_json()
        else:
            logger.warning(
                'No %s for catchment %s event %s; using the default '
                'recovery breakpoints. Re-run compute_adjusted_k_c to '
                'persist them.',
                const.EVENT_DEFINITION_NAME, self.catchment, self.event,
            )
            data = {}
        return EventDefinition.from_dict(
            data,
            fire_start_date=self.fire_start_date,
            fire_end_date=self.fire_end_date,
        )

    def set_recovery_breakpoints(self, breakpoints) -> EventDefinition:
        """Persist this event's recovery breakpoints to event.json.

        Validates the breakpoints (at least two, strictly increasing)
        before writing. Returns the resulting EventDefinition.

        Merges into any existing event.json rather than replacing it, so
        parameter overrides written by set_parameter_overrides survive.
        """
        const.recovery_windows(breakpoints)
        definition = EventDefinition(
            fire_start_date=self.fire_start_date,
            fire_end_date=self.fire_end_date,
            recovery_breakpoints=list(breakpoints),
        )
        self._update_event_json(definition.to_dict())
        return definition

    # -- Calibration parameters ---------------------------------------------

    def event_parameter_overrides(self) -> dict:
        """Return this event's parameter overrides from event.json.

        Empty dict when the event has none (or has no event.json yet).
        Raises if this context has no event.
        """
        return self._read_event_json().get('parameters', {})

    def set_event_parameter_overrides(self, overrides) -> dict:
        """Persist event-scope parameter overrides to event.json.

        Parameters:
        - overrides: a ModelParameters instance, or a nested dict of
          {group: {field: value}}. Either way only what differs from the
          package defaults is stored, so the file records choices rather
          than pinning every default. A ModelParameters carrying
          catchment-scoped changes is still refused — those belong in the
          catchment file.

        Merges into any existing event.json, leaving recovery_breakpoints
        untouched. Validated before writing.
        """
        from .params import ModelParameters, check_scope, sparse_overrides
        if isinstance(overrides, ModelParameters):
            # Only what differs from the defaults — see sparse_overrides.
            data = sparse_overrides(overrides)
        else:
            data = dict(overrides or {})
            ModelParameters.from_dict(data)   # validate; raises on bad keys
        # An event file may not carry catchment-scoped values: the layers
        # they control are written once per catchment, so an event-level
        # value would either be ignored or would overwrite a file the
        # sibling events share.
        check_scope(data, 'event')
        self._update_event_json({'parameters': data})
        return data

    # -- Run identity --------------------------------------------------------

    def ensure_run_directory(self) -> str:
        """Create this run's output directory and record what it is.

        The directory is named by :attr:`run_label`, which need not be
        the ensemble name, so the ensemble cannot be read back off the
        path. ``run.json`` carries it, and is written here — when the
        directory is created — rather than by save_ensemble_run, which
        only runs on success. A run that crashes midway would otherwise
        leave a directory nothing could identify.

        Raises if the directory already belongs to a different
        (event, ensemble): two ensembles given the same label would
        otherwise write into each other's results.

        Returns:
        - The run directory path.
        """
        root = self.run_path()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, const.RUN_DEFINITION_NAME)

        identity = {
            'event': self.event,
            'ensemble': self.ensemble,
            'label': self.run_label,
        }
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
            clash = {
                key: (existing.get(key), identity[key])
                for key in ('event', 'ensemble')
                if existing.get(key) != identity[key]
            }
            if clash:
                detail = '; '.join(
                    f'{key}: directory belongs to {was!r}, this run is {now!r}'
                    for key, (was, now) in sorted(clash.items())
                )
                raise ValueError(
                    f'{root} already holds a different run — {detail}. Two '
                    f'runs cannot share a label unless they are the same '
                    f'(event, ensemble); give this one a different label.'
                )
            return root

        with open(path, 'w') as f:
            json.dump(identity, f, indent=2)
        return root

    def run_identity(self) -> dict | None:
        """Return this run directory's recorded identity, or None."""
        path = os.path.join(self.run_path(), const.RUN_DEFINITION_NAME)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # -- Input bindings ------------------------------------------------------

    def event_binding_overrides(self) -> dict:
        """Return this event's input bindings from event.json."""
        return self._read_event_json().get('bindings', {})

    def set_event_binding_overrides(self, bindings) -> dict:
        """Persist event-scope input bindings to event.json.

        Parameters:
        - bindings: an InputBindings instance, or a nested dict of
          {input: {source: ..., ...}}.

        Validated before writing, and merged into event.json alongside
        the recovery breakpoints and parameter overrides.
        """
        from .bindings import InputBindings, check_binding_scope

        if isinstance(bindings, InputBindings):
            data = {
                name: binding
                for name, binding in bindings.to_dict().items()
                if binding.get('source') != 'derived'
            }
        else:
            data = dict(bindings or {})
            InputBindings.from_dict(data)   # validate
        # An event file may only carry event-scoped inputs. c_factor
        # builds a catchment-level layer every fire in the catchment
        # shares, so binding it per event would have one event rewrite
        # the others' input.
        check_binding_scope(data, 'event')
        self._update_event_json({'bindings': data})
        return data

    def catchment_binding_overrides(self) -> dict:
        """Return this context's catchment-scope input bindings."""
        return self.project.catchment_binding_overrides(self.catchment)

    def set_catchment_binding_overrides(self, bindings) -> dict:
        """Persist catchment-scope bindings for this context's catchment.

        Where a c_factor binding belongs; set_event_binding_overrides
        refuses it.
        """
        return self.project.set_catchment_binding_overrides(
            self.catchment, bindings)

    def bindings(self):
        """Resolve this context's input bindings.

        Merges the project, catchment and event layers, most specific
        winning. Unlike :meth:`parameters` there is deliberately no
        call-site layer: resolving a binding writes a raster, so an
        override at simulation time would either rewrite a preprocessing
        artefact mid-run or be silently ignored. Bindings are applied at
        exactly one point, by the function that materialises them.

        Returns:
        - An :class:`fire_impacts.bindings.InputBindings`.
        """
        from .bindings import InputBindings, check_binding_scope

        merged: dict = {}
        for layer, data, where in (
            ('project', self.project.binding_overrides(),
             const.PARAMETERS_FILE_NAME),
            ('catchment',
             self.project.catchment_binding_overrides(self.catchment),
             f'{self.catchment}/{const.PARAMETERS_FILE_NAME}'),
            ('event',
             self.event_binding_overrides() if self.event else {},
             f'{self.event}/{const.EVENT_DEFINITION_NAME}'),
        ):
            if not data:
                continue
            # Checked on read as well as on write: these files are
            # hand-editable, so the setters are not the only way in.
            try:
                check_binding_scope(data, layer)
            except ValueError as exc:
                raise ValueError(f'In {where}: {exc}') from None
            merged.update(data)
        return InputBindings.from_dict(merged)

    # -- Catchment-scope overrides (delegate to the project store) ----------

    def catchment_parameter_overrides(self) -> dict:
        """Return this context's catchment-scope parameter overrides."""
        return self.project.catchment_parameter_overrides(self.catchment)

    def set_catchment_parameter_overrides(self, overrides) -> dict:
        """Persist catchment-scope overrides for this context's catchment.

        This is where catchment-scoped groups (topography, delivery) must
        be set; :meth:`set_parameter_overrides` rejects them.
        """
        return self.project.set_catchment_parameter_overrides(
            self.catchment, overrides,
        )

    def parameters(self, **overrides):
        """Resolve this context's calibration parameters.

        Merges five layers, most specific winning: the package defaults,
        ``<project>/parameters.json``, this catchment's
        ``Catchments/<c>/parameters.json``, this event's ``parameters``
        key in event.json (skipped for a catchment-only context), and any
        call-site overrides given here as ``group__field=value``.

        Each parameter declares the scope of the output it controls, and
        every persisted layer is checked against it on read as well as on
        write — so a catchment-scoped value hand-added to an event file
        raises rather than silently rewriting the layers the other events
        share (see :func:`fire_impacts.params.check_scope`). Call-site
        overrides are deliberately unrestricted.

        Parameters:
        - overrides: call-site overrides, e.g.
          ``ctx.parameters(delivery__max_sdr=0.9)``.

        Returns:
        - A :class:`fire_impacts.params.ParameterRecord` carrying the
          resolved values, the origin of each one, and a digest.

        Notes:
        - Unknown group or field names raise rather than being ignored,
          at every layer.
        """
        from .params import nest_overrides, resolve_parameters

        # Check the persisted layers on the way IN, not only when they are
        # written. Both files are documented as hand-editable, so the
        # setters are not the only way a value gets into them — and a
        # catchment-scoped value hand-added to an event file is exactly the
        # cross-event corruption the scope system exists to prevent
        # (fire A's event.json rewriting the shared SDR_baseline.tif).
        layers = []
        for layer, data, where in (
            ('project', self.project.parameter_overrides(),
             const.PARAMETERS_FILE_NAME),
            ('catchment', self.catchment_parameter_overrides(),
             f'{self.catchment}/{const.PARAMETERS_FILE_NAME}'),
        ):
            _check_layer_scope(data, layer, where)
            layers.append((layer, data))
        if self.event is not None:
            event_data = self.event_parameter_overrides()
            _check_layer_scope(
                event_data, 'event',
                f'{self.event}/{const.EVENT_DEFINITION_NAME}',
            )
            layers.append(('event', event_data))
        if overrides:
            # The call layer carries exactly the keys the caller named,
            # NOT a diff against the package defaults. Diffing would drop
            # an explicit override that happens to equal the default, so
            # ctx.parameters(delivery__max_sdr=0.8) against a project file
            # holding 0.9 would silently resolve to 0.9 — the failure mode
            # this whole system exists to prevent. If the caller typed it,
            # they chose it.
            layers.append(('call', nest_overrides(overrides)))
        return resolve_parameters(layers)

    def _resolved_params(self, params=None, **overrides):
        """Resolve parameters for a model function.

        The single entry point every public pre/sim function uses, so the
        precedence rules live in one place:

        - ``params`` given: it is authoritative and returned as a record
          (a bare :class:`~fire_impacts.params.ModelParameters` is wrapped,
          attributing every leaf to 'call'). The caller has already
          resolved, so the layers are not re-read.
        - ``params`` not given: the five layers are merged as in
          :meth:`parameters`, with ``overrides`` as the call layer.

        Passing both ``params`` and an override for the same value is
        ambiguous, so it raises rather than silently picking one.

        Parameters:
        - params: a ParameterRecord or ModelParameters, or None.
        - overrides: call-layer overrides as ``group__field=value``,
          typically built by
          :func:`~fire_impacts.params.deprecated_overrides` from legacy
          keyword arguments.

        Returns:
        - A :class:`~fire_impacts.params.ParameterRecord`.
        """
        from .params import as_record

        if params is None:
            return self.parameters(**overrides)
        if overrides:
            clashes = sorted(
                key.replace('__', '.') for key in overrides
            )
            raise ValueError(
                f'Cannot pass params= together with an override for '
                f'{clashes}: which one wins is ambiguous. Either set the '
                f'value in the params you pass, or drop params= and let '
                f'the context resolve.'
            )
        return as_record(params)

    def write_provenance(self, record, *, scope: str, section=None,
                         groups=None, extra=None) -> str:
        """Write a resolved parameter record beside the outputs it produced.

        Parameters:
        - record: the ParameterRecord the step resolved.
        - scope: 'catchment', 'event' or 'run' — which output tree this
          step wrote to.
        - section: for run scope, the results sub-folder (e.g. 'Results').
          Ignored for catchment and event scope, which have no sections.
        - groups: parameter groups this step actually consumed. When
          given, only those groups are updated and the rest of an
          existing record is preserved — several steps write catchment
          scope (extract_headwaters, compute_lsi, the base C/K build) and
          the last to run would otherwise erase what the others recorded.
          None writes the whole record.
        - extra: additional top-level keys to store alongside the record,
          such as the run signature used to detect an overwrite. Kept
          outside 'values' so the record's digest still describes exactly
          the parameters.

        Returns:
        - The path written.

        Notes:
        - Deliberately a different file from parameters.json: that is the
          sparse override *input* a user edits, this is the full resolved
          *output*. Writing the record back into an override file would
          turn every package default into an explicit user setting on the
          first run, which destroys the default/chosen distinction the
          record exists to preserve.
        """
        path = self._provenance_path(scope, section)
        data = record.to_dict()
        if groups is not None:
            existing = self.read_provenance(scope=scope, section=section)
            if existing is not None:
                merged = existing.to_dict()
                for group in groups:
                    if group in data['values']:
                        merged['values'][group] = data['values'][group]
                prefixes = tuple(f'{g}.' for g in groups)
                merged['sources'] = {
                    **{k: v for k, v in merged['sources'].items()
                       if not k.startswith(prefixes)},
                    **{k: v for k, v in data['sources'].items()
                       if k.startswith(prefixes)},
                }
                merged['resolved_at'] = data['resolved_at']
                merged['digest'] = _digest_of(merged['values'])
                data = merged
        if extra:
            data = {**data, **extra}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info('Wrote parameter provenance to %s', path)
        return path

    def read_provenance(self, *, scope: str, section=None):
        """Read back a provenance record, or None when absent.

        Returns a :class:`~fire_impacts.params.ParameterRecord`.
        """
        from .params import ParameterRecord

        path = self._provenance_path(scope, section)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return ParameterRecord.from_dict(json.load(f))

    def _provenance_path(self, scope: str, section=None) -> str:
        """Resolve the provenance path for a scope (shared by read/write)."""
        if scope == 'catchment':
            return self.catchment_path(const.PROVENANCE_FILE_NAME)
        if scope == 'event':
            return self.event_path(const.PROVENANCE_FILE_NAME)
        if scope == 'run':
            parts = [section] if section else []
            return self.run_path(*parts, const.PROVENANCE_FILE_NAME)
        raise ValueError(
            f"Unknown provenance scope {scope!r}; expected 'catchment', "
            f"'event' or 'run'."
        )

    # -- event.json read/write ----------------------------------------------

    def _read_event_json(self) -> dict:
        """Return the parsed event.json, or {} when it does not exist."""
        path = self._event_definition_path()
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _update_event_json(self, updates: dict) -> dict:
        """Merge updates into event.json and write it back.

        event.json holds several independent concerns (recovery
        breakpoints, parameter overrides). Each writer must preserve the
        others rather than replacing the file wholesale.
        """
        path = self._event_definition_path()
        data = self._read_event_json()
        data.update(updates)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return data

    def simulation_period(self):
        """Return the (start, end) pandas Timestamps of this event's
        recovery simulation period.

        The period spans the recovery windows recorded in event.json —
        from the fire end date through the end of the last window — so
        the simulation-period end never has to be hard-coded. Suitable
        for get_rainfall_replicates and aggregate_rainfall_data.
        """
        return self.event_definition().simulation_period()

    def _event_definition_path(self) -> str:
        """Path to this event's event.json (raises if event is None)."""
        return self.event_path(const.EVENT_DEFINITION_NAME)

    def _fire_meta_date(self, key):
        """Read one date from the event-scoped FireMeta.csv."""
        path = self.event_path(
            const.FIRE_SEVERITY_FOLDER_NAME, 'FireMeta.csv',
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'No fire metadata at {path}. Run '
                f'calculate_fire_severity for event {self.event!r} '
                f'first.'
            )
        fire_meta = pd.read_csv(path, index_col=0)
        return pd.to_datetime(fire_meta.loc[key, 'Value'])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digest_of(values):
    """Digest a plain values dict (used when merging provenance records)."""
    from .params import _digest
    return _digest(values)


def _check_layer_scope(data, layer, where):
    """Scope-check one persisted override layer, naming the file on failure."""
    from .params import check_scope

    if not data:
        return
    try:
        check_scope(data, layer)
    except ValueError as exc:
        raise ValueError(f'In {where}: {exc}') from None


def _resolve_catchments(project, catchment):
    """Return the list of catchment names to iterate, filtered by name."""
    if catchment is None:
        return list(project.catchments)
    if catchment not in project.catchments:
        raise ValueError(
            f"Catchment {catchment!r} is not registered with the "
            f"project. Known: {list(project.catchments)}"
        )
    return [catchment]


def _assert_catchment_registered(project, catchment):
    """Raise if a catchment name is not registered with the project."""
    if catchment not in project.catchments:
        raise ValueError(
            f"Catchment {catchment!r} is not registered with the "
            f"project. Known: {list(project.catchments)}"
        )


def _resolve_solo_catchment(project, catchment):
    """Resolve None to the project's sole catchment, or require an explicit name."""
    if catchment is not None:
        return catchment
    catchments = list(project.catchments)
    if len(catchments) == 1:
        return catchments[0]
    raise ValueError(
        f"Project has {len(catchments)} catchments — catchment must "
        f"be specified explicitly. Known: {catchments}"
    )
