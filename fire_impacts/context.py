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
    ) -> 'RunContext':
        """Convenience for the single-catchment case.

        Returns the unique (event, ensemble) run context. If ``catchment``
        is None and the project has exactly one catchment, that catchment
        is used; otherwise ``catchment`` must be supplied.
        """
        catchment = _resolve_solo_catchment(project, catchment)
        _assert_catchment_registered(project, catchment)
        return cls(
            project=project, catchment=catchment,
            event=event, ensemble=ensemble,
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

    def run_path(self, *args) -> str:
        """Resolve a path under this context's run folder.

        Raises if this context has no ensemble.
        """
        if self.ensemble is None:
            raise ValueError(
                'This context has no ensemble; run_path() requires a '
                'run-level context.'
            )
        return self.project.run_path(
            self.catchment, *args,
            event=self.event, ensemble=self.ensemble,
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
            with open(path) as f:
                data = json.load(f)
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
        """
        const.recovery_windows(breakpoints)
        path = self._event_definition_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        definition = EventDefinition(
            fire_start_date=self.fire_start_date,
            fire_end_date=self.fire_end_date,
            recovery_breakpoints=list(breakpoints),
        )
        with open(path, 'w') as f:
            json.dump(definition.to_dict(), f, indent=2)
        return definition

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
