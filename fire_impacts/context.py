"""
Run-context object bundling project + catchment + event + ensemble.

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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .pre.project import FireImpactsProject


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
