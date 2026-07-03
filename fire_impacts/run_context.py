"""
Event run-context: the persisted description of a fire event and the
recovery windows modelled for it.

A run-context bundles the fire dates and the recovery-time breakpoints so
they are specified once (at preprocessing) and read back by the simulation
steps, rather than re-declared in every notebook. It is written per
catchment on this branch; in the multi-event model the same record is
written per event (the schema is unchanged — only the storage scope gains
an event dimension).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fire_impacts import const as c


def _to_iso(value):
    """Serialise a date-like value to an ISO date string, or None."""
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _parse(value):
    """Parse a date-like value to a pandas Timestamp, or None."""
    if value is None:
        return None
    return pd.Timestamp(value)


@dataclass(frozen=True)
class EventRunContext:
    """
    Immutable description of a fire event's recovery modelling setup.

    Attributes:
    - fire_start_date / fire_end_date: pandas Timestamps (or None until
      set by fire-severity preprocessing).
    - recovery_breakpoints: monotonically increasing years-since-fire
      boundaries. n+1 breakpoints define n contiguous recovery windows;
      window i is [b_i, b_{i+1}) and is modelled at recovery time b_i.
    """

    fire_start_date: object = None
    fire_end_date: object = None
    recovery_breakpoints: list = field(
        default_factory=lambda: list(c.DEFAULT_RECOVERY_BREAKPOINTS)
    )

    # -- Derived recovery structure -------------------------------------

    def recovery_times(self):
        """Return the window-start recovery times (breakpoints[:-1])."""
        return list(self.recovery_breakpoints[:-1])

    def windows(self):
        """Return [(start, end), ...] window pairs in years since fire."""
        return c.recovery_windows(self.recovery_breakpoints)

    def window_for(self, recovery_time):
        """Return the (start, end) years for the window starting at
        recovery_time. Raises ValueError if recovery_time is not a
        window start."""
        for start, end in self.windows():
            if start == recovery_time:
                return (start, end)
        raise ValueError(
            f"recovery_time {recovery_time} is not a window start in "
            f"breakpoints {self.recovery_breakpoints!r}."
        )

    def absolute_window(self, recovery_time):
        """Return the (start, end) pandas Timestamps of the window
        starting at recovery_time, measured from fire_end_date."""
        start_years, end_years = self.window_for(recovery_time)
        if self.fire_end_date is None:
            raise ValueError(
                "run-context has no fire_end_date; cannot resolve an "
                "absolute recovery window."
            )
        base = pd.Timestamp(self.fire_end_date)
        start = base + pd.DateOffset(days=int(start_years * 365.25))
        end = base + pd.DateOffset(days=int(end_years * 365.25))
        return (start, end)

    # -- Serialisation --------------------------------------------------

    def to_dict(self):
        """Return a JSON-serialisable dict representation."""
        return {
            "fire_start_date": _to_iso(self.fire_start_date),
            "fire_end_date": _to_iso(self.fire_end_date),
            "recovery_breakpoints": list(self.recovery_breakpoints),
        }

    @classmethod
    def from_dict(cls, data):
        """Build an EventRunContext from a dict (as produced by
        to_dict). Missing breakpoints fall back to the package default."""
        breakpoints = data.get("recovery_breakpoints")
        return cls(
            fire_start_date=_parse(data.get("fire_start_date")),
            fire_end_date=_parse(data.get("fire_end_date")),
            recovery_breakpoints=(
                list(breakpoints) if breakpoints
                else list(c.DEFAULT_RECOVERY_BREAKPOINTS)
            ),
        )
