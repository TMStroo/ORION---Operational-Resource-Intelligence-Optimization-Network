
"""Time, units and interval primitives.

ORION measures all time in **integer minutes from the scenario epoch**. Integer
minutes are used deliberately:

* Deadlines, releases, durations and travel legs are all discrete in the
  operational domains ORION targets (a 15-minute slot is a planning unit in
  emergency response, a 30-minute slot in field maintenance).
* Integer arithmetic removes floating-point tie-breaking noise from solver
  comparisons, which matters when ORION reports objective values to one decimal
  place and claims a solution is optimal.
* Scheduling constraints become difference constraints with integer bounds,
  which both CP-SAT and the MILP backend handle without scaling tricks.

Distances are in kilometres. Costs are in abstract currency units; a scenario
supplies the weights, never a hard-coded price table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Minutes in one hour / one day, exposed so call sites avoid magic numbers.
MINUTES_PER_HOUR: Final[int] = 60
MINUTES_PER_DAY: Final[int] = 24 * MINUTES_PER_HOUR

#: One week of simulated time. A scenario horizon longer than this is almost
#: certainly a unit error, so validation rejects it rather than letting a
#: misconfigured scenario run for a simulated month.
MAX_HORIZON_MINUTES: Final[int] = 7 * MINUTES_PER_DAY


def hhmm(minutes: int) -> str:
    """Format minutes-from-epoch as ``HH:MM`` for human-readable output."""
    if minutes < 0:
        return f"-{hhmm(-minutes)}"
    return f"{minutes // MINUTES_PER_HOUR:02d}:{minutes % MINUTES_PER_HOUR:02d}"


def parse_hhmm(value: str) -> int:
    """Parse ``HH:MM`` (or ``HH:MM:SS``, truncating seconds) into minutes."""
    parts = value.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"cannot parse time {value!r}, expected HH:MM")
    try:
        hours = int(parts[0])
        mins = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"cannot parse time {value!r}, expected HH:MM") from exc
    if not (0 <= hours < 100 and 0 <= mins < 60):
        raise ValueError(f"time {value!r} out of range")
    return hours * MINUTES_PER_HOUR + mins


@dataclass(frozen=True, slots=True, order=True)
class Interval:
    """A half-open interval ``[start, end)`` in minutes from the epoch.

    Half-open semantics are chosen so that two consecutive assignments on the
    same resource do not overlap when one ends exactly when the next begins.
    With closed intervals, a task ending at 10:00 and another starting at 10:00
    would be reported as a spurious conflict.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Interval end {self.end} precedes start {self.start} "
                f"({hhmm(self.start)} -> {hhmm(self.end)})"
            )

    @property
    def duration(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Interval) -> bool:
        """True when the two intervals share at least one minute."""
        return self.start < other.end and other.start < self.end

    def contains(self, point: int) -> bool:
        return self.start <= point < self.end

    def intersection(self, other: Interval) -> Interval | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if end < start:
            return None
        return Interval(start, end)

    def with_start(self, start: int) -> Interval:
        """Shift the interval so it begins at *start*, preserving duration."""
        return Interval(start, start + self.duration)

    def shifted(self, delta: int) -> Interval:
        return Interval(self.start + delta, self.end + delta)

    def clamp_to(self, bounds: Interval) -> Interval | None:
        return self.intersection(bounds)

    def __str__(self) -> str:
        return f"[{hhmm(self.start)}-{hhmm(self.end)})"


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping/touching intervals into a minimal covering set."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start <= last.end:
            if interval.end > last.end:
                merged[-1] = Interval(last.start, interval.end)
        else:
            merged.append(interval)
    return merged


def subtract_intervals(base: Interval, holes: list[Interval]) -> list[Interval]:
    """Return ``base`` minus every interval in *holes*."""
    remaining = [base]
    for hole in merge_intervals(holes):
        nxt: list[Interval] = []
        for piece in remaining:
            if not piece.overlaps(hole):
                nxt.append(piece)
                continue
            if hole.start > piece.start:
                nxt.append(Interval(piece.start, hole.start))
            if hole.end < piece.end:
                nxt.append(Interval(hole.end, piece.end))
        remaining = nxt
        if not remaining:
            break
    return remaining


def total_duration(intervals: list[Interval]) -> int:
    return sum(interval.duration for interval in intervals)


__all__ = [
    "MINUTES_PER_HOUR",
    "MINUTES_PER_DAY",
    "MAX_HORIZON_MINUTES",
    "Interval",
    "hhmm",
    "parse_hhmm",
    "merge_intervals",
    "subtract_intervals",
    "total_duration",
]
