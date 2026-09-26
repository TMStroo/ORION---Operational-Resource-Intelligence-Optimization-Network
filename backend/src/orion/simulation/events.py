
"""Discrete-event primitives for the simulator.

The simulator is a standard event-scheduling loop: a priority queue of timed
events, popped in non-decreasing time order, each mutating world state and
possibly scheduling follow-up events. Every state change is emitted as a
:class:`SimEvent`, so the run produces a complete, replayable trace.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterator, Mapping

from orion.domain.time_model import hhmm


class EventType(IntEnum):
    """Simulation event taxonomy. Values order the trace deterministically.

    Ordering matters: at the same simulated minute, a resource becoming
    unavailable must be processed before the task that uses it starts, or the
    simulator would execute a plan against a world state that no longer exists.
    That is why ``RESOURCE_FAIL`` (0) precedes ``TASK_ARRIVE`` (2).
    """

    RESOURCE_FAIL = 0
    MAINTENANCE_START = 1
    TASK_ARRIVE = 2
    TASK_START = 3
    TASK_COMPLETE = 4
    NEW_TASK = 5
    MAINTENANCE_END = 6
    RESOURCE_RETURN = 7
    DISRUPTION = 8
    REPLAN_START = 9
    REPLAN_END = 10


EVENT_LABELS: Mapping[EventType, str] = {
    EventType.RESOURCE_FAIL: "resource_failed",
    EventType.MAINTENANCE_START: "maintenance_started",
    EventType.TASK_ARRIVE: "resource_arrived",
    EventType.TASK_START: "task_started",
    EventType.TASK_COMPLETE: "task_completed",
    EventType.NEW_TASK: "new_task_appeared",
    EventType.MAINTENANCE_END: "maintenance_ended",
    EventType.RESOURCE_RETURN: "resource_returned",
    EventType.DISRUPTION: "disruption_registered",
    EventType.REPLAN_START: "replanning_started",
    EventType.REPLAN_END: "replanning_finished",
}


@dataclass(frozen=True, slots=True)
class SimEvent:
    """One entry in the simulation trace."""

    time: int
    type: EventType
    subject_id: str | None = None
    resource_id: str | None = None
    detail: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"simulation event at negative time {self.time}")

    @property
    def label(self) -> str:
        return EVENT_LABELS.get(self.type, self.type.name.lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "clock": hhmm(self.time),
            "type": self.type.name,
            "label": self.label,
            "subject_id": self.subject_id,
            "resource_id": self.resource_id,
            "detail": self.detail,
            "payload": dict(self.payload),
        }

    def to_csv_row(self) -> list[Any]:
        return [
            self.time,
            hhmm(self.time),
            self.type.name,
            self.subject_id or "",
            self.resource_id or "",
            self.detail,
        ]


class EventQueue:
    """Deterministic min-heap of simulation events.

    Ties are broken by ``(time, event_type, sequence)``. The sequence counter is
    what makes two events at the same instant pop in insertion order, which is
    what lets a test assert an exact trace.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, SimEvent]] = []
        self._counter = itertools.count()
        self._by_id: dict[int, SimEvent] = {}

    def push(self, event: SimEvent) -> int:
        """Schedule *event*. Returns its handle for cancellation."""
        event_id = next(self._counter)
        heapq.heappush(self._heap, (event.time, int(event.type), event_id, event))
        self._by_id[event_id] = event
        return event_id

    def pop(self) -> SimEvent | None:
        if not self._heap:
            return None
        _, _, event_id, event = heapq.heappop(self._heap)
        self._by_id.pop(event_id, None)
        return event

    def peek_time(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def cancel(self, event_id: int) -> bool:
        """Remove a scheduled event. Returns True if it was still pending."""
        if event_id not in self._by_id:
            return False
        del self._by_id[event_id]
        for i, (_, _, eid, _) in enumerate(self._heap):
            if eid == event_id:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                return True
        return False

    def cancel_for(self, *, subject_id: str | None = None, resource_id: str | None = None,
                   types: tuple[EventType, ...] | None = None) -> int:
        """Cancel every pending event matching the given filters."""
        victims = [
            eid
            for eid, event in self._by_id.items()
            if (subject_id is None or event.subject_id == subject_id)
            and (resource_id is None or event.resource_id == resource_id)
            and (types is None or event.type in types)
        ]
        removed = 0
        for eid in victims:
            if self.cancel(eid):
                removed += 1
        return removed

    def drain_until(self, time: int) -> Iterator[SimEvent]:
        """Yield every event with ``event.time <= time`` in order."""
        while self._heap and self._heap[0][0] <= time:
            event = self.pop()
            if event is not None:
                yield event

    def pending(self) -> list[SimEvent]:
        return sorted(
            (e for _, _, _, e in self._heap),
            key=lambda e: (e.time, int(e.type)),
        )

    def __len__(self) -> int:
        return len(self._heap)


__all__ = [
    "EventType",
    "EVENT_LABELS",
    "SimEvent",
    "EventQueue",
]
