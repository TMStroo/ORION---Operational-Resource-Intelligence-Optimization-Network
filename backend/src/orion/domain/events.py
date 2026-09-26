
"""Disruptions and the world-state transition they cause.

A :class:`Disruption` is a *declarative* description of an environmental change.
Applying it produces a **new** Scenario via :meth:`Disruption.apply`. Nothing is
mutated in place. That is the mechanism behind the required flow

    preserve state -> apply disruption -> detect impact -> repair -> compare

: the "previous world" is simply the Scenario object you already had a reference
to, and there is no way for a disruption to leak backwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from orion.domain.entities import (
    Resource,
    ResourceStatus,
    Scenario,
    Task,
    TaskStatus,
)
from orion.domain.errors import UnknownEntityError, ValidationError
from orion.domain.ids import format_id, validate_id
from orion.domain.time_model import Interval, hhmm


class DisruptionType(str):
    """Disruption taxonomy.

    Every type here is one ORION knows how to *apply* to a Scenario and detect
    impact on. A type that cannot be applied is not offered: ORION does not have
    a "generic disruption" that silently does nothing.
    """

    VEHICLE_FAILURE = "VEHICLE_FAILURE"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    RESOURCE_RETURN = "RESOURCE_RETURN"
    NEW_URGENT_TASK = "NEW_URGENT_TASK"
    DEADLINE_CHANGE = "DEADLINE_CHANGE"
    TRAVEL_TIME_INCREASE = "TRAVEL_TIME_INCREASE"
    CAPACITY_REDUCTION = "CAPACITY_REDUCTION"
    MAINTENANCE_EVENT = "MAINTENANCE_EVENT"
    DEMAND_SURGE = "DEMAND_SURGE"


#: Human-facing descriptions, surfaced in the UI disruption list.
DISRUPTION_DESCRIPTIONS: Mapping[str, str] = {
    DisruptionType.VEHICLE_FAILURE: "A vehicle fails and is out of service for the rest of the horizon.",
    DisruptionType.RESOURCE_UNAVAILABLE: "A resource becomes unavailable for a window.",
    DisruptionType.RESOURCE_RETURN: "A previously unavailable resource comes back.",
    DisruptionType.NEW_URGENT_TASK: "A new high-priority task appears.",
    DisruptionType.DEADLINE_CHANGE: "A task deadline moves earlier or later.",
    DisruptionType.TRAVEL_TIME_INCREASE: "Congestion slows every leg in the network.",
    DisruptionType.CAPACITY_REDUCTION: "A resource loses capacity in one dimension.",
    DisruptionType.MAINTENANCE_EVENT: "A resource is pulled into maintenance for a window.",
    DisruptionType.DEMAND_SURGE: "Several new urgent tasks appear at once.",
}


class DisruptionSeverity(str):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


#: Ordered low < medium < high, used by the disruption-rate stress test.
SEVERITY_ORDER: Mapping[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass(frozen=True, slots=True)
class NewTaskSpec:
    """Payload for a task introduced by a disruption."""

    task_id: str
    location: str
    release_time: int
    deadline: int
    duration: int
    priority: int = 4
    required_capabilities: tuple[str, ...] = ()
    required_capacity: tuple[tuple[str, float], ...] = ()

    def to_task(self) -> Task:
        from orion.domain.capacity import CapacityDemand, parse_priority

        return Task(
            id=self.task_id,
            location=self.location,
            release_time=self.release_time,
            deadline=self.deadline,
            duration=self.duration,
            priority=parse_priority(self.priority),
            required_capabilities=frozenset(self.required_capabilities),
            required_capacity=tuple(
                CapacityDemand(dim, amt) for dim, amt in self.required_capacity
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "location": self.location,
            "release_time": self.release_time,
            "deadline": self.deadline,
            "duration": self.duration,
            "priority": self.priority,
            "required_capabilities": list(self.required_capabilities),
            "required_capacity": [list(c) for c in self.required_capacity],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NewTaskSpec:
        return cls(
            task_id=str(data["task_id"]),
            location=str(data["location"]),
            release_time=int(data["release_time"]),
            deadline=int(data["deadline"]),
            duration=int(data["duration"]),
            priority=int(data.get("priority", 4)),
            required_capabilities=tuple(str(c) for c in data.get("required_capabilities", ())),
            required_capacity=tuple(
                (str(c[0]), float(c[1])) for c in data.get("required_capacity", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class Disruption:
    """A declarative environmental change at a point in simulated time."""

    id: str
    type: str
    timestamp: int
    target_id: str | None = None
    magnitude: float = 1.0
    duration: int | None = None
    description: str = ""
    severity: str = DisruptionSeverity.MEDIUM
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_id(self.id, field="disruption.id")
        if self.type not in DISRUPTION_DESCRIPTIONS:
            raise ValidationError(
                f"disruption {self.id}: unknown type {self.type!r}; allowed: "
                f"{sorted(DISRUPTION_DESCRIPTIONS)}"
            )
        if self.timestamp < 0:
            raise ValidationError(f"disruption {self.id}: timestamp {self.timestamp} < 0")
        if self.severity not in SEVERITY_ORDER:
            raise ValidationError(
                f"disruption {self.id}: unknown severity {self.severity!r}; allowed: "
                f"{sorted(SEVERITY_ORDER)}"
            )
        if self.magnitude <= 0:
            raise ValidationError(
                f"disruption {self.id}: magnitude {self.magnitude} must be > 0"
            )
        if self.duration is not None and self.duration <= 0:
            raise ValidationError(f"disruption {self.id}: duration {self.duration} must be > 0")
        if self.type in {
            DisruptionType.VEHICLE_FAILURE,
            DisruptionType.RESOURCE_UNAVAILABLE,
            DisruptionType.RESOURCE_RETURN,
            DisruptionType.CAPACITY_REDUCTION,
            DisruptionType.MAINTENANCE_EVENT,
        } and not self.target_id:
            raise ValidationError(f"disruption {self.id}: type {self.type} requires target_id")
        if self.type in {DisruptionType.NEW_URGENT_TASK, DisruptionType.DEADLINE_CHANGE} and (
            "task" not in self.payload
        ):
            raise ValidationError(
                f"disruption {self.id}: type {self.type} requires payload['task']"
            )

    @property
    def window(self) -> Interval:
        """When the disruption is in effect. Failures last to the horizon end."""
        if self.duration is None:
            return Interval(self.timestamp, 24 * 60 * 7)
        return Interval(self.timestamp, self.timestamp + self.duration)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER[self.severity]

    def summary(self) -> str:
        base = DISRUPTION_DESCRIPTIONS.get(self.type, self.type)
        target = f" -> {self.target_id}" if self.target_id else ""
        return f"[{hhmm(self.timestamp)}] {self.type}{target} (sev={self.severity}) {base}"

    # -- application -------------------------------------------------------
    def apply(self, scenario: Scenario) -> Scenario:
        """Return a new Scenario with this disruption applied.

        Raises ``UnknownEntityError`` when the target does not exist, so a
        mis-typed disruption id surfaces immediately instead of quietly doing
        nothing. A test asserts that a disruption referencing a missing entity
        raises rather than applying.
        """
        handler: Callable[[Scenario, Disruption], Scenario] | None = _HANDLERS.get(self.type)
        if handler is None:  # pragma: no cover - __post_init__ already guards
            raise ValidationError(f"disruption {self.id}: no handler for {self.type}")
        return handler(scenario, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "target_id": self.target_id,
            "magnitude": self.magnitude,
            "duration": self.duration,
            "description": self.description,
            "severity": self.severity,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Disruption:
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            timestamp=int(data["timestamp"]),
            target_id=data.get("target_id"),
            magnitude=float(data.get("magnitude", 1.0)),
            duration=(int(data["duration"]) if data.get("duration") is not None else None),
            description=str(data.get("description", "")),
            severity=str(data.get("severity", DisruptionSeverity.MEDIUM)),
            payload=dict(data.get("payload") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ==========================================================================
# Handlers - one per disruption type
# ==========================================================================
def _blocked_window(resource: Any, start: int, end: int) -> Interval | None:
    """The part of ``[start, end)`` that falls inside the resource's own shift.

    ``Resource`` validation rejects an unavailable block longer than the shift,
    so a block derived from the *scenario* horizon is invalid whenever the
    resource works a shorter shift than the horizon. Intersecting here means a
    disruption only removes time the resource was ever going to be on duty -
    which is also the semantically correct reading: a vehicle that fails at
    03:00 is not "unavailable" for hours it was never rostered to work.

    Returns ``None`` when the window misses the shift entirely, so the caller
    can skip adding a redundant interval.
    """
    lo = max(int(start), int(resource.shift_start))
    hi = min(int(end), int(resource.shift_end))
    if hi <= lo:
        return None
    return Interval(lo, hi)


def _apply_vehicle_failure(scenario: Scenario, disruption: Disruption) -> Scenario:
    """Vehicle fails: unusable for the remainder of the horizon."""
    target = disruption.target_id
    assert target is not None
    resource = _require_resource(scenario, target)
    window = _blocked_window(resource, disruption.timestamp, scenario.horizon.end)
    failed = replace(
        resource,
        status=ResourceStatus.FAILED,
        unavailable=tuple(
            sorted(set(resource.unavailable) | ({window} if window else set()))
        ),
    )
    out = scenario.with_resource(failed)
    return out.with_metadata(
        last_disruption=disruption.id,
        last_disruption_type=disruption.type,
    )


def _apply_resource_unavailable(scenario: Scenario, disruption: Disruption) -> Scenario:
    target = disruption.target_id
    assert target is not None
    resource = _require_resource(scenario, target)
    end = (
        disruption.timestamp + disruption.duration
        if disruption.duration
        else scenario.horizon.end
    )
    window = _blocked_window(resource, disruption.timestamp, end)
    blocked = replace(
        resource,
        unavailable=tuple(sorted(set(resource.unavailable) | ({window} if window else set()))),
    )
    return scenario.with_resource(blocked).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_resource_return(scenario: Scenario, disruption: Disruption) -> Scenario:
    """Bring a failed/unavailable resource back into service."""
    target = disruption.target_id
    assert target is not None
    resource = _require_resource(scenario, target)
    end = (
        disruption.timestamp + disruption.duration
        if disruption.duration
        else scenario.horizon.end
    )
    # only drop blocks that were created by the matching unavailability
    remaining = tuple(
        block
        for block in resource.unavailable
        if not (block.start == disruption.timestamp and block.end == end)
    )
    revived = replace(
        resource,
        status=ResourceStatus.AVAILABLE,
        unavailable=remaining,
    )
    return scenario.with_resource(revived).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_new_urgent_task(scenario: Scenario, disruption: Disruption) -> Scenario:
    spec = NewTaskSpec.from_dict(disruption.payload["task"])  # type: ignore[index]
    if any(t.id == spec.task_id for t in scenario.tasks):
        raise ValidationError(
            f"disruption {disruption.id}: task {spec.task_id} already exists in "
            f"scenario {scenario.id}"
        )
    task = spec.to_task()
    if not scenario.travel.has(task.location):
        raise ValidationError(
            f"disruption {disruption.id}: new task location {task.location!r} is not in "
            f"the travel matrix"
        )
    if task.deadline > scenario.horizon.end:
        # extend the horizon so a late-breaking urgent task is still schedulable
        new_horizon = Interval(scenario.horizon.start, max(scenario.horizon.end, task.deadline + 60))
        scenario = scenario.with_horizon(new_horizon)
    return scenario.with_task(task).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_deadline_change(scenario: Scenario, disruption: Disruption) -> Scenario:
    task_id = str(disruption.payload["task"])  # type: ignore[index]
    try:
        task = scenario.task(task_id)
    except UnknownEntityError as exc:
        raise UnknownEntityError(
            f"disruption {disruption.id}: cannot change deadline of unknown task {task_id!r}"
        ) from exc
    new_deadline = int(disruption.payload["new_deadline"])  # type: ignore[index]
    if new_deadline > scenario.horizon.end:
        scenario = scenario.with_horizon(
            Interval(scenario.horizon.start, new_deadline + 60)
        )
    if new_deadline < task.release_time:
        raise ValidationError(
            f"disruption {disruption.id}: new deadline {new_deadline} precedes release "
            f"{task.release_time} of task {task_id}; the task would be impossible"
        )
    updated = replace(task, deadline=new_deadline, status=TaskStatus.PENDING)
    return scenario.with_task(updated).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_travel_increase(scenario: Scenario, disruption: Disruption) -> Scenario:
    factor = float(disruption.magnitude)
    if factor < 1.0:
        raise ValidationError(
            f"disruption {disruption.id}: TRAVEL_TIME_INCREASE magnitude {factor} must be "
            ">= 1.0 (use a value < 1 to model faster travel, which is not a disruption)"
        )
    return scenario.with_travel(scenario.travel.scaled(factor)).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_capacity_reduction(scenario: Scenario, disruption: Disruption) -> Scenario:
    target = disruption.target_id
    assert target is not None
    resource = _require_resource(scenario, target)
    dimension = str(disruption.payload.get("dimension", "units"))  # type: ignore[union-attr]
    factor = float(disruption.payload.get("factor", 1.0 - disruption.magnitude))  # type: ignore[union-attr]
    if not 0.0 <= factor <= 1.0:
        raise ValidationError(
            f"disruption {disruption.id}: capacity factor {factor} must be in [0, 1]"
        )
    new_capacity = dict(resource.capacity)
    if dimension in new_capacity:
        new_capacity[dimension] = new_capacity[dimension] * factor
    else:
        new_capacity[dimension] = 0.0
    updated = replace(resource, capacity=new_capacity)
    return scenario.with_resource(updated).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_maintenance_event(scenario: Scenario, disruption: Disruption) -> Scenario:
    target = disruption.target_id
    assert target is not None
    resource = _require_resource(scenario, target)
    duration = disruption.duration or max(30, int(60 * disruption.magnitude))
    window = Interval(
        disruption.timestamp,
        min(scenario.horizon.end, disruption.timestamp + duration),
    )
    updated = replace(
        resource,
        unavailable=tuple(sorted(set(resource.unavailable) | {window})),
    )
    return scenario.with_resource(updated).with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _apply_demand_surge(scenario: Scenario, disruption: Disruption) -> Scenario:
    """A batch of new urgent tasks arrives."""
    specs = disruption.payload.get("tasks", [])  # type: ignore[union-attr]
    if not specs:
        raise ValidationError(
            f"disruption {disruption.id}: DEMAND_SURGE requires payload['tasks']"
        )
    out = scenario
    for index, spec in enumerate(specs, start=1):
        child = Disruption(
            id=f"{disruption.id}-{index:02d}",
            type=DisruptionType.NEW_URGENT_TASK,
            timestamp=disruption.timestamp,
            severity=disruption.severity,
            payload={"task": dict(spec)},
        )
        out = child.apply(out)
    return out.with_metadata(
        last_disruption=disruption.id, last_disruption_type=disruption.type
    )


def _require_resource(scenario: Scenario, resource_id: str) -> Resource:
    try:
        return scenario.resource(resource_id)
    except UnknownEntityError as exc:
        raise UnknownEntityError(
            f"scenario {scenario.id} has no resource {resource_id!r}"
        ) from exc


_HANDLERS: Mapping[str, Callable[[Scenario, Disruption], Scenario]] = {
    DisruptionType.VEHICLE_FAILURE: _apply_vehicle_failure,
    DisruptionType.RESOURCE_UNAVAILABLE: _apply_resource_unavailable,
    DisruptionType.RESOURCE_RETURN: _apply_resource_return,
    DisruptionType.NEW_URGENT_TASK: _apply_new_urgent_task,
    DisruptionType.DEADLINE_CHANGE: _apply_deadline_change,
    DisruptionType.TRAVEL_TIME_INCREASE: _apply_travel_increase,
    DisruptionType.CAPACITY_REDUCTION: _apply_capacity_reduction,
    DisruptionType.MAINTENANCE_EVENT: _apply_maintenance_event,
    DisruptionType.DEMAND_SURGE: _apply_demand_surge,
}


# ==========================================================================
# Impact analysis
# ==========================================================================
@dataclass(frozen=True, slots=True)
class DisruptionImpact:
    """Which parts of an existing plan a disruption puts at risk.

    Computed *before* replanning, from the disruption and the pre-disruption
    plan. This is what the UI shows under "Affected tasks".
    """

    disruption_id: str
    affected_resources: tuple[str, ...]
    affected_tasks: tuple[str, ...]
    at_risk_deadlines: tuple[str, ...]
    newly_infeasible_pairs: tuple[tuple[str, str], ...]
    severity: str

    @property
    def affected_count(self) -> int:
        return len(self.affected_tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disruption_id": self.disruption_id,
            "severity": self.severity,
            "affected_resources": list(self.affected_resources),
            "affected_tasks": list(self.affected_tasks),
            "at_risk_deadlines": list(self.at_risk_deadlines),
            "newly_infeasible_pairs": [list(p) for p in self.newly_infeasible_pairs],
            "affected_count": self.affected_count,
        }

    def describe(self) -> str:
        if not self.affected_tasks and not self.affected_resources:
            return f"{self.disruption_id}: no assignment impact detected"
        return (
            f"{self.disruption_id}: {len(self.affected_tasks)} task(s) affected across "
            f"{len(self.affected_resources)} resource(s)"
        )


def analyse_impact(
    disruption: Disruption,
    before: Scenario,
    after: Scenario,
    plan: Any,
) -> DisruptionImpact:
    """Determine which assignments a disruption endangers.

    The rules are derived from the disruption *type*, not guessed:

    * resource-scoped disruptions -> every task assigned to that resource;
    * travel/congestion changes -> every task with a tight deadline, because
      only those can absorb a leg-length increase;
    * deadline tightening -> the specific task plus anything downstream of it;
    * new tasks -> no existing assignment is invalidated (the new task simply
      competes for capacity), so affected_tasks is empty and the churn appears
      as *new* assignments.
    """
    from orion.domain.plans import Plan

    assert isinstance(plan, Plan)
    assigned_tasks = {a.task_id for a in plan.assignments}
    by_resource: dict[str, list[str]] = {}
    for a in plan.assignments:
        by_resource.setdefault(a.resource_id, []).append(a.task_id)

    affected_resources: set[str] = set()
    affected_tasks: set[str] = set()
    at_risk: set[str] = set()
    newly_infeasible: list[tuple[str, str]] = []

    dtype = disruption.type
    if dtype in {
        DisruptionType.VEHICLE_FAILURE,
        DisruptionType.RESOURCE_UNAVAILABLE,
        DisruptionType.RESOURCE_RETURN,
        DisruptionType.MAINTENANCE_EVENT,
        DisruptionType.CAPACITY_REDUCTION,
    }:
        target = disruption.target_id
        if target:
            affected_resources.add(target)
            affected_tasks.update(by_resource.get(target, ()))

    elif dtype == DisruptionType.TRAVEL_TIME_INCREASE:
        factor = after.travel.congestion_factor / max(1e-9, before.travel.congestion_factor)
        for a in plan.assignments:
            task = before.task(a.task_id)
            slack = task.deadline - a.end
            if slack < task.duration * max(0.0, factor - 1.0) + 30:
                at_risk.add(a.task_id)
        affected_tasks.update(at_risk)

    elif dtype == DisruptionType.DEADLINE_CHANGE:
        target_task = str(disruption.payload.get("task", ""))  # type: ignore[union-attr]
        if target_task in assigned_tasks:
            affected_tasks.add(target_task)
        # downstream dependents of the changed task
        changed = True
        while changed:
            changed = False
            for task in after.tasks:
                if task.id in affected_tasks:
                    continue
                if any(dep in affected_tasks for dep in task.dependencies):
                    affected_tasks.add(task.id)
                    changed = True

    elif dtype in {DisruptionType.NEW_URGENT_TASK, DisruptionType.DEMAND_SURGE}:
        pass  # no existing assignment is invalidated

    # pairs that were feasible before and are not feasible now
    before_pairs = set(before.eligible_pairs())
    after_pairs = set(after.eligible_pairs())
    for pair in sorted(before_pairs - after_pairs):
        task_id, resource_id = pair
        plan_assignment = plan.assignment_for(task_id)
        if plan_assignment is not None and plan_assignment.resource_id == resource_id:
            affected_tasks.add(task_id)
        newly_infeasible.append(pair)

    return DisruptionImpact(
        disruption_id=disruption.id,
        affected_resources=tuple(sorted(affected_resources)),
        affected_tasks=tuple(sorted(affected_tasks)),
        at_risk_deadlines=tuple(sorted(at_risk)),
        newly_infeasible_pairs=tuple(newly_infeasible[:200]),
        severity=disruption.severity,
    )


__all__ = [
    "Disruption",
    "DisruptionType",
    "DisruptionSeverity",
    "DisruptionImpact",
    "DISRUPTION_DESCRIPTIONS",
    "SEVERITY_ORDER",
    "NewTaskSpec",
    "analyse_impact",
]
