
"""Discrete-event simulation engine.

The engine executes a :class:`~orion.domain.plans.Plan` against a
:class:`~orion.domain.entities.Scenario` over simulated time and produces a
complete event trace plus live world state. It is the operational layer that
makes ORION a *system* rather than an optimiser: the plan is a proposal, and
this is where the world either honours it or does not.

Execution model
---------------

For every assignment the engine schedules:

1. ``TASK_ARRIVE``   - the resource completes its travel leg and is on site.
2. ``TASK_START``    - work begins (must be after arrival and release).
3. ``TASK_COMPLETE`` - work ends, the resource is free.

A disruption is injected as a ``DISRUPTION`` event plus whatever world change it
implies (``RESOURCE_FAIL``, ``MAINTENANCE_START``, ``NEW_TASK``). When a
disruption invalidates work that is *already in progress*, the engine emits
``TASK_INTERRUPTED`` semantics by cancelling the pending completion and marking
the assignment as ``INTERRUPTED`` - the task is not silently completed.

That cancellation is the important part. A simulator that keeps completing tasks
on a failed vehicle produces a trace that looks fine and is fiction, and the
recovery numbers computed from it would be meaningless.

Disruption injection
--------------------

Disruptions are applied at their declared timestamp, once. A test asserts that
running the same disruption twice does not double-apply it: the engine tracks
applied disruption ids in the run and skips duplicates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from orion.domain.entities import (
    Resource,
    ResourceStatus,
    Scenario,
    Task,
    TaskStatus,
)
from orion.domain.events import Disruption, DisruptionType
from orion.domain.plans import Assignment, Plan
from orion.domain.time_model import Interval, hhmm
from orion.simulation.events import EVENT_LABELS, EventQueue, EventType, SimEvent

#: Signature of a replanning callback: (disruption, current world snapshot) -> Plan | None
ReplanCallback = Callable[[Disruption, "WorldState"], Plan | None]


@dataclass(slots=True)
class ResourceState:
    """Live state of one resource during a run."""

    resource_id: str
    status: str = ResourceStatus.AVAILABLE
    location: str = ""
    current_task: str | None = None
    available_at: int = 0
    completed_tasks: tuple[str, ...] = ()
    failed_at: int | None = None
    working_minutes: int = 0
    travel_minutes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "status": self.status,
            "location": self.location,
            "current_task": self.current_task,
            "available_at": self.available_at,
            "completed_tasks": list(self.completed_tasks),
            "failed_at": self.failed_at,
            "working_minutes": self.working_minutes,
            "travel_minutes": self.travel_minutes,
        }


@dataclass(slots=True)
class TaskState:
    """Live state of one task during a run."""

    task_id: str
    status: str = TaskStatus.PENDING
    assigned_resource: str | None = None
    started_at: int | None = None
    completed_at: int | None = None
    late: bool = False
    interrupted: bool = False
    work_done: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "assigned_resource": self.assigned_resource,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "late": self.late,
            "interrupted": self.interrupted,
            "work_done": self.work_done,
        }


@dataclass(slots=True)
class WorldState:
    """Complete live world state at a point in simulated time."""

    now: int = 0
    resources: dict[str, ResourceState] = field(default_factory=dict)
    tasks: dict[str, TaskState] = field(default_factory=dict)
    applied_disruptions: set[str] = field(default_factory=set)
    scenario: Scenario | None = None

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable snapshot, used by replanning callbacks and the UI."""
        return {
            "now": self.now,
            "clock": hhmm(self.now),
            "resources": [s.to_dict() for s in self.resources.values()],
            "tasks": [s.to_dict() for s in self.tasks.values()],
            "applied_disruptions": sorted(self.applied_disruptions),
        }

    def resource(self, resource_id: str) -> ResourceState:
        return self.resources[resource_id]

    def task(self, task_id: str) -> TaskState:
        return self.tasks[task_id]

    def idle_resources(self) -> list[str]:
        return sorted(
            rid
            for rid, state in self.resources.items()
            if state.status == ResourceStatus.AVAILABLE and state.current_task is None
        )

    def active_tasks(self) -> list[str]:
        return sorted(
            tid for tid, state in self.tasks.items() if state.status == TaskStatus.IN_PROGRESS
        )

    def completed_count(self) -> int:
        return sum(1 for s in self.tasks.values() if s.status == TaskStatus.COMPLETED)

    def late_count(self) -> int:
        return sum(1 for s in self.tasks.values() if s.late)


@dataclass(slots=True)
class SimulationResult:
    """Everything one simulation run produces."""

    trace: tuple[SimEvent, ...]
    final_state: WorldState
    final_plan: Plan
    start_time: int
    end_time: int
    completed_tasks: int
    late_tasks: int
    failed_tasks: int
    replans_performed: int
    runtime_s: float
    status: str = "COMPLETED"

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    def events_between(self, start: int, end: int) -> list[SimEvent]:
        return [e for e in self.trace if start <= e.time <= end]

    def disruptions_in_trace(self) -> list[SimEvent]:
        return [e for e in self.trace if e.type == EventType.DISRUPTION]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "completed_tasks": self.completed_tasks,
            "late_tasks": self.late_tasks,
            "failed_tasks": self.failed_tasks,
            "replans_performed": self.replans_performed,
            "runtime_s": round(self.runtime_s, 6),
            "status": self.status,
            "event_count": len(self.trace),
            "final_state": self.final_state.snapshot(),
            "trace": [e.to_dict() for e in self.trace],
        }

    def trace_csv(self) -> list[list[Any]]:
        return [
            [
                "time",
                "clock",
                "event_type",
                "subject_id",
                "resource_id",
                "detail",
            ]
            * [e.to_csv_row() for e in self.trace]
        ]

    def summary_line(self) -> str:
        return (
            f"sim {self.start_time}-{self.end_time} ({hhmm(self.start_time)}-"
            f"{hhmm(self.end_time)}): {self.completed_tasks} completed, "
            f"{self.late_tasks} late, {self.failed_tasks} failed, "
            f"{self.replans_performed} replan(s), {len(self.trace)} events"
        )


class SimulationEngine:
    """Executes a plan over simulated time, with disruption injection."""

    def __init__(
        self,
        scenario: Scenario,
        plan: Plan,
        *,
        disruptions: Sequence[Disruption] = (),
        replan_callback: ReplanCallback | None = None,
        respect_failures: bool = True,
    ) -> None:
        self.scenario = scenario
        self.plan = plan
        self.disruptions = sorted(disruptions, key=lambda d: (d.timestamp, d.id))
        self.replan_callback = replan_callback
        self.respect_failures = respect_failures
        # Per-instance state. These were class attributes in an earlier draft,
        # which meant two concurrent simulations would corrupt each other; the
        # API serves requests concurrently, so they must not be shared.
        self._dynamic_tasks: dict[str, Task] = {}
        self._latest_plan: Plan | None = None
        self._latest_scenario: Scenario | None = None

    # -- setup -------------------------------------------------------------
    def _initial_world(self) -> WorldState:
        world = WorldState(
            now=self.scenario.horizon.start,
            scenario=self.scenario,
        )
        for resource in self.scenario.resources:
            world.resources[resource.id] = ResourceState(
                resource_id=resource.id,
                status=resource.status,
                location=resource.home_location,
                available_at=resource.shift_start,
            )
        for task in self.scenario.tasks:
            world.tasks[task.id] = TaskState(task_id=task.id, status=TaskStatus.PENDING)
        return world

    def _seed_plan_events(self, queue: EventQueue, world: WorldState) -> None:
        """Schedule arrival/start/complete for every assignment in the plan."""
        for a in self.plan.assignments:
            task_state = world.tasks.get(a.task_id)
            resource_state = world.resources.get(a.resource_id)
            if task_state is None or resource_state is None:
                continue
            if a.start >= a.end:
                continue
            task_state.assigned_resource = a.resource_id
            task_state.status = TaskStatus.ASSIGNED
            queue.push(
                SimEvent(
                    time=a.start - a.travel_before,
                    type=EventType.TASK_ARRIVE,
                    subject_id=a.task_id,
                    resource_id=a.resource_id,
                    detail=(
                        f"{a.resource_id} arrives at {a.location} for {a.task_id}"
                        f" (travel {a.travel_before} min)"
                    ),
                    payload={"location": a.location, "travel_minutes": a.travel_before},
                )
            )
            queue.push(
                SimEvent(
                    time=a.start,
                    type=EventType.TASK_START,
                    subject_id=a.task_id,
                    resource_id=a.resource_id,
                    detail=f"{a.resource_id} starts {a.task_id} at {a.location}",
                    payload={"location": a.location},
                )
            )
            queue.push(
                SimEvent(
                    time=a.end,
                    type=EventType.TASK_COMPLETE,
                    subject_id=a.task_id,
                    resource_id=a.resource_id,
                    detail=f"{a.resource_id} completes {a.task_id}",
                )
            )

    def _seed_disruption_events(self, queue: EventQueue, world: WorldState) -> None:
        for disruption in self.disruptions:
            queue.push(
                SimEvent(
                    time=disruption.timestamp,
                    type=EventType.DISRUPTION,
                    subject_id=disruption.target_id,
                    detail=disruption.summary(),
                    payload={"disruption": disruption.to_dict()},
                )
            )
            self._seed_disruption_effect(queue, world, disruption)

    def _seed_disruption_effect(
        self, queue: EventQueue, world: WorldState, disruption: Disruption
    ) -> None:
        """Schedule the world-state change a disruption implies."""
        dtype = disruption.type
        if dtype in {DisruptionType.VEHICLE_FAILURE, DisruptionType.RESOURCE_UNAVAILABLE}:
            if disruption.target_id:
                queue.push(
                    SimEvent(
                        time=disruption.timestamp,
                        type=EventType.RESOURCE_FAIL,
                        subject_id=None,
                        resource_id=disruption.target_id,
                        detail=f"{disruption.target_id} out of service",
                        payload={"disruption_id": disruption.id},
                    )
                )
        elif dtype == DisruptionType.MAINTENANCE_EVENT and disruption.target_id:
            queue.push(
                SimEvent(
                    time=disruption.timestamp,
                    type=EventType.MAINTENANCE_START,
                    subject_id=None,
                    resource_id=disruption.target_id,
                    detail=f"{disruption.target_id} enters maintenance",
                    payload={"disruption_id": disruption.id},
                )
            )
            if disruption.duration:
                queue.push(
                    SimEvent(
                        time=disruption.timestamp + disruption.duration,
                        type=EventType.MAINTENANCE_END,
                        subject_id=None,
                        resource_id=disruption.target_id,
                        detail=f"{disruption.target_id} leaves maintenance",
                        payload={"disruption_id": disruption.id},
                    )
                )
        elif dtype == DisruptionType.RESOURCE_RETURN and disruption.target_id:
            end = (
                disruption.timestamp + disruption.duration
                if disruption.duration
                else self.scenario.horizon.end
            )
            queue.push(
                SimEvent(
                    time=end,
                    type=EventType.RESOURCE_RETURN,
                    subject_id=None,
                    resource_id=disruption.target_id,
                    detail=f"{disruption.target_id} returns to service",
                    payload={"disruption_id": disruption.id},
                )
            )
        elif dtype == DisruptionType.NEW_URGENT_TASK:
            task_payload = disruption.payload.get("task") if disruption.payload else None
            if task_payload:
                queue.push(
                    SimEvent(
                        time=disruption.timestamp,
                        type=EventType.NEW_TASK,
                        subject_id=str(task_payload.get("task_id")),
                        detail=f"urgent task {task_payload.get('task_id')} appears",
                        payload={"disruption_id": disruption.id, "task": dict(task_payload)},
                    )
                )
        elif dtype == DisruptionType.DEMAND_SURGE:
            for spec in (disruption.payload.get("tasks") if disruption.payload else []) or []:
                queue.push(
                    SimEvent(
                        time=disruption.timestamp,
                        type=EventType.NEW_TASK,
                        subject_id=str(spec.get("task_id")),
                        detail=f"surge task {spec.get('task_id')} appears",
                        payload={"disruption_id": disruption.id, "task": dict(spec)},
                    )
                )
        elif dtype == DisruptionType.DEADLINE_CHANGE:
            task_id = str((disruption.payload or {}).get("task", ""))
            if task_id:
                queue.push(
                    SimEvent(
                        time=disruption.timestamp,
                        type=EventType.DISRUPTION,
                        subject_id=task_id,
                        detail=f"deadline for {task_id} changed",
                        payload={"disruption_id": disruption.id},
                    )
                )
        elif dtype == DisruptionType.TRAVEL_TIME_INCREASE:
            queue.push(
                SimEvent(
                    time=disruption.timestamp,
                    type=EventType.DISRUPTION,
                    detail=f"congestion x{disruption.magnitude:.2f}",
                    payload={"disruption_id": disruption.id},
                )
            )
        elif dtype == DisruptionType.CAPACITY_REDUCTION:
            queue.push(
                SimEvent(
                    time=disruption.timestamp,
                    type=EventType.DISRUPTION,
                    subject_id=disruption.target_id,
                    detail=f"{disruption.target_id} capacity reduced",
                    payload={"disruption_id": disruption.id},
                )
            )
        _ = world

    # -- main loop ---------------------------------------------------------
    def run(self, *, until: int | None = None, max_events: int = 1_000_000) -> SimulationResult:
        started = time.perf_counter()
        world = self._initial_world()
        queue = EventQueue()
        self._seed_plan_events(queue, world)
        self._seed_disruption_events(queue, world)

        end = until if until is not None else self.scenario.horizon.end
        trace: list[SimEvent] = []
        processed = 0
        replans = 0
        current_plan = self.plan
        current_scenario = self.scenario

        while len(queue) and processed < max_events:
            next_time = queue.peek_time()
            if next_time is None or next_time > end:
                break
            event = queue.pop()
            if event is None:  # pragma: no cover - guarded by len(queue)
                break
            world.now = max(world.now, event.time)
            processed += 1

            handler = self._HANDLERS.get(event.type)
            if handler is not None:
                emitted = handler(self, event, world, queue, current_plan, current_scenario)
                if emitted:
                    trace.extend(emitted)
                    # a replan may replace the plan; refresh the local reference
                    if any(e.type == EventType.REPLAN_END for e in emitted):
                        replans += 1
                        if self._latest_plan is not None:
                            current_plan = self._latest_plan
                        if self._latest_scenario is not None:
                            current_scenario = self._latest_scenario
            trace.append(event)

        completed = sum(1 for s in world.tasks.values() if s.status == TaskStatus.COMPLETED)
        late = sum(1 for s in world.tasks.values() if s.late)
        failed = sum(1 for s in world.tasks.values() if s.interrupted)
        return SimulationResult(
            trace=tuple(trace),
            final_state=world,
            final_plan=current_plan,
            start_time=self.scenario.horizon.start,
            end_time=world.now,
            completed_tasks=completed,
            late_tasks=late,
            failed_tasks=failed,
            replans_performed=replans,
            runtime_s=time.perf_counter() - started,
            status="COMPLETED" if processed < max_events else "EVENT_LIMIT",
        )

    # -- event handlers ----------------------------------------------------
    def _on_task_arrive(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        assert event.resource_id is not None
        state = world.resources[event.resource_id]
        state.location = str(event.payload.get("location", state.location))
        state.travel_minutes += int(event.payload.get("travel_minutes", 0))
        # The event loop already appends the popped event to the trace, so a
        # handler must return only *additional* events. Re-emitting the arrival
        # here would duplicate it in the trace.
        return []

    def _on_task_start(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        task_state = world.tasks[event.subject_id]  # type: ignore[index]
        resource_state = world.resources[event.resource_id]  # type: ignore[index]
        if self.respect_failures and resource_state.status in {
            ResourceStatus.FAILED,
            ResourceStatus.UNAVAILABLE,
            ResourceStatus.MAINTENANCE,
        }:
            # the world moved on: the task cannot start
            queue.cancel_for(subject_id=event.subject_id, types=(EventType.TASK_COMPLETE,))
            task_state.status = TaskStatus.UNASSIGNED
            task_state.assigned_resource = None
            return [
                SimEvent(
                    time=event.time,
                    type=EventType.DISRUPTION,
                    subject_id=event.subject_id,
                    resource_id=event.resource_id,
                    detail=(
                        f"{event.subject_id} did not start: "
                        f"{event.resource_id} is {resource_state.status}"
                    ),
                    payload={"reason": "resource_unavailable_at_start"},
                )
            ]
        task_state.status = TaskStatus.IN_PROGRESS
        task_state.started_at = event.time
        task_state.assigned_resource = event.resource_id
        resource_state.status = ResourceStatus.ON_TASK
        resource_state.current_task = event.subject_id
        return []

    def _on_task_complete(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        task_state = world.tasks[event.subject_id]  # type: ignore[index]
        resource_state = world.resources[event.resource_id]  # type: ignore[index]
        task_state.status = TaskStatus.COMPLETED
        task_state.completed_at = event.time
        resource_state.completed_tasks = (
            *resource_state.completed_tasks,
            str(event.subject_id),
        )
        resource_state.status = ResourceStatus.AVAILABLE
        resource_state.current_task = None
        resource_state.available_at = event.time
        assignment = plan.assignment_for(str(event.subject_id)) if plan else None
        if assignment is not None:
            resource_state.working_minutes += assignment.duration
            task = self._find_task(str(event.subject_id))
            if task is not None and event.time > task.deadline:
                task_state.late = True
        return []

    def _on_resource_fail(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        assert event.resource_id is not None
        state = world.resources[event.resource_id]
        state.status = ResourceStatus.FAILED
        state.failed_at = event.time
        emitted: list[SimEvent] = []
        # cancel the work in progress: the vehicle cannot finish the job
        if state.current_task is not None:
            task_state = world.tasks[state.current_task]
            task_state.status = TaskStatus.UNASSIGNED
            task_state.interrupted = True
            task_state.assigned_resource = None
            queue.cancel_for(
                subject_id=state.current_task,
                types=(EventType.TASK_START, EventType.TASK_COMPLETE),
            )
            emitted.append(
                SimEvent(
                    time=event.time,
                    type=EventType.DISRUPTION,
                    subject_id=state.current_task,
                    resource_id=event.resource_id,
                    detail=f"{state.current_task} interrupted by failure of {event.resource_id}",
                    payload={"reason": "resource_failure"},
                )
            )
        state.current_task = None
        return emitted

    def _on_maintenance_start(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        assert event.resource_id is not None
        state = world.resources[event.resource_id]
        if state.current_task is not None:
            task_state = world.tasks[state.current_task]
            task_state.status = TaskStatus.UNASSIGNED
            task_state.interrupted = True
            task_state.assigned_resource = None
            queue.cancel_for(
                subject_id=state.current_task,
                types=(EventType.TASK_START, EventType.TASK_COMPLETE),
            )
        state.status = ResourceStatus.MAINTENANCE
        state.current_task = None
        return []

    def _on_maintenance_end(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        assert event.resource_id is not None
        state = world.resources[event.resource_id]
        if state.status == ResourceStatus.MAINTENANCE:
            state.status = ResourceStatus.AVAILABLE
        return []

    def _on_resource_return(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        assert event.resource_id is not None
        state = world.resources[event.resource_id]
        state.status = ResourceStatus.AVAILABLE
        return []

    def _on_new_task(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        """Register a task that appears mid-run.

        The task is added to world state and immediately triggers a replan, since
        an unassigned urgent task is exactly the situation the replanning loop
        exists for. If no replan callback is wired, the task simply sits
        unassigned and the trace records that.
        """
        task_payload = dict(event.payload.get("task") or {})
        task_id = event.subject_id or str(task_payload.get("task_id", ""))
        if not task_id:
            return []
        from orion.domain.capacity import CapacityDemand, parse_priority
        from orion.domain.entities import Task as DomainTask

        disruption_id = str(event.payload.get("disruption_id", ""))
        try:
            task = DomainTask(
                id=task_id,
                location=str(task_payload.get("location", scenario.depot)),
                release_time=int(task_payload.get("release_time", event.time)),
                deadline=int(task_payload.get("deadline", event.time + 120)),
                duration=int(task_payload.get("duration", 30)),
                priority=parse_priority(task_payload.get("priority", 4)),
                required_capabilities=frozenset(
                    task_payload.get("required_capabilities", ())
                ),
                required_capacity=tuple(
                    CapacityDemand(str(c[0]), float(c[1]))
                    for c in task_payload.get("required_capacity", ())
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a malformed task must not kill the run
            return [
                SimEvent(
                    time=event.time,
                    type=EventType.DISRUPTION,
                    subject_id=task_id,
                    detail=f"rejected new task {task_id}: {exc}",
                    payload={"reason": "invalid_new_task"},
                )
            ]
        world.tasks[task_id] = TaskState(task_id=task_id, status=TaskStatus.PENDING)
        self._dynamic_tasks[task_id] = task
        return self._request_replan(event, world, task_id, disruption_id)

    def _on_disruption(
        self, event: SimEvent, world: WorldState, queue: EventQueue,
        plan: Plan, scenario: Scenario,
    ) -> list[SimEvent]:
        payload = event.payload.get("disruption")
        disruption = None
        if isinstance(payload, Mapping):
            from orion.domain.events import Disruption as D

            try:
                disruption = D.from_dict(payload)
            except Exception:  # noqa: BLE001
                disruption = None
        if disruption is None:
            disruption = self._find_disruption(str(event.payload.get("disruption_id", "")))
        if disruption is None:
            return []
        if disruption.id in world.applied_disruptions:
            # exactly-once application, enforced and testable
            return []
        world.applied_disruptions.add(disruption.id)
        return self._request_replan(event, world, event.subject_id, disruption.id)

    def _find_task(self, task_id: str) -> Task | None:
        for task in self.scenario.tasks:
            if task.id == task_id:
                return task
        return self._dynamic_tasks.get(task_id)

    def _find_disruption(self, disruption_id: str) -> Disruption | None:
        for disruption in self.disruptions:
            if disruption.id == disruption_id:
                return disruption
        return None

    def _request_replan(
        self, event: SimEvent, world: WorldState, task_id: str | None, disruption_id: str
    ) -> list[SimEvent]:
        """Invoke the replan callback and emit REPLAN_START/END around it."""
        if self.replan_callback is None:
            return []
        disruption = self._find_disruption(disruption_id)
        if disruption is None:
            disruption = Disruption(
                id=disruption_id or "D-UNKNOWN",
                type=DisruptionType.NEW_URGENT_TASK,
                timestamp=event.time,
                payload={"task": {"task_id": task_id}} if task_id else {},
            )
        start = SimEvent(
            time=event.time,
            type=EventType.REPLAN_START,
            subject_id=task_id,
            detail=f"replanning triggered by {disruption.id}",
            payload={"disruption_id": disruption.id},
        )
        try:
            new_plan = self.replan_callback(disruption, world)
        except Exception as exc:  # noqa: BLE001 - a failed replan is a trace event
            return [
                start,
                SimEvent(
                    time=event.time,
                    type=EventType.REPLAN_END,
                    subject_id=task_id,
                    detail=f"replan failed: {type(exc).__name__}: {exc}",
                    payload={"disruption_id": disruption.id, "success": False},
                ),
            ]
        if new_plan is None:
            return [
                start,
                SimEvent(
                    time=event.time,
                    type=EventType.REPLAN_END,
                    subject_id=task_id,
                    detail="replan produced no plan",
                    payload={"disruption_id": disruption.id, "success": False},
                ),
            ]
        self._latest_plan = new_plan
        candidate = (getattr(new_plan, "metadata", None) or {}).get("scenario")
        if isinstance(candidate, Scenario):
            self._latest_scenario = candidate
        return [
            start,
            SimEvent(
                time=event.time,
                type=EventType.REPLAN_END,
                subject_id=task_id,
                detail=(
                    f"replan complete: {new_plan.id} score={new_plan.score:.2f} "
                    f"assigned={new_plan.tasks_assigned}/{new_plan.tasks_total}"
                ),
                payload={
                    "disruption_id": disruption.id,
                    "success": True,
                    "plan_id": new_plan.id,
                    "score": round(new_plan.score, 4),
                },
            ),
        ]

    _HANDLERS: Mapping[EventType, Callable[..., list[SimEvent]]] = {}


SimulationEngine._HANDLERS = {  # type: ignore[attr-defined]
    EventType.TASK_ARRIVE: SimulationEngine._on_task_arrive,
    EventType.TASK_START: SimulationEngine._on_task_start,
    EventType.TASK_COMPLETE: SimulationEngine._on_task_complete,
    EventType.RESOURCE_FAIL: SimulationEngine._on_resource_fail,
    EventType.MAINTENANCE_START: SimulationEngine._on_maintenance_start,
    EventType.MAINTENANCE_END: SimulationEngine._on_maintenance_end,
    EventType.RESOURCE_RETURN: SimulationEngine._on_resource_return,
    EventType.NEW_TASK: SimulationEngine._on_new_task,
    EventType.DISRUPTION: SimulationEngine._on_disruption,
}


__all__ = [
    "SimulationEngine",
    "SimulationResult",
    "WorldState",
    "ResourceState",
    "TaskState",
    "ReplanCallback",
]
