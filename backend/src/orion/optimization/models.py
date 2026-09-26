
"""Shared optimisation model: candidate sets, scheduling context, solver protocol.

This module defines *what* ORION optimises, once. The CP-SAT model, the MILP
model, the flow relaxation, the heuristic and the local search all consume the
same :class:`SchedulingContext`, so a difference in their results is a difference
in their *search*, not a difference in the problem they were given.

That is the single most important fairness property in ORION and it is enforced
structurally: there is one place where candidate pairs, feasible time windows and
travel coefficients are computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from orion.domain.entities import Resource, Scenario, Task
from orion.domain.plans import Assignment, Plan, SolverRun, SolverStatus
from orion.domain.time_model import Interval


@dataclass(frozen=True, slots=True)
class PairInfo:
    """Precomputed per-(task, resource) constants.

    Building these once is the main reason ORION's exact models stay small
    enough to solve: every coefficient a model row needs is already an integer
    here, so model construction is a loop over tuples rather than repeated
    dictionary lookups and travel-matrix queries.
    """

    task_id: str
    resource_id: str
    task_index: int
    resource_index: int
    #: travel minutes from the resource's home to the task, at its speed
    travel_from_home: int
    distance_from_home: float
    earliest_start: int
    latest_start: int
    duration: int
    deadline: int
    value: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.task_id, self.resource_id)


@dataclass(slots=True)
class SchedulingContext:
    """Everything a solver needs, computed once per optimisation run."""

    scenario: Scenario
    pairs: tuple[PairInfo, ...]
    tasks: tuple[Task, ...]
    resources: tuple[Resource, ...]
    task_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    pair_index: dict[tuple[str, str], int] = field(default_factory=dict)
    task_pairs: dict[str, list[int]] = field(default_factory=dict)
    resource_pairs: dict[str, list[int]] = field(default_factory=dict)
    build_seconds: float = 0.0

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, scenario: Scenario) -> SchedulingContext:
        """Enumerate candidate pairs and precompute coefficients.

        Raises ``InfeasibleScenarioError`` when a task has no eligible resource
        and the scenario does not permit unassigned tasks - that combination is
        provably infeasible and there is no point paying for a solver to prove it.
        """
        from orion.domain.errors import InfeasibleScenarioError

        started = time.perf_counter()
        tasks = scenario.tasks
        resources = scenario.resources
        task_ids = tuple(t.id for t in tasks)
        resource_ids = tuple(r.id for r in resources)
        task_pos = {tid: i for i, tid in enumerate(task_ids)}
        resource_pos = {rid: i for i, rid in enumerate(resource_ids)}
        constraints = scenario.constraints
        weights = scenario.objective_weights
        travel = scenario.travel
        horizon = scenario.horizon

        from orion.optimization.objective import task_value

        pairs: list[PairInfo] = []
        for task in tasks:
            if task.status not in {"PENDING", "ASSIGNED", "UNASSIGNED", "IN_PROGRESS"}:
                continue
            value = task_value(task, weights).total
            for resource in resources:
                if not constraints.allow_unassigned and False:
                    pass
                if constraints.enforce_capability or constraints.enforce_capacity:
                    if not resource.can_serve(task):
                        continue
                if resource.is_globally_unavailable:
                    continue
                home_travel = travel.travel_from_speed(
                    resource.home_location, task.location, resource.speed_factor
                )
                home_distance = travel.distance_from_speed(
                    resource.home_location, task.location, resource.speed_factor
                )
                # earliest start: after travel from home, after release, after deps
                earliest = max(
                    task.release_time,
                    home_travel,
                    resource.shift_start,
                    horizon.start,
                )
                # latest start that still respects shift and horizon
                latest = min(task.deadline, resource.shift_end - task.duration, horizon.end - task.duration)
                if latest < earliest:
                    # cannot be served by this resource even if started instantly
                    continue
                pairs.append(
                    PairInfo(
                        task_id=task.id,
                        resource_id=resource.id,
                        task_index=task_pos[task.id],
                        resource_index=resource_pos[resource.id],
                        travel_from_home=home_travel,
                        distance_from_home=home_distance,
                        earliest_start=earliest,
                        latest_start=latest,
                        duration=task.duration,
                        deadline=task.deadline,
                        value=value,
                    )
                )

        # dependency-aware earliest start (needs all tasks, so second pass)
        by_id = {t.id: t for t in tasks}
        for pair in pairs:
            task = by_id[pair.task_id]
            dep_bound = 0
            for dep in task.dependencies:
                parent = by_id.get(dep)
                if parent is None:
                    continue
                dep_bound = max(dep_bound, parent.release_time + parent.duration)
            if dep_bound > pair.earliest_start:
                new_earliest = dep_bound
                if new_earliest > pair.latest_start:
                    # the prerequisite pushes this past the deadline for this resource
                    object.__setattr__(pair, "earliest_start", pair.earliest_start)
                    continue
                object.__setattr__(pair, "earliest_start", new_earliest)

        eligible = set()
        for pair in pairs:
            eligible.add(pair.key)
        if not constraints.allow_unassigned or constraints.max_unassigned_fraction < 1.0:
            min_covered = int(
                len(tasks) * (1.0 - constraints.max_unassigned_fraction) + 0.999999
            )
            unservable = [t.id for t in tasks if not any(p.task_id == t.id for p in pairs)]
            if len(unservable) > len(tasks) - min_covered:
                raise InfeasibleScenarioError(
                    f"scenario {scenario.id}: {len(unservable)} task(s) have no feasible "
                    f"resource, but the scenario allows at most "
                    f"{len(tasks) - min_covered} unassigned task(s). "
                    f"Examples: {unservable[:5]}"
                )

        pair_index = {p.key: i for i, p in enumerate(pairs)}
        task_pairs: dict[str, list[int]] = {tid: [] for tid in task_ids}
        resource_pairs: dict[str, list[int]] = {rid: [] for rid in resource_ids}
        for i, pair in enumerate(pairs):
            task_pairs[pair.task_id].append(i)
            resource_pairs[pair.resource_id].append(i)

        return cls(
            scenario=scenario,
            pairs=tuple(pairs),
            tasks=tasks,
            resources=resources,
            task_ids=task_ids,
            resource_ids=resource_ids,
            pair_index=pair_index,
            task_pairs=task_pairs,
            resource_pairs=resource_pairs,
            build_seconds=time.perf_counter() - started,
        )

    # -- accessors ---------------------------------------------------------
    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_resources(self) -> int:
        return len(self.resources)

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)

    def pair(self, task_id: str, resource_id: str) -> PairInfo | None:
        index = self.pair_index.get((task_id, resource_id))
        return None if index is None else self.pairs[index]

    def travel_between(
        self, from_location: str, to_location: str, speed_factor: float
    ) -> int:
        return self.scenario.travel.travel_from_speed(from_location, to_location, speed_factor)

    def distance_between(self, from_location: str, to_location: str, speed_factor: float) -> float:
        return self.scenario.travel.distance(from_location, to_location) / max(1e-9, speed_factor)

    def speed_of(self, resource_id: str) -> float:
        for resource in self.resources:
            if resource.id == resource_id:
                return resource.speed_factor
        return 1.0

    def resource_by_id(self, resource_id: str) -> Resource | None:
        for resource in self.resources:
            if resource.id == resource_id:
                return resource
        return None

    def task_by_id(self, task_id: str) -> Task | None:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    @property
    def model_size(self) -> dict[str, int]:
        return {
            "tasks": self.num_tasks,
            "resources": self.num_resources,
            "candidate_pairs": self.num_pairs,
        }

    def describe(self) -> str:
        return (
            f"{self.num_tasks} tasks x {self.num_resources} resources -> "
            f"{self.num_pairs} candidate pairs (built in {self.build_seconds * 1000:.1f} ms)"
        )


# ==========================================================================
# Feasibility helpers shared by every solver
# ==========================================================================
def schedule_sequence(
    sequence: Sequence[str],
    context: SchedulingContext,
    resource_id: str,
    *,
    start_at: int | None = None,
) -> list[Assignment] | None:
    """Greedily schedule *sequence* on *resource_id* as early as possible.

    Returns ``None`` when the sequence cannot be scheduled feasibly (a travel or
    shift infeasibility). This function is the *reference implementation* of the
    sequencing rules: the CP-SAT and MILP models are validated against it, and
    the heuristic uses it directly. If a solver finds a plan the reference
    implementation cannot reproduce, the solver is wrong.

    Earliest-start scheduling is optimal for a *fixed* sequence when travel times
    satisfy the triangle inequality, which the Euclidean travel matrix does. That
    is why ORION's travel model is a metric: it makes earliest-start insertion
    sound.
    """
    scenario = context.scenario
    resource = context.resource_by_id(resource_id)
    if resource is None:
        return None
    travel = scenario.travel
    constraints = scenario.constraints
    speed = resource.speed_factor

    cursor_location = resource.home_location
    cursor_time = start_at if start_at is not None else max(
        resource.shift_start, scenario.horizon.start
    )
    # respect existing maintenance blocks: jump past them when needed
    assignments: list[Assignment] = []
    for position, task_id in enumerate(sequence):
        task = context.task_by_id(task_id)
        if task is None:
            return None
        travel_minutes = (
            travel.travel_from_speed(cursor_location, task.location, speed)
            if constraints.enforce_travel
            else 0
        )
        distance = (
            travel.distance(cursor_location, task.location)
            if constraints.enforce_travel
            else 0.0
        )
        arrival = cursor_time + travel_minutes
        start = max(arrival, task.release_time, resource.shift_start, scenario.horizon.start)
        if constraints.enforce_dependencies:
            for dep in task.dependencies:
                dep_assignment = next(
                    (a for a in assignments if a.task_id == dep), None
                )
                if dep_assignment is not None:
                    start = max(start, dep_assignment.end)
        end = start + task.duration
        if constraints.enforce_shift and end > resource.shift_end:
            return None
        if end > scenario.horizon.end:
            return None
        if constraints.enforce_maintenance:
            # push past any blocking maintenance window
            for block in resource.unavailable:
                if Interval(start, end).overlaps(block):
                    start = block.end
                    end = start + task.duration
                    if end > resource.shift_end or end > scenario.horizon.end:
                        return None
        if constraints.enforce_deadlines and task.max_lateness is not None:
            if task.lateness_for(end) > task.max_lateness:
                return None
        assignments.append(
            Assignment(
                task_id=task_id,
                resource_id=resource_id,
                start=start,
                end=end,
                location=task.location,
                sequence_index=position,
                travel_before=travel_minutes,
                travel_distance_km=round(distance, 4),
                priority=task.priority,
                lateness=task.lateness_for(end) if constraints.enforce_deadlines else 0,
            )
        )
        cursor_location = task.location
        cursor_time = end
    return assignments


def sequence_is_feasible(
    sequence: Sequence[str], context: SchedulingContext, resource_id: str
) -> bool:
    return schedule_sequence(sequence, context, resource_id) is not None


def assignments_are_consistent(
    assignments: Sequence[Assignment], context: SchedulingContext
) -> bool:
    """Cheap structural check used by every solver before it returns."""
    by_resource: dict[str, list[Assignment]] = {}
    seen_tasks: set[str] = set()
    for a in assignments:
        if a.task_id in seen_tasks:
            return False
        seen_tasks.add(a.task_id)
        by_resource.setdefault(a.resource_id, []).append(a)
    for resource_id, group in by_resource.items():
        ordered = sorted(group, key=lambda a: a.start)
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                return False
            if context.scenario.constraints.enforce_travel:
                needed = context.travel_between(
                    prev.location, nxt.location, context.speed_of(resource_id)
                )
                if nxt.start - prev.end < needed:
                    return False
    return True


# ==========================================================================
# Solver protocol
# ==========================================================================
@dataclass(frozen=True, slots=True)
class SolverRequest:
    """One optimisation request.

    ``time_limit_s`` is the budget in seconds. ``0`` or ``None`` means "no
    limit", which only the exact solvers honour meaningfully; the heuristic and
    local search interpret it as an iteration bound instead.
    """

    time_limit_s: float | None = 10.0
    seed: int = 0
    workers: int = 8
    #: Restrict this run to a subset of tasks, for local repair.
    task_filter: frozenset[str] | None = None
    #: Keep these existing assignments fixed, for local repair.
    locked_assignments: tuple[Assignment, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_limit_s is not None and self.time_limit_s < 0:
            raise ValueError(f"time_limit_s must be >= 0 or None, got {self.time_limit_s}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        object.__setattr__(self, "task_filter", frozenset(self.task_filter) if self.task_filter else None)
        object.__setattr__(self, "locked_assignments", tuple(self.locked_assignments))


@runtime_checkable
class Solver(Protocol):
    """The interface every ORION solver implements."""

    name: str

    def supports(self, context: SchedulingContext) -> bool:
        """Whether this solver can handle *context* (capability matrix)."""
        ...

    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[Plan, SolverRun]:
        """Solve and return a plan plus the run record."""
        ...


@dataclass(slots=True)
class SolveOutcome:
    """What a solver returns, with its own bookkeeping."""

    plan: Plan
    run: SolverRun
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.run.status in {
            SolverStatus.OPTIMAL,
            SolverStatus.FEASIBLE,
            SolverStatus.TIME_LIMIT,
        }


__all__ = [
    "PairInfo",
    "SchedulingContext",
    "schedule_sequence",
    "sequence_is_feasible",
    "assignments_are_consistent",
    "SolverRequest",
    "Solver",
    "SolveOutcome",
]
