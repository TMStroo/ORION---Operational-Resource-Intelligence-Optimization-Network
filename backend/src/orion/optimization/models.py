
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
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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
def dependency_bounds(
    context: SchedulingContext,
    sequences: dict[str, list[str]],
    *,
    max_rounds: int = 40,
) -> dict[str, int]:
    """Earliest start each placed task may take from its prerequisites.

    Materialising every resource's sequence and returning ``{task_id: end}`` is
    the only way to honour a dependency whose two endpoints sit on *different*
    resources. A per-resource scheduler cannot see across resources by
    construction, so this has to be computed at the plan level and fed back in.

    The pass repeats until no end time moves, because a single pass is not
    enough: sequencing a prerequisite late has to push its dependent out, which
    moves that dependent's own dependents, and so on. A 10-operation job chain
    needs several rounds to settle.

    Callers must pass the same per-resource ordering both here and when they
    apply the bounds, and must not discard a resource whose sequence comes back
    ``None`` from this helper: a bound that is missing only means "no information
    yet", not "this resource is empty". Dropping those resources is what turned a
    100/100-operation solution into 50/100 when this loop was first written.
    """
    #
    # The iteration must be monotone, and it is not automatic. A round in which
    # one resource's sequence no longer fits *removes that resource's tasks from
    # the bound map*, and those tasks are the prerequisites other resources were
    # being held back by - so the next round schedules them earlier and the whole
    # thing oscillates. On ft10 the bound map went 96, 56, 96, 56 tasks across
    # rounds and never converged.
    #
    # Monotonicity is restored by accumulating instead of replacing: a bound is
    # only ever raised, never lowered or removed, because `updated` is merged as
    # `max(existing, new)`. End times are then non-decreasing and bounded by the
    # horizon, so the loop terminates at a genuine fixed point. Tasks with no
    # entry carry no constraint at all, which is the correct reading of "this
    # resource could not place the task" - it is the caller's job to decide
    # whether that means dropping work, and `materialise_sequences` does.
    bounds: dict[str, int] = {}
    for _ in range(max_rounds):
        updated = dict(bounds)
        for resource_id in sorted(sequences):
            result = schedule_sequence(
                sequences[resource_id],
                context,
                resource_id,
                external_bounds=bounds or None,
            )
            if result is None:
                continue
            for a in result:
                previous = updated.get(a.task_id)
                updated[a.task_id] = (
                    a.end if previous is None else max(previous, a.end)
                )
        if updated == bounds:
            break
        bounds = updated
    return bounds


def validate_materialised(
    assignments: Sequence[Assignment], context: SchedulingContext
) -> list[str]:
    """Check a rebuilt schedule against the constraints it claims to honour.

    A solver's *model* may legitimately relax a constraint - MILP's precedence is
    enforced with a big-M term and continuous start variables, so it can return a
    solution that violates precedence. But a plan handed to a caller must be a
    real schedule. So the rebuilt plan is checked here, and a solver whose model
    produced something unreconstructable is told so rather than being allowed to
    present it as a successful result.

    This is deliberately independent of `schedule_sequence`: it re-derives the
    facts from the returned assignments rather than trusting the scheduler that
    produced them.
    """
    problems: list[str] = []
    scenario = context.scenario
    constraints = scenario.constraints
    by_task: dict[str, Assignment] = {}
    for a in assignments:
        if a.task_id in by_task:
            problems.append(f"{a.task_id} assigned more than once")
        by_task[a.task_id] = a

    # Deadline misses are a scored cost, not a hard violation: ORION's objective
    # prices lateness explicitly, so a late task is a valid plan. Only structural
    # errors - precedence, double-booking, shift, horizon - are reported here.

    if constraints.enforce_max_work:
        worked: dict[str, int] = {}
        for a in assignments:
            worked[a.resource_id] = worked.get(a.resource_id, 0) + (a.end - a.start)
        for resource_id, minutes in sorted(worked.items()):
            resource = context.resource_by_id(resource_id)
            if resource is None:
                continue
            excess = minutes - resource.max_work_minutes
            if excess > 0:
                problems.append(
                    f"{resource_id} is worked {minutes} minutes, over its limit of "
                    f"{resource.max_work_minutes}"
                )

    if constraints.enforce_dependencies:
        for task in context.tasks:
            a = by_task.get(task.id)
            if a is None:
                continue
            for dep in task.dependencies:
                parent = by_task.get(dep)
                if parent is None:
                    continue
                if a.start < parent.end:
                    problems.append(
                        f"{task.id} starts {a.start} before prerequisite {dep} ends {parent.end}"
                    )

    # one resource cannot be in two places at once
    per_resource: dict[str, list[Assignment]] = {}
    for a in assignments:
        per_resource.setdefault(a.resource_id, []).append(a)
    for resource_id, items in per_resource.items():
        items.sort(key=lambda a: a.start)
        for i in range(len(items) - 1):
            if items[i].end > items[i + 1].start:
                problems.append(
                    f"{resource_id} double-booked: {items[i].task_id} ends "
                    f"{items[i].end} but {items[i + 1].task_id} starts {items[i + 1].start}"
                )

    if constraints.enforce_shift:
        for a in assignments:
            resource = context.resource_by_id(a.resource_id)
            if resource is None:
                continue
            if a.start < resource.shift_start or a.end > resource.shift_end:
                problems.append(
                    f"{a.task_id} runs {a.start}-{a.end} outside {resource_id}'s shift "
                    f"{resource.shift_start}-{resource.shift_end}"
                )

    if constraints.enforce_travel:
        for a in assignments:
            if a.end > scenario.horizon.end:
                problems.append(
                    f"{a.task_id} ends at {a.end}, past the horizon {scenario.horizon.end}"
                )

    return problems


def order_within_resource(
    sequence: Sequence[str], context: SchedulingContext
) -> list[str]:
    """Order *sequence* so no task precedes a prerequisite sharing its resource.

    A stable topological sort restricted to the dependencies that are *inside*
    the sequence. `schedule_sequence` rejects a sequence that lists a task before
    one of its own prerequisites, so without this a resource whose model solution
    arrived in arbitrary order could be thrown away wholesale.

    Tasks on a chain keep their relative order; independent tasks keep the order
    they were given in.
    """
    members = set(sequence)
    indegree: dict[str, int] = {t: 0 for t in sequence}
    successors: dict[str, list[str]] = {t: [] for t in sequence}
    for task_id in sequence:
        task = context.task_by_id(task_id)
        if task is None:
            continue
        for dep in task.dependencies:
            if dep in members and dep != task_id:
                successors[dep].append(task_id)
                indegree[task_id] += 1
    # ready tasks in the order they were supplied keeps the sort deterministic and
    # as close to the caller's intent as the constraints allow
    ready = [t for t in sequence if indegree[t] == 0]
    out: list[str] = []
    remaining = set(indegree)
    while ready:
        current = ready.pop(0)
        out.append(current)
        remaining.discard(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    # a cycle inside the sequence cannot be scheduled; keep those tasks in their
    # original order so schedule_sequence can reject them explicitly rather than
    # having them silently vanish
    out.extend(t for t in sequence if t in remaining)
    return out


def materialise_sequences(
    context: SchedulingContext,
    sequences: Mapping[str, Sequence[str]],
    *,
    order: Mapping[str, Sequence[str]] | None = None,
) -> list[Assignment]:
    """Schedule every resource's sequence, tolerating partial failure.

    ``schedule_sequence`` returns ``None`` when a whole sequence cannot fit -
    a shift or travel infeasibility. Callers that treat that as "this resource
    contributes nothing" silently delete work the solver had already decided to
    do. On the job-shop ft10 that cost the constructive heuristic 80 of its 100
    operations, which is not a scheduling decision but a crash in the
    reconstruction step.

    So a failed sequence falls back to its longest feasible prefix, exactly as
    the local repairer does. The prefix is found by dropping trailing tasks, so
    an infeasible *middle* still costs the rest of that resource's route; the
    common case, a route that simply runs past the end of the shift, is fixed.
    """
    # Every solver funnels through here, so the per-resource ordering is decided
    # in ONE place rather than in each solver's extraction step. They disagreed:
    # CP-SAT sorted by the model's start variable and MILP did not sort at all,
    # so two solvers could hand the same assignment set to the same scheduler in
    # different orders and get different schedules - which is exactly what the
    # job-shop run showed (MILP 42 precedence violations against CP-SAT's 0).
    #
    # The order used is topological within a resource, ties broken by task id:
    # a task can never be sequenced before a prerequisite that happens to share
    # its resource, which is the one ordering that cannot be wrong regardless of
    # what the model decided. Any model-specific start ordering is then a
    # refinement, not a correctness requirement.
    # Within-resource ordering is necessary but not sufficient. A job's operations
    # are spread across *different* resources, so no single resource's list holds
    # its own chain and the topological sort has nothing to act on. The ordering
    # that matters is the one the *solver's model* chose, and only the solver has
    # it - it is expressed in model variables, not in task ids. So a caller that
    # knows the model's own sequence order passes it in as `order`; without it,
    # sequences are kept in the order supplied and only repaired within a resource.
    #
    # CP-SAT used to sort here while MILP did not, so the two fed identical
    # assignment sets to the same scheduler in different orders and got different
    # schedules - which is how MILP reported a 62-unit makespan against CP-SAT's
    # valid 93 on ft06. Deciding the order once, from the caller that owns the
    # model, is the only place where it can be decided correctly.
    prepared: dict[str, list[str]] = {}
    for resource_id, seq in sequences.items():
        if not seq:
            continue
        ranked = order.get(resource_id) if order else None
        members = set(seq)
        ordered = (
            [t for t in ranked if t in members] + [t for t in seq if t not in members]
            if ranked
            else list(seq)
        )
        prepared[resource_id] = order_within_resource(ordered, context)

    bounds = dependency_bounds(context, prepared)
    out: list[Assignment] = []
    for resource_id in sorted(prepared):
        sequence = list(prepared[resource_id])
        if not sequence:
            continue
        result = schedule_sequence(
            sequence, context, resource_id, external_bounds=bounds
        )
        while result is None and sequence:
            sequence = sequence[:-1]
            result = schedule_sequence(
                sequence, context, resource_id, external_bounds=bounds
            )
        if result:
            out.extend(result)
    return out


def schedule_sequence(
    sequence: Sequence[str],
    context: SchedulingContext,
    resource_id: str,
    *,
    start_at: int | None = None,
    external_bounds: Mapping[str, int] | None = None,
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

    ``external_bounds`` maps an already-scheduled task to its finish time on
    *another* resource, and is how cross-resource precedence is enforced. A
    dependency resolved inside *this* sequence is handled locally; one resolved
    elsewhere can only be seen through this parameter, which is why every caller
    that schedules a subset of a plan must supply it.
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
    # a sequence that lists a task before its own prerequisite cannot be fixed by
    # earliest-start scheduling alone, so it is rejected outright
    if constraints.enforce_dependencies:
        seen: set[str] = set()
        for task_id in sequence:
            task = context.task_by_id(task_id)
            if task is None:
                return None
            for dep in task.dependencies:
                if dep in sequence and dep not in seen:
                    return None
            seen.add(task_id)

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
                # a prerequisite on this resource: already in `assignments`
                dep_assignment = next(
                    (a for a in assignments if a.task_id == dep), None
                )
                if dep_assignment is not None:
                    start = max(start, dep_assignment.end)
                elif external_bounds is not None and dep in external_bounds:
                    # a prerequisite on a different resource
                    start = max(start, external_bounds[dep])
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
    "dependency_bounds",
    "SchedulingContext",
    "schedule_sequence",
    "sequence_is_feasible",
    "assignments_are_consistent",
    "SolverRequest",
    "Solver",
    "SolveOutcome",
]
