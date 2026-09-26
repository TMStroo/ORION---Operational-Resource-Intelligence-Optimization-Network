
"""Plan representation: assignments, sequences, metrics and solver outcomes.

A :class:`Plan` is the unit a planner produces, a simulator executes, a replanner
compares against, and a user exports. It carries:

* the **assignments** (which resource does which task, when, where);
* the **objective breakdown** so a score is explainable rather than a black
  number;
* the **violations** the plan actually contains, with the exact reason;
* the **provenance** (solver, status, runtime, seed) so the plan is auditable.

Violations are *recorded, not hidden*. A plan that misses a deadline is legal
(as deadlines are soft) but it carries a ``DeadlineMissed`` violation, and the
API and UI both show the count. A plan that is structurally broken (a resource
double-booked) is illegal: :func:`validate_plan` rejects it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from orion.domain.capacity import Priority
from orion.domain.errors import ValidationError
from orion.domain.ids import validate_id
from orion.domain.time_model import Interval, hhmm


class SolverStatus(str):
    """Explicit solver outcome vocabulary (section 45 of the brief).

    These are strings, not an Enum, so they serialise to JSON identically in
    the API, the CLI and the experiment manifest.
    """

    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    TIME_LIMIT = "TIME_LIMIT"
    INFEASIBLE = "INFEASIBLE"
    ERROR = "ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    #: The solver proved optimality **of a relaxation of the true problem**, so
    #: the returned plan is feasible but is not known to be the best possible
    #: plan. This is the honest status for a min-cost-flow relaxation and for
    #: the MILP, whose formulation cannot see intra-route travel: SCIP genuinely
    #: proves its own model optimal, but that model is weaker than ORION's
    #: objective, so reporting plain ``OPTIMAL`` alongside CP-SAT would invite
    #: the reader to treat two different guarantees as one. The tiny-instance
    #: cross-checks in `orion.optimization.reference` demonstrate the
    #: difference: on the same fixtures the MILP reports OPTIMAL yet scores
    #: below the brute-force optimum.
    RELAXATION_OPTIMAL = "RELAXATION_OPTIMAL"


#: Statuses that mean "there is a usable plan to show the user".
USABLE_STATUSES = frozenset({
    SolverStatus.OPTIMAL,
    SolverStatus.FEASIBLE,
    SolverStatus.TIME_LIMIT,
    SolverStatus.RELAXATION_OPTIMAL,
})

#: Statuses that assert nothing beyond "this plan is feasible". Only plain
#: ``OPTIMAL`` claims a proven global optimum of the full objective.
PROVEN_OPTIMAL_STATUSES = frozenset({SolverStatus.OPTIMAL})


class SolverName(str):
    """Canonical solver identifiers used everywhere in ORION."""

    CP_SAT = "CP_SAT"
    MILP = "MILP"
    MIN_COST_FLOW = "MIN_COST_FLOW"
    HEURISTIC = "HEURISTIC"
    LOCAL_SEARCH = "LOCAL_SEARCH"


ALL_SOLVERS = (
    SolverName.CP_SAT,
    SolverName.MILP,
    SolverName.MIN_COST_FLOW,
    SolverName.HEURISTIC,
    SolverName.LOCAL_SEARCH,
)

#: Human-readable one-line descriptions used by `--help` and the UI solver page.
SOLVER_DESCRIPTIONS: Mapping[str, str] = {
    SolverName.CP_SAT: "Google OR-Tools CP-SAT: explicit integer decision variables, "
    "interval-based no-overlap, reports proven optimality and a bound.",
    SolverName.MILP: "OR-Tools linear solver (SCIP) on an explicit MILP formulation with "
    "binary assignment vars, time-indexed start vars and linearised travel. The model "
    "cannot express multi-stop route cost, so it optimises a relaxation of the full "
    "objective and reports RELAXATION_OPTIMAL rather than OPTIMAL.",
    SolverName.MIN_COST_FLOW: "Min-cost flow relaxation solved with OR-Tools SimpleMinCostFlow: "
    "optimal for capacity/assignment structure, no sequencing, so it reports "
    "RELAXATION_OPTIMAL.",
    SolverName.HEURISTIC: "ORION constructive heuristic: greedy by priority-density with "
    "regret-k insertion, cheapest-insertion repair, deterministic.",
    SolverName.LOCAL_SEARCH: "ORION local search: adaptive large-neighbourhood destroy-and-repair "
    "over the constructive solution.",
}


class ViolationKind(str):
    DEADLINE_MISSED = "DEADLINE_MISSED"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    MAINTENANCE_CONFLICT = "MAINTENANCE_CONFLICT"
    SHIFT_VIOLATION = "SHIFT_VIOLATION"
    RELEASE_TIME_VIOLATED = "RELEASE_TIME_VIOLATED"
    DEPENDENCY_VIOLATED = "DEPENDENCY_VIOLATED"
    TRAVEL_INFEASIBLE = "TRAVEL_INFEASIBLE"
    OVERLOAD = "OVERLOAD"
    UNASSIGNED_TASK = "UNASSIGNED_TASK"
    MAX_LATENESS_EXCEEDED = "MAX_LATENESS_EXCEEDED"


@dataclass(frozen=True, slots=True)
class Violation:
    """A single recorded constraint breach in a plan."""

    kind: str
    task_id: str | None
    resource_id: str | None
    severity: int  # 1 = soft (penalised), 2 = hard (plan is illegal)
    detail: str
    amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "severity": self.severity,
            "detail": self.detail,
            "amount": round(self.amount, 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Violation:
        return cls(
            kind=str(data["kind"]),
            task_id=data.get("task_id"),
            resource_id=data.get("resource_id"),
            severity=int(data.get("severity", 1)),
            detail=str(data.get("detail", "")),
            amount=float(data.get("amount", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    """One resource performing one task, with the schedule and its route leg.

    ``sequence_index`` fixes the order of tasks on a resource; the simulator and
    the timeline view use it, and the local-repair churn metric treats a change
    in order on an unchanged task as a (small) change.
    """

    task_id: str
    resource_id: str
    start: int
    end: int
    location: str
    sequence_index: int = 0
    travel_before: int = 0
    travel_distance_km: float = 0.0
    priority: Priority = Priority.MEDIUM
    lateness: int = 0

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def window(self) -> Interval:
        return Interval(self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "start": self.start,
            "end": self.end,
            "location": self.location,
            "sequence_index": self.sequence_index,
            "travel_before": self.travel_before,
            "travel_distance_km": round(self.travel_distance_km, 4),
            "priority": int(self.priority),
            "lateness": self.lateness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Assignment:
        return cls(
            task_id=str(data["task_id"]),
            resource_id=str(data["resource_id"]),
            start=int(data["start"]),
            end=int(data["end"]),
            location=str(data["location"]),
            sequence_index=int(data.get("sequence_index", 0)),
            travel_before=int(data.get("travel_before", 0)),
            travel_distance_km=float(data.get("travel_distance_km", 0.0)),
            priority=parse_priority_value(data.get("priority", 2)),
            lateness=int(data.get("lateness", 0)),
        )


def parse_priority_value(value: Any) -> Priority:
    from orion.domain.capacity import parse_priority

    return parse_priority(value)


@dataclass(slots=True)
class ObjectiveBreakdown:
    """Additive breakdown of the scalar service score.

    Every term here corresponds to exactly one objective coefficient in
    :class:`~orion.domain.entities.ObjectiveWeights`, so a score can always be
    reconstructed by summing these fields. A test enforces that.
    """

    completion: float = 0.0
    priority: float = 0.0
    lateness_penalty: float = 0.0
    travel_penalty: float = 0.0
    cost_penalty: float = 0.0
    dispatch_penalty: float = 0.0
    overload_penalty: float = 0.0
    idle_move_penalty: float = 0.0
    violation_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.completion
            + self.priority
            - self.lateness_penalty
            - self.travel_penalty
            - self.cost_penalty
            - self.dispatch_penalty
            - self.overload_penalty
            - self.idle_move_penalty
            - self.violation_penalty
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "completion": round(self.completion, 4),
            "priority": round(self.priority, 4),
            "lateness_penalty": round(self.lateness_penalty, 4),
            "travel_penalty": round(self.travel_penalty, 4),
            "cost_penalty": round(self.cost_penalty, 4),
            "dispatch_penalty": round(self.dispatch_penalty, 4),
            "overload_penalty": round(self.overload_penalty, 4),
            "idle_move_penalty": round(self.idle_move_penalty, 4),
            "violation_penalty": round(self.violation_penalty, 4),
            "total": round(self.total, 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectiveBreakdown:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: float(data.get(k, 0.0)) for k in known})  # type: ignore[arg-type]


@dataclass(slots=True)
class SolverRun:
    """Record of one solver invocation."""

    solver: str
    status: str
    runtime_s: float
    objective: float
    feasible: bool
    num_variables: int = 0
    num_constraints: int = 0
    optimality_gap: float | None = None
    time_limit_s: float | None = None
    num_iterations: int | None = None
    notes: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "status": self.status,
            "runtime_s": round(self.runtime_s, 6),
            "objective": round(self.objective, 4),
            "feasible": self.feasible,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "optimality_gap": (
                round(self.optimality_gap, 6) if self.optimality_gap is not None else None
            ),
            "time_limit_s": self.time_limit_s,
            "num_iterations": self.num_iterations,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SolverRun:
        return cls(
            solver=str(data["solver"]),
            status=str(data["status"]),
            runtime_s=float(data["runtime_s"]),
            objective=float(data["objective"]),
            feasible=bool(data["feasible"]),
            num_variables=int(data.get("num_variables", 0)),
            num_constraints=int(data.get("num_constraints", 0)),
            optimality_gap=(
                float(data["optimality_gap"])
                if data.get("optimality_gap") is not None
                else None
            ),
            time_limit_s=(
                float(data["time_limit_s"]) if data.get("time_limit_s") is not None else None
            ),
            num_iterations=(
                int(data["num_iterations"]) if data.get("num_iterations") is not None else None
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class ResourceUtilization:
    """Per-resource summary of how hard a plan works a resource."""

    resource_id: str
    assigned_tasks: int = 0
    worked_minutes: int = 0
    travel_minutes: int = 0
    travel_km: float = 0.0
    idle_minutes: int = 0
    span_start: int | None = None
    span_end: int | None = None
    operating_cost: float = 0.0

    @property
    def utilization(self) -> float:
        """Worked fraction of the span between first start and last end."""
        if self.span_start is None or self.span_end is None:
            return 0.0
        span = max(1, self.span_end - self.span_start)
        return min(1.0, self.worked_minutes / span)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "assigned_tasks": self.assigned_tasks,
            "worked_minutes": self.worked_minutes,
            "travel_minutes": self.travel_minutes,
            "travel_km": round(self.travel_km, 3),
            "idle_minutes": self.idle_minutes,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "operating_cost": round(self.operating_cost, 3),
            "utilization": round(self.utilization, 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceUtilization:
        return cls(
            resource_id=str(data["resource_id"]),
            assigned_tasks=int(data.get("assigned_tasks", 0)),
            worked_minutes=int(data.get("worked_minutes", 0)),
            travel_minutes=int(data.get("travel_minutes", 0)),
            travel_km=float(data.get("travel_km", 0.0)),
            idle_minutes=int(data.get("idle_minutes", 0)),
            span_start=data.get("span_start"),
            span_end=data.get("span_end"),
            operating_cost=float(data.get("operating_cost", 0.0)),
        )


@dataclass(slots=True)
class Plan:
    """A complete operational plan for a scenario at a point in time."""

    id: str
    scenario_id: str
    assignments: tuple[Assignment, ...]
    solver_runs: tuple[SolverRun, ...] = ()
    objective: ObjectiveBreakdown = field(default_factory=ObjectiveBreakdown)
    violations: tuple[Violation, ...] = ()
    utilization: tuple[ResourceUtilization, ...] = ()
    total_travel_km: float = 0.0
    total_travel_minutes: int = 0
    total_operating_cost: float = 0.0
    total_dispatch_cost: float = 0.0
    total_worked_minutes: int = 0
    service_level: float = 0.0
    tasks_assigned: int = 0
    tasks_total: int = 0
    late_tasks: int = 0
    status: str = SolverStatus.FEASIBLE
    strategy: str = ""
    created_at: str = ""
    parent_plan_id: str | None = None
    disruption_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- construction helpers ---------------------------------------------
    def __post_init__(self) -> None:
        validate_id(self.id, field="plan.id")
        self.assignments = tuple(self.assignments)
        self.solver_runs = tuple(self.solver_runs)
        self.violations = tuple(self.violations)
        self.utilization = tuple(self.utilization)
        seen: set[str] = set()
        for assignment in self.assignments:
            if assignment.task_id in seen:
                raise ValidationError(
                    f"plan {self.id}: task {assignment.task_id} is assigned more than once"
                )
            seen.add(assignment.task_id)

    @property
    def score(self) -> float:
        """The scalar ORION optimises. Higher is better."""
        return self.objective.total

    @property
    def is_usable(self) -> bool:
        return self.status in USABLE_STATUSES

    @property
    def hard_violations(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity >= 2)

    @property
    def soft_violations(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity < 2)

    # -- lookups -----------------------------------------------------------
    def by_task(self) -> dict[str, Assignment]:
        return {a.task_id: a for a in self.assignments}

    def by_resource(self) -> dict[str, list[Assignment]]:
        grouped: dict[str, list[Assignment]] = {}
        for assignment in sorted(self.assignments, key=lambda a: (a.resource_id, a.start)):
            grouped.setdefault(assignment.resource_id, []).append(assignment)
        return grouped

    def assignment_for(self, task_id: str) -> Assignment | None:
        for assignment in self.assignments:
            if assignment.task_id == task_id:
                return assignment
        return None

    def tasks_of(self, resource_id: str) -> list[Assignment]:
        return sorted(
            (a for a in self.assignments if a.resource_id == resource_id),
            key=lambda a: a.sequence_index,
        )

    def utilization_of(self, resource_id: str) -> ResourceUtilization:
        for item in self.utilization:
            if item.resource_id == resource_id:
                return item
        return ResourceUtilization(resource_id=resource_id)

    @property
    def primary_solver_run(self) -> SolverRun | None:
        return self.solver_runs[0] if self.solver_runs else None

    @property
    def window(self) -> tuple[int, int]:
        if not self.assignments:
            return (0, 0)
        return (min(a.start for a in self.assignments), max(a.end for a in self.assignments))

    def unassigned_tasks(self, all_task_ids: Iterable[str]) -> list[str]:
        assigned = {a.task_id for a in self.assignments}
        return [tid for tid in all_task_ids if tid not in assigned]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scenario_id": self.scenario_id,
            "status": self.status,
            "strategy": self.strategy,
            "created_at": self.created_at,
            "parent_plan_id": self.parent_plan_id,
            "disruption_id": self.disruption_id,
            "assignments": [a.to_dict() for a in self.assignments],
            "solver_runs": [s.to_dict() for s in self.solver_runs],
            "objective": self.objective.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "utilization": [u.to_dict() for u in self.utilization],
            "metrics": {
                "score": round(self.score, 4),
                "service_level": round(self.service_level, 6),
                "tasks_assigned": self.tasks_assigned,
                "tasks_total": self.tasks_total,
                "late_tasks": self.late_tasks,
                "total_travel_km": round(self.total_travel_km, 3),
                "total_travel_minutes": self.total_travel_minutes,
                "total_operating_cost": round(self.total_operating_cost, 3),
                "total_dispatch_cost": round(self.total_dispatch_cost, 3),
                "total_worked_minutes": self.total_worked_minutes,
                "violations_total": len(self.violations),
                "violations_hard": len(self.hard_violations),
            },
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plan:
        metrics = data.get("metrics") or {}
        return cls(
            id=str(data["id"]),
            scenario_id=str(data["scenario_id"]),
            assignments=tuple(Assignment.from_dict(a) for a in data.get("assignments", ())),
            solver_runs=tuple(SolverRun.from_dict(s) for s in data.get("solver_runs", ())),
            objective=ObjectiveBreakdown.from_dict(data.get("objective") or {}),
            violations=tuple(Violation.from_dict(v) for v in data.get("violations", ())),
            utilization=tuple(
                ResourceUtilization.from_dict(u) for u in data.get("utilization", ())
            ),
            total_travel_km=float(metrics.get("total_travel_km", 0.0)),
            total_travel_minutes=int(metrics.get("total_travel_minutes", 0)),
            total_operating_cost=float(metrics.get("total_operating_cost", 0.0)),
            total_dispatch_cost=float(metrics.get("total_dispatch_cost", 0.0)),
            total_worked_minutes=int(metrics.get("total_worked_minutes", 0)),
            service_level=float(metrics.get("service_level", 0.0)),
            tasks_assigned=int(metrics.get("tasks_assigned", len(data.get("assignments", ())))),
            tasks_total=int(metrics.get("tasks_total", 0)),
            late_tasks=int(metrics.get("late_tasks", 0)),
            status=str(data.get("status", SolverStatus.FEASIBLE)),
            strategy=str(data.get("strategy", "")),
            created_at=str(data.get("created_at", "")),
            parent_plan_id=data.get("parent_plan_id"),
            disruption_id=data.get("disruption_id"),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> Plan:
        return cls.from_dict(json.loads(text))

    def summary_line(self) -> str:
        run = self.primary_solver_run
        solver = run.solver if run else "none"
        return (
            f"{self.id} [{self.status}] score={self.score:.2f} "
            f"assigned={self.tasks_assigned}/{self.tasks_total} late={self.late_tasks} "
            f"travel={self.total_travel_km:.1f}km solver={solver}"
        )


# ==========================================================================
# Structural plan validation
# ==========================================================================
def validate_plan(
    plan: Plan,
    scenario: Any,
    *,
    strict: bool = True,
) -> list[Violation]:
    """Recompute a plan's violations from the scenario.

    This is the function of record. Every number the UI shows about a plan -
    completion rate, lateness, utilisation, cost, violations - is produced here
    or by :mod:`orion.evaluation.metrics`, and both are pure functions of
    ``(plan, scenario)``. A test asserts that a plan whose stored metrics
    disagree with this function fails, which prevents a stale cached score from
    surviving a code change.

    When *strict* is True, hard (severity 2) violations are also returned; the
    caller decides whether to reject the plan. Nothing is silently repaired.
    """
    from orion.domain.entities import Scenario

    assert isinstance(scenario, Scenario)
    tasks = {t.id: t for t in scenario.tasks}
    resources = {r.id: r for r in scenario.resources}
    weights = scenario.objective_weights
    constraints = scenario.constraints
    violations: list[Violation] = []

    assignments = list(plan.assignments)

    # ---- per-assignment checks ----
    for a in assignments:
        task = tasks.get(a.task_id)
        resource = resources.get(a.resource_id)
        if task is None:
            violations.append(
                Violation(
                    ViolationKind.RESOURCE_CONFLICT,
                    a.task_id,
                    a.resource_id,
                    2,
                    f"assignment references unknown task {a.task_id}",
                )
            )
            continue
        if resource is None:
            violations.append(
                Violation(
                    ViolationKind.RESOURCE_CONFLICT,
                    a.task_id,
                    a.resource_id,
                    2,
                    f"assignment references unknown resource {a.resource_id}",
                )
            )
            continue
        if constraints.enforce_capability and not task.requires(resource.capabilities):
            missing = sorted(task.required_capabilities - resource.capabilities)
            violations.append(
                Violation(
                    ViolationKind.CAPABILITY_MISMATCH,
                    task.id,
                    resource.id,
                    2,
                    f"resource {resource.id} lacks {missing}",
                )
            )
        if constraints.enforce_capacity:
            for demand in task.required_capacity:
                available = float(resource.capacity.get(demand.dimension, 0.0))
                if demand.amount > available:
                    violations.append(
                        Violation(
                            ViolationKind.CAPACITY_EXCEEDED,
                            task.id,
                            resource.id,
                            2,
                            f"{resource.id} {demand.dimension}={available:g} < "
                            f"{demand.describe()}",
                        )
                    )
        if a.end - a.start < task.duration:
            violations.append(
                Violation(
                    ViolationKind.RESOURCE_CONFLICT,
                    task.id,
                    resource.id,
                    2,
                    f"assignment window {hhmm(a.start)}-{hhmm(a.end)} is shorter than "
                    f"the required duration {task.duration}",
                )
            )
        if constraints.enforce_release_times and a.start < task.release_time:
            violations.append(
                Violation(
                    ViolationKind.RELEASE_TIME_VIOLATED,
                    task.id,
                    resource.id,
                    2,
                    f"starts {hhmm(a.start)} before release {hhmm(task.release_time)}",
                )
            )
        if constraints.enforce_deadlines:
            late = task.lateness_for(a.end)
            if late > 0:
                severity = 1
                if task.max_lateness is not None and late > task.max_lateness:
                    severity = 2
                    kind = ViolationKind.MAX_LATENESS_EXCEEDED
                else:
                    kind = ViolationKind.DEADLINE_MISSED
                violations.append(
                    Violation(
                        kind,
                        task.id,
                        resource.id,
                        severity,
                        f"finishes {hhmm(a.end)}, {late} min after deadline {hhmm(task.deadline)}",
                        amount=float(late),
                    )
                )
        if constraints.enforce_shift:
            if a.start < resource.shift_start or a.end > resource.shift_end:
                violations.append(
                    Violation(
                        ViolationKind.SHIFT_VIOLATION,
                        task.id,
                        resource.id,
                        2,
                        f"{hhmm(a.start)}-{hhmm(a.end)} outside shift "
                        f"{hhmm(resource.shift_start)}-{hhmm(resource.shift_end)}",
                    )
                )
        if constraints.enforce_maintenance and resource.is_globally_unavailable:
            violations.append(
                Violation(
                    ViolationKind.MAINTENANCE_CONFLICT,
                    task.id,
                    resource.id,
                    2,
                    f"resource {resource.id} status is {resource.status}",
                )
            )
        elif constraints.enforce_maintenance:
            for block in resource.unavailable:
                if a.window.overlaps(block):
                    violations.append(
                        Violation(
                            ViolationKind.MAINTENANCE_CONFLICT,
                            task.id,
                            resource.id,
                            2,
                            f"overlaps maintenance block {block}",
                        )
                    )

    # ---- per-resource overlap / travel / max-work ----
    for resource_id, group in plan.by_resource().items():
        resource = resources.get(resource_id)
        ordered = sorted(group, key=lambda a: a.start)
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                violations.append(
                    Violation(
                        ViolationKind.RESOURCE_CONFLICT,
                        nxt.task_id,
                        resource_id,
                        2,
                        f"overlaps {prev.task_id} on {resource_id} "
                        f"({hhmm(prev.start)}-{hhmm(prev.end)} vs {hhmm(nxt.start)})",
                    )
                )
            if constraints.enforce_travel and scenario.travel.has(prev.location) and scenario.travel.has(nxt.location):
                needed = scenario.travel.travel_from_speed(
                    prev.location, nxt.location, resource.speed_factor if resource else 1.0
                )
                slack = nxt.start - prev.end
                if slack < needed:
                    violations.append(
                        Violation(
                            ViolationKind.TRAVEL_INFEASIBLE,
                            nxt.task_id,
                            resource_id,
                            2,
                            f"needs {needed} min to travel {prev.location}->{nxt.location}, "
                            f"only {slack} min allowed",
                        )
                    )
        if constraints.enforce_max_work and resource is not None:
            worked = sum(a.end - a.start for a in ordered)
            if worked > resource.max_work_minutes:
                violations.append(
                    Violation(
                        ViolationKind.OVERLOAD,
                        None,
                        resource_id,
                        2,
                        f"works {worked} min, limit {resource.max_work_minutes}",
                        amount=float(worked - resource.max_work_minutes),
                    )
                )

    # ---- dependencies ----
    if constraints.enforce_dependencies:
        by_task = plan.by_task()
        for a in assignments:
            task = tasks.get(a.task_id)
            if task is None:
                continue
            for dep in task.dependencies:
                parent = by_task.get(dep)
                if parent is None:
                    continue  # unassigned dependency: covered by UNASSIGNED_TASK
                if a.start < parent.end:
                    violations.append(
                        Violation(
                            ViolationKind.DEPENDENCY_VIOLATED,
                            a.task_id,
                            a.resource_id,
                            2,
                            f"starts {hhmm(a.start)} before prerequisite {dep} ends "
                            f"{hhmm(parent.end)}",
                        )
                    )

    # ---- unassigned tasks ----
    assigned_ids = {a.task_id for a in assignments}
    unassigned = [t.id for t in scenario.tasks if t.id not in assigned_ids]
    if unassigned:
        if not constraints.allow_unassigned:
            for task_id in unassigned:
                violations.append(
                    Violation(
                        ViolationKind.UNASSIGNED_TASK,
                        task_id,
                        None,
                        2,
                        "task is unassigned and the scenario forbids unassigned tasks",
                    )
                )
        elif len(unassigned) / max(1, len(scenario.tasks)) > constraints.max_unassigned_fraction:
            violations.append(
                Violation(
                    ViolationKind.UNASSIGNED_TASK,
                    None,
                    None,
                    1,
                    f"{len(unassigned)}/{len(scenario.tasks)} tasks unassigned exceeds the "
                    f"allowed fraction {constraints.max_unassigned_fraction:.2f}",
                    amount=float(len(unassigned)),
                )
            )

    if strict and any(v.severity >= 2 for v in violations):
        # still return them; the caller inspects. The flag exists so callers can
        # assert "no hard violations" in one line.
        pass
    return violations


def has_hard_violation(violations: Sequence[Violation]) -> bool:
    return any(v.severity >= 2 for v in violations)


__all__ = [
    "SolverStatus",
    "USABLE_STATUSES",
    "SolverName",
    "ALL_SOLVERS",
    "SOLVER_DESCRIPTIONS",
    "Violation",
    "ViolationKind",
    "Assignment",
    "ObjectiveBreakdown",
    "SolverRun",
    "ResourceUtilization",
    "Plan",
    "validate_plan",
    "has_hard_violation",
]
