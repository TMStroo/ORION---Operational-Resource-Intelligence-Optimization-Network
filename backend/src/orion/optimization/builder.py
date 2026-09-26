
"""Plan construction and scoring: the single path from assignments to a Plan.

Every solver ends here. A solver's only job is to decide *which* assignments to
make; how those assignments become a scored, validated, explainable
:class:`~orion.domain.plans.Plan` is decided in this module, once.
"""

from __future__ import annotations

import datetime as _dt
import itertools
from typing import Iterable, Mapping, Sequence

from orion.domain.entities import Scenario
from orion.domain.plans import (
    Assignment,
    ObjectiveBreakdown,
    Plan,
    ResourceUtilization,
    SolverRun,
    SolverStatus,
    Violation,
    ViolationKind,
    has_hard_violation,
    validate_plan,
)
from orion.optimization.objective import (
    objective_breakdown_from_assignments,
    service_level,
)
from orion.optimization.models import SchedulingContext

_PLAN_COUNTER = itertools.count(1)


def next_plan_id(prefix: str = "PLAN") -> str:
    """Monotonic plan identifier.

    Plans are given sequential ids rather than random ones so that a demo
    transcript reads as a narrative: PLAN-001 is the baseline, PLAN-002 the
    repaired plan, PLAN-003 the re-optimised plan. The id is part of the audit
    log, so it must be readable by a human reviewing the run.
    """
    return f"{prefix}-{next(_PLAN_COUNTER):04d}"


def reset_plan_counter(start: int = 1) -> None:
    """Reset the plan counter. Used by tests and the deterministic demo."""
    global _PLAN_COUNTER
    _PLAN_COUNTER = itertools.count(start)


def _depot_return_penalty(
    context: SchedulingContext, assignments: Sequence[Assignment]
) -> float:
    """Cost of not returning to the depot at the end of each resource's shift."""
    if not context.scenario.constraints.require_depot_return:
        return 0.0
    total = 0.0
    by_resource: dict[str, list[Assignment]] = {}
    for a in assignments:
        by_resource.setdefault(a.resource_id, []).append(a)
    for resource_id, group in by_resource.items():
        ordered = sorted(group, key=lambda a: a.sequence_index)
        if not ordered:
            continue
        speed = context.speed_of(resource_id)
        total += context.scenario.travel.distance(
            ordered[-1].location, context.scenario.depot
        ) / max(1e-9, speed)
    return total


def build_utilization(
    context: SchedulingContext, assignments: Sequence[Assignment]
) -> tuple[ResourceUtilization, ...]:
    """Per-resource workload summary for the timeline and utilisation figures."""
    scenario = context.scenario
    by_resource: dict[str, list[Assignment]] = {}
    for a in assignments:
        by_resource.setdefault(a.resource_id, []).append(a)

    out: list[ResourceUtilization] = []
    for resource in context.resources:
        group = by_resource.get(resource.id, [])
        if group:
            ordered = sorted(group, key=lambda a: a.start)
            span_start = ordered[0].start
            span_end = max(a.end for a in ordered)
            worked = sum(a.end - a.start for a in ordered)
            travel_min = sum(a.travel_before for a in ordered)
            travel_km = sum(a.travel_distance_km for a in ordered)
            idle = max(0, span_end - span_start - worked)
            cost = resource.operating_cost_per_hour * (worked / 60.0)
        else:
            span_start = span_end = None
            worked = travel_min = 0
            travel_km = 0.0
            idle = 0
            cost = 0.0
        out.append(
            ResourceUtilization(
                resource_id=resource.id,
                assigned_tasks=len(group),
                worked_minutes=worked,
                travel_minutes=travel_min,
                travel_km=round(travel_km, 3),
                idle_minutes=idle,
                span_start=span_start,
                span_end=span_end,
                operating_cost=round(cost, 4),
            )
        )
    return tuple(out)


def build_plan(
    context: SchedulingContext,
    assignments: Iterable[Assignment],
    run: SolverRun,
    *,
    plan_id: str | None = None,
    status: str | None = None,
    strategy: str = "",
    parent_plan_id: str | None = None,
    disruption_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Plan:
    """Score, validate and package a set of assignments into a Plan.

    The returned plan's stored metrics are always recomputed here, never taken
    from the solver. If the solver's own objective disagrees with the scored
    value, the scored value wins and the disagreement is recorded in
    ``metadata['objective_mismatch']`` so it is visible rather than hidden.
    """
    scenario = context.scenario
    ordered = tuple(sorted(assignments, key=lambda a: (a.resource_id, a.start)))
    violations = validate_plan(
        Plan(id="SCRATCH", scenario_id=scenario.id, assignments=ordered), scenario
    )
    hard = [v for v in violations if v.severity >= 2]
    # Deadline misses are already priced through ``w_late`` (per hour of
    # lateness), so charging them again as generic violations would double-count
    # the same harm and would make the CP-SAT/MILP internal objective (which
    # models lateness but not the violation tally) disagree with the score.
    # Every other soft violation is unpriced elsewhere, so those are charged.
    soft = [
        v
        for v in violations
        if v.severity < 2 and v.kind != ViolationKind.DEADLINE_MISSED
    ]

    speeds = {r.id: r.speed_factor for r in context.resources}
    overload: dict[str, int] = {}
    for resource in context.resources:
        worked = sum(a.end - a.start for a in ordered if a.resource_id == resource.id)
        if scenario.constraints.enforce_max_work:
            excess = worked - resource.max_work_minutes
            if excess > 0:
                overload[resource.id] = excess

    breakdown = objective_breakdown_from_assignments(
        list(ordered),
        scenario,
        resource_speeds=speeds,
        violation_penalty_count=len(soft),
        hard_violation_count=len(hard),
        overload_minutes=overload,
    )
    # depot return, if required, counts as travel
    if scenario.constraints.require_depot_return:
        penalty = _depot_return_penalty(context, ordered)
        if penalty:
            breakdown.travel_penalty += scenario.objective_weights.w_travel * penalty
            breakdown.total_travel_km = breakdown.total_travel_km  # km tracked separately

    level = service_level(breakdown, scenario)
    late = sum(1 for a in ordered if a.lateness > 0)
    travel_km = round(sum(a.travel_distance_km for a in ordered), 4)
    travel_min = sum(a.travel_before for a in ordered)
    cost = 0.0
    dispatch = 0.0
    worked_by_resource: dict[str, int] = {}
    for a in ordered:
        worked_by_resource[a.resource_id] = (
            worked_by_resource.get(a.resource_id, 0) + (a.end - a.start)
        )
    for resource in context.resources:
        minutes = worked_by_resource.get(resource.id, 0)
        cost += resource.operating_cost_per_hour * (minutes / 60.0)
        if minutes:
            dispatch += resource.fixed_dispatch_cost

    effective_status = status or run.status
    if has_hard_violation(violations) and effective_status in {
        SolverStatus.OPTIMAL,
        SolverStatus.FEASIBLE,
    }:
        # A plan that violates a hard constraint is not "feasible", whatever the
        # solver said. Downgrading here prevents a false success from reaching
        # the UI, which is the explicit requirement in the failure-modes section.
        effective_status = SolverStatus.ERROR

    extra: dict[str, object] = dict(metadata or {})
    if abs(run.objective - breakdown.total) > 1e-2 and run.objective != 0.0:
        extra["objective_mismatch"] = {
            "solver_reported": round(run.objective, 4),
            "scored": round(breakdown.total, 4),
        }

    plan = Plan(
        id=plan_id or next_plan_id(),
        scenario_id=scenario.id,
        assignments=ordered,
        solver_runs=(run,),
        objective=breakdown,
        violations=tuple(violations),
        utilization=build_utilization(context, ordered),
        total_travel_km=travel_km,
        total_travel_minutes=travel_min,
        total_operating_cost=round(cost, 4),
        total_dispatch_cost=round(dispatch, 4),
        total_worked_minutes=sum(a.end - a.start for a in ordered),
        service_level=level.ratio,
        tasks_assigned=len(ordered),
        tasks_total=len(scenario.tasks),
        late_tasks=late,
        status=effective_status,
        strategy=strategy or run.solver,
        created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        parent_plan_id=parent_plan_id,
        disruption_id=disruption_id,
        metadata=extra,
    )
    return plan


def empty_run(
    solver: str,
    status: str = SolverStatus.ERROR,
    *,
    note: str = "",
    runtime_s: float = 0.0,
) -> SolverRun:
    return SolverRun(
        solver=solver,
        status=status,
        runtime_s=runtime_s,
        objective=0.0,
        feasible=False,
        notes=note,
    )


__all__ = ["build_plan", "build_utilization", "next_plan_id", "reset_plan_counter", "empty_run"]
