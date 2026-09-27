"""Convert domain objects into API schemas.

Kept separate from the routers so the mapping is testable on its own and so the
wiring code stays readable. Every conversion is total: a missing optional field
produces an empty list, never an exception. A plan with no explanations should
render, not 500.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from orion.api.schemas import (
    AssignmentView,
    ChurnView,
    ComparisonView,
    DisruptionView,
    MetricDeltaView,
    ObjectiveView,
    PlanDetail,
    PlanSummary,
    RepairView,
    ResourceView,
    ScenarioDetail,
    ScenarioSummary,
    SimEventView,
    SolverRunView,
    TaskView,
    UtilizationView,
    ValidationIssue,
)


def _name(value: Any) -> str:
    """Enum member or plain string -> a stable readable string.

    ``Priority`` is an IntEnum, so ``str(p.value)`` is "3" and a client reading
    `priority` would have to know the numbering. The member name is what a
    human wants and what stays stable if the ordering is ever renumbered.

    Plain strings pass through unchanged, so this is safe on values that are
    already strings.
    """
    if isinstance(value, str):
        return value
    return str(getattr(value, "name", None) or getattr(value, "value", value))


def scenario_summary(row: Any) -> ScenarioSummary:
    return ScenarioSummary(
        id=row.id,
        name=row.name,
        description=row.description or "",
        task_count=row.task_count,
        resource_count=row.resource_count,
        horizon_end=row.horizon_end,
        source=row.source or "generated",
        tags=[t for t in (row.tags or "").split(",") if t],
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


def scenario_detail(row: Any, scenario: Any) -> ScenarioDetail:
    base = scenario_summary(row)
    # The domain names these w_*, not the ObjectiveBreakdown names they weight.
    # Reading the wrong ones silently returns an empty dict rather than raising,
    # which is how the weights went missing from the response body.
    weights = {
        name: float(getattr(scenario.objective_weights, name))
        for name in (
            "w_completion",
            "w_priority",
            "w_late",
            "w_travel",
            "w_cost",
            "w_dispatch",
            "w_overload",
            "w_move_idle",
            "w_violation",
        )
        if hasattr(scenario.objective_weights, name)
    }
    return ScenarioDetail(
        **base.model_dump(),
        objective_weights=weights,
        tasks=[
            TaskView(
                id=t.id,
                location=t.location,
                release_time=t.release_time,
                deadline=t.deadline,
                duration=t.duration,
                priority=_name(t.priority),
                required_capabilities=sorted(t.required_capabilities),
                dependencies=list(t.dependencies),
                status=_name(t.status),
            )
            for t in scenario.tasks
        ],
        resources=[
            ResourceView(
                id=r.id,
                kind=_name(r.kind),
                capabilities=sorted(r.capabilities),
                home_location=r.home_location,
                shift_start=r.shift_start,
                shift_end=r.shift_end,
                status=_name(r.status),
                max_work_minutes=r.max_work_minutes,
                capacity=dict(r.capacity),
            )
            for r in scenario.resources
        ],
        horizon={"start": scenario.horizon.start, "end": scenario.horizon.end},
        depot=scenario.depot,
    )


def plan_summary(row: Any) -> PlanSummary:
    return PlanSummary(
        id=row.id,
        scenario_id=row.scenario_id,
        label=row.label or "",
        status=row.status or "",
        solver=row.solver or "",
        objective=float(row.objective or 0.0),
        completion=float(row.completion or 0.0),
        late_tasks=int(row.late_tasks or 0),
        tasks_assigned=int(row.tasks_assigned or 0),
        tasks_total=int(row.tasks_total or 0),
        runtime_s=float(row.runtime_s or 0.0),
        infeasible=bool(row.infeasible),
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


def plan_detail(row: Any, plan: Any) -> PlanDetail:
    """Full plan view.

    `row` supplies the queryable columns, `plan` the fidelity payload. Using
    both means the table and the object cannot drift: the columns are written
    from the same object in the same transaction.
    """
    return PlanDetail(
        **plan_summary(row).model_dump(),
        assignments=[assignment_view(a) for a in plan.assignments],
        violations=[issue(v) for v in plan.violations],
        utilization=[utilization_view(u) for u in plan.utilization],
        solver_runs=[solver_run_view(r) for r in plan.solver_runs],
        objective_breakdown=objective_view(plan.objective),
        total_travel_km=float(plan.total_travel_km or 0.0),
        total_travel_minutes=int(plan.total_travel_minutes or 0),
        total_operating_cost=float(plan.total_operating_cost or 0.0),
        service_level=float(plan.service_level or 0.0),
        strategy=str(plan.strategy or ""),
        parent_plan_id=plan.parent_plan_id,
        disruption_id=plan.disruption_id,
        metadata=dict(plan.metadata or {}),
    )


def assignment_view(a: Any) -> AssignmentView:
    return AssignmentView(
        task_id=a.task_id,
        resource_id=a.resource_id,
        start=a.start,
        end=a.end,
        location=a.location,
        sequence_index=int(getattr(a, "sequence_index", 0) or 0),
        travel_before=int(getattr(a, "travel_before", 0) or 0),
        travel_distance_km=float(getattr(a, "travel_distance_km", 0.0) or 0.0),
        priority=_name(a.priority),
        lateness=int(getattr(a, "lateness", 0) or 0),
    )


def solver_run_view(r: Any) -> SolverRunView:
    raw = getattr(r, "raw", {}) or {}
    return SolverRunView(
        solver=r.solver,
        status=r.status,
        objective=float(r.objective or 0.0),
        runtime_s=float(r.runtime_s or 0.0),
        optimality_gap=(
            float(r.optimality_gap) if getattr(r, "optimality_gap", None) is not None else None
        ),
        feasible=bool(getattr(r, "feasible", False)),
        notes=str(getattr(r, "notes", "") or ""),
    )


def objective_view(o: Any) -> ObjectiveView | None:
    if o is None:
        return None
    return ObjectiveView(
        total=float(getattr(o, "total", 0.0) or 0.0),
        completion=float(getattr(o, "completion", 0.0) or 0.0),
        priority=float(getattr(o, "priority", 0.0) or 0.0),
        lateness_penalty=float(getattr(o, "lateness_penalty", 0.0) or 0.0),
        travel_penalty=float(getattr(o, "travel_penalty", 0.0) or 0.0),
        cost_penalty=float(getattr(o, "cost_penalty", 0.0) or 0.0),
        dispatch_penalty=float(getattr(o, "dispatch_penalty", 0.0) or 0.0),
        overload_penalty=float(getattr(o, "overload_penalty", 0.0) or 0.0),
        idle_move_penalty=float(getattr(o, "idle_move_penalty", 0.0) or 0.0),
        violation_penalty=float(getattr(o, "violation_penalty", 0.0) or 0.0),
    )


def utilization_view(u: Any) -> UtilizationView:
    util = getattr(u, "utilization", None)
    if util is None:
        util = getattr(u, "utilisation", 0.0)
    return UtilizationView(
        resource_id=u.resource_id,
        kind=_name(getattr(u, "kind", "")),
        assigned_tasks=int(getattr(u, "assigned_tasks", 0) or 0),
        worked_minutes=int(getattr(u, "worked_minutes", 0) or 0),
        utilization=float(util or 0.0),
        travel_km=float(getattr(u, "travel_km", 0.0) or 0.0),
    )


def issue(v: Any) -> ValidationIssue:
    return ValidationIssue(
        kind=_name(getattr(v, "kind", "")),
        detail=str(getattr(v, "detail", "") or ""),
        task_id=getattr(v, "task_id", None),
        resource_id=getattr(v, "resource_id", None),
        severity=int(getattr(v, "severity", 0) or 0),
        amount=float(getattr(v, "amount", 0.0) or 0.0),
    )


def disruption_view(d: Any, impact: Any | None = None) -> DisruptionView:
    return DisruptionView(
        id=d.id,
        type=_name(d.type),
        severity=_name(getattr(d, "severity", "")),
        target_id=getattr(d, "target_id", None),
        timestamp=int(d.timestamp or 0),
        magnitude=float(d.magnitude or 0.0),
        duration=getattr(d, "duration", None),
        description=str(d.description or ""),
        affected_tasks=list(getattr(impact, "affected_tasks", ()) or ()),
        affected_resources=list(getattr(impact, "affected_resources", ()) or ()),
        at_risk_deadlines=list(getattr(impact, "at_risk_deadlines", ()) or ()),
    )


def repair_view(repair: Any) -> RepairView:
    return RepairView(
        strategy=str(getattr(repair, "strategy", "") or ""),
        seconds=float(getattr(repair, "seconds", 0.0) or 0.0),
        candidates_considered=int(getattr(repair, "candidates_considered", 0) or 0),
        recovered_tasks=list(getattr(repair, "recovered_tasks", ()) or ()),
        unrecovered_tasks=list(getattr(repair, "unrecovered_tasks", ()) or ()),
        produced_plan=getattr(repair, "plan", None) is not None,
        reason=str(getattr(repair, "reason", "") or ""),
    )


def churn_view(churn: Any) -> ChurnView:
    total_before = int(getattr(churn, "total_before", 0) or 0)
    changed = int(getattr(churn, "changed_assignments", 0) or 0)
    # A ratio against zero assignments is reported as 0, not a division error.
    ratio = (changed / total_before) if total_before else 0.0
    return ChurnView(
        changed_assignments=changed,
        rescheduled_tasks=int(getattr(churn, "rescheduled_tasks", 0) or 0),
        total_before=total_before,
        total_after=int(getattr(churn, "total_after", 0) or 0),
        tolerance_minutes=int(getattr(churn, "tolerance_minutes", 0) or 0),
        travel_delta_km=float(getattr(churn, "travel_delta_km", 0.0) or 0.0),
        cost_delta=float(getattr(churn, "cost_delta", 0.0) or 0.0),
        added=list(getattr(churn, "added", ()) or ()),
        removed=list(getattr(churn, "removed", ()) or ()),
        delayed=list(getattr(churn, "delayed", ()) or ()),
        churn_ratio=ratio,
    )


def comparison_view(
    comparison: Any, *, kind: str, baseline_plan_id: str, candidate_plan_id: str
) -> ComparisonView:
    return ComparisonView(
        kind=kind,
        baseline_plan_id=baseline_plan_id,
        candidate_plan_id=candidate_plan_id,
        churn=churn_view(getattr(comparison, "churn", None)),
        deltas=[
            MetricDeltaView(
                name=d.name,
                label=d.label,
                before=float(d.before),
                after=float(d.after),
                change=float(d.after) - float(d.before),
                unit=d.unit,
                higher_is_better=bool(d.higher_is_better),
            )
            for d in getattr(comparison, "deltas", ()) or ()
        ],
        newly_assigned=list(getattr(comparison, "newly_assigned", ()) or ()),
        removed_assignments=list(getattr(comparison, "removed_assignments", ()) or ()),
        delayed=list(getattr(comparison, "delayed", ()) or ()),
        newly_violated=list(getattr(comparison, "newly_violated", ()) or ()),
        newly_resolved=list(getattr(comparison, "newly_resolved", ()) or ()),
    )


def _event_type_name(value: Any) -> str:
    """IntEnum -> its name, e.g. TASK_START.

    ``.value`` on an IntEnum is the integer, so a client reading `type` would
    have to hard-code the numbering. The name is stable and readable.
    """
    return str(getattr(value, "name", None) or _name(value))


def sim_event_view(event: Any, ordinal: int) -> SimEventView:
    return SimEventView(
        ordinal=ordinal,
        time=int(getattr(event, "time", 0) or 0),
        type=_event_type_name(getattr(event, "type", getattr(event, "kind", ""))),
        subject_id=getattr(event, "subject_id", None),
        resource_id=getattr(event, "resource_id", None),
        detail=str(getattr(event, "detail", "") or ""),
        payload=dict(getattr(event, "payload", {}) or {}),
    )
