
"""Plan comparison and recovery metrics.

The comparison view is the answer to "what changed and why". It is computed
purely from two plans (and the scenario), so the same function serves the API,
the CLI, the report and the figures.

Definitions used throughout, stated precisely because a recovery percentage
with an unstated denominator is meaningless:

**service level**
    ``(completed value + priority value) / (total value of all tasks)``, in
    ``[0, 1]``. A plan that serves nothing is 0; a plan that serves everything on
    time is 1.

**degradation**
    ``service(baseline) - service(degraded)``, as a fraction. This is how much
    service the disruption destroyed *in the plan*, measured on the same
    denominator for both.

**recovery**
    ``(service(repaired) - service(degraded)) / (service(baseline) - service(degraded))``
    when the disruption caused any loss; ``1.0`` when it caused none. Clamped to
    ``[0, 1]`` because a repair that ends up *better* than the baseline has not
    "recovered more than all of the loss" - it has found a better plan, and
    reporting that as 140% recovery would be false precision.

**time to recovery**
    wall-clock seconds from the disruption to a usable plan, measured inside the
    replanner rather than estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from orion.domain.entities import Scenario
from orion.domain.plans import Plan, SolverStatus
from orion.planning.replanner import ChurnReport, compute_churn


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """A before/after pair for one metric, with the change and its direction."""

    name: str
    label: str
    before: float
    after: float
    unit: str = ""
    higher_is_better: bool = True

    @property
    def change(self) -> float:
        return self.after - self.before

    @property
    def change_pct(self) -> float:
        if abs(self.before) < 1e-12:
            return 0.0
        return 100.0 * self.change / abs(self.before)

    @property
    def improved(self) -> bool:
        return self.change > 0 if self.higher_is_better else self.change < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "before": round(self.before, 4),
            "after": round(self.after, 4),
            "change": round(self.change, 4),
            "change_pct": round(self.change_pct, 3),
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "improved": self.improved,
        }


@dataclass(slots=True)
class PlanComparison:
    """Side-by-side comparison of two plans for the same scenario."""

    baseline: Plan
    candidate: Plan
    scenario: Scenario
    churn: ChurnReport
    deltas: tuple[MetricDelta, ...]
    newly_assigned: tuple[str, ...] = ()
    removed_assignments: tuple[str, ...] = ()
    delayed: tuple[str, ...] = ()
    newly_violated: tuple[str, ...] = ()
    newly_resolved: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": {
                "plan_id": self.baseline.id,
                "status": self.baseline.status,
                "score": round(self.baseline.score, 4),
                "service_level": round(self.baseline.service_level, 6),
                "tasks_assigned": self.baseline.tasks_assigned,
                "tasks_total": self.baseline.tasks_total,
                "late_tasks": self.baseline.late_tasks,
                "travel_km": round(self.baseline.total_travel_km, 3),
                "operating_cost": round(self.baseline.total_operating_cost, 3),
                "violations": len(self.baseline.violations),
            },
            "candidate": {
                "plan_id": self.candidate.id,
                "status": self.candidate.status,
                "score": round(self.candidate.score, 4),
                "service_level": round(self.candidate.service_level, 6),
                "tasks_assigned": self.candidate.tasks_assigned,
                "tasks_total": self.candidate.tasks_total,
                "late_tasks": self.candidate.late_tasks,
                "travel_km": round(self.candidate.total_travel_km, 3),
                "operating_cost": round(self.candidate.total_operating_cost, 3),
                "violations": len(self.candidate.violations),
            },
            "deltas": [d.to_dict() for d in self.deltas],
            "churn": self.churn.to_dict(),
            "diff": {
                "newly_assigned": list(self.newly_assigned),
                "removed_assignments": list(self.removed_assignments),
                "delayed": list(self.delayed),
                "newly_violated": list(self.newly_violated),
                "newly_resolved": list(self.newly_resolved),
            },
        }

    def table(self) -> list[list[Any]]:
        """A plain-text table for the CLI."""
        header = ["Metric", "Baseline", "Candidate", "Change", "Better?"]
        rows: list[list[Any]] = [header, ["-" * 16] * 5]
        for delta in self.deltas:
            rows.append(
                [
                    delta.label,
                    f"{delta.before:.2f}{delta.unit}",
                    f"{delta.after:.2f}{delta.unit}",
                    f"{delta.change:+.2f}",
                    "yes" if delta.improved else "no",
                ]
            )
        rows.append(["-" * 16] * 5)
        rows.append(
            [
                "Churn",
                f"{self.churn.changed_assignments}/{self.churn.total_before}",
                f"{self.churn.total_after} assigned",
                f"{self.churn.churn_ratio * 100:.1f}% changed",
                "-",
            ]
        )
        return rows

    def render(self) -> str:
        widths = [max(len(str(r[i])) for r in self.table()) for i in range(5)]
        out = []
        for row in self.table():
            out.append(
                "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
            )
        return "\n".join(out)


def _metric_deltas(baseline: Plan, candidate: Plan) -> list[MetricDelta]:
    return [
        MetricDelta(
            "score",
            "Plan score",
            baseline.score,
            candidate.score,
            higher_is_better=True,
        ),
        MetricDelta(
            "service_level",
            "Service level",
            baseline.service_level,
            candidate.service_level,
            unit="",
            higher_is_better=True,
        ),
        MetricDelta(
            "tasks_assigned",
            "Tasks assigned",
            float(baseline.tasks_assigned),
            float(candidate.tasks_assigned),
            higher_is_better=True,
        ),
        MetricDelta(
            "late_tasks",
            "Late tasks",
            float(baseline.late_tasks),
            float(candidate.late_tasks),
            higher_is_better=False,
        ),
        MetricDelta(
            "travel_km",
            "Travel (km)",
            baseline.total_travel_km,
            candidate.total_travel_km,
            unit=" km",
            higher_is_better=False,
        ),
        MetricDelta(
            "operating_cost",
            "Operating cost",
            baseline.total_operating_cost,
            candidate.total_operating_cost,
            higher_is_better=False,
        ),
        MetricDelta(
            "utilization",
            "Mean utilisation",
            _mean_utilization(baseline),
            _mean_utilization(candidate),
            higher_is_better=True,
        ),
        MetricDelta(
            "violations",
            "Violations",
            float(len(baseline.violations)),
            float(len(candidate.violations)),
            higher_is_better=False,
        ),
        MetricDelta(
            "runtime_s",
            "Solver runtime (s)",
            float(baseline.primary_solver_run.runtime_s if baseline.primary_solver_run else 0.0),
            float(candidate.primary_solver_run.runtime_s if candidate.primary_solver_run else 0.0),
            unit=" s",
            higher_is_better=False,
        ),
    ]


def _mean_utilization(plan: Plan) -> float:
    active = [u for u in plan.utilization if u.assigned_tasks > 0]
    if not active:
        return 0.0
    return sum(u.utilization for u in active) / len(active)


def compare_plans(
    baseline: Plan, candidate: Plan, scenario: Scenario
) -> PlanComparison:
    """Compare two plans for the same scenario."""
    if baseline.scenario_id != candidate.scenario_id:
        raise ValueError(
            f"cannot compare plans from different scenarios: "
            f"{baseline.scenario_id} vs {candidate.scenario_id}"
        )
    churn = compute_churn(baseline, candidate)

    before_tasks = set(baseline.by_task())
    after_tasks = set(candidate.by_task())
    before_violations = {v.task_id for v in baseline.violations if v.task_id}
    after_violations = {v.task_id for v in candidate.violations if v.task_id}

    return PlanComparison(
        baseline=baseline,
        candidate=candidate,
        scenario=scenario,
        churn=churn,
        deltas=tuple(_metric_deltas(baseline, candidate)),
        newly_assigned=tuple(sorted(after_tasks - before_tasks)),
        removed_assignments=tuple(sorted(before_tasks - after_tasks)),
        delayed=churn.delayed,
        newly_violated=tuple(sorted(after_violations - before_violations)),
        newly_resolved=tuple(sorted(before_violations - after_violations)),
    )


# ==========================================================================
# Recovery metrics
# ==========================================================================
@dataclass(slots=True)
class RecoveryReport:
    """Disruption -> degradation -> repair, in precise numbers."""

    disruption_id: str
    baseline_score: float
    degraded_score: float
    repaired_score: float
    baseline_service: float
    degraded_service: float
    repaired_service: float
    repair_seconds: float
    full_seconds: float
    full_score: float
    full_service: float
    repair_churn: ChurnReport
    full_churn: ChurnReport
    unrecovered_tasks: tuple[str, ...] = ()
    notes: str = ""

    @property
    def degradation(self) -> float:
        return max(0.0, self.baseline_service - self.degraded_service)

    @property
    def recovery(self) -> float:
        """Fraction of the lost service that the repair recovered, in [0, 1]."""
        loss = self.degradation
        if loss <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (self.repaired_service - self.degraded_service) / loss))

    @property
    def full_recovery(self) -> float:
        loss = self.degradation
        if loss <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (self.full_service - self.degraded_service) / loss))

    def to_dict(self) -> dict[str, Any]:
        return {
            "disruption_id": self.disruption_id,
            "baseline_score": round(self.baseline_score, 4),
            "degraded_score": round(self.degraded_score, 4),
            "repaired_score": round(self.repaired_score, 4),
            "full_score": round(self.full_score, 4),
            "baseline_service": round(self.baseline_service, 6),
            "degraded_service": round(self.degraded_service, 6),
            "repaired_service": round(self.repaired_service, 6),
            "degradation": round(self.degradation, 6),
            "recovery": round(self.recovery, 6),
            "recovery_pct": round(self.recovery * 100.0, 2),
            "full_recovery": round(self.full_recovery, 6),
            "full_recovery_pct": round(self.full_recovery * 100.0, 2),
            "repair_seconds": round(self.repair_seconds, 6),
            "full_seconds": round(self.full_seconds, 6),
            "repair_churn": self.repair_churn.to_dict(),
            "full_churn": self.full_churn.to_dict(),
            "unrecovered_tasks": list(self.unrecovered_tasks),
            "notes": self.notes,
        }

    def render(self) -> str:
        return (
            f"disruption {self.disruption_id}\n"
            f"  baseline   score={self.baseline_score:8.2f} service={self.baseline_service * 100:5.1f}%\n"
            f"  degraded   score={self.degraded_score:8.2f} service={self.degraded_service * 100:5.1f}%\n"
            f"  repaired   score={self.repaired_score:8.2f} service={self.repaired_service * 100:5.1f}%\n"
            f"  recovery   {self.recovery * 100:5.1f}%  "
            f"(churn {self.repair_churn.changed_assignments}/{self.repair_churn.total_before}, "
            f"{self.repair_seconds * 1000:.1f} ms)\n"
            f"  full       score={self.full_score:8.2f} recovery={self.full_recovery * 100:5.1f}%  "
            f"(churn {self.full_churn.changed_assignments}/{self.full_churn.total_before}, "
            f"{self.full_seconds * 1000:.1f} ms)"
        )


def build_recovery_report(
    *,
    disruption_id: str,
    baseline: Plan,
    degraded: Plan,
    repaired: Plan,
    full: Plan | None,
    repair_seconds: float,
    full_seconds: float,
    unrecovered_tasks: Sequence[str] = (),
    notes: str = "",
) -> RecoveryReport:
    """Assemble the recovery report from concrete plans.

    ``degraded`` is the plan that *would* stand if the disruption were simply
    absorbed with no replanning. ORION computes it by taking the pre-disruption
    plan and applying the disruption's own impact, which is precisely the "no
    replanning" arm of the disruption study.
    """
    full_plan = full or repaired
    return RecoveryReport(
        disruption_id=disruption_id,
        baseline_score=baseline.score,
        degraded_score=degraded.score,
        repaired_score=repaired.score,
        full_score=full_plan.score,
        baseline_service=baseline.service_level,
        degraded_service=degraded.service_level,
        repaired_service=repaired.service_level,
        repair_seconds=repair_seconds,
        full_seconds=full_seconds,
        full_service=full_plan.service_level,
        repair_churn=compute_churn(baseline, repaired),
        full_churn=compute_churn(baseline, full_plan),
        unrecovered_tasks=tuple(unrecovered_tasks),
        notes=notes,
    )


def degraded_plan_from(
    scenario_before: Scenario, plan: Plan, impact: Any
) -> Plan:
    """The 'no replanning' arm: keep the plan, but mark it as it now stands.

    Assignments that the disruption invalidated are dropped, because the world
    will not honour them - the simulation engine cancels them too. What remains
    is what the organisation would actually be able to deliver if it did nothing.
    """
    from orion.optimization.models import SchedulingContext

    affected = set(getattr(impact, "affected_tasks", ()) or ())
    kept = tuple(a for a in plan.assignments if a.task_id not in affected)
    from orion.domain.plans import Plan as _Plan
    from orion.domain.plans import SolverRun

    degraded = _Plan(
        id=f"{plan.id}-DEGRADED",
        scenario_id=plan.scenario_id,
        assignments=kept,
        status=SolverStatus.FEASIBLE,
        strategy="no_replan",
        tasks_total=plan.tasks_total,
        metadata={"dropped_tasks": sorted(affected), "note": "disruption absorbed without replanning"},
    )
    from orion.optimization.builder import build_plan as _build

    context = SchedulingContext.build(scenario_before)
    run = SolverRun(
        solver="NONE",
        status=SolverStatus.FEASIBLE,
        runtime_s=0.0,
        objective=0.0,
        feasible=True,
        notes="no replanning: the plan is used as-is minus infeasible assignments",
    )
    return _build(context, kept, run, strategy="no_replan")


__all__ = [
    "PlanComparison",
    "MetricDelta",
    "RecoveryReport",
    "compare_plans",
    "build_recovery_report",
    "degraded_plan_from",
]
