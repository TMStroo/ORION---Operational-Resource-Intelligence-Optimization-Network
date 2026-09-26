"""Plan, disruption and solver metrics.

Every definition here is *exact* - the formulas are the ones the README and the
technical report quote, so a reviewer can recompute any number from the raw plan.
Metrics are computed from a :class:`~orion.domain.plans.Plan`, never stored
separately, so they cannot drift from the plan they describe.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from orion.domain.plans import Plan, SolverName


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One metric with a human label, value and unit.

    ``value`` is always the *raw* fraction in ``[0, 1]`` for fractional metrics,
    so downstream code never has to guess the scale. ``unit`` is a display hint:
    ``"%"`` means "render this fraction as a percentage". :attr:`display` is the
    only place the conversion happens, which is what keeps a report's
    percentages and a figure's percentages in agreement.
    """

    key: str
    label: str
    value: float
    unit: str = ""
    higher_is_better: bool | None = None

    @property
    def is_fraction(self) -> bool:
        return self.unit == "%"

    @property
    def display(self) -> str:
        """Human-readable rendering of the value."""
        if self.unit == "%":
            return f"{self.value * 100:,.1f}%"
        if self.unit == "min":
            return f"{self.value:,.1f} min"
        if self.unit == "km":
            return f"{self.value:,.2f} km"
        if self.unit == "s":
            return f"{self.value:,.3f} s"
        if self.unit:
            return f"{self.value:,.2f} {self.unit}"
        return f"{self.value:,.2f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "display": self.display,
        }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _lateness(plan: Plan) -> list[float]:
    """Per-assignment lateness in minutes (0.0 when on time)."""
    return [float(a.lateness) for a in plan.assignments]


def _mean_utilisation(plan: Plan) -> float:
    """Mean utilisation across the resources the plan actually dispatched.

    Undispatched resources are excluded: including a resource that was never
    asked to work would make any plan that uses *fewer* resources look better,
    which is the opposite of what a planner wants to see.
    """
    used = {a.resource_id for a in plan.assignments}
    return _mean([u.utilization for u in plan.utilization if u.resource_id in used])


def _solver_runtime(plan: Plan) -> float:
    """Wall-clock of the run that produced this plan, in seconds."""
    return max((r.runtime_s for r in plan.solver_runs), default=0.0)


def _model_size(plan: Plan) -> tuple[int, int]:
    """Largest (variables, constraints) seen among the plan's solver runs."""
    variables = max((r.num_variables for r in plan.solver_runs), default=0)
    constraints = max((r.num_constraints for r in plan.solver_runs), default=0)
    return int(variables), int(constraints)


def _optimality_gap(plan: Plan) -> float | None:
    """Best available relative optimality gap, or ``None`` if never proven."""
    gaps = [r.optimality_gap for r in plan.solver_runs if r.optimality_gap is not None]
    if not gaps:
        return None
    best = min(gaps)
    return None if best < 0 else best


def plan_metric_rows(plan: Plan, scenario: Any) -> list[MetricRow]:
    """Return the full planning metric vector for *plan*.

    Definitions (all denominators are the scenario's task count, so the
    completion rate is "share of the scenario's demand that got served"):

    ``task_completion_rate``
        ``tasks_assigned / num_tasks``. A task left unassigned counts as not
        completed - it is a service failure, not a neutral omission.
    ``priority_weighted_completion``
        ``sum(priority * assigned) / sum(priority)`` over all tasks, so dropping
        a CRITICAL task costs far more than dropping a LOW one.
    ``deadline_violation_rate``
        ``late_tasks / tasks_assigned``: the share of *dispatched* work that
        misses its deadline. Reported next to the completion rate because the
        two move in opposite directions under lateness pressure.
    ``mean_lateness`` / ``max_lateness``
        Minutes past deadline, over late tasks only. A plan with no late task
        reports 0.0 rather than NaN, so aggregates stay computable.
    """
    n_tasks = len(scenario.tasks)
    priority_total = sum(t.priority for t in scenario.tasks) or 1
    assigned_priorities = 0.0
    for a in plan.assignments:
        assigned_priorities += scenario.task(a.task_id).priority

    rows = [
        MetricRow("plan_score", "Plan score", plan.objective.total, higher_is_better=True),
        MetricRow("service_level", "Service level", plan.service_level, unit="%", higher_is_better=True),
        MetricRow(
            "task_completion_rate",
            "Task completion rate",
            (plan.tasks_assigned / n_tasks) if n_tasks else 0.0,
            unit="%",
            higher_is_better=True,
        ),
        MetricRow(
            "priority_weighted_completion",
            "Priority-weighted completion",
            assigned_priorities / priority_total,
            unit="%",
            higher_is_better=True,
        ),
        MetricRow(
            "deadline_violation_rate",
            "Deadline violation rate",
            (plan.late_tasks / plan.tasks_assigned) if plan.tasks_assigned else 0.0,
            unit="%",
            higher_is_better=False,
        ),
        MetricRow("mean_lateness", "Mean lateness", _mean(_lateness(plan)), unit="min", higher_is_better=False),
        MetricRow("max_lateness", "Max lateness", max(_lateness(plan), default=0.0), unit="min", higher_is_better=False),
        MetricRow("travel_distance", "Travel distance", plan.total_travel_km, unit="km", higher_is_better=False),
        MetricRow("operating_cost", "Operating cost", plan.total_operating_cost, unit="cost", higher_is_better=False),
        MetricRow("mean_utilisation", "Mean utilisation", _mean_utilisation(plan), unit="%", higher_is_better=True),
        MetricRow("violations", "Constraint violations", float(len(plan.violations)), higher_is_better=False),
        MetricRow("tasks_assigned", "Tasks assigned", float(plan.tasks_assigned), higher_is_better=True),
        MetricRow("late_tasks", "Late tasks", float(plan.late_tasks), higher_is_better=False),
        MetricRow("solver_runtime", "Solver runtime", _solver_runtime(plan), unit="s", higher_is_better=False),
        MetricRow("model_variables", "Model variables", float(_model_size(plan)[0]), higher_is_better=False),
        MetricRow("model_constraints", "Model constraints", float(_model_size(plan)[1]), higher_is_better=False),
    ]
    gap = _optimality_gap(plan)
    if gap is not None:
        rows.append(MetricRow("optimality_gap", "Optimality gap", float(gap), unit="%", higher_is_better=False))
    return rows


def objective_components(plan: Plan) -> dict[str, float]:
    """The plan's objective split, for explaining where score comes from."""
    return plan.objective.to_dict()


class FailureCategory(str, Enum):
    """Categories ORION is known to fail in.

    The report quantifies each of these from the benchmark suite rather than
    asserting them, so the taxonomy is driven by what the runs actually show.
    """

    INFEASIBLE_SCENARIO = "infeasible_scenario"
    DEADLINE_COLLAPSE = "deadline_collapse"
    RESOURCE_BOTTLENECK = "resource_bottleneck"
    SOLVER_TIMEOUT = "solver_timeout"
    HEURISTIC_QUALITY_GAP = "heuristic_quality_gap"
    POOR_LOCAL_REPAIR = "poor_local_repair"
    EXCESSIVE_CHURN = "excessive_churn"
    MISLEADING_OBJECTIVE = "misleading_objective"
    TRIVIAL_PROBLEM = "trivial_problem"

    #: The solver raised and produced no plan. Distinct from every other
    #: category because the row is not a *bad* result, it is the *absence* of
    #: one, and it must never be averaged into a quality table.
    SOLVER_ERROR = "solver_error"
    #: The solver reported success but its plan violates a hard constraint, or
    #: is empty when the instance demonstrably admits work. The silent-empty-plan
    #: case is why this category exists: it is a success status wrapping a
    #: failure.
    SOLUTION_INVALID = "solution_invalid"
    #: A scenario for which no resource can serve any task. Legitimate, and
    #: distinct from SOLVER_ERROR: the solver is correct, the instance is not.
    NO_ELIGIBLE_PAIR = "no_eligible_pair"


@dataclass(slots=True)
class FailureRecord:
    """A single observed failure instance."""

    category: FailureCategory
    scenario_name: str
    detail: str
    root_cause: str
    consequence: str
    mitigation: str
    severity: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "scenario": self.scenario_name,
            "detail": self.detail,
            "root_cause": self.root_cause,
            "consequence": self.consequence,
            "mitigation": self.mitigation,
            "severity": self.severity,
        }


def detect_failures(
    plan: Plan,
    scenario: Any,
    *,
    reference_score: float | None = None,
    reference_gap: float | None = None,
    churn_fraction: float | None = None,
    repair_service: float | None = None,
    baseline_service: float | None = None,
) -> list[FailureRecord]:
    """Derive failure records from one plan's own numbers.

    Thresholds are deliberately conservative: a record is only emitted when the
    plan's metrics cross a line that a planner would care about, so the failure
    analysis stays a list of real problems rather than a list of every run.
    """
    out: list[FailureRecord] = []
    name = scenario.name
    n = len(scenario.tasks)
    # A plan records the solver inside `solver_runs`, not as its own attributes;
    # an ad-hoc plan can carry none at all, so both reads are guarded rather
    # than assumed.
    last_run = plan.solver_runs[-1] if plan.solver_runs else None
    solver_name = last_run.solver if last_run else plan.strategy
    solver_runtime = last_run.runtime_s if last_run else 0.0

    if plan.status == "INFEASIBLE":
        out.append(
            FailureRecord(
                FailureCategory.INFEASIBLE_SCENARIO,
                name,
                "solver proved no feasible plan exists",
                root_cause="demand exceeds capability, capacity or horizon",
                consequence="no plan can be served; the planner must change inputs",
                mitigation="run `orion scenario validate` to surface the binding constraint",
                severity=3,
            )
        )
    if plan.status == "TIME_LIMIT":
        out.append(
            FailureRecord(
                FailureCategory.SOLVER_TIMEOUT,
                name,
                f"no proven optimum within {solver_runtime:.1f}s",
                root_cause="model size above the budget the planner allowed",
                consequence="quality is unknown, not optimal",
                mitigation="raise the time budget, or accept the FEASIBLE plan and raise a follow-up",
                severity=1,
            )
        )
    if n and plan.late_tasks / max(1, plan.tasks_assigned) > 0.5 and plan.tasks_assigned:
        out.append(
            FailureRecord(
                FailureCategory.DEADLINE_COLLAPSE,
                name,
                f"{plan.late_tasks}/{plan.tasks_assigned} dispatched tasks are late",
                root_cause="deadline slack too small relative to travel and duration",
                consequence="service level collapses; the plan is technically feasible but useless",
                mitigation="widen w_late or the time windows before trusting the plan",
                severity=2,
            )
        )
    if reference_gap is not None and reference_gap > 0.05 and solver_name == SolverName.HEURISTIC:
        out.append(
            FailureRecord(
                FailureCategory.HEURISTIC_QUALITY_GAP,
                name,
                f"heuristic is {reference_gap:.1%} below the best known plan",
                root_cause="greedy insertion ignores the opportunity cost of committing a resource early",
                consequence="cheap answers can be materially worse on clustered instances",
                mitigation="use local search or CP-SAT when the budget allows",
                severity=2,
            )
        )
    if churn_fraction is not None and churn_fraction > 0.5:
        out.append(
            FailureRecord(
                FailureCategory.EXCESSIVE_CHURN,
                name,
                f"{churn_fraction:.0%} of assignments changed",
                root_cause="disruption touched a resource that served most of the horizon",
                consequence="the planner loses continuity; field teams must be re-briefed",
                mitigation="prefer local repair, or accept churn as the cost of a better score",
                severity=2,
            )
        )
    if (
        repair_service is not None
        and baseline_service is not None
        and baseline_service > 0
        and repair_service < 0.6 * baseline_service
    ):
        out.append(
            FailureRecord(
                FailureCategory.POOR_LOCAL_REPAIR,
                name,
                f"repair kept only {repair_service:.0%} of baseline service",
                root_cause="no replacement resource satisfies the affected tasks' hard constraints",
                consequence="fast repair gives a badly degraded plan",
                mitigation="fall back to full re-optimization when repair service < 0.8 * baseline",
                severity=3,
            )
        )
    if reference_score is not None and plan.objective.total < 0.0:
        out.append(
            FailureRecord(
                FailureCategory.MISLEADING_OBJECTIVE,
                name,
                f"best available plan still scores {plan.objective.total:.1f} (negative)",
                root_cause="penalty weights exceed the scenario's gross service value",
                consequence="every dispatch decision is dominated by cost, and the score is unreadable",
                mitigation="rescale weights; see the calibration rule in ObjectiveWeights",
                severity=3,
            )
        )
    return out


def aggregate_failures(records: Iterable[FailureRecord]) -> dict[str, dict[str, Any]]:
    """Frequency, worst severity and an example per failure category."""
    buckets: dict[str, list[FailureRecord]] = {}
    for r in records:
        buckets.setdefault(r.category.value, []).append(r)
    summary: dict[str, dict[str, Any]] = {}
    for cat, items in buckets.items():
        items = sorted(items, key=lambda r: -r.severity)
        summary[cat] = {
            "count": len(items),
            "worst_severity": max(i.severity for i in items),
            "representative": items[0].to_dict(),
        }
    return summary


def is_finite(value: float) -> bool:
    """Guard used by the report generator: NaN must never reach a headline."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))
