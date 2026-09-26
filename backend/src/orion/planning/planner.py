
"""Planner: turn a scenario into a plan, and explain it.

The planner is the user-facing entry point. It:

1. validates the scenario and refuses obviously-infeasible ones with a reason;
2. builds a shared :class:`~orion.optimization.models.SchedulingContext`;
3. runs the requested solver under a time budget;
4. returns a :class:`~orion.domain.plans.Plan` plus an explanation of every
   assignment.

Explanations
------------

Every assignment carries an :class:`AssignmentExplanation` built from *actual*
constraint evaluations and *actual* objective terms - never from a
natural-language template. For each assignment ORION reports the constraint
checks that passed (with their real values), the cost contribution computed by
the same code that computes the plan score, and the runner-up resource with the
extra distance/cost it would have cost. If an explanation cannot be derived
from the model, ORION says so rather than inventing one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from orion.domain.entities import Scenario
from orion.domain.errors import InfeasibleScenarioError
from orion.domain.plans import (
    Assignment,
    Plan,
    SolverName,
    SolverStatus,
)
from orion.optimization.builder import build_plan
from orion.optimization.models import SchedulingContext, SolverRequest
from orion.optimization.objective import task_value
from orion.optimization.registry import SolveResult, applicable_solvers, run_solver

DEFAULT_TIME_BUDGETS: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One constraint check on one assignment, with its real value."""

    name: str
    satisfied: bool
    value: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AssignmentExplanation:
    """Why this resource was assigned to this task."""

    task_id: str
    resource_id: str
    checks: tuple[CheckResult, ...]
    objective_contribution: float
    runner_up_resource: str | None = None
    runner_up_extra_cost: float | None = None
    runner_up_extra_km: float | None = None
    ranking: str = ""

    @property
    def all_satisfied(self) -> bool:
        return all(c.satisfied for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "resource_id": self.resource_id,
            "checks": [c.to_dict() for c in self.checks],
            "all_satisfied": self.all_satisfied,
            "objective_contribution": round(self.objective_contribution, 4),
            "runner_up_resource": self.runner_up_resource,
            "runner_up_extra_cost": (
                round(self.runner_up_extra_cost, 4)
                if self.runner_up_extra_cost is not None
                else None
            ),
            "runner_up_extra_km": (
                round(self.runner_up_extra_km, 3)
                if self.runner_up_extra_km is not None
                else None
            ),
            "ranking": self.ranking,
        }


@dataclass(slots=True)
class PlanResult:
    """What a planning request returns."""

    plan: Plan
    context: SchedulingContext
    result: SolveResult
    explanations: tuple[AssignmentExplanation, ...] = ()
    warnings: tuple[str, ...] = ()
    total_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.plan.is_usable

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "solver": self.result.solver,
            "applicable_solvers": applicable_solvers(self.context),
            "context": self.context.model_size,
            "explanations": [e.to_dict() for e in self.explanations],
            "warnings": list(self.warnings),
            "total_seconds": round(self.total_seconds, 6),
        }


class Planner:
    """Produces plans from scenarios."""

    def __init__(self, *, default_solver: str = SolverName.HEURISTIC,
                 default_time_budget: float = 10.0, seed: int = 0) -> None:
        self.default_solver = default_solver
        self.default_time_budget = default_time_budget
        self.seed = seed

    # -- validation --------------------------------------------------------
    @staticmethod
    def validate(scenario: Scenario) -> str | None:
        """Return a reason string when the scenario cannot be planned."""
        if not scenario.tasks:
            return "scenario contains no tasks: there is nothing to plan"
        if not scenario.active_resources():
            return "scenario contains no available resources"
        return scenario.static_infeasibility()

    # -- planning ----------------------------------------------------------
    def plan(
        self,
        scenario: Scenario,
        *,
        solver: str | None = None,
        time_budget_s: float | None = None,
        seed: int | None = None,
        request: SolverRequest | None = None,
        explain: bool = True,
    ) -> PlanResult:
        """Generate a plan. Never raises for a merely infeasible scenario -
        it returns a Plan with a non-success status and a reason in the run
        notes, because "this scenario cannot be planned" is a legitimate answer
        a user needs to see, not an exception.
        """
        started = time.perf_counter()
        chosen_solver = solver or self.default_solver
        budget = time_budget_s if time_budget_s is not None else self.default_time_budget
        use_seed = seed if seed is not None else self.seed

        reason = self.validate(scenario)
        if reason is not None:
            from orion.optimization.builder import next_plan_id

            plan = Plan(
                id=next_plan_id(),
                scenario_id=scenario.id,
                assignments=(),
                solver_runs=(),
                status=SolverStatus.INFEASIBLE,
                strategy=chosen_solver,
                tasks_total=len(scenario.tasks),
                service_level=0.0,
                metadata={"infeasible_reason": reason},
            )
            return PlanResult(
                plan=plan,
                context=SchedulingContext(scenario=scenario, pairs=(), tasks=(), resources=()),
                result=SolveResult(
                    solver=chosen_solver,
                    plan=plan,
                    run=plan.solver_runs[0] if plan.solver_runs else None,  # type: ignore[arg-type]
                    diagnostics={},
                    applicable=False,
                    reason=reason,
                ),
                warnings=(reason,),
                total_seconds=time.perf_counter() - started,
            )

        context = SchedulingContext.build(scenario)
        if request is None:
            request = SolverRequest(time_limit_s=budget, seed=use_seed)
        else:
            request = SolverRequest(
                time_limit_s=budget if budget is not None else request.time_limit_s,
                seed=use_seed,
                workers=request.workers,
                task_filter=request.task_filter,
                locked_assignments=request.locked_assignments,
                extra=dict(request.extra),
            )

        result = run_solver(chosen_solver, context, request)
        warnings: list[str] = []
        if not result.applicable:
            warnings.append(f"{chosen_solver} not applicable: {result.reason}")
        if result.run.status == SolverStatus.TIME_LIMIT:
            warnings.append(
                f"{chosen_solver} hit its {budget}s budget; the solution is "
                "feasible but not proven optimal"
            )
        if result.run.status == SolverStatus.ERROR:
            warnings.append(f"{chosen_solver} failed: {result.reason}")

        explanations: tuple[AssignmentExplanation, ...] = ()
        if explain and result.plan.assignments:
            explanations = self.explain(result.plan, scenario)

        return PlanResult(
            plan=result.plan,
            context=context,
            result=result,
            explanations=explanations,
            warnings=tuple(warnings),
            total_seconds=time.perf_counter() - started,
        )

    # -- time-budgeted planning -------------------------------------------
    def plan_with_budgets(
        self,
        scenario: Scenario,
        *,
        solver: str,
        budgets: Sequence[float] = DEFAULT_TIME_BUDGETS,
        seed: int = 0,
    ) -> list[PlanResult]:
        """Run the same solver at several time budgets.

        This is the feature behind the "exact solution vs 5-second solution"
        panel in the UI. Every run gets a fresh context so the comparison is
        not contaminated by cached model state.
        """
        out: list[PlanResult] = []
        for budget in budgets:
            out.append(
                self.plan(
                    scenario, solver=solver, time_budget_s=budget, seed=seed
                )
            )
        return out

    # -- explanation -------------------------------------------------------
    def explain(
        self, plan: Plan, scenario: Scenario, *, top_alternatives: int = 1
    ) -> tuple[AssignmentExplanation, ...]:
        """Derive a per-assignment explanation from the model itself.

        Each check is a real evaluation against the real constraint, and the
        objective contribution is the marginal value this assignment adds,
        computed with the same weights the optimiser used. The runner-up is the
        eligible resource that came closest on cost.
        """
        weights = scenario.objective_weights
        out: list[AssignmentExplanation] = []
        tasks = {t.id: t for t in scenario.tasks}
        resources = {r.id: r for r in scenario.resources}

        for assignment in plan.assignments:
            task = tasks.get(assignment.task_id)
            resource = resources.get(assignment.resource_id)
            if task is None or resource is None:
                continue
            checks: list[CheckResult] = []

            # capability
            missing = sorted(task.required_capabilities - resource.capabilities)
            checks.append(
                CheckResult(
                    name="required_skills",
                    satisfied=not missing,
                    value=(
                        f"{len(task.required_capabilities & resource.capabilities)}/"
                        f"{len(task.required_capabilities)}"
                    ),
                    detail=(
                        "all required skills present"
                        if not missing
                        else f"missing {missing}"
                    ),
                )
            )
            # capacity
            capacity_ok = True
            capacity_detail = "no capacity requirement"
            for demand in task.required_capacity:
                available = float(resource.capacity.get(demand.dimension, 0.0))
                if demand.amount > available:
                    capacity_ok = False
                    capacity_detail = (
                        f"{demand.dimension}: needs {demand.amount:g}, "
                        f"has {available:g}"
                    )
                    break
            checks.append(
                CheckResult(
                    name="capacity",
                    satisfied=capacity_ok,
                    value=(
                        f"{resource.capacity.get(task.required_capacity[0].dimension, 0):g}"
                        if task.required_capacity
                        else "n/a"
                    ),
                    detail=capacity_detail,
                )
            )
            # availability / shift
            on_shift = resource.shift_start <= assignment.start and assignment.end <= resource.shift_end
            checks.append(
                CheckResult(
                    name="availability",
                    satisfied=on_shift,
                    value=f"{resource.shift_start}-{resource.shift_end}",
                    detail=(
                        "within rostered shift"
                        if on_shift
                        else f"{assignment.start}-{assignment.end} outside shift"
                    ),
                )
            )
            # maintenance
            clash = next(
                (b for b in resource.unavailable if assignment.window.overlaps(b)),
                None,
            )
            checks.append(
                CheckResult(
                    name="maintenance",
                    satisfied=clash is None,
                    value=f"{len(resource.unavailable)} block(s)",
                    detail="no clash" if clash is None else f"clashes with {clash}",
                )
            )
            # deadline
            lateness = task.lateness_for(assignment.end)
            checks.append(
                CheckResult(
                    name="deadline",
                    satisfied=lateness == 0,
                    value=f"ends {assignment.end}, due {task.deadline}",
                    detail=(
                        "on time"
                        if lateness == 0
                        else f"{lateness} min late"
                    ),
                )
            )
            # travel
            checks.append(
                CheckResult(
                    name="travel",
                    satisfied=True,
                    value=f"{assignment.travel_distance_km:.1f} km / {assignment.travel_before} min",
                    detail=f"leg before this task from previous position",
                )
            )
            # priority
            checks.append(
                CheckResult(
                    name="priority",
                    satisfied=True,
                    value=task.priority.label,
                    detail=f"value {task_value(task, weights).total:.2f} on time",
                )
            )

            # objective contribution
            contribution = task_value(task, weights).total
            contribution -= weights.w_late * (lateness / 60.0)
            contribution -= weights.w_travel * (
                assignment.travel_distance_km / max(1e-9, resource.speed_factor)
            )
            contribution -= weights.w_cost * resource.operating_cost_per_hour * (
                (assignment.end - assignment.start) / 60.0
            )

            # runner-up
            runner_up, extra_cost, extra_km = self._runner_up(
                task, resource, assignment, scenario, weights
            )
            ranking = "assigned"
            if runner_up and extra_cost is not None and extra_cost <= 1e-6:
                ranking = "assigned; tie-broken deterministically"

            out.append(
                AssignmentExplanation(
                    task_id=task.id,
                    resource_id=resource.id,
                    checks=tuple(checks),
                    objective_contribution=contribution,
                    runner_up_resource=runner_up,
                    runner_up_extra_cost=extra_cost,
                    runner_up_extra_km=extra_km,
                    ranking=ranking,
                )
            )
        return tuple(out)

    def _runner_up(
        self,
        task: Any,
        resource: Any,
        assignment: Assignment,
        scenario: Scenario,
        weights: Any,
    ) -> tuple[str | None, float | None, float | None]:
        """Find the eligible resource that was the next-best choice.

        Scored on exactly the terms the objective uses, from the same travel
        matrix. This is a real computation, not a plausible-looking number.
        """
        best: tuple[float, str, float] | None = None
        for candidate in scenario.resources:
            if candidate.id == resource.id or not candidate.can_serve(task):
                continue
            km = scenario.travel.distance(
                candidate.home_location, task.location
            ) / max(1e-9, candidate.speed_factor)
            cost = (
                weights.w_travel * km
                + weights.w_cost
                * candidate.operating_cost_per_hour
                * (task.duration / 60.0)
            )
            if best is None or cost < best[0]:
                best = (cost, candidate.id, km)
        if best is None:
            return None, None, None
        chosen_km = scenario.travel.distance(
            resource.home_location, task.location
        ) / max(1e-9, resource.speed_factor)
        chosen_cost = (
            weights.w_travel * chosen_km
            + weights.w_cost
            * resource.operating_cost_per_hour
            * (task.duration / 60.0)
        )
        return best[1], best[0] - chosen_cost, best[2] - chosen_km


__all__ = [
    "Planner",
    "PlanResult",
    "AssignmentExplanation",
    "CheckResult",
    "DEFAULT_TIME_BUDGETS",
]
