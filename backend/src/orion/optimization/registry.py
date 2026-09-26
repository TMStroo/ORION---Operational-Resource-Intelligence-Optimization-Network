
"""Solver registry and the capability matrix.

ORION does not force every solver onto every scenario. This module is the one
place that decides which solvers *apply*, and it derives that decision from the
scenario's own structure rather than from a hard-coded table in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from orion.domain.plans import (
    ALL_SOLVERS,
    SOLVER_DESCRIPTIONS,
    Plan,
    SolverName,
    SolverRun,
    SolverStatus,
)
from orion.domain.time_model import Interval
from orion.optimization.builder import build_plan
from orion.optimization.cp_sat_solver import CpSatSolver
from orion.optimization.heuristics import ConstructiveHeuristic
from orion.optimization.local_search import LocalSearchSolver
from orion.optimization.milp_solver import MilpSolver, available_backend
from orion.optimization.min_cost_flow import MinCostFlowSolver
from orion.optimization.models import SchedulingContext, SolverRequest

SolverFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """Static description of a solver, used for the capability matrix and docs."""

    name: str
    kind: str
    exact: bool
    supports_dependencies: bool
    supports_travel: bool
    supports_max_work: bool
    produces_proven_optimality: bool
    typical_scale: str
    description: str
    factory: SolverFactory | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "exact": self.exact,
            "supports_dependencies": self.supports_dependencies,
            "supports_travel": self.supports_travel,
            "supports_max_work": self.supports_max_work,
            "proven_optimality": self.produces_proven_optimality,
            "typical_scale": self.typical_scale,
            "description": self.description,
        }


SOLVER_SPECS: tuple[SolverSpec, ...] = (
    SolverSpec(
        name=SolverName.CP_SAT,
        kind="constraint_programming",
        exact=True,
        supports_dependencies=True,
        supports_travel=True,
        supports_max_work=True,
        produces_proven_optimality=True,
        typical_scale="<= ~150 tasks",
        description=SOLVER_DESCRIPTIONS[SolverName.CP_SAT],
        factory=CpSatSolver,
    ),
    SolverSpec(
        name=SolverName.MILP,
        kind="mixed_integer_linear",
        # `exact` means the formulation uses no heuristic shortcuts; it does not
        # mean it models the full objective. The linear model prices travel
        # between consecutive assignments but cannot express the cost of a
        # multi-stop route, so it is a relaxation of the true scheduling
        # problem - hence produces_proven_optimality=False and the
        # RELAXATION_OPTIMAL status it now reports.
        exact=True,
        supports_dependencies=True,
        supports_travel=True,
        supports_max_work=True,
        produces_proven_optimality=False,
        typical_scale="<= ~60 tasks (big-M grows quadratically)",
        description=SOLVER_DESCRIPTIONS[SolverName.MILP],
        factory=MilpSolver,
    ),
    SolverSpec(
        name=SolverName.MIN_COST_FLOW,
        kind="combinatorial_optimization",
        exact=False,
        supports_dependencies=False,
        supports_travel=False,
        supports_max_work=True,
        produces_proven_optimality=False,
        typical_scale="any (polynomial)",
        description=SOLVER_DESCRIPTIONS[SolverName.MIN_COST_FLOW],
        factory=MinCostFlowSolver,
    ),
    SolverSpec(
        name=SolverName.HEURISTIC,
        kind="constructive_heuristic",
        exact=False,
        supports_dependencies=True,
        supports_travel=True,
        supports_max_work=True,
        produces_proven_optimality=False,
        typical_scale="any (sub-second)",
        description=SOLVER_DESCRIPTIONS[SolverName.HEURISTIC],
        factory=ConstructiveHeuristic,
    ),
    SolverSpec(
        name=SolverName.LOCAL_SEARCH,
        kind="metaheuristic",
        exact=False,
        supports_dependencies=True,
        supports_travel=True,
        supports_max_work=True,
        produces_proven_optimality=False,
        typical_scale="any (budget-bounded)",
        description=SOLVER_DESCRIPTIONS[SolverName.LOCAL_SEARCH],
        factory=LocalSearchSolver,
    ),
)

SOLVER_INDEX: Mapping[str, SolverSpec] = {spec.name: spec for spec in SOLVER_SPECS}


def make_solver(name: str, **kwargs: Any) -> Any:
    """Instantiate a solver by canonical name."""
    spec = SOLVER_INDEX.get(name)
    if spec is None:
        raise ValueError(
            f"unknown solver {name!r}; available: {sorted(SOLVER_INDEX)}"
        )
    if spec.factory is None:  # pragma: no cover - all specs have factories
        raise ValueError(f"solver {name} has no factory")
    return spec.factory(**kwargs)


def solver_available(name: str) -> bool:
    """Whether a solver can run in this installation."""
    try:
        instance = make_solver(name)
    except Exception:  # pragma: no cover
        return False
    return bool(getattr(instance, "available", True))


def capability_matrix(context: SchedulingContext) -> dict[str, dict[str, Any]]:
    """Per-solver applicability for this specific scenario.

    Returned shape is what the UI's solver page and the report's capability
    table both consume, so the documentation cannot drift from the code.
    """
    scenario = context.scenario
    out: dict[str, dict[str, Any]] = {}
    for spec in SOLVER_SPECS:
        reasons: list[str] = []
        try:
            instance = make_solver(spec.name)
            applicable = bool(instance.supports(context))
        except Exception as exc:  # pragma: no cover
            applicable = False
            reasons.append(f"instantiation failed: {exc}")
        else:
            checker = getattr(instance, "unsupported_reason", None)
            if not applicable and callable(checker):
                reason = checker(context)
                if reason:
                    reasons.append(reason)
        if not solver_available(spec.name):
            reasons.append("backend not installed in this environment")
        out[spec.name] = {
            "applicable": applicable and solver_available(spec.name),
            "reasons": reasons,
            **spec.to_dict(),
        }
    return out


def applicable_solvers(context: SchedulingContext) -> list[str]:
    return [name for name, info in capability_matrix(context).items() if info["applicable"]]


@dataclass(slots=True)
class SolveResult:
    """A single solver's contribution to a comparison."""

    solver: str
    plan: Plan
    run: SolverRun
    diagnostics: Mapping[str, Any]
    applicable: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.run.status in {
            SolverStatus.OPTIMAL,
            SolverStatus.FEASIBLE,
            SolverStatus.TIME_LIMIT,
        }

    def to_row(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "status": self.run.status,
            "applicable": self.applicable,
            "reason": self.reason,
            "runtime_s": round(self.run.runtime_s, 4),
            "objective": round(self.plan.score, 4),
            "feasible": self.run.feasible,
            "num_variables": self.run.num_variables,
            "num_constraints": self.run.num_constraints,
            "optimality_gap": (
                round(self.run.optimality_gap, 6)
                if self.run.optimality_gap is not None
                else None
            ),
            "tasks_assigned": self.plan.tasks_assigned,
            "tasks_total": self.plan.tasks_total,
            "late_tasks": self.plan.late_tasks,
            "travel_km": round(self.plan.total_travel_km, 3),
            "operating_cost": round(self.plan.total_operating_cost, 3),
            "service_level": round(self.plan.service_level, 6),
            "violations": len(self.plan.violations),
            "hard_violations": len(self.plan.hard_violations),
        }


#: Statuses for which a reported objective is a real, comparable score. An
#: errored or infeasible run has no solution quality to report, and its 0.0 is
#: left alone so a missing result is never mistaken for a perfect one.
_COMPARABLE_STATUSES = frozenset(
    {
        getattr(SolverStatus, "OPTIMAL", "OPTIMAL"),
        getattr(SolverStatus, "FEASIBLE", "FEASIBLE"),
        getattr(SolverStatus, "TIME_LIMIT", "TIME_LIMIT"),
        getattr(SolverStatus, "RELAXATION_OPTIMAL", "RELAXATION_OPTIMAL"),
    }
)


def run_solver(
    name: str,
    context: SchedulingContext,
    request: SolverRequest,
    *,
    plan_id: str | None = None,
    **solver_kwargs: Any,
) -> SolveResult:
    """Run one solver, handling every failure mode explicitly.

    A solver that raises, times out, or is not applicable produces a
    :class:`SolveResult` with a non-success status and a human-readable reason -
    never an exception that escapes into the API, and never a plan that looks
    usable when it is not.
    """
    from orion.domain.errors import SolverUnavailableError

    spec = SOLVER_INDEX.get(name)
    if spec is None:
        return SolveResult(
            solver=name,
            plan=Plan(id=plan_id or "PLAN-INVALID", scenario_id=context.scenario.id, assignments=()),
            run=SolverRun(
                solver=name,
                status=SolverStatus.ERROR,
                runtime_s=0.0,
                objective=0.0,
                feasible=False,
                notes=f"unknown solver name {name!r}",
            ),
            diagnostics={},
            applicable=False,
            reason=f"unknown solver {name!r}",
        )

    try:
        instance = make_solver(name, **solver_kwargs)
    except (SolverUnavailableError, Exception) as exc:
        return SolveResult(
            solver=name,
            plan=Plan(id=plan_id or "PLAN-ERROR", scenario_id=context.scenario.id, assignments=()),
            run=SolverRun(
                solver=name,
                status=SolverStatus.ERROR,
                runtime_s=0.0,
                objective=0.0,
                feasible=False,
                notes=f"could not instantiate: {exc}",
            ),
            diagnostics={},
            applicable=False,
            reason=str(exc),
        )

    if not getattr(instance, "supports", lambda _c: True)(context):
        checker = getattr(instance, "unsupported_reason", None)
        reason = checker(context) if callable(checker) else "solver not applicable to this scenario"
        # The declined run is attached to the plan as a real SolverRun. Without
        # this the plan carried an empty `solver_runs`, so downstream code saw an
        # empty plan whose own status field read FEASIBLE and recorded a score of
        # 0.0 for a solver that had explicitly declined the instance. "Not
        # applicable" has to be visible in the plan, not only in a warning string.
        declined_run = SolverRun(
            solver=name,
            status=SolverStatus.NOT_APPLICABLE,
            runtime_s=0.0,
            objective=0.0,
            feasible=False,
            notes=f"not applicable: {reason}",
        )
        return SolveResult(
            solver=name,
            plan=Plan(
                id=plan_id or "PLAN-NA",
                scenario_id=context.scenario.id,
                assignments=(),
                solver_runs=(declined_run,),
                status=SolverStatus.NOT_APPLICABLE,
                strategy=name,
                tasks_total=len(context.scenario.tasks),
                service_level=0.0,
                metadata={"not_applicable_reason": reason},
            ),
            run=declined_run,
            diagnostics={},
            applicable=False,
            reason=reason,
        )

    try:
        assignments, run, diagnostics = instance.solve(context, request)
        plan = build_plan(
            context, assignments, run, plan_id=plan_id, strategy=name
        )
        # Fairness contract: a solver's own reported objective is whatever that
        # solver chose to write down, and solvers disagree - CP-SAT reports a
        # scaled integer value, the constructive heuristics and the flow
        # relaxation report nothing at all (0.0), and the MILP reports its SCIP
        # objective. Comparing those numbers across solvers would compare
        # reporting conventions, not solution quality.
        #
        # So the single value that is ever compared is recomputed here, by
        # ORION's own objective, from the assignments the solver produced. The
        # solver's own figure is preserved as `internal_objective` for
        # diagnosis. This is what makes the benchmark table meaningful: every
        # row is scored the same way, by code that did not participate in
        # solving.
        scored = plan.objective.total
        internal = run.objective
        if run.status in _COMPARABLE_STATUSES and plan.assignments:
            run = replace(
                run,
                objective=scored,
                raw={**dict(run.raw), "internal_objective": internal},
            )
        return SolveResult(
            solver=name, plan=plan, run=run, diagnostics=diagnostics, applicable=True
        )
    except Exception as exc:  # noqa: BLE001 - every solver failure is reported, not raised
        return SolveResult(
            solver=name,
            plan=Plan(id=plan_id or "PLAN-ERROR", scenario_id=context.scenario.id, assignments=()),
            run=SolverRun(
                solver=name,
                status=SolverStatus.ERROR,
                runtime_s=0.0,
                objective=0.0,
                feasible=False,
                notes=f"{type(exc).__name__}: {exc}",
            ),
            diagnostics={},
            applicable=True,
            reason=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "SolverSpec",
    "SOLVER_SPECS",
    "SOLVER_INDEX",
    "make_solver",
    "solver_available",
    "capability_matrix",
    "applicable_solvers",
    "SolveResult",
    "run_solver",
    "available_backend",
]
