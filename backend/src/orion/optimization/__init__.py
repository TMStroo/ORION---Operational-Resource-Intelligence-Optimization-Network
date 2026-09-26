
"""Optimisation layer: explicit formulations and five solution methods.

Solvers
-------

``CP_SAT``          exact constraint programming, proven optimality + dual bound
``MILP``            exact mixed-integer linear, big-M disjunctions
``MIN_COST_FLOW``   transportation relaxation, polynomial time, a valid bound
``HEURISTIC``       constructive greedy insertion, sub-second, always applicable
``LOCAL_SEARCH``    ALNS destroy-and-repair, budget-bounded

Shared infrastructure
---------------------

:class:`~orion.optimization.models.SchedulingContext`
    One computation of the candidate set, time windows and travel coefficients,
    consumed by every solver. This is what makes the comparison fair.
:func:`~orion.optimization.builder.build_plan`
    One scoring and validation implementation, so no solver's number comes from
    its own objective function.
"""

from __future__ import annotations

from orion.optimization.builder import build_plan, build_utilization, reset_plan_counter
from orion.optimization.cp_sat_solver import CpSatSolver
from orion.optimization.heuristics import ConstructiveHeuristic, constructive_plan
from orion.optimization.local_search import DESTRUCTION_OPERATORS, LocalSearchSolver
from orion.optimization.milp_solver import MilpSolver, available_backend
from orion.optimization.min_cost_flow import MinCostFlowSolver
from orion.optimization.models import (
    PairInfo,
    SchedulingContext,
    SolverRequest,
    assignments_are_consistent,
    schedule_sequence,
)
from orion.optimization.objective import (
    ServiceLevel,
    describe_weights,
    lateness_penalty,
    objective_breakdown_from_assignments,
    service_level,
    task_value,
)
from orion.optimization.registry import (
    SOLVER_INDEX,
    SOLVER_SPECS,
    SolveResult,
    SolverSpec,
    applicable_solvers,
    capability_matrix,
    make_solver,
    run_solver,
    solver_available,
)

__all__ = [
    "SchedulingContext",
    "PairInfo",
    "SolverRequest",
    "schedule_sequence",
    "assignments_are_consistent",
    "CpSatSolver",
    "MilpSolver",
    "MinCostFlowSolver",
    "ConstructiveHeuristic",
    "LocalSearchSolver",
    "DESTRUCTION_OPERATORS",
    "build_plan",
    "build_utilization",
    "reset_plan_counter",
    "constructive_plan",
    "task_value",
    "lateness_penalty",
    "objective_breakdown_from_assignments",
    "service_level",
    "ServiceLevel",
    "describe_weights",
    "SOLVER_SPECS",
    "SOLVER_INDEX",
    "SolverSpec",
    "SolveResult",
    "run_solver",
    "capability_matrix",
    "applicable_solvers",
    "make_solver",
    "solver_available",
    "available_backend",
]
