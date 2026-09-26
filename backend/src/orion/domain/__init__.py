
"""Domain layer: entities, constraints, plans, events.

Pure data and pure functions. Nothing in this package imports a solver, a
database, or the web layer.
"""

from __future__ import annotations

from orion.domain.capacity import (
    CapacityDemand,
    Priority,
    has_all_capabilities,
    missing_capabilities,
    normalize_capabilities,
    parse_priority,
)
from orion.domain.constraints import (
    CONSTRAINT_CATALOGUE,
    CONSTRAINT_INDEX,
    ConstraintSpec,
    ConstraintType,
    FLOW_INCOMPATIBLE,
    SolverSupport,
    active_constraints,
)
from orion.domain.entities import (
    ObjectiveWeights,
    Resource,
    ResourceKind,
    ResourceStatus,
    Scenario,
    ScenarioConstraints,
    Task,
    TaskStatus,
    make_task,
)
from orion.domain.errors import (
    DuplicateEntityError,
    InfeasibleScenarioError,
    OrionError,
    TravelGraphError,
    UnknownEntityError,
    ValidationError,
)
from orion.domain.events import (
    DISRUPTION_DESCRIPTIONS,
    Disruption,
    DisruptionImpact,
    DisruptionSeverity,
    DisruptionType,
    NewTaskSpec,
    analyse_impact,
)
from orion.domain.ids import DEPOT_ID, format_id, is_valid_id, validate_id
from orion.domain.plans import (
    ALL_SOLVERS,
    SOLVER_DESCRIPTIONS,
    Assignment,
    ObjectiveBreakdown,
    Plan,
    ResourceUtilization,
    SolverName,
    SolverRun,
    SolverStatus,
    USABLE_STATUSES,
    Violation,
    ViolationKind,
    has_hard_violation,
    validate_plan,
)
from orion.domain.time_model import (
    MAX_HORIZON_MINUTES,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    Interval,
    hhmm,
    merge_intervals,
    parse_hhmm,
    subtract_intervals,
)
from orion.domain.travel import TravelMatrix

__all__ = [
    # capacity / priority
    "CapacityDemand",
    "Priority",
    "parse_priority",
    "normalize_capabilities",
    "missing_capabilities",
    "has_all_capabilities",
    # entities
    "Task",
    "TaskStatus",
    "Resource",
    "ResourceKind",
    "ResourceStatus",
    "Scenario",
    "ScenarioConstraints",
    "ObjectiveWeights",
    "make_task",
    # plans
    "Plan",
    "Assignment",
    "Violation",
    "ViolationKind",
    "ObjectiveBreakdown",
    "ResourceUtilization",
    "SolverRun",
    "SolverStatus",
    "USABLE_STATUSES",
    "SolverName",
    "ALL_SOLVERS",
    "SOLVER_DESCRIPTIONS",
    "validate_plan",
    "has_hard_violation",
    # constraints
    "CONSTRAINT_CATALOGUE",
    "CONSTRAINT_INDEX",
    "ConstraintSpec",
    "ConstraintType",
    "SolverSupport",
    "FLOW_INCOMPATIBLE",
    "active_constraints",
    # events
    "Disruption",
    "DisruptionType",
    "DisruptionSeverity",
    "DisruptionImpact",
    "DISRUPTION_DESCRIPTIONS",
    "NewTaskSpec",
    "analyse_impact",
    # infra
    "Interval",
    "hhmm",
    "parse_hhmm",
    "merge_intervals",
    "subtract_intervals",
    "MINUTES_PER_HOUR",
    "MINUTES_PER_DAY",
    "MAX_HORIZON_MINUTES",
    "TravelMatrix",
    "DEPOT_ID",
    "format_id",
    "validate_id",
    "is_valid_id",
    "OrionError",
    "ValidationError",
    "DuplicateEntityError",
    "UnknownEntityError",
    "InfeasibleScenarioError",
    "TravelGraphError",
]
