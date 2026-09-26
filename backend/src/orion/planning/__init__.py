"""Planning layer: planner, replanner, plan comparison and what-if analysis."""

from __future__ import annotations

from orion.planning.comparison import (
    MetricDelta,
    PlanComparison,
    RecoveryReport,
    build_recovery_report,
    compare_plans,
    degraded_plan_from,
)
from orion.planning.planner import (
    AssignmentExplanation,
    CheckResult,
    DEFAULT_TIME_BUDGETS,
    Planner,
    PlanResult,
)
from orion.planning.replanner import (
    ChurnReport,
    LocalRepairer,
    REPAIR_STRATEGIES,
    ReplanOutcome,
    Replanner,
    RepairOutcome,
    compute_churn,
)
from orion.planning.whatif import (
    WHAT_IF_DESCRIPTIONS,
    WHAT_IF_OPERATORS,
    WhatIfEngine,
    WhatIfResult,
)

__all__ = [
    "Planner",
    "PlanResult",
    "AssignmentExplanation",
    "CheckResult",
    "DEFAULT_TIME_BUDGETS",
    "Replanner",
    "ReplanOutcome",
    "RepairOutcome",
    "LocalRepairer",
    "ChurnReport",
    "compute_churn",
    "REPAIR_STRATEGIES",
    "PlanComparison",
    "MetricDelta",
    "RecoveryReport",
    "compare_plans",
    "build_recovery_report",
    "degraded_plan_from",
    "WhatIfEngine",
    "WhatIfResult",
    "WHAT_IF_OPERATORS",
    "WHAT_IF_DESCRIPTIONS",
]
