"""Pydantic request and response models for the ORION API.

Deliberately separate from the domain entities. The domain is a costed
scheduling graph with frozen dataclasses and enum-valued statuses; the API is a
wire format. They change for different reasons - a new solver status is not a
breaking API change, and adding a field to a plan's metadata is not a reason to
touch the domain.

One convention is load-bearing: every status string is passed through verbatim.
The API does not translate ERROR into a 200 with an empty body, and it does not
flatten INFEASIBLE and TIME_LIMIT into "failed". A client that cannot tell those
apart cannot build an honest comparison view.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base for every schema: forbid unknown fields rather than drop them."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ health


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    store_url: str
    rows: dict[str, int]
    live_experiments: int
    superseded_experiments: int
    solvers: list[str]


# --------------------------------------------------------------- scenarios


class ScenarioSummary(ApiModel):
    id: str
    name: str
    description: str = ""
    task_count: int
    resource_count: int
    horizon_end: int
    source: str = "generated"
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""


class ScenarioListResponse(ApiModel):
    items: list[ScenarioSummary]
    total: int
    limit: int
    offset: int


class CreateScenarioRequest(ApiModel):
    """Generate a scenario.

    Size parameters are explicit rather than a free-form spec so that a request
    is reproducible: the same request with the same seed yields the same
    scenario, which is what makes an API-level determinism test meaningful.
    """

    name: str = "api-scenario"
    task_count: int = Field(default=25, ge=1, le=1000)
    resource_count: int = Field(default=8, ge=1, le=200)
    seed: int = 0
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    description: str = ""


class TaskView(ApiModel):
    id: str
    location: str
    release_time: int
    deadline: int
    duration: int
    priority: str
    required_capabilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str


class ResourceView(ApiModel):
    id: str
    kind: str
    capabilities: list[str] = Field(default_factory=list)
    home_location: str
    shift_start: int
    shift_end: int
    status: str
    max_work_minutes: int
    capacity: dict[str, float] = Field(default_factory=dict)
    # The blocks of the shift a resource has lost. A disruption that removes
    # working time does not change `status` - the resource is still a vehicle,
    # it is simply off the road for part of the day. Without this the client
    # cannot tell an intact scenario from a disrupted one, because every
    # resource still reads AVAILABLE.
    unavailable: list[dict[str, int]] = Field(default_factory=list)


class ScenarioDetail(ScenarioSummary):
    objective_weights: dict[str, float] = Field(default_factory=dict)
    tasks: list[TaskView] = Field(default_factory=list)
    resources: list[ResourceView] = Field(default_factory=list)
    horizon: dict[str, int] = Field(default_factory=dict)
    depot: str = ""


class ValidationIssue(ApiModel):
    kind: str
    detail: str
    task_id: str | None = None
    resource_id: str | None = None
    severity: int = 0
    amount: float = 0.0


class ValidationResponse(ApiModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    task_count: int
    resource_count: int


# -------------------------------------------------------------------- plans


class AssignmentView(ApiModel):
    task_id: str
    resource_id: str
    start: int
    end: int
    location: str
    sequence_index: int
    travel_before: int
    travel_distance_km: float
    priority: str
    lateness: int


class SolverRunView(ApiModel):
    solver: str
    status: str
    objective: float
    runtime_s: float
    optimality_gap: float | None = None
    feasible: bool = False
    notes: str = ""


class ObjectiveView(ApiModel):
    total: float
    completion: float
    priority: float
    lateness_penalty: float
    travel_penalty: float
    cost_penalty: float
    dispatch_penalty: float
    overload_penalty: float
    idle_move_penalty: float
    violation_penalty: float


class UtilizationView(ApiModel):
    resource_id: str
    kind: str
    assigned_tasks: int
    worked_minutes: int
    utilization: float
    travel_km: float = 0.0


class PlanSummary(ApiModel):
    id: str
    scenario_id: str
    label: str = ""
    status: str
    solver: str
    objective: float
    completion: float
    late_tasks: int
    tasks_assigned: int
    tasks_total: int
    runtime_s: float
    infeasible: bool = False
    created_at: str = ""


class PlanDetail(PlanSummary):
    assignments: list[AssignmentView] = Field(default_factory=list)
    violations: list[ValidationIssue] = Field(default_factory=list)
    utilization: list[UtilizationView] = Field(default_factory=list)
    solver_runs: list[SolverRunView] = Field(default_factory=list)
    objective_breakdown: ObjectiveView | None = None
    total_travel_km: float = 0.0
    total_travel_minutes: int = 0
    total_operating_cost: float = 0.0
    service_level: float = 0.0
    strategy: str = ""
    parent_plan_id: str | None = None
    disruption_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanListResponse(ApiModel):
    items: list[PlanSummary]
    total: int


# ------------------------------------------------------------------ requests


class OptimizeRequest(ApiModel):
    """Solve a stored scenario.

    `solver` and `time_budget_s` are passed straight through to the planner.
    A budget the solver cannot meet is reported as TIME_LIMIT, not converted
    into an error.
    """

    solver: str = "HEURISTIC"
    time_budget_s: float | None = None
    seed: int | None = None
    label: str = ""
    explain: bool = False


class SimulateRequest(ApiModel):
    until: int | None = None
    max_events: int = 1_000_000
    respect_failures: bool = True
    seed: int = 0
    plan_id: str | None = Field(
        default=None,
        description="Simulate this plan. Defaults to the scenario's newest. "
        "Pass it explicitly: the newest plan may be a 1 ms-budget CP-SAT run "
        "that assigned nothing, and simulating that reports an empty day.",
    )


class SimEventView(ApiModel):
    ordinal: int
    time: int
    type: str
    subject_id: str | None = None
    resource_id: str | None = None
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class SimulateResponse(ApiModel):
    run_id: str
    plan_id: str = ""
    plan_status: str = ""
    status: str
    start_time: int
    end_time: int
    completed_tasks: int
    late_tasks: int
    failed_tasks: int
    replans_performed: int
    runtime_s: float
    events: list[SimEventView] = Field(default_factory=list)
    event_count: int = 0


class DisruptRequest(ApiModel):
    """Trigger a disruption and report its impact."""

    type: str = Field(
        description="One of DISRUPTION_KINDS, e.g. VEHICLE_FAILURE, RESOURCE_UNAVAILABLE, "
        "CAPACITY_REDUCTION, DEMAND_SURGE, DEADLINE_CHANGE, TRAVEL_INCREASE"
    )
    target_id: str | None = None
    magnitude: float = 0.3
    duration: int | None = None
    at_time: int | None = None
    seed: int = 0


class DisruptionView(ApiModel):
    id: str
    type: str
    severity: str
    target_id: str | None = None
    timestamp: int
    magnitude: float
    duration: int | None = None
    description: str
    affected_tasks: list[str] = Field(default_factory=list)
    affected_resources: list[str] = Field(default_factory=list)
    at_risk_deadlines: list[str] = Field(default_factory=list)


class DisruptResponse(ApiModel):
    disruption: DisruptionView
    recorded: bool = Field(
        description="False when this disruption id was already applied; "
        "a second application is refused, not repeated"
    )
    after_scenario_id: str = ""


class ReplanRequest(ApiModel):
    """Local repair, optionally followed by full re-optimization."""

    disruption_type: str = "VEHICLE_FAILURE"
    target_id: str | None = None
    magnitude: float = 0.3
    duration: int | None = None
    at_time: int | None = None
    seed: int = 0
    run_full: bool = True
    solver: str = "HEURISTIC"
    time_budget_s: float | None = None
    strategies: list[str] | None = None


class RepairView(ApiModel):
    strategy: str
    seconds: float
    candidates_considered: int
    recovered_tasks: list[str] = Field(default_factory=list)
    unrecovered_tasks: list[str] = Field(default_factory=list)
    produced_plan: bool = False
    reason: str = ""


class ChurnView(ApiModel):
    changed_assignments: int
    rescheduled_tasks: int
    total_before: int
    total_after: int
    tolerance_minutes: int
    travel_delta_km: float
    cost_delta: float
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    delayed: list[str] = Field(default_factory=list)
    churn_ratio: float = 0.0


class MetricDeltaView(ApiModel):
    name: str
    label: str
    before: float
    after: float
    change: float
    unit: str = ""
    higher_is_better: bool = True


class ComparisonView(ApiModel):
    kind: str
    baseline_plan_id: str
    candidate_plan_id: str
    churn: ChurnView
    deltas: list[MetricDeltaView] = Field(default_factory=list)
    newly_assigned: list[str] = Field(default_factory=list)
    removed_assignments: list[str] = Field(default_factory=list)
    delayed: list[str] = Field(default_factory=list)
    newly_violated: list[str] = Field(default_factory=list)
    newly_resolved: list[str] = Field(default_factory=list)


class ReplanResponse(ApiModel):
    disruption: DisruptionView
    impact: DisruptionView
    repair: RepairView
    repair_plan: PlanSummary | None = None
    full_plan: PlanSummary | None = None
    repair_seconds: float = 0.0
    full_seconds: float = 0.0
    full_status: str = ""
    full_reason: str = ""
    solver: str = ""
    time_budget_s: float = 0.0
    repair_comparison: ComparisonView | None = None
    full_comparison: ComparisonView | None = None
    repair_preserved_ratio: float = Field(
        default=0.0,
        description="Fraction of the baseline plan's assignments that local repair "
        "left untouched. Preserved is not the same as good: see recovery_pct.",
    )
    recovery_pct: float = Field(
        default=0.0,
        description="Repair quality relative to full re-optimization, 0-1. "
        "Below 1.0 means full re-optimization produced a better plan.",
    )


class WhatIfRequest(ApiModel):
    operator: str = Field(
        description="One of capacity_reduction, demand_increase, deadline_tighten, "
        "resource_outage, travel_increase, availability_drop"
    )
    parameter: float | str = 0.8
    seed: int = 0


class WhatIfResponse(ApiModel):
    operator: str
    question: str
    parameter: float | str
    modification: str
    status: str
    seconds: float
    baseline_plan_id: str
    scenario_plan_id: str
    comparison: ComparisonView


# ------------------------------------------------------------------ reports


class PlanComparisonResponse(ApiModel):
    plan_id: str
    items: list[ComparisonView]
    baseline_plan_id: str = ""
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="metric -> {plan_id: value}, so a table can be built without "
        "walking the comparison list",
    )


class ExperimentSummary(ApiModel):
    id: str
    kind: str
    status: str
    commit: str = ""
    seed: int = 0
    row_count: int = 0
    created_at: str = ""
    superseded_by: str | None = None
    supersede_reason: str = ""


class ExperimentListResponse(ApiModel):
    items: list[ExperimentSummary]
    total: int
    live: int
    superseded: int


class ExperimentDetail(ExperimentSummary):
    manifest: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rows: list[dict[str, str]] = Field(default_factory=list)


class ErrorResponse(ApiModel):
    error: str
    detail: str
    context: dict[str, Any] = Field(default_factory=dict)

# ----------------------------------------------------------------- exports


class ExportArtifact(ApiModel):
    """One written export. ``content_type`` says how to interpret ``body``."""

    kind: str
    filename: str
    content_type: str
    body: str
    bytes: int
    provenance: dict[str, Any] = Field(default_factory=dict)


class ImportResult(ApiModel):
    """The result of reloading an export, with what survived reported."""

    kind: str
    entity_id: str
    status: str
    objective: float | None = None
    tasks_assigned: int | None = None
    tasks_total: int | None = None
    runtime_s: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    round_trip_identical: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
