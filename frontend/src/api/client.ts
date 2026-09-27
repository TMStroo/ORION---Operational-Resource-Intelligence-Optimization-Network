/**
 * Typed client for the ORION backend.
 *
 * Two rules, both deliberate:
 *
 * 1. No mock data. There is no fallback object anywhere in this file. If a
 *    request fails the caller gets an ApiError carrying the server's own
 *    message, never a plausible-looking placeholder.
 * 2. The shapes below mirror `orion.api.schemas` field for field. They were
 *    transcribed from the real Pydantic models rather than guessed, so a rename
 *    on the backend surfaces here as a TypeScript error instead of silently
 *    rendering `undefined`.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (cause) {
    throw new ApiError(
      `The backend at ${BASE} is unreachable. Is the server running? (${(cause as Error).message})`,
      0,
    );
  }
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(`Response was not JSON: ${text.slice(0, 200)}`, response.status);
    }
  }
  if (!response.ok) {
    const detail = (payload as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => (d as { msg?: string }).msg ?? JSON.stringify(d)).join("; ")
          : `HTTP ${response.status}`;
    throw new ApiError(message, response.status, detail);
  }
  return payload as T;
}

const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });

// ------------------------------------------------------------------ domain

export interface ValidationIssue {
  kind: string;
  detail: string;
  task_id: string | null;
  resource_id: string | null;
  severity: number;
  amount: number;
}

export interface TaskView {
  id: string;
  location: string;
  release_time: number;
  deadline: number;
  duration: number;
  priority: string;
  required_capabilities: string[];
  dependencies: string[];
  status: string;
}

export interface ResourceView {
  id: string;
  kind: string;
  capabilities: string[];
  home_location: string;
  shift_start: number;
  shift_end: number;
  status: string;
  max_work_minutes: number;
  capacity: Record<string, number> | null;
}

export interface ScenarioSummary {
  id: string;
  name: string;
  task_count: number;
  resource_count: number;
  horizon_end: number;
  source: string;
  created_at: string;
}

export interface ScenarioDetail extends ScenarioSummary {
  description: string;
  tags: string[];
  objective_weights: Record<string, number>;
  tasks: TaskView[];
  resources: ResourceView[];
  horizon: { start: number; end: number };
  depot: string | null;
}

export interface ValidationResponse {
  valid: boolean;
  issues: ValidationIssue[];
  task_count: number;
  resource_count: number;
}

export interface AssignmentView {
  task_id: string;
  resource_id: string;
  start: number;
  end: number;
  location: string;
  sequence_index: number;
  travel_before: number;
  travel_distance_km: number;
  priority: string;
  lateness: number;
}

export interface UtilizationView {
  resource_id: string;
  kind: string;
  assigned_tasks: number;
  worked_minutes: number;
  utilization: number;
  travel_km: number;
}

export interface SolverRunView {
  solver: string;
  status: string;
  objective: number;
  runtime_s: number;
  optimality_gap: number | null;
  feasible: boolean;
  notes: string;
}

export interface ObjectiveView {
  total: number;
  completion: number;
  priority: number;
  lateness_penalty: number;
  travel_penalty: number;
  cost_penalty: number;
  dispatch_penalty: number;
  overload_penalty: number;
  idle_move_penalty: number;
  violation_penalty: number;
}

export interface PlanSummary {
  id: string;
  scenario_id: string;
  label: string;
  status: string;
  solver: string;
  objective: number;
  completion: number;
  late_tasks: number;
  tasks_assigned: number;
  tasks_total: number;
  runtime_s: number;
  infeasible: boolean;
  created_at: string;
}

export interface PlanDetail extends PlanSummary {
  assignments: AssignmentView[];
  violations: ValidationIssue[];
  utilization: UtilizationView[];
  solver_runs: SolverRunView[];
  objective_breakdown: ObjectiveView | null;
  total_travel_km: number;
  total_travel_minutes: number;
  total_operating_cost: number;
  service_level: number;
  strategy: string | null;
  parent_plan_id: string | null;
  disruption_id: string | null;
  metadata: Record<string, unknown> | null;
}

export interface PlanListResponse {
  items: PlanSummary[];
  total: number;
}

export interface SimEventView {
  ordinal: number;
  time: number;
  type: string;
  subject_id: string | null;
  resource_id: string | null;
  detail: string;
  payload: Record<string, unknown> | null;
}

export interface SimulateResponse {
  run_id: string;
  plan_id: string;
  plan_status: string;
  /** NOT_APPLICABLE when the plan assigns nothing; there is then no trace. */
  status: string;
  start_time: number;
  end_time: number;
  completed_tasks: number;
  late_tasks: number;
  failed_tasks: number;
  replans_performed: number;
  runtime_s: number;
  events: SimEventView[];
  event_count: number;
}

export interface DisruptionView {
  id: string;
  type: string;
  severity: string;
  target_id: string | null;
  timestamp: number;
  magnitude: number | null;
  duration: number | null;
  description: string;
  affected_tasks: string[];
  affected_resources: string[];
  at_risk_deadlines: string[];
}

export interface DisruptResponse {
  disruption: DisruptionView;
  recorded: boolean;
  after_scenario_id: string | null;
}

export interface RepairView {
  strategy: string;
  seconds: number;
  candidates_considered: number;
  recovered_tasks: number;
  unrecovered_tasks: number;
  produced_plan: boolean;
  reason: string;
}

export interface ChurnView {
  changed_assignments: number;
  rescheduled_tasks: number;
  total_before: number;
  total_after: number;
  tolerance_minutes: number;
  travel_delta_km: number;
  cost_delta: number;
  churn_ratio: number;
}

export interface MetricDeltaView {
  name: string;
  label: string;
  before: number;
  after: number;
  change: number;
  unit: string;
  higher_is_better: boolean;
}

export interface ComparisonView {
  kind: string;
  baseline_plan_id: string;
  candidate_plan_id: string;
  churn: ChurnView;
  deltas: MetricDeltaView[];
  newly_assigned: string[];
  removed_assignments: string[];
  delayed: string[];
  newly_violated: string[];
  newly_resolved: string[];
}

export interface ReplanResponse {
  disruption: DisruptionView;
  impact: DisruptionView;
  repair: RepairView;
  repair_plan: PlanSummary | null;
  full_plan: PlanSummary | null;
  repair_seconds: number;
  full_seconds: number;
  full_status: string;
  full_reason: string;
  solver: string;
  time_budget_s: number;
  repair_comparison: ComparisonView | null;
  full_comparison: ComparisonView | null;
  repair_preserved_ratio: number;
  recovery_pct: number;
}

export interface WhatIfResponse {
  operator: string;
  question: string;
  parameter: number | string;
  modification: string;
  status: string;
  seconds: number;
  baseline_plan_id: string;
  scenario_plan_id: string;
  comparison: ComparisonView;
}

export interface PlanComparisonResponse {
  plan_id: string;
  items: ComparisonView[];
  baseline_plan_id: string;
  metrics: Record<string, Record<string, number>>;
}

export interface ExperimentSummary {
  id: string;
  kind: string;
  status: string;
  commit: string;
  seed: number;
  row_count: number;
  created_at: string;
  superseded_by: string | null;
  supersede_reason: string;
}

export interface ExperimentListResponse {
  items: ExperimentSummary[];
  total: number;
  live: number;
  superseded: number;
}

export interface ExperimentDetail extends ExperimentSummary {
  manifest: Record<string, unknown>;
  summary: Record<string, unknown>;
  metrics: Record<string, unknown>;
  rows: Record<string, string>[];
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  database: string;
  store_url: string;
  rows: Record<string, number>;
  live_experiments: number;
  superseded_experiments: number;
  solvers: string[];
}

// --------------------------------------------------------------- constants

/**
 * `orion.optimization.registry` solver names. The label says what each method
 * actually is; MIN_COST_FLOW is a transportation relaxation and is never
 * presented as an exact solve.
 */
export const SOLVERS = [
  { id: "HEURISTIC", label: "HEURISTIC — constructive, fast, approximate" },
  { id: "LOCAL_SEARCH", label: "LOCAL_SEARCH / ALNS — improves a plan in place" },
  { id: "CP_SAT", label: "CP-SAT — exact within the time budget" },
  { id: "MILP", label: "MILP — exact where the full model applies" },
  { id: "MIN_COST_FLOW", label: "MIN_COST_FLOW — transportation relaxation" },
] as const;

export const SOLVER_STATUSES = [
  "OPTIMAL",
  "FEASIBLE",
  "TIME_LIMIT",
  "INFEASIBLE",
  "ERROR",
  "NOT_APPLICABLE",
  "RELAXATION_OPTIMAL",
] as const;

export type SolverStatus = (typeof SOLVER_STATUSES)[number];

/** `orion.planning.whatif.WHAT_IF_OPERATORS` — the six, verbatim. */
export const WHAT_IF_OPERATORS = [
  "availability_drop",
  "demand_increase",
  "deadline_tighten",
  "resource_outage",
  "travel_increase",
  "capacity_reduction",
] as const;

export type WhatIfOperator = (typeof WHAT_IF_OPERATORS)[number];

/** What each operator's `parameter` means, from `WHAT_IF_DESCRIPTIONS`. */
export const WHAT_IF_PARAMETERS: Record<
  WhatIfOperator,
  { hint: string; default: number | string }
> = {
  // These defaults come from the engine's own validation bounds, not from
  // intuition. `demand_increase` is a *fraction of the existing task count*
  // (0.3 = add 30% more tasks), not a multiplier - a 1.3 here would try to add
  // 130% more tasks and the operator would reject it. `resource_outage` needs a
  // real resource id, so it has no valid default and is filled in from the
  // loaded scenario.
  availability_drop: { hint: "Fraction of availability lost, e.e. 0.2 for a 20% drop.", default: 0.2 },
  demand_increase: { hint: "Fraction more tasks, e.g. 0.3 for +30% demand.", default: 0.3 },
  deadline_tighten: { hint: "Fraction of deadline slack removed, e.g. 0.15.", default: 0.15 },
  resource_outage: { hint: "A named resource becomes unavailable entirely.", default: "" },
  travel_increase: { hint: "Multiplier on all travel times, e.g. 1.25.", default: 1.25 },
  capacity_reduction: { hint: "Capacity falls to this fraction, e.g. 0.8 for -20%.", default: 0.8 },
};

/**
 * True only for a status that claims proven optimality of the actual problem.
 *
 * RELAXATION_OPTIMAL is deliberately excluded. A relaxation being solved
 * optimally says nothing about the real problem, and any code path that treats
 * the two alike would report a bound as a solution.
 */
export function isProvenOptimal(status: string): boolean {
  return status === "OPTIMAL";
}

/** `orion.simulation.disruptions.DISRUPTION_KINDS` — the five the engine models. */
export const DISRUPTION_TYPES = [
  "VEHICLE_FAILURE",
  "RESOURCE_UNAVAILABLE",
  "MAINTENANCE",
  "DEMAND_SURGE",
  "TRAVEL_INCREASE",
] as const;

export type DisruptionType = (typeof DISRUPTION_TYPES)[number];

// --------------------------------------------------------------------- api

export const api = {
  health: () => get<HealthResponse>("/health"),

  listScenarios: (limit = 50) =>
    get<{ items: ScenarioSummary[]; total: number }>(`/scenarios?limit=${limit}`),

  createScenario: (body: {
    name: string;
    task_count: number;
    resource_count: number;
    seed: number;
    difficulty: "easy" | "medium" | "hard";
  }) => post<ScenarioDetail>("/scenarios", body),

  getScenario: (id: string) => get<ScenarioDetail>(`/scenarios/${encodeURIComponent(id)}`),

  validateScenario: (id: string) =>
    post<ValidationResponse>(`/scenarios/${encodeURIComponent(id)}/validate`),

  optimize: (
    scenarioId: string,
    body: { solver: string; time_budget_s: number; seed?: number; label?: string },
  ) => post<PlanDetail>(`/scenarios/${encodeURIComponent(scenarioId)}/optimize`, body),

  listPlans: (scenarioId: string) =>
    get<PlanListResponse>(`/plans?scenario_id=${encodeURIComponent(scenarioId)}`),

  getPlan: (planId: string) => get<PlanDetail>(`/plans/${encodeURIComponent(planId)}`),

  /**
   * `plan_id` is explicit. The backend otherwise falls back to the newest plan,
   * which under a short time budget is a run that scheduled nothing.
   */
  simulate: (
    scenarioId: string,
    body: { plan_id: string; seed?: number; until?: number; max_events?: number },
  ) => post<SimulateResponse>(`/scenarios/${encodeURIComponent(scenarioId)}/simulate`, body),

  disrupt: (scenarioId: string, body: Record<string, unknown>) =>
    post<DisruptResponse>(`/scenarios/${encodeURIComponent(scenarioId)}/disrupt`, body),

  replan: (scenarioId: string, body: Record<string, unknown>) =>
    post<ReplanResponse>(`/scenarios/${encodeURIComponent(scenarioId)}/replan`, body),

  whatIf: (scenarioId: string, body: { operator: string; parameter: number | string; seed?: number }) =>
    post<WhatIfResponse>(`/scenarios/${encodeURIComponent(scenarioId)}/what-if`, body),

  planComparison: (planId: string) =>
    get<PlanComparisonResponse>(`/plans/${encodeURIComponent(planId)}/comparison`),

  listExperiments: (liveOnly = false, kind?: string) => {
    const q = new URLSearchParams({ live_only: String(liveOnly) });
    if (kind) q.set("kind", kind);
    return get<ExperimentListResponse>(`/experiments?${q.toString()}`);
  },

  getExperiment: (id: string) => get<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`),
};
