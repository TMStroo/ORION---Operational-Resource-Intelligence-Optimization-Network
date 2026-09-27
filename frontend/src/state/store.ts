/**
 * Application state.
 *
 * One reducer holds the scenario being worked on, the plan the user is looking
 * at, the simulation trace, the disruption, and the replan outcome. Keeping the
 * active plan id separate from the plan object is deliberate: the backend
 * mints the id it will serve from, and every follow-up call has to cite it.
 */

import { createContext, useContext } from "react";
import type {
  DisruptResponse,
  ExperimentSummary,
  HealthResponse,
  PlanDetail,
  PlanSummary,
  ReplanResponse,
  ScenarioDetail,
  ScenarioSummary,
  SimulateResponse,
  ValidationResponse,
} from "../api/client";

export interface AppState {
  health: HealthResponse | null;
  scenarios: ScenarioSummary[];
  scenario: ScenarioDetail | null;
  validation: ValidationResponse | null;
  plans: PlanSummary[];
  activePlanId: string | null;
  plan: PlanDetail | null;
  simulation: SimulateResponse | null;
  disruption: DisruptResponse | null;
  replan: ReplanResponse | null;
  experiments: ExperimentSummary[];
  busy: string | null;
  error: string | null;
}

export type Action =
  | { type: "health"; payload: HealthResponse | null }
  | { type: "scenarios"; payload: ScenarioSummary[] }
  | { type: "openScenario"; id: string; scenario: ScenarioDetail }
  | { type: "validation"; payload: ValidationResponse | null }
  | { type: "plansRefreshed"; payload: PlanSummary[] }
  | { type: "planLoaded"; plan: PlanDetail; planId: string }
  | { type: "planCreated"; plan: PlanDetail; planId: string }
  | { type: "simulation"; payload: SimulateResponse | null }
  | { type: "disruption"; payload: DisruptResponse | null }
  | { type: "replan"; payload: ReplanResponse | null }
  | { type: "experiments"; payload: ExperimentSummary[] }
  | { type: "busy"; label: string | null }
  | { type: "error"; message: string | null }
  | { type: "reset" };

export const initialState: AppState = {
  health: null,
  scenarios: [],
  scenario: null,
  validation: null,
  plans: [],
  activePlanId: null,
  plan: null,
  simulation: null,
  disruption: null,
  replan: null,
  experiments: [],
  busy: null,
  error: null,
};

/**
 * `scenarioChanged` invalidates everything downstream of a scenario. A plan,
 * trace or disruption left over from the previous scenario would be shown
 * against the new one's numbers, which is the exact class of bug the backend
 * test for an empty plan was written to catch.
 */
const scenarioChanged = (state: AppState, scenario: ScenarioDetail): AppState => ({
  ...state,
  scenario,
  plans: [],
  activePlanId: null,
  plan: null,
  simulation: null,
  disruption: null,
  replan: null,
  validation: null,
});

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "health":
      return { ...state, health: action.payload };
    case "scenarios":
      return { ...state, scenarios: action.payload };
    case "openScenario":
      return scenarioChanged(state, action.scenario);
    case "validation":
      return { ...state, validation: action.payload };
    case "plansRefreshed": {
      // Keep the open plan only if it survived the refresh.
      const stillThere = state.activePlanId && action.payload.some((p) => p.id === state.activePlanId);
      return {
        ...state,
        plans: action.payload,
        plan: stillThere ? state.plan : null,
        activePlanId: stillThere ? state.activePlanId : null,
      };
    }
    case "planLoaded":
      return { ...state, plan: action.plan, activePlanId: action.planId, error: null };
    case "planCreated":
      return {
        ...state,
        plan: action.plan,
        activePlanId: action.planId,
        // A new plan invalidates a trace and a disruption measured against the old one.
        simulation: null,
        disruption: null,
        replan: null,
        error: null,
      };
    case "simulation":
      return { ...state, simulation: action.payload, error: null };
    case "disruption":
      return { ...state, disruption: action.payload, replan: null, error: null };
    case "replan":
      return { ...state, replan: action.payload, error: null };
    case "experiments":
      return { ...state, experiments: action.payload };
    case "busy":
      return { ...state, busy: action.label };
    case "error":
      return { ...state, error: action.message, busy: null };
    case "reset":
      return initialState;
    default:
      return state;
  }
}

export interface Store {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

export const StoreContext = createContext<Store | null>(null);

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore must be used inside <StoreProvider>");
  return store;
}
