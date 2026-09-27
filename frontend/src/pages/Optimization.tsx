/**
 * Optimization: run solvers against the open scenario and read what came back.
 *
 * The status vocabulary here is the engine's own. A relaxation is never shown
 * as an exact solve, TIME_LIMIT is not dressed up as a failure, and INFEASIBLE
 * is reported as an answer rather than an error.
 */

import React, { useState } from "react";
import { api, SOLVERS, type PlanDetail } from "../api/client";
import { ErrorBanner, Panel, Spinner, StatusBadge, Table, num, pct } from "../components/ui";
import type { AppState, Action } from "../state/store";

/** What each status means, shown on hover and in the legend. */
const STATUS_MEANING: Record<string, string> = {
  OPTIMAL: "Proven optimal for this scenario and budget.",
  FEASIBLE: "A valid plan was found; optimality was not proven.",
  TIME_LIMIT: "The budget expired with a valid partial plan, or with none.",
  INFEASIBLE: "No plan satisfies the constraints as given.",
  ERROR: "The solver raised. The failure is reported, not hidden.",
  NOT_APPLICABLE: "The method does not apply to this scenario.",
  RELAXATION_OPTIMAL: "Optimal for a relaxation, so a bound on the true problem, not a solution to it.",
};

export function Optimization({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const { scenario, plan, plans, validation } = state;
  const [budgets, setBudgets] = useState<Record<string, number>>(
    Object.fromEntries(SOLVERS.map((s) => [s.id, 5])),
  );
  const [running, setRunning] = useState<string | null>(null);
  const [compare, setCompare] = useState<PlanDetail[]>([]);

  if (!scenario) {
    return (
      <div className="page">
        <h1>Optimization</h1>
        <Panel title="No scenario open">
          <p>Open a scenario to run solvers against it.</p>
        </Panel>
      </div>
    );
  }

  const run = async (solver: string) => {
    setRunning(solver);
    dispatch({ type: "busy", label: `${solver} at ${budgets[solver] ?? 5}s` });
    try {
      const result = await api.optimize(scenario.id, {
        solver,
        time_budget_s: budgets[solver] ?? 5,
        seed: 0,
        label: `${solver} @ ${budgets[solver] ?? 5}s`,
      });
      dispatch({ type: "planCreated", plan: result, planId: result.id });
      setCompare((prev) => [...prev.filter((p) => p.solver !== solver), result]);
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setRunning(null);
      dispatch({ type: "busy", label: null });
    }
  };

  const runAll = async () => {
    for (const s of SOLVERS) await run(s.id);
  };

  return (
    <div className="page">
      <h1>Optimization</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title="Run a solver"
        subtitle="Each solver optimises the same explicit objective, through ORION's common rescoring layer."
        actions={
          <button type="button" onClick={() => void runAll()} disabled={running !== null}>
            {running === "__all__" ? "Running all…" : "Run all five"}
          </button>
        }
      >
        <div className="solver-grid">
          {SOLVERS.map((s) => (
            <div key={s.id} className="solver-card">
              <h4>{s.id}</h4>
              <p className="hint">{s.label.split("— ")[1] ?? s.label}</p>
              <label className="inline-label">
                budget (s)
                <input
                  type="number"
                  min={0.001}
                  step={0.5}
                  value={budgets[s.id] ?? 5}
                  onChange={(e) =>
                    setBudgets((prev) => ({ ...prev, [s.id]: Number(e.target.value) }))
                  }
                />
              </label>
              <div className="row-actions">
                <button
                  type="button"
                  onClick={() => void run(s.id)}
                  disabled={running !== null || (validation ? !validation.valid : false)}
                >
                  {running === s.id ? <Spinner label="solving" /> : "Solve"}
                </button>
              </div>
            </div>
          ))}
        </div>
        {validation && !validation.valid && (
          <p className="note">
            Solving is blocked: the backend reports {validation.issues.length} validation issue(s)
            on this scenario.
          </p>
        )}
      </Panel>

      <Panel title="Status vocabulary" subtitle="The only statuses the engine returns, and what each one claims.">
        <div className="legend">
          {Object.entries(STATUS_MEANING).map(([status, meaning]) => (
            <div key={status} className="legend-row">
              <StatusBadge status={status} />
              <span className="hint">{meaning}</span>
            </div>
          ))}
        </div>
      </Panel>

      {compare.length > 0 && (
        <Panel
          title="Solvers on this scenario"
          subtitle="Same objective, same rescoring, different budgets. A lower objective is only better among rows that actually produced a plan."
        >
          <div className="scroll">
            <Table
              rows={compare}
              rowKey={(p) => p.solver}
              empty="Nothing run yet."
              columns={[
                { key: "solver", label: "Solver" },
                { key: "status", label: "Status", render: (p) => <StatusBadge status={p.status} /> },
                {
                  key: "objective",
                  label: "Objective",
                  render: (p) => (p.infeasible ? "—" : num(p.objective, 1)),
                  numeric: true,
                },
                {
                  key: "tasks",
                  label: "Tasks",
                  render: (p) => `${p.tasks_assigned}/${p.tasks_total}`,
                  numeric: true,
                },
                { key: "late", label: "Late", render: (p) => String(p.late_tasks), numeric: true },
                {
                  key: "travel",
                  label: "Travel km",
                  render: (p) => num(p.total_travel_km, 1),
                  numeric: true,
                },
                {
                  key: "cost",
                  label: "Cost",
                  render: (p) => num(p.total_operating_cost),
                  numeric: true,
                },
                {
                  key: "runtime",
                  label: "Runtime",
                  render: (p) => `${num(p.runtime_s, 3)} s`,
                  numeric: true,
                },
                {
                  key: "gap",
                  label: "Gap",
                  render: (p) => {
                    const run = p.solver_runs.find((r) => r.optimality_gap !== null);
                    return run?.optimality_gap !== null && run
                      ? run.optimality_gap!.toExponential(2)
                      : "—";
                  },
                  numeric: true,
                },
                { key: "completion", label: "Completion", render: (p) => pct(p.completion) },
              ]}
            />
          </div>
          <p className="note">
            MIN_COST_FLOW solves a transportation relaxation: it has no route order and no time
            windows, so it can be labelled RELAXATION_OPTIMAL and still not be a feasible schedule.
            That is a property of the method, not a defect in its run.
          </p>
        </Panel>
      )}

      {plan && plan.solver_runs.length > 0 && (
        <Panel title={`Solver runs on plan ${plan.id}`}>
          <Table
            rows={plan.solver_runs}
            rowKey={(r, i) => `${r.solver}-${i}`}
            empty="No runs recorded."
            columns={[
              { key: "solver", label: "Solver" },
              { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
              {
                key: "objective",
                label: "Objective",
                render: (r) => (r.feasible ? num(r.objective, 1) : "—"),
                numeric: true,
              },
              {
                key: "feasible",
                label: "Feasible",
                render: (r) => (r.feasible ? "yes" : "no"),
              },
              { key: "runtime_s", label: "Runtime", render: (r) => `${num(r.runtime_s, 3)} s`, numeric: true },
              {
                key: "optimality_gap",
                label: "Gap",
                render: (r) => (r.optimality_gap === null ? "—" : r.optimality_gap.toExponential(2)),
                numeric: true,
              },
              { key: "notes", label: "Notes" },
            ]}
          />
        </Panel>
      )}

      {plans.length > 0 && (
        <Panel title={`Stored plans (${plans.length})`}>
          <div className="scroll">
            <Table
              rows={plans}
              rowKey={(p) => p.id}
              highlight={(p) => p.id === state.activePlanId}
              onRowClick={async (p) => {
                dispatch({ type: "busy", label: "loading plan" });
                try {
                  const loaded = await api.getPlan(p.id);
                  dispatch({ type: "planLoaded", plan: loaded, planId: loaded.id });
                } catch (e) {
                  dispatch({ type: "error", message: (e as Error).message });
                } finally {
                  dispatch({ type: "busy", label: null });
                }
              }}
              empty="No plans."
              columns={[
                { key: "id", label: "Plan" },
                { key: "label", label: "Label" },
                { key: "solver", label: "Solver" },
                { key: "status", label: "Status", render: (p) => <StatusBadge status={p.status} /> },
                { key: "objective", label: "Objective", render: (p) => num(p.objective, 1), numeric: true },
                {
                  key: "tasks",
                  label: "Tasks",
                  render: (p) => `${p.tasks_assigned}/${p.tasks_total}`,
                  numeric: true,
                },
                {
                  key: "runtime_s",
                  label: "Runtime",
                  render: (p) => `${num(p.runtime_s, 3)} s`,
                  numeric: true,
                },
              ]}
            />
          </div>
        </Panel>
      )}
    </div>
  );
}
