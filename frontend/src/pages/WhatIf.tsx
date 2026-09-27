/**
 * What-if: the six operators, each a re-solve from the same baseline.
 *
 * The backend returns a comparison rather than two objectives, so the deltas
 * shown are the comparison's own metric deltas. A zero objective delta is
 * labelled "absorbed" and explained — it means this configuration had the
 * headroom for this perturbation, not that the constraint never binds.
 */

import React, { useState } from "react";
import {
  api,
  WHAT_IF_OPERATORS,
  WHAT_IF_PARAMETERS,
  type WhatIfOperator,
  type WhatIfResponse,
} from "../api/client";
import { ErrorBanner, Panel, Spinner, StatusBadge, Stat, Table, num, pct } from "../components/ui";
import type { AppState, Action } from "../state/store";

export function WhatIf({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const [results, setResults] = useState<Record<string, WhatIfResponse>>({});
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Partial<Record<WhatIfOperator, string>>>({});
  const { scenario, plan, activePlanId } = state;

  if (!scenario) {
    return (
      <div className="page">
        <h1>What-if</h1>
        <Panel title="No scenario open">
          <p>Open a scenario and solve it; a what-if question is asked of a plan.</p>
        </Panel>
      </div>
    );
  }

  if (!plan || !activePlanId) {
    return (
      <div className="page">
        <h1>What-if</h1>
        <Panel title="No baseline plan">
          <p>
            Solve the scenario on the Planning page first. Every operator re-solves from the plan
            that is currently loaded, so differences are attributable to the operator.
          </p>
        </Panel>
      </div>
    );
  }

  const parameterFor = (op: WhatIfOperator): number | string => {
    const override = overrides[op];
    if (override !== undefined && override !== "") return override;
    // `resource_outage` has no sensible default: the engine requires a real
    // resource id and rejects an empty one. Pick the busiest resource actually
    // present in the loaded scenario, so "run all six" is a complete run
    // rather than five operators and one error.
    if (op === "resource_outage") {
      return scenario?.resources?.[0]?.id ?? "";
    }
    return WHAT_IF_PARAMETERS[op].default;
  };

  const runOne = async (op: WhatIfOperator) => {
    setRunning(op);
    setError(null);
    dispatch({ type: "busy", label: `what-if: ${op}` });
    try {
      const res = await api.whatIf(scenario.id, {
        operator: op,
        parameter: parameterFor(op),
        seed: 0,
      });
      setResults((prev) => ({ ...prev, [op]: res }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(null);
      dispatch({ type: "busy", label: null });
    }
  };

  const runAll = async () => {
    setRunning("__all__");
    setError(null);
    dispatch({ type: "busy", label: "running all six what-if operators" });
    try {
      const next: Record<string, WhatIfResponse> = {};
      const failures: string[] = [];
      for (const op of WHAT_IF_OPERATORS) {
        // One operator failing must not abandon the other five: a reviewer
        // asking six questions should see five answers and one named failure,
        // not an empty page.
        try {
          next[op] = await api.whatIf(scenario.id, {
            operator: op,
            parameter: parameterFor(op),
            seed: 0,
          });
        } catch (e) {
          failures.push(`${op}: ${(e as Error).message}`);
        }
        setResults((prev) => ({ ...prev, ...next } as Record<string, WhatIfResponse>));
      }
      if (failures.length) setError(`Some operators failed. ${failures.join("; ")}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(null);
      dispatch({ type: "busy", label: null });
    }
  };

  const objectiveDelta = (r: WhatIfResponse) => {
    const d = r.comparison.deltas.find((x) => /objective/i.test(x.name));
    return d ? d.change : null;
  };

  return (
    <div className="page">
      <h1>What-if</h1>
      {state.error && <ErrorBanner message={state.error} />}
      {error && <ErrorBanner message={error} />}

      <Panel
        title="Baseline"
        subtitle={`Every operator below re-solves from plan ${activePlanId}.`}
      >
        <div className="stat-grid">
          <Stat label="Baseline objective" value={num(plan.objective)} hint={plan.solver} />
          <Stat label="Completion" value={pct(plan.completion)} />
          <Stat
            label="Late tasks"
            value={plan.late_tasks}
            tone={plan.late_tasks ? "warn" : "good"}
          />
          <Stat label="Assigned" value={`${plan.tasks_assigned}/${plan.tasks_total}`} />
        </div>
      </Panel>

      <Panel
        title="The six operators"
        actions={
          <button type="button" onClick={() => void runAll()} disabled={running !== null}>
            {running === "__all__" ? <Spinner label="running all" /> : "Run all six"}
          </button>
        }
      >
        <div className="operator-grid">
          {WHAT_IF_OPERATORS.map((op) => {
            const result = results[op];
            const delta = result ? objectiveDelta(result) : null;
            return (
              <div key={op} className="operator">
                <h4>{op.replace(/_/g, " ")}</h4>
                <p className="hint">{WHAT_IF_PARAMETERS[op].hint}</p>
                <label className="inline-label">
                  parameter
                  <input
                    value={overrides[op] ?? String(WHAT_IF_PARAMETERS[op].default)}
                    onChange={(e) => setOverrides((prev) => ({ ...prev, [op]: e.target.value }))}
                    disabled={op === "resource_outage"}
                    placeholder={op === "resource_outage" ? "resource id" : undefined}
                  />
                </label>
                <div className="row-actions">
                  <button type="button" onClick={() => void runOne(op)} disabled={running !== null}>
                    {running === op ? <Spinner label="running" /> : "Run"}
                  </button>
                </div>
                {result && (
                  <div
                    className={`operator-result ${
                      delta !== null && Math.abs(delta) < 1e-6 ? "absorbed" : "impacted"
                    }`}
                  >
                    <div className="row">
                      <span>Question</span>
                      <strong>{result.question}</strong>
                    </div>
                    <div className="row">
                      <span>Modification</span>
                      <strong>{result.modification}</strong>
                    </div>
                    <div className="row">
                      <span>Objective Δ</span>
                      <strong className={delta === null ? "" : delta > 1e-6 ? "late" : delta < -1e-6 ? "good-text" : ""}>
                        {delta === null
                          ? "not reported"
                          : Math.abs(delta) < 1e-6
                            ? "0 — absorbed"
                            : num(delta, 1)}
                      </strong>
                    </div>
                    <div className="row">
                      <span>Churn</span>
                      <strong>{result.comparison.churn.changed_assignments}</strong>
                    </div>
                    <div className="row">
                      <span>Solve time</span>
                      <strong>{`${num(result.seconds, 2)} s`}</strong>
                    </div>
                    {delta !== null && Math.abs(delta) < 1e-6 && (
                      <p className="hint">
                        This scenario absorbed the perturbation at a baseline of{" "}
                        {num(plan.objective, 0)}. That is a statement about this configuration, not
                        evidence the constraint never binds.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Panel>

      {Object.keys(results).length > 0 && (
        <Panel
          title="Comparison"
          subtitle="Sorted by the size of the objective change. Every row is a re-solve, not an estimate."
        >
          <div className="scroll">
            <Table
              rows={Object.values(results).sort((a, b) => {
                const da = Math.abs(objectiveDelta(a) ?? 0);
                const db = Math.abs(objectiveDelta(b) ?? 0);
                return db - da;
              })}
              rowKey={(r) => r.operator}
              empty="No results."
              columns={[
                { key: "operator", label: "Operator" },
                { key: "parameter", label: "Parameter", numeric: true },
                { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
                {
                  key: "objective",
                  label: "Objective Δ",
                  render: (r) => {
                    const d = objectiveDelta(r);
                    if (d === null) return "—";
                    return Math.abs(d) < 1e-6 ? (
                      <span className="muted">0 absorbed</span>
                    ) : (
                      <span className={d > 0 ? "late" : "good-text"}>{num(d, 1)}</span>
                    );
                  },
                  numeric: true,
                },
                {
                  key: "churn",
                  label: "Churn",
                  render: (r) => String(r.comparison.churn.changed_assignments),
                  numeric: true,
                },
                {
                  key: "churn_ratio",
                  label: "Churn ratio",
                  render: (r) => pct(r.comparison.churn.churn_ratio),
                  numeric: true,
                },
                { key: "seconds", label: "Solve", render: (r) => `${num(r.seconds, 2)} s`, numeric: true },
                { key: "scenario_plan_id", label: "Plan" },
              ]}
            />
          </div>
        </Panel>
      )}
    </div>
  );
}
