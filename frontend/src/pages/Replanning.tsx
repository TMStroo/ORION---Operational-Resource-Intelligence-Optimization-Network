/**
 * Replanning: local repair against full re-optimization.
 *
 * Both are run and both are shown, including when repair is the better answer.
 * The four states are Baseline, Disrupted, Local Repair, Full Re-optimization.
 * The recovery ratio is a comparison of two computed objectives, not a bound
 * on the true optimum, and the page says so where it is displayed.
 */

import React, { useState } from "react";
import { api } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, StatusBadge, Table, num, pct } from "../components/ui";
import type { AppState, Action } from "../state/store";

export function Replanning({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const [budget, setBudget] = useState<number>(10);
  const [solver, setSolver] = useState<string>("HEURISTIC");
  const [working, setWorking] = useState(false);
  const { scenario, plan, disruption, replan, activePlanId } = state;

  if (!scenario) {
    return (
      <div className="page">
        <h1>Replanning</h1>
        <Panel title="No scenario open">
          <p>Open a scenario, solve it, and apply a disruption before replanning.</p>
        </Panel>
      </div>
    );
  }

  if (!disruption) {
    return (
      <div className="page">
        <h1>Replanning</h1>
        <Panel title="No disruption to recover from">
          <p>
            Recovery is measured against a disrupted baseline. Apply a disruption on the
            Disruptions page first.
          </p>
        </Panel>
      </div>
    );
  }

  const runRecovery = async () => {
    if (!activePlanId) return;
    setWorking(true);
    dispatch({ type: "busy", label: "running local repair and full re-optimization" });
    try {
      const result = await api.replan(scenario.id, {
        disruption_type: disruption.disruption.type,
        target_id: disruption.disruption.target_id,
        magnitude: disruption.disruption.magnitude,
        duration: disruption.disruption.duration,
        seed: 0,
        run_full: true,
        solver,
        time_budget_s: budget,
      });
      dispatch({ type: "replan", payload: result });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setWorking(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const repair = replan?.repair_plan;
  const full = replan?.full_plan;

  return (
    <div className="page">
      <h1>Replanning</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title="Recover"
        subtitle={`Against disruption ${disruption.disruption.type} (${disruption.disruption.severity}), re-applied server-side so the repaired state matches.`}
      >
        <div className="form-grid">
          <label>
            Full re-optimization solver
            <select value={solver} onChange={(e) => setSolver(e.target.value)}>
              <option value="HEURISTIC">HEURISTIC</option>
              <option value="LOCAL_SEARCH">LOCAL_SEARCH</option>
              <option value="CP_SAT">CP-SAT</option>
              <option value="MILP">MILP</option>
            </select>
          </label>
          <label>
            Time budget (s)
            <input
              type="number"
              min={0.001}
              step={1}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="row-actions">
          <button type="button" onClick={() => void runRecovery()} disabled={working || !activePlanId}>
            {working ? <Spinner label="recovering" /> : "Run local repair + full re-optimization"}
          </button>
        </div>
      </Panel>

      <Panel
        title="Four states"
        subtitle="Baseline, disrupted, local repair, full re-optimization — the same objective, scored by the same layer."
      >
        <div className="four-up">
          <StateCard
            label="Baseline"
            note="The plan before the disruption."
            plan={plan ?? null}
            status={plan?.status}
          />
          <StateCard
            label="Disrupted"
            note="The impact the disruption has on the baseline. No repair attempted."
            plan={null}
            status={null}
            riskCount={disruption.disruption.at_risk_deadlines.length}
            affectedCount={disruption.disruption.affected_tasks.length}
          />
          <StateCard
            label="Local repair"
            note="Reassignment around the disruption only. Does not re-solve the rest of the schedule."
            plan={repair ?? null}
            status={replan?.repair.produced_plan ? "REPAIRED" : null}
            seconds={replan?.repair_seconds}
            strategy={replan?.repair.strategy}
            extra={
              replan
                ? `${replan.repair.recovered_tasks} recovered, ${replan.repair.unrecovered_tasks} not`
                : null
            }
          />
          <StateCard
            label="Full re-optimization"
            note="The whole scenario re-solved from scratch. Slower; no local minimum."
            plan={full ?? null}
            status={replan?.full_status ?? null}
            seconds={replan?.full_seconds}
            reason={replan?.full_reason}
          />
        </div>
      </Panel>

      {replan && (
        <>
          <Panel
            title="Recovery quality"
            subtitle="How close the cheap repair came to the expensive answer, and how much of the plan it left alone."
          >
            <div className="stat-grid">
              <Stat
                label="Recovery"
                value={pct(replan.recovery_pct, 0)}
                hint="full objective ÷ repair objective, capped at 1"
                tone={
                  replan.recovery_pct >= 0.99 ? "good" : replan.recovery_pct >= 0.9 ? "warn" : "bad"
                }
              />
              <Stat
                label="Churn"
                value={`${replan.repair_comparison?.churn.changed_assignments ?? 0}`}
                hint="baseline assignments that changed"
              />
              <Stat
                label="Preserved"
                value={pct(replan.repair_preserved_ratio, 0)}
                hint="baseline assignments left untouched by the repair"
              />
              <Stat label="Repair runtime" value={`${num(replan.repair_seconds, 3)} s`} />
              <Stat
                label="Full runtime"
                value={
                  replan.full_status === "TIME_LIMIT" || replan.full_status === "INFEASIBLE"
                    ? replan.full_status
                    : `${num(replan.full_seconds, 3)} s`
                }
                hint={replan.full_reason}
              />
              <Stat
                label="Speed-up"
                value={
                  replan.repair_seconds > 0 && replan.full_seconds > 0
                    ? `${num(replan.full_seconds / replan.repair_seconds, 1)}x`
                    : "—"
                }
                hint="full re-optimization ÷ local repair"
              />
            </div>
            <p className="note">
              A recovery below 100% means the full re-optimization found a better plan than the local
              repair did. That is reported rather than smoothed over, and the ratio compares two
              computed values — it is not a bound on the true optimum.
            </p>
          </Panel>

          {replan.repair_comparison && replan.repair_comparison.deltas.length > 0 && (
            <Panel
              title="Repair versus baseline"
              subtitle="Metric deltas from the stored comparison, with the direction each metric is better in."
            >
              <MetricDeltas comparison={replan.repair_comparison} />
            </Panel>
          )}

          {replan.full_comparison && replan.full_comparison.deltas.length > 0 && (
            <Panel title="Full re-optimization versus baseline">
              <MetricDeltas comparison={replan.full_comparison} />
            </Panel>
          )}

          {(repair || full) && (
            <Panel title="Open a resulting plan" subtitle="Loads it into the Planning view.">
              <div className="row-actions">
                {repair && (
                  <button
                    type="button"
                    onClick={async () => {
                      dispatch({ type: "busy", label: "loading repaired plan" });
                      try {
                        const loaded = await api.getPlan(repair.id);
                        dispatch({ type: "planLoaded", plan: loaded, planId: loaded.id });
                      } catch (e) {
                        dispatch({ type: "error", message: (e as Error).message });
                      } finally {
                        dispatch({ type: "busy", label: null });
                      }
                    }}
                  >
                    Open local-repair plan
                  </button>
                )}
                {full && (
                  <button
                    type="button"
                    onClick={async () => {
                      dispatch({ type: "busy", label: "loading re-optimized plan" });
                      try {
                        const loaded = await api.getPlan(full.id);
                        dispatch({ type: "planLoaded", plan: loaded, planId: loaded.id });
                      } catch (e) {
                        dispatch({ type: "error", message: (e as Error).message });
                      } finally {
                        dispatch({ type: "busy", label: null });
                      }
                    }}
                  >
                    Open full re-optimization plan
                  </button>
                )}
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}

function StateCard({
  label,
  note,
  plan,
  status,
  seconds,
  strategy,
  reason,
  extra,
  riskCount,
  affectedCount,
}: {
  label: string;
  note: string;
  plan: {
    id: string;
    objective: number;
    tasks_assigned: number;
    tasks_total: number;
    late_tasks: number;
    completion: number;
    runtime_s: number;
  } | null;
  status?: string | null;
  seconds?: number;
  strategy?: string;
  reason?: string;
  extra?: string | null;
  riskCount?: number;
  affectedCount?: number;
}) {
  return (
    <div className="state-card">
      <header>
        <h4>{label}</h4>
        {status ? <StatusBadge status={status} /> : <span className="muted">—</span>}
      </header>
      <p className="note">{note}</p>
      {plan ? (
        <>
          <div className="state-metric">
            <span>Objective</span>
            <strong>{num(plan.objective, 1)}</strong>
          </div>
          <div className="state-metric">
            <span>Assigned</span>
            <strong>
              {plan.tasks_assigned}/{plan.tasks_total}
            </strong>
          </div>
          <div className="state-metric">
            <span>Late</span>
            <strong className={plan.late_tasks ? "late" : "good-text"}>{plan.late_tasks}</strong>
          </div>
          <div className="state-metric">
            <span>Completion</span>
            <strong>{pct(plan.completion)}</strong>
          </div>
          <div className="state-metric">
            <span>Runtime</span>
            <strong>{`${num(plan.runtime_s, 3)} s`}</strong>
          </div>
        </>
      ) : riskCount !== undefined ? (
        <>
          <div className="state-metric">
            <span>Tasks affected</span>
            <strong>{affectedCount ?? 0}</strong>
          </div>
          <div className="state-metric">
            <span>Deadlines at risk</span>
            <strong className={riskCount ? "late" : "good-text"}>{riskCount}</strong>
          </div>
          <p className="note">
            No plan exists for this state. The disruption's effect is on the baseline's deadlines, not
            on a separate schedule.
          </p>
        </>
      ) : (
        <p className="empty">Not run yet.</p>
      )}
      {seconds !== undefined && (
        <div className="state-metric">
          <span>Elapsed</span>
          <strong>{`${num(seconds, 3)} s`}</strong>
        </div>
      )}
      {strategy && <div className="state-metric"><span>Strategy</span><strong>{strategy}</strong></div>}
      {extra && <div className="state-metric"><span>Tasks</span><strong>{extra}</strong></div>}
      {reason && <p className="note">{reason}</p>}
    </div>
  );
}

function MetricDeltas({ comparison }: { comparison: NonNullable<AppState["replan"]>["repair_comparison"] }) {
  if (!comparison) return null;
  return (
    <Table
      rows={comparison.deltas}
      rowKey={(d) => d.name}
      empty="No deltas recorded."
      columns={[
        { key: "label", label: "Metric" },
        { key: "before", label: "Before", render: (d) => num(d.before, 2), numeric: true },
        { key: "after", label: "After", render: (d) => num(d.after, 2), numeric: true },
        {
          key: "change",
          label: "Change",
          render: (d) => {
            const worse = d.higher_is_better ? d.change < 0 : d.change > 0;
            return (
              <span className={Math.abs(d.change) < 1e-9 ? "muted" : worse ? "late" : "good-text"}>
                {d.change > 0 ? "+" : ""}
                {num(d.change, 2)}
                {d.unit ? ` ${d.unit}` : ""}
              </span>
            );
          },
          numeric: true,
        },
      ]}
    />
  );
}
