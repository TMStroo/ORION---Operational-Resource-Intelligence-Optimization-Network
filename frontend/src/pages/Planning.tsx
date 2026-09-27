/**
 * Planning: the plan as an inspector can read it.
 *
 * The selected task is shown with the reasons behind its assignment, so the
 * question "why did that go there" is answerable without reading a log.
 */

import React, { useEffect, useMemo, useState } from "react";
import { api, SOLVERS } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, StatusBadge, Table, clock, num, pct } from "../components/ui";
import { Timeline, UtilisationBars } from "../components/Timeline";
import type { AppState, Action } from "../state/store";

export function Planning({ state, dispatch }: { state: AppState; dispatch: React.Dispatch<Action> }) {
  const [solver, setSolver] = useState<string>("HEURISTIC");
  const [budget, setBudget] = useState<number>(10);
  const [selected, setSelected] = useState<string | null>(null);
  const [solving, setSolving] = useState(false);

  const { scenario, plan, plans, validation } = state;
  const invalid = validation ? !validation.valid : false;

  const late = useMemo(
    () => new Set((plan?.assignments ?? []).filter((a) => a.lateness > 0).map((a) => a.task_id)),
    [plan],
  );

  // Reload the plan list whenever this page is opened. Without it the list is
  // whatever the create flow happened to leave in the store, so returning to
  // the page after a repair elsewhere would show a stale set of plans.
  useEffect(() => {
    if (!scenario) return;
    let cancelled = false;
    api
      .listPlans(scenario.id)
      .then(({ items }) => {
        if (!cancelled) dispatch({ type: "plansRefreshed", payload: items });
      })
      .catch((e) => {
        if (!cancelled) dispatch({ type: "error", message: (e as Error).message });
      });
    return () => {
      cancelled = true;
    };
  }, [scenario, dispatch]);

  if (!scenario) return <NeedScenario />;

  const solve = async () => {
    if (!scenario) return;
    setSolving(true);
    dispatch({ type: "busy", label: `solving with ${solver}` });
    try {
      const result = await api.optimize(scenario.id, { solver, time_budget_s: budget, seed: 0 });
      dispatch({ type: "planCreated", plan: result, planId: result.id });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setSolving(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const openPlan = async (id: string) => {
    dispatch({ type: "busy", label: "loading plan" });
    try {
      const loaded = await api.getPlan(id);
      dispatch({ type: "planLoaded", plan: loaded, planId: loaded.id });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      dispatch({ type: "busy", label: null });
    }
  };

  return (
    <div className="page">
      <h1>Planning</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title="Optimise"
        subtitle="The chosen solver and budget are passed to the engine; the status it returns is shown unchanged."
      >
        <div className="form-grid">
          <label>
            Solver
            <select value={solver} onChange={(e) => setSolver(e.target.value)}>
              {SOLVERS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Time budget (s)
            <input
              type="number"
              min={0.001}
              step={0.5}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="row-actions">
          <button type="button" onClick={() => void solve()} disabled={solving || invalid}>
            {solving ? <Spinner label="solving" /> : "Solve"}
          </button>
          {invalid && <span className="note">Blocked: the scenario failed validation.</span>}
        </div>
      </Panel>

      {plans.length > 0 && (
        <Panel title={`Plans for this scenario (${plans.length})`}>
          <Table
            rows={plans}
            rowKey={(p) => p.id}
            onRowClick={(p) => void openPlan(p.id)}
            highlight={(p) => p.id === state.activePlanId}
            empty="No plans."
            columns={[
              { key: "id", label: "Plan" },
              { key: "label", label: "Label" },
              { key: "solver", label: "Solver" },
              {
                key: "status",
                label: "Status",
                render: (p) => <StatusBadge status={p.status} />,
              },
              { key: "objective", label: "Objective", render: (p) => num(p.objective), numeric: true },
              {
                key: "tasks",
                label: "Tasks",
                render: (p) => `${p.tasks_assigned}/${p.tasks_total}`,
                numeric: true,
              },
              { key: "runtime_s", label: "Runtime", render: (p) => `${num(p.runtime_s, 3)} s`, numeric: true },
            ]}
          />
        </Panel>
      )}

      {!plan ? (
        <Panel title="No plan selected">
          <p>Solve the scenario, or open a stored plan from the table above.</p>
        </Panel>
      ) : (
        <>
          <Panel
            title="Plan summary"
            subtitle={plan.label || plan.id}
            actions={<StatusBadge status={plan.status} />}
          >
            <div className="stat-grid">
              <Stat label="Objective" value={num(plan.objective)} hint={plan.solver} />
              <Stat label="Completion" value={pct(plan.completion)} />
              <Stat label="Late tasks" value={plan.late_tasks} tone={plan.late_tasks ? "bad" : "good"} />
              <Stat label="Assigned" value={`${plan.tasks_assigned}/${plan.tasks_total}`} />
              <Stat label="Travel" value={`${num(plan.total_travel_km, 1)} km`} hint={`${plan.total_travel_minutes} min`} />
              <Stat label="Operating cost" value={num(plan.total_operating_cost)} />
              <Stat label="Runtime" value={`${num(plan.runtime_s, 3)} s`} />
              <Stat label="Service level" value={pct(plan.service_level)} />
            </div>

            {plan.objective_breakdown && (
              <div className="breakdown">
                {Object.entries(plan.objective_breakdown)
                  .filter(([k]) => k !== "total")
                  .map(([k, v]) => (
                    <div key={k} className="breakdown-row">
                      <span>{k.replace(/_/g, " ")}</span>
                      <span className="num">{num(v as number)}</span>
                    </div>
                  ))}
              </div>
            )}
          </Panel>

          {plan.solver_runs.length > 0 && (
            <Panel title="Solver runs" subtitle="Every run recorded against this plan, including declines.">
              <Table
                rows={plan.solver_runs}
                rowKey={(r, i) => `${r.solver}-${i}`}
                empty="No runs recorded."
                columns={[
                  { key: "solver", label: "Solver" },
                  { key: "status", label: "Status", render: (r) => <StatusBadge status={r.status} /> },
                  { key: "objective", label: "Objective", render: (r) => num(r.objective), numeric: true },
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

          {plan.violations.length > 0 && (
            <Panel title={`Violations (${plan.violations.length})`}>
              <ul className="issue-list">
                {plan.violations.map((v, i) => (
                  <li key={i}>
                    <span className="issue-kind">{v.kind}</span> {v.detail}
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel
            title="Timeline"
            subtitle="One row per resource. Travel legs are drawn separately so idle time is not mistaken for movement."
          >
            <Timeline
              scenario={scenario}
              assignments={plan.assignments}
              highlight={late}
              selected={selected}
              onSelect={setSelected}
            />
          </Panel>

          <Panel title="Resource utilisation" subtitle="Measured against each resource's own shift.">
            <UtilisationBars scenario={scenario} utilization={plan.utilization} />
          </Panel>

          {selected && (
            <Panel title={`Task ${selected}`}>
              <AssignmentDetail plan={plan} taskId={selected} />
            </Panel>
          )}

          <Panel title={`Assignments (${plan.assignments.length})`}>
            <div className="scroll">
              <Table
                rows={plan.assignments}
                rowKey={(a) => a.task_id}
                onRowClick={(a) => setSelected(a.task_id)}
                highlight={(a) => a.task_id === selected}
                empty="This plan assigns nothing."
                columns={[
                  { key: "task_id", label: "Task" },
                  { key: "resource_id", label: "Resource" },
                  { key: "start", label: "Start", render: (a) => clock(a.start), numeric: true },
                  { key: "end", label: "End", render: (a) => clock(a.end), numeric: true },
                  { key: "location", label: "Location" },
                  { key: "priority", label: "Priority" },
                  {
                    key: "lateness",
                    label: "Late (min)",
                    render: (a) => (a.lateness > 0 ? <span className="late">{a.lateness}</span> : "0"),
                    numeric: true,
                  },
                  {
                    key: "travel",
                    label: "Travel",
                    render: (a) => `${a.travel_before} min / ${a.travel_distance_km.toFixed(1)} km`,
                    numeric: true,
                  },
                ]}
              />
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

function AssignmentDetail({ plan, taskId }: { plan: NonNullable<AppState["plan"]>; taskId: string }) {
  const a = plan.assignments.find((x) => x.task_id === taskId);
  if (!a) return <p className="empty">No assignment for {taskId}.</p>;
  const others = plan.assignments.filter((x) => x.resource_id !== a.resource_id);
  const busiest = others.length
    ? [...others].sort((x, y) => (y.end - y.start) - (x.end - x.start))[0]
    : undefined;
  return (
    <dl className="detail-list">
      <div>
        <dt>Assigned to</dt>
        <dd>{a.resource_id}</dd>
      </div>
      <div>
        <dt>Window</dt>
        <dd>
          {clock(a.start)}–{clock(a.end)} ({a.end - a.start} min)
        </dd>
      </div>
      <div>
        <dt>Location</dt>
        <dd>{a.location}</dd>
      </div>
      <div>
        <dt>Travel to arrive</dt>
        <dd>
          {a.travel_before} min · {a.travel_distance_km.toFixed(1)} km
        </dd>
      </div>
      <div>
        <dt>Priority</dt>
        <dd>{a.priority}</dd>
      </div>
      <div>
        <dt>Lateness</dt>
        <dd>{a.lateness > 0 ? <span className="late">{a.lateness} min late</span> : "on time"}</dd>
      </div>
      {busiest && (
        <div>
          <dt>Other work on other resources</dt>
          <dd>
            {busiest.resource_id} is busy {clock(busiest.start)}–{clock(busiest.end)} with{" "}
            {busiest.task_id}
          </dd>
        </div>
      )}
    </dl>
  );
}

function NeedScenario() {
  return (
    <Panel title="No scenario open">
      <p>Open a scenario from the Scenarios page first.</p>
    </Panel>
  );
}
