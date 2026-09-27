/**
 * Scenarios: list, create, inspect, validate.
 *
 * Validation is a gate, not a decoration. The Optimize button is disabled on a
 * scenario the backend reports as invalid, and the issues are listed - the
 * point is to stop an unsatisfiable scenario from reaching the solver silently.
 */

import React, { useState } from "react";
import { api } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, Table, clock } from "../components/ui";
import type { AppState, Action } from "../state/store";

export function Scenarios({ state, dispatch }: { state: AppState; dispatch: React.Dispatch<Action> }) {
  type Difficulty = "easy" | "medium" | "hard";
  const [form, setForm] = useState({
    name: "ops-day",
    task_count: 40,
    resource_count: 12,
    seed: 7,
    difficulty: "medium" as Difficulty,
  });
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const refresh = async () => {
    dispatch({ type: "busy", label: "loading scenarios" });
    try {
      const { items } = await api.listScenarios(100);
      dispatch({ type: "scenarios", payload: items });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    }
  };

  const create = async () => {
    setCreating(true);
    setCreateError(null);
    dispatch({ type: "busy", label: "creating scenario" });
    try {
      const scenario = await api.createScenario(form);
      dispatch({ type: "openScenario", id: scenario.id, scenario });
      const validation = await api.validateScenario(scenario.id);
      dispatch({ type: "validation", payload: validation });
      const { items } = await api.listPlans(scenario.id);
      dispatch({ type: "plansRefreshed", payload: items });
      await refresh();
    } catch (e) {
      setCreateError((e as Error).message);
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setCreating(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const open = async (id: string) => {
    dispatch({ type: "busy", label: "opening scenario" });
    try {
      const scenario = await api.getScenario(id);
      dispatch({ type: "openScenario", id, scenario });
      const validation = await api.validateScenario(id);
      dispatch({ type: "validation", payload: validation });
      const { items } = await api.listPlans(id);
      dispatch({ type: "plansRefreshed", payload: items });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      dispatch({ type: "busy", label: null });
    }
  };

  return (
    <div className="page">
      <h1>Scenarios</h1>
      {state.error && <ErrorBanner message={state.error} onRetry={refresh} />}

      <Panel
        title="Stored scenarios"
        actions={
          <button type="button" onClick={refresh} disabled={state.busy !== null}>
            {state.busy === "loading scenarios" ? <Spinner label="loading" /> : "Refresh"}
          </button>
        }
      >
        <Table
          rows={state.scenarios}
          rowKey={(s) => s.id}
          onRowClick={(s) => void open(s.id)}
          empty="No scenarios stored yet. Create one below."
          columns={[
            { key: "name", label: "Name" },
            { key: "id", label: "Id" },
            { key: "task_count", label: "Tasks", numeric: true },
            { key: "resource_count", label: "Resources", numeric: true },
            { key: "source", label: "Source" },
            {
              key: "created_at",
              label: "Created",
              render: (s) => (s.created_at ? s.created_at.replace("T", " ").slice(0, 19) : "—"),
            },
          ]}
        />
      </Panel>

      <Panel
        title="Create a scenario"
        subtitle="The same request with the same seed produces the same scenario."
      >
        <div className="form-grid">
          <label>
            Name
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </label>
          <label>
            Tasks
            <input
              type="number"
              min={1}
              max={1000}
              value={form.task_count}
              onChange={(e) => setForm({ ...form, task_count: Number(e.target.value) })}
            />
          </label>
          <label>
            Resources
            <input
              type="number"
              min={1}
              max={200}
              value={form.resource_count}
              onChange={(e) => setForm({ ...form, resource_count: Number(e.target.value) })}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={form.seed}
              onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
            />
          </label>
          <label>
            Difficulty
            <select
              value={form.difficulty}
              onChange={(e) => setForm({ ...form, difficulty: e.target.value as Difficulty })}
            >
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
            </select>
          </label>
        </div>
        {createError && <ErrorBanner message={createError} />}
        <div className="row-actions">
          <button type="button" onClick={() => void create()} disabled={creating}>
            {creating ? "Creating…" : "Create scenario"}
          </button>
        </div>
      </Panel>

      {state.scenario && <ScenarioDetailPanel state={state} />}
    </div>
  );
}

function ScenarioDetailPanel({ state }: { state: AppState }) {
  const scenario = state.scenario!;
  const validation = state.validation;
  const invalid = validation ? !validation.valid : false;

  return (
    <>
      <Panel
        title={`${scenario.name} — validation`}
        subtitle={
          validation
            ? validation.valid
              ? "The backend reports this scenario as valid."
              : `${validation.issues.length} issue(s) must be resolved before optimising.`
            : "Validation has not been run."
        }
      >
        <div className="stat-grid">
          <Stat label="Tasks" value={scenario.task_count} />
          <Stat label="Resources" value={scenario.resource_count} />
          <Stat
            label="Horizon"
            value={`${clock(scenario.horizon.start)}–${clock(scenario.horizon.end)}`}
          />
          <Stat label="Depot" value={scenario.depot || "—"} />
        </div>
        {validation && !validation.valid && (
          <ul className="issue-list">
            {validation.issues.map((issue, i) => (
              <li key={i}>
                <span className="issue-kind">{issue.kind}</span> {issue.detail}
              </li>
            ))}
          </ul>
        )}
        {invalid && (
          <p className="note">
            Optimisation is blocked for this scenario. A scenario with a task no resource can serve
            produces NO_ELIGIBLE_PAIR at solve time; catching it here is the point.
          </p>
        )}
      </Panel>

      <Panel title={`Tasks (${scenario.tasks.length})`}>
        <div className="scroll">
          <Table
            rows={scenario.tasks}
            rowKey={(t) => t.id}
            empty="No tasks."
            columns={[
              { key: "id", label: "Task" },
              { key: "location", label: "Location" },
              { key: "priority", label: "Priority" },
              { key: "release_time", label: "Release", render: (t) => clock(t.release_time), numeric: true },
              { key: "deadline", label: "Deadline", render: (t) => clock(t.deadline), numeric: true },
              { key: "duration", label: "Minutes", numeric: true },
              {
                key: "required_capabilities",
                label: "Requires",
                render: (t) => (t.required_capabilities.length ? t.required_capabilities.join(", ") : "—"),
              },
              {
                key: "dependencies",
                label: "After",
                render: (t) => (t.dependencies.length ? t.dependencies.join(", ") : "—"),
              },
            ]}
          />
        </div>
      </Panel>

      <Panel title={`Resources (${scenario.resources.length})`}>
        <div className="scroll">
          <Table
            rows={scenario.resources}
            rowKey={(r) => r.id}
            empty="No resources."
            columns={[
              { key: "id", label: "Resource" },
              { key: "kind", label: "Kind" },
              { key: "home_location", label: "Home" },
              {
                key: "shift",
                label: "Shift",
                render: (r) => `${clock(r.shift_start)}–${clock(r.shift_end)}`,
              },
              { key: "max_work_minutes", label: "Max minutes", numeric: true },
              {
                key: "capabilities",
                label: "Capabilities",
                render: (r) => (r.capabilities.length ? r.capabilities.join(", ") : "—"),
              },
              { key: "status", label: "Status" },
            ]}
          />
        </div>
      </Panel>

      <Panel title="Objective weights" subtitle="The single objective every solver optimises.">
        <div className="weight-grid">
          {Object.entries(scenario.objective_weights).map(([k, v]) => (
            <div key={k} className="weight">
              <span>{k.replace(/^w_/, "").replace(/_/g, " ")}</span>
              <strong>{v}</strong>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}
