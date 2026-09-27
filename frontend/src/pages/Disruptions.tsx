/**
 * Disruptions: apply one, see what it puts at risk.
 *
 * The affected-task list is the backend's, computed by the same eligibility
 * rule the solvers use — not a client-side guess. The disrupted state is
 * stored under its own id, so the pre-disruption scenario is never overwritten.
 */

import React, { useState } from "react";
import { api, DISRUPTION_TYPES, type DisruptionType } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, clock } from "../components/ui";
import type { AppState, Action } from "../state/store";

interface Form {
  type: DisruptionType;
  target_id: string;
  magnitude: number;
  duration: number;
}

const MAGNITUDE_HINT: Record<DisruptionType, string> = {
  VEHICLE_FAILURE: "0–1. The fraction of capacity the failed vehicle loses for the duration.",
  RESOURCE_UNAVAILABLE: "0–1. Share of the resource's available time removed.",
  MAINTENANCE: "0–1. Share of capacity diverted to maintenance.",
  DEMAND_SURGE: "Multiplier on demand, e.g. 1.3 for a 30% surge.",
  TRAVEL_INCREASE: "Multiplier on travel times, e.g. 1.25.",
};

export function Disruptions({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const [form, setForm] = useState<Form>({
    type: "RESOURCE_UNAVAILABLE",
    target_id: state.scenario?.resources[0]?.id ?? "",
    magnitude: 0.5,
    duration: 240,
  });
  const [working, setWorking] = useState(false);

  const { scenario, disruption, activePlanId } = state;

  if (!scenario) {
    return (
      <div className="page">
        <h1>Disruptions</h1>
        <Panel title="No scenario open">
          <p>Open a scenario to apply a disruption to it.</p>
        </Panel>
      </div>
    );
  }

  const submit = async () => {
    if (!activePlanId) return;
    setWorking(true);
    dispatch({ type: "busy", label: "applying disruption" });
    try {
      const result = await api.disrupt(scenario.id, {
        type: form.type,
        target_id: form.target_id || null,
        magnitude: form.magnitude,
        duration: form.duration,
        seed: 0,
      });
      dispatch({ type: "disruption", payload: result });
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setWorking(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const canSubmit = Boolean(activePlanId) && Boolean(form.target_id);

  return (
    <div className="page">
      <h1>Disruptions</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title="Apply a disruption"
        subtitle={
          activePlanId
            ? `Measured against plan ${activePlanId}.`
            : "Solve the scenario first — a disruption is always reported against a plan."
        }
      >
        <div className="form-grid">
          <label>
            Type
            <select
              value={form.type}
              onChange={(e) => setForm({ ...form, type: e.target.value as DisruptionType })}
            >
              {DISRUPTION_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, " ").toLowerCase()}
                </option>
              ))}
            </select>
          </label>

          <label>
            Resource
            <select
              value={form.target_id}
              onChange={(e) => setForm({ ...form, target_id: e.target.value })}
            >
              {scenario.resources.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} ({r.kind})
                </option>
              ))}
            </select>
          </label>

          <label>
            Magnitude
            <input
              type="number"
              step={0.05}
              min={0}
              max={10}
              value={form.magnitude}
              onChange={(e) => setForm({ ...form, magnitude: Number(e.target.value) })}
            />
            <span className="hint">{MAGNITUDE_HINT[form.type]}</span>
          </label>

          <label>
            Duration (min)
            <input
              type="number"
              min={0}
              value={form.duration}
              onChange={(e) => setForm({ ...form, duration: Number(e.target.value) })}
            />
          </label>
        </div>

        <div className="row-actions">
          <button type="button" onClick={() => void submit()} disabled={!canSubmit || working}>
            {working ? <Spinner label="applying" /> : "Apply disruption"}
          </button>
          {!activePlanId && <span className="note">No plan open yet.</span>}
        </div>
      </Panel>

      {disruption && (
        <>
          <Panel
            title={`Applied: ${disruption.disruption.type}`}
            subtitle={disruption.disruption.description}
            actions={!disruption.recorded ? <span className="note">Not recorded (duplicate id)</span> : null}
          >
            <div className="stat-grid">
              <Stat
                label="Severity"
                value={disruption.disruption.severity}
                tone={disruption.disruption.severity === "high" ? "bad" : "warn"}
              />
              <Stat
                label="Affected tasks"
                value={disruption.disruption.affected_tasks.length}
                hint="by the backend's own eligibility rule"
              />
              <Stat
                label="Affected resources"
                value={disruption.disruption.affected_resources.length}
              />
              <Stat
                label="Deadlines at risk"
                value={disruption.disruption.at_risk_deadlines.length}
                tone={disruption.disruption.at_risk_deadlines.length ? "warn" : "good"}
              />
              <Stat label="At" value={clock(disruption.disruption.timestamp)} />
              <Stat
                label="Disrupted scenario"
                value={disruption.after_scenario_id ?? "—"}
                hint={disruption.after_scenario_id ? "stored separately; the original is untouched" : undefined}
              />
            </div>

            {disruption.disruption.affected_tasks.length > 0 && (
              <>
                <h4>Affected tasks</h4>
                <div className="tag-row">
                  {disruption.disruption.affected_tasks.map((t) => (
                    <span key={t} className="tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            )}
            {disruption.disruption.at_risk_deadlines.length > 0 && (
              <>
                <h4>Deadlines at risk</h4>
                <div className="tag-row">
                  {disruption.disruption.at_risk_deadlines.map((t) => (
                    <span key={t} className="tag warn-tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            )}
          </Panel>

          <Panel title="Continue">
            <p className="note">
              Go to <strong>Replanning</strong> to run a local repair and a full re-optimization
              against this disruption, and compare the two. Neither is presented as the answer.
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}
