/**
 * Simulation: playback over the real event stream.
 *
 * The backend runs the discrete-event engine to completion and returns the
 * ordered trace. Playback here is a cursor over that trace - it replays what
 * actually happened rather than simulating a second time in the browser, so
 * the display cannot disagree with the engine's own numbers.
 */

import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, StatusBadge, Table, clock, num } from "../components/ui";
import type { AppState, Action } from "../state/store";

const SPEEDS = [
  { label: "0.5x", ms: 600 },
  { label: "1x", ms: 300 },
  { label: "4x", ms: 80 },
  { label: "16x", ms: 20 },
];

export function Simulation({ state, dispatch }: { state: AppState; dispatch: React.Dispatch<Action> }) {
  const [playing, setPlaying] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [speedIndex, setSpeedIndex] = useState(1);
  const [running, setRunning] = useState(false);
  const timer = useRef<number | null>(null);

  const { scenario, plan, simulation, activePlanId } = state;

  // A new run invalidates the old cursor.
  useEffect(() => {
    setCursor(0);
    setPlaying(false);
  }, [simulation?.run_id]);

  useEffect(() => {
    if (!playing) {
      if (timer.current) window.clearTimeout(timer.current);
      return;
    }
    timer.current = window.setTimeout(() => {
      setCursor((c) => {
        const next = c + 1;
        if (simulation && next >= simulation.events.length) {
          setPlaying(false);
          return simulation.events.length;
        }
        return next;
      });
    }, SPEEDS[speedIndex]?.ms ?? 300);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [playing, cursor, speedIndex, simulation]);

  if (!scenario) {
    return (
      <div className="page">
        <h1>Simulation</h1>
        <Panel title="No scenario open">
          <p>Open a scenario and solve it first; the engine runs against a plan.</p>
        </Panel>
      </div>
    );
  }

  const run = async () => {
    if (!plan || !activePlanId) return;
    setRunning(true);
    dispatch({ type: "busy", label: "running the simulation engine" });
    try {
      // The plan is named explicitly. "The newest" would change under the user
      // and could be a run that scheduled nothing.
      const result = await api.simulate(scenario.id, { plan_id: activePlanId, seed: 0 });
      dispatch({ type: "simulation", payload: result });
      setCursor(0);
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setRunning(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const events = simulation?.events ?? [];
  const visible = events.slice(0, cursor);
  const now = visible.length ? (visible[visible.length - 1]?.time ?? 0) : 0;

  const stateNow = useMemo(() => summarise(visible), [visible]);

  if (simulation && simulation.status === "NOT_APPLICABLE") {
    return (
      <div className="page">
        <h1>Simulation</h1>
        <Panel title="Nothing to simulate">
          <p>
            Plan <code>{simulation.plan_id}</code> has status{" "}
            <StatusBadge status={simulation.plan_status} /> and assigns no tasks, so there is no
            event stream. This is the backend saying so explicitly, rather than returning an empty
            trace that would be indistinguishable from a simulation that did nothing.
          </p>
        </Panel>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Simulation</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title="Run the discrete-event engine"
        subtitle={
          plan
            ? `Plan ${plan.id} · ${plan.tasks_assigned} assignments`
            : "No plan loaded. Solve the scenario on the Planning page first."
        }
      >
        <div className="row-actions">
          <button type="button" onClick={() => void run()} disabled={!plan || running}>
            {running ? <Spinner label="running" /> : "Run simulation"}
          </button>
        </div>
      </Panel>

      {simulation && (
        <>
          <Panel
            title="Playback"
            subtitle={`Run ${simulation.run_id} · ${simulation.event_count} events from the engine`}
            actions={
              simulation.status !== "COMPLETED" && (
                <StatusBadge status={simulation.status} title="The engine did not run to completion" />
              )
            }
          >
            <div className="transport">
              <button type="button" onClick={() => setPlaying((p) => !p)} disabled={events.length === 0}>
                {playing ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setPlaying(false);
                  setCursor((c) => Math.max(0, c - 1));
                }}
                disabled={events.length === 0}
              >
                Step back
              </button>
              <button
                type="button"
                onClick={() => {
                  setPlaying(false);
                  setCursor((c) => Math.min(events.length, c + 1));
                }}
                disabled={events.length === 0}
              >
                Step
              </button>
              <button
                type="button"
                onClick={() => {
                  setPlaying(false);
                  setCursor(events.length);
                }}
                disabled={events.length === 0}
              >
                Skip to end
              </button>
              <label className="speed">
                Speed
                <select value={speedIndex} onChange={(e) => setSpeedIndex(Number(e.target.value))}>
                  {SPEEDS.map((s, i) => (
                    <option key={s.label} value={i}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </label>
              <span className="clock-display">{clock(now)}</span>
            </div>

            <input
              className="scrubber"
              type="range"
              min={0}
              max={events.length}
              value={cursor}
              onChange={(e) => {
                setPlaying(false);
                setCursor(Number(e.target.value));
              }}
              aria-label="Event cursor"
            />

            <div className="stat-grid">
              <Stat label="Simulated time" value={clock(now)} />
              <Stat label="Events replayed" value={`${visible.length} / ${events.length}`} />
              <Stat label="Tasks completed" value={stateNow.completed} />
              <Stat label="Active" value={stateNow.active} />
              <Stat label="In progress" value={stateNow.started} />
              <Stat label="Resources moving" value={stateNow.busyResources} />
              <Stat
                label="Final completed"
                value={simulation.completed_tasks}
                hint={`${simulation.late_tasks} late, ${simulation.failed_tasks} failed`}
              />
              <Stat label="Engine runtime" value={`${num(simulation.runtime_s, 3)} s`} />
            </div>
          </Panel>

          <Panel title={`Event log (${visible.length} shown)`}>
            <div className="scroll tall">
              <Table
                rows={[...visible].reverse()}
                rowKey={(e) => String(e.ordinal)}
                empty="Press play or step to advance through the trace."
                columns={[
                  { key: "ordinal", label: "#", numeric: true },
                  { key: "time", label: "Time", render: (e) => clock(e.time), numeric: true },
                  { key: "type", label: "Event" },
                  { key: "subject_id", label: "Subject" },
                  { key: "resource_id", label: "Resource" },
                  { key: "detail", label: "Detail" },
                ]}
              />
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

interface SimEventish {
  time: number;
  type: string;
  subject_id: string | null;
  resource_id: string | null;
}

/**
 * Derive world state at the cursor.
 *
 * This replays the engine's own events; it does not re-run the simulation. If
 * the backend had replayed it, the two could disagree, and the one on screen is
 * the one a reviewer would trust.
 */
function summarise(events: SimEventish[]) {
  const completed = new Set<string>();
  const active = new Set<string>();
  const busy = new Set<string>();
  for (const e of events) {
    const type = e.type.toUpperCase();
    if (type.includes("TASK_COMPLETE") && e.subject_id) {
      completed.add(e.subject_id);
      active.delete(e.subject_id);
      if (e.resource_id) busy.delete(e.resource_id);
    } else if (type.includes("TASK_START") && e.subject_id) {
      active.add(e.subject_id);
      if (e.resource_id) busy.add(e.resource_id);
    }
  }
  return {
    completed: completed.size,
    active: active.size,
    started: active.size + completed.size,
    busyResources: busy.size,
  };
}
