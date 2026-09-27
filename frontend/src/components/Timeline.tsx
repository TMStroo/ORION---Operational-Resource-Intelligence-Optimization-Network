/**
 * Timeline visualisation for a plan.
 *
 * Drawn from the plan's own assignments: one row per resource, one bar per
 * assignment, positioned by its real start and end. Nothing is smoothed or
 * rounded into looking better than it is - a bar that starts late and ends late
 * looks exactly that way.
 */

import type { AssignmentView, ScenarioDetail } from "../api/client";
import { clock, pct } from "./ui";

const LATE = "bar-late";
const OK = "bar-ok";
const TRAVEL = "bar-travel";

interface Props {
  scenario: ScenarioDetail;
  assignments: AssignmentView[];
  highlight?: Set<string>;
  onSelect?: (taskId: string) => void;
  selected?: string | null;
}

export function Timeline({ scenario, assignments, highlight, onSelect, selected }: Props) {
  if (assignments.length === 0) {
    return <p className="empty">This plan assigns no tasks, so there is no timeline to draw.</p>;
  }

  const horizonStart = scenario.horizon.start;
  const horizonEnd = Math.max(
    scenario.horizon.end,
    ...assignments.map((a) => a.end),
  );
  const span = Math.max(1, horizonEnd - horizonStart);

  const byResource = new Map<string, AssignmentView[]>();
  for (const a of assignments) {
    const list = byResource.get(a.resource_id) ?? [];
    list.push(a);
    byResource.set(a.resource_id, list);
  }
  for (const list of byResource.values()) list.sort((x, y) => x.start - y.start);

  const ticks = buildTicks(horizonStart, horizonEnd, 8);

  return (
    <div className="timeline">
      <div className="timeline-axis">
        <div className="timeline-label-col" />
        <div className="timeline-track">
          {ticks.map((t) => (
            <span key={t} className="tick" style={{ left: `${((t - horizonStart) / span) * 100}%` }}>
              {clock(t)}
            </span>
          ))}
        </div>
      </div>

      {[...byResource.entries()].map(([resourceId, list]) => (
        <div key={resourceId} className="timeline-row">
          <div className="timeline-label-col" title={resourceId}>
            {resourceId}
          </div>
          <div className="timeline-track">
            {/* Travel legs, so the gaps between tasks are visible as movement
                rather than looking like idle time. */}
            {list.map((a, i) => {
              const prev = i === 0 ? null : list[i - 1];
              if (!prev) return null;
              const from = prev.end;
              const travel = a.travel_before;
              if (travel <= 0) return null;
              return (
                <div
                  key={`t-${a.task_id}`}
                  className={`bar ${TRAVEL}`}
                  title={`Travel ${travel} min to ${a.location}`}
                  style={{
                    left: `${((from - horizonStart) / span) * 100}%`,
                    width: `${(travel / span) * 100}%`,
                  }}
                />
              );
            })}
            {list.map((a) => {
              const isLate = a.lateness > 0;
              const isSel = selected === a.task_id;
              const isHi = highlight?.has(a.task_id);
              return (
                <button
                  type="button"
                  key={a.task_id}
                  className={`bar ${isLate ? LATE : OK} ${isSel ? "bar-selected" : ""} ${isHi ? "bar-highlight" : ""}`}
                  style={{
                    left: `${((a.start - horizonStart) / span) * 100}%`,
                    width: `${Math.max(0.4, ((a.end - a.start) / span) * 100)}%`,
                  }}
                  title={
                    `${a.task_id} on ${a.resource_id}\n` +
                    `${clock(a.start)}-${clock(a.end)} (${a.end - a.start} min)\n` +
                    `location ${a.location}, priority ${a.priority}` +
                    (isLate ? `\nLATE by ${a.lateness} min` : "")
                  }
                  onClick={() => onSelect?.(a.task_id)}
                >
                  <span className="bar-text">{a.task_id}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      <p className="timeline-legend">
        <span className={`legend-swatch ${OK}`} /> on time
        <span className={`legend-swatch ${LATE}`} /> late
        <span className={`legend-swatch ${TRAVEL}`} /> travel
      </p>
    </div>
  );
}

function buildTicks(start: number, end: number, count: number): number[] {
  const step = Math.max(1, Math.round((end - start) / count));
  const ticks: number[] = [];
  for (let t = start; t <= end; t += step) ticks.push(t);
  return ticks;
}

/**
 * Utilisation per resource, as measured against the resource's own shift.
 *
 * The denominator is the shift length, not the horizon: a resource idle at 3am
 * is not under-used, it is off shift, and dividing by the horizon would make
 * every team look permanently 70% idle.
 */
export function UtilisationBars({
  scenario,
  utilization,
}: {
  scenario: ScenarioDetail;
  utilization: { resource_id: string; kind: string; assigned_tasks: number; worked_minutes: number; utilization: number; travel_km: number }[];
}) {
  if (utilization.length === 0) return <p className="empty">No utilisation data.</p>;
  const byId = new Map(scenario.resources.map((r) => [r.id, r]));
  return (
    <div className="util-grid">
      {utilization.map((u) => {
        const res = byId.get(u.resource_id);
        const shift = res ? Math.max(1, res.shift_end - res.shift_start) : 1;
        const ratio = Math.min(1, u.worked_minutes / shift);
        return (
          <div key={u.resource_id} className="util-row">
            <div className="util-name">
              {u.resource_id}
              <span className="util-kind">{u.kind}</span>
            </div>
            <div className="util-bar">
              <div
                className={`util-fill ${ratio > 0.95 ? "util-hot" : ""}`}
                style={{ width: `${ratio * 100}%` }}
              />
            </div>
            <div className="util-figures">
              {u.assigned_tasks} tasks · {u.worked_minutes} min · {pct(ratio, 0)} of shift
              {u.travel_km > 0 && <> · {u.travel_km.toFixed(1)} km</>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
