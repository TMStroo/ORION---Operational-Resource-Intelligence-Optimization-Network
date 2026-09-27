/**
 * Overview: the state of the work in one screen.
 *
 * Every figure is the backend's. When there is no plan yet, the panel says so
 * rather than showing zeros, because a grid of zeros reads as "everything is
 * fine and nothing is scheduled".
 */

import { Link } from "react-router-dom";
import { Panel, Stat, StatusBadge, clock, num, pct } from "../components/ui";
import type { AppState } from "../state/store";

export function Overview({ state }: { state: AppState }) {
  const { scenario, plan, simulation, disruption, replan, health, busy } = state;

  if (!scenario) {
    return (
      <div className="page">
        <h1>Overview</h1>
        <Panel title="No scenario open">
          <p>
            Open a scenario from the <Link to="/scenarios">Scenarios</Link> page, or create one, to
            begin. The backend is {health ? "reachable" : "not reachable yet"} and holds{" "}
            {health ? `${health.rows.plans ?? 0} plans` : "no data"}.
          </p>
        </Panel>
      </div>
    );
  }

  const completion = plan ? plan.completion : null;
  const utilization = plan
    ? plan.utilization.reduce((sum, u) => sum + u.utilization, 0) / Math.max(1, plan.utilization.length)
    : null;

  return (
    <div className="page">
      <h1>Overview</h1>

      <Panel
        title={scenario.name}
        subtitle={
          <>
            {scenario.id} · horizon {clock(scenario.horizon.start)}–{clock(scenario.horizon.end)} ·
            source {scenario.source}
          </>
        }
        actions={busy && <span className="busy">{busy}</span>}
      >
        <div className="stat-grid">
          <Stat label="Tasks" value={scenario.task_count} />
          <Stat label="Resources" value={scenario.resource_count} />
          <Stat
            label="Completion"
            value={completion === null ? "—" : pct(completion)}
            hint={plan ? `${plan.tasks_assigned} of ${plan.tasks_total} assigned` : "no plan yet"}
            tone={completion === null ? "neutral" : completion > 0.95 ? "good" : "warn"}
          />
          <Stat
            label="Late tasks"
            value={plan ? plan.late_tasks : "—"}
            tone={plan && plan.late_tasks > 0 ? "bad" : "good"}
          />
          <Stat
            label="Objective"
            value={plan ? num(plan.objective) : "—"}
            hint={plan ? `${plan.solver} · ${num(plan.runtime_s, 3)} s` : undefined}
          />
          <Stat
            label="Mean utilisation"
            value={utilization === null ? "—" : pct(utilization)}
            hint={plan ? `across ${plan.utilization.length} resources` : undefined}
          />
          <Stat
            label="Simulated time"
            value={simulation ? clock(simulation.end_time) : "—"}
            hint={simulation ? `${simulation.event_count} events` : "not run"}
          />
          <Stat
            label="Latest disruption"
            value={disruption ? disruption.disruption.type : "—"}
            hint={
              disruption
                ? `${disruption.disruption.severity} · ${disruption.disruption.affected_tasks.length} tasks affected`
                : "none applied"
            }
            tone={disruption ? "warn" : "neutral"}
          />
        </div>

        <div className="status-line">
          <span>Plan status</span>
          {plan ? <StatusBadge status={plan.status} /> : <span className="muted">no plan</span>}
          {plan?.violations.length ? (
            <span className="violations">
              {plan.violations.length} violation{plan.violations.length === 1 ? "" : "s"}
            </span>
          ) : null}
        </div>
      </Panel>

      {replan && (
        <Panel
          title="Recovery state"
          subtitle="Local repair and full re-optimization are both reported; neither is presented as the answer."
        >
          <div className="stat-grid">
            <Stat label="Repair runtime" value={`${num(replan.repair_seconds, 3)} s`} />
            <Stat
              label="Full re-optimization"
              value={replan.full_plan ? `${num(replan.full_seconds, 3)} s` : replan.full_status}
              hint={replan.full_reason}
            />
            <Stat
              label="Plan preserved"
              value={pct(replan.repair_preserved_ratio, 0)}
              hint="share of baseline assignments left untouched"
            />
            <Stat
              label="Recovery"
              value={pct(replan.recovery_pct, 0)}
              hint="repair quality relative to full re-optimization"
              tone={replan.recovery_pct >= 0.98 ? "good" : "warn"}
            />
          </div>
        </Panel>
      )}

      <Panel title="Next steps">
        <ol className="steps">
          <li>
            <Link to="/planning">Inspect the plan</Link> — assignments, timeline, utilisation.
          </li>
          <li>
            <Link to="/simulation">Run the simulation</Link> — the discrete-event engine on this plan.
          </li>
          <li>
            <Link to="/disruptions">Apply a disruption</Link> and see what it puts at risk.
          </li>
          <li>
            <Link to="/replanning">Compare local repair against full re-optimization</Link>.
          </li>
          <li>
            <Link to="/what-if">Ask what-if questions</Link> about the same baseline.
          </li>
          <li>
            <Link to="/experiments">Read the benchmark evidence</Link>.
          </li>
        </ol>
      </Panel>
    </div>
  );
}
