/**
 * Plan comparison: every recorded comparison involving a plan.
 *
 * Reads `/plans/{id}/comparison`, which returns the comparisons the backend
 * stored. A plan nobody compared against is a real state and is shown as such.
 */

import React, { useEffect, useState } from "react";
import { api, type ComparisonView, type PlanComparisonResponse } from "../api/client";
import { ErrorBanner, Panel, Spinner, Stat, Table, num, pct } from "../components/ui";
import type { AppState, Action } from "../state/store";

export function Comparison({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const [data, setData] = useState<PlanComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const { activePlanId, plan } = state;

  useEffect(() => {
    if (!activePlanId) {
      setData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    dispatch({ type: "busy", label: "loading comparisons" });
    api
      .planComparison(activePlanId)
      .then((res) => {
        if (cancelled) return;
        setData(res);
        setSelected(res.items[0]?.kind ?? null);
      })
      .catch((e) => {
        if (!cancelled) dispatch({ type: "error", message: (e as Error).message });
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          dispatch({ type: "busy", label: null });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activePlanId, dispatch]);

  if (!activePlanId) {
    return (
      <div className="page">
        <h1>Plan comparison</h1>
        <Panel title="No plan selected">
          <p>Open a plan on the Planning page to compare it against the plans it was compared to.</p>
        </Panel>
      </div>
    );
  }

  const item = data?.items.find((i) => i.kind === selected) ?? data?.items[0] ?? null;

  return (
    <div className="page">
      <h1>Plan comparison</h1>
      {state.error && <ErrorBanner message={state.error} />}

      <Panel
        title={`Comparisons for ${plan?.label || activePlanId}`}
        actions={loading ? <Spinner label="loading" /> : null}
      >
        {data && data.items.length > 0 ? (
          <div className="tab-row">
            {data.items.map((i) => (
              <button
                key={i.kind}
                type="button"
                className={i.kind === item?.kind ? "tab active" : "tab"}
                onClick={() => setSelected(i.kind)}
              >
                {i.kind}
              </button>
            ))}
          </div>
        ) : (
          <p className="empty">Loading comparisons…</p>
        )}
      </Panel>

      {item && <ComparisonDetail item={item} />}

      {data && Object.keys(data.metrics).length > 0 && (
        <Panel
          title="Metric matrix"
          subtitle="Flattened by the backend so a client can render a table without walking the comparison list."
        >
          <div className="scroll">
            <Table
              rows={Object.entries(data.metrics).flatMap(([kind, bucket]) =>
                Object.entries(bucket).map(([metric, value]) => ({ kind, metric, value })),
              )}
              rowKey={(r) => `${r.kind}:${r.metric}`}
              empty="No metrics."
              columns={[
                { key: "kind", label: "Comparison" },
                { key: "metric", label: "Metric" },
                { key: "value", label: "Value", render: (r) => num(r.value, 2), numeric: true },
              ]}
            />
          </div>
        </Panel>
      )}
    </div>
  );
}

function ComparisonDetail({ item }: { item: ComparisonView }) {
  const isNone = item.kind === "none";
  return (
    <>
      <Panel
        title={isNone ? "No recorded comparison" : item.kind}
        subtitle={
          isNone
            ? "Nothing has been compared against this plan yet. That is a real state, not a failure."
            : `${item.baseline_plan_id} → ${item.candidate_plan_id}`
        }
      >
        {isNone ? (
          <p className="note">
            Run a what-if, a local repair, or a full re-optimization and a comparison will appear
            here.
          </p>
        ) : (
          <>
            <div className="stat-grid">
              <Stat label="Churn" value={item.churn.changed_assignments} hint="assignments that changed" />
              <Stat label="Churn ratio" value={pct(item.churn.churn_ratio)} />
              <Stat label="Before" value={item.churn.total_before} />
              <Stat label="After" value={item.churn.total_after} />
              <Stat
                label="Newly assigned"
                value={item.newly_assigned.length}
                tone="good"
              />
              <Stat
                label="Removed"
                value={item.removed_assignments.length}
                tone={item.removed_assignments.length ? "warn" : "neutral"}
              />
              <Stat
                label="Delayed"
                value={item.delayed.length}
                tone={item.delayed.length ? "warn" : "neutral"}
              />
              <Stat
                label="Newly violated"
                value={item.newly_violated.length}
                tone={item.newly_violated.length ? "bad" : "good"}
              />
            </div>

            {item.deltas.length > 0 && (
              <>
                <h4>Metric deltas</h4>
                <Table
                  rows={item.deltas}
                  rowKey={(d) => d.name}
                  empty="No deltas."
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
                          <span
                            className={
                              Math.abs(d.change) < 1e-9 ? "muted" : worse ? "late" : "good-text"
                            }
                          >
                            {d.change > 0 ? "+" : ""}
                            {num(d.change, 2)}
                            {d.unit ? ` ${d.unit}` : ""}
                          </span>
                        );
                      },
                      numeric: true,
                    },
                    { key: "higher_is_better", label: "Better when", render: (d) => (d.higher_is_better ? "higher" : "lower") },
                  ]}
                />
              </>
            )}

            {item.newly_assigned.length > 0 && (
              <>
                <h4>Newly assigned</h4>
                <div className="tag-row">
                  {item.newly_assigned.map((t) => (
                    <span key={t} className="tag good-tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            )}
            {item.removed_assignments.length > 0 && (
              <>
                <h4>Removed</h4>
                <div className="tag-row">
                  {item.removed_assignments.map((t) => (
                    <span key={t} className="tag warn-tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            )}
            {item.newly_violated.length > 0 && (
              <>
                <h4>Newly violated</h4>
                <div className="tag-row">
                  {item.newly_violated.map((t) => (
                    <span key={t} className="tag bad-tag">
                      {t}
                    </span>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </Panel>
    </>
  );
}
