/**
 * Experiments: the stored benchmark evidence.
 *
 * Superseded runs are shown, but never as current results. Every row carries
 * its status and the successor it was replaced by, because a run that was
 * withdrawn for being wrong must not read like a live measurement.
 */

import React, { useEffect, useState } from "react";
import { api, type ExperimentDetail } from "../api/client";
import { ErrorBanner, Panel, Spinner, StatusBadge, Table } from "../components/ui";
import type { AppState, Action } from "../state/store";

export function Experiments({
  state,
  dispatch,
}: {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}) {
  const [liveOnly, setLiveOnly] = useState(false);
  const [kind, setKind] = useState<string>("");
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = React.useCallback(async () => {
    setLoading(true);
    dispatch({ type: "busy", label: "loading experiments" });
    try {
      const res = await api.listExperiments(liveOnly, kind || undefined);
      dispatch({ type: "experiments", payload: res.items });
      (window as unknown as { __expCounts?: unknown }).__expCounts = {
        total: res.total,
        live: res.live,
        superseded: res.superseded,
      };
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setLoading(false);
      dispatch({ type: "busy", label: null });
    }
  }, [dispatch, liveOnly, kind]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const open = async (id: string) => {
    setLoading(true);
    dispatch({ type: "busy", label: "loading experiment" });
    try {
      setDetail(await api.getExperiment(id));
    } catch (e) {
      dispatch({ type: "error", message: (e as Error).message });
    } finally {
      setLoading(false);
      dispatch({ type: "busy", label: null });
    }
  };

  const kinds = Array.from(new Set(state.experiments.map((e) => e.kind))).sort();

  return (
    <div className="page">
      <h1>Experiments</h1>
      {state.error && <ErrorBanner message={state.error} onRetry={refresh} />}

      <Panel
        title="Stored experiments"
        subtitle="The backend imports the experiments directory on startup. A superseded run is kept, and marked."
        actions={
          <div className="filter-row">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={liveOnly}
                onChange={(e) => setLiveOnly(e.target.checked)}
              />
              live only
            </label>
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="">all kinds</option>
              {kinds.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
            <button type="button" onClick={() => void refresh()} disabled={loading}>
              {loading ? <Spinner label="loading" /> : "Refresh"}
            </button>
          </div>
        }
      >
        <div className="scroll">
          <Table
            rows={state.experiments}
            rowKey={(e) => e.id}
            onRowClick={(e) => void open(e.id)}
            empty="No experiments stored."
            columns={[
              { key: "id", label: "Experiment" },
              { key: "kind", label: "Benchmark" },
              {
                key: "status",
                label: "Status",
                render: (e) => <StatusBadge status={e.status === "succeeded" ? "OPTIMAL" : e.status} />,
              },
              { key: "row_count", label: "Rows", numeric: true },
              { key: "seed", label: "Seed", numeric: true },
              { key: "commit", label: "Commit" },
              {
                key: "created_at",
                label: "Recorded",
                render: (e) => (e.created_at ? e.created_at.replace("T", " ").slice(0, 19) : "—"),
              },
              {
                key: "superseded_by",
                label: "Superseded by",
                render: (e) => (e.superseded_by ? <code>{e.superseded_by}</code> : <span className="muted">—</span>),
              },
            ]}
          />
        </div>
      </Panel>

      {detail && (
        <Panel
          title={detail.id}
          subtitle={`${detail.kind} · ${detail.row_count} rows · commit ${detail.commit || "unrecorded"}`}
          actions={<StatusBadge status={detail.status === "succeeded" ? "OPTIMAL" : detail.status} />}
        >
          {detail.superseded_by && (
            <p className="note">
              <strong>Superseded by {detail.superseded_by}.</strong> {detail.supersede_reason} This run
              is retained for history and is excluded from the evidence layer.
            </p>
          )}

          <h4>Manifest</h4>
          <pre className="json-block">{JSON.stringify(detail.manifest, null, 2)}</pre>

          <h4>Summary</h4>
          <pre className="json-block">{JSON.stringify(detail.summary, null, 2)}</pre>

          {detail.rows.length > 0 && (
            <>
              <h4>Rows ({detail.rows.length})</h4>
              <div className="scroll tall">
                <Table
                  rows={detail.rows}
                  rowKey={(_, i) => String(i)}
                  empty="No rows."
                  columns={Object.keys(detail.rows[0] ?? {}).map((col) => ({
                    key: col,
                    label: col,
                    render: (r: Record<string, string>) => r[col] ?? "—",
                  }))}
                />
              </div>
            </>
          )}
        </Panel>
      )}
    </div>
  );
}
