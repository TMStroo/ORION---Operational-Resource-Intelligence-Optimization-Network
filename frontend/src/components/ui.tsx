/**
 * Shared presentational components.
 *
 * The status badge is the important one: it is the single place a solver status
 * becomes a colour and a word, and it distinguishes proven optimality from a
 * relaxation. A relaxation gets its own colour and an explicit tooltip, so it
 * can never be mistaken for an exact result at a glance.
 */

import React from "react";

export function Panel({
  title,
  subtitle,
  actions,
  children,
}: {
  title?: string;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      {title && (
        <header className="panel-header">
          <div>
            <h2>{title}</h2>
            {subtitle && <p className="panel-subtitle">{subtitle}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  return (
    <div className={`stat stat-${tone}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

/**
 * Status -> CSS class suffix. The stylesheet has one rule per status so a new
 * status falls through to the neutral tone rather than inheriting another
 * status's colour.
 */
const STATUS_TONE: Record<string, "optimal" | "feasible" | "warn" | "bad" | "relaxation_optimal" | "not_applicable" | "unknown"> = {
  OPTIMAL: "optimal",
  FEASIBLE: "feasible",
  RELAXATION_OPTIMAL: "relaxation_optimal",
  TIME_LIMIT: "warn",
  INFEASIBLE: "bad",
  ERROR: "bad",
  NOT_APPLICABLE: "not_applicable",
};

/**
 * Render a solver status verbatim.
 *
 * RELAXATION_OPTIMAL is deliberately a different colour from OPTIMAL and
 * carries a tooltip saying the result is optimal for a relaxation. Collapsing
 * the two would be the single most misleading thing this UI could do.
 */
export function StatusBadge({ status, title }: { status: string; title?: string }) {
  const tone = STATUS_TONE[status] ?? "unknown";
  const hint =
    title ??
    (status === "RELAXATION_OPTIMAL"
      ? "Optimal for a relaxation of the problem, not an exact solution"
      : status === "TIME_LIMIT"
        ? "The time budget expired. The plan is usable but not proven optimal."
        : status === "NOT_APPLICABLE"
          ? "The solver declined this instance; it is not a failure"
          : undefined);
  return (
    <span className={`badge s-${tone}`} title={hint}>
      {status}
    </span>
  );
}

export function Latency({ ms }: { ms: number }) {
  return <span className="latency">{ms.toFixed(0)} ms</span>;
}

export function Bar({ value, max, tone = "accent" }: { value: number; max: number; tone?: string }) {
  const pct = max > 0 ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
  return (
    <div className="bar" role="presentation">
      <div className={`bar-fill bar-${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  empty = "No rows.",
  onRowClick,
  highlight,
}: {
  columns: { key: string; label: string; render?: (row: T) => React.ReactNode; numeric?: boolean }[];
  rows: T[];
  rowKey: (row: T, index: number) => string;
  empty?: string;
  onRowClick?: (row: T) => void;
  highlight?: (row: T) => boolean;
}) {
  if (rows.length === 0) return <p className="empty">{empty}</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "num" : undefined}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={rowKey(row, i)}
              className={[onRowClick ? "clickable" : "", highlight?.(row) ? "highlight" : ""]
                .filter(Boolean)
                .join(" ")}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "num" : undefined}>
                  {c.render ? c.render(row) : String((row as Record<string, unknown>)[c.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="banner banner-error" role="alert">
      <div>
        <strong>Request failed.</strong> {message}
      </div>
      {onRetry && (
        <button type="button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {children && <div>{children}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="spinner" role="status">
      <span className="spinner-dot" />
      {label && <span>{label}</span>}
    </div>
  );
}

/** Format an objective or cost. Fixed precision so columns line up. */
export function num(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function pct(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Minutes-since-midnight as HH:MM. */
export function clock(minutes: number): string {
  if (!Number.isFinite(minutes)) return "--:--";
  const m = ((Math.round(minutes) % 1440) + 1440) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}
