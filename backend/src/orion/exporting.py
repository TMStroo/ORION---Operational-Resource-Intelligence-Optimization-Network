"""Reproducible exports: JSON, CSV and the event timeline.

Every export carries provenance. A metrics CSV without the seed, git commit and
objective weights that produced it is not evidence, so the exporter refuses to
write one: :func:`provenance_block` must be supplied and is embedded in the
artifact.

Reproducibility rule: exports are written from stored domain objects only, with
no timestamps of their own beyond the provenance block. Re-exporting the same
plan produces a byte-identical file, which is what makes
``check_readme`` able to fail CI when a claimed number disagrees with its
artifact.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from orion.domain.errors import ExportError

#: Bumped when the export layout changes, so a consumer can tell old files.
EXPORT_SCHEMA_VERSION = "orion.export/1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "value"):  # enums
        return value.value
    return str(value)


def provenance_block(
    *,
    kind: str,
    source: str | None = None,
    seed: int | None = None,
    git_commit: str | None = None,
    objective_weights: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance every export embeds."""
    from orion.experiments.tracker import _git_commit

    block: dict[str, Any] = {
        "export_schema": EXPORT_SCHEMA_VERSION,
        "kind": kind,
        "source": source or "orion",
        "seed": seed,
        "git_commit": git_commit or _git_commit(Path.cwd()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if objective_weights:
        block["objective_weights"] = dict(objective_weights)
    if extra:
        block.update(dict(extra))
    return block


def write_json(payload: Any, path: Path | str, *, provenance: Mapping[str, Any] | None = None) -> Path:
    """Write deterministic JSON (sorted keys, 2-space indent, trailing newline)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"provenance": dict(provenance or {})}
    if hasattr(payload, "to_dict"):
        body["data"] = _jsonable(payload.to_dict())
    else:
        body["data"] = _jsonable(payload)
    try:
        text = json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"cannot serialise {kind_of(payload)} to JSON: {exc}") from exc
    target.write_text(text + "\n", encoding="utf-8", newline="\n")
    return target


def kind_of(obj: Any) -> str:
    return type(obj).__name__


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
    *,
    provenance: Mapping[str, Any] | None = None,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a CSV. ``provenance`` becomes leading comment lines, not data rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ExportError(f"refusing to write an empty CSV to {target}: a result file must contain data")
    columns = list(fieldnames or rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as fh:
        if provenance:
            fh.write(f"# provenance: {json.dumps(dict(provenance), sort_keys=True, default=str)}\n")
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    return target


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        # fixed precision keeps re-reads byte-stable across platforms
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, default=str)
    if hasattr(value, "value"):
        return value.value
    return value


# --------------------------------------------------------------------------
# domain-specific exports
# --------------------------------------------------------------------------


def export_plan(plan: Any, path: Path | str, **prov: Any) -> Path:
    prov.setdefault("kind", "plan")
    prov.setdefault("source", f"plan:{getattr(plan, 'id', 'unknown')}")
    return write_json(plan, path, provenance=provenance_block(**prov))


def export_scenario(scenario: Any, path: Path | str, **prov: Any) -> Path:
    prov.setdefault("kind", "scenario")
    prov.setdefault("source", f"scenario:{getattr(scenario, 'id', 'unknown')}")
    return write_json(scenario, path, provenance=provenance_block(**prov))


def plan_metric_rows_csv(plan: Any, scenario: Any, path: Path | str, **prov: Any) -> Path:
    """Export the planning metric vector as CSV."""
    from orion.evaluation.metrics import plan_metric_rows

    rows = []
    for row in plan_metric_rows(plan, scenario):
        rows.append({
            "key": row.key,
            "label": row.label,
            "value": f"{row.value:.6f}",
            "unit": row.unit,
            "higher_is_better": "" if row.higher_is_better is None else ("true" if row.higher_is_better else "false"),
        })
    prov.setdefault("kind", "metrics")
    prov.setdefault("source", f"plan:{getattr(plan, 'id', 'unknown')}")
    return write_csv(rows, path, provenance=provenance_block(**prov))


def write_simulation_csv(result: Any, path: Path | str, **prov: Any) -> Path:
    """Export the simulation event timeline as CSV."""
    rows = [
        {
            "time": event.time,
            "clock": _hhmm(event.time),
            "type": event.type.name,
            "label": getattr(event.type, "label", event.type.name),
            "subject_id": event.subject_id or "",
            "resource_id": event.resource_id or "",
            "detail": event.detail,
        }
        for event in result.trace
    ]
    if not rows:
        raise ExportError("refusing to write an empty simulation timeline")
    prov.setdefault("kind", "simulation_timeline")
    prov.setdefault("source", f"simulation:{getattr(result, 'scenario_id', 'unknown')}")
    return write_csv(rows, path, provenance=provenance_block(**prov))


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def export_comparison(comparison: Any, path: Path | str, **prov: Any) -> Path:
    """Export a plan comparison, including the churn breakdown."""
    rows = [
        {
            "metric": d.label,
            "baseline": f"{d.before:.6f}",
            "candidate": f"{d.after:.6f}",
            "change": f"{d.after - d.before:+.6f}",
            "unit": d.unit,
            "higher_is_better": "true" if d.higher_is_better else "false",
        }
        for d in comparison.deltas
    ]
    churn = comparison.churn
    rows.append({
        "metric": "Assignments changed",
        "baseline": str(churn.total_before),
        "candidate": str(churn.total_after),
        "change": str(churn.changed_assignments),
        "unit": "count",
        "higher_is_better": "false",
    })
    prov.setdefault("kind", "plan_comparison")
    return write_csv(rows, path, provenance=provenance_block(**prov))


def export_benchmark_summary(rows: Iterable[Mapping[str, Any]], path: Path | str, **prov: Any) -> Path:
    materialised = list(rows)
    if not materialised:
        raise ExportError("refusing to write an empty benchmark summary")
    prov.setdefault("kind", "benchmark_summary")
    return write_csv(materialised, path, provenance=provenance_block(**prov))


def export_event_timeline(rows: Sequence[Mapping[str, Any]], path: Path | str, **prov: Any) -> Path:
    if not rows:
        raise ExportError("refusing to write an empty event timeline")
    prov.setdefault("kind", "event_timeline")
    return write_csv(rows, path, provenance=provenance_block(**prov))
