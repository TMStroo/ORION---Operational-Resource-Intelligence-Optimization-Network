"""Generate report figures from live experiment artifacts.

Every figure reads through :class:`orion.evidence.Evidence`, so a figure cannot
show a result the project has superseded and cannot drift from the tables in the
report. A figure that cannot be produced because its suite is missing is
skipped and reported, never faked with placeholder data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orion.evidence import Evidence
from orion.figures import (
    figure_constraint_pressure,
    figure_disruption_stress,
    figure_failures,
    figure_quality_vs_runtime,
    figure_recovery,
    figure_repair_vs_full,
    figure_runtime_vs_size,
    figure_solver_comparison,
    figure_whatif,
)

# README shows a small subset; the report embeds the full set.
README_FIGURES = [
    "03_runtime_scaling.png",
    "04_quality_vs_runtime.png",
    "06_repair_vs_full.png",
    "08_constraint_pressure.png",
    "09_disruption_stress.png",
]


def generate_figures(
    project_root: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Render every figure the live evidence supports.

    Returns a report of what was written and what was skipped, so a caller can
    surface gaps rather than discovering them from a missing file.
    """
    root = Path(project_root)
    evidence = Evidence.load(root)
    out = Path(output_dir) if output_dir else root / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: dict[str, str] = {}

    def attempt(name: str, kind: str, fn) -> None:
        if name in written:
            return
        try:
            fn(out / name)
            written.append(name)
        except Exception as exc:  # a missing suite must be visible, not fatal
            skipped[name] = f"{kind}: {type(exc).__name__}: {exc}"

    # --- 01/02: solver comparison -------------------------------------
    if evidence.has("solver_comparison"):
        rows = evidence.get("solver_comparison").rows
        # One budget, so the bars compare solvers rather than budgets.
        budgets = sorted({r.get("time_budget_s", "") for r in rows}, key=float)
        preferred = budgets[-1] if budgets else ""
        comparison = [r for r in rows if r.get("time_budget_s") == preferred]
        attempt(
            "01_solver_comparison.png",
            "solver_comparison",
            lambda p: figure_solver_comparison(comparison, p, title="Solver comparison (public benchmarks)"),
        )
        # Synthetic suite too, if present in scalability
    if evidence.has("solver_comparison"):
        rows = evidence.get("solver_comparison").rows
        synthetic = [r for r in rows if r.get("dataset") not in ("solomon", "jobshop")]
        if synthetic:
            attempt(
                "02_solver_comparison_synthetic.png",
                "solver_comparison (synthetic)",
                lambda p: figure_solver_comparison(synthetic, p, title="Solver comparison (synthetic scenarios)"),
            )

    # --- 03/04: scalability -------------------------------------------
    if evidence.has("scalability"):
        rows = evidence.scalability()
        attempt(
            "03_runtime_scaling.png",
            "scalability",
            lambda p: figure_runtime_vs_size(
                [{**r, "num_tasks": r["tasks"]} for r in rows],
                p,
                title="Solver runtime versus problem size",
            ),
        )
        attempt(
            "04_quality_vs_runtime.png",
            "scalability",
            lambda p: figure_quality_vs_runtime(
                rows, p, title="Solution quality versus runtime (scalability ladder)"
            ),
        )
    else:
        skipped["03_runtime_scaling.png"] = "no live scalability experiment"
        skipped["04_quality_vs_runtime.png"] = "no live scalability experiment"

    # --- 05: public benchmark quality ---------------------------------
    if evidence.has("solver_comparison"):
        public = [
            {
                "solver": r["solver"],
                "objective": r["objective"] or 0.0,
                "runtime_s": r["runtime_s"] or 0.0,
                "status": r["status"],
            }
            for r in evidence.public_benchmarks()
            if r["dataset"] == "solomon" and r["budget_s"] == max(
                (x["budget_s"] for x in evidence.public_benchmarks() if x["dataset"] == "solomon"),
                key=float,
                default="",
            )
        ]
        if public:
            attempt(
                "05_public_benchmark.png",
                "solver_comparison (solomon)",
                lambda p: figure_quality_vs_runtime(public, p, title="Solomon c101: quality versus runtime"),
            )

    # --- 06: repair vs full -------------------------------------------
    if evidence.has("disruption_stress"):
        rows = evidence.get("disruption_stress").rows
        grouped: dict[str, dict[str, float]] = {}
        for r in rows:
            strategy = r.get("strategy", "")
            if strategy == "none":
                continue
            bucket = grouped.setdefault(strategy, {"recovered_pct": 0.0, "replan_seconds": 0.0, "n": 0.0})
            bucket["recovered_pct"] += float(r.get("recovery_pct") or 0.0)
            bucket["replan_seconds"] += float(r.get("replan_seconds") or 0.0)
            bucket["n"] += 1
        averages = [
            {
                "recovered_pct": g["recovered_pct"] / g["n"] if g["n"] else 0.0,
                "replan_seconds": g["replan_seconds"] / g["n"] if g["n"] else 0.0,
            }
            for _, g in sorted(grouped.items(), key=lambda kv: -kv[0].count("repair"))
        ]
        if averages:
            attempt(
                "06_repair_vs_full.png",
                "disruption_stress",
                lambda p: figure_repair_vs_full(
                    averages, p, title="Local repair vs full re-optimization (mean over all disruptions)"
                ),
            )
    else:
        skipped["06_repair_vs_full.png"] = "no live disruption_stress experiment"

    # --- 07: recovery by severity -------------------------------------
    if evidence.has("disruption_stress"):
        summary = evidence.disruption_recovery()
        stages = [
            (k.split("/", 1)[1], v)  # severity -> mean recovery
            for k, v in summary.get("by_strategy_severity", {}).items()
            if k.startswith("local_repair/")
        ]
        if stages:
            attempt(
                "07_recovery.png",
                "disruption_stress",
                lambda p: figure_recovery(stages, p, title="Service recovery by disruption severity (local repair)"),
            )
    else:
        skipped["07_recovery.png"] = "no live disruption_stress experiment"

    # --- 08: constraint pressure --------------------------------------
    if evidence.has("constraint_pressure"):
        rows = evidence.get("constraint_pressure").rows
        attempt(
            "08_constraint_pressure.png",
            "constraint_pressure",
            lambda p: figure_constraint_pressure(rows, p, title="Service level under resource scarcity and deadline pressure"),
        )
    else:
        skipped["08_constraint_pressure.png"] = "no live constraint_pressure experiment"

    # --- 09: disruption stress ----------------------------------------
    if evidence.has("disruption_stress"):
        rows = evidence.get("disruption_stress").rows
        attempt(
            "09_disruption_stress.png",
            "disruption_stress",
            lambda p: figure_disruption_stress(rows, p, title="Service level by disruption rate and recovery strategy"),
        )
    else:
        skipped["09_disruption_stress.png"] = "no live disruption_stress experiment"

    # --- 10: what-if (from the demo artifacts) -------------------------
    whatif_csv = root / "results" / "demo" / "whatif_comparison.csv"
    if whatif_csv.is_file():
        import csv as _csv
        import io as _io

        rows = list(_csv.DictReader(_io.StringIO(whatif_csv.read_text(encoding="utf-8"))))
        attempt(
            "10_whatif.png",
            "demo what-if",
            lambda p: figure_whatif(rows, p, title="What-if analysis: baseline versus modified scenario"),
        )
    else:
        skipped["10_whatif.png"] = "results/demo/whatif_comparison.csv not present"

    # --- 11: failure taxonomy -----------------------------------------
    # Failure records live in metrics.json, not summary.json - summary.json holds
    # only the numeric roll-up, so reading it found nothing and skipped the
    # figure. The severity key there is `worst_severity`, per `aggregate_failures`.
    failure_rows: list[dict[str, Any]] = []
    for kind, suite in evidence.suites.items():
        metrics_path = suite.directory / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for category, info in ((metrics.get("failures") or {})).items():
            if not isinstance(info, dict):
                continue
            failure_rows.append(
                {
                    "category": str(category),
                    "count": int(info.get("count", 0) or 0),
                    "severity": int(info.get("worst_severity", 1) or 1),
                    "suite": kind,
                }
            )
    if failure_rows:
        attempt(
            "11_failure_taxonomy.png",
            "failure taxonomy",
            lambda p: figure_failures(failure_rows, p, title="Observed failure categories"),
        )
    else:
        skipped["11_failure_taxonomy.png"] = "no failure records in the live summaries"

    return {
        "output_dir": out,
        "written": sorted(written),
        "skipped": skipped,
        "integrity_problems": evidence.integrity_problems(),
    }
