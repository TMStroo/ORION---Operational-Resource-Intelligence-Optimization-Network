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
    figure_schedule,
    figure_solver_comparison,
    figure_utilization_rate,
    figure_whatif,
)

#: The one authoritative figure inventory. Every figure the project claims to
#: have lives here, in order, with the file name the README and the report link
#: to. The checker, the report builder and the README all read this list rather
#: than hard-coding their own numbering, which is how the numbering drifted
#: apart from the docs in the first place.
FIGURE_INVENTORY: list[dict[str, str]] = [
    {"file": "01_baseline_schedule.png", "title": "Baseline schedule",
     "source": "results/demo/baseline_plan.json"},
    {"file": "02_resource_utilization.png", "title": "Resource utilisation",
     "source": "results/demo/baseline_plan.json + results/demo/scenario.json"},
    {"file": "03_solver_comparison.png", "title": "Solver comparison",
     "source": "solver_comparison"},
    {"file": "04_runtime_scaling.png", "title": "Runtime scaling",
     "source": "scalability"},
    {"file": "05_quality_vs_runtime.png", "title": "Quality versus runtime",
     "source": "scalability"},
    {"file": "06_constraint_pressure.png", "title": "Constraint pressure",
     "source": "constraint_pressure"},
    {"file": "07_disruption_stress.png", "title": "Disruption impact",
     "source": "disruption_stress"},
    {"file": "08_repair_vs_full.png", "title": "Local repair versus full re-optimization",
     "source": "disruption_stress"},
    {"file": "09_recovery.png", "title": "Recovery comparison",
     "source": "disruption_stress"},
    {"file": "10_whatif.png", "title": "What-if analysis",
     "source": "results/demo/whatif_comparison.csv"},
    {"file": "11_failure_taxonomy.png", "title": "Failure taxonomy",
     "source": "experiments/*/metrics.json"},
]

#: The subset promoted into the README, straight after the opening sections.
README_FIGURES = [
    "01_baseline_schedule.png",
    "05_quality_vs_runtime.png",
    "04_runtime_scaling.png",
    "08_repair_vs_full.png",
    "10_whatif.png",
]


def figure_names() -> list[str]:
    """Every file name the project claims, in inventory order."""
    return [entry["file"] for entry in FIGURE_INVENTORY]


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

    # --- 01/02: the product's own baseline plan -------------------------
    # These come from the demo's persisted plan rather than from a benchmark
    # suite, because they depict the product's actual output: the schedule a
    # user is handed and how hard each resource worked. The plan is reloaded
    # through orion.importing, i.e. the same path a user takes when importing
    # an export, so the figure cannot show a plan the product could not
    # actually reload.
    demo_plan = root / "results" / "demo" / "baseline_plan.json"
    demo_scenario = root / "results" / "demo" / "scenario.json"
    if demo_plan.is_file():
        def _plan_figures() -> None:
            from orion.importing import load_plan, load_scenario

            plan, _prov = load_plan(demo_plan)
            attempt(
                "01_baseline_schedule.png",
                "demo baseline plan",
                lambda p: figure_schedule(plan, p, title="Baseline schedule"),
            )
            if demo_scenario.is_file():
                scenario, _sprov = load_scenario(demo_scenario)
                attempt(
                    "02_resource_utilization.png",
                    "demo baseline plan + scenario",
                    lambda p: figure_utilization_rate(plan, scenario, p),
                )
            else:
                skipped["02_resource_utilization.png"] = (
                    f"{demo_scenario.name} not present, so availability is unknown"
                )

        try:
            _plan_figures()
        except Exception as exc:  # a missing demo must be visible, not fatal
            skipped["01_baseline_schedule.png"] = f"demo plan: {type(exc).__name__}: {exc}"
            skipped["02_resource_utilization.png"] = f"demo plan: {type(exc).__name__}: {exc}"
    else:
        skipped["01_baseline_schedule.png"] = "results/demo/baseline_plan.json not present"
        skipped["02_resource_utilization.png"] = "results/demo/baseline_plan.json not present"

    # --- 03: solver comparison -----------------------------------------
    if evidence.has("solver_comparison"):
        rows = evidence.get("solver_comparison").rows
        # One budget, so the bars compare solvers rather than budgets.
        budgets = sorted({r.get("time_budget_s", "") for r in rows}, key=float)
        preferred = budgets[-1] if budgets else ""
        comparison = [r for r in rows if r.get("time_budget_s") == preferred]
        attempt(
            "03_solver_comparison.png",
            "solver_comparison",
            lambda p: figure_solver_comparison(comparison, p, title="Solver comparison"),
        )
    else:
        skipped["03_solver_comparison.png"] = "no live solver_comparison experiment"

    # --- 04/05: scalability -------------------------------------------
    if evidence.has("scalability"):
        rows = evidence.scalability()
        attempt(
            "04_runtime_scaling.png",
            "scalability",
            lambda p: figure_runtime_vs_size(
                [{**r, "num_tasks": r["tasks"]} for r in rows],
                p,
                title="Solver runtime versus problem size",
            ),
        )
        attempt(
            "05_quality_vs_runtime.png",
            "scalability",
            lambda p: figure_quality_vs_runtime(
                rows, p, title="Solution quality versus runtime"
            ),
        )
    else:
        skipped["04_runtime_scaling.png"] = "no live scalability experiment"
        skipped["05_quality_vs_runtime.png"] = "no live scalability experiment"

    # --- 08: local repair versus full re-optimization ------------
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
                "08_repair_vs_full.png",
                "disruption_stress",
                lambda p: figure_repair_vs_full(
                    averages, p, title="Local repair vs full re-optimization"
                ),
            )
    else:
        skipped["08_repair_vs_full.png"] = "no live disruption_stress experiment"

    # --- 09: recovery by severity -------------------------------------
    if evidence.has("disruption_stress"):
        summary = evidence.disruption_recovery()
        stages = [
            (k.split("/", 1)[1], v)  # severity -> mean recovery
            for k, v in summary.get("by_strategy_severity", {}).items()
            if k.startswith("local_repair/")
        ]
        if stages:
            attempt(
                "09_recovery.png",
                "disruption_stress",
                lambda p: figure_recovery(stages, p, title="Recovery by disruption severity (local repair)"),
            )
    else:
        skipped["09_recovery.png"] = "no live disruption_stress experiment"

    # --- 06: constraint pressure -------------------------------------
    if evidence.has("constraint_pressure"):
        rows = evidence.get("constraint_pressure").rows
        attempt(
            "06_constraint_pressure.png",
            "constraint_pressure",
            lambda p: figure_constraint_pressure(rows, p, title="Service level under constraint pressure"),
        )
    else:
        skipped["06_constraint_pressure.png"] = "no live constraint_pressure experiment"

    # --- 07: disruption impact ---------------------------------------
    if evidence.has("disruption_stress"):
        rows = evidence.get("disruption_stress").rows
        attempt(
            "07_disruption_stress.png",
            "disruption_stress",
            lambda p: figure_disruption_stress(rows, p, title="Service level by disruption rate and strategy"),
        )
    else:
        skipped["07_disruption_stress.png"] = "no live disruption_stress experiment"

    # --- 10: what-if (from the demo artifacts) ------------------------
    whatif_csv = root / "results" / "demo" / "whatif_comparison.csv"
    if whatif_csv.is_file():
        import csv as _csv
        import io as _io

        rows = list(_csv.DictReader(_io.StringIO(whatif_csv.read_text(encoding="utf-8"))))
        attempt(
            "10_whatif.png",
            "demo what-if",
            lambda p: figure_whatif(rows, p, title="What-if analysis"),
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

    # Anything in the inventory that produced no file is reported, so a missing
    # figure surfaces here instead of as a broken image link in the README.
    for name in figure_names():
        if name not in written and name not in skipped:
            skipped[name] = "not produced and not attempted"

    return {
        "output_dir": out,
        "written": sorted(written),
        "skipped": skipped,
        "inventory": FIGURE_INVENTORY,
        "integrity_problems": evidence.integrity_problems(),
    }
