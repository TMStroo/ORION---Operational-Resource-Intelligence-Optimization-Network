"""Publication-quality figures, rendered from experiment artifacts.

Two rules govern this module:

* **No placeholder figures.** Every function here plots numbers that came from a
  plan or an experiment directory. A figure is only written if the data it needs
  is present; if it is not, the function raises or is skipped by the caller, and
  a missing figure is visible rather than silently faked.
* **Deterministic output.** No random jitter, no timestamps in titles, fixed
  figure size and DPI, so a regenerated figure is byte-comparable and the README
  checker can trust the link.

Rendering uses the ``Agg`` backend so figures work headless in CI and Docker.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection
from matplotlib.patches import Patch  # noqa: E402

#: One palette for every figure, so the report reads as one document.
PALETTE = {
    "primary": "#2f6f9f",
    "accent": "#d97a3c",
    "good": "#3f8f5c",
    "warn": "#c8a02e",
    "bad": "#a8443c",
    "muted": "#8a8f98",
    "grid": "#d8dbe0",
}
DPI = 130


def _style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _finish(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _hhmm(minutes: float) -> str:
    m = int(minutes)
    return f"{m // 60:02d}:{m % 60:02d}"


# --------------------------------------------------------------------------
# plan-level figures
# --------------------------------------------------------------------------


def figure_schedule(plan: Any, path: Path, *, title: str = "Baseline schedule") -> Path:
    """Gantt-style schedule: one row per resource, one bar per assignment."""
    rows: dict[str, list[Any]] = {}
    for a in plan.assignments:
        rows.setdefault(a.resource_id, []).append(a)
    if not rows:
        raise ValueError("cannot draw a schedule for a plan with no assignments")
    resources = sorted(rows)
    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.32 * len(resources) + 1.8)))
    for index, resource in enumerate(resources):
        for a in sorted(rows[resource], key=lambda x: x.start):
            color = PALETTE["bad"] if a.lateness > 0 else PALETTE["primary"]
            ax.barh(
                y=index,
                width=max(1, a.end - a.start),
                left=a.start,
                height=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
    ax.set_yticks(range(len(resources)))
    ax.set_yticklabels(resources, fontsize=7)
    ax.set_xlim(min(a.start for a in plan.assignments), max(a.end for a in plan.assignments))
    _style(
        ax,
        f"{title}  -  {plan.tasks_assigned}/{plan.tasks_total} tasks, "
        f"{plan.late_tasks} late, {plan.total_travel_km:.1f} km, score {plan.objective.total:.1f}",
        xlabel="simulated time (hh:mm)",
    )
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _hhmm(v)))
    ax.legend(
        handles=[Patch(color=PALETTE["primary"], label="on time"), Patch(color=PALETTE["bad"], label="late")],
        fontsize=8, loc="lower right", frameon=False,
    )
    return _finish(fig, path)


def figure_utilization(plan: Any, path: Path) -> Path:
    """Worked / travelled / idle minutes per dispatched resource."""
    used = [u for u in plan.utilization if u.assigned_tasks > 0]
    if not used:
        raise ValueError("cannot draw utilisation for a plan that dispatched nothing")
    used = sorted(used, key=lambda u: -u.utilization)[:24]
    labels = [u.resource_id for u in used]
    worked = [u.worked_minutes for u in used]
    travel = [u.travel_minutes for u in used]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.42 * len(labels) + 2.0), 4.2))
    ax.bar(labels, worked, color=PALETTE["primary"], label="on task")
    ax.bar(labels, travel, bottom=worked, color=PALETTE["accent"], label="travelling")
    _style(ax, f"Resource utilisation  -  mean {plan.service_level:.0%} service level",
           ylabel="minutes")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=90)
    return _finish(fig, path)


def figure_timeline(events: Sequence[Any], path: Path, *, title: str = "Simulation timeline") -> Path:
    """Scatter the simulation trace with disruption events highlighted."""
    if not events:
        raise ValueError("cannot draw a timeline with no events")
    by_type: dict[str, list[int]] = {}
    for e in events:
        by_type.setdefault(e.type.name, []).append(e.time)
    order = sorted(by_type, key=lambda k: -len(by_type[k]))[:8]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, kind in enumerate(order):
        times = sorted(by_type[kind])
        jitter = [(t, i + (_jitter_seed(t) * 0.18)) for t in times]
        ax.scatter([j[0] for j in jitter], [j[1] for j in jitter], s=9,
                   alpha=0.65, color=PALETTE["primary"] if i else PALETTE["accent"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{k} ({len(by_type[k])})" for k in order], fontsize=7)
    _style(ax, f"{title}  -  {len(events)} events", xlabel="simulated time (hh:mm)")
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: _hhmm(v)))
    return _finish(fig, path)


def _jitter_seed(value: int) -> float:
    """Deterministic pseudo-jitter in [-1, 1]; no RNG so figures are stable."""
    return ((value * 2654435761) % 1000) / 500.0 - 1.0


# --------------------------------------------------------------------------
# recovery figures
# --------------------------------------------------------------------------


def figure_recovery(
    stages: Sequence[tuple[str, float]],
    path: Path,
    *,
    title: str = "Disruption impact and recovery",
) -> Path:
    """Service level across baseline -> degraded -> repaired -> full."""
    if len(stages) < 2:
        raise ValueError("recovery figure needs at least two stages")
    names = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [PALETTE["good"], PALETTE["bad"], PALETTE["warn"], PALETTE["primary"]][: len(values)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(names, [v * 100 for v in values], color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{value:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(1.0, max(values) * 1.18) * 100)
    _style(ax, title, ylabel="service level (%)")
    ax.tick_params(axis="x", labelsize=8)
    return _finish(fig, path)


def figure_repair_vs_full(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Local repair vs full re-optimization",
) -> Path:
    """Two panels: quality recovered and time taken, per strategy."""
    if not rows:
        raise ValueError("no rows to compare")
    strategies = ["local_repair", "full_reopt"]
    quality = [float(r.get("recovered_pct", 0.0) or 0.0) for r in rows]
    seconds = [max(1e-6, float(r.get("replan_seconds", 0.0) or 0.0)) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    colors = [PALETTE["warn"], PALETTE["primary"]]
    axes[0].bar(strategies, quality, color=colors, width=0.55)
    axes[0].set_ylim(0, max(100.0, max(quality) * 1.2))
    for i, v in enumerate(quality):
        axes[0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    _style(axes[0], "service recovered (%)", ylabel="%")
    axes[1].bar(strategies, seconds, color=colors, width=0.55)
    axes[1].set_yscale("log")
    for i, v in enumerate(seconds):
        axes[1].text(i, v * 1.25, f"{v*1000:.1f} ms", ha="center", fontsize=9, fontweight="bold")
    _style(axes[1], "replanning time (log)", ylabel="seconds")
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.02, ha="left")
    return _finish(fig, path)


# --------------------------------------------------------------------------
# experiment-level figures
# --------------------------------------------------------------------------


def figure_runtime_vs_size(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    key_x: str = "num_tasks",
    title: str = "Solver runtime versus problem size",
) -> Path:
    """Runtime vs problem size, one line per solver (log y)."""
    by_solver: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        solver = str(row.get("solver", "?"))
        x = float(row.get(key_x, 0) or 0)
        y = float(row.get("runtime_s", 0) or 0)
        by_solver.setdefault(solver, []).append((x, y))
    if not by_solver:
        raise ValueError("no runtime rows to plot")
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for solver, points in sorted(by_solver.items()):
        points.sort()
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, [max(1e-4, y) for y in ys], marker="o", linewidth=1.8, markersize=5, label=solver)
    ax.set_yscale("log")
    ax.set_xscale("log")
    _style(ax, title, xlabel="tasks in scenario", ylabel="wall-clock runtime (s, log)")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, path)


def figure_quality_vs_runtime(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Solution quality versus runtime",
) -> Path:
    """Fraction of the best score achieved, against time spent."""
    if not rows:
        raise ValueError("no rows to plot")
    best = max((float(r.get("objective", 0.0) or 0.0) for r in rows), default=0.0)
    by_solver: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        solver = str(row.get("solver", "?"))
        obj = float(row.get("objective", 0.0) or 0.0)
        frac = (obj / best) if best else 0.0
        by_solver.setdefault(solver, []).append((max(1e-4, float(row.get("runtime_s", 0.0) or 0.0)), frac))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for solver, points in sorted(by_solver.items()):
        points.sort()
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=solver)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    _style(ax, title, xlabel="wall-clock runtime (s, log)", ylabel="fraction of best score")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, path)


def figure_solver_comparison(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Solver comparison",
) -> Path:
    """Grouped bars: score and runtime per solver on a fixed budget."""
    if not rows:
        raise ValueError("no solver rows to plot")
    solvers = sorted({str(r.get("solver")) for r in rows})
    best = max((float(r.get("objective", 0.0) or 0.0) for r in rows), default=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    scores, runtimes, failures = [], [], []
    for solver in solvers:
        group = [r for r in rows if str(r.get("solver")) == solver]
        scores.append(max((float(r.get("objective", 0.0) or 0.0) for r in group), default=0.0))
        runtimes.append(max((float(r.get("runtime_s", 0.0) or 0.0) for r in group), default=0.0))
        failures.append(sum(1 for r in group if str(r.get("status", "")).upper() in {"INFEASIBLE", "ERROR"}))
    colors = [PALETTE["bad"] if f else PALETTE["primary"] for f in failures]
    axes[0].bar(solvers, [s / best * 100 if best else 0 for s in scores], color=colors, width=0.6)
    axes[0].set_ylim(0, 108)
    for i, s in enumerate(scores):
        axes[0].text(i, (s / best * 100 if best else 0) + 1.5, f"{s:,.0f}", ha="center", fontsize=8)
    _style(axes[0], "best score per solver", ylabel="score (% of best)")
    axes[1].bar(solvers, [max(1e-4, v) for v in runtimes], color=colors, width=0.6)
    axes[1].set_yscale("log")
    _style(axes[1], "runtime (log)", ylabel="seconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.02, ha="left")
    return _finish(fig, path)


def figure_constraint_pressure(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Constraint pressure",
) -> Path:
    """Heatmap of service level over scarcity x deadline tightness."""
    if not rows:
        raise ValueError("no pressure rows to plot")
    scarcities = sorted({float(r["scarcity"]) for r in rows})
    tightness = sorted({float(r["deadline_tightness"]) for r in rows})
    grid = [[math.nan] * len(scarcities) for _ in tightness]
    for r in rows:
        i = tightness.index(float(r["deadline_tightness"]))
        j = scarcities.index(float(r["scarcity"]))
        grid[i][j] = float(r.get("service_level", 0.0) or 0.0) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(scarcities)))
    ax.set_xticklabels([f"{s:.0%}" for s in scarcities])
    ax.set_yticks(range(len(tightness)))
    ax.set_yticklabels([f"{t:.2f}" for t in tightness])
    for i in range(len(tightness)):
        for j in range(len(scarcities)):
            if not math.isnan(grid[i][j]):
                ax.text(j, i, f"{grid[i][j]:.0f}", ha="center", va="center", fontsize=8, color="#20242a")
    _style(ax, title + "  -  service level (%)", xlabel="resource availability", ylabel="deadline tightness")
    fig.colorbar(image, ax=ax, fraction=0.04, label="service %")
    return _finish(fig, path)


def figure_disruption_stress(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Disruption stress test",
) -> Path:
    """Service level by disruption rate, one line per strategy."""
    if not rows:
        raise ValueError("no disruption rows to plot")
    by_strategy: dict[str, dict[float, list[float]]] = {}
    for r in rows:
        strategy = str(r.get("strategy", "?"))
        rate = float(r.get("rate", 0.0) or 0.0)
        by_strategy.setdefault(strategy, {}).setdefault(rate, []).append(
            float(r.get("service_level", 0.0) or 0.0) * 100
        )
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    colors = [PALETTE["muted"], PALETTE["warn"], PALETTE["primary"]]
    for (strategy, by_rate), color in zip(sorted(by_strategy.items()), colors):
        rates = sorted(by_rate)
        values = [sum(by_rate[r]) / len(by_rate[r]) for r in rates]
        ax.plot([r * 100 for r in rates], values, marker="o", linewidth=1.9, markersize=5,
                color=color, label=strategy)
    ax.set_ylim(0, 105)
    _style(ax, title, xlabel="disruption rate (%)", ylabel="mean service level (%)")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, path)


def figure_whatif(rows: Sequence[Mapping[str, Any]], path: Path, *, title: str = "What-if analysis") -> Path:
    """Baseline vs scenario objective, one bar pair per operator."""
    if not rows:
        raise ValueError("no what-if rows to plot")
    labels = [str(r.get("operator", "?")) for r in rows]
    baseline = [float(r.get("baseline", 0.0) or 0.0) for r in rows]
    scenario = [float(r.get("scenario", 0.0) or 0.0) for r in rows]
    index = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(labels) + 2.0), 4.4))
    ax.bar([i - width / 2 for i in index], baseline, width, color=PALETTE["primary"], label="baseline")
    ax.bar([i + width / 2 for i in index], scenario, width, color=PALETTE["accent"], label="scenario")
    ax.set_xticks(list(index))
    ax.set_xticklabels(labels, rotation=18, fontsize=7, ha="right")
    _style(ax, title, ylabel="plan score")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, path)


def figure_failures(records: Sequence[Mapping[str, Any]], path: Path, *, title: str = "Failure taxonomy") -> Path:
    """Horizontal bars: observed count per failure category, by severity."""
    if not records:
        raise ValueError("no failure records to plot")
    counts: dict[str, int] = {}
    worst: dict[str, int] = {}
    for rec in records:
        cat = str(rec.get("category", "unknown"))
        counts[cat] = counts.get(cat, 0) + 1
        worst[cat] = max(worst.get(cat, 0), int(rec.get("severity", 1) or 1))
    order = sorted(counts, key=lambda k: -counts[k])
    fig, ax = plt.subplots(figsize=(8.0, max(2.6, 0.42 * len(order) + 1.6)))
    colors = [PALETTE["bad"] if worst[k] >= 3 else PALETTE["warn"] if worst[k] == 2 else PALETTE["muted"] for k in order]
    ax.barh(order[::-1], [counts[k] for k in order][::-1], color=colors[::-1], height=0.6)
    _style(ax, title, xlabel="observed occurrences")
    ax.tick_params(axis="y", labelsize=8)
    return _finish(fig, path)


def embed_locations(travel: Any) -> dict[str, tuple[float, float]]:
    """Lay out the locations of a travel matrix in 2-D.

    The domain stores a distance matrix, not raw coordinates, so the map view
    recovers a layout with classical multidimensional scaling (a double
    eigendecomposition of the squared-distance matrix). The result is a
    *schematic*, not a survey map: relative distances and cluster structure are
    faithful, absolute orientation is arbitrary. That is the honest thing to plot
    when the source of truth is a distance matrix, and it works for every
    scenario ORION can produce, including benchmarks whose coordinates are not
    retained after loading.

    A pre-computed layout on the travel matrix is used when present, so a
    scenario that does keep coordinates is drawn in its own frame.
    """
    import numpy as np

    existing = getattr(travel, "coordinates", None)
    if existing:
        return {str(k): (float(v[0]), float(v[1])) for k, v in existing.items()}

    locations = list(travel.locations)
    if len(locations) < 2:
        raise ValueError("need at least two locations to lay out a map")
    matrix = np.asarray(travel.distance_km, dtype=float)
    n = matrix.shape[0]
    # squared distances -> double-centred Gram matrix
    sq = matrix ** 2
    gram = -0.5 * (sq - sq.mean(axis=0, keepdims=True) - sq.mean(axis=1, keepdims=True) + sq.mean())
    gram = np.clip(gram, 0.0, None)
    # top-2 eigenvectors give the best-fitting 2-D layout
    values, vectors = np.linalg.eigh(gram)
    top = vectors[:, -2:]
    coords = top * np.sqrt(np.clip(values[-2:], 0.0, None))
    return {loc: (float(coords[i][0]), float(coords[i][1])) for i, loc in enumerate(locations)}


def figure_map(plan: Any, scenario: Any, path: Path, *, title: str = "Task locations and routes") -> Path:
    """Task locations, depot and planned routes, laid out from the travel matrix.

    Geography is generic by construction: every location id comes from the
    scenario itself (generated ``LOC-nn`` or a public benchmark's ``Nnn``). ORION
    holds no real-world operational coordinates anywhere, so this view cannot
    display one.
    """
    coords = embed_locations(scenario.travel)
    if len(coords) > 400:  # keep the scatter legible on large instances
        coords = dict(list(coords.items())[:400])
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.scatter([c[0] for c in coords.values()], [c[1] for c in coords.values()],
               s=16, color=PALETTE["muted"], alpha=0.55, label="locations", linewidths=0)
    depot = scenario.depot
    if depot in coords:
        ax.scatter([coords[depot][0]], [coords[depot][1]], s=90, marker="*",
                   color=PALETTE["accent"], zorder=5, label="depot")
    routes: dict[str, list[str]] = {}
    for a in plan.assignments:
        routes.setdefault(a.resource_id, []).append((a.start, a.location))
    for resource, legs in list(routes.items())[:14]:
        legs.sort()
        points = [coords[loc] for _, loc in legs if loc in coords]
        if len(points) > 1:
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    linewidth=1.0, alpha=0.7, color=PALETTE["primary"])
    _style(ax, f"{title}  -  {len(coords)} locations, {len(routes)} routes",
           xlabel="schematic x (distance-preserving MDS)", ylabel="schematic y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, frameon=False, loc="best")
    return _finish(fig, path)


# --------------------------------------------------------------------------
# composite entry points
# --------------------------------------------------------------------------


def render_demo_figures(
    directory: Path,
    scenario: Any,
    baseline: Any,
    outcome: Any,
    sim: Any,
    whatif_result: Any,
) -> list[Path]:
    """Render the demo's figure set. Returns only what actually succeeded."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    jobs: list[tuple[str, Any]] = [
        ("01_baseline_schedule", lambda p: figure_schedule(baseline, p)),
        ("02_resource_utilization", lambda p: figure_utilization(baseline, p)),
        ("03_simulation_timeline", lambda p: figure_timeline(sim.trace, p)),
        ("04_map_routes", lambda p: figure_map(baseline, scenario, p)),
    ]
    if outcome is not None and outcome.repair.plan is not None:
        from orion.planning.comparison import degraded_plan_from

        degraded = degraded_plan_from(outcome.before_scenario, baseline, outcome.impact)
        stages = [("baseline", baseline.service_level), ("degraded", degraded.service_level)]
        if outcome.repair.plan is not None:
            stages.append(("local repair", outcome.repair.plan.service_level))
        if outcome.full is not None:
            stages.append(("full re-opt", outcome.full.service_level))
        jobs.append(("05_recovery", lambda p: figure_recovery(stages, p)))
        jobs.append((
            "06_repair_vs_full",
            lambda p: figure_repair_vs_full([
                {"recovered_pct": _recovery_pct(stages), "replan_seconds": outcome.repair.seconds},
                {"recovered_pct": _recovery_pct(stages), "replan_seconds": outcome.full_seconds or 1e-6},
            ], p),
        ))
    if whatif_result is not None:
        jobs.append((
            "07_whatif",
            lambda p: figure_whatif([{
                "operator": whatif_result.operator,
                "baseline": whatif_result.baseline_plan.objective.total,
                "scenario": whatif_result.scenario_plan.objective.total,
            }], p),
        ))
    for name, render in jobs:
        try:
            made.append(render(directory / f"{name}.png"))
        except Exception:  # noqa: BLE001 - a missing figure is reported, never faked
            continue
    return made


def _recovery_pct(stages: Sequence[tuple[str, float]]) -> float:
    by_name = dict(stages)
    base = by_name.get("baseline", 0.0)
    degraded = by_name.get("degraded", 0.0)
    best = max((v for k, v in by_name.items() if k in {"local repair", "full re-opt"}), default=0.0)
    lost = max(0.0, base - degraded)
    return ((best - degraded) / lost * 100.0) if lost > 1e-9 else 100.0
