"""``python -m orion demo`` - the deterministic five-minute walkthrough.

The demo is the project's front door, so it runs the *real* pipeline: no mocked
solver, no canned output. Every number it prints is read back from the plan the
optimizer actually produced, and every artifact it writes lands in a directory a
reviewer can inspect.

Sequence (matching the README walkthrough):

1. build and validate the scenario
2. produce the baseline plan and print its metrics
3. simulate the plan
4. inject a vehicle failure; show the affected assignments
5. local repair -> report recovered tasks, churn and runtime
6. full re-optimization -> report the quality/time trade-off
7. inject an urgent demand surge and replan
8. run one what-if comparison
9. write exports, figures and an audit log

Determinism: every source of randomness is seeded from the config, so two runs
on the same machine produce identical numbers. That is asserted by the demo
integration test rather than merely claimed here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from orion.config import ExperimentConfig
from orion.data.scenario_generator import difficulty_config, generate
from orion.domain.entities import Scenario
from orion.domain.errors import OrionError
from orion.domain.events import Disruption, NewTaskSpec
from orion.evaluation.metrics import plan_metric_rows
from orion.observability import AuditAction, AuditLog, configure_logging, get_logger
from orion.planning.comparison import compare_plans, compute_churn, degraded_plan_from
from orion.planning.planner import Planner
from orion.planning.replanner import Replanner
from orion.planning.whatif import WhatIfEngine
from orion.simulation.engine import SimulationEngine
from orion.simulation.events import EventType

LOG = get_logger("demo")


@dataclass(slots=True)
class DemoStage:
    """One numbered step of the demo, with the facts it observed."""

    number: int
    title: str
    facts: tuple[str, ...] = ()
    seconds: float = 0.0

    def render(self) -> str:
        head = f"[{self.number:2d}] {self.title}  ({self.seconds:.2f}s)"
        body = "".join(f"\n      {f}" for f in self.facts)
        return head + body


@dataclass(slots=True)
class DemoSummary:
    """Everything the demo produced, for the CLI and for tests to assert on."""

    scenario_id: str
    output_dir: Path
    stages: list[DemoStage] = field(default_factory=list)
    baseline_score: float = 0.0
    baseline_service: float = 0.0
    degraded_score: float = 0.0
    repaired_score: float = 0.0
    full_score: float = 0.0
    repair_seconds: float = 0.0
    full_seconds: float = 0.0
    recovery_pct: float = 0.0
    churn_changed: int = 0
    churn_total: int = 0
    affected_tasks: int = 0
    surge_recovered: int = 0
    whatif_delta: float = 0.0
    artifacts: tuple[str, ...] = ()

    def highlights(self) -> list[str]:
        return [
            f"scenario            {self.scenario_id}",
            f"baseline score      {self.baseline_score:.2f}  service {self.baseline_service:.1%}",
            f"disruption impact   {self.affected_tasks} task(s) affected",
            f"local repair        {self.repaired_score:.2f} in {self.repair_seconds*1000:.1f} ms, "
            f"churn {self.churn_changed}/{self.churn_total}",
            f"full re-optimize    {self.full_score:.2f} in {self.full_seconds*1000:.1f} ms",
            f"recovery            {self.recovery_pct:.1f}% of the lost service",
            f"demand surge        {self.surge_recovered} task(s) recovered",
            f"what-if delta       {self.whatif_delta:+.2f} objective",
            f"artifacts           {self.output_dir}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "output_dir": str(self.output_dir),
            "baseline_score": self.baseline_score,
            "baseline_service": self.baseline_service,
            "degraded_score": self.degraded_score,
            "repaired_score": self.repaired_score,
            "full_score": self.full_score,
            "repair_seconds": self.repair_seconds,
            "full_seconds": self.full_seconds,
            "recovery_pct": self.recovery_pct,
            "churn_changed": self.churn_changed,
            "churn_total": self.churn_total,
            "affected_tasks": self.affected_tasks,
            "surge_recovered": self.surge_recovered,
            "whatif_delta": self.whatif_delta,
            "artifacts": list(self.artifacts),
            "stages": [
                {"number": s.number, "title": s.title, "facts": list(s.facts), "seconds": s.seconds}
                for s in self.stages
            ],
        }


def _build_scenario(cfg: ExperimentConfig, quick: bool) -> Scenario:
    level = "medium" if quick else cfg.scenario.level
    scenario_cfg = difficulty_config(
        level,
        seed=cfg.scenario.seed,
        maintenance_count=cfg.scenario.maintenance_count,
        resource_scarcity=cfg.scenario.scarcity,
        deadline_tightness=cfg.scenario.deadline_tightness,
        dependency_rate=cfg.scenario.dependency_rate,
        weight_overrides=cfg.weights().to_dict(),
    )
    return generate(scenario_cfg)


def _metric_line(plan: Any, scenario: Scenario, keys: Sequence[str]) -> list[str]:
    rows = {r.key: r for r in plan_metric_rows(plan, scenario)}
    return [f"{rows[key].label}: {rows[key].display}" for key in keys if key in rows]


def _explanation_facts(explanations: Sequence[Any], limit: int = 1) -> list[str]:
    """Render one real assignment explanation, straight from the solver's checks.

    The text is assembled from the constraint checks the explanation already
    carries - skills, capacity, availability, maintenance, deadline, travel - plus
    the objective contribution and the best alternative resource. Nothing here
    is invented: if a check failed, it is printed as failed.
    """
    if not explanations:
        return ["explanations: none attached"]
    facts: list[str] = []
    for exp in list(explanations)[:limit]:
        checks = "; ".join(
            f"{c.name}={'ok' if c.satisfied else 'FAIL'}({c.value})" for c in exp.checks
        )
        runner = ""
        if exp.runner_up_resource:
            runner = (
                f" | alternative {exp.runner_up_resource}: "
                f"{exp.runner_up_extra_km:+.1f} km, {exp.runner_up_extra_cost:+.2f} cost"
            )
        facts.append(
            f"{exp.task_id} -> {exp.resource_id}: {checks} | objective {exp.objective_contribution:+.2f}{runner}"
        )
    facts.append(f"explanations attached: {len(explanations)}")
    return facts


def _pick_failure_target(scenario: Scenario, plan: Any) -> str:
    """Choose a resource that actually carries work.

    Disrupting an unused resource produces a zero-impact event, which would make
    the demo look like the recovery machinery does nothing. The demo therefore
    targets the busiest dispatched resource.
    """
    counts: dict[str, int] = {}
    for a in plan.assignments:
        counts[a.resource_id] = counts.get(a.resource_id, 0) + 1
    if not counts:
        return scenario.resources[0].id
    return max(sorted(counts), key=lambda rid: counts[rid])


def run_demo(
    *,
    config_path: str | Path | None = "configs/demo.yaml",
    output_dir: str | Path = "results/demo",
    quick: bool = False,
    figures: bool = True,
) -> DemoSummary:
    """Run the full demo and return a machine-checkable summary."""
    configure_logging()
    cfg = (
        ExperimentConfig.from_yaml(config_path)
        if config_path and Path(config_path).exists()
        else ExperimentConfig(kind="demo", description="built-in demo")
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(out_dir / "audit.log", actor="demo")
    solver = cfg.solvers[0]
    budget = cfg.time_budgets[0]
    planner = Planner(default_solver=solver, default_time_budget=budget, seed=cfg.seed)
    summary = DemoSummary(scenario_id="", output_dir=out_dir)
    artifacts: list[str] = []

    def finish(number: int, title: str, facts: Sequence[str], started: float) -> None:
        summary.stages.append(DemoStage(number, title, tuple(facts), time.perf_counter() - started))
        LOG.info(f"demo stage {number}: {title}")

    # ---- 1. scenario -----------------------------------------------------
    t0 = time.perf_counter()
    scenario = _build_scenario(cfg, quick)
    summary.scenario_id = scenario.id
    infeasible = scenario.static_infeasibility()
    demand = scenario.demand_summary()
    audit.record(AuditAction.SCENARIO_CREATED, scenario.id, num_tasks=len(scenario.tasks), num_resources=len(scenario.resources))
    if infeasible:
        raise OrionError(f"demo scenario is statically infeasible: {infeasible}")
    audit.record(AuditAction.SCENARIO_VALIDATED, scenario.id, feasible=True)
    finish(1, "Scenario created and validated", [
        f"{len(scenario.tasks)} tasks, {len(scenario.resources)} resources "
        f"({sum(1 for r in scenario.resources if r.kind == 'TEAM')} teams / "
        f"{sum(1 for r in scenario.resources if r.kind == 'VEHICLE')} vehicles)",
        f"horizon {scenario.horizon}  demand {demand['total_demand_minutes']} min  "
        f"available {demand['available_minutes']} min",
        f"capabilities {sorted({c for r in scenario.resources for c in r.capabilities})}",
        f"static feasibility: FEASIBLE",
    ], t0)

    # ---- 2. baseline plan ------------------------------------------------
    t0 = time.perf_counter()
    result = planner.plan(scenario, explain=True)
    baseline = result.plan
    summary.baseline_score = baseline.objective.total
    summary.baseline_service = baseline.service_level
    audit.record(AuditAction.PLAN_GENERATED, scenario.id, plan_id=baseline.id,
                 solver=solver, status=baseline.status, objective=baseline.objective.total,
                 runtime_s=max((r.runtime_s for r in baseline.solver_runs), default=0.0))
    artifacts.append(str(export_plan_file(baseline, scenario, out_dir, seed=cfg.seed)))
    finish(2, "Baseline plan generated", [
        baseline.summary_line(),
        *_metric_line(baseline, scenario, [
            "plan_score", "service_level", "task_completion_rate", "deadline_violation_rate",
            "mean_lateness", "travel_distance", "operating_cost", "mean_utilisation",
        ]),
        *_explanation_facts(result.explanations),
    ], t0)

    # ---- 3. simulate -----------------------------------------------------
    t0 = time.perf_counter()
    sim = SimulationEngine(scenario, baseline).run()
    audit.record(AuditAction.SIMULATION_STARTED, scenario.id, plan_id=baseline.id, events=len(sim.trace))
    finish(3, "Simulation executed", [
        sim.summary_line(),
        f"first events: " + " | ".join(f"{_hhmm(e.time)} {e.type.name}" for e in sim.trace[:3]),
    ], t0)

    # ---- 4. vehicle failure ---------------------------------------------
    t0 = time.perf_counter()
    target = _pick_failure_target(scenario, baseline)
    when = max(scenario.horizon.start + 60, scenario.horizon.end // 4)
    failure = Disruption(
        id=f"{scenario.id}-FAIL-01",
        type="VEHICLE_FAILURE",
        timestamp=when,
        target_id=target,
        magnitude=1.0,
        duration=scenario.horizon.end - when,
        description=f"{target} fails at {_hhmm(when)} and is out of service for the rest of the horizon",
        severity="HIGH",
    )
    original = [a for a in baseline.assignments if a.resource_id == target]
    audit.record(AuditAction.DISRUPTION_INJECTED, scenario.id, disruption_id=failure.id,
                 disruption_type=failure.type, target=target)
    finish(4, f"Disruption injected: {target} fails at {_hhmm(when)}", [
        f"affected tasks: {len(original)}",
        ", ".join(f"{a.task_id} [{_hhmm(a.start)}-{_hhmm(a.end)}]" for a in original[:6]) or "  (none)",
    ], t0)

    # ---- 5/6. local repair vs full re-optimization -----------------------
    t0 = time.perf_counter()
    replanner = Replanner(solver=solver, time_budget_s=budget, seed=cfg.seed)
    outcome = replanner.replan(scenario, baseline, failure)
    # The degraded plan is what a planner would keep if it did *nothing*: the
    # same assignments, evaluated against a world in which the resource is gone.
    degraded = degraded_plan_from(scenario, baseline, outcome.impact)
    outcome_degraded_service = degraded.service_level
    repair = outcome.repair
    audit.record(AuditAction.LOCAL_REPAIR_ATTEMPTED, scenario.id, plan_id=baseline.id,
                 disruption_id=failure.id, strategy=repair.strategy,
                 repaired_tasks=len(repair.recovered_tasks), seconds=repair.seconds)
    repaired_plan = repair.plan
    churn = compute_churn(baseline, repaired_plan) if repaired_plan else None
    summary.repaired_score = repaired_plan.objective.total if repaired_plan else 0.0
    summary.repair_seconds = repair.seconds
    summary.churn_changed = churn.changed_assignments if churn else 0
    summary.churn_total = churn.total_before if churn else 0
    summary.affected_tasks = len(outcome.impact.affected_tasks)
    summary.degraded_score = degraded.objective.total
    finish(5, "Local repair", [
        f"strategy: {repair.strategy}  candidates considered: {repair.candidates_considered}",
        f"recovered {len(repair.recovered_tasks)} task(s) in {repair.seconds*1000:.1f} ms",
        f"unrecovered: {list(repair.unrecovered_tasks) or 'none'}  ({repair.reason})" if repair.reason else f"unrecovered: {list(repair.unrecovered_tasks) or 'none'}",
        f"plan churn: {summary.churn_changed}/{summary.churn_total} assignments changed"
        + (f", travel {churn.travel_delta_km:+.1f} km" if churn else ""),
    ], t0)

    t0 = time.perf_counter()
    if outcome.full is not None:
        audit.record(AuditAction.FULL_REOPT_ATTEMPTED, scenario.id, disruption_id=failure.id,
                     solver=solver, status=outcome.full.status, objective=outcome.full.objective.total)
        summary.full_score = outcome.full.objective.total
        summary.full_seconds = outcome.full_seconds
        full_churn = compute_churn(baseline, outcome.full)
        lost_service = max(0.0, summary.baseline_service - outcome_degraded_service)
        recovered = max(0.0, outcome.full.service_level - outcome_degraded_service)
        summary.recovery_pct = (recovered / lost_service * 100.0) if lost_service > 1e-9 else 100.0
        finish(6, "Full re-optimization (for comparison)", [
            outcome.full.summary_line(),
            f"runtime {outcome.full_seconds*1000:.1f} ms vs local repair {repair.seconds*1000:.1f} ms "
            f"({outcome.full_seconds/max(repair.seconds,1e-9):.0f}x slower)",
            f"churn {full_churn.changed_assignments}/{full_churn.total_before} assignments changed",
            f"service: baseline {summary.baseline_service:.1%} -> degraded {outcome_degraded_service:.1%} "
            f"-> full {outcome.full.service_level:.1%}  (recovery {summary.recovery_pct:.1f}%)",
        ], t0)
    else:
        finish(6, "Full re-optimization", [f"not attempted: {outcome.full_reason or 'disabled'}"], t0)

    # ---- 7. demand surge -------------------------------------------------
    t0 = time.perf_counter()
    surge_count = max(3, round(len(scenario.tasks) * 0.30))
    surge = _build_demand_surge(scenario, surge_count, cfg.seed)
    new_ids = {spec["task_id"] for spec in surge.payload["tasks"]}
    audit.record(AuditAction.DISRUPTION_INJECTED, scenario.id, disruption_id=surge.id,
                 disruption_type=surge.type, count=surge_count)
    # Let the domain apply it: DEMAND_SURGE validates each new task against the
    # travel matrix, refuses duplicate ids, and widens the horizon if needed.
    disrupted_scenario = surge.apply(scenario)
    surge_plan = planner.plan(disrupted_scenario, explain=False).plan
    handled = sum(1 for tid in new_ids if surge_plan.assignment_for(tid) is not None)
    summary.surge_recovered = handled
    finish(7, f"Urgent demand surge: +{surge_count} high-priority tasks", [
        f"replanned scenario: {len(disrupted_scenario.tasks)} tasks (was {len(scenario.tasks)})",
        f"urgent tasks accepted into the plan: {handled}/{surge_count} ({handled / surge_count:.0%})",
        surge_plan.summary_line(),
    ], t0)

    # ---- 8. what-if ------------------------------------------------------
    t0 = time.perf_counter()
    whatif = WhatIfEngine(planner)
    wi = whatif.run(scenario, baseline, "deadline_tighten", 0.15)
    score_delta = wi.scenario_plan.objective.total - wi.baseline_plan.objective.total
    summary.whatif_delta = score_delta
    audit.record(AuditAction.WHAT_IF_RUN, scenario.id, operator=wi.operator, parameter=wi.parameter)
    top = wi.comparison.deltas[:4]
    finish(8, "What-if: deadlines 15% tighter", [
        wi.question,
        f"modification: {wi.modification}",
        f"baseline {wi.baseline_plan.objective.total:,.2f} -> scenario {wi.scenario_plan.objective.total:,.2f} ({score_delta:+,.2f})",
        "; ".join(f"{d.label} {d.before:,.1f}->{d.after:,.1f}" for d in top),
    ], t0)

    # ---- 9. artifacts ----------------------------------------------------
    t0 = time.perf_counter()
    summary.artifacts = tuple(_write_artifacts(out_dir, scenario, baseline, outcome, sim, cfg, artifacts))
    finish(9, "Artifacts written", [f"{len(summary.artifacts)} files under {out_dir}"], t0)

    audit.record(
        AuditAction.SCENARIO_EXPORTED, scenario.id,
        output_dir=str(out_dir), artifacts=len(summary.artifacts),
    )
    (out_dir / "demo_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    summary.artifacts = summary.artifacts + (str(out_dir / "demo_summary.json"),)

    if figures:
        try:
            from orion.figures import render_demo_figures

            made = render_demo_figures(out_dir / "figures", scenario, baseline, outcome, sim, wi)
            summary.artifacts = summary.artifacts + tuple(str(m) for m in made)
        except Exception as exc:  # noqa: BLE001 - figures are evidence, not correctness
            LOG.info("figure generation skipped", extra={"fields": {"error": str(exc), "error_type": type(exc).__name__}})
            print(f"  (figure generation skipped: {exc})")

    for stage in summary.stages:
        print(stage.render())
    return summary


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _build_demand_surge(scenario: Scenario, count: int, seed: int) -> Disruption:
    """A ``DEMAND_SURGE`` disruption carrying *count* new high-priority tasks.

    Built through the domain's own :class:`NewTaskSpec` so the payload is exactly
    what ``_apply_demand_surge`` expects. New tasks are placed at locations that
    already exist in the travel matrix, with deadlines inside the horizon, so the
    surge measures *demand pressure* rather than a malformed scenario.
    """
    import random

    rng = random.Random(seed ^ 0x5EED)
    locations = sorted({t.location for t in scenario.tasks}) or [scenario.depot]
    horizon = scenario.horizon
    span = max(1, (horizon.end - horizon.start) // 2)
    specs = []
    for i in range(count):
        release = horizon.start + rng.randrange(0, span)
        duration = rng.randrange(20, 60, 5)
        deadline = min(horizon.end - 1, release + duration + rng.randrange(90, 300))
        specs.append(
            NewTaskSpec(
                task_id=f"{scenario.id}-URGENT-{i + 1:02d}",
                location=rng.choice(locations),
                release_time=release,
                deadline=max(release + duration, deadline),
                duration=duration,
                priority=3,
            ).to_dict()
        )
    return Disruption(
        id=f"{scenario.id}-SURGE-01",
        type="DEMAND_SURGE",
        timestamp=horizon.start + max(30, (horizon.end - horizon.start) // 3),
        magnitude=float(count),
        description=f"{count} new urgent tasks arrive",
        severity="HIGH",
        payload={"tasks": specs},
    )


def export_plan_file(plan: Any, scenario: Scenario, out_dir: Path, *, seed: int) -> Path:
    from orion.exporting import export_plan

    return export_plan(plan, out_dir / "baseline_plan.json", seed=seed,
                       objective_weights=scenario.objective_weights.to_dict())


def _write_artifacts(
    out_dir: Path,
    scenario: Scenario,
    baseline: Any,
    outcome: Any,
    sim: Any,
    cfg: ExperimentConfig,
    already: list[str],
) -> list[str]:
    from orion.exporting import (
        export_comparison,
        export_scenario,
        plan_metric_rows_csv,
        write_simulation_csv,
    )

    prov = {"seed": cfg.seed, "objective_weights": scenario.objective_weights.to_dict(),
            "source": f"demo:{scenario.id}"}
    out: list[str] = [str(export_scenario(scenario, out_dir / "scenario.json", **prov))]
    out.append(str(plan_metric_rows_csv(baseline, scenario, out_dir / "baseline_metrics.csv", **prov)))
    out.append(str(write_simulation_csv(sim, out_dir / "simulation_events.csv", **prov)))
    if outcome.repair.plan is not None:
        from orion.exporting import export_plan

        out.append(str(export_plan(outcome.repair.plan, out_dir / "repaired_plan.json", **prov)))
    if outcome.full is not None:
        from orion.exporting import export_plan

        out.append(str(export_plan(outcome.full, out_dir / "full_reoptimized_plan.json", **prov)))
        out.append(str(compare_export(outcome, scenario, out_dir, **prov)))
    out.extend(already)
    return out


def compare_export(outcome: Any, scenario: Scenario, out_dir: Path, **prov: Any) -> Path:
    """Local repair vs full re-optimization, side by side."""
    from orion.exporting import export_comparison

    left = outcome.repair.plan or outcome.full
    right = outcome.full or outcome.repair.plan
    if left is None or right is None:
        raise OrionError("cannot compare: neither repair nor full re-optimization produced a plan")
    comparison = compare_plans(left, right, scenario)
    return export_comparison(comparison, out_dir / "repair_vs_full.csv", **prov)
