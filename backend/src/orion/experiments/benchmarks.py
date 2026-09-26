"""Reproducible experiment suites.

Five suites, each answering one of the project's secondary questions:

``scalability``        runtime and quality vs problem size (exact vs CP-SAT vs
                       heuristic vs local search)
``time_budget``        quality lost when the time cap binds
``constraint_pressure``  feasibility and quality as constraints tighten
``disruption_stress``  service loss and recovery vs disruption rate/severity
``solver_comparison``  like-for-like solver comparison on public benchmarks

Fair-comparison rules enforced here, not just documented:

* Every solver in a run sees the *same* scenario object, the same objective
  weights and the same input ordering.
* A run with a different time budget is recorded with that budget, so a 60 s
  heuristic is never silently compared against an unbounded exact solve.
* Each run is stamped with the instance's split, and ``eval``-split results are
  the only ones the report headlines.
* Failures are recorded as failures. A solver that times out or proves
  infeasibility still writes its run record.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from orion.config import BenchmarkConfig, ExperimentConfig, ScenarioConfig
from orion.data.benchmark_adapters import (
    all_data_sources,
    jobshop_to_scenario,
    load_jobshop,
    load_solomon,
    solomon_to_scenario,
)
from orion.data.scenario_generator import difficulty_config, generate
from orion.domain.plans import Plan
from orion.evaluation.metrics import (
    FailureCategory,
    FailureRecord,
    aggregate_failures,
    detect_failures,
    plan_metric_rows,
)
from orion.experiments.tracker import ExperimentStore, ExperimentTimer, RunRecord
from orion.optimization.registry import run_solver
from orion.planning.comparison import build_recovery_report, compute_churn
from orion.planning.planner import Planner
from orion.planning.replanner import Replanner
from orion.simulation.disruptions import generate_disruptions
from orion.simulation.engine import SimulationEngine

#: The dev/val/eval split of *synthetic* levels. Eval is reserved: it is the
#: only split the report headlines, and the only split a tuning loop may not
#: read. Solomon and job-shop splits are declared in the config.
SYNTHETIC_SPLIT = {"small": "dev", "medium": "val", "large": "eval", "stress": "eval"}


def scenario_for(cfg: ExperimentConfig, benchmark: BenchmarkConfig | None = None) -> Any:
    """Build the scenario a run should optimize, per the config."""
    if benchmark and benchmark.dataset in {"solomon", "jobshop"}:
        if benchmark.dataset == "solomon":
            key = benchmark.instances[0]
            return solomon_to_scenario(load_solomon(cfg.cache_dir, (key,))[key], weights=cfg.weights())
        key = benchmark.instances[0]
        return jobshop_to_scenario(load_jobshop(cfg.cache_dir, (key,))[key], weights=cfg.weights())
    level = cfg.scenario.level
    scen_cfg = difficulty_config(
        level,
        seed=cfg.scenario.seed,
        maintenance_count=cfg.scenario.maintenance_count,
    )
    scen_cfg = dataclass_replace(
        scen_cfg,
        resource_scarcity=cfg.scenario.scarcity,
        deadline_tightness=cfg.scenario.deadline_tightness,
        dependency_rate=cfg.scenario.dependency_rate,
        weight_overrides=cfg.weights().to_dict(),
    )
    return generate(scen_cfg)


def dataclass_replace(obj: Any, **kwargs: Any) -> Any:
    import dataclasses

    return dataclasses.replace(obj, **kwargs)


def run_single(
    scenario: Any,
    solver: str,
    time_budget_s: float,
    *,
    seed: int,
    split: str,
    benchmark: str,
    instance: str,
) -> tuple[Plan, list[FailureRecord]]:
    """Solve one (scenario, solver, budget) triple and record the outcome."""
    planner = Planner(
        default_solver=solver,
        default_time_budget=time_budget_s,
        seed=seed,
    )
    result = planner.plan(scenario, explain=False)
    plan = result.plan
    records = detect_failures(plan, scenario, reference_score=plan.objective.total)
    run = RunRecord(
        benchmark=benchmark,
        instance=instance,
        solver=solver,
        time_budget_s=time_budget_s,
        seed=seed,
        status=str(plan.status),
        runtime_s=max((r.runtime_s for r in plan.solver_runs), default=0.0),
        objective=plan.objective.total,
        service_level=plan.service_level,
        tasks_assigned=plan.tasks_assigned,
        late_tasks=plan.late_tasks,
        travel_km=plan.total_travel_km,
        operating_cost=plan.total_operating_cost,
        violations=len(plan.hard_violations),
        model_variables=max((r.num_variables for r in plan.solver_runs), default=0),
        model_constraints=max((r.num_constraints for r in plan.solver_runs), default=0),
        optimality_gap=_gap(plan),
        split=split,
        notes=(result.warnings[0] if getattr(result, "warnings", None) else ""),
    )
    return plan, records, run


def _gap(plan: Plan) -> float | None:
    gaps = [r.optimality_gap for r in plan.solver_runs if r.optimality_gap is not None]
    if not gaps:
        return None
    g = min(gaps)
    return None if g < 0 else g


def make_record(**kw: Any) -> RunRecord:  # small helper for readability
    return RunRecord(**kw)


# --------------------------------------------------------------------------
# suite 1: scalability
# --------------------------------------------------------------------------

SCALABILITY_LEVELS = ("small", "medium", "large", "stress")


def run_scalability(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    manifest: Any,
    directory: Path,
) -> dict[str, Any]:
    """Runtime and quality vs problem size, for the configured solvers."""
    timer = ExperimentTimer()
    results: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    best_by_level: dict[str, float] = {}

    for level in SCALABILITY_LEVELS:
        sc_cfg = difficulty_config(level, seed=cfg.seed, maintenance_count=cfg.scenario.maintenance_count)
        scenario = generate(sc_cfg, weights=cfg.weights())
        split = SYNTHETIC_SPLIT[level]
        for solver in cfg.solver_names():
            for budget in cfg.time_budgets:
                with timer.time("solve"):
                    plan, records, run = run_single(
                        scenario, solver, budget,
                        seed=cfg.seed, split=split, benchmark="synthetic", instance=level,
                    )
                store.add_run(directory, run)
                results.append({
                    "level": level,
                    "num_tasks": len(scenario.tasks),
                    "num_resources": len(scenario.resources),
                    "solver": solver,
                    "time_budget_s": budget,
                    "status": str(plan.status),
                    "objective": plan.objective.total,
                    "runtime_s": run.runtime_s,
                    "service_level": plan.service_level,
                    "late_tasks": plan.late_tasks,
                    "travel_km": plan.total_travel_km,
                    "optimality_gap": run.optimality_gap,
                    "split": split,
                })
                failures.extend(records)
                # best achievable score at this size, for gap-vs-best
                best_by_level[level] = max(
                    best_by_level.get(level, -1e18), plan.objective.total
                )
    for row in results:
        best = best_by_level.get(row["level"])
        row["gap_to_best"] = (
            (best - row["objective"]) / abs(best) if best not in (None, 0) else None
        )
    return {
        "rows": results,
        "timings": timer.to_dict(),
        "failures": aggregate_failures(failures),
    }


# --------------------------------------------------------------------------
# suite 2: time budget
# --------------------------------------------------------------------------


def run_time_budget(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    manifest: Any,
    directory: Path,
) -> dict[str, Any]:
    """How much quality is lost as the time cap tightens."""
    sc_cfg = difficulty_config(cfg.scenario.level, seed=cfg.seed, maintenance_count=cfg.scenario.maintenance_count)
    scenario = generate(sc_cfg, weights=cfg.weights())
    budgets = (1.0, 5.0, 10.0, 30.0, 60.0)
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for solver in cfg.solver_names():
        for budget in budgets:
            plan, records, run = run_single(
                scenario, solver, budget,
                seed=cfg.seed, split=SYNTHETIC_SPLIT[cfg.scenario.level],
                benchmark="synthetic", instance=cfg.scenario.level,
            )
            store.add_run(directory, run)
            rows.append({
                "solver": solver,
                "time_budget_s": budget,
                "status": str(plan.status),
                "objective": plan.objective.total,
                "runtime_s": run.runtime_s,
                "service_level": plan.service_level,
                "late_tasks": plan.late_tasks,
                "optimality_gap": run.optimality_gap,
            })
            failures.extend(records)
    best = max((r["objective"] for r in rows), default=0.0)
    for row in rows:
        row["fraction_of_best"] = (row["objective"] / best) if best else None
    return {"rows": rows, "failures": aggregate_failures(failures)}


# --------------------------------------------------------------------------
# suite 3: constraint pressure
# --------------------------------------------------------------------------

PRESSURE_SCARCITY = (1.0, 0.8, 0.6, 0.4, 0.2)
PRESSURE_TIGHTNESS = (0.2, 0.5, 0.8, 0.95)


def run_constraint_pressure(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    manifest: Any,
    directory: Path,
) -> dict[str, Any]:
    """Feasibility and quality as resource availability and deadlines tighten."""
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for scarcity in PRESSURE_SCARCITY:
        for tightness in PRESSURE_TIGHTNESS:
            sc_cfg = difficulty_config(
                cfg.scenario.level,
                seed=cfg.seed,
                maintenance_count=cfg.scenario.maintenance_count,
                scarcity=scarcity,
                deadline_tightness=tightness,
            )
            scenario = generate(sc_cfg, weights=cfg.weights())
            solver = cfg.solver_names()[0]
            plan, records, run = run_single(
                scenario, solver, cfg.time_budgets[0],
                seed=cfg.seed, split="eval", benchmark="synthetic",
                instance=f"scarcity{scarcity}_tight{tightness}",
            )
            store.add_run(directory, run)
            rows.append({
                "scarcity": scarcity,
                "deadline_tightness": tightness,
                "status": str(plan.status),
                "objective": plan.objective.total,
                "runtime_s": run.runtime_s,
                "service_level": plan.service_level,
                "late_tasks": plan.late_tasks,
                "feasible": len(plan.hard_violations) == 0,
            })
            failures.extend(records)
    return {"rows": rows, "failures": aggregate_failures(failures)}


# --------------------------------------------------------------------------
# suite 4: disruption stress
# --------------------------------------------------------------------------

STRESS_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)
STRESS_SEVERITIES = ("low", "medium", "high")
STRESS_STRATEGIES = ("none", "local_repair", "full_reopt")


def run_disruption_stress(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    manifest: Any,
    directory: Path,
) -> dict[str, Any]:
    """Service loss and recovery vs disruption rate and severity.

    For each (rate, severity) the suite replans with three strategies and records
    service level, replanning time, churn and remaining violations, so the
    local-repair vs full-re-optimization trade-off is measured, not asserted.
    """
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    for rate in STRESS_RATES:
        for severity in STRESS_SEVERITIES:
            sc_cfg = difficulty_config("medium", seed=cfg.seed, maintenance_count=cfg.scenario.maintenance_count)
            scenario = generate(sc_cfg, weights=cfg.weights())
            solver_name = cfg.solvers[0]
            planner = Planner(
                default_solver=solver_name,
                default_time_budget=cfg.time_budgets[0],
                seed=cfg.seed,
            )
            baseline = planner.plan(scenario, explain=False).plan
            SimulationEngine(scenario, baseline).run()
            disruptions = generate_disruptions(
                scenario, rate=rate, severity=severity, seed=cfg.disruption.seed
            )
            if not disruptions:
                continue
            replanner = Replanner(
                solver=solver_name, time_budget_s=cfg.time_budgets[0], seed=cfg.seed
            )
            for disruption in disruptions[:2]:  # bounded: keep the suite tractable
                out = replanner.replan(scenario, baseline, disruption)
                if out.repair.plan is None and out.full is None:
                    continue
                report = build_recovery_report(
                    disruption_id=out.disruption.id,
                    baseline=baseline,
                    degraded=out.degraded,
                    repaired=out.repair.plan if out.repair.plan is not None else out.full,
                    full=out.full,
                    repair_seconds=out.repair.seconds,
                    full_seconds=out.full_seconds,
                )
                for strategy, plan, seconds in (
                    ("none", out.degraded, 0.0),
                    ("local_repair", out.repair.plan, out.repair.seconds),
                    ("full_reopt", out.full, out.full_seconds),
                ):
                    if plan is None:
                        continue
                    rows.append({
                        "rate": rate,
                        "severity": severity,
                        "strategy": strategy,
                        "disruption_id": disruption.id,
                        "service_level": plan.service_level,
                        "recovered_pct": report.recovery_pct,
                        "replan_seconds": seconds,
                        "churn_fraction": (
                            (compute_churn(baseline, plan).changed_assignments
                             / max(1, compute_churn(baseline, plan).total_before))
                            if strategy != "none" else None
                        ),
                        "violations": len(plan.hard_violations),
                        "objective": plan.objective.total,
                    })
                if out.full is not None:
                    failures.extend(
                        detect_failures(out.full, scenario, reference_score=out.full.objective.total)
                    )
    return {"rows": rows, "failures": aggregate_failures(failures)}


# --------------------------------------------------------------------------
# suite 5: public benchmark solver comparison
# --------------------------------------------------------------------------


def run_solver_comparison(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    manifest: Any,
    directory: Path,
) -> dict[str, Any]:
    """Like-for-like comparison on public instances + the synthetic suite."""
    rows: list[dict[str, Any]] = []
    failures: list[FailureRecord] = []
    datasets: list[BenchmarkConfig] = list(cfg.benchmarks) or [
        BenchmarkConfig(name="solomon", dataset="solomon", instances=("c101", "r101", "rc101"), split="eval")
    ]
    for benchmark in datasets:
        try:
            scenario = scenario_for(cfg, benchmark)
        except Exception as exc:  # noqa: BLE001 - a missing instance must not abort the suite
            failures.append(FailureRecord(
                FailureCategory.TRIVIAL_PROBLEM, benchmark.name,
                detail=f"could not load {benchmark.name}: {exc}",
                root_cause="benchmark download or parse failure",
                consequence="instance excluded from the comparison",
                mitigation="check data/PROVENANCE.md and the network path",
                severity=1,
            ))
            continue
        for solver in cfg.solver_names():
            for budget in cfg.time_budgets:
                plan, records, run = run_single(
                    scenario, solver, budget,
                    seed=cfg.seed, split=benchmark.split,
                    benchmark=benchmark.dataset, instance=benchmark.instances[0] if benchmark.instances else benchmark.name,
                )
                store.add_run(directory, run)
                rows.append({
                    "benchmark": benchmark.name,
                    "dataset": benchmark.dataset,
                    "instance": run.instance,
                    "solver": solver,
                    "time_budget_s": budget,
                    "status": str(plan.status),
                    "objective": plan.objective.total,
                    "runtime_s": run.runtime_s,
                    "service_level": plan.service_level,
                    "late_tasks": plan.late_tasks,
                    "travel_km": plan.total_travel_km,
                    "violations": len(plan.hard_violations),
                    "split": benchmark.split,
                })
                failures.extend(records)
    return {"rows": rows, "failures": aggregate_failures(failures)}


SUITES: dict[str, Callable[..., dict[str, Any]]] = {
    "scalability": run_scalability,
    "time_budget": run_time_budget,
    "constraint_pressure": run_constraint_pressure,
    "disruption_stress": run_disruption_stress,
    "solver_comparison": run_solver_comparison,
}


def run_experiment(
    cfg: ExperimentConfig,
    store: ExperimentStore,
    suite: str,
    *,
    experiment_id: str | None = None,
) -> tuple[Any, Path, dict[str, Any]]:
    """Run one named suite and seal its experiment directory."""
    cfg.validate()
    if suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; available: {sorted(SUITES)}")
    checksums = {d.name: d.sha256 for d in all_data_sources()}
    manifest, directory = store.create(
        kind=f"{cfg.kind}-{suite}",
        description=cfg.description,
        config=cfg.to_dict(),
        objective_weights=cfg.weights().to_dict(),
        seeds=[cfg.seed, cfg.disruption.seed],
        experiment_id=experiment_id,
        dataset_checksums=checksums,
        splits=dict(SYNTHETIC_SPLIT),
    )
    try:
        outcome = SUITES[suite](cfg, store, manifest, directory)
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
        store.finish(directory, status="failed", summary={"error": str(exc), "type": type(exc).__name__})
        raise
    status = "succeeded" if outcome.get("rows") else "empty"
    store.write_artifacts(
        directory,
        metrics={k: v for k, v in outcome.items() if k != "timings"},
        timings=outcome.get("timings", {}),
        tables={"results": outcome.get("rows", [])},
    )
    store.finish(directory, status=status, summary={"rows": len(outcome.get("rows", []))})
    return manifest, directory, outcome
