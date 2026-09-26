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
from orion.domain.plans import SolverStatus
from orion.evaluation.metrics import (
    FailureCategory,
    FailureRecord,
    aggregate_failures,
    detect_failures,
    plan_metric_rows,
)
from orion.experiments.tracker import ExperimentStore, ExperimentTimer, RunRecord
from orion.optimization.registry import run_solver
from orion.planning.comparison import build_recovery_report, compute_churn, degraded_plan_from
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

    # A solver that raised must never be recorded as a successful run. This
    # guard exists because it previously was not: an exception inside a solver
    # produced a Plan with no assignments, status FEASIBLE and objective 0.0,
    # and that row went into the benchmark table indistinguishable from a real
    # result. An empty plan from a solver that reports success is legitimate
    # (a genuinely infeasible instance), so the distinction is made on the
    # solver's reported status, not on emptiness.
    # Report the status the *solver* returned, not the plan's own status field.
    # A solver that declines a scenario (MIN_COST_FLOW cannot represent
    # precedence) produces an empty plan whose status reads FEASIBLE, which put
    # a "0.00 objective, FEASIBLE" row into the 250-task table. A run that
    # produced no solution must say so.
    declined = [r for r in plan.solver_runs if r.status == SolverStatus.NOT_APPLICABLE]
    if declined:
        return plan, [
            FailureRecord(
                FailureCategory.NO_ELIGIBLE_PAIR,
                scenario.name,
                f"solver {declined[0].solver} does not apply to this scenario: {declined[0].notes}",
                root_cause=(
                    "the scenario contains a constraint the solver's formulation "
                    "cannot represent (MIN_COST_FLOW has no notion of precedence)"
                ),
                consequence="no row for this solver at this size; not a solver failure",
                mitigation="report the cell as not-applicable rather than as a score",
                severity=1,
            )
        ], RunRecord(
            benchmark=benchmark,
            instance=instance,
            solver=solver,
            time_budget_s=time_budget_s,
            seed=seed,
            status=SolverStatus.NOT_APPLICABLE,
            runtime_s=max((r.runtime_s for r in plan.solver_runs), default=0.0),
            objective=None,
            service_level=None,
            tasks_assigned=0,
            late_tasks=None,
            travel_km=None,
            operating_cost=None,
            violations=None,
            model_variables=0,
            model_constraints=0,
            optimality_gap=None,
            split=split,
            notes=(declined[0].notes or "")[:500],
        )

    failed_runs = [r for r in plan.solver_runs if r.status == SolverStatus.ERROR]
    if failed_runs:
        message = "; ".join(f"{r.solver}: {r.notes or 'no detail recorded'}" for r in failed_runs)
        return plan, [
            FailureRecord(
                FailureCategory.SOLVER_ERROR,
                scenario.name,
                f"solver {failed_runs[0].solver} raised: {message}",
                root_cause="the solver hit an unexpected exception; no plan was produced",
                consequence="this row carries no solution and must not be compared",
                mitigation="fix the solver, or exclude it from this suite",
                severity=3,
            )
        ], RunRecord(
            benchmark=benchmark,
            instance=instance,
            solver=solver,
            time_budget_s=time_budget_s,
            seed=seed,
            status=SolverStatus.ERROR,
            runtime_s=max((r.runtime_s for r in plan.solver_runs), default=0.0),
            objective=None,
            service_level=None,
            tasks_assigned=0,
            late_tasks=None,
            travel_km=None,
            operating_cost=None,
            violations=None,
            model_variables=max((r.num_variables for r in plan.solver_runs), default=0),
            model_constraints=max((r.num_constraints for r in plan.solver_runs), default=0),
            optimality_gap=None,
            split=split,
            notes=message[:500],
        )

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
        violations=sum(1 for v in plan.violations if v.severity >= 1),
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

#: The scalability ladder is an explicit task count rather than the generator's
#: named difficulty presets, so the axis means exactly what the report says it
#: means. Resource count scales with it (see `_resources_for_tasks`) to keep the
#: demand-to-capacity ratio roughly constant, so a growth curve reflects size
#: rather than a steadily worsening ratio.
SCALABILITY_LEVELS = ("small", "medium", "large", "stress")
SCALABILITY_TASK_COUNTS = (10, 25, 50, 100, 250)

#: Resource count per task count. Kept explicit instead of derived so the
#: experiment is reproducible by reading the value.
SCALABILITY_RESOURCES: Mapping[int, int] = {10: 5, 25: 8, 50: 12, 100: 16, 250: 28}

#: Task counts for which an exact solver is worth attempting at all. A 250-task
#: CP-SAT instance is a minutes-long solve, not a useful data point on a plot
#: whose axis is seconds, so it is attempted only to obtain an honest
#: TIME_LIMIT row - never skipped, because a skipped size is an invented
#: measurement.
SCALABILITY_EXACT_CEILING = 100


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
    best_by_size: dict[int, float] = {}

    for num_tasks in SCALABILITY_TASK_COUNTS:
        resources = SCALABILITY_RESOURCES.get(num_tasks, max(3, num_tasks // 9))
        sc_cfg = difficulty_config(
            "stress" if num_tasks >= 250 else ("large" if num_tasks >= 100 else "medium"),
            seed=cfg.seed,
            num_tasks=num_tasks,
            num_teams=max(1, resources // 2),
            num_vehicles=max(1, resources - resources // 2),
            maintenance_count=cfg.scenario.maintenance_count,
        )
        scenario = generate(sc_cfg)
        split = SYNTHETIC_SPLIT.get(sc_cfg.name.rsplit("-", 2)[-1] if "-" in sc_cfg.name else "medium", "eval")
        for solver in cfg.solver_names():
            for budget in cfg.time_budgets:
                with timer.time("solve"):
                    plan, records, run = run_single(
                        scenario, solver, budget,
                        seed=cfg.seed, split=split, benchmark="synthetic", instance=f"synthetic-{num_tasks}t",
                    )
                store.add_run(directory, run)
                results.append({
                    "task_count": num_tasks,
                    "num_tasks": len(scenario.tasks),
                    "num_resources": len(scenario.resources),
                    "solver": solver,
                    "time_budget_s": budget,
                    "status": str(run.status),
                    "objective": plan.objective.total,
                        "tasks_assigned": len(plan.assignments),
                    "runtime_s": run.runtime_s,
                    "service_level": plan.service_level,
                    "late_tasks": plan.late_tasks,
                    "travel_km": plan.total_travel_km,
                    "optimality_gap": run.optimality_gap,
                    "split": split,
                })
                failures.extend(records)
                # best achievable score at this size, for gap-vs-best
                best_by_size[num_tasks] = max(
                    best_by_size.get(num_tasks, -1e18), plan.objective.total
                )
    for row in results:
        best = best_by_size.get(row["task_count"])
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
    scenario = generate(sc_cfg)
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
                "status": str(run.status),
                "objective": plan.objective.total,
                        "tasks_assigned": len(plan.assignments),
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
                resource_scarcity=scarcity,
                deadline_tightness=tightness,
            )
            scenario = generate(sc_cfg)
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
                "status": str(run.status),
                "objective": plan.objective.total,
                        "tasks_assigned": len(plan.assignments),
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


def recovery_fraction(baseline: float, degraded: float, recovered: float) -> float | None:
    """Share of the lost service a recovery strategy gave back.

    ``(recovered - degraded) / (baseline - degraded)``. Returns ``None`` when the
    disruption cost no measurable service, because the fraction is then
    undefined - reporting 0.0 or 100.0 there would be an invention.
    """
    lost = baseline - degraded
    if lost <= 1e-9:
        return None
    return round(max(0.0, min(1.0, (recovered - degraded) / lost)), 6)


def _spread_sample(items: Sequence[Any], *, limit: int) -> list[Any]:
    """Take up to *limit* items, spread evenly across the sequence.

    Deterministic (the input is sorted first) and order-independent, so raising
    the rate genuinely increases coverage instead of only reordering the sample.
    """
    ordered = sorted(items, key=lambda d: getattr(d, "id", str(d)))
    if limit <= 0 or len(ordered) <= limit:
        return list(ordered)
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    step = (len(ordered) - 1) / (limit - 1)
    return [ordered[round(i * step)] for i in range(limit)]


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
            scenario = generate(sc_cfg)
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
            # Bounded, but bounded *per rate* rather than by a flat slice. An
            # earlier `disruptions[:2]` truncated whatever the generator happened
            # to emit first, so raising the disruption rate changed which
            # disruptions were sampled rather than how many were applied - the
            # measured recovery was flat across 5% and 30%. The sample is now
            # deterministic (sorted by id) and spread across the whole list.
            sample = _spread_sample(disruptions, limit=cfg.disruption.sample_per_rate)
            for disruption in sample:
                out = replanner.replan(scenario, baseline, disruption)
                if out.repair.plan is None and out.full is None:
                    continue
                # ReplanOutcome does not carry a "do nothing" plan; the degraded
                # state is derived from the impact, exactly as the demo and the
                # API do, so the suite measures the same baseline->degraded delta.
                degraded_plan = degraded_plan_from(scenario, baseline, out.impact)
                report = build_recovery_report(
                    disruption_id=out.disruption.id,
                    baseline=baseline,
                    degraded=degraded_plan,
                    repaired=out.repair.plan if out.repair.plan is not None else out.full,
                    full=out.full,
                    repair_seconds=out.repair.seconds,
                    full_seconds=out.full_seconds,
                )
                for strategy, plan, seconds in (
                    ("none", degraded_plan, 0.0),
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
                        "objective": plan.objective.total,
                        "tasks_assigned": len(plan.assignments),
                        "late_tasks": plan.late_tasks,
                        "tasks_assigned": plan.tasks_assigned,
                        "travel_km": plan.total_travel_km,
                        "operating_cost": plan.total_operating_cost,
                        "violations": sum(1 for v in plan.violations if v.severity >= 1),
                        "replan_seconds": seconds,
                        # Recovery is measured per strategy against that
                        # strategy's own service level, using the same
                        # baseline -> degraded -> recovered definition the report
                        # uses. `none` is by construction 0% recovery.
                        "recovery_pct": recovery_fraction(
                            report.baseline_service,
                            report.degraded_service,
                            plan.service_level,
                        ),
                        "baseline_service": report.baseline_service,
                        "degraded_service": report.degraded_service,
                        "churn_changed": (
                            None if strategy == "none"
                            else compute_churn(baseline, plan).changed_assignments
                        ),
                        "churn_total_before": (
                            None if strategy == "none"
                            else compute_churn(baseline, plan).total_before
                        ),
                        "churn_fraction": (
                            None if strategy == "none"
                            else round(
                                compute_churn(baseline, plan).changed_assignments
                                / max(1, compute_churn(baseline, plan).total_before),
                                6,
                            )
                        ),
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
                    "status": str(run.status),
                    "objective": plan.objective.total,
                        "tasks_assigned": len(plan.assignments),
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
    started = time.perf_counter()
    try:
        outcome = SUITES[suite](cfg, store, manifest, directory)
    except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
        store.finish(
            directory,
            status="failed",
            summary={"error": str(exc), "type": type(exc).__name__, "rows": 0},
        )
        raise
    wall = time.perf_counter() - started

    # Every suite records real wall-clock. An empty timings.json reads like an
    # experiment that measured nothing, so the harness always fills in what it
    # can know; a suite that does its own stage timing keeps its numbers.
    rows = list(outcome.get("rows", []))
    timings: dict[str, Any] = dict(outcome.get("timings") or {})
    timings.setdefault("suite_wall_seconds", round(wall, 4))
    solver_seconds = sum(
        float(r.get("runtime_s") or 0.0)
        for r in rows
        if isinstance(r, dict) and r.get("runtime_s") is not None
    )
    if solver_seconds and wall:
        timings.setdefault("solver_seconds", round(solver_seconds, 4))
        timings.setdefault("solver_fraction_of_wall", round(solver_seconds / wall, 4))
    timings.setdefault("rows", len(rows))

    summary = summarise(outcome, rows, suite=suite, wall_seconds=wall)
    store.write_artifacts(
        directory,
        metrics={k: v for k, v in outcome.items() if k not in ("timings", "rows")},
        timings=timings,
        tables={"results": rows},
    )
    store.finish(directory, status="succeeded" if rows else "empty", summary=summary)
    return manifest, directory, outcome


def summarise(outcome: dict[str, Any], rows: list[Any], *, suite: str, wall_seconds: float) -> dict[str, Any]:
    """A small, always-populated summary of what an experiment actually found.

    Derived from the rows rather than hand-written, so a suite cannot claim a
    result it did not measure. The status breakdown is included on purpose:
    "three of sixteen runs hit the time limit" is a finding, not a footnote.
    """
    summary: dict[str, Any] = {"suite": suite, "rows": len(rows), "wall_seconds": round(wall_seconds, 4)}
    for key in ("findings", "notes", "transitions", "recommendations"):
        if outcome.get(key):
            summary[key] = outcome[key]
    if not rows:
        return summary

    def column(name: str) -> list[Any]:
        return [r.get(name) for r in rows if isinstance(r, dict) and r.get(name) is not None]

    statuses = column("status")
    if statuses:
        counts: dict[str, int] = {}
        for value in statuses:
            counts[str(value)] = counts.get(str(value), 0) + 1
        summary["status_counts"] = dict(sorted(counts.items()))

    for name in ("objective", "runtime_s", "service_level", "late_tasks", "travel_km", "violations"):
        values = [float(v) for v in column(name) if isinstance(v, (int, float))]
        if values:
            summary.setdefault("numeric", {})[name] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "mean": round(statistics.fmean(values), 6),
                "n": len(values),
            }

    failures = outcome.get("failures")
    if isinstance(failures, list) and failures:
        categories: dict[str, int] = {}
        for record in failures:
            key = str(record.get("category") if isinstance(record, dict) else getattr(record, "category", "UNKNOWN"))
            categories[key] = categories.get(key, 0) + 1
        summary["failure_records"] = len(failures)
        summary["failure_categories"] = dict(sorted(categories.items()))
    return summary
