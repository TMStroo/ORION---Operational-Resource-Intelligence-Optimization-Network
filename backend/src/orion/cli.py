"""``python -m orion`` - the complete workflow without the frontend.

Every major capability is reachable here: scenario creation and validation,
optimization, simulation, disruption, replanning, what-if, benchmarking,
experiment listing, reporting and auditing. A reviewer who never opens the web
UI can still reproduce every number in the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from orion.config import ExperimentConfig
from orion.domain.errors import ConfigError, OrionError
from orion.observability import AuditLog, configure_logging, get_logger

LOG = get_logger("cli")


def _print(obj: Any) -> None:
    """Human output on stdout, machine output on --json."""
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def _load_config(path: str | None) -> ExperimentConfig:
    if not path:
        return ExperimentConfig(kind="adhoc", description="ad-hoc CLI run")
    return ExperimentConfig.from_yaml(path)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    if args.action == "validate":
        cfg.validate()
        print(f"config OK: {args.config}")
        print(f"  kind={cfg.kind} suite-seed={cfg.seed} solvers={list(cfg.solvers)} budgets={list(cfg.time_budgets)}")
    else:
        _print(cfg.to_dict())
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    from orion.data.scenario_generator import difficulty_config, generate, scaling_configs

    if args.action == "create":
        cfg = _load_config(args.config)
        scenario = generate(
            difficulty_config(
                cfg.scenario.level, seed=cfg.scenario.seed, maintenance_count=cfg.scenario.maintenance_count
            )
        )
        out = Path(args.out) if args.out else Path(cfg.output_dir) / f"scenario-{scenario.id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(scenario.to_json(indent=2), encoding="utf-8")
        print(f"wrote {out}")
        print(f"  {scenario.name}: {len(scenario.tasks)} tasks, {len(scenario.resources)} resources, horizon {scenario.horizon}")
    elif args.action == "validate":
        scenario = _load_scenario(args.scenario)
        infeasible = scenario.static_infeasibility()
        demand = scenario.demand_summary()
        print(f"scenario {scenario.id}: {'INFEASIBLE' if infeasible else 'feasible'}")
        if infeasible:
            print(f"  reason: {infeasible}")
            return 2
        print(f"  tasks={demand['tasks']} resources={demand['resources']} demand_min={demand['total_demand_minutes']} available_min={demand['available_minutes']}")
        print(f"  horizon={scenario.horizon}")
        return 0
    elif args.action == "list":
        for cfg in scaling_configs():
            print(f"  {cfg.name:24s} tasks={cfg.num_tasks:4d} teams={cfg.num_teams:3d} vehicles={cfg.num_vehicles:3d}")
    return 0


def _load_scenario(spec: str):
    from orion.data.scenario_generator import difficulty_config, generate

    path = Path(spec)
    if path.exists():
        from orion.data.loaders import load_scenario

        return load_scenario(path)
    # treat a bare level name as a generated preset
    return generate(difficulty_config(spec, seed=2026))


def cmd_optimize(args: argparse.Namespace) -> int:
    from orion.planning.planner import Planner

    cfg = _load_config(args.config)
    scenario = _load_scenario(args.scenario or cfg.scenario.level)
    result = Planner(
        default_solver=args.solver or cfg.solvers[0],
        default_time_budget=args.time_budget or cfg.time_budgets[0],
        seed=cfg.seed,
    ).plan(scenario, explain=not args.no_explain)
    plan = result.plan
    print(plan.summary_line())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(plan.to_json(indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        _print(plan)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from orion.planning.planner import Planner
    from orion.simulation.engine import SimulationEngine

    cfg = _load_config(args.config)
    scenario = _load_scenario(args.scenario or cfg.scenario.level)
    plan = Planner(default_solver=cfg.solvers[0], default_time_budget=cfg.time_budgets[0], seed=cfg.seed).plan(scenario).plan
    result = SimulationEngine(scenario, plan).run()
    print(f"simulation: {result.summary_line()}")
    for event in result.trace[: args.limit]:
        print(f"  {event.time:4d} {event.type.name:16s} {event.subject_id or ''} {event.detail}")
    if args.out:
        from orion.exporting import write_simulation_csv

        write_simulation_csv(result, args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_disrupt(args: argparse.Namespace) -> int:
    from orion.domain.events import Disruption
    from orion.simulation.disruptions import generate_disruptions

    cfg = _load_config(args.config)
    scenario = _load_scenario(args.scenario or cfg.scenario.level)
    if args.rate is not None:
        items = generate_disruptions(scenario, rate=args.rate, severity=args.severity or cfg.disruption.severity, seed=cfg.disruption.seed)
    elif args.kind:
        items = (
            Disruption(
                id=f"{scenario.id}-D01",
                type=args.kind,
                timestamp=args.at if args.at is not None else scenario.horizon.start,
                target_id=args.target,
                magnitude=1.0,
                duration=args.duration,
                description=args.description or f"{args.kind} on {args.target or 'scenario'}",
                severity="MEDIUM",
            ),
        )
    else:
        raise ConfigError("provide --rate or --kind to generate a disruption")
    print(f"generated {len(items)} disruption(s):")
    for d in items:
        print(f"  {d.id} {d.type} @{d.timestamp} target={d.target_id} {d.description}")
    return 0


def cmd_replan(args: argparse.Namespace) -> int:
    from orion.planning.planner import Planner
    from orion.planning.replanner import Replanner
    from orion.simulation.disruptions import generate_disruptions

    cfg = _load_config(args.config)
    scenario = _load_scenario(args.scenario or cfg.scenario.level)
    baseline = Planner(default_solver=cfg.solvers[0], default_time_budget=cfg.time_budgets[0], seed=cfg.seed).plan(scenario).plan
    print(f"baseline: {baseline.summary_line()}")
    items = generate_disruptions(scenario, rate=args.rate if args.rate is not None else cfg.disruption.rate,
                                 severity=args.severity or cfg.disruption.severity, seed=cfg.disruption.seed)
    if not items:
        print("no disruption generated (rate 0) - nothing to replan")
        return 0
    replanner = Replanner(solver=cfg.solvers[0], time_budget_s=cfg.time_budgets[0], seed=cfg.seed)
    for disruption in items:
        out = replanner.replan(scenario, baseline, disruption)
        print(f"\n{disruption.id}: {disruption.type} - {disruption.description}")
        print(f"  impact: {len(out.impact.affected_tasks)} task(s) affected across {len(out.impact.affected_resources)} resource(s)")
        repair = out.repair
        print(f"  local repair: recovered={list(repair.recovered_tasks)} unrecovered={list(repair.unrecovered_tasks)} in {repair.seconds*1000:.1f}ms")
        if out.full:
            print(f"  full re-opt: {out.full.summary_line()} in {out.full_seconds:.2f}s")
    return 0


def cmd_whatif(args: argparse.Namespace) -> int:
    from orion.planning.planner import Planner
    from orion.planning.whatif import WHAT_IF_DESCRIPTIONS, WHAT_IF_OPERATORS, WhatIfEngine

    cfg = _load_config(args.config)
    scenario = _load_scenario(args.scenario or cfg.scenario.level)
    if args.operator not in WHAT_IF_OPERATORS:
        raise ConfigError(
            f"unknown what-if operator {args.operator!r}; available: {sorted(WHAT_IF_OPERATORS)}"
        )
    planner = Planner(
        default_solver=cfg.solvers[0], default_time_budget=cfg.time_budgets[0], seed=cfg.seed
    )
    engine = WhatIfEngine(planner)
    value = float(args.value) if args.value is not None else 0.2
    result = engine.run(scenario, args.operator, value)
    print(f"what-if: {result.question}")
    print(f"  modification: {result.modification}")
    print(f"  baseline: {result.baseline_plan.summary_line()}")
    print(f"  scenario: {result.scenario_plan.summary_line()}")
    print(f"  {'metric':26s} {'baseline':>10s} {'scenario':>10s} {'change':>10s}")
    for delta in result.comparison.deltas[: args.limit]:
        print(f"  {delta.label:26s} {delta.before:10.2f} {delta.after:10.2f} {delta.after - delta.before:+10.2f} {delta.unit}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from orion.experiments.benchmarks import SUITES, run_experiment
    from orion.experiments.tracker import ExperimentStore

    cfg = _load_config(args.config)
    store = ExperimentStore(cfg.output_dir)
    suite = args.suite or "solver_comparison"
    manifest, directory, outcome = run_experiment(cfg, store, suite)
    print(f"experiment {manifest.experiment_id} -> {directory}")
    print(f"  status={manifest.status} rows={len(outcome.get('rows', []))} runs={len(manifest.runs)}")
    for row in outcome.get("rows", [])[: args.limit]:
        print("  ", json.dumps(row, default=str))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from orion.experiments.tracker import ExperimentStore

    store = ExperimentStore(args.root or "experiments")
    if args.action == "list":
        for eid in store.list_ids():
            m = store.load(eid)
            print(f"  {eid:44s} {m.status:10s} runs={len(m.runs):3d} commit={m.git_commit[:8]}")
    elif args.action == "show":
        m = store.load(args.experiment_id)
        _print(m.to_dict())
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from orion.experiments.tracker import ExperimentStore

    store = ExperimentStore("experiments")
    if args.action == "verify":
        problems: list[str] = []
        for eid in store.list_ids():
            m = store.load(eid)
            if not m.git_commit:
                problems.append(f"{eid}: no git commit")
            if not m.objective_weights:
                problems.append(f"{eid}: no objective weights")
            if m.status == "succeeded" and not m.runs:
                problems.append(f"{eid}: marked succeeded with zero runs")
            for run in m.runs:
                if run.status == "succeeded" and run.violations > 0:
                    problems.append(f"{eid}/{run.instance}: succeeded but {run.violations} hard violations")
        if problems:
            print(f"AUDIT FAILED ({len(problems)} problem(s)):")
            for p in problems:
                print(f"  - {p}")
            return 2
        print(f"AUDIT OK: {len(store.list_ids())} experiment(s) consistent")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from orion.report.build import build_report

    result = build_report(experiments_dir=args.experiments, output=args.out, figures_dir=args.figures)
    print(f"report written: {result.html_path}")
    print(f"  pdf:  {result.pdf_path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from orion.demo import run_demo

    summary = run_demo(config_path=args.config, output_dir=args.out, quick=args.quick)
    print("\n=== DEMO COMPLETE ===")
    for line in summary.highlights():
        print(f"  {line}")
    print(f"\nartifacts: {summary.output_dir}")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from orion.data.benchmark_adapters import all_data_sources, load_jobshop, load_solomon

    if args.action == "fetch":
        if args.dataset in ("solomon", "all"):
            inst = load_solomon(args.cache_dir, args.instances or None)
            print(f"solomon: loaded {len(inst)} instance(s): {sorted(inst)[:6]}")
        if args.dataset in ("jobshop", "all"):
            inst = load_jobshop(args.cache_dir, args.instances or None)
            print(f"jobshop: loaded {len(inst)} instance(s): {sorted(inst)[:6]}")
    elif args.action == "sources":
        for src in all_data_sources():
            print(f"\n{src.name}")
            print(f"  source:  {src.source}")
            print(f"  url:     {src.url}")
            print(f"  license: {src.license}")
            print(f"  sha256:  {src.sha256}")
            print(f"  purpose: {src.purpose}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orion",
        description="ORION - operational resource allocation, simulation and disruption recovery",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="validate or show an experiment config")
    p.add_argument("action", choices=["validate", "show"])
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("scenario", help="create/validate/list scenarios")
    p.add_argument("action", choices=["create", "validate", "list"])
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--out")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("optimize", help="produce a plan for a scenario")
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--solver")
    p.add_argument("--time-budget", type=float)
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-explain", action="store_true")
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("simulate", help="execute a plan in the DES engine")
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("disrupt", help="generate a disruption list")
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--rate", type=float)
    p.add_argument("--severity", choices=["low", "medium", "high"])
    p.add_argument("--kind")
    p.add_argument("--target")
    p.add_argument("--at", type=int)
    p.add_argument("--duration", type=int)
    p.add_argument("--description")
    p.set_defaults(func=cmd_disrupt)

    p = sub.add_parser("replan", help="inject a disruption and replan")
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--rate", type=float)
    p.add_argument("--severity", choices=["low", "medium", "high"])
    p.set_defaults(func=cmd_replan)

    p = sub.add_parser("what-if", help="run a what-if comparison")
    p.add_argument("--config")
    p.add_argument("--scenario")
    p.add_argument("--operator", default="availability_drop")
    p.add_argument("--value", type=float)
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_whatif)

    p = sub.add_parser("benchmark", help="run a reproducible experiment suite")
    p.add_argument("--config", required=True)
    p.add_argument("--suite", default=None)
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("experiment", help="inspect experiment records")
    p.add_argument("action", choices=["list", "show"])
    p.add_argument("--root")
    p.add_argument("experiment_id", nargs="?")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("audit", help="verify experiment provenance and integrity")
    p.add_argument("action", choices=["verify"])
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("report", help="build the technical report from artifacts")
    p.add_argument("--experiments", default="experiments")
    p.add_argument("--out", default="docs/report.html")
    p.add_argument("--figures", default="docs/figures")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="run the deterministic end-to-end demo")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--out", default="results/demo")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("data", help="fetch or list benchmark data sources")
    p.add_argument("action", choices=["fetch", "sources"])
    p.add_argument("--dataset", default="all", choices=["solomon", "jobshop", "all"])
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--instances", nargs="*")
    p.set_defaults(func=cmd_data)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except OrionError as exc:
        LOG.error("command failed", extra={"fields": {"error": str(exc), "error_type": type(exc).__name__}})
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130
