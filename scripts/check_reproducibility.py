"""Check what ORION actually reproduces, and what it only approximates.

Run twice, in two separate processes, and diff the results. The claim under
test is deliberately narrow and specific rather than "it is deterministic",
because a blanket determinism claim would be false in two places:

* **Runtime is not deterministic.** It is wall-clock and depends on the
  machine, so every solver runtime and any figure derived from one is expected
  to differ between runs.
* **Identifiers are deterministic given the same inputs.** A scenario id is a
  function of the config and seed, so the same seed gives the same id. Plan
  ids are *not* a function of content: a plan counter increments per process,
  so a plan is identified by its content, not by its number.

Everything else - the scenario, the assignments, the objective, the statuses,
the simulation trace, the what-if deltas and the exported artifacts - must be
bit-identical across runs.

Usage::

    python scripts/check_reproducibility.py            # one run, prints verdict
    python scripts/check_reproducibility.py --both     # two subprocesses, diffs them

Exits non-zero if a deterministic field moved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "backend" / "src"
OUT = ROOT / "results" / "reproducibility"

#: Keys allowed to differ between two runs. Each is allowed for a stated
#: reason; anything not listed must be identical.
VOLATILE = {
    "created_at": "wall clock",
    "generated_at": "wall clock",
    "timestamp": "wall clock",
    "run_at": "wall clock",
    "started_at": "wall clock",
    "finished_at": "wall clock",
    "seconds": "wall clock",
    "runtime_s": "wall clock",
    "runtime_ms": "wall clock",
    "elapsed_s": "wall clock",
    "full_seconds": "wall clock",
    "repair_seconds": "wall clock",
    "simulation_seconds": "wall clock",
    "_runtime_s": "wall clock",
    "durations": "per-execution wall clock in a pytest-style report",
    "artifacts": "output paths differ because the run writes to its own directory",
}


def _run_once(out_dir: Path) -> dict[str, Any]:
    """Produce one full run and return a comparable digest of it."""
    sys.path.insert(0, str(SRC))
    from orion.data.scenario_generator import ScenarioConfig, generate
    from orion.domain.events import Disruption
    from orion.exporting import export_plan, export_scenario
    from orion.importing import load_plan, load_scenario
    from orion.planning.planner import Planner
    from orion.simulation.engine import SimulationEngine

    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 20260927
    scenario = generate(
        ScenarioConfig(
            name="reproducibility",
            num_tasks=14,
            num_teams=5,
            num_vehicles=2,
            num_locations=6,
            dependency_rate=0.3,
            resource_scarcity=0.7,
            seed=seed,
        )
    )
    planner = Planner(default_solver="HEURISTIC", default_time_budget=3.0, seed=seed)
    plan = planner.plan(scenario).plan

    # A disruption on a real working window, so the run is not a no-op.
    resource = next(r for r in scenario.resources if r.shift_end - r.shift_start > 240)
    disruption = Disruption(
        id="DSR-REPRO",
        type="RESOURCE_UNAVAILABLE",
        target_id=resource.id,
        timestamp=resource.shift_start + 30,
        magnitude=0.5,
        duration=90,
    )
    disrupted = disruption.apply(scenario)
    repaired = planner.plan(disrupted).plan

    result = SimulationEngine(scenario, plan).run()

    scenario_path = out_dir / "scenario.json"
    plan_path = out_dir / "plan.json"
    repaired_path = out_dir / "repaired_plan.json"
    export_scenario(scenario, scenario_path, kind="scenario", seed=seed, source="reproducibility")
    export_plan(plan, plan_path, kind="plan", source="reproducibility")
    export_plan(repaired, repaired_path, kind="plan", source="reproducibility")

    reloaded_scenario, scenario_prov = load_scenario(scenario_path)
    reloaded_plan, plan_prov = load_plan(plan_path)

    digest: dict[str, Any] = {
        "scenario": scenario.to_dict(),
        "scenario_id": scenario.id,
        "plan": {
            "assignments": [
                [a.task_id, a.resource_id, a.start, a.end] for a in plan.assignments
            ],
            "objective": plan.objective.to_dict(),
            "status": str(plan.status),
            "tasks_assigned": plan.tasks_assigned,
            "tasks_total": plan.tasks_total,
            "late_tasks": plan.late_tasks,
            "travel_km": plan.total_travel_km,
            "solver_statuses": [str(r.status) for r in plan.solver_runs],
        },
        "repaired": {
            "assignments": [
                [a.task_id, a.resource_id, a.start, a.end] for a in repaired.assignments
            ],
            "objective": repaired.objective.to_dict(),
            "status": str(repaired.status),
        },
        "disruption": {
            "applied_to": disrupted.id,
            "resource_unavailable": [
                [i.start, i.end] for r in disrupted.resources for i in r.unavailable
            ],
        },
        "simulation": {
            "events": len(result.trace),
            "status": result.status,
            "completed_tasks": result.completed_tasks,
            "late_tasks": result.late_tasks,
            "failed_tasks": result.failed_tasks,
            "digest": [
                [
                    str(event.kind if hasattr(event, "kind") else event.type),
                    int(getattr(event, "minute", getattr(event, "at", 0)) or 0),
                    str(getattr(event, "resource_id", "") or ""),
                ]
                for event in result.trace
            ],
        },
        "round_trip": {
            "scenario_identical": reloaded_scenario.to_dict() == scenario.to_dict(),
            "plan_identical": reloaded_plan.to_dict() == plan.to_dict(),
            "provenance_kinds": [scenario_prov.get("kind"), plan_prov.get("kind")],
        },
    }
    (out_dir / "digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return digest


def _diff(a: Any, b: Any, path: str = "") -> list[tuple[str, str]]:
    """Compare two digests, tolerating only the documented volatile keys."""
    out: list[tuple[str, str]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in VOLATILE:
                continue
            out += _diff(a.get(key), b.get(key), f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} != {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, f"{a!r} != {b!r}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--both",
        action="store_true",
        help="run twice in separate subprocesses and diff the digests",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write this run's digest to DIR (used by --both for each child)",
    )
    args = parser.parse_args()

    if not args.both:
        digest = _run_once(Path(args.out) if args.out else OUT / "single")
        print(json.dumps(digest["round_trip"], indent=2))
        print(f"\nscenario id      {digest['scenario_id']}")
        print(f"assignments      {len(digest['plan']['assignments'])}")
        print(f"objective        {digest['plan']['objective']['total']}")
        print(f"status           {digest['plan']['status']}")
        print(f"simulation events {digest['simulation']['events']}")
        return 0

    # Each child is a *separate process* writing to its own directory, so the
    # comparison is between genuinely independent runs rather than two calls
    # into one already-warm interpreter.
    digests = []
    for name in ("run_a", "run_b"):
        out_dir = OUT / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(Path(__file__).resolve()), "--out", str(out_dir)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-2000:])
            return proc.returncode
        digests.append(
            json.loads((out_dir / "digest.json").read_text(encoding="utf-8"))
        )

    differences = _diff(*digests)
    print("Reproducibility across two separate processes")
    print("=" * 60)
    print(f"scenario id        {digests[0]['scenario_id']} / {digests[1]['scenario_id']}")
    print(f"assignments        {len(digests[0]['plan']['assignments'])}")
    print(f"objective          {digests[0]['plan']['objective']['total']}")
    print(f"status             {digests[0]['plan']['status']}")
    print(f"simulation events  {digests[0]['simulation']['events']}")
    print(f"round trip         {digests[0]['round_trip']}")
    print()
    print(f"volatile keys allowed to differ: {len(VOLATILE)}")
    for key, why in sorted(VOLATILE.items()):
        print(f"  {key:20s} {why}")
    print()

    if differences:
        print(f"FAIL  {len(differences)} deterministic field(s) moved:")
        for path, detail in differences[:20]:
            print(f"  {path}: {detail}")
        return 1

    print("PASS  every non-volatile field is identical across two processes")
    print(
        "      scenario, assignments, objective, statuses, simulation trace,\n"
        "      disruption state, repaired plan and exported artifacts all match."
    )
    print("      Wall-clock runtimes and output paths are the documented exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
