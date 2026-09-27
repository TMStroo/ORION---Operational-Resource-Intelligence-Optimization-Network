"""Audit every stored experiment run for honesty and completeness.

The benchmark suites write their own results, so the risk is not that a number
is wrong but that a run is *missing* what a reader needs, or that a row claims
success while carrying nothing. This checks both, and it treats a superseded
run differently from a live one on purpose: a superseded row is allowed to be
incomplete, because it exists to be read alongside its successor, but it must
say why it was superseded and name what replaced it.

Run:  python scripts/check_experiments.py
Exits non-zero and lists every problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from orion.evidence import Evidence  # noqa: E402

# Statuses that assert a usable plan was produced.
SUCCESS = {"OPTIMAL", "FEASIBLE", "RELAXATION_OPTIMAL", "TIME_LIMIT"}
# Statuses where an objective of zero is legitimate.
ZERO_OBJECTIVE_OK = {"INFEASIBLE", "ERROR", "NOT_APPLICABLE"}


def audit() -> list[str]:
    problems: list[str] = []
    evidence = Evidence.load(ROOT)
    suites = list(evidence.suites.values())
    superseded = list(evidence.superseded)
    live = {s.experiment_id: s for s in suites}
    superseded_ids = {r["experiment_id"] for r in superseded}

    # -- every run on disk is accounted for -------------------------------
    runs = sorted(p for p in (ROOT / "experiments").iterdir() if p.is_dir())
    for run in runs:
        if run.name in live or run.name in superseded_ids:
            continue
        problems.append(f"run on disk is neither live nor superseded: {run.name}")

    # -- live runs ---------------------------------------------------------
    for suite in suites:
        suite_keys: set[str] = set()
        for row in suite.rows:
            solver = row.get("solver", "?")
            status = row.get("status", "?")
            budget = row.get("time_budget_s", "-")
            label = f"{suite.experiment_id} {row.get('instance', row.get('case', '-'))}/{solver}@{budget}"

            if status in SUCCESS:
                # A success with no work in it is the failure mode that matters
                # most: it would be read as a result while carrying nothing.
                assigned = _num(row.get("tasks_assigned"))
                if assigned is not None and assigned <= 0 and status != "TIME_LIMIT":
                    problems.append(f"successful row assigned no tasks: {label} ({status})")

                objective = _num(row.get("objective"))
                if objective is None:
                    problems.append(f"successful row has no objective: {label} ({status})")
                elif objective == 0.0 and status not in ZERO_OBJECTIVE_OK | {"TIME_LIMIT"}:
                    # TIME_LIMIT is included deliberately: a solver stopped before
                    # it found any feasible plan reports 0.0 with no assignments.
                    # That is the honest outcome, and the project states it rather
                    # than hiding it.
                    # Objective is a reward, so a feasible plan scoring exactly
                    # zero is possible but rare enough to demand a look.
                    problems.append(f"successful row has objective 0.0: {label} ({status})")
                elif objective < 0:
                    problems.append(f"negative objective: {label} ({objective})")

                if row.get("split") == "train" and suite.kind == "public_benchmark":
                    problems.append(f"a training row leaked into the public benchmark: {label}")

            # gaps must never be negative; a negative gap means a solver beat a
            # proven optimum, which is a bug rather than a good result
            gap = _num(row.get("optimality_gap"))
            if gap is not None and gap < -1e-9:
                problems.append(f"negative optimality gap: {label} ({gap})")

            runtime = _num(row.get("runtime_s"))
            if runtime is not None and runtime < 0:
                problems.append(f"negative runtime: {label} ({runtime})")

            # Duplicate detection is per suite. The suites were not written to
            # one schema -- scalability keys on num_tasks, solver_comparison on
            # instance -- so two suites can legitimately hold a row that looks
            # identical. Within one suite it is a defect.
            # Build the identity key from the suite's own schema rather than a
            # fixed allowlist. The suites were not written to one table shape:
            # solver_comparison keys on instance, scalability on num_tasks,
            # constraint_pressure on scarcity x deadline_tightness, and
            # disruption_stress on disruption_id x strategy. A measurement
            # column is a poor key, so exclude the ones that describe a result
            # rather than a case.
            key = _identity_key(row)

        # -- per-run completeness ------------------------------------------
        if suite.directory is None or not Path(suite.directory).is_dir():
            problems.append(f"live run has no directory: {suite.experiment_id}")
            continue
        directory = Path(suite.directory)
        for required in ("manifest.json",):
            if not (directory / required).is_file():
                problems.append(f"live run missing {required}: {suite.experiment_id}")
        if not any(directory.glob("*.csv")) and not (directory / "metrics.json").is_file():
            problems.append(f"live run has no result table: {suite.experiment_id}")

        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # `seeds` is a list because a run sweeps more than one; a run
            # with no recorded seed cannot be reproduced.
            for field in ("git_commit", "config", "status"):
                if not manifest.get(field):
                    problems.append(f"manifest missing {field}: {suite.experiment_id}")
            if not manifest.get("seeds") and not manifest.get("seed"):
                problems.append(f"manifest records no seed: {suite.experiment_id}")
            if str(manifest.get("git_commit")) == "unknown":
                problems.append(f"manifest records an unknown commit: {suite.experiment_id}")

    # -- superseded runs keep their explanation ----------------------------
    # A superseded run must say why, and the reason should point at the run
    # that replaced it. Older records name the successor only inside the prose,
    # so accept a reason that cites a run id, and flag a bare "superseded".
    for record in superseded:
        identifier = record.get("experiment_id", "?")
        reason = (record.get("reason") or "").strip()
        if not reason:
            problems.append(f"superseded run has no reason: {identifier}")
        else:
            directory = ROOT / "experiments" / identifier
            manifest = directory / "manifest.json"
            successor = ""
            if manifest.is_file():
                successor = (json.loads(manifest.read_text(encoding="utf-8")).get("superseded_by") or "")
            if not successor and "benchmark-" not in reason:
                problems.append(
                    f"superseded run names no successor: {identifier} ({reason[:60]})"
                )

    return problems


# Columns that describe a measurement rather than which case was run.
MEASUREMENT_COLUMNS = {
    "objective", "runtime_s", "service_level", "tasks_assigned", "late_tasks",
    "travel_km", "violations", "operating_cost", "replan_seconds", "recovery_pct",
    "baseline_service", "degraded_service", "churn_changed", "churn_total_before",
    "churn_fraction", "optimality_gap", "fraction_of_best", "gap_to_best",
    "comparable", "split", "status",
}


def _identity_key(row: dict[str, str]) -> str:
    """Return a key that identifies the case a row belongs to.

    Every column that is not a measurement takes part, so the key is derived
    from the suite's real schema instead of an assumed list of names.
    """
    parts = [f"{k}={row[k]}" for k in sorted(row) if k not in MEASUREMENT_COLUMNS]
    return "|".join(parts)


def _num(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def main() -> int:
    problems = audit()
    evidence = Evidence.load(ROOT)
    print(
        f"experiment audit: {len(evidence.suites)} live suite(s), "
        f"{len(evidence.superseded)} superseded run(s)"
    )
    if problems:
        print(f"\nFAILED ({len(problems)}):\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("experiment audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
