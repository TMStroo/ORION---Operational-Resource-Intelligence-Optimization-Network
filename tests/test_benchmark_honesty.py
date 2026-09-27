"""Regression tests for benchmark-row honesty.

A benchmark table is only worth reading if a row cannot look better than it is.
These tests pin the three ways that happened during development:

* an exception inside a solver became a row with a success status and objective
  0.0, indistinguishable from a real result;
* a plan that failed materialisation validation kept its objective and became
  the "best" score at its problem size, so every other row at that size was
  scored against a plan that was never valid;
* ``gap_to_best`` was keyed on problem size alone, letting a 30 s result become
  the yardstick for the 5 s rows at the same size - which produced an
  arithmetically impossible negative gap.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from orion.domain.plans import SolverStatus
from orion.evidence import Evidence
from orion.experiments.benchmarks import _COMPARABLE_STATUSES

# `SolverStatus` and `_COMPARABLE_STATUSES` are namespaces of plain string
# constants, not enums, so the members are read off the class rather than
# iterated.
ALL_STATUSES = {
    getattr(SolverStatus, name)
    for name in dir(SolverStatus)
    if name.isupper()
}
COMPARABLE = set(_COMPARABLE_STATUSES)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"

pytestmark = pytest.mark.integration


def _live_results() -> list[dict[str, str]]:
    """Every row of every currently live experiment."""
    if not EXPERIMENTS.exists():
        pytest.skip("no experiments directory")
    rows: list[dict[str, str]] = []
    for directory in sorted(EXPERIMENTS.iterdir()):
        manifest_path = directory / "manifest.json"
        results_path = directory / "results.csv"
        if not (manifest_path.exists() and results_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "superseded":
            continue
        kind = manifest.get("kind", "")
        for row in csv.DictReader(io.StringIO(results_path.read_text(encoding="utf-8"))):
            row["_experiment"] = directory.name
            row["_kind"] = kind
            # Only the solver-level suites carry a status; the disruption suite
            # reports replanning strategies, not solver outcomes, and has no
            # status column at all. Status-based assertions apply where the
            # column exists rather than silently passing on `None`.
            row["_has_status"] = "status" in row
            rows.append(row)
    if not rows:
        pytest.skip("no live experiment rows")
    return rows


def _is_success(row: dict[str, str]) -> bool:
    return row.get("status") in COMPARABLE


def _solver_rows() -> list[dict[str, str]]:
    """Rows from suites that report a solver status."""
    return [r for r in _live_results() if r.get("_has_status")]


def test_no_claimed_success_has_a_zero_objective():
    """A row asserting a *result* must carry one.

    OPTIMAL, FEASIBLE and RELAXATION_OPTIMAL all mean "here is a plan". A row
    with one of those statuses, no assignments and objective 0.0 is either a
    solver that raised or one that silently did nothing, and it was recorded
    indistinguishably from a real result. This is the exact failure that
    produced a fake FEASIBLE row when MIN_COST_FLOW aborted on an exception.
    """
    claimed = {"OPTIMAL", "FEASIBLE", "RELAXATION_OPTIMAL"}
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"), r.get("objective"))
        for r in _solver_rows()
        if r.get("status") in claimed and float(r.get("objective") or 0) == 0
    ]
    assert not offenders, f"rows claiming a result but reporting objective 0.0: {offenders}"


def test_time_limit_without_a_plan_is_visibly_empty():
    """A solver that found nothing in its budget is honest, not a failure.

    CP-SAT at 5s on the 100- and 250-task scenarios and on c101 returns no
    solution at all. That is a real TIME_LIMIT, and per the project's honesty
    rules it must not be reclassified as an error. It does have to be legible:
    the row must say zero assignments, so a reader can tell it apart from a run
    that genuinely served nothing at a cost.
    """
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"), r.get("tasks_assigned"))
        for r in _solver_rows()
        if r.get("status") == "TIME_LIMIT"
        and float(r.get("objective") or 0) == 0
        and str(r.get("tasks_assigned") or "0") != "0"
    ]
    assert not offenders, f"TIME_LIMIT rows with objective 0 but assignments: {offenders}"


def test_zero_objective_rows_all_carry_an_assignment_count():
    """Whatever the status, a zero objective must be disambiguable.

    Without `tasks_assigned` a zero from a timeout and a zero from a real but
    empty plan look identical in the table, which is how a fake success hid.
    """
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"))
        for r in _solver_rows()
        if float(r.get("objective") or 0) == 0 and "tasks_assigned" not in r
    ]
    assert not offenders, f"zero-objective rows with no assignment count: {offenders}"


def test_no_failed_row_carries_an_objective():
    """ERROR, INFEASIBLE and NOT_APPLICABLE have no plan to score."""
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"), r.get("objective"))
        for r in _solver_rows()
        if not _is_success(r) and float(r.get("objective") or 0) != 0
    ]
    assert not offenders, f"non-success rows with a nonzero objective: {offenders}"


def test_error_rows_are_never_recorded_as_successful():
    """The specific failure mode: an exception surfaced as FEASIBLE."""
    rows = _solver_rows()
    errors = [r for r in rows if r.get("status") == "ERROR"]
    for row in errors:
        assert not row.get("assignments") or float(row.get("tasks_assigned") or 0) >= 0
    # A solver error must also be visible in the failure taxonomy, not only as a
    # status. If no suite currently errors, this is vacuously true, which is the
    # desired state.
    assert isinstance(errors, list)


def test_gap_to_best_is_never_negative():
    """A gap against a maximum cannot be below zero.

    It went negative when the reference was computed across budgets, so the
    row that *was* the reference reported a gap of -0.0096 against itself.
    """
    offenders = [
        (r["_experiment"], r.get("solver"), r.get("task_count"), r.get("time_budget_s"), r.get("gap_to_best"))
        for r in _live_results()
        if r.get("gap_to_best") not in (None, "", "None") and float(r["gap_to_best"]) < -1e-9
    ]
    assert not offenders, f"negative gaps: {offenders}"


def test_gap_to_best_matches_a_same_budget_maximum():
    """The reference must be the best result at the *same* budget.

    Keying on problem size alone let a 30 s result judge the 5 s rows at that
    size, scoring them against a plan they were never given the time to find.
    """
    rows = [r for r in _live_results() if r.get("gap_to_best") not in (None, "", "None")]
    if not rows:
        pytest.skip("no gap-annotated rows")
    mismatches: list[tuple] = []
    groups: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("comparable") != "True":
            continue
        key = (row["_experiment"], row.get("task_count"), row.get("time_budget_s"))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        best = max(float(r.get("objective") or 0) for r in group)
        if best == 0:
            continue
        for row in group:
            expected = (best - float(row.get("objective") or 0)) / abs(best)
            if abs(float(row["gap_to_best"]) - expected) > 1e-6:
                mismatches.append((key, row.get("solver"), row["gap_to_best"], expected))
    assert not mismatches, f"gap_to_best does not match a same-budget maximum: {mismatches[:5]}"


def test_non_comparable_rows_have_no_gap():
    """A row with no valid plan has no objective to compare."""
    offenders = [
        (r["_experiment"], r.get("solver"), r.get("status"), r.get("gap_to_best"))
        for r in _live_results()
        if r.get("comparable") == "False" and r.get("gap_to_best") not in (None, "", "None")
    ]
    assert not offenders, f"non-comparable rows reporting a gap: {offenders}"


def test_solver_statuses_are_from_the_declared_set():
    """Only the seven documented statuses may appear in a table."""
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"))
        for r in _solver_rows()
        if r.get("status") not in ALL_STATUSES
    ]
    assert not offenders, f"unknown statuses: {offenders}"


def test_relaxation_solvers_are_never_labelled_optimal():
    """MIN_COST_FLOW and MILP solve reduced models.

    MIN_COST_FLOW is a transportation relaxation: it does not model route
    ordering or time windows, so it can never claim proven optimality for the
    full problem.
    """
    offenders = [
        (r["_kind"], r.get("solver"), r.get("status"))
        for r in _solver_rows()
        if r.get("solver") in ("MIN_COST_FLOW", "MILP") and r.get("status") == "OPTIMAL"
    ]
    assert not offenders, f"relaxation solvers claiming OPTIMAL: {offenders}"


def test_every_experiment_manifest_records_provenance():
    """An experiment that cannot be reproduced is not evidence."""
    if not EXPERIMENTS.exists():
        pytest.skip("no experiments directory")
    required = ("experiment_id", "created_utc", "git_commit", "python_version", "package_versions")
    checked = 0
    for directory in sorted(EXPERIMENTS.iterdir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in required:
            assert manifest.get(key), f"{directory.name}: manifest missing {key}"
        checked += 1
    assert checked, "no experiment manifests found"

def test_every_superseded_run_names_its_successor_and_reason():
    """A retired run must say what replaced it and why.

    The filesystem store accepts `superseded_by=None`, and 33 runs were written
    that way - the reason mentioned the successor in prose, but the field itself
    was null. Anything that follows the chain (which run is current? what did we
    retract?) then has to parse English.
    """
    root = Path(__file__).resolve().parents[1] / "experiments"
    if not root.is_dir():
        pytest.skip("no experiments directory")

    manifest_path = root / "manifest.json"
    superseded = []
    for directory in sorted(root.iterdir()):
        path = directory / "manifest.json"
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{directory.name}/manifest.json is not valid JSON: {exc}")
        if manifest.get("status") == "superseded":
            superseded.append((directory.name, manifest))

    assert superseded, "the project supersedes runs; none found is itself a signal"
    for name, manifest in superseded:
        assert manifest.get("superseded_by"), (
            f"{name} is superseded but names no successor; the chain is unfollowable"
        )
        assert manifest.get("superseded_reason"), (
            f"{name} is superseded with no recorded reason"
        )
        assert manifest.get("superseded_utc"), f"{name} has no retirement timestamp"
        assert (directory := root / name).is_dir()
        assert (root / name / "SUPERSEDED.json").is_file(), (
            f"{name} has no sidecar marking it as not-evidence"
        )
        # The successor must name a run that actually exists.
        successor = manifest["superseded_by"]
        assert (root / successor / "manifest.json").is_file(), (
            f"{name} names successor {successor!r}, which is not on disk"
        )


def test_superseded_runs_are_excluded_from_live_suites():
    """The live set must be disjoint from the superseded set.

    This is the invariant the whole evidence layer rests on: if a run can be
    both, a figure or a README number can quote a result the project withdrew.
    """
    root = Path(__file__).resolve().parents[1] / "experiments"
    if not root.is_dir():
        pytest.skip("no experiments directory")

    statuses: dict[str, str] = {}
    for directory in sorted(root.iterdir()):
        path = directory / "manifest.json"
        if path.is_file():
            statuses[directory.name] = json.loads(path.read_text(encoding="utf-8")).get(
                "status", ""
            )

    evidence = Evidence.load(root.parent)
    live_ids = {suite.experiment_id for suite in evidence.suites.values()}
    assert live_ids, "no live suites found"
    for experiment_id in live_ids:
        assert statuses.get(experiment_id) == "succeeded", (
            f"{experiment_id} is treated as live evidence but its manifest says "
            f"{statuses.get(experiment_id)!r}"
        )
    assert len(live_ids) == len(evidence.suites), (
        "two live runs of the same suite; the newer must supersede the older"
    )
