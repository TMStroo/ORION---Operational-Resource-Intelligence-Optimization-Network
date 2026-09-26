"""Regression tests for defects found while validating the job-shop benchmark.

Every test here corresponds to a specific, reproduced failure. They are written
so that reintroducing the bug fails the test, and each names the observation that
exposed it.
"""

from __future__ import annotations

import pytest

from orion.data.benchmark_adapters import jobshop_to_scenario, load_jobshop
from orion.optimization import models
from orion.optimization.jobshop_check import validate_jobshop
from orion.optimization.models import SchedulingContext, validate_materialised
from orion.optimization.registry import run_solver

pytestmark = pytest.mark.solver

CACHE = "data/cache"
ALL_SOLVERS = ("HEURISTIC", "LOCAL_SEARCH", "CP_SAT", "MILP", "MIN_COST_FLOW")
# Instances small enough to solve in a unit test, spanning the two families in
# ft06/ft10 (Fisher-Thompson) and la01/la02 (Lawrence).
SMALL = ("ft06", "ft10", "la01", "la02")


@pytest.fixture(scope="module")
def jobshop():
    data = load_jobshop(CACHE, list(SMALL))
    return {name: jobshop_to_scenario(data[name]) for name in SMALL}


def _solved(scenario, solver: str):
    context = SchedulingContext.build(scenario)
    return run_solver(solver, context, _request())


def _request():
    from orion.optimization.registry import SolverRequest

    return SolverRequest(time_limit_s=20.0, seed=0)


# ---------------------------------------------------------------------------
# The heuristic placed a task before its own prerequisite
# ---------------------------------------------------------------------------


def test_heuristic_respects_dependencies_within_a_resource(jobshop):
    """The pool is ordered by value density, not by dependency.

    A value-dense successor could therefore be inserted ahead of its own
    prerequisite. The insertion gate now refuses those positions, and
    `dependency_bounds` alone cannot catch it because a not-yet-placed
    prerequisite contributes no bound.
    """
    for name in SMALL:
        result = _solved(jobshop[name], "HEURISTIC")
        problems = validate_materialised(result.plan.assignments, SchedulingContext.build(jobshop[name]))
        assert not problems, f"{name}: heuristic produced {len(problems)} problem(s): {problems[:3]}"


def test_heuristic_insertion_gate_ignores_unplaced_prerequisites(jobshop):
    """Refusing any insertion whose prerequisite is unplaced deadlocks a job.

    An earlier version of the fix did exactly that and placed 0/36 operations
    on every instance: the job's later operations can never become placeable
    because the predecessor that would allow them is itself refused.
    """
    for name in SMALL:
        result = _solved(jobshop[name], "HEURISTIC")
        assert result.plan.assignments, f"{name}: heuristic placed nothing"


# ---------------------------------------------------------------------------
# Heuristic aborted with IndexError at 250 tasks
# ---------------------------------------------------------------------------


def test_heuristic_survives_a_stress_instance():
    """`schedule_sequence` returns None for an unschedulable sequence.

    The gate indexed `[-1]` of that result, so the whole solver died with
    IndexError, the planner returned an empty plan, and the benchmark recorded
    FEASIBLE with 0 assignments and 0.0s.
    """
    from orion.data.scenario_generator import difficulty_config, generate

    scenario = generate(difficulty_config("stress", seed=2026, num_tasks=250, num_teams=14,
                                         num_vehicles=14, maintenance_count=2))
    context = SchedulingContext.build(scenario)
    result = run_solver("HEURISTIC", context, _request())
    assert str(result.run.status) == "FEASIBLE", result.run.notes
    assert result.plan.assignments, "heuristic placed nothing on the 250-task instance"


# ---------------------------------------------------------------------------
# MILP did not pass its model ordering to the shared materialiser
# ---------------------------------------------------------------------------


def test_milp_and_cpsat_agree_on_validity(jobshop):
    """Both solvers optimise the same objective over the same model family.

    They disagreed because only CP-SAT sorted each resource's sequence before
    materialising: MILP reported 42 precedence violations where CP-SAT had 0.
    """
    for name in SMALL:
        context = SchedulingContext.build(jobshop[name])
        for solver in ("CP_SAT", "MILP"):
            result = run_solver(solver, context, _request())
            problems = validate_materialised(result.plan.assignments, context)
            assert not problems, f"{name}/{solver}: {len(problems)} problem(s): {problems[:3]}"


# ---------------------------------------------------------------------------
# Dependency bounds went stale after a truncation
# ---------------------------------------------------------------------------


def test_bounds_are_recomputed_after_truncation(jobshop, monkeypatch):
    """Bounds must be recomputed whenever a sequence loses a task.

    They were computed once, before the truncation loop, so a dropped task's
    lower bound vanished for every other resource, which was then free to start
    before it. On the 250-task stress scenario this produced T-182 starting at
    652 while its prerequisite T-053 ended at 655.
    """
    calls: list[dict[str, list[str]]] = []
    original = models.dependency_bounds

    def recording(context, sequences, *args, **kwargs):
        calls.append({k: list(v) for k, v in sequences.items()})
        return original(context, sequences, *args, **kwargs)

    monkeypatch.setattr(models, "dependency_bounds", recording)
    context = SchedulingContext.build(jobshop["ft10"])
    run_solver("CP_SAT", context, _request())
    assert len(calls) > 1, "bounds were computed exactly once, so a truncation could not invalidate them"


# ---------------------------------------------------------------------------
# The independent validator must agree with the internal one
# ---------------------------------------------------------------------------


def test_internal_and_independent_validators_agree(jobshop):
    """Two independent checks of the same plan.

    `validate_materialised` reads the scheduling context; `validate_jobshop`
    re-reads the instance's own machine and duration tables. If they disagree
    about a plan, one of them is wrong - which is how two genuine bugs in the
    independent checker were found.
    """
    data = load_jobshop(CACHE, list(SMALL))
    for name in SMALL:
        instance = data[name]
        context = SchedulingContext.build(jobshop[name])
        for solver in ("HEURISTIC", "LOCAL_SEARCH", "CP_SAT", "MILP"):
            result = run_solver(solver, context, _request())
            internal = validate_materialised(result.plan.assignments, context)
            external = validate_jobshop(instance, result.plan.assignments)
            assert not internal, f"{name}/{solver}: internal {internal[:2]}"
            assert external.valid, f"{name}/{solver}: independent {[f.message for f in external.findings][:2]}"


def test_validator_reports_missing_operations_not_ordering_errors(jobshop):
    """A dropped middle operation must not fabricate a precedence violation.

    Comparing consecutive *present* operations of a job checks a relation that
    does not exist when one of them is absent. Doing so reported 12 phantom
    violations on LOCAL_SEARCH's ft10 plan.
    """
    data = load_jobshop(CACHE, ["ft10"])
    instance = data["ft10"]
    context = SchedulingContext.build(jobshop["ft10"])
    result = run_solver("LOCAL_SEARCH", context, _request())
    verdict = validate_jobshop(instance, result.plan.assignments)
    kinds = {f.kind for f in verdict.findings}
    assert "PRECEDENCE_VIOLATED" not in kinds
    # The job-shop scenario sets allow_unassigned, so leaving work out is a
    # scheduling decision, not a validity problem: the default check reports
    # nothing and the plan is valid with partial coverage.
    assert verdict.valid, [f.message for f in verdict.findings][:3]
    assert verdict.operations_assigned < verdict.operations_total, (
        "this test only means something if the solver actually dropped operations"
    )


# ---------------------------------------------------------------------------
# Every solver returns a schedule that passes validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ALL_SOLVERS)
def test_every_solver_returns_a_checkable_plan(jobshop, solver):
    """A solver that cannot be reconstructed must not claim success.

    NOT_APPLICABLE and ERROR are honest outcomes; a plan that fails validation
    is not, and the registry downgrades it to ERROR.
    """
    context = SchedulingContext.build(jobshop["ft10"])
    result = run_solver(solver, context, _request())
    status = str(result.run.status)
    if result.plan.assignments:
        problems = validate_materialised(result.plan.assignments, context)
        assert status != "ERROR", f"{solver}: {status} but plan has problems: {problems[:2]}"
        assert not problems, f"{solver}: {len(problems)} problem(s) on a plan reported as {status}"
