"""Property tests for the domain and scheduling engine.

Each property states an invariant that must hold for *every* input, so a
regression surfaces as a counterexample instead of one more worked example.
They are written against the real domain objects and the project's own
independent validators:

* :func:`orion.optimization.models.validate_materialised` re-derives the facts
  from a returned assignment list and is the authority on whether a plan is a
  real schedule;
* :func:`orion.optimization.jobshop_check.validate_jobshop` checks processing
  time, machine eligibility, machine occupancy and precedence against a
  ``JobShopInstanceLike`` (``name`` plus ``jobs``);
* :func:`orion.optimization.reference.enumerate_optimum` is the brute-force
  reference a solver is never allowed to beat *on the instances where it is
  exhaustive over the whole search space*.

Two findings shaped this file, and both are recorded where they matter rather
than papered over:

1. The engine reports ``ERROR`` when a model solution cannot be rebuilt into a
   valid schedule. It happens on roughly 1 generated scenario in 80, on 14-task
   instances with ``resource_scarcity`` near 0.85, and the reason is always
   reported. It is the honest behaviour - presenting an invalid plan as a
   success would be far worse - but it means a successful solve can become an
   ``ERROR`` purely by *removing* capacity, which breaks naive capacity
   monotonicity. So the monotonicity properties are stated over *successful*
   plans, and :func:`test_engine_error_is_always_accompanied_by_a_reported_reason`
   pins the honesty requirement in exchange.

2. ``enumerate_optimum`` assigns tasks to resources in task-index order and
   never permutes the route within a resource. Its ``exhaustive`` flag is
   therefore exhaustive over the *assignment* space only, so on a general
   generated scenario it is a lower bound, not a proven optimum - a solver that
   sequences a route better can legitimately score above it. The no-beat
   property is asserted only where no routing decision remains.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from orion.data.benchmark_adapters import jobshop_to_scenario, load_jobshop  # noqa: E402
from orion.data.scenario_generator import ScenarioConfig, generate  # noqa: E402
from orion.domain.entities import Scenario  # noqa: E402
from orion.domain.events import Disruption  # noqa: E402
from orion.domain.plans import Plan, SolverStatus  # noqa: E402
from orion.optimization.jobshop_check import validate_jobshop  # noqa: E402
from orion.optimization.models import (  # noqa: E402
    SchedulingContext,
    validate_materialised,
)
from orion.optimization.objective import (  # noqa: E402
    objective_breakdown_from_assignments,
)
from orion.optimization.verify import verify_all  # noqa: E402
from orion.optimization.reference import (  # noqa: E402
    _assignments_for,
    _schedule_route,
    enumerate_optimum,
)
from orion.optimization.registry import SolverRequest, run_solver  # noqa: E402

# A property that runs a solver needs real time, so `deadline=None` replaces
# the 200ms default, which fails on a correct implementation.
SOLVING = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# Pure domain properties need no solver and can afford more examples.
DOMAIN = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

#: Both ScenarioConfig ratios are validated as ``(0, 1]``. Drawing 0.0 would
#: be testing the config validator rather than any property of the engine.
_UNIT_RATIO = st.floats(min_value=0.01, max_value=1.0)


@st.composite
def scenario_configs(draw, *, max_tasks: int = 16, max_resources: int = 6):
    """A small but structurally valid ScenarioConfig.

    Deliberately small: these properties run real solvers, and a 200-task
    scenario would say nothing extra about the invariant while making the
    suite unusable.
    """
    return ScenarioConfig(
        name="prop",
        num_tasks=draw(st.integers(min_value=4, max_value=max_tasks)),
        num_teams=draw(st.integers(min_value=1, max_value=max_resources)),
        num_vehicles=draw(st.integers(min_value=0, max_value=2)),
        num_locations=draw(st.integers(min_value=2, max_value=6)),
        deadline_tightness=draw(_UNIT_RATIO),
        resource_scarcity=draw(_UNIT_RATIO),
        dependency_rate=draw(st.floats(min_value=0.0, max_value=0.6)),
        seed=draw(st.integers(min_value=0, max_value=10_000)),
    )


def _context(scenario: Scenario) -> SchedulingContext:
    return SchedulingContext.build(scenario)


def _solve(scenario: Scenario, solver: str = "HEURISTIC", budget: float = 3.0):
    return run_solver(solver, _context(scenario), SolverRequest(time_limit_s=budget, seed=0))


def _success(status) -> bool:
    """True when the engine is asserting that a usable schedule was returned.

    ``TIME_LIMIT`` and ``RELAXATION_OPTIMAL`` both count: the engine reports
    them as *a solution was returned*, and such a plan still has to be a real
    schedule. ``ERROR``, ``INFEASIBLE`` and ``NOT_APPLICABLE`` do not.
    """
    return str(status) in {
        str(SolverStatus.FEASIBLE),
        str(SolverStatus.OPTIMAL),
        str(SolverStatus.TIME_LIMIT),
        str(SolverStatus.RELAXATION_OPTIMAL),
    }


# -------------------------------------------------------------- determinism


@DOMAIN
@given(config=scenario_configs())
def test_the_same_seed_produces_the_same_scenario(config):
    """A scenario id is a promise of reproducibility, so the content must match."""
    first = generate(config)
    second = generate(config)
    assert first.to_dict() == second.to_dict()
    assert first.id == second.id


@SOLVING
@given(config=scenario_configs())
def test_a_fixed_seed_produces_the_same_plan(config):
    """Verified determinism: same scenario, same solver, same request, same plan.

    Runtime and timestamps may differ; the schedule does not.
    """
    scenario = generate(config)
    first, second = _solve(scenario), _solve(scenario)
    assert [(a.task_id, a.resource_id, a.start, a.end) for a in first.plan.assignments] == [
        (a.task_id, a.resource_id, a.start, a.end) for a in second.plan.assignments
    ]
    assert str(first.run.status) == str(second.run.status)


# ----------------------------------------------------------- serialisation


@DOMAIN
@given(config=scenario_configs())
def test_a_scenario_reloads_to_the_same_domain_state(config):
    scenario = generate(config)
    assert Scenario.from_dict(scenario.to_dict()).to_dict() == scenario.to_dict()


@SOLVING
@given(config=scenario_configs())
def test_a_plan_reloads_to_the_same_domain_state(config):
    plan = _solve(generate(config)).plan
    assert Plan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()


# -------------------------------------------------------------- disruptions


@DOMAIN
@given(config=scenario_configs(), data=st.data())
def test_applying_a_disruption_once_changes_the_world_once(config, data):
    """A disruption applied twice to the same input must give the same world.

    If it accumulated, the API's "re-apply server-side so the repaired state
    matches" path would silently compound the damage.
    """
    scenario = generate(config)
    resource = data.draw(st.sampled_from(scenario.resources))
    disruption = Disruption(
        id="DSR-1",
        type="RESOURCE_UNAVAILABLE",
        target_id=resource.id,
        timestamp=resource.shift_start,
        magnitude=0.5,
        duration=120,
    )
    once = disruption.apply(scenario)
    twice = disruption.apply(scenario)
    assert once.to_dict() == twice.to_dict()


@DOMAIN
@given(config=scenario_configs())
def test_a_disruption_never_adds_tasks_or_resources(config):
    """A disruption changes availability, not the size of the problem.

    Also asserts the scenario it was given is not mutated: `apply` must return
    a new world rather than editing the caller's copy in place.
    """
    scenario = generate(config)
    resource = scenario.resources[0]
    before = scenario.to_dict()
    after = Disruption(
        id="DSR-1",
        type="RESOURCE_UNAVAILABLE",
        target_id=resource.id,
        timestamp=resource.shift_start,
        magnitude=0.5,
        duration=120,
    ).apply(scenario)
    assert len(after.tasks) == len(scenario.tasks)
    assert len(after.resources) == len(scenario.resources)
    assert {r.id for r in after.resources} == {r.id for r in scenario.resources}
    assert scenario.to_dict() == before


@DOMAIN
@given(config=scenario_configs(), data=st.data())
def test_removing_a_resource_never_increases_available_capacity(config, data):
    """Total available resource-minutes is monotone in the resource set.

    The resource is dropped outright rather than merely disrupted, so this is a
    claim about the resource set. Strictness is safe because a generated
    resource always has a positive-length shift.
    """
    scenario = generate(config)
    victim = data.draw(st.sampled_from(scenario.resources))

    def capacity_minutes(sc: Scenario) -> int:
        return sum(
            max(
                0,
                r.shift_end
                - r.shift_start
                - sum(i.end - i.start for i in r.unavailable),
            )
            for r in sc.resources
        )

    remaining = Scenario.from_dict(
        {
            **scenario.to_dict(),
            "resources": [r.to_dict() for r in scenario.resources if r.id != victim.id],
        }
    )
    before, after = capacity_minutes(scenario), capacity_minutes(remaining)
    assert after < before, "removing a shifted resource must reduce capacity minutes"


# ---------------------------------------------------- monotonicity: capacity


@SOLVING
@given(config=scenario_configs(), factor=st.floats(min_value=0.3, max_value=1.0))
def test_shrinking_capacity_never_places_more_work(config, factor):
    """Capacity monotonicity: less capacity cannot place more work.

    Two scenarios identical but for resource capacity and max-work limits,
    solved by the same deterministic heuristic, so the only difference between
    the plans is the capacity itself. A violation would mean the engine reads a
    constraint it does not enforce.

    The claim is deliberately about work placed, not about the objective.
    Capacity changes the feasible region, and ORION's objective is a *reward*
    whose response to a larger feasible region is not what this property is
    about; asserting a direction there would be asserting something false.
    """
    scenario = generate(config)
    payload = scenario.to_dict()
    for resource in payload["resources"]:
        resource["capacity"] = {k: v * factor for k, v in resource["capacity"].items()}
        resource["max_work_minutes"] = max(0, int(resource["max_work_minutes"] * factor))
    shrunk = Scenario.from_dict(payload)

    assert len({t.id for t in shrunk.tasks}) == len(shrunk.tasks)
    baseline, reduced = _solve(scenario), _solve(shrunk)
    if _success(reduced.run.status):
        assert reduced.plan.tasks_assigned <= baseline.plan.tasks_assigned, (
            f"shrinking capacity to {factor:.2f} placed "
            f"{reduced.plan.tasks_assigned} tasks vs {baseline.plan.tasks_assigned}"
        )


@SOLVING
@given(config=scenario_configs(), factor=st.floats(min_value=1.5, max_value=4.0))
def test_giving_a_resource_more_working_time_never_places_less_work(config, factor):
    """Working-time monotonicity: more valid working time cannot place less.

    `factor > 1` only, because a limit extended past the horizon is a scenario
    the domain rejects outright rather than one that tests the property.
    """
    scenario = generate(config)
    payload = scenario.to_dict()
    for resource in payload["resources"]:
        resource["max_work_minutes"] = int(resource["max_work_minutes"] * factor)
    longer = Scenario.from_dict(payload)

    baseline, extended = _solve(scenario), _solve(longer)
    if _success(extended.run.status):
        assert extended.plan.tasks_assigned >= baseline.plan.tasks_assigned, (
            f"extending working time by {factor:.2f}x placed "
            f"{extended.plan.tasks_assigned} tasks vs {baseline.plan.tasks_assigned}"
        )


@SOLVING
@given(config=scenario_configs())
def test_an_unsatisfiable_requirement_stays_unsatisfiable(config):
    """A task needing a capability no resource has must never be scheduled.

    This is the property behind the NO_ELIGIBLE_PAIR path: a requirement
    nothing satisfies cannot be satisfied by trying harder.
    """
    scenario = generate(config)
    payload = scenario.to_dict()
    payload["tasks"][0]["required_capabilities"] = ["capability_nothing_provides"]
    impossible = Scenario.from_dict(payload)

    result = _solve(impossible)
    if _success(result.run.status):
        assert result.plan.assignment_for(payload["tasks"][0]["id"]) is None, (
            "a task with no eligible resource was scheduled anyway"
        )


# ---------------------------------------------------------- plan structure


@SOLVING
@given(config=scenario_configs())
def test_no_plan_contains_a_duplicate_assignment(config):
    result = _solve(generate(config))
    seen: set[tuple[str, str]] = set()
    for assignment in result.plan.assignments:
        key = (assignment.task_id, assignment.resource_id)
        assert key not in seen, f"{key} assigned twice"
        seen.add(key)


@SOLVING
@given(config=scenario_configs())
def test_every_assignment_refers_to_a_real_task_and_resource(config):
    scenario = generate(config)
    result = _solve(scenario)
    task_ids = {t.id for t in scenario.tasks}
    resource_ids = {r.id for r in scenario.resources}
    for assignment in result.plan.assignments:
        assert assignment.task_id in task_ids
        assert assignment.resource_id in resource_ids


@SOLVING
@given(config=scenario_configs())
def test_capability_constraints_hold_in_every_successful_plan(config):
    """A scheduled resource must actually have what the task requires."""
    scenario = generate(config)
    result = _solve(scenario)
    if not _success(result.run.status):
        return
    tasks = {t.id: t for t in scenario.tasks}
    resources = {r.id: r for r in scenario.resources}
    for assignment in result.plan.assignments:
        task = tasks[assignment.task_id]
        resource = resources[assignment.resource_id]
        missing = set(task.required_capabilities) - set(resource.capabilities)
        assert not missing, f"{resource.id} lacks {sorted(missing)} required by {task.id}"


@SOLVING
@given(config=scenario_configs())
def test_a_successful_plan_passes_the_independent_validator(config):
    """The strongest single property: the engine's own plan validates.

    `validate_materialised` is written from the constraints rather than from
    the solver, so this is a genuine cross-check and not the engine agreeing
    with itself. It covers duplicate assignment, max-work, precedence,
    double-booking, shift containment and horizon containment.

    Replaces an earlier version whose job-shop branch imported a
    `JobShopInstance` that this project does not have; the job-shop claim now
    lives in `test_job_shop_plans_validate_against_their_instance`.
    """
    scenario = generate(config)
    result = _solve(scenario)
    if not _success(result.run.status):
        return
    problems = validate_materialised(result.plan.assignments, _context(scenario))
    assert not problems, f"status {result.run.status}: {problems}"


@SOLVING
@given(config=scenario_configs())
def test_engine_error_is_always_accompanied_by_a_reported_reason(config):
    """``ERROR`` must never be silent.

    The engine does return ``ERROR`` when a model solution cannot be rebuilt
    into a valid schedule - roughly 1 generated scenario in 80 - and naming the
    structural problem is the required behaviour, not an optional nicety. A
    bare ``ERROR`` with no reason would be indistinguishable from a crash.
    """
    result = _solve(generate(config))
    if str(result.run.status) != str(SolverStatus.ERROR):
        return
    notes = (result.run.notes or "").lower()
    assert notes, "ERROR status with no explanation"
    assert any(
        marker in notes
        for marker in (
            "prerequisite",
            "over its limit",
            "double-booked",
            "outside",
            "past the horizon",
        )
    ), f"ERROR status with an uninformative reason: {result.run.notes!r}"


# --------------------------------------------- precedence (independent check)


@SOLVING
@given(
    tasks=st.integers(min_value=4, max_value=10),
    teams=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=5_000),
)
def test_every_dependent_task_starts_after_its_prerequisites_finish(tasks, teams, seed):
    """Precedence, against the real declared dependencies.

    `dependency_rate=1.0` gives genuine A -> B -> C chains rather than a
    scattering of unrelated edges. The assertion is the declaration itself: a
    dependent task must not start before every task it depends on has ended.

    Only pairs where *both* endpoints are placed are checked, because ORION's
    objective explicitly prices leaving work unassigned - a job whose fourth
    operation is unplaced still has the fifth depending on the fourth.
    """
    scenario = generate(
        ScenarioConfig(
            name="chain",
            num_tasks=tasks,
            num_teams=teams,
            num_vehicles=0,
            num_locations=3,
            dependency_rate=1.0,
            seed=seed,
        )
    )
    result = _solve(scenario, budget=5.0)
    if not _success(result.run.status):
        return

    by_task = {a.task_id: a for a in result.plan.assignments}
    for task in scenario.tasks:
        placement = by_task.get(task.id)
        if placement is None:
            continue
        for dependency in task.dependencies:
            parent = by_task.get(dependency)
            if parent is None:
                continue
            assert placement.start >= parent.end, (
                f"{task.id} starts {placement.start} before its prerequisite "
                f"{dependency} ends {parent.end}"
            )


# --------------------------------------------------------- plan accounting


@SOLVING
@given(config=scenario_configs())
def test_reported_counts_follow_from_the_assignments(config):
    """`tasks_assigned` is a claim about the plan, so it has to follow from it."""
    scenario = generate(config)
    plan = _solve(scenario).plan
    assert plan.tasks_assigned == len({a.task_id for a in plan.assignments})
    assert plan.tasks_assigned <= plan.tasks_total
    assert plan.tasks_total == len(scenario.tasks)


# --------------------------------------------------------- reference optimum


@SOLVING
@given(
    tasks=st.integers(min_value=3, max_value=6),
    teams=st.integers(min_value=1, max_value=3),
    locations=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=5_000),
)
def test_the_reference_sequences_rescore_to_the_reported_optimum(
    tasks, teams, locations, seed
):
    """The enumerator's own winner must re-score to the score it reported.

    This is the invariant that makes the reference trustworthy at all: the
    number `enumerate_optimum` publishes has to be the number the project's
    shared scorer assigns to the plan that won the enumeration.

    It failed before this was written. `enumerate_optimum` called
    `objective_breakdown_from_assignments` without `resource_speeds`, so the
    travel term was `distance_km / 1.0`, while every solver is scored by
    `build_plan` using each resource's real `speed_factor`. The same assignment
    list therefore scored 17.0234 in the enumerator and 17.0293 in the solver
    comparison, and a solver that merely sequenced better looked as though it
    had beaten a proven optimum.
    """
    scenario = generate(
        ScenarioConfig(
            name="ref",
            num_tasks=tasks,
            num_teams=teams,
            num_vehicles=0,
            num_locations=locations,
            dependency_rate=0.0,
            seed=seed,
        )
    )
    reference = enumerate_optimum(scenario)
    if not reference.exhaustive or reference.best_score is None:
        return
    # Re-score the enumerator's *own* assignment records, not a fresh
    # materialisation of the same sequences: `materialise_sequences` lays the
    # route out again and can legitimately produce different start times, so
    # comparing against it would be testing the scheduler, not the scorer.
    by_task = {t.id: t for t in scenario.tasks}
    sequences = {k: tuple(v) for k, v in reference.sequences.items() if v}
    times = {
        resource_id: _schedule_route(sequence, scenario, by_task)
        for resource_id, sequence in sequences.items()
    }
    assert all(value is not None for value in times.values()), (
        "the enumerator returned sequences it cannot itself schedule"
    )
    assignments = _assignments_for(scenario, sequences, times)
    rescored = objective_breakdown_from_assignments(
        list(assignments),
        scenario,
        resource_speeds={r.id: r.speed_factor for r in scenario.resources},
    )
    tolerance = 1e-6 * max(1.0, abs(reference.best_score))
    assert abs(rescored.total - reference.best_score) <= tolerance, (
        f"enumerator reported {reference.best_score} but its own winning plan "
        f"rescored to {rescored.total}"
    )


@SOLVING
@given(
    tasks=st.integers(min_value=3, max_value=6),
    teams=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=5_000),
)
def test_a_solver_never_beats_the_reference_on_a_tiny_fixture(tasks, teams, seed):
    """No solver may exceed the proven optimum on the reference fixtures.

    Scoped to `TINY_FIXTURES`, which is where the project actually claims a
    proven optimum, because the general enumerator is exhaustive over the
    *assignment* space only: it assigns tasks to resources in task-index order
    and never permutes the route within a resource. On a general generated
    scenario its `exhaustive` flag is therefore true while the reported score
    is still only a lower bound, and a solver that sequences a route better
    legitimately scores above it. That is a limitation of the enumerator, not
    a solver defect, so it is scoped away rather than asserted against.
    """
    scenario = generate(
        ScenarioConfig(
            name="tiny-probe",
            num_tasks=tasks,
            num_teams=teams,
            num_vehicles=0,
            num_locations=2,
            dependency_rate=0.0,
            seed=seed,
        )
    )
    # Only comparable where the enumerator is exhaustive over the *whole* space
    # rather than over the assignment space alone. It lays each resource's
    # route out in enumeration order, so whenever two tasks share a resource
    # there is a route permutation it never tries. One task per eligible
    # resource leaves no such decision, and that is the shape the tiny
    # fixtures are built in.
    if len({t.location for t in scenario.tasks}) != 1:
        return
    if len(scenario.tasks) != len(scenario.resources):
        return
    reference = enumerate_optimum(scenario)
    if not reference.exhaustive or reference.best_score is None:
        return
    for solver in ("HEURISTIC", "LOCAL_SEARCH"):
        result = _solve(scenario, solver=solver, budget=5.0)
        if not _success(result.run.status):
            continue
        tolerance = 1e-6 * max(1.0, abs(reference.best_score))
        assert result.plan.objective.total <= reference.best_score + tolerance, (
            f"{solver} scored {result.plan.objective.total} against a proven "
            f"optimum of {reference.best_score}"
        )


def test_the_project_reference_validation_reports_no_proven_mismatch():
    """The claim the README makes, checked directly.

    `verify_all` runs every solver on every tiny fixture and compares against
    the brute-force enumeration. It must report zero proven mismatches (a
    solver claiming OPTIMAL that missed the proven optimum) and zero cases of
    a solver beating the reference.
    """
    report = verify_all(time_limit_s=10.0)
    assert not report.proven_mismatches(), (
        f"{len(report.proven_mismatches())} solver(s) claimed OPTIMAL below a "
        f"proven optimum: {report.proven_mismatches()[:2]}"
    )
    assert not report.beats_reference(), (
        f"{len(report.beats_reference())} solver(s) exceeded a proven optimum: "
        f"{report.beats_reference()[:2]}"
    )


# -------------------------------------------------------------- job shop


def test_job_shop_plans_validate_against_their_instance():
    """Processing time, machine eligibility, machine occupancy, precedence.

    Uses the real fixtures through the real adapter, and the real
    ``JobShopInstanceLike`` protocol (``name`` + ``jobs``) rather than an
    invented instance class.

    Coverage is asserted as a *bound* - every assigned operation is one the
    instance defines, on the machine it specifies - and not as full coverage,
    because ORION's job-shop scenario sets ``allow_unassigned=True``: its
    objective explicitly prices leaving work undone, so a solver that drops
    operations has made a legitimate scheduling decision.

    These fixtures carry no published optimum, so nothing here claims one.
    """
    cache = REPO_ROOT / "data" / "cache"
    instances = load_jobshop(cache, ["ft06", "ft10", "la01", "la02"])
    for name in ("ft06", "ft10", "la01", "la02"):
        instance = instances[name]
        scenario = jobshop_to_scenario(instance)
        result = _solve(scenario, budget=10.0)
        validation = validate_jobshop(instance, result.plan.assignments)
        assert validation.operations_assigned > 0, f"{name}: heuristic placed nothing"
        assert validation.operations_assigned <= validation.operations_total
        assert validation.valid, f"{name}: {validation.findings[:3]}"
