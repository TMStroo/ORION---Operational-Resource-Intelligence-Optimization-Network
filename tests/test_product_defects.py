"""Regression tests for defects the live-server and real-browser runs exposed.

Each of these was a real bug found by running the product, not by reading it.
They are written against the observable API/store behaviour so they fail if the
underlying cause is reintroduced.
"""

from __future__ import annotations

import os

import pytest

from orion.domain.plans import Plan, SolverRun, SolverStatus
from orion.optimization.builder import build_plan
from orion.optimization.models import SchedulingContext
from orion.store import Store


@pytest.fixture()
def store(tmp_path):
    st = Store(f"sqlite:///{tmp_path / 'reg.db'}")
    st.create_all()
    try:
        yield st
    finally:
        st.dispose()


@pytest.fixture(scope="module")
def solved():
    """A real plan from the real Planner on a small deterministic scenario."""
    from orion.data.scenario_generator import ScenarioConfig, generate
    from orion.planning.planner import Planner

    config = ScenarioConfig(
        name="regression", num_tasks=12, num_teams=3, num_vehicles=2, seed=11
    )
    scenario = generate(config)
    outcome = Planner(default_solver="HEURISTIC", default_time_budget=5.0).plan(scenario)
    return scenario, outcome.plan


# --------------------------------------------------------------- builder bug


def test_build_plan_writes_the_scored_objective_back_onto_the_run(solved):
    """heuristics.py builds its SolverRun with objective=0.0 and a comment
    promising build_plan fills it in. If it does not, the API shows a solver
    run reporting 0.0 beside a plan scored at several hundred."""
    _, plan = solved

    assert plan is not None
    assert plan.objective.total > 0.0
    run = plan.solver_runs[0]
    assert run.objective != 0.0, (
        "solver run still reports 0.0 while the plan is scored at "
        f"{plan.objective.total:.2f}"
    )
    assert run.objective == pytest.approx(plan.objective.total, rel=1e-3)


# ----------------------------------------------------------------- store bug


def test_stored_plan_reports_real_completion_runtime_and_solver(store, solved):
    """Plan has no `completion`, `runtime_seconds` or `solver_name` field.
    Reading them with getattr defaulted every stored plan to 0% completion, 0.0 s
    and a blank solver, and those became the numbers the API served."""
    scenario, plan = solved
    store.save_scenario(scenario)
    plan_id = store.save_plan(plan, scenario_id=scenario.id, label="test")

    row = store.get_plan_row(plan_id)
    assert row.solver == "HEURISTIC", f"stored solver was {row.solver!r}"
    assert row.completion > 0.0, f"stored completion was {row.completion!r}"
    assert row.runtime_s > 0.0, f"stored runtime was {row.runtime_s!r}"
    assert row.objective == pytest.approx(plan.objective.total, rel=1e-6)
    # completion must equal assigned/total, not a stale or zeroed column
    assert row.completion == pytest.approx(
        row.tasks_assigned / row.tasks_total, rel=1e-6
    )


def test_store_reports_no_invented_solver_for_a_runless_plan(store, solved):
    """A plan with no solver runs must report an empty solver, not a guess."""
    scenario, plan = solved
    store.save_scenario(scenario)
    stripped = Plan(
        id=plan.id,
        scenario_id=plan.scenario_id,
        assignments=plan.assignments,
        objective=plan.objective,
        tasks_assigned=plan.tasks_assigned,
        tasks_total=plan.tasks_total,
        status=plan.status,
    )
    plan_id = store.save_plan(stripped, scenario_id=scenario.id)
    row = store.get_plan_row(plan_id)
    assert row.solver == ""
    assert row.runtime_s == 0.0


# ------------------------------------------------------------- id coherence


def test_plan_id_survives_the_round_trip(store, solved):
    """A plan's id must be the id the API will serve it under, or every
    cross-plan reference in a comparison is a dangling pointer."""
    scenario, plan = solved
    store.save_scenario(scenario)
    plan_id = store.save_plan(plan, scenario_id=scenario.id)
    assert plan_id == plan.id
    loaded = store.load_plan(plan_id)
    assert loaded.id == plan.id


def test_a_plan_scenario_id_is_not_shadowed_by_a_disrupted_derangement(store, solved):
    """Disruption.apply keeps the scenario id, so persisting the disrupted state
    under its own id must not overwrite the original."""
    import dataclasses

    from orion.simulation.disruptions import Disruption

    scenario, _plan = solved
    store.save_scenario(scenario)
    original = store.load_scenario(scenario.id)
    disruption = Disruption(
        id="DSR-1",
        type="RESOURCE_UNAVAILABLE",
        target_id=scenario.resources[0].id,
        timestamp=0,
        magnitude=0.5,
        duration=240,
    )

    after = dataclasses.replace(disruption.apply(original), id=f"{original.id}-after-DSR-1")
    after_id = store.save_scenario(after, source="disrupted")

    assert after_id != original.id
    # the pre-disruption world is still readable and unchanged
    still = store.load_scenario(original.id)
    assert len(still.resources) == len(original.resources)
    assert all(r.status == "AVAILABLE" for r in still.resources)


# ------------------------------------------------- comparison persistence bug


def test_stored_comparison_deltas_round_trip_through_the_api(store, solved):
    """The store hand-rolled a 4-key delta dict and dropped name, unit and
    higher_is_better. Reading a stored comparison back raised a
    ValidationError, so GET /plans/{id}/comparison returned 500 for every
    comparison the product itself had written."""
    from orion.planning.comparison import compare_plans

    scenario, plan = solved
    store.save_scenario(scenario)
    store.save_plan(plan, scenario_id=scenario.id, label="baseline")
    other = store.save_plan(plan, scenario_id=scenario.id, label="candidate", plan_id="PLAN-CAND")

    comparison = compare_plans(plan, plan, scenario)
    store.record_comparison(
        other, plan.id, other, "what_if:capacity_reduction", comparison
    )

    row = store.list_comparisons(other)[-1]
    import json

    deltas = json.loads(row.deltas_json)
    assert deltas, "no deltas were persisted"
    for d in deltas:
        assert set(("name", "label", "before", "after", "change", "unit", "higher_is_better")) <= set(
            d
        ), f"delta missing fields the API schema requires: {sorted(d)}"

    # The endpoint must survive reading it back. Exercise the same projection
    # the route uses, not a splat, because MetricDelta.to_dict carries derived
    # fields the response model does not declare.
    from orion.api.schemas import MetricDeltaView

    for d in deltas:
        MetricDeltaView(
            name=str(d.get("name", "")),
            label=str(d.get("label", "")),
            before=float(d.get("before", 0.0)),
            after=float(d.get("after", 0.0)),
            change=float(d.get("change", 0.0)),
            unit=str(d.get("unit", "")),
            higher_is_better=bool(d.get("higher_is_better", True)),
        )


# --------------------------------------------- resource unavailability is visible


def test_a_disrupted_resource_reports_its_lost_working_time(solved):
    """A disruption removes time, not the resource. `status` stays AVAILABLE,
    so the API has to expose the `unavailable` windows or a client cannot tell a
    disrupted scenario from an intact one."""
    import dataclasses
    import types

    from orion.api.serializers import scenario_detail
    from orion.simulation.disruptions import Disruption

    scenario, _plan = solved
    vehicle = next(
        (r for r in scenario.resources if r.kind == "VEHICLE"), scenario.resources[0]
    )
    disruption = Disruption(
        id="DSR-1",
        type="RESOURCE_UNAVAILABLE",
        target_id=vehicle.id,
        timestamp=vehicle.shift_start,
        magnitude=0.5,
        duration=240,
    )
    after = disruption.apply(scenario)

    # scenario_detail takes the persisted row plus the domain object; the row
    # only supplies the summary columns.
    def row_for(sc):
        return types.SimpleNamespace(
            id=sc.id,
            name=sc.name,
            description=sc.description,
            task_count=len(sc.tasks),
            resource_count=len(sc.resources),
            horizon_end=sc.horizon.end,
            source="test",
            tags=[],
            created_at=None,
        )

    before_view = {r.id: r for r in scenario_detail(row_for(scenario), scenario).resources}
    after_view = {r.id: r for r in scenario_detail(row_for(after), after).resources}

    assert before_view[vehicle.id].unavailable == []
    assert len(after_view[vehicle.id].unavailable) == 1, (
        "the API does not report the lost window, so the disruption is invisible "
        "to a client"
    )
    window = after_view[vehicle.id].unavailable[0]
    assert window["end"] > window["start"]
    assert vehicle.shift_start <= window["start"] < window["end"] <= vehicle.shift_end
    # the resource itself is unchanged apart from the window
    assert after_view[vehicle.id].status == before_view[vehicle.id].status
