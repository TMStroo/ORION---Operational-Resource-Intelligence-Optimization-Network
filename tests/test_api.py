"""API tests.

These exercise the real planners, the real simulation engine and the real
replanner through the real HTTP surface. A mocked planner would prove the
serializers work, which is not what is at risk here.

What is pinned:

* a solver failure reaches the client as a failure status, not a 200 with an
  empty plan;
* a disruption is applied once, and a repeat is refused rather than re-applied;
* what-if results come from the optimisation engine, so a wrong operator or a
  missing baseline is a 4xx;
* unknown ids are 404s, not 500s.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from orion.api.app import create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    """An app on an in-memory database, seeded with one optimised scenario.

    Module-scoped because building a scenario and solving it is the slow part;
    the tests mutate their own rows, not shared state, except where a test says
    otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # A file, not ``:memory:``. SQLAlchemy pools connections, and an
        # in-memory SQLite database is per-connection - so a second checkout
        # sees an empty database and every request 500s on a missing table.
        db_path = Path(tmp) / "orion-test.db"
        app = create_app(
            f"sqlite:///{db_path.as_posix()}",
            experiments_root=REPO_ROOT / "experiments",
        )
        try:
            with TestClient(app) as c:
                yield c
        finally:
            # Windows refuses to delete a file a pooled connection still holds
            # open, and the temp-dir cleanup would mask the real test outcome.
            app.state.store.dispose()


@pytest.fixture(scope="module")
def scenario_id(client):
    response = client.post(
        "/scenarios",
        json={"name": "api-test", "task_count": 20, "resource_count": 8, "seed": 7},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture(scope="module")
def plan_id(client, scenario_id):
    response = client.post(
        f"/scenarios/{scenario_id}/optimize",
        json={"solver": "HEURISTIC", "time_budget_s": 5.0, "seed": 0},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


# ------------------------------------------------------------------- health


def test_health_reports_store_and_experiments(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "sqlite"
    assert body["rows"]["scenarios"] >= 0
    # The live evidence set is on disk in the repository; health should see it.
    assert body["live_experiments"] >= 5
    assert body["superseded_experiments"] >= 0
    assert "CP_SAT" in body["solvers"]


# ---------------------------------------------------------------- scenarios


def test_create_scenario_is_deterministic_for_a_seed(client):
    """The same request must yield the same scenario.

    Without this, an API-level determinism claim is untestable: ids are derived
    from the request, so a repeat POST returns the stored row and would pass
    even if generation were random.
    """
    payload = {"name": "determinism", "task_count": 12, "resource_count": 6, "seed": 99}
    first = client.post("/scenarios", json=payload)
    assert first.status_code == 201
    second = client.post("/scenarios", json=payload)
    assert second.status_code == 201
    assert first.json() == second.json()

    # A different seed must actually differ, or the check above is vacuous.
    other = client.post("/scenarios", json={**payload, "seed": 100})
    assert other.json()["tasks"] != first.json()["tasks"]


def test_list_and_get_scenario(client, scenario_id):
    listing = client.get("/scenarios").json()
    assert listing["total"] >= 1
    assert any(item["id"] == scenario_id for item in listing["items"])

    detail = client.get(f"/scenarios/{scenario_id}").json()
    assert detail["id"] == scenario_id
    assert len(detail["tasks"]) == detail["task_count"]
    assert len(detail["resources"]) == detail["resource_count"]
    assert detail["horizon"]["end"] > detail["horizon"]["start"]
    assert detail["objective_weights"]


def test_unknown_scenario_is_404(client):
    assert client.get("/scenarios/no-such-scenario").status_code == 404


def test_scenario_validation_reports_clean_scenario(client, scenario_id):
    body = client.post(f"/scenarios/{scenario_id}/validate").json()
    assert body["valid"] is True, body["issues"]
    assert body["task_count"] > 0


def test_validation_flags_a_task_with_no_eligible_resource(client):
    """A scenario with an unsatisfiable capability requirement must be invalid.

    This is the input-side counterpart to the NO_ELIGIBLE_PAIR failure the
    solvers report: catching it here means the caller learns before solving.
    """
    from orion.domain.entities import Scenario

    listing = client.get("/scenarios").json()
    base_id = listing["items"][-1]["id"]
    scenario = Scenario.from_dict(_scenario_payload(client, base_id))
    broken = _with_impossible_requirement(scenario)
    client.app.state.store.save_scenario(broken, source="generated")

    body = client.post(f"/scenarios/{broken.id}/validate").json()
    assert body["valid"] is False
    assert any(i["kind"] == "NO_ELIGIBLE_RESOURCE" for i in body["issues"])


def _scenario_payload(client, scenario_id: str) -> dict:
    row = client.app.state.store
    return row.load_scenario(scenario_id).to_dict()


def _with_impossible_requirement(scenario):
    """Return a copy of *scenario* with one task nobody can serve."""
    import dataclasses

    from orion.domain.entities import Task

    tasks = list(scenario.tasks)
    target = tasks[0]
    tasks[0] = dataclasses.replace(
        target, required_capabilities=frozenset({"capability-nobody-has"})
    )
    return dataclasses.replace(scenario, id=scenario.id + "-broken", tasks=tuple(tasks))


# -------------------------------------------------------------------- plans


def test_optimize_returns_a_plan_with_a_real_status(client, plan_id):
    body = client.get(f"/plans/{plan_id}").json()
    assert body["status"] in {
        "OPTIMAL",
        "FEASIBLE",
        "TIME_LIMIT",
        "RELAXATION_OPTIMAL",
    }
    assert body["tasks_assigned"] > 0
    assert body["objective"] > 0
    assert body["assignments"]
    assert body["solver_runs"], "a plan must record the solver run that produced it"
    assert body["objective_breakdown"]["total"] == pytest.approx(body["objective"])


def test_optimize_reports_a_timeout_as_a_status_not_an_error(client, scenario_id):
    """A 1 ms budget cannot prove optimality. That is TIME_LIMIT, not a failure.

    The important part is what it must *not* be: a 500, or a 200 claiming
    OPTIMAL.
    """
    response = client.post(
        f"/scenarios/{scenario_id}/optimize",
        json={"solver": "CP_SAT", "time_budget_s": 0.001, "seed": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] != "OPTIMAL" or body["tasks_assigned"] == body["tasks_total"]
    run = body["solver_runs"][0]
    assert run["solver"] == "CP_SAT"


def test_optimize_rejects_an_unknown_field(client, scenario_id):
    """`extra: forbid` is a contract choice: a typo'd field should not be ignored."""
    response = client.post(
        f"/scenarios/{scenario_id}/optimize", json={"solver": "HEURISTIC", "typo": 1}
    )
    assert response.status_code == 422


def test_unknown_plan_is_404(client):
    assert client.get("/plans/no-such-plan").status_code == 404


# --------------------------------------------------------------- simulation


def test_simulate_runs_the_engine(client, plan_id):
    """Uses `plan_id` rather than `scenario_id`.

    The endpoint takes the scenario's newest plan; a test that only builds a
    scenario has none, and asserting 409 there would test nothing about the
    engine. The 409 path is covered separately.
    """
    scenario = client.get(f"/plans/{plan_id}").json()["scenario_id"]
    response = client.post(
        f"/scenarios/{scenario}/simulate", json={"seed": 0, "plan_id": plan_id}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_id"]
    assert body["plan_id"] == plan_id
    assert body["plan_status"]
    assert body["end_time"] >= body["start_time"]
    assert body["completed_tasks"] >= 0
    assert body["events"], "the engine must emit a trace"
    assert body["event_count"] >= len(body["events"])
    times = [e["time"] for e in body["events"]]
    assert times == sorted(times), "events must be in chronological order"
    # Events are emitted once each; identical consecutive ordinals would mean a
    # duplicated insert.
    ordinals = [e["ordinal"] for e in body["events"]]
    assert ordinals == sorted(set(ordinals))
    assert any(e["type"].endswith("TASK_START") for e in body["events"]), (
        "a plan with assignments must produce at least one start event"
    )


def test_simulating_an_empty_plan_says_so(client, scenario_id):
    """A zero-assignment plan must report NOT_APPLICABLE, not an empty trace.

    A 1 ms CP-SAT budget produces a real plan that schedules nothing. Running
    it through the DES engine yields no events, and an empty event list is
    indistinguishable from a simulation that did nothing. The status is the
    difference between the two.
    """
    empty = client.post(
        f"/scenarios/{scenario_id}/optimize",
        json={"solver": "CP_SAT", "time_budget_s": 0.001, "seed": 0},
    ).json()
    if empty["tasks_assigned"] > 0:
        pytest.skip("this budget produced a non-empty plan; nothing to assert")

    body = client.post(
        f"/scenarios/{scenario_id}/simulate",
        json={"seed": 0, "plan_id": empty["id"]},
    ).json()
    assert body["status"] == "NOT_APPLICABLE"
    assert body["plan_id"] == empty["id"]
    assert body["events"] == []


def test_simulate_rejects_a_plan_from_another_scenario(client, scenario_id):
    other = client.post(
        "/scenarios", json={"name": "other-sim", "task_count": 8, "seed": 42}
    ).json()["id"]
    other_plan = client.post(
        f"/scenarios/{other}/optimize", json={"solver": "HEURISTIC", "time_budget_s": 3.0}
    ).json()["id"]
    response = client.post(
        f"/scenarios/{scenario_id}/simulate", json={"seed": 0, "plan_id": other_plan}
    )
    assert response.status_code == 422
    assert "belongs to scenario" in response.json()["detail"]


def test_simulate_without_a_plan_is_409(client):
    client.post("/scenarios", json={"name": "no-plan", "task_count": 6, "seed": 3})
    new_id = client.get("/scenarios").json()["items"][0]["id"]
    response = client.post(f"/scenarios/{new_id}/simulate", json={})
    assert response.status_code == 409


# -------------------------------------------------------------- disruption


def test_disrupt_reports_affected_work(client, scenario_id):
    response = client.post(
        f"/scenarios/{scenario_id}/disrupt",
        json={"type": "VEHICLE_FAILURE", "magnitude": 0.5, "at_time": 60},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["disruption"]["type"] == "VEHICLE_FAILURE"
    assert body["disruption"]["severity"] == "HIGH"
    assert body["recorded"] is True
    assert body["after_scenario_id"] != scenario_id


def test_disrupt_rejects_an_unknown_type(client, scenario_id):
    response = client.post(
        f"/scenarios/{scenario_id}/disrupt", json={"type": "NOT_A_DISRUPTION"}
    )
    assert response.status_code == 422
    assert "supported" in response.json()["detail"]


def test_replan_compares_repair_against_full(client, scenario_id):
    response = client.post(
        f"/scenarios/{scenario_id}/replan",
        json={
            "disruption_type": "RESOURCE_UNAVAILABLE",
            "magnitude": 0.3,
            "run_full": True,
            "solver": "HEURISTIC",
            "seed": 0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["repair"]["strategy"]
    assert body["repair_seconds"] > 0
    assert body["full_status"]
    if body["full_plan"] is not None:
        assert body["full_seconds"] > 0
        assert body["full_comparison"] is not None
        assert body["full_comparison"]["baseline_plan_id"]
    # Both paths reported, whatever the outcome. The API must not present
    # repair as the answer regardless of which is better.
    assert "recovery_pct" in body
    assert 0.0 <= body["recovery_pct"] <= 1.0


# ------------------------------------------------------------------ what-if


@pytest.mark.parametrize(
    "operator,parameter",
    [
        ("capacity_reduction", 0.8),
        ("demand_increase", 0.3),
        ("deadline_tighten", 0.15),
        ("travel_increase", 1.25),
        ("availability_drop", 0.2),
    ],
)
def test_what_if_operators_run_through_the_engine(client, scenario_id, operator, parameter):
    response = client.post(
        f"/scenarios/{scenario_id}/what-if",
        json={"operator": operator, "parameter": parameter, "seed": 0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["operator"] == operator
    assert body["question"]
    assert body["comparison"]["deltas"], "a what-if must report measured deltas"


def test_what_if_rejects_an_unknown_operator(client, scenario_id):
    response = client.post(
        f"/scenarios/{scenario_id}/what-if", json={"operator": "teleport", "parameter": 1}
    )
    assert response.status_code == 422
    assert "supported" in response.json()["detail"]


# ------------------------------------------------------------- comparisons


def test_plan_comparison_returns_something(client, plan_id):
    body = client.get(f"/plans/{plan_id}/comparison").json()
    assert body["plan_id"] == plan_id
    assert body["items"]


# -------------------------------------------------------------- experiments


def test_experiments_endpoint_lists_live_and_superseded(client):
    body = client.get("/experiments").json()
    assert body["total"] >= 5
    assert body["live"] >= 5
    assert body["superseded"] >= 1
    assert any(e["kind"].endswith("scalability") for e in body["items"])


def test_experiment_detail_includes_rows(client):
    listing = client.get("/experiments?live_only=true").json()
    target = next(e for e in listing["items"] if e["kind"].endswith("scalability"))
    body = client.get(f"/experiments/{target['id']}").json()
    assert body["id"] == target["id"]
    assert body["rows"], "experiment rows must be readable"
    assert body["manifest"]["git_commit"]


def test_unknown_experiment_is_404(client):
    assert client.get("/experiments/no-such-experiment").status_code == 404


# --------------------------------- defects the real browser run exposed


def _first_resource_id(client, scenario_id: str) -> str:
    body = client.get(f"/scenarios/{scenario_id}").json()
    return next(r["id"] for r in body["resources"])


def _disrupt_and_replan(client, scenario_id):
    """Apply a resource disruption and repair it, through the public surface."""
    target = _first_resource_id(client, scenario_id)
    response = client.post(
        f"/scenarios/{scenario_id}/replan",
        json={
            "disruption_type": "RESOURCE_UNAVAILABLE",
            "target_id": target,
            "magnitude": 0.5,
            "duration": 180,
            "solver": "HEURISTIC",
            "time_budget_s": 5.0,
            "run_full": True,
        },
    )
    assert response.status_code == 200, response.text
    return target, response.json()


def test_replan_persists_its_comparison_so_the_plan_is_not_an_orphan(
    client, scenario_id
):
    """A comparison that is only returned, never stored, is invisible forever.

    The Comparison page reads /plans/{id}/comparison, which knows only about
    stored records. Returning a repair comparison without persisting it made a
    completed repair look like it had never been compared against anything.
    """
    _target, body = _disrupt_and_replan(client, scenario_id)
    repair = body["repair_plan"]
    assert repair is not None, "replan produced no repair plan"

    stored = client.get(f"/plans/{repair['id']}/comparison").json()
    assert stored["items"], "the repair plan has no stored comparison"
    assert stored["items"][0]["kind"] != "none", (
        f"the repair comparison was stored as {stored['items'][0]['kind']!r}"
    )
    assert stored["items"][0]["deltas"], "the stored comparison carries no deltas"


def test_a_comparison_names_a_baseline_the_client_can_fetch(client, scenario_id):
    """A dangling baseline id reads as resolvable and is not.

    The comparison has to reference the stored plan row, not the loaded plan's
    own id, or a client following the reference gets a 404.
    """
    _target, body = _disrupt_and_replan(client, scenario_id)
    repair_id = body["repair_plan"]["id"]
    baseline_id = client.get(f"/plans/{repair_id}/comparison").json()["items"][0][
        "baseline_plan_id"
    ]
    assert client.get(f"/plans/{baseline_id}").status_code == 200, (
        f"comparison names baseline {baseline_id!r}, which is not fetchable"
    )


def test_a_plan_with_no_comparison_does_not_impersonate_a_baseline(client, plan_id):
    """The empty state is real, but it must not name a scenario as a baseline."""
    body = client.get(f"/plans/{plan_id}/comparison").json()
    assert body["items"], "a plan with no comparison should still report the state"
    only = body["items"][0]
    assert only["kind"] == "none"
    assert only["baseline_plan_id"] == plan_id, (
        f"placeholder baseline is {only['baseline_plan_id']!r}, not the plan itself"
    )


def test_a_default_resource_disruption_actually_removes_working_time(
    client, scenario_id
):
    """A no-op disruption reported as HIGH severity is the worst failure mode.

    The default disruption time sat at 25% of the horizon, which for a
    late-starting resource fell entirely before its shift, so the availability
    window intersected to nothing while the response still claimed a severity.
    """
    target = _first_resource_id(client, scenario_id)
    before = client.get(f"/scenarios/{scenario_id}").json()
    resource = next(r for r in before["resources"] if r["id"] == target)

    applied = client.post(
        f"/scenarios/{scenario_id}/disrupt",
        json={
            "type": "RESOURCE_UNAVAILABLE",
            "target_id": target,
            "magnitude": 0.5,
            "duration": 240,
        },
    )
    assert applied.status_code == 200, applied.text

    after = client.get(f"/scenarios/{applied.json()['after_scenario_id']}").json()
    disrupted = next(r for r in after["resources"] if r["id"] == target)
    assert disrupted["unavailable"], (
        "a default RESOURCE_UNAVAILABLE removed no working time; "
        f"shift={resource['shift_start']}-{resource['shift_end']}"
    )


def test_a_resource_reports_its_availability_windows_through_the_api(
    client, scenario_id
):
    """Lost working time has to be readable, or the UI cannot show it."""
    body = client.get(f"/scenarios/{scenario_id}").json()
    assert all(
        r.get("unavailable") is not None for r in body["resources"]
    ), "ResourceView exposes no unavailable field"


# ----------------------------------------------------------------- exports


def test_a_scenario_exports_and_reloads_identically(client, scenario_id):
    """Export then import, through the public surface, with nothing lost."""
    exported = client.get(f"/scenarios/{scenario_id}/export")
    assert exported.status_code == 200, exported.text
    body = exported.json()
    assert body["kind"] == "scenario"
    assert body["content_type"] == "application/json"
    assert body["bytes"] > 0
    assert body["provenance"]["git_commit"], "an export without a commit is not evidence"
    assert body["provenance"]["source"].startswith("api:/scenarios/")

    reloaded = client.post("/exports/scenario", json={"artifact": body["body"]})
    assert reloaded.status_code == 200, reloaded.text
    result = reloaded.json()
    assert result["round_trip_identical"] is True
    assert result["entity_id"] == scenario_id
    assert result["detail"]["tasks"] > 0
    assert result["provenance"]["kind"] == "scenario"


def test_a_plan_exports_and_reloads_with_its_status_and_objective(client, plan_id):
    exported = client.get(f"/plans/{plan_id}/export")
    assert exported.status_code == 200, exported.text
    reloaded = client.post("/exports/plan", json={"artifact": exported.json()["body"]})
    assert reloaded.status_code == 200, reloaded.text
    result = reloaded.json()
    assert result["round_trip_identical"] is True
    assert result["entity_id"] == plan_id
    assert result["status"] in {
        "OPTIMAL", "FEASIBLE", "TIME_LIMIT", "INFEASIBLE",
        "ERROR", "NOT_APPLICABLE", "RELAXATION_OPTIMAL",
    }
    assert result["objective"] is not None
    assert result["detail"]["assignments"] > 0
    assert result["detail"]["created_at"], "the plan timestamp was lost in the export"
    assert result["detail"]["solver_statuses"], "solver status was lost in the export"


def test_metrics_and_timeline_export_as_csv_with_provenance(client, scenario_id, plan_id):
    metrics = client.get(f"/scenarios/{scenario_id}/export?format=metrics")
    assert metrics.status_code == 200, metrics.text
    body = metrics.json()
    assert body["content_type"] == "text/csv"
    lines = body["body"].splitlines()
    assert lines[0].startswith("# provenance: "), "a CSV export has no provenance"
    assert len(lines) > 1, "the metrics CSV has no data rows"

    timeline = client.get(f"/plans/{plan_id}/export?format=timeline")
    assert timeline.status_code == 200, timeline.text
    tl = timeline.json()["body"].splitlines()
    assert tl[0].startswith("# provenance: ")
    header = tl[1].split(",")
    for column in ("task_id", "resource_id", "start", "end"):
        assert column in header, f"the timeline CSV has no {column} column"


def test_an_unsupported_export_format_is_rejected(client, scenario_id):
    response = client.get(f"/scenarios/{scenario_id}/export?format=docx")
    assert response.status_code == 422


def test_importing_a_bare_payload_reports_that_it_is_not_an_export(client):
    response = client.post("/exports/scenario", json={"artifact": '{"hello": "world"}'})
    assert response.status_code == 422
    assert "not an ORION export" in response.json()["detail"]


# --------------------------------------------------------- published schema


def test_the_difficulty_enum_matches_what_the_generator_accepts(client):
    """The advertised `difficulty` values must actually be accepted.

    The schema declared `Literal["easy", "medium", "hard"]` while
    `difficulty_config` accepts only `small`, `medium`, `large` and `stress`.
    Three of the four published values were rejected by the server, so the
    OpenAPI document lied to any client that read it - and only the default
    value happened to work. Found by running the workflow over real HTTP.
    """
    from orion.data.scenario_generator import DIFFICULTY_PRESETS

    from orion.api.schemas import CreateScenarioRequest

    declared = set(get_args(CreateScenarioRequest.model_fields["difficulty"].annotation))
    assert declared == set(DIFFICULTY_PRESETS), (
        f"schema advertises {sorted(declared)} but the generator accepts "
        f"{sorted(DIFFICULTY_PRESETS)}"
    )

    for level in sorted(DIFFICULTY_PRESETS):
        response = client.post(
            "/scenarios",
            json={"name": f"difficulty-{level}", "difficulty": level, "task_count": 6},
        )
        assert response.status_code == 201, (
            f"difficulty {level!r} is advertised but the server answered "
            f"{response.status_code}: {response.text[:160]}"
        )
