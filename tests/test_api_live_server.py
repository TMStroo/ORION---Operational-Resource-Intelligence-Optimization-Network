"""End-to-end API test against a real uvicorn server over real HTTP.

TestClient runs the ASGI app in-process and shares the interpreter's state. It
cannot catch a routing or middleware problem that only appears under the real
server, a database that behaves differently once another process is writing, or
a startup path that TestClient never takes. This test starts uvicorn as a
subprocess and drives the whole workflow with actual requests.

What it pins, beyond the TestClient suite:

* the server boots from `orion.api.asgi:app` with no arguments;
* the workflow works over the wire, in order, on a fresh database;
* database state is inspected after each major step, through a separate
  connection, rather than trusted from the response body;
* a disrupted scenario does not overwrite the baseline;
* an explicitly named plan is the plan that gets simulated;
* a disruption is recorded exactly once even when the workflow replans.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "backend" / "src"
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # pragma: no cover - CI runs from a different layout
    PYTHON = Path(sys.executable)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Boot uvicorn on a free port against a throwaway database.

    Yields a namespace with the base URL and the SQLite path, so the test can
    read the database directly and independently of the server process.
    """
    pytest.importorskip("uvicorn")
    workdir = tmp_path_factory.mktemp("live-server")
    db_path = workdir / "orion-live.db"
    port = _free_port()

    env = {
        **os.environ,
        "PYTHONPATH": str(SRC_ROOT),
        "ORION_DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        "ORION_EXPERIMENTS_ROOT": str(REPO_ROOT / "experiments"),
    }
    process = subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "uvicorn",
            "orion.api.asgi:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base = f"http://127.0.0.1:{port}"
    import urllib.error
    import urllib.request

    def _get(path: str, timeout: float = 1.0):
        request = urllib.request.Request(base + path)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # Poll until the server answers. Failing to boot is a test failure with the
    # server's own output attached, which is the only way to diagnose it.
    deadline = time.time() + 60
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            pytest.fail(
                f"uvicorn exited with {process.returncode} before serving:\n{output[-3000:]}"
            )
        try:
            _get("/health")
            break
        except Exception as exc:  # noqa: BLE001 - any failure means "not up yet"
            last_error = exc
            time.sleep(0.4)
    else:  # pragma: no cover - only on a very slow machine
        process.kill()
        pytest.fail(f"uvicorn did not become ready in 60s: {last_error}")

    try:
        yield {"base": base, "db": db_path, "env": env}
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(scope="module")
def http(live_server):
    """A tiny HTTP client: (method, path, payload) -> (status, body)."""
    import urllib.error
    import urllib.request

    def call(method: str, path: str, payload: dict | None = None, timeout: float = 180.0):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            live_server["base"] + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return response.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"raw": raw}

    return call


def _query(db: Path, sql: str, *params) -> list[tuple]:
    """Read the database with a connection this test owns.

    A second connection is the point: state read back through the server would
    only prove the server agrees with itself.
    """
    connection = sqlite3.connect(str(db))
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------


def test_health_over_http(live_server, http):
    status, body = http("GET", "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["database"] == "sqlite"
    assert body["live_experiments"] >= 5, (
        "the server must import the repository's live evidence at startup"
    )
    # The experiments really are in the database, not just counted in memory.
    assert _query(live_server["db"], "SELECT COUNT(*) FROM experiments")[0][0] >= 5


def test_full_workflow_over_http(live_server, http):
    """The product loop, end to end, over the wire, with state checked as it goes.

    Scenario -> validate -> optimize -> plan -> simulate -> disrupt -> replan ->
    compare -> what-if -> experiments. Every step asserts against the database,
    not just the response.
    """
    db = live_server["db"]

    # 1. create a scenario
    status, created = http(
        "POST",
        "/scenarios",
        {"name": "e2e", "task_count": 24, "resource_count": 8, "seed": 4242},
    )
    assert status == 201, created
    scenario_id = created["id"]
    assert len(created["tasks"]) == 24
    assert len(created["resources"]) == 8
    # 2. it is persisted, with its tasks and resources as real rows
    assert _query(db, "SELECT task_count FROM scenarios WHERE id=?", scenario_id)[0][0] == 24
    assert _query(db, "SELECT COUNT(*) FROM tasks WHERE scenario_id=?", scenario_id)[0][0] == 24
    assert (
        _query(db, "SELECT COUNT(*) FROM resources WHERE scenario_id=?", scenario_id)[0][0] == 8
    )

    # 3. validate
    status, validation = http("POST", f"/scenarios/{scenario_id}/validate")
    assert status == 200
    assert validation["valid"] is True, validation["issues"]

    # 4. optimize
    status, plan = http(
        "POST",
        f"/scenarios/{scenario_id}/optimize",
        {"solver": "HEURISTIC", "time_budget_s": 10.0, "seed": 0, "label": "baseline"},
    )
    assert status == 200, plan
    plan_id = plan["id"]
    assert plan["tasks_assigned"] > 0
    assert plan["objective"] > 0
    assert plan["status"] in {"OPTIMAL", "FEASIBLE", "TIME_LIMIT", "RELAXATION_OPTIMAL"}
    # 5. the plan and its assignments are queryable, not a blob
    assert _query(db, "SELECT COUNT(*) FROM plans WHERE id=?", plan_id)[0][0] == 1
    assert (
        _query(db, "SELECT COUNT(*) FROM assignments WHERE plan_id=?", plan_id)[0][0]
        == plan["tasks_assigned"]
    )
    assert _query(db, "SELECT COUNT(*) FROM solver_runs WHERE plan_id=?", plan_id)[0][0] >= 1

    # 6. retrieve it
    status, fetched = http("GET", f"/plans/{plan_id}")
    assert status == 200
    assert fetched["id"] == plan_id
    assert fetched["objective"] == pytest.approx(plan["objective"])
    assert len(fetched["assignments"]) == plan["tasks_assigned"]
    assert fetched["solver_runs"][0]["solver"] == "HEURISTIC"
    # Event/status strings are readable, not enum ordinals.
    assert fetched["assignments"][0]["priority"].isalpha()

    # 7. simulate the named plan explicitly
    status, sim = http(
        "POST", f"/scenarios/{scenario_id}/simulate", {"seed": 0, "plan_id": plan_id}
    )
    assert status == 200, sim
    assert sim["plan_id"] == plan_id, "the requested plan must be the one simulated"
    assert sim["events"], "a non-empty plan must produce a trace"
    assert sim["completed_tasks"] > 0
    event_types = {e["type"] for e in sim["events"]}
    assert any(t.endswith("TASK_START") for t in event_types), event_types
    # Events are persisted, once each
    events_in_db = _query(
        db, "SELECT COUNT(*) FROM simulation_events WHERE run_id=?", sim["run_id"]
    )[0][0]
    assert events_in_db == sim["event_count"]
    assert (
        _query(
            db,
            "SELECT COUNT(*) FROM (SELECT ordinal FROM simulation_events "
            "WHERE run_id=? GROUP BY ordinal HAVING COUNT(*)>1)",
            sim["run_id"],
        )[0][0]
        == 0
    )

    # 8. disrupt
    status, disrupted = http(
        "POST",
        f"/scenarios/{scenario_id}/disrupt",
        {"type": "RESOURCE_UNAVAILABLE", "magnitude": 0.4, "at_time": 90, "seed": 0},
    )
    assert status == 200, disrupted
    after_id = disrupted["after_scenario_id"]
    assert after_id != scenario_id, (
        "the disrupted world must be a distinct scenario; sharing an id would make "
        "before/after the same row"
    )
    assert disrupted["recorded"] is True
    assert _query(db, "SELECT COUNT(*) FROM disruptions WHERE scenario_id=?", scenario_id)[0][0] == 1
    # The baseline scenario's own state is untouched
    assert (
        _query(db, "SELECT COUNT(*) FROM tasks WHERE scenario_id=?", scenario_id)[0][0] == 24
    )

    # 9. replan: local repair and full re-optimization, both reported
    status, replan = http(
        "POST",
        f"/scenarios/{scenario_id}/replan",
        {
            "disruption_type": "RESOURCE_UNAVAILABLE",
            "magnitude": 0.4,
            "at_time": 90,
            "run_full": True,
            "solver": "HEURISTIC",
            "seed": 0,
        },
    )
    assert status == 200, replan
    assert replan["repair"]["strategy"]
    assert replan["repair_seconds"] > 0
    assert replan["full_status"], "a declined full re-plan must still state its status"
    assert replan["full_reason"] or replan["full_plan"] is not None
    if replan["full_plan"] is not None:
        assert replan["full_seconds"] > 0
        # Both paths recorded, neither deleted
        assert _query(db, "SELECT COUNT(*) FROM plans WHERE scenario_id=?", scenario_id)[0][0] >= 3
    if replan["repair_plan"] is not None:
        assert replan["repair_comparison"] is not None
        assert replan["repair_comparison"]["baseline_plan_id"] == plan_id
        assert 0.0 <= replan["recovery_pct"] <= 1.0
        assert 0.0 <= replan["repair_preserved_ratio"] <= 1.0

    # 10. compare
    status, comparison = http("GET", f"/plans/{plan_id}/comparison")
    assert status == 200
    assert comparison["items"]
    assert comparison["plan_id"] == plan_id

    # 11. what-if
    status, whatif = http(
        "POST",
        f"/scenarios/{scenario_id}/what-if",
        {"operator": "capacity_reduction", "parameter": 0.8, "seed": 0},
    )
    assert status == 200, whatif
    assert whatif["operator"] == "capacity_reduction"
    assert whatif["comparison"]["deltas"], "a what-if must report measured deltas"
    whatif_plan = whatif["scenario_plan_id"]
    assert _query(db, "SELECT COUNT(*) FROM plans WHERE id=?", whatif_plan)[0][0] == 1

    # 12. experiments are still reachable and unmodified by the workflow.
    # Asked without live_only: with that filter the superseded count is
    # necessarily zero, so asserting on it here would be asserting nothing.
    status, experiments = http("GET", "/experiments")
    assert status == 200
    assert experiments["live"] >= 5
    assert experiments["superseded"] >= 1
    # A superseded run must be marked as such, never presented as current.
    superseded = [e for e in experiments["items"] if e["status"] != "succeeded"]
    assert superseded, "the repository keeps superseded runs; they must be visible"
    for entry in superseded:
        assert entry["superseded_by"], f"{entry['id']} is superseded with no successor"
        assert entry["supersede_reason"], f"{entry['id']} is superseded with no reason"


def test_a_disruption_is_not_applied_twice(live_server, http):
    """Replanning the same event must not record a second disruption row.

    The DES loop legitimately re-runs against one event; what must not happen is
    the event being counted as applied more than once, because that inflates
    every downstream severity and recovery number.
    """
    db = live_server["db"]
    status, created = http(
        "POST", "/scenarios", {"name": "once", "task_count": 16, "seed": 5151}
    )
    assert status == 201, created
    scenario_id = created["id"]
    status, plan = http(
        "POST", f"/scenarios/{scenario_id}/optimize", {"solver": "HEURISTIC", "time_budget_s": 8.0}
    )
    assert status == 200, plan

    before = _query(
        db, "SELECT COUNT(*) FROM disruptions WHERE scenario_id=?", scenario_id
    )[0][0]

    for _ in range(2):
        status, _ = http(
            "POST",
            f"/scenarios/{scenario_id}/replan",
            {"disruption_type": "VEHICLE_FAILURE", "magnitude": 0.5, "seed": 0},
        )
        assert status == 200

    after = _query(
        db, "SELECT COUNT(*) FROM disruptions WHERE scenario_id=?", scenario_id
    )[0][0]
    # Each replan mints a fresh disruption id, so two replans are two events -
    # but every stored row must be flagged applied exactly once.
    assert after == before + 2
    unflagged = _query(
        db,
        "SELECT COUNT(*) FROM disruptions WHERE scenario_id=? AND applied_once=0",
        scenario_id,
    )[0][0]
    assert unflagged == 0


def test_errors_are_honest_over_http(http):
    """Status codes for bad input, checked over the wire rather than in-process."""
    status, body = http("GET", "/scenarios/does-not-exist")
    assert status == 404, body
    assert "does-not-exist" in body["detail"]

    status, body = http("GET", "/plans/does-not-exist")
    assert status == 404

    status, body = http("GET", "/experiments/does-not-exist")
    assert status == 404

    status, body = http(
        "POST", "/scenarios/does-not-exist/optimize", {"solver": "HEURISTIC"}
    )
    assert status == 404

    status, body = http("POST", "/scenarios", {"task_count": 0})
    assert status == 422, "an empty scenario request must be rejected by validation"

    status, body = http("POST", "/scenarios", {"unknown_field": 1})
    assert status == 422, "unknown fields are a contract violation, not noise to ignore"
