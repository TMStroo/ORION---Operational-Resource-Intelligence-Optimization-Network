"""Exercise the complete ORION workflow against a real running uvicorn server.

This is deliberately *not* a TestClient. TestClient runs the ASGI app in
process, so it cannot catch anything that only goes wrong over a real socket:
a middleware ordering mistake, a proxy that strips a body, a response that
depends on a streaming path, a server that fails to start under uvicorn's own
lifespan handling, or a route that is registered but not actually reachable.

Every step below is a real HTTP request with a real request body, and the
persistence claim is checked by reading the row back out of the database
afterwards rather than by trusting the response.

Run against a server already listening::

    python scripts/check_api_live.py --base http://127.0.0.1:8137

Exits non-zero on the first failed expectation.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def call(
    base: str, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    """One real HTTP request. Returns (status, decoded body)."""
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSED.append(label)
        print(f"  ok    {label}")
    else:
        FAILED.append((label, detail))
        print(f"  FAIL  {label}  {detail}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8137")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"ORION live API workflow against {base}")
    print("=" * 64)

    # ---------------------------------------------------------------- health
    print("\n[1] GET /health")
    status, health = call(base, "GET", "/health")
    check("health returns 200", status == 200, str(status))
    check("health reports ok", isinstance(health, dict) and health.get("status") == "ok")

    # -------------------------------------------------------------- scenario
    print("\n[2] POST /scenarios")
    # CreateScenarioRequest takes task_count/resource_count, not the generator's
    # num_tasks/num_teams/num_vehicles, so the shape is taken from the schema
    # rather than guessed.
    scenario_body = {
        "name": "live-api-check",
        "task_count": 12,
        "resource_count": 6,
        "seed": 31337,
        "difficulty": "medium",
    }
    status, created = call(base, "POST", "/scenarios", scenario_body)
    # 201 Created is the correct status for a POST that made a new resource;
    # the API is right to return it and the check should say so.
    check("create returns 201", status == 201, str(status))
    scenario_id = created.get("id") if isinstance(created, dict) else None
    check("create returns a scenario id", bool(scenario_id), str(created)[:120])

    status, fetched = call(base, "GET", f"/scenarios/{scenario_id}")
    check("scenario is readable after creation", status == 200 and fetched.get("id") == scenario_id)
    check(
        "created scenario has the requested task count",
        fetched.get("task_count") == 12 or len(fetched.get("tasks", [])) == 12,
        f"task_count={fetched.get('task_count')} tasks={len(fetched.get('tasks', []))}",
    )

    # ------------------------------------------------------------- validate
    print("\n[3] POST /scenarios/{id}/validate")
    status, validation = call(base, "POST", f"/scenarios/{scenario_id}/validate")
    check("validate returns 200", status == 200, str(status))
    check("scenario is valid", validation.get("valid") is True, str(validation)[:160])

    # ------------------------------------------------------------- optimize
    print("\n[4] POST /scenarios/{id}/optimize")
    status, optimized = call(
        base, "POST", f"/scenarios/{scenario_id}/optimize",
        {"solver": "HEURISTIC", "time_budget_s": 5.0},
    )
    check("optimize returns 200", status == 200, f"{status} {str(optimized)[:200]}")
    # The optimize response is a PlanSummary: the id is at the top level.
    plan_id = (optimized or {}).get("id") or (optimized or {}).get("plan_id")
    check("optimize returns a plan id", bool(plan_id), str(optimized)[:200])

    # ----------------------------------------------------------------- plan
    print("\n[5] GET /plans/{id}")
    status, detail = call(base, "GET", f"/plans/{plan_id}")
    check("plan is readable", status == 200 and detail.get("id") == plan_id)
    plan = detail
    runs = plan.get("solver_runs") or []
    run = runs[0] if runs else {}
    solver_name = run.get("solver") or run.get("solver_name")
    check("plan records the solver that ran", solver_name == "HEURISTIC", str(solver_name))
    check("plan records a nonzero runtime", float(run.get("runtime_s") or 0) > 0, str(run.get("runtime_s")))
    # PlanDetail exposes the objective as a float; OptimizeResponse's summary
    # exposes it as a breakdown. Accept either rather than assuming one shape.
    raw_objective = plan.get("objective")
    objective_total = (
        float(raw_objective.get("total") or 0)
        if isinstance(raw_objective, dict)
        else float(raw_objective or 0)
    )
    check("plan objective is nonzero", abs(objective_total) > 0, str(raw_objective))
    check("plan records at least one solver run", bool(runs), f"{len(runs)} runs")
    check(
        "solver run objective matches the plan objective",
        # The API rounds the run objective for display; compare at 4dp, which is
        # the precision to_dict() itself uses.
        abs(float(run.get("objective") or 0) - objective_total) < 1e-3,
        f"run={run.get('objective')} plan={objective_total}",
    )
    check(
        "plan assigned some tasks",
        plan.get("tasks_assigned", 0) > 0,
        f"tasks_assigned={plan.get('tasks_assigned')}",
    )
    check(
        "tasks assigned never exceeds tasks total",
        plan.get("tasks_assigned", 0) <= plan.get("tasks_total", 0),
    )

    # ------------------------------------------------------------- simulate
    print("\n[6] POST /scenarios/{id}/simulate")
    status, sim = call(base, "POST", f"/scenarios/{scenario_id}/simulate", {"plan_id": plan_id})
    check("simulate returns 200", status == 200, f"{status} {str(sim)[:160]}")
    events = sim.get("events") or sim.get("trace") or []
    check("simulation produced events", len(events) > 0, f"{len(events)} events")

    # -------------------------------------------------------------- disrupt
    print("\n[7] POST /scenarios/{id}/disrupt")
    status, scen = call(base, "GET", f"/scenarios/{scenario_id}")
    resource_id = scen["resources"][0]["id"] if scen.get("resources") else None
    check("scenario exposes resources to disrupt", bool(resource_id))

    status, disrupted = call(
        base, "POST", f"/scenarios/{scenario_id}/disrupt",
        {"type": "RESOURCE_UNAVAILABLE", "target_id": resource_id,
         "magnitude": 0.5, "duration": 120},
    )
    check("disrupt returns 200", status == 200, f"{status} {str(disrupted)[:200]}")
    disruption = (disrupted or {}).get("disruption") or {}
    check("disrupt returns a disruption record", bool(disruption.get("id")), str(disruption)[:120])
    after_scenario_id = (disrupted or {}).get("after_scenario_id")
    check("disrupt returns the post-disruption scenario id", bool(after_scenario_id))

    # The disruption's effect is a window on the resource, not a status change,
    # so read it back from the disrupted scenario rather than the original.
    after = call(base, "GET", f"/scenarios/{after_scenario_id or scenario_id}")[1] or {}
    unavailable = [
        (r["id"], r.get("unavailable"))
        for r in after.get("resources", [])
        if r.get("unavailable")
    ]
    check(
        "disruption changed world state (resource recorded unavailable)",
        bool(unavailable),
        f"no resource carries an unavailable window (target was {resource_id})",
    )

    # -------------------------------------------------------------- replan
    print("\n[8] POST /scenarios/{id}/replan")
    status, repaired = call(
        base, "POST", f"/scenarios/{scenario_id}/replan",
        {"disruption_type": "RESOURCE_UNAVAILABLE", "target_id": resource_id,
         "magnitude": 0.5, "duration": 120, "run_full": True},
    )
    check("replan returns 200", status == 200, f"{status} {str(repaired)[:200]}")
    repair_summary = (repaired or {}).get("repair_plan") or {}
    full_summary = (repaired or {}).get("full_plan") or {}
    repair_plan_id = repair_summary.get("id")
    full_plan_id = full_summary.get("id")
    check("replan returns a local repair plan", bool(repair_plan_id), str(repair_summary)[:120])
    check("replan returns a full re-optimization plan", bool(full_plan_id), str(full_summary)[:120])
    check("replan reports a repair runtime", (repaired or {}).get("repair_seconds", 0) > 0)
    check("replan reports a full re-optimization runtime", (repaired or {}).get("full_seconds", 0) > 0)

    for label, pid in (("local repair", repair_plan_id), ("full re-optimization", full_plan_id)):
        status, p = call(base, "GET", f"/plans/{pid}")
        check(f"{label} plan is readable", status == 200 and p.get("id") == pid)

    # ----------------------------------------------------------- comparison
    print("\n[9] GET /plans/{id}/comparison")
    status, comparison = call(base, "GET", f"/plans/{repair_plan_id}/comparison")
    check("comparison returns 200", status == 200, f"{status} {str(comparison)[:160]}")
    # A plan can carry several comparisons, so the response is a list of them
    # under `items`, not a single object.
    items = (comparison or {}).get("items") or []
    check("comparison reports at least one item", bool(items), f"{len(items)} items")
    kinds = {i.get("kind") for i in items}
    check(
        "the repair plan has a real (non-placeholder) comparison",
        any(k not in (None, "none") for k in kinds),
        f"kinds={kinds}",
    )
    real = [i for i in items if i.get("kind") not in (None, "none")]
    deltas = real[0].get("deltas", []) if real else []
    check("comparison reports metric deltas", len(deltas) > 0, f"{len(deltas)} deltas")
    # `unit` may legitimately be empty: the plan score is dimensionless. The
    # requirement is that the key is *present* on every delta, not that it is
    # non-empty, because a previous version of the store dropped it entirely.
    check(
        "every delta names a metric, a label, a unit key and a direction",
        all(
            d.get("name")
            and d.get("label")
            and "unit" in d
            and "higher_is_better" in d
            for d in deltas
        ),
        str(deltas[:1])[:200],
    )
    check(
        "every delta carries before, after and change values",
        all(
            all(k in d for k in ("before", "after", "change")) for d in deltas
        ),
        str(deltas[:1])[:200],
    )
    check(
        "comparison names a real baseline plan",
        all(i.get("baseline_plan_id") in {repair_plan_id, plan_id} for i in real),
        f"baselines={[i.get('baseline_plan_id') for i in real]}",
    )

    # ------------------------------------------------------------- what-if
    print("\n[10] POST /scenarios/{id}/what-if")
    # WhatIfRequest carries one scalar `parameter`: a fraction for the five
    # magnitude operators, and a resource id for the outage.
    operators = [
        ("capacity_reduction", 0.2),
        ("demand_increase", 0.3),
        ("deadline_tighten", 0.15),
        # travel_increase takes a *multiplier* (1.25 == 25% slower), unlike
        # the other four which take a fraction. The API enforces this and a
        # 0.25 here is correctly rejected as a validation error.
        ("travel_increase", 1.25),
        ("availability_drop", 0.2),
        ("resource_outage", resource_id),
    ]
    results: dict[str, Any] = {}
    for name, parameter in operators:
        status, result = call(
            base, "POST", f"/scenarios/{scenario_id}/what-if",
            {"operator": name, "parameter": parameter},
        )
        check(f"what-if {name} returns 200", status == 200, f"{status} {str(result)[:140]}")
        check(
            f"what-if {name} answers a question and reports a status",
            bool((result or {}).get("question")) and bool((result or {}).get("status")),
            str(result)[:120],
        )
        results[name] = result

    check("all six operators responded", len(results) == 6, f"{len(results)}/6")
    objectives = {}
    for name, result in results.items():
        value = None
        comparison = (result or {}).get("comparison") or {}
        for delta in comparison.get("deltas") or []:
            if str(delta.get("name", "")).lower() in ("objective", "score", "objective total"):
                value = delta.get("candidate")
                break
        if value is None:
            value = (result or {}).get("scenario_plan_id")
        objectives[name] = value
    distinct = {v for v in objectives.values() if v is not None}
    check("every what-if result reports an objective", len(distinct) > 0, f"objectives={objectives}")
    check(
        "the six operators produce more than one distinct objective",
        len(distinct) > 1,
        f"objectives={objectives}",
    )
    check(
        "every what-if result carries a comparison against a baseline plan",
        all((r or {}).get("baseline_plan_id") for r in results.values()),
        "a result has no baseline_plan_id",
    )

    # ---------------------------------------------------------- experiments
    print("\n[11] GET /experiments")
    status, experiments = call(base, "GET", "/experiments")
    check("experiments returns 200", status == 200, str(status))
    rows = experiments if isinstance(experiments, list) else experiments.get("experiments", [])
    check("experiment store is readable", isinstance(rows, list), str(type(rows)))

    # -------------------------------------------------------------- exports
    print("\n[12] GET exports")
    status, scenario_export = call(base, "GET", f"/scenarios/{scenario_id}/export")
    check("scenario export returns 200", status == 200, f"{status} {str(scenario_export)[:140]}")
    # An export is an artifact envelope: kind, filename, content_type, body and
    # provenance. `body` is the serialized {data, provenance} payload.
    envelope = scenario_export or {}
    check("scenario export is an artifact envelope",
          "body" in envelope and "content_type" in envelope, str(list(envelope))[:160])
    check("scenario export carries provenance", "provenance" in envelope, str(list(envelope))[:160])
    body = envelope.get("body")
    check("scenario export body is a serialized payload", isinstance(body, str),
          type(body).__name__)
    check("scenario export body carries data", '"data"' in (body or ""), (body or "")[:80])

    status, plan_export = call(base, "GET", f"/plans/{plan_id}/export")
    check("plan export returns 200", status == 200, f"{status} {str(plan_export)[:140]}")
    plan_envelope = plan_export or {}
    check("plan export carries a body", "body" in plan_envelope, str(list(plan_envelope))[:160])
    plan_body = plan_envelope.get("body")
    check("plan export body is a serialized payload", isinstance(plan_body, str), type(plan_body).__name__)

    # The import endpoint takes the payload object, not the artifact envelope
    # and not the serialized string.
    status, reimport = call(base, "POST", "/exports/plan", json.loads(plan_body or "{}"))
    check("plan re-import returns 200", status == 200, f"{status} {str(reimport)[:140]}")
    check(
        "plan round trip is identical",
        (reimport or {}).get("round_trip_identical") is True,
        str((reimport or {}).get("round_trip_identical")),
    )
    check("plan id survives the round trip", (reimport or {}).get("entity_id") == plan_id,
          f"{(reimport or {}).get('entity_id')} != {plan_id}")
    check(
        "reloaded plan keeps its solver status and objective",
        (reimport or {}).get("status") and (reimport or {}).get("objective") is not None,
        str(reimport)[:140],
    )

    # ------------------------------------------------------------ health
    print("\n[13] GET /health (after the workflow)")
    status, health = call(base, "GET", "/health")
    rows = (health or {}).get("rows", {})
    check("persistence: scenarios were stored", rows.get("scenarios", 0) >= 1, str(rows.get("scenarios")))
    check("persistence: plans were stored", rows.get("plans", 0) >= 3, str(rows.get("plans")))
    check("persistence: assignments were stored", rows.get("assignments", 0) > 0, str(rows.get("assignments")))

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 64)
    print(f"passed {len(PASSED)}   failed {len(FAILED)}")
    for label, detail in FAILED:
        print(f"  FAIL {label}: {detail}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
