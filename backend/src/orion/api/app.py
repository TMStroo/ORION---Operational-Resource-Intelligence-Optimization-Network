"""FastAPI application for ORION.

The API is a thin shell over the planning core. It does not reimplement
scheduling, does not cache solver results, and does not paper over a solver
failure: a plan that is INFEASIBLE is served as INFEASIBLE with a 200, because
"this scenario has no feasible plan" is an answer, not a server error. Errors
that *are* server errors - unknown ids, a store that will not open - map to 404
and 500 respectively, with the detail in the body.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from orion import __version__
from orion.api import serializers as S
from orion.api.schemas import (
    CreateScenarioRequest,
    DisruptRequest,
    DisruptResponse,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentSummary,
    HealthResponse,
    OptimizeRequest,
    PlanComparisonResponse,
    PlanDetail,
    PlanListResponse,
    PlanSummary,
    ReplanRequest,
    ReplanResponse,
    ScenarioDetail,
    ScenarioListResponse,
    SimulateRequest,
    SimulateResponse,
    ValidationIssue,
    ValidationResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from orion.domain.errors import OrionError, UnknownEntityError
from orion.domain.events import analyse_impact
from orion.planning.comparison import compare_plans
from orion.planning.planner import Planner
from orion.planning.replanner import Replanner
from orion.planning.whatif import WHAT_IF_OPERATORS, WhatIfEngine
from orion.simulation.engine import SimulationEngine
from orion.store import Store

#: How many events a simulate response returns by default. A 60-task scenario
#: produces thousands; sending them all makes the overview page unusable.
DEFAULT_EVENT_LIMIT = 500


def default_database_url() -> str:
    return os.environ.get("ORION_DATABASE_URL", "sqlite:///orion.db")


def default_experiments_root() -> Path:
    return Path(os.environ.get("ORION_EXPERIMENTS_ROOT", "experiments"))


def create_app(
    database_url: str | None = None,
    *,
    experiments_root: str | Path | None = None,
) -> FastAPI:
    """Build the application.

    Dependencies are wired here rather than at import time so tests can run
    against an in-memory database without touching the filesystem.
    """
    store = Store(
        database_url or default_database_url(),
        experiments_root=Path(experiments_root or default_experiments_root()),
    )
    store.create_all()
    # One planner and one replanner, reused. Both are stateless with respect to
    # a request; the seed is passed per call.
    planner = Planner()
    whatif = WhatIfEngine(planner)

    app = FastAPI(
        title="ORION",
        version=__version__,
        description=(
            "Operational Resource Intelligence & Optimization Network - "
            "constrained resource allocation, simulation, disruption recovery "
            "and what-if planning."
        ),
    )
    app.state.store = store
    app.state.planner = planner
    app.state.whatif = whatif
    app.state.experiments_root = Path(experiments_root or default_experiments_root())
    app.state.database_url = store.url

    # Mirror the filesystem evidence into the database once at startup. Without
    # this the /experiments endpoints answer correctly but with zero rows,
    # because the benchmarks write to disk and never touch the relational store.
    app.state.experiments_imported = 0
    if app.state.experiments_root.is_dir():
        try:
            app.state.experiments_imported = store.import_experiments(app.state.experiments_root)
        except Exception as exc:  # pragma: no cover - a bad evidence dir must not stop the API
            logging.getLogger("orion.api").warning(
                "could not import experiments from %s: %s", app.state.experiments_root, exc
            )

    # ---------------------------------------------------------------- errors

    @app.exception_handler(OrionError)
    def _domain_error(_request: Request, exc: OrionError) -> JSONResponse:
        # A domain error is a real answer about the problem, so it keeps its
        # message rather than being flattened to "error".
        #
        # UnknownEntityError is the exception: it means the *caller* asked for
        # something that does not exist, which is a 404, not a 422. Reporting it
        # as 422 tells a client the request was malformed when it was well-formed
        # and simply pointed at a stale id.
        code = (
            status.HTTP_404_NOT_FOUND
            if isinstance(exc, UnknownEntityError)
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        return JSONResponse(
            status_code=code,
            content={"error": type(exc).__name__, "detail": str(exc), "context": {}},
        )

    def get_store() -> Iterator[Store]:
        yield app.state.store

    def _scenario_or_404(scenario_id: str, store: Store) -> Any:
        if not store.scenario_exists(scenario_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown scenario {scenario_id!r}",
            )
        return store.load_scenario(scenario_id)

    def _plan_or_404(plan_id: str, store: Store) -> tuple[Any, Any]:
        row = store.get_plan_row(plan_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown plan {plan_id!r}"
            )
        return row, store.load_plan(plan_id)

    # ---------------------------------------------------------------- health

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(store: Store = Depends(get_store)) -> HealthResponse:
        stats = store.stats()
        live = superseded = 0
        root = app.state.experiments_root
        if root.is_dir():
            from orion.experiments.tracker import ExperimentStore

            filesystem = ExperimentStore(root)
            live = len(filesystem.live())
            superseded = len(filesystem.list_ids()) - live
        return HealthResponse(
            status="ok",
            version=__version__,
            database=store.url.split(":", 1)[0],
            store_url=store.url,
            rows=stats.as_dict(),
            live_experiments=live,
            superseded_experiments=superseded,
            solvers=list(SOLVER_DESCRIPTIONS),
        )

    # ------------------------------------------------------------- scenarios

    @app.get("/scenarios", response_model=ScenarioListResponse, tags=["scenarios"])
    def list_scenarios(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        store: Store = Depends(get_store),
    ) -> ScenarioListResponse:
        rows = store.list_scenarios(limit=limit, offset=offset)
        return ScenarioListResponse(
            items=[S.scenario_summary(r) for r in rows],
            total=len(rows),
            limit=limit,
            offset=offset,
        )

    @app.post(
        "/scenarios",
        response_model=ScenarioDetail,
        status_code=status.HTTP_201_CREATED,
        tags=["scenarios"],
    )
    def create_scenario(
        request: CreateScenarioRequest,
        store: Store = Depends(get_store),
    ) -> ScenarioDetail:
        from orion.data.scenario_generator import difficulty_config, generate

        # The scenario id is a pure function of the request, so a repeat POST
        # is a repeat ask for the same thing. Returning the stored row keeps the
        # endpoint idempotent; regenerating and overwriting would silently reset
        # a scenario that already has plans against it.
        existing_id = f"SCN-{request.name}-{request.seed}"
        if store.scenario_exists(existing_id):
            row = next(
                (r for r in store.list_scenarios(limit=500) if r.id == existing_id), None
            )
            if row is not None:
                return S.scenario_detail(row, store.load_scenario(existing_id))

        config = difficulty_config(
            request.difficulty,
            seed=request.seed,
            name=request.name,
            num_tasks=request.task_count,
            num_teams=max(1, request.resource_count // 2),
            num_vehicles=max(1, request.resource_count - request.resource_count // 2),
            description=request.description,
        )
        scenario = generate(config)
        store.save_scenario(scenario, source="generated")
        row = next(
            (r for r in store.list_scenarios(limit=500) if r.id == scenario.id), None
        )
        if row is None:
            # The row was just written; if it is not visible, the store wrote
            # under a different id and the response would be wrong.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"scenario {scenario.id!r} was saved but could not be read back",
            )
        return S.scenario_detail(row, scenario)

    @app.get("/scenarios/{scenario_id}", response_model=ScenarioDetail, tags=["scenarios"])
    def get_scenario(scenario_id: str, store: Store = Depends(get_store)) -> ScenarioDetail:
        scenario = _scenario_or_404(scenario_id, store)
        row = next(r for r in store.list_scenarios(limit=500) if r.id == scenario_id)
        return S.scenario_detail(row, scenario)

    @app.post(
        "/scenarios/{scenario_id}/validate", response_model=ValidationResponse, tags=["scenarios"]
    )
    def validate_scenario(
        scenario_id: str, store: Store = Depends(get_store)
    ) -> ValidationResponse:
        """Structural validation of the scenario itself.

        This checks the *input*, not a plan: a task with no eligible resource is
        a property of the scenario, and reporting it here is what stops it
        showing up later as an unexplained NO_ELIGIBLE_PAIR failure.
        """
        scenario = _scenario_or_404(scenario_id, store)
        issues: list[ValidationIssue] = []
        eligible = _eligible_pairs(scenario)
        for task in scenario.tasks:
            if not eligible.get(task.id):
                issues.append(
                    ValidationIssue(
                        kind="NO_ELIGIBLE_RESOURCE",
                        detail=(
                            f"task {task.id} requires "
                            f"{sorted(task.required_capabilities) or ['(any)']}, "
                            "and no resource satisfies that"
                        ),
                        task_id=task.id,
                        severity=2,
                    )
                )
            if task.deadline < task.release_time:
                issues.append(
                    ValidationIssue(
                        kind="NEGATIVE_WINDOW",
                        detail=(
                            f"task {task.id} has deadline {task.deadline} before "
                            f"release {task.release_time}"
                        ),
                        task_id=task.id,
                        severity=2,
                    )
                )
        return ValidationResponse(
            valid=not issues,
            issues=issues,
            task_count=len(scenario.tasks),
            resource_count=len(scenario.resources),
        )

    # ----------------------------------------------------------------- plans

    @app.post(
        "/scenarios/{scenario_id}/optimize", response_model=PlanDetail, tags=["plans"]
    )
    def optimize(
        scenario_id: str, request: OptimizeRequest, store: Store = Depends(get_store)
    ) -> PlanDetail:
        """Solve a scenario and persist the result.

        The response is the plan whatever the solver's status. A TIME_LIMIT
        with a partial plan returns 200 and `status: TIME_LIMIT`, because the
        client needs to know a plan exists and that it is not proven optimal.
        """
        scenario = _scenario_or_404(scenario_id, store)
        result = app.state.planner.plan(
            scenario,
            solver=request.solver,
            time_budget_s=request.time_budget_s,
            seed=request.seed,
            explain=request.explain,
        )
        plan = result.plan
        plan_id = f"PLAN-{scenario_id}-{uuid.uuid4().hex[:8]}"
        store.save_plan(
            plan,
            scenario_id=scenario_id,
            label=request.label or f"{request.solver} plan",
            plan_id=plan_id,
        )
        row = store.get_plan_row(plan_id)
        return S.plan_detail(row, plan)

    @app.get("/plans", response_model=PlanListResponse, tags=["plans"])
    def list_plans(
        scenario_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        store: Store = Depends(get_store),
    ) -> PlanListResponse:
        rows = store.list_plans(scenario_id=scenario_id, limit=limit, offset=offset)
        return PlanListResponse(items=[S.plan_summary(r) for r in rows], total=len(rows))

    @app.get("/plans/{plan_id}", response_model=PlanDetail, tags=["plans"])
    def get_plan(plan_id: str, store: Store = Depends(get_store)) -> PlanDetail:
        row, plan = _plan_or_404(plan_id, store)
        return S.plan_detail(row, plan)

    # ------------------------------------------------------------ simulation

    @app.post(
        "/scenarios/{scenario_id}/simulate", response_model=SimulateResponse, tags=["simulation"]
    )
    def simulate(
        scenario_id: str, request: SimulateRequest, store: Store = Depends(get_store)
    ) -> SimulateResponse:
        """Run the discrete-event engine on the scenario's newest plan."""
        scenario = _scenario_or_404(scenario_id, store)
        plan_id: str | None = None
        if request.plan_id is not None:
            plan_id = request.plan_id
            row, plan = _plan_or_404(request.plan_id, store)
            if row.scenario_id != scenario_id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"plan {request.plan_id!r} belongs to scenario "
                        f"{row.scenario_id!r}, not {scenario_id!r}"
                    ),
                )
        else:
            plans = store.list_plans(scenario_id=scenario_id, limit=1)
            if not plans:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"scenario {scenario_id!r} has no plan; call "
                        f"POST /scenarios/{scenario_id}/optimize first"
                    ),
                )
            plan_id = plans[0].id
            plan = store.load_plan(plan_id)

        # A plan with no assignments simulates to an empty trace, which reads as
        # "nothing happened" rather than "this solver produced no schedule".
        # Saying so is the difference between a bug report and a fact.
        if not plan.assignments:
            return SimulateResponse(
                run_id="",
                plan_id=plan_id or plan.id,
                plan_status=plan.status,
                status="NOT_APPLICABLE",
                start_time=0,
                end_time=0,
                completed_tasks=0,
                late_tasks=0,
                failed_tasks=0,
                replans_performed=0,
                runtime_s=0.0,
                events=[],
                event_count=0,
            )

        engine = SimulationEngine(scenario, plan, respect_failures=request.respect_failures)
        result = engine.run(until=request.until, max_events=request.max_events)

        run_id = f"SIM-{uuid.uuid4().hex[:10]}"
        # Persist the run so the event stream is inspectable later. A storage
        # failure is reported, not swallowed: an earlier version caught it
        # broadly and returned a plausible-looking run id with an empty trace,
        # which is indistinguishable from a simulation that did nothing.
        store.save_simulation_run(run_id, scenario_id, result, seed=request.seed)

        events = list(result.trace)
        return SimulateResponse(
            run_id=run_id,
            plan_id=plan_id or plan.id,
            plan_status=plan.status,
            status=result.status,
            start_time=result.start_time,
            end_time=result.end_time,
            completed_tasks=result.completed_tasks,
            late_tasks=result.late_tasks,
            failed_tasks=result.failed_tasks,
            replans_performed=result.replans_performed,
            runtime_s=result.runtime_s,
            events=[S.sim_event_view(e, i) for i, e in enumerate(events[:DEFAULT_EVENT_LIMIT])],
            event_count=len(events),
        )

    # ------------------------------------------------------------ disruption

    @app.post(
        "/scenarios/{scenario_id}/disrupt", response_model=DisruptResponse, tags=["disruption"]
    )
    def disrupt(
        scenario_id: str, request: DisruptRequest, store: Store = Depends(get_store)
    ) -> DisruptResponse:
        """Apply a disruption and report which work it puts at risk.

        Records the event against the scenario. A repeated id is refused rather
        than applied twice, and `recorded: false` says so explicitly.
        """
        scenario = _scenario_or_404(scenario_id, store)
        disruption = _build_disruption(request, scenario)
        after = disruption.apply(scenario)
        plans = store.list_plans(scenario_id=scenario_id, limit=1)
        plan = store.load_plan(plans[0].id) if plans else None
        impact = analyse_impact(disruption, scenario, after, plan) if plan else None
        recorded = store.record_disruption(scenario_id, disruption, impact)
        # Disruption.apply keeps the scenario id, so persisting `after` as-is
        # would overwrite the pre-disruption world. The disrupted state is a
        # distinct scenario and needs its own id, or "before" and "after" become
        # the same row and the comparison is vacuous.
        after = _derange(after, f"{scenario_id}-after-{disruption.id}")
        after_id = store.save_scenario(after, source="disrupted")
        return DisruptResponse(
            disruption=S.disruption_view(disruption, impact),
            recorded=recorded,
            after_scenario_id=after_id,
        )

    # -------------------------------------------------------------- replanning

    @app.post(
        "/scenarios/{scenario_id}/replan", response_model=ReplanResponse, tags=["replanning"]
    )
    def replan(
        scenario_id: str, request: ReplanRequest, store: Store = Depends(get_store)
    ) -> ReplanResponse:
        """Local repair, and optionally full re-optimization, after a disruption.

        Both paths are reported side by side, including the cases where repair
        wins on churn and full re-optimization wins on objective. Neither is
        presented as the answer.
        """
        scenario = _scenario_or_404(scenario_id, store)
        plans = store.list_plans(scenario_id=scenario_id, limit=1)
        if not plans:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"scenario {scenario_id!r} has no plan to repair",
            )
        baseline_row = plans[0]
        baseline = store.load_plan(baseline_row.id)

        disruption = _build_disruption(
            DisruptRequest(
                type=request.disruption_type,
                target_id=request.target_id,
                magnitude=request.magnitude,
                duration=request.duration,
                at_time=request.at_time,
                seed=request.seed,
            ),
            scenario,
        )
        replanner = Replanner(solver=request.solver, seed=request.seed)
        if request.time_budget_s is not None:
            replanner.time_budget_s = request.time_budget_s
        outcome = replanner.replan(
            scenario,
            baseline,
            disruption,
            run_full=request.run_full,
            solver=request.solver,
            time_budget_s=request.time_budget_s,
            **({"strategies": tuple(request.strategies)} if request.strategies else {}),
        )
        store.record_disruption(scenario_id, disruption, outcome.impact)
        store.save_scenario(
            _derange(outcome.after_scenario, f"{scenario_id}-after-{disruption.id}"),
            source="disrupted",
        )

        repair_row = None
        repair_comparison = None
        preserved = 0.0
        recovery = 0.0
        if outcome.repair.plan is not None:
            repair_id = f"PLAN-{scenario_id}-repair-{uuid.uuid4().hex[:8]}"
            store.save_plan(
                outcome.repair.plan,
                scenario_id=scenario_id,
                label="local repair",
                plan_id=repair_id,
            )
            repair_row = store.get_plan_row(repair_id)
            comparison = compare_plans(baseline, outcome.repair.plan, outcome.after_scenario)
            repair_comparison = S.comparison_view(
                comparison,
                kind="local_repair",
                baseline_plan_id=baseline.id,
                candidate_plan_id=repair_id,
            )
            preserved = (
                1.0 - repair_comparison.churn.churn_ratio
                if repair_comparison.churn.total_before
                else 0.0
            )
            # Recovery is repair quality relative to full re-optimization. It is
            # reported as-is: below 1.0 means the full re-plan was better, and
            # hiding that would make repair look universal.
            if outcome.full is not None and outcome.full.objective.total:
                recovery = min(
                    1.0,
                    max(0.0, outcome.full.objective.total / outcome.repair.plan.objective.total)
                    if outcome.repair.plan.objective.total
                    else 0.0,
                )

        full_row = None
        full_comparison = None
        if outcome.full is not None:
            full_id = f"PLAN-{scenario_id}-full-{uuid.uuid4().hex[:8]}"
            store.save_plan(
                outcome.full,
                scenario_id=scenario_id,
                label="full re-optimization",
                plan_id=full_id,
            )
            full_row = store.get_plan_row(full_id)
            full_comparison = S.comparison_view(
                compare_plans(baseline, outcome.full, outcome.after_scenario),
                kind="full_reoptimization",
                baseline_plan_id=baseline.id,
                candidate_plan_id=full_id,
            )

        return ReplanResponse(
            disruption=S.disruption_view(outcome.disruption, outcome.impact),
            impact=S.disruption_view(outcome.disruption, outcome.impact),
            repair=S.repair_view(outcome.repair),
            repair_plan=S.plan_summary(repair_row) if repair_row else None,
            full_plan=S.plan_summary(full_row) if full_row else None,
            repair_seconds=outcome.repair.seconds,
            full_seconds=outcome.full_seconds,
            full_status=outcome.full_status,
            full_reason=outcome.full_reason,
            solver=outcome.solver,
            time_budget_s=outcome.time_budget_s,
            repair_comparison=repair_comparison,
            full_comparison=full_comparison,
            repair_preserved_ratio=preserved,
            recovery_pct=recovery,
        )

    # --------------------------------------------------------------- what-if

    @app.post(
        "/scenarios/{scenario_id}/what-if", response_model=WhatIfResponse, tags=["whatif"]
    )
    def what_if(
        scenario_id: str, request: WhatIfRequest, store: Store = Depends(get_store)
    ) -> WhatIfResponse:
        """Re-solve a modified scenario and compare it with the baseline."""
        if request.operator not in WHAT_IF_OPERATORS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"unknown what-if operator {request.operator!r}; "
                    f"supported: {sorted(WHAT_IF_OPERATORS)}"
                ),
            )
        scenario = _scenario_or_404(scenario_id, store)
        plans = store.list_plans(scenario_id=scenario_id, limit=1)
        if not plans:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"scenario {scenario_id!r} has no baseline plan to compare against",
            )
        baseline = store.load_plan(plans[0].id)
        result = app.state.whatif.run(
            scenario, baseline, request.operator, request.parameter, seed=request.seed
        )
        candidate_id = f"PLAN-{scenario_id}-whatif-{uuid.uuid4().hex[:8]}"
        store.save_plan(
            result.scenario_plan,
            scenario_id=scenario_id,
            label=f"what-if: {request.operator}",
            plan_id=candidate_id,
        )
        store.record_comparison(
            candidate_id, baseline.id, candidate_id, f"what_if:{request.operator}",
            result.comparison,
        )
        return WhatIfResponse(
            operator=result.operator,
            question=result.question,
            parameter=result.parameter,
            modification=result.modification,
            status=result.status,
            seconds=result.seconds,
            baseline_plan_id=baseline.id,
            scenario_plan_id=candidate_id,
            comparison=S.comparison_view(
                result.comparison,
                kind=f"what_if:{result.operator}",
                baseline_plan_id=baseline.id,
                candidate_plan_id=candidate_id,
            ),
        )

    # ---------------------------------------------------------- comparisons

    @app.get(
        "/plans/{plan_id}/comparison", response_model=PlanComparisonResponse, tags=["plans"]
    )
    def plan_comparison(
        plan_id: str, store: Store = Depends(get_store)
    ) -> PlanComparisonResponse:
        """Every recorded comparison involving a plan.

        The flattened `metrics` map exists so a client can render a table
        without walking the comparison list and matching on labels.
        """
        row, _ = _plan_or_404(plan_id, store)
        stored = store.list_comparisons(plan_id)
        items: list[Any] = []
        metrics: dict[str, dict[str, float]] = {}
        for record in stored:
            import json as _json

            deltas = _json.loads(record.deltas_json or "[]")
            churn = S.ChurnView(
                changed_assignments=record.churn_changed,
                rescheduled_tasks=record.churn_changed,
                total_before=record.churn_total,
                total_after=record.churn_total,
                tolerance_minutes=0,
                travel_delta_km=0.0,
                cost_delta=0.0,
                churn_ratio=record.churn_ratio,
            )
            items.append(
                S.ComparisonView(
                    kind=record.kind,
                    baseline_plan_id=record.baseline_plan_id,
                    candidate_plan_id=record.candidate_plan_id,
                    churn=churn,
                    deltas=[S.MetricDeltaView(**d) for d in deltas],
                )
            )
            bucket = metrics.setdefault(record.kind, {})
            for delta in deltas:
                bucket[str(delta["label"])] = float(delta["after"])
        if not items:
            # A plan nobody compared against is a real state, not an error.
            items.append(
                S.ComparisonView(
                    kind="none",
                    baseline_plan_id=row.scenario_id,
                    candidate_plan_id=plan_id,
                    churn=S.ChurnView(
                        changed_assignments=0,
                        rescheduled_tasks=0,
                        total_before=int(row.tasks_assigned or 0),
                        total_after=int(row.tasks_assigned or 0),
                        tolerance_minutes=0,
                        travel_delta_km=0.0,
                        cost_delta=0.0,
                    ),
                )
            )
        return PlanComparisonResponse(
            plan_id=plan_id,
            items=items,
            baseline_plan_id=items[0].baseline_plan_id,
            metrics=metrics,
        )

    # ----------------------------------------------------------- experiments

    @app.get("/experiments", response_model=ExperimentListResponse, tags=["experiments"])
    def list_experiments(
        live_only: bool = False,
        kind: str | None = None,
        store: Store = Depends(get_store),
    ) -> ExperimentListResponse:
        rows = store.list_experiments(live_only=live_only, kind=kind)
        live = sum(1 for r in rows if r.status == "succeeded")
        return ExperimentListResponse(
            items=[_experiment_summary(r) for r in rows],
            total=len(rows),
            live=live,
            superseded=len(rows) - live,
        )

    @app.get(
        "/experiments/{experiment_id}", response_model=ExperimentDetail, tags=["experiments"]
    )
    def get_experiment(
        experiment_id: str, store: Store = Depends(get_store)
    ) -> ExperimentDetail:
        import json as _json

        record = store.get_experiment(experiment_id)
        return ExperimentDetail(
            **_experiment_summary(record).model_dump(),
            manifest=_json.loads(record.manifest_json or "{}"),
            summary=_json.loads(record.summary_json or "{}"),
            metrics=_json.loads(record.metrics_json or "{}"),
            rows=store.experiment_rows(experiment_id),
        )

    return app


def _experiment_summary(row: Any) -> ExperimentSummary:
    return ExperimentSummary(
        id=row.id,
        kind=row.kind,
        status=row.status,
        commit=row.commit or "",
        seed=int(row.seed or 0),
        row_count=int(row.row_count or 0),
        created_at=row.created_at.isoformat() if row.created_at else "",
        superseded_by=row.superseded_by,
        supersede_reason=row.supersede_reason or "",
    )


def _eligible_pairs(scenario: Any) -> dict[str, int]:
    """Count eligible resources per task, using the same rule the solvers use."""
    counts: dict[str, int] = {}
    for task in scenario.tasks:
        n = 0
        for res in scenario.resources:
            if not task.required_capabilities.issubset(res.capabilities):
                continue
            if res.max_work_minutes < task.duration:
                continue
            n += 1
        counts[task.id] = n
    return counts


def _derange(scenario: Any, new_id: str) -> Any:
    """Return a copy of *scenario* under a different id."""
    import dataclasses

    return dataclasses.replace(scenario, id=new_id)


def _build_disruption(request: DisruptRequest, scenario: Any) -> Any:
    """Build a Disruption, defaulting a sensible target when none is given.

    A disruption aimed at nothing is a no-op that looks like a result, so an
    unknown type is a 422 and a scenario with no resources is a 422 too.
    DisruptionType is a namespace of string constants, not an Enum, so
    membership is checked against the known names rather than by calling it.
    """
    from orion.domain.events import Disruption, DisruptionSeverity, DisruptionType

    known = {n for n in dir(DisruptionType) if n.isupper()}
    if request.type not in known:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown disruption type {request.type!r}; supported: {sorted(known)}",
        )

    target = request.target_id
    if target is None:
        candidates = [r.id for r in scenario.resources if r.kind in ("vehicle", "team")]
        target = candidates[0] if candidates else (
            scenario.resources[0].id if scenario.resources else None
        )
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="scenario has no resources to disrupt",
            )

    magnitude = float(request.magnitude or 0.0)
    severity = (
        DisruptionSeverity.HIGH
        if magnitude >= 0.5
        else DisruptionSeverity.MEDIUM
        if magnitude >= 0.2
        else DisruptionSeverity.LOW
    )
    at_time = (
        request.at_time
        if request.at_time is not None
        else int(
            scenario.horizon.start
            + (scenario.horizon.end - scenario.horizon.start) * 0.25
        )
    )
    return Disruption(
        id=f"DSR-{uuid.uuid4().hex[:8]}",
        type=request.type,
        timestamp=at_time,
        target_id=target,
        magnitude=magnitude,
        duration=request.duration,
        description=f"{request.type} affecting {target} at t={at_time}",
        severity=severity,
        payload={},
    )


# Imported at the bottom to avoid a circular import at module load.
from orion.domain.plans import SOLVER_DESCRIPTIONS  # noqa: E402


app = None


def get_app() -> FastAPI:
    """Lazily built module-level app, for `uvicorn orion.api.app:get_app`."""
    global app
    if app is None:
        app = create_app()
    return app
