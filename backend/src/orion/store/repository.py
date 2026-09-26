"""Repository layer: domain objects in, rows out, and back again.

Round-tripping is the point. ``save_scenario`` then ``load_scenario`` returns a
scenario whose optimisation result is identical to the original's, which is
what lets the API serve a stored plan without re-solving it and getting a
different answer.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from orion.domain.entities import Scenario
from orion.domain.plans import Plan
from orion.domain.travel import TravelMatrix
from orion.experiments.tracker import ExperimentStore
from orion.domain.errors import ConfigError, UnknownEntityError
from orion.store.models import (
    AssignmentRow,
    Base,
    ComparisonRow,
    DisruptionRow,
    ExperimentRow,
    PlanRow,
    ResourceRow,
    ScenarioRow,
    SimulationEventRow,
    SimulationRunRow,
    SolverRunRow,
    StoreStats,
    TaskRow,
    dumps,
    join_values,
    loads,
    split_values,
)

DEFAULT_URL = "sqlite:///orion.db"


def _build_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine, enabling foreign keys on SQLite.

    SQLite does not enforce foreign keys unless asked. Without this, deleting a
    scenario would leave its tasks behind and the row counts in the audit would
    quietly disagree with reality.
    """
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_fks(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


class Store:
    """Relational store for scenarios, plans, disruptions and experiments.

    SQLite by default so the demo needs no server. A PostgreSQL URL works
    unchanged; the schema uses no SQLite-specific type.
    """

    def __init__(
        self,
        url: str | Path | None = None,
        *,
        echo: bool = False,
        experiments_root: str | Path | None = None,
    ) -> None:
        self.url = self._normalise(url)
        self._experiments_root = Path(experiments_root or "experiments")
        self.engine = _build_engine(self.url, echo=echo)
        self._sessionmaker = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )

    @staticmethod
    def _normalise(url: str | Path | None) -> str:
        if url is None:
            return DEFAULT_URL
        text = str(url)
        if "://" in text:
            return text
        if text in {"", ":memory:"}:
            return "sqlite:///:memory:"
        # A bare path is interpreted as a SQLite file, which is the common case
        # in tests and in the demo.
        return f"sqlite:///{Path(text).resolve().as_posix()}"

    # ------------------------------------------------------------------ schema

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        return self._sessionmaker()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.dispose()

    def dispose(self) -> None:
        self.engine.dispose()

    def stats(self) -> StoreStats:
        with self.session() as session:
            def count(model: type[Base]) -> int:
                return int(session.execute(select(func.count()).select_from(model)).scalar_one())

            return StoreStats(
                scenarios=count(ScenarioRow),
                tasks=count(TaskRow),
                resources=count(ResourceRow),
                plans=count(PlanRow),
                assignments=count(AssignmentRow),
                solver_runs=count(SolverRunRow),
                disruptions=count(DisruptionRow),
                simulation_events=count(SimulationEventRow),
                comparisons=count(ComparisonRow),
                experiments=count(ExperimentRow),
            )

    # --------------------------------------------------------------- scenarios

    def save_scenario(self, scenario: Scenario, *, source: str = "generated") -> str:
        """Insert or replace a scenario and its task/resource rows."""
        with self.session() as session:
            existing = session.get(ScenarioRow, scenario.id)
            if existing is not None:
                session.delete(existing)
                session.flush()

            row = ScenarioRow(
                id=scenario.id,
                name=scenario.name,
                description=scenario.description,
                task_count=len(scenario.tasks),
                resource_count=len(scenario.resources),
                horizon_end=scenario.horizon.end,
                source=source,
                tags=join_values(scenario.tags),
                graph_json=dumps(_scenario_graph(scenario)),
            )
            for task in scenario.tasks:
                row.tasks.append(
                    TaskRow(
                        task_id=task.id,
                        location=task.location,
                        release_time=task.release_time,
                        deadline=task.deadline,
                        duration=task.duration,
                        priority=str(getattr(task.priority, "value", task.priority)),
                        status=str(getattr(task.status, "value", task.status)),
                        required_capabilities=join_values(sorted(task.required_capabilities)),
                        dependencies=dumps(list(task.dependencies)),
                        extra_json=dumps(
                            {
                                "required_capacity": [
                                    {
                                        "dimension": str(
                                            getattr(c.dimension, "value", c.dimension)
                                        ),
                                        "amount": c.amount,
                                    }
                                    for c in task.required_capacity
                                ],
                                "service_minutes_target": task.service_minutes_target,
                                "max_lateness": task.max_lateness,
                            }
                        ),
                    )
                )
            for res in scenario.resources:
                row.resources.append(
                    ResourceRow(
                        resource_id=res.id,
                        kind=str(getattr(res.kind, "value", res.kind)),
                        home_location=res.home_location,
                        shift_start=res.shift_start,
                        shift_end=res.shift_end,
                        status=str(getattr(res.status, "value", res.status)),
                        capabilities=join_values(sorted(res.capabilities)),
                        max_work_minutes=res.max_work_minutes,
                        operating_cost_per_hour=res.operating_cost_per_hour,
                        capacity_json=dumps(dict(res.capacity)),
                        unavailable_json=dumps(
                            [[w.start, w.end] for w in res.unavailable]
                        ),
                        extra_json=dumps(
                            {
                                "speed_factor": res.speed_factor,
                                "fixed_dispatch_cost": res.fixed_dispatch_cost,
                            }
                        ),
                    )
                )
            session.add(row)
            session.commit()
        return scenario.id

    def load_scenario(self, scenario_id: str) -> Scenario:
        """Rebuild the exact domain scenario from stored rows."""
        with self.session() as session:
            row = session.get(ScenarioRow, scenario_id)
            if row is None:
                raise UnknownEntityError(f"unknown scenario {scenario_id!r}")
            return _scenario_from_row(row)

    def list_scenarios(self, *, limit: int = 100, offset: int = 0) -> list[ScenarioRow]:
        with self.session() as session:
            stmt = (
                select(ScenarioRow)
                .order_by(ScenarioRow.created_at.desc(), ScenarioRow.id)
                .limit(limit)
                .offset(offset)
                .options(
                    selectinload(ScenarioRow.tasks),
                    selectinload(ScenarioRow.resources),
                )
            )
            return list(session.execute(stmt).unique().scalars())

    def scenario_exists(self, scenario_id: str) -> bool:
        with self.session() as session:
            return session.get(ScenarioRow, scenario_id) is not None

    def delete_scenario(self, scenario_id: str) -> bool:
        with self.session() as session:
            row = session.get(ScenarioRow, scenario_id)
            if row is None:
                return False
            session.delete(row)
            session.commit()
        return True

    # ------------------------------------------------------------------- plans

    def save_plan(
        self,
        plan: Plan,
        *,
        scenario_id: str | None = None,
        label: str = "",
        plan_id: str | None = None,
    ) -> str:
        """Store a plan with its assignments and every solver run.

        All solver runs are persisted, including NOT_APPLICABLE and ERROR ones.
        A plan whose only recorded solver run is a decline is a different fact
        from a plan that was never attempted, and the table should say which.
        """
        resolved_scenario = scenario_id or getattr(plan, "scenario_id", "") or ""
        identifier = plan_id or getattr(plan, "id", "") or f"plan-{abs(hash(plan))}"
        with self.session() as session:
            if session.get(ScenarioRow, resolved_scenario) is None:
                raise UnknownEntityError(
                    f"cannot store a plan for unknown scenario {resolved_scenario!r}"
                )
            existing = session.get(PlanRow, identifier)
            if existing is not None:
                session.delete(existing)
                session.flush()

            objective = plan.objective
            row = PlanRow(
                id=identifier,
                scenario_id=resolved_scenario,
                label=label,
                status=str(getattr(plan, "status", "")),
                solver=str(getattr(plan, "solver_name", "") or ""),
                objective=float(getattr(objective, "total", 0.0) or 0.0),
                completion=float(getattr(plan, "completion", 0.0) or 0.0),
                late_tasks=int(getattr(plan, "late_tasks", 0) or 0),
                tasks_assigned=len(plan.assignments),
                runtime_s=float(getattr(plan, "runtime_seconds", 0.0) or 0.0),
                infeasible=bool(getattr(plan, "infeasible", False)),
                plan_json=dumps(plan.to_dict()),
            )
            for idx, assignment in enumerate(plan.assignments):
                row.assignments.append(
                    AssignmentRow(
                        task_id=assignment.task_id,
                        resource_id=assignment.resource_id,
                        sequence_index=int(getattr(assignment, "sequence_index", idx) or 0),
                        start=int(assignment.start),
                        end=int(assignment.end),
                        location=assignment.location,
                        travel_before=int(getattr(assignment, "travel_before", 0) or 0),
                        travel_distance_km=float(
                            getattr(assignment, "travel_distance_km", 0.0) or 0.0
                        ),
                        priority=str(
                            getattr(assignment.priority, "value", assignment.priority)
                        ),
                        lateness=int(getattr(assignment, "lateness", 0) or 0),
                    )
                )
            for run in getattr(plan, "solver_runs", ()) or ():
                row.solver_runs.append(
                    SolverRunRow(
                        solver=str(getattr(run, "solver", "")),
                        status=str(getattr(run, "status", "")),
                        objective=float(getattr(run, "objective", 0.0) or 0.0),
                        # The solver's own figure lives in `raw`; it is kept
                        # because it is what the fairness contract compares
                        # against the independently rescored objective.
                        internal_objective=float(
                            (getattr(run, "raw", {}) or {}).get("internal_objective", 0.0) or 0.0
                        ),
                        runtime_s=float(getattr(run, "runtime_s", 0.0) or 0.0),
                        tasks_assigned=int(
                            (getattr(run, "raw", {}) or {}).get("tasks_assigned", 0) or 0
                        ),
                        feasible=bool(getattr(run, "feasible", False)),
                        optimality_gap=(
                            float(run.optimality_gap)
                            if getattr(run, "optimality_gap", None) is not None
                            else None
                        ),
                        note=str(getattr(run, "notes", "") or ""),
                        detail_json=dumps(getattr(run, "raw", {}) or {}),
                    )
                )
            session.add(row)
            session.commit()
        return identifier

    def load_plan(self, plan_id: str) -> Plan:
        with self.session() as session:
            row = session.get(PlanRow, plan_id)
            if row is None:
                raise UnknownEntityError(f"unknown plan {plan_id!r}")
            return _plan_from_row(row)

    def list_plans(
        self, *, scenario_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[PlanRow]:
        with self.session() as session:
            stmt = select(PlanRow)
            if scenario_id is not None:
                stmt = stmt.where(PlanRow.scenario_id == scenario_id)
            stmt = (
                stmt.order_by(PlanRow.created_at.desc(), PlanRow.id)
                .limit(limit)
                .offset(offset)
                .options(
                    selectinload(PlanRow.assignments),
                    selectinload(PlanRow.solver_runs),
                )
            )
            return list(session.execute(stmt).unique().scalars())

    def get_plan_row(self, plan_id: str) -> PlanRow | None:
        with self.session() as session:
            return session.execute(
                select(PlanRow)
                .where(PlanRow.id == plan_id)
                .options(
                    selectinload(PlanRow.assignments),
                    selectinload(PlanRow.solver_runs),
                )
            ).unique().scalar_one_or_none()

    # ------------------------------------------------------------- disruptions

    def record_disruption(
        self, scenario_id: str, disruption: Any, impact: Any | None = None
    ) -> bool:
        """Record a disruption. Returns False if it was already recorded.

        The `applied_once` guard lives here so that a second application is
        reported rather than silently duplicating the event.
        """
        disruption_id = str(getattr(disruption, "id", "") or "")
        with self.session() as session:
            existing = session.execute(
                select(DisruptionRow).where(
                    DisruptionRow.scenario_id == scenario_id,
                    DisruptionRow.disruption_id == disruption_id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return False
            session.add(
                DisruptionRow(
                    scenario_id=scenario_id,
                    disruption_id=disruption_id,
                    type=str(getattr(disruption, "type", "")),
                    severity=str(getattr(disruption, "severity", "")),
                    target_id=getattr(disruption, "target_id", None),
                    timestamp=int(getattr(disruption, "timestamp", 0) or 0),
                    magnitude=float(getattr(disruption, "magnitude", 0.0) or 0.0),
                    duration=getattr(disruption, "duration", None),
                    description=str(getattr(disruption, "description", "") or ""),
                    applied_once=True,
                    affected_tasks=len(getattr(impact, "affected_tasks", ()) or ()),
                    affected_resources=len(getattr(impact, "affected_resources", ()) or ()),
                    impact_json=dumps(
                        {
                            "affected_resources": list(
                                getattr(impact, "affected_resources", ()) or ()
                            ),
                            "affected_tasks": list(getattr(impact, "affected_tasks", ()) or ()),
                            "at_risk_deadlines": list(
                                getattr(impact, "at_risk_deadlines", ()) or ()
                            ),
                            "severity": str(getattr(impact, "severity", "") or ""),
                        }
                        if impact is not None
                        else {}
                    ),
                )
            )
            session.commit()
        return True

    def list_disruptions(
        self, scenario_id: str | None = None
    ) -> list[DisruptionRow]:
        with self.session() as session:
            stmt = select(DisruptionRow)
            if scenario_id is not None:
                stmt = stmt.where(DisruptionRow.scenario_id == scenario_id)
            return list(session.execute(stmt.order_by(DisruptionRow.timestamp)).scalars())

    # -------------------------------------------------------------- simulation

    def save_simulation_run(
        self, run_id: str, scenario_id: str, result: Any, *, seed: int = 0
    ) -> str:
        """Store a simulation run and every event it emitted."""
        with self.session() as session:
            if session.get(SimulationRunRow, run_id) is not None:
                raise ConfigError(f"simulation run {run_id!r} already recorded")
            session.add(
                SimulationRunRow(
                    id=run_id,
                    scenario_id=scenario_id,
                    plan_id=str(getattr(result, "plan_id", "") or ""),
                    start_time=int(getattr(result, "start_time", 0) or 0),
                    end_time=int(getattr(result, "end_time", 0) or 0),
                    completed_tasks=int(getattr(result, "completed_tasks", 0) or 0),
                    late_tasks=int(getattr(result, "late_tasks", 0) or 0),
                    failed_tasks=int(getattr(result, "failed_tasks", 0) or 0),
                    replans_performed=int(getattr(result, "replans_performed", 0) or 0),
                    status=str(getattr(result, "status", "") or ""),
                    runtime_s=float(getattr(result, "runtime_s", 0.0) or 0.0),
                    seed=int(seed),
                    state_json=dumps(
                        {
                            "completed": list(getattr(result.final_state, "completed", ()) or ()),
                            "active": list(getattr(result.final_state, "active", ()) or ()),
                        }
                    ),
                )
            )
            for ordinal, event in enumerate(getattr(result, "trace", ()) or ()):
                session.add(
                    SimulationEventRow(
                        run_id=run_id,
                        ordinal=ordinal,
                        time=int(getattr(event, "time", 0) or 0),
                        kind=str(getattr(event, "kind", "") or getattr(event, "type", "")),
                        subject=str(getattr(event, "subject", "") or ""),
                        detail=str(getattr(event, "detail", "") or ""),
                        payload_json=dumps(getattr(event, "detail_json", {}) or {}),
                    )
                )
            session.commit()
        return run_id

    def list_simulation_events(
        self, run_id: str, *, limit: int = 1000
    ) -> list[SimulationEventRow]:
        with self.session() as session:
            stmt = (
                select(SimulationEventRow)
                .where(SimulationEventRow.run_id == run_id)
                .order_by(SimulationEventRow.ordinal)
                .limit(limit)
            )
            return list(session.execute(stmt).scalars())

    def list_simulation_runs(self, scenario_id: str | None = None) -> list[SimulationRunRow]:
        with self.session() as session:
            stmt = select(SimulationRunRow)
            if scenario_id is not None:
                stmt = stmt.where(SimulationRunRow.scenario_id == scenario_id)
            return list(session.execute(stmt.order_by(SimulationRunRow.created_at)).scalars())

    # -------------------------------------------------------------- comparisons

    def record_comparison(
        self,
        plan_id: str,
        baseline_plan_id: str,
        candidate_plan_id: str,
        kind: str,
        comparison: Any,
        *,
        recovery_pct: float = 0.0,
    ) -> int:
        churn = getattr(comparison, "churn", None)
        with self.session() as session:
            row = ComparisonRow(
                plan_id=plan_id,
                baseline_plan_id=baseline_plan_id,
                candidate_plan_id=candidate_plan_id,
                kind=kind,
                churn_changed=int(getattr(churn, "changed", 0) or 0),
                churn_total=int(getattr(churn, "total", 0) or 0),
                churn_ratio=float(getattr(churn, "ratio", 0.0) or 0.0),
                recovery_pct=float(recovery_pct or 0.0),
                deltas_json=dumps(
                    [
                        {
                            "label": str(getattr(d, "label", "")),
                            "before": float(getattr(d, "before", 0.0) or 0.0),
                            "after": float(getattr(d, "after", 0.0) or 0.0),
                            "change": float(getattr(d, "change", 0.0) or 0.0),
                        }
                        for d in getattr(comparison, "deltas", ()) or ()
                    ]
                ),
            )
            session.add(row)
            session.commit()
            return int(row.id)

    def list_comparisons(self, plan_id: str) -> list[ComparisonRow]:
        with self.session() as session:
            return list(
                session.execute(
                    select(ComparisonRow)
                    .where(ComparisonRow.plan_id == plan_id)
                    .order_by(ComparisonRow.id)
                ).scalars()
            )

    # -------------------------------------------------------------- experiments

    def import_experiments(self, root: str | Path) -> int:
        """Mirror the filesystem experiment store into the database.

        Superseded runs are imported with their status, not dropped, so the
        database shows the same history the filesystem does. The manifest is the
        authority for status; the CSV and JSON files beside it are copied
        verbatim.
        """
        filesystem = ExperimentStore(Path(root))
        imported = 0
        with self.session() as session:
            for experiment_id in filesystem.list_ids():
                if session.get(ExperimentRow, experiment_id) is not None:
                    continue
                manifest = filesystem.load(experiment_id)
                directory = filesystem.path_for(experiment_id)
                session.add(
                    ExperimentRow(
                        id=manifest.experiment_id,
                        kind=manifest.kind,
                        status=manifest.status,
                        commit=manifest.git_commit,
                        seed=int(manifest.seeds[0]) if manifest.seeds else 0,
                        superseded_by=manifest.superseded_by,
                        supersede_reason=manifest.superseded_reason or "",
                        row_count=len(manifest.runs),
                        manifest_json=dumps(manifest.to_dict()),
                        summary_json=_read_json(directory / "summary.json"),
                        metrics_json=_read_json(directory / "metrics.json"),
                    )
                )
                imported += 1
            session.commit()
        return imported

    def experiment_rows(self, experiment_id: str) -> list[dict[str, str]]:
        """The result CSV rows for a run, read from the filesystem.

        Rows stay on disk where the experiment harness writes them. Copying them
        into a second table would create two sources for the same number, and the
        two would eventually disagree.
        """
        path = ExperimentStore(self._experiments_root).path_for(experiment_id) / "results.csv"
        if not path.is_file():
            return []
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))

    def list_experiments(
        self, *, live_only: bool = False, kind: str | None = None
    ) -> list[ExperimentRow]:
        with self.session() as session:
            stmt = select(ExperimentRow)
            if live_only:
                stmt = stmt.where(ExperimentRow.status == "succeeded")
            if kind is not None:
                stmt = stmt.where(ExperimentRow.kind.like(f"%{kind}%"))
            return list(session.execute(stmt.order_by(ExperimentRow.created_at)).scalars())

    def get_experiment(self, experiment_id: str) -> ExperimentRow:
        with self.session() as session:
            row = session.get(ExperimentRow, experiment_id)
            if row is None:
                raise UnknownEntityError(f"unknown experiment {experiment_id!r}")
            return row


def _read_json(path: Path) -> str:
    if not path.is_file():
        return "{}"
    try:
        return dumps(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return "{}"


# --------------------------------------------------------------------- helpers


def _scenario_graph(scenario: Scenario) -> dict[str, Any]:
    """Fidelity copy for reconstruction.

    Scenario already knows how to serialise itself; duplicating that here would
    give the store a second serialisation to keep in step with the first. The
    queryable columns in TaskRow/ResourceRow are still written separately - the
    point of the graph copy is only to rebuild the object exactly.
    """
    return scenario.to_dict()


def _scenario_from_row(row: ScenarioRow) -> Scenario:
    """Rebuild a domain Scenario from the stored fidelity copy."""
    graph = loads(row.graph_json, {})
    if not graph:
        raise ConfigError(f"scenario {row.id!r} has no stored graph")
    return Scenario.from_dict(graph)


def _coerce(enum_value: Any, fallback: str) -> Any:
    """Prefer a real enum member, but never fail the load over one bad row."""
    return enum_value if enum_value is not None else fallback


def _plan_from_row(row: PlanRow) -> Plan:
    payload = loads(row.plan_json, {})
    if not payload:
        raise ConfigError(f"plan {row.id!r} has no stored payload")
    return Plan.from_dict(payload)
