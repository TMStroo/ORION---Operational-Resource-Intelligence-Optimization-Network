"""Relational persistence for ORION.

Design note: rows that a reviewer needs to query - tasks, resources, assignments,
disruptions, simulation events - are real columns. They are not folded into an
opaque JSON blob, because "how many tasks were served by vehicle V3 after the
disruption" is a question this schema is supposed to answer.

The full domain graph is not normalised either. A scenario's task graph, travel
matrix and objective weights are stored as JSON *alongside* the queryable
columns, not instead of them. The reason is asymmetry: the columns that matter
for analysis are the ones with stable, flat meaning, and everything else is a
fidelity copy whose only job is to reconstruct the exact domain object that was
optimised. A fully normalised schema would add joins to every load and would
still not let you reproduce a run bit-for-bit unless the graph round-tripped
exactly.

Deliberately absent: a generic blob table. If a row cannot be queried, it does
not belong here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every ORION table."""


class ScenarioRow(Base):
    """A stored optimisation scenario."""

    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    # Queryable scalars, denormalised from the graph for filtering and sorting.
    task_count: Mapped[int] = mapped_column(Integer, default=0, index=True)
    resource_count: Mapped[int] = mapped_column(Integer, default=0, index=True)
    horizon_end: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(64), default="generated")
    tags: Mapped[str] = mapped_column(String(512), default="")
    # Fidelity copies used to rebuild the exact domain object.
    graph_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    tasks: Mapped[list["TaskRow"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="TaskRow.id"
    )
    resources: Mapped[list["ResourceRow"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="ResourceRow.id"
    )
    plans: Mapped[list["PlanRow"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="PlanRow.created_at"
    )

    @property
    def tag_list(self) -> list[str]:
        return [t for t in self.tags.split(",") if t]


class TaskRow(Base):
    """One unit of work. Kept relational so task-level analysis is a query."""

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_scenario_priority", "scenario_id", "priority"),
        Index("ix_tasks_scenario_deadline", "scenario_id", "deadline"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[str] = mapped_column(String(128), default="")
    release_time: Mapped[int] = mapped_column(Integer, default=0)
    deadline: Mapped[int] = mapped_column(Integer, default=0)
    duration: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), default="")
    required_capabilities: Mapped[str] = mapped_column(String(512), default="")
    dependencies: Mapped[str] = mapped_column(Text, default="")
    extra_json: Mapped[str] = mapped_column(Text, default="{}")

    scenario: Mapped[ScenarioRow] = relationship(back_populates="tasks")

    __table_args_unique__ = UniqueConstraint("scenario_id", "task_id")


class ResourceRow(Base):
    """One resource: a team, a vehicle, a technician."""

    __tablename__ = "resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    resource_id: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(64), default="", index=True)
    home_location: Mapped[str] = mapped_column(String(128), default="")
    shift_start: Mapped[int] = mapped_column(Integer, default=0)
    shift_end: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="")
    capabilities: Mapped[str] = mapped_column(String(512), default="")
    max_work_minutes: Mapped[int] = mapped_column(Integer, default=0)
    operating_cost_per_hour: Mapped[float] = mapped_column(Float, default=0.0)
    capacity_json: Mapped[str] = mapped_column(Text, default="{}")
    unavailable_json: Mapped[str] = mapped_column(Text, default="[]")
    extra_json: Mapped[str] = mapped_column(Text, default="{}")

    scenario: Mapped[ScenarioRow] = relationship(back_populates="resources")

    __table_args_unique__ = UniqueConstraint("scenario_id", "resource_id")


class PlanRow(Base):
    """An optimised plan plus the run metadata needed to audit it."""

    __tablename__ = "plans"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(48), default="", index=True)
    solver: Mapped[str] = mapped_column(String(48), default="", index=True)
    objective: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    completion: Mapped[float] = mapped_column(Float, default=0.0)
    late_tasks: Mapped[int] = mapped_column(Integer, default=0)
    tasks_assigned: Mapped[int] = mapped_column(Integer, default=0)
    #: How many tasks the scenario offered. Stored because "12 of 20 assigned" is
    #: the question a plan is usually asked, and it cannot be derived from a
    #: plan row alone once the scenario's task set has moved on.
    tasks_total: Mapped[int] = mapped_column(Integer, default=0)
    runtime_s: Mapped[float] = mapped_column(Float, default=0.0)
    infeasible: Mapped[bool] = mapped_column(Boolean, default=False)
    # The full Plan, so a stored plan reconstructs exactly rather than approximately.
    plan_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scenario: Mapped[ScenarioRow] = relationship(back_populates="plans")
    assignments: Mapped[list["AssignmentRow"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="AssignmentRow.sequence_index"
    )
    solver_runs: Mapped[list["SolverRunRow"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="SolverRunRow.id"
    )
    comparisons: Mapped[list["ComparisonRow"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="ComparisonRow.id"
    )


class AssignmentRow(Base):
    """One task-to-resource assignment inside a plan."""

    __tablename__ = "assignments"
    __table_args__ = (Index("ix_assignments_plan_resource", "plan_id", "resource_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("plans.id", ondelete="CASCADE"), index=True
    )
    task_id: Mapped[str] = mapped_column(String(128), index=True)
    resource_id: Mapped[str] = mapped_column(String(128), index=True)
    sequence_index: Mapped[int] = mapped_column(Integer, default=0)
    start: Mapped[int] = mapped_column(Integer, default=0)
    end: Mapped[int] = mapped_column(Integer, default=0)
    location: Mapped[str] = mapped_column(String(128), default="")
    travel_before: Mapped[int] = mapped_column(Integer, default=0)
    travel_distance_km: Mapped[float] = mapped_column(Float, default=0.0)
    priority: Mapped[str] = mapped_column(String(32), default="")
    lateness: Mapped[int] = mapped_column(Integer, default=0)

    plan: Mapped[PlanRow] = relationship(back_populates="assignments")


class SolverRunRow(Base):
    """One solver invocation, including declines.

    Declines are stored deliberately. A solver that reports NOT_APPLICABLE is
    evidence about the problem, and dropping those rows is how a benchmark ends
    up looking better than it is.
    """

    __tablename__ = "solver_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("plans.id", ondelete="CASCADE"), index=True
    )
    solver: Mapped[str] = mapped_column(String(48), default="", index=True)
    status: Mapped[str] = mapped_column(String(48), default="", index=True)
    objective: Mapped[float] = mapped_column(Float, default=0.0)
    internal_objective: Mapped[float] = mapped_column(Float, default=0.0)
    runtime_s: Mapped[float] = mapped_column(Float, default=0.0)
    tasks_assigned: Mapped[int] = mapped_column(Integer, default=0)
    feasible: Mapped[bool] = mapped_column(Boolean, default=False)
    optimality_gap: Mapped[float | None] = mapped_column(Float, default=None)
    note: Mapped[str] = mapped_column(Text, default="")
    detail_json: Mapped[str] = mapped_column(Text, default="{}")

    plan: Mapped[PlanRow] = relationship(back_populates="solver_runs")


class DisruptionRow(Base):
    """A disruption that was applied to a scenario.

    `applied_once` exists so that re-applying a disruption is detectable. DES
    replanning can be re-run many times against the same event; without this flag
    a double application is invisible.
    """

    __tablename__ = "disruptions"
    __table_args__ = (Index("ix_disruptions_scenario_time", "scenario_id", "timestamp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), index=True
    )
    disruption_id: Mapped[str] = mapped_column(String(128), index=True)
    type: Mapped[str] = mapped_column(String(64), default="", index=True)
    severity: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str | None] = mapped_column(String(128), default=None)
    timestamp: Mapped[int] = mapped_column(Integer, default=0)
    magnitude: Mapped[float] = mapped_column(Float, default=0.0)
    duration: Mapped[int | None] = mapped_column(Integer, default=None)
    description: Mapped[str] = mapped_column(Text, default="")
    applied_once: Mapped[bool] = mapped_column(Boolean, default=True)
    affected_tasks: Mapped[int] = mapped_column(Integer, default=0)
    affected_resources: Mapped[int] = mapped_column(Integer, default=0)
    impact_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args_unique__ = UniqueConstraint("scenario_id", "disruption_id")


class SimulationEventRow(Base):
    """A single DES event.

    `event_key` is unique per simulation run: the engine can legitimately emit
    the same (time, kind, subject) triple twice, and those are two distinct
    events, so the key includes the run and an ordinal.
    """

    __tablename__ = "simulation_events"
    __table_args__ = (
        Index("ix_simevents_run_time", "run_id", "time"),
        UniqueConstraint("run_id", "ordinal"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    time: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(64), default="", index=True)
    subject: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")

    __table_args_unique__ = UniqueConstraint("run_id", "ordinal")


class SimulationRunRow(Base):
    """A whole simulation run and its headline numbers."""

    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(128), index=True)
    plan_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    start_time: Mapped[int] = mapped_column(Integer, default=0)
    end_time: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    late_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    replans_performed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(48), default="")
    runtime_s: Mapped[float] = mapped_column(Float, default=0.0)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ComparisonRow(Base):
    """A plan-vs-plan comparison (repair vs full, what-if vs baseline)."""

    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("plans.id", ondelete="CASCADE"), index=True
    )
    baseline_plan_id: Mapped[str] = mapped_column(String(128), default="", index=True)
    candidate_plan_id: Mapped[str] = mapped_column(String(128), default="")
    kind: Mapped[str] = mapped_column(String(64), default="", index=True)
    churn_changed: Mapped[int] = mapped_column(Integer, default=0)
    churn_total: Mapped[int] = mapped_column(Integer, default=0)
    churn_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    recovery_pct: Mapped[float] = mapped_column(Float, default=0.0)
    deltas_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    plan: Mapped[PlanRow] = relationship(back_populates="comparisons")


class ExperimentRow(Base):
    """A benchmark or experiment run.

    Experiment ids are unique and never reused. The filesystem store enforces
    this; the relational store has to enforce it too, otherwise importing a
    store into the database would silently weaken the immutability guarantee.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), default="", index=True)
    status: Mapped[str] = mapped_column(String(32), default="", index=True)
    commit: Mapped[str] = mapped_column(String(64), default="")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    superseded_by: Mapped[str | None] = mapped_column(String(128), default=None)
    supersede_reason: Mapped[str] = mapped_column(Text, default="")
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_json: Mapped[str] = mapped_column(Text, default="{}")
    summary_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


@dataclass(frozen=True)
class StoreStats:
    """Row counts, for the health endpoint and the audit."""

    scenarios: int
    tasks: int
    resources: int
    plans: int
    assignments: int
    solver_runs: int
    disruptions: int
    simulation_events: int
    comparisons: int
    experiments: int

    def as_dict(self) -> dict[str, int]:
        return {
            "scenarios": self.scenarios,
            "tasks": self.tasks,
            "resources": self.resources,
            "plans": self.plans,
            "assignments": self.assignments,
            "solver_runs": self.solver_runs,
            "disruptions": self.disruptions,
            "simulation_events": self.simulation_events,
            "comparisons": self.comparisons,
            "experiments": self.experiments,
        }


def dumps(value: Any) -> str:
    """Serialise to JSON deterministically.

    sort_keys is on so that a row written twice with equal content is
    byte-identical, which is what makes the reproducibility check meaningful.
    """
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {} if default is None else default


def split_values(raw: str) -> list[str]:
    """Decode a comma-joined string column back into a list."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def join_values(values: Mapping[str, Any] | tuple[str, ...] | list[str] | None) -> str:
    """Encode an iterable of strings into a comma-joined column."""
    if not values:
        return ""
    if isinstance(values, Mapping):
        values = values.keys()
    return ",".join(str(v) for v in values)
