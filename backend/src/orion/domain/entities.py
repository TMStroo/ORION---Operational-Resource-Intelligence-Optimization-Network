
"""Core domain entities: Task, Resource, Scenario and their invariants.

These are the only objects the optimisers, the simulator and the API share.
They are deliberately plain dataclasses with explicit validation rather than
ORM rows or pydantic models, because:

* the solvers need cheap, immutable, in-memory access with no serialisation
  overhead in the inner loop;
* the database layer and the API layer each need their own shape;
* validation errors must be raised at construction with a message naming the
  entity, not at query time.

Every ``__post_init__`` below encodes an invariant that, if violated, would
silently corrupt a plan. That is the whole reason these checks exist: ORION
refuses to optimise a scenario it cannot model correctly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from orion.domain.capacity import (
    CapacityDemand,
    Priority,
    normalize_capabilities,
    parse_priority,
)
from orion.domain.errors import (
    DuplicateEntityError,
    InfeasibleScenarioError,
    UnknownEntityError,
    ValidationError,
)
from orion.domain.ids import DEPOT_ID, format_id, validate_id
from orion.domain.time_model import (
    MAX_HORIZON_MINUTES,
    MINUTES_PER_DAY,
    Interval,
)
from orion.domain.travel import TravelMatrix


# ==========================================================================
# Task
# ==========================================================================
class TaskStatus(str):
    """String constants for task status (not an Enum, so JSON stays stable)."""

    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    LATE = "LATE"
    UNASSIGNED = "UNASSIGNED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Task:
    """A unit of operational work that requires a resource for a duration.

    Time window semantics: the task may start no earlier than ``release_time``
    and should finish by ``deadline``. A start after the deadline is allowed by
    the model (soft constraint) but penalised through
    ``ObjectiveWeights.lateness``; making the deadline hard would render whole
    scenario classes infeasible and hide the interesting planning behaviour.
    """

    id: str
    location: str
    release_time: int
    deadline: int
    duration: int
    priority: Priority = Priority.MEDIUM
    required_capabilities: frozenset[str] = frozenset()
    required_capacity: tuple[CapacityDemand, ...] = ()
    dependencies: tuple[str, ...] = ()
    status: str = TaskStatus.PENDING
    service_minutes_target: int | None = None
    max_lateness: int | None = None

    def __post_init__(self) -> None:
        validate_id(self.id, field="task.id")
        if not self.location:
            raise ValidationError(f"task {self.id}: location must be non-empty")
        if self.duration <= 0:
            raise ValidationError(
                f"task {self.id}: duration {self.duration} must be > 0 "
                "(a zero or negative duration cannot be scheduled)"
            )
        if self.deadline < self.release_time:
            raise ValidationError(
                f"task {self.id}: deadline {self.deadline} precedes release "
                f"{self.release_time}; the task could never be feasible"
            )
        if self.release_time < 0:
            raise ValidationError(f"task {self.id}: release_time {self.release_time} < 0")
        if self.deadline > MAX_HORIZON_MINUTES:
            raise ValidationError(
                f"task {self.id}: deadline {self.deadline} exceeds the "
                f"{MAX_HORIZON_MINUTES}-minute scenario horizon"
            )
        self.priority = parse_priority(self.priority)
        self.required_capabilities = normalize_capabilities(self.required_capabilities)
        self.dependencies = tuple(dict.fromkeys(self.dependencies))
        for dep in self.dependencies:
            if dep == self.id:
                raise ValidationError(f"task {self.id}: task depends on itself")
        if self.service_minutes_target is not None and self.service_minutes_target <= 0:
            raise ValidationError(f"task {self.id}: service_minutes_target must be > 0")
        if self.max_lateness is not None and self.max_lateness < 0:
            raise ValidationError(f"task {self.id}: max_lateness must be >= 0")
        # capacity demands are keyed by dimension; duplicates are a config error
        seen: set[str] = set()
        for demand in self.required_capacity:
            if demand.dimension in seen:
                raise ValidationError(
                    f"task {self.id}: duplicate capacity dimension {demand.dimension!r}"
                )
            seen.add(demand.dimension)

    # -- derived -----------------------------------------------------------
    @property
    def slack(self) -> int:
        """Minutes between the earliest possible finish and the deadline."""
        return self.deadline - self.release_time - self.duration

    @property
    def is_tight(self) -> bool:
        """True when the deadline leaves no room for travel or waiting."""
        return self.slack <= 0

    def lateness_for(self, finish_time: int) -> int:
        """Minutes late, 0 when on time. Negative values are clamped to 0."""
        return max(0, finish_time - self.deadline)

    def has_hard_lateness_limit(self) -> bool:
        return self.max_lateness is not None

    def requires(self, available: frozenset[str]) -> bool:
        return self.required_capabilities <= available

    def capacity_for(self, dimension: str) -> float:
        for demand in self.required_capacity:
            if demand.dimension == dimension:
                return demand.amount
        return 0.0

    @property
    def capacity_dimensions(self) -> tuple[str, ...]:
        return tuple(d.dimension for d in self.required_capacity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "location": self.location,
            "release_time": self.release_time,
            "deadline": self.deadline,
            "duration": self.duration,
            "priority": int(self.priority),
            "required_capabilities": sorted(self.required_capabilities),
            "required_capacity": [
                {"dimension": d.dimension, "amount": d.amount} for d in self.required_capacity
            ],
            "dependencies": list(self.dependencies),
            "status": self.status,
            "service_minutes_target": self.service_minutes_target,
            "max_lateness": self.max_lateness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Task:
        return cls(
            id=str(data["id"]),
            location=str(data["location"]),
            release_time=int(data["release_time"]),
            deadline=int(data["deadline"]),
            duration=int(data["duration"]),
            priority=parse_priority(data.get("priority", Priority.MEDIUM)),
            required_capabilities=frozenset(data.get("required_capabilities", ())),
            required_capacity=tuple(
                CapacityDemand(str(d["dimension"]), float(d["amount"]))
                for d in data.get("required_capacity", ())
            ),
            dependencies=tuple(str(x) for x in data.get("dependencies", ())),
            status=str(data.get("status", TaskStatus.PENDING)),
            service_minutes_target=(
                int(data["service_minutes_target"])
                if data.get("service_minutes_target") is not None
                else None
            ),
            max_lateness=(
                int(data["max_lateness"]) if data.get("max_lateness") is not None else None
            ),
        )


# ==========================================================================
# Resource
# ==========================================================================
class ResourceKind(str):
    """Resource kind constants. Drives which scenario constraints apply."""

    TEAM = "TEAM"
    VEHICLE = "VEHICLE"
    EQUIPMENT = "EQUIPMENT"


class ResourceStatus(str):
    AVAILABLE = "AVAILABLE"
    ON_TASK = "ON_TASK"
    UNAVAILABLE = "UNAVAILABLE"
    MAINTENANCE = "MAINTENANCE"
    FAILED = "FAILED"


@dataclass(slots=True)
class Resource:
    """A resource that can perform tasks: a team, a vehicle or a piece of equipment.

    A vehicle carries a team conceptually, but ORION does **not** couple them.
    Coupling would require a two-level routing problem (which team rides which
    vehicle) and would change the class of solver needed. Instead a scenario
    models a crewed vehicle as a single :class:`Resource` with the union of the
    crew's skills and the vehicle's capabilities. This is stated as a limitation
    in the technical report rather than hidden.
    """

    id: str
    kind: str
    capabilities: frozenset[str] = frozenset()
    home_location: str = DEPOT_ID
    #: Busy periods: maintenance windows, committed jobs, rest.
    unavailable: tuple[Interval, ...] = ()
    #: Calendar bounds during which this resource may work at all.
    shift_start: int = 0
    shift_end: int = MAX_HORIZON_MINUTES
    capacity: Mapping[str, float] = field(default_factory=dict)
    operating_cost_per_hour: float = 0.0
    status: str = ResourceStatus.AVAILABLE
    speed_factor: float = 1.0
    max_work_minutes: int = MAX_HORIZON_MINUTES
    #: Cost multiplier applied while the resource is on a task.
    fixed_dispatch_cost: float = 0.0

    def __post_init__(self) -> None:
        validate_id(self.id, field="resource.id")
        if self.kind not in {ResourceKind.TEAM, ResourceKind.VEHICLE, ResourceKind.EQUIPMENT}:
            raise ValidationError(
                f"resource {self.id}: unknown kind {self.kind!r}; expected one of "
                f"TEAM, VEHICLE, EQUIPMENT"
            )
        if not self.home_location:
            raise ValidationError(f"resource {self.id}: home_location must be non-empty")
        self.capabilities = normalize_capabilities(self.capabilities)
        if self.speed_factor <= 0:
            raise ValidationError(
                f"resource {self.id}: speed_factor {self.speed_factor} must be > 0"
            )
        if self.operating_cost_per_hour < 0:
            raise ValidationError(
                f"resource {self.id}: operating_cost_per_hour must be >= 0"
            )
        if self.fixed_dispatch_cost < 0:
            raise ValidationError(f"resource {self.id}: fixed_dispatch_cost must be >= 0")
        if self.shift_end <= self.shift_start:
            raise ValidationError(
                f"resource {self.id}: shift_end {self.shift_end} must exceed "
                f"shift_start {self.shift_start}"
            )
        if self.max_work_minutes <= 0:
            raise ValidationError(f"resource {self.id}: max_work_minutes must be > 0")
        for interval in self.unavailable:
            if interval.duration > self.shift_end - self.shift_start:
                raise ValidationError(
                    f"resource {self.id}: unavailable block {interval} is longer than its shift"
                )
        self.unavailable = tuple(sorted(self.unavailable))
        self.capacity = {str(k): float(v) for k, v in self.capacity.items()}
        for dim, amount in self.capacity.items():
            if amount < 0:
                raise ValidationError(f"resource {self.id}: capacity {dim}={amount} must be >= 0")
        if self.status not in {
            ResourceStatus.AVAILABLE,
            ResourceStatus.ON_TASK,
            ResourceStatus.UNAVAILABLE,
            ResourceStatus.MAINTENANCE,
            ResourceStatus.FAILED,
        }:
            raise ValidationError(f"resource {self.id}: unknown status {self.status!r}")

    # -- availability ------------------------------------------------------
    @property
    def shift(self) -> Interval:
        return Interval(self.shift_start, self.shift_end)

    @property
    def is_globally_unavailable(self) -> bool:
        """True when the resource cannot work at any point in the horizon."""
        return self.status in {ResourceStatus.UNAVAILABLE, ResourceStatus.FAILED}

    def is_free_at(self, window: Interval) -> bool:
        """True when the resource is on shift and not blocked during *window*."""
        if self.is_globally_unavailable:
            return False
        if not self.shift.contains(window.start) or not self.shift.contains(window.end - 1):
            # a window ending exactly at shift_end is fine; one past it is not
            if not (self.shift_start <= window.start and window.end <= self.shift_end):
                return False
        return all(not window.overlaps(block) for block in self.unavailable)

    def feasible_windows(self, horizon: Interval) -> list[Interval]:
        """Shift minus maintenance blocks, clipped to *horizon*."""
        shift = self.shift.intersection(horizon)
        if shift is None:
            return []
        remaining = [shift]
        for block in self.unavailable:
            nxt: list[Interval] = []
            for piece in remaining:
                if not piece.overlaps(block):
                    nxt.append(piece)
                    continue
                if block.start > piece.start:
                    nxt.append(Interval(piece.start, block.start))
                if block.end < piece.end:
                    nxt.append(Interval(block.end, piece.end))
            remaining = nxt
        return remaining

    def can_serve(self, task: Task) -> bool:
        """Capability + capacity feasibility, ignoring time."""
        if not task.requires(self.capabilities):
            return False
        for demand in task.required_capacity:
            available = float(self.capacity.get(demand.dimension, 0.0))
            if demand.amount > available:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "capabilities": sorted(self.capabilities),
            "home_location": self.home_location,
            "unavailable": [[iv.start, iv.end] for iv in self.unavailable],
            "shift_start": self.shift_start,
            "shift_end": self.shift_end,
            "capacity": dict(self.capacity),
            "operating_cost_per_hour": self.operating_cost_per_hour,
            "status": self.status,
            "speed_factor": self.speed_factor,
            "max_work_minutes": self.max_work_minutes,
            "fixed_dispatch_cost": self.fixed_dispatch_cost,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Resource:
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            capabilities=frozenset(data.get("capabilities", ())),
            home_location=str(data.get("home_location", DEPOT_ID)),
            unavailable=tuple(Interval(int(a), int(b)) for a, b in data.get("unavailable", ())),
            shift_start=int(data.get("shift_start", 0)),
            shift_end=int(data.get("shift_end", MAX_HORIZON_MINUTES)),
            capacity={str(k): float(v) for k, v in (data.get("capacity") or {}).items()},
            operating_cost_per_hour=float(data.get("operating_cost_per_hour", 0.0)),
            status=str(data.get("status", ResourceStatus.AVAILABLE)),
            speed_factor=float(data.get("speed_factor", 1.0)),
            max_work_minutes=int(data.get("max_work_minutes", MAX_HORIZON_MINUTES)),
            fixed_dispatch_cost=float(data.get("fixed_dispatch_cost", 0.0)),
        )

    def with_status(self, status: str) -> Resource:
        return replace(self, status=status)

    def with_unavailable(self, blocks: Sequence[Interval]) -> Resource:
        return replace(self, unavailable=tuple(sorted(set(self.unavailable) | set(blocks))))


# ==========================================================================
# Objective weights
# ==========================================================================
@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Configurable objective coefficients.

    ORION **maximises** a single scalar ``service_score``. The default weights
    are a starting point, not a prescription: a scenario may set any of them, and
    the technical report documents how weight choice changes the plan. ``w_late``
    is expressed per hour of lateness so that a value of 4.0 means "one hour
    late costs as much as a medium-priority task is worth".
    """

    w_completion: float = 10.0
    w_priority: float = 2.0
    w_late: float = 4.0
    w_travel: float = 0.35
    w_cost: float = 0.2
    w_dispatch: float = 1.5
    w_overload: float = 0.8
    w_move_idle: float = 0.15
    w_violation: float = 50.0

    def __post_init__(self) -> None:
        for name in (
            "w_completion",
            "w_priority",
            "w_late",
            "w_travel",
            "w_cost",
            "w_dispatch",
            "w_overload",
            "w_move_idle",
            "w_violation",
        ):
            value = float(getattr(self, name))  # type: ignore[arg-type]
            if value < 0:
                raise ValidationError(f"objective weight {name}={value} must be >= 0")
        if self.w_completion == 0 and self.w_priority == 0:
            raise ValidationError(
                "objective weights w_completion and w_priority are both zero; "
                "the planner would have no reason to complete any task"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ObjectiveWeights:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValidationError(
                f"unknown objective weights {sorted(unknown)}; allowed: {sorted(known)}"
            )
        return cls(**{k: float(v) for k, v in data.items()})

    @property
    def total_positive_weight(self) -> float:
        return self.w_completion + self.w_priority


# ==========================================================================
# Scenario-level constraints
# ==========================================================================
@dataclass(frozen=True, slots=True)
class ScenarioConstraints:
    """Global constraint toggles for a scenario.

    These are switches on constraint *classes*, not hard-coded rules. ORION's
    formulations read them to decide which constraint rows to emit, which is how
    the constraint-pressure experiment tightens a scenario in a controlled way.
    """

    enforce_capability: bool = True
    enforce_capacity: bool = True
    enforce_deadlines: bool = True
    enforce_release_times: bool = True
    enforce_dependencies: bool = True
    enforce_maintenance: bool = True
    enforce_shift: bool = True
    enforce_max_work: bool = True
    enforce_travel: bool = True
    allow_late: bool = True
    allow_unassigned: bool = True
    max_unassigned_fraction: float = 1.0
    require_depot_return: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_unassigned_fraction <= 1.0:
            raise ValidationError(
                f"max_unassigned_fraction {self.max_unassigned_fraction} must be in [0, 1]"
            )
        if not self.allow_unassigned and self.max_unassigned_fraction == 1.0:
            # 0 tolerated + fully fraction allowed is contradictory
            object.__setattr__(self, "max_unassigned_fraction", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ScenarioConstraints:
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValidationError(
                f"unknown scenario constraints {sorted(unknown)}; allowed: {sorted(known)}"
            )
        return cls(**{k: data[k] for k in data})


# ==========================================================================
# Scenario
# ==========================================================================
@dataclass(slots=True)
class Scenario:
    """A complete, self-contained operational problem instance.

    A scenario is immutable in practice: disruptions produce a *new* Scenario
    via :meth:`with_disruption`, which is what makes "preserve the current world
    state, then apply the disruption" (section 12 of the project brief) a
    structural guarantee rather than a convention.
    """

    id: str
    name: str
    tasks: tuple[Task, ...]
    resources: tuple[Resource, ...]
    travel: TravelMatrix
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    constraints: ScenarioConstraints = field(default_factory=ScenarioConstraints)
    horizon: Interval = field(default_factory=lambda: Interval(0, MINUTES_PER_DAY))
    depot: str = DEPOT_ID
    description: str = ""
    tags: tuple[str, ...] = ()
    #: Free-form provenance: seed, generator version, source dataset.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_id(self.id, field="scenario.id")
        if not self.name.strip():
            raise ValidationError(f"scenario {self.id}: name must be non-empty")
        self.tasks = tuple(self.tasks)
        self.resources = tuple(self.resources)
        self.tags = tuple(self.tags)

        task_ids = [t.id for t in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            dupes = sorted({i for i in task_ids if task_ids.count(i) > 1})
            raise DuplicateEntityError(
                f"scenario {self.id}: duplicate task ids {dupes[:5]}"
            )
        resource_ids = [r.id for r in self.resources]
        if len(set(resource_ids)) != len(resource_ids):
            dupes = sorted({i for i in resource_ids if resource_ids.count(i) > 1})
            raise DuplicateEntityError(
                f"scenario {self.id}: duplicate resource ids {dupes[:5]}"
            )

        if self.horizon.duration <= 0:
            raise ValidationError(f"scenario {self.id}: horizon duration must be > 0")

        # every referenced location must exist in the travel matrix
        known = set(self.travel.locations)
        for task in self.tasks:
            if not self.travel.has(task.location):
                raise ValidationError(
                    f"scenario {self.id}: task {task.id} references unknown location "
                    f"{task.location!r}"
                )
        for resource in self.resources:
            if not self.travel.has(resource.home_location):
                raise ValidationError(
                    f"scenario {self.id}: resource {resource.id} references unknown "
                    f"location {resource.home_location!r}"
                )
        if self.constraints.require_depot_return and not self.travel.has(self.depot):
            raise ValidationError(
                f"scenario {self.id}: depot {self.depot!r} is not in the travel matrix"
            )

        # dependencies must resolve and must be acyclic
        id_set = set(task_ids)
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in id_set:
                    raise UnknownEntityError(
                        f"scenario {self.id}: task {task.id} depends on unknown task {dep!r}"
                    )
        self._check_acyclic()

        # releases/deadlines must lie inside the horizon
        for task in self.tasks:
            if task.deadline > self.horizon.end:
                raise ValidationError(
                    f"scenario {self.id}: task {task.id} deadline {task.deadline} lies "
                    f"beyond the horizon end {self.horizon.end}"
                )

    def _check_acyclic(self) -> None:
        """Kahn's algorithm; raises on a dependency cycle."""
        indegree = {t.id: len(t.dependencies) for t in self.tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in self.tasks}
        for task in self.tasks:
            for dep in task.dependencies:
                dependents[dep].append(task.id)
        queue = [tid for tid, deg in indegree.items() if deg == 0]
        seen = 0
        while queue:
            current = queue.pop()
            seen += 1
            for child in dependents[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self.tasks):
            remaining = sorted(tid for tid, deg in indegree.items() if deg > 0)
            raise ValidationError(
                f"scenario {self.id}: dependency cycle among tasks {remaining[:5]}"
            )

    # -- lookups -----------------------------------------------------------
    @property
    def task_index(self) -> dict[str, int]:
        return {t.id: i for i, t in enumerate(self.tasks)}

    @property
    def resource_index(self) -> dict[str, int]:
        return {r.id: i for i, r in enumerate(self.resources)}

    def task(self, task_id: str) -> Task:
        for candidate in self.tasks:
            if candidate.id == task_id:
                return candidate
        raise UnknownEntityError(f"scenario {self.id}: unknown task {task_id!r}")

    def resource(self, resource_id: str) -> Resource:
        for candidate in self.resources:
            if candidate.id == resource_id:
                return candidate
        raise UnknownEntityError(f"scenario {self.id}: unknown resource {resource_id!r}")

    def active_resources(self) -> tuple[Resource, ...]:
        """Resources that can work at some point in the horizon."""
        return tuple(
            r for r in self.resources if not r.is_globally_unavailable and r.shift.end > self.horizon.start
        )

    def eligible_pairs(self) -> list[tuple[str, str]]:
        """(task_id, resource_id) pairs satisfying capability and capacity.

        This is the candidate set every solver works over. Precomputing it means
        the CP-SAT and MILP models contain only variables that can legally be 1,
        which is the single biggest model-size lever ORION has.
        """
        pairs: list[tuple[str, str]] = []
        for task in self.tasks:
            for resource in self.resources:
                if not self.constraints.enforce_capability and not self.constraints.enforce_capacity:
                    pairs.append((task.id, resource.id))
                    continue
                if resource.can_serve(task):
                    pairs.append((task.id, resource.id))
        return pairs

    def static_infeasibility(self) -> str | None:
        """Detect obviously-infeasible scenarios before invoking any solver.

        Returns a human-readable reason, or ``None`` when no *structural*
        infeasibility is detectable. This is deliberately conservative: ORION
        only claims infeasibility it can prove cheaply, and otherwise lets the
        solver decide. A wrong "infeasible" here would be a lie to the user.
        """
        if not self.tasks:
            return "scenario contains no tasks"
        if not self.active_resources():
            return "scenario contains no available resources"
        if not self.constraints.allow_unassigned and self.constraints.max_unassigned_fraction == 0.0:
            pairs = set(self.eligible_pairs())
            for task in self.tasks:
                if not any(t == task.id for t, _ in pairs):
                    return (
                        f"task {task.id} has no capable resource "
                        f"(needs {sorted(task.required_capabilities) or 'no capability'})"
                    )
        for task in self.tasks:
            earliest = task.release_time
            for dep in task.dependencies:
                parent = self.task(dep)
                earliest = max(earliest, parent.deadline + parent.duration)
            if earliest + task.duration > self.horizon.end:
                return (
                    f"task {task.id} cannot finish inside the horizon: earliest start "
                    f"{earliest} + duration {task.duration} > {self.horizon.end}"
                )
        return None

    def demand_summary(self) -> dict[str, int]:
        total = sum(t.duration for t in self.tasks)
        by_priority: dict[str, int] = {}
        for task in self.tasks:
            key = task.priority.label
            by_priority[key] = by_priority.get(key, 0) + 1
        return {
            "tasks": len(self.tasks),
            "resources": len(self.resources),
            "total_demand_minutes": total,
            "available_minutes": sum(
                r.shift.duration for r in self.active_resources()
            ),
            **{f"priority_{k.lower()}": v for k, v in sorted(by_priority.items())},
        }

    # -- immutable updates -------------------------------------------------
    def with_task(self, task: Task) -> Scenario:
        tasks = tuple(t if t.id != task.id else task for t in self.tasks)
        if all(t.id != task.id for t in self.tasks):
            tasks = self.tasks + (task,)
        return replace(self, tasks=tasks)

    def with_resource(self, resource: Resource) -> Scenario:
        resources = tuple(r if r.id != resource.id else resource for r in self.resources)
        if all(r.id != resource.id for r in self.resources):
            resources = self.resources + (resource,)
        return replace(self, resources=resources)

    def with_travel(self, travel: TravelMatrix) -> Scenario:
        return replace(self, travel=travel)

    def with_weights(self, weights: ObjectiveWeights) -> Scenario:
        return replace(self, objective_weights=weights)

    def with_constraints(self, constraints: ScenarioConstraints) -> Scenario:
        return replace(self, constraints=constraints)

    def with_horizon(self, horizon: Interval) -> Scenario:
        return replace(self, horizon=horizon)

    def with_metadata(self, **updates: Any) -> Scenario:
        merged = dict(self.metadata)
        merged.update(updates)
        return replace(self, metadata=merged)

    def restart(self, suffix: str = "-r") -> Scenario:
        """Return a copy with all tasks reset to PENDING (used after disruption)."""
        tasks = tuple(replace(t, status=TaskStatus.PENDING) for t in self.tasks)
        rid = f"{self.id}{suffix}"
        if len(rid) > 64:
            rid = rid[:64]
        return replace(self, id=rid, tasks=tasks, resources=self.resources)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "depot": self.depot,
            "horizon": [self.horizon.start, self.horizon.end],
            "tasks": [t.to_dict() for t in self.tasks],
            "resources": [r.to_dict() for r in self.resources],
            "travel": self.travel.to_dict(),
            "objective_weights": self.objective_weights.to_dict(),
            "constraints": self.constraints.to_dict(),
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            tasks=tuple(Task.from_dict(t) for t in data.get("tasks", ())),
            resources=tuple(Resource.from_dict(r) for r in data.get("resources", ())),
            travel=TravelMatrix.from_dict(data["travel"]),
            objective_weights=ObjectiveWeights.from_dict(data.get("objective_weights")),
            constraints=ScenarioConstraints.from_dict(data.get("constraints")),
            horizon=Interval(int(data["horizon"][0]), int(data["horizon"][1])),
            depot=str(data.get("depot", DEPOT_ID)),
            description=str(data.get("description", "")),
            tags=tuple(str(t) for t in data.get("tags", ())),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> Scenario:
        return cls.from_dict(json.loads(text))


def make_task(
    index: int,
    *,
    location: str,
    release_time: int,
    deadline: int,
    duration: int,
    priority: Priority = Priority.MEDIUM,
    required_capabilities: Iterable[str] = (),
    required_capacity: Iterable[CapacityDemand] = (),
    dependencies: Iterable[str] = (),
    prefix: str = "T",
) -> Task:
    """Convenience constructor used by the scenario generator and tests."""
    return Task(
        id=format_id(prefix, index),
        location=location,
        release_time=release_time,
        deadline=deadline,
        duration=duration,
        priority=priority,
        required_capabilities=frozenset(required_capabilities),
        required_capacity=tuple(required_capacity),
        dependencies=tuple(dependencies),
    )


__all__ = [
    "Task",
    "TaskStatus",
    "Resource",
    "ResourceKind",
    "ResourceStatus",
    "Scenario",
    "ScenarioConstraints",
    "ObjectiveWeights",
    "make_task",
]
