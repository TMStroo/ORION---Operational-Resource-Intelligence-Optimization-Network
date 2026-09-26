
"""Reproducible scenario generator.

Every scenario ORION generates is a pure function of its
:class:`ScenarioConfig` and its ``seed``. Re-running the generator with the same
config and seed produces a byte-identical scenario, which is what makes the
scalability, constraint-pressure and disruption experiments comparable.

Design
------

The generator is parameterised along the axes that actually change algorithmic
behaviour, not along cosmetic ones:

``num_tasks`` / ``num_teams`` / ``num_vehicles``
    problem size and resource mix
``geographic_density``
    coordinate spread; low density means everything is close, high density
    means travel dominates
``deadline_tightness``
    scales the slack between a task's earliest possible finish and its deadline;
    this is the primary driver of how often lateness becomes unavoidable
``resource_scarcity``
    fraction of the full capability set each resource actually has; controls how
    often the candidate pair set becomes the binding constraint
``dependency_rate``
    probability a task has a prerequisite, which makes the problem a partial
    order rather than a set of independent jobs
``priority_distribution``
    weights over LOW..CRITICAL
``shift_pattern``
    rostered hours per resource, including overlap and gaps
``maintenance_count``
    number of scheduled downtime blocks

Difficulty levels
-----------------

``small`` / ``medium`` / ``large`` / ``stress`` are presets. They are *not*
just bigger: ``stress`` also tightens deadlines and raises scarcity, so it
changes the regime of the problem rather than only its size. That matters
because a scalability curve over four points that all sit in the same easy
regime proves very little.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from orion.domain.capacity import CapacityDemand, Priority
from orion.domain.entities import (
    ObjectiveWeights,
    Resource,
    ResourceKind,
    ResourceStatus,
    Scenario,
    ScenarioConstraints,
    Task,
)
from orion.domain.errors import ValidationError
from orion.domain.ids import DEPOT_ID, format_id
from orion.domain.time_model import (
    MINUTES_PER_DAY,
    Interval,
    hhmm,
)
from orion.domain.travel import TravelMatrix

CAPABILITY_POOL: tuple[str, ...] = (
    "electrical",
    "plumbing",
    "hvac",
    "structural",
    "hazmat",
    "crane_heavy",
    "refrigerated",
    "network",
    "heavy_lift",
    "water_treatment",
)

CAPACITY_POOL: tuple[str, ...] = ("kg", "volume_l", "seats")


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Complete, serialisable generator configuration.

    The whole config is stored in the scenario metadata and in the experiment
    manifest, so any generated instance can be reproduced exactly.
    """

    name: str
    num_tasks: int = 25
    num_teams: int = 6
    num_vehicles: int = 4
    num_locations: int = 10
    geographic_density: float = 1.0
    deadline_tightness: float = 0.5
    resource_scarcity: float = 0.6
    dependency_rate: float = 0.0
    priority_weights: tuple[float, ...] = (0.30, 0.30, 0.25, 0.15)
    shift_pattern: str = "day"
    maintenance_count: int = 0
    duration_choices: tuple[int, ...] = (20, 30, 45, 60, 90)
    release_window: int = 180
    horizon_minutes: int = 12 * 60
    max_work_fraction: float = 0.85
    require_depot_return: bool = False
    travel_speed_kmh: float = 45.0
    congestion_factor: float = 1.0
    weight_overrides: Mapping[str, float] = field(default_factory=dict)
    constraint_overrides: Mapping[str, bool] = field(default_factory=dict)
    seed: int = 20260101
    tags: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if self.num_tasks <= 0:
            raise ValidationError(f"num_tasks must be > 0, got {self.num_tasks}")
        if self.num_teams + self.num_vehicles <= 0:
            raise ValidationError("a scenario needs at least one team or vehicle")
        if self.num_locations < 2:
            raise ValidationError(
                f"num_locations must be >= 2 (depot plus work sites), got {self.num_locations}"
            )
        for name, value in (
            ("geographic_density", self.geographic_density),
            ("deadline_tightness", self.deadline_tightness),
            ("resource_scarcity", self.resource_scarcity),
            ("max_work_fraction", self.max_work_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValidationError(f"{name} must be in (0, 1], got {value}")
        if not 0.0 <= self.dependency_rate <= 1.0:
            raise ValidationError(
                f"dependency_rate must be in [0, 1], got {self.dependency_rate}"
            )
        if len(self.priority_weights) != 4:
            raise ValidationError(
                f"priority_weights must have 4 entries (LOW..CRITICAL), got "
                f"{len(self.priority_weights)}"
            )
        if sum(self.priority_weights) <= 0:
            raise ValidationError("priority_weights must sum to a positive value")
        if self.shift_pattern not in {"day", "extended", "night", "split"}:
            raise ValidationError(
                f"unknown shift_pattern {self.shift_pattern!r}; expected day, "
                "extended, night or split"
            )
        if self.maintenance_count < 0:
            raise ValidationError("maintenance_count must be >= 0")
        if not self.duration_choices or any(d <= 0 for d in self.duration_choices):
            raise ValidationError("duration_choices must be non-empty and positive")
        if self.horizon_minutes <= 0 or self.horizon_minutes > MINUTES_PER_DAY:
            raise ValidationError(
                f"horizon_minutes must be in (0, {MINUTES_PER_DAY}], got "
                f"{self.horizon_minutes}"
            )
        if self.release_window >= self.horizon_minutes:
            raise ValidationError(
                "release_window must be smaller than horizon_minutes, otherwise "
                "every task is released at the very end"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority_weights"] = list(self.priority_weights)
        data["duration_choices"] = list(self.duration_choices)
        data["weight_overrides"] = dict(self.weight_overrides)
        data["constraint_overrides"] = dict(self.constraint_overrides)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioConfig:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValidationError(
                f"unknown scenario config keys {sorted(unknown)}; allowed: {sorted(known)}"
            )
        payload = dict(data)
        if "priority_weights" in payload:
            payload["priority_weights"] = tuple(float(x) for x in payload["priority_weights"])
        if "duration_choices" in payload:
            payload["duration_choices"] = tuple(int(x) for x in payload["duration_choices"])
        if "tags" in payload:
            payload["tags"] = tuple(str(x) for x in payload["tags"])
        if "weight_overrides" in payload:
            payload["weight_overrides"] = dict(payload["weight_overrides"])
        if "constraint_overrides" in payload:
            payload["constraint_overrides"] = dict(payload["constraint_overrides"])
        return cls(**payload)  # type: ignore[arg-type]

    def with_seed(self, seed: int) -> ScenarioConfig:
        return replace(self, seed=seed)

    def scaled_tasks(self, num_tasks: int) -> ScenarioConfig:
        return replace(self, num_tasks=num_tasks)


#: Difficulty presets. Each changes size *and* regime, deliberately.
DIFFICULTY_PRESETS: Mapping[str, Mapping[str, Any]] = {
    "small": dict(
        num_tasks=10,
        num_teams=3,
        num_vehicles=2,
        num_locations=6,
        deadline_tightness=0.6,
        resource_scarcity=0.7,
    ),
    "medium": dict(
        num_tasks=25,
        num_teams=5,
        num_vehicles=3,
        num_locations=8,
        deadline_tightness=0.5,
        resource_scarcity=0.6,
    ),
    "large": dict(
        num_tasks=60,
        num_teams=9,
        num_vehicles=5,
        num_locations=14,
        deadline_tightness=0.45,
        resource_scarcity=0.55,
    ),
    "stress": dict(
        num_tasks=120,
        num_teams=10,
        num_vehicles=6,
        num_locations=20,
        deadline_tightness=0.8,
        resource_scarcity=0.35,
        dependency_rate=0.15,
    ),
}


def difficulty_config(
    level: str, *, seed: int = 20260101, **overrides: Any
) -> ScenarioConfig:
    """Build a config from a named difficulty preset."""
    if level not in DIFFICULTY_PRESETS:
        raise ValidationError(
            f"unknown difficulty {level!r}; expected one of {sorted(DIFFICULTY_PRESETS)}"
        )
    params = dict(DIFFICULTY_PRESETS[level])
    params.update(overrides)
    params.setdefault("name", f"orion-{level}")
    params["seed"] = seed
    params.setdefault("tags", (level, "generated"))
    params.setdefault(
        "description",
        f"ORION generated {level} scenario "
        f"({params.get('num_tasks')} tasks, seed {seed})",
    )
    return ScenarioConfig(**params)


SHIFTS: Mapping[str, tuple[tuple[int, int], ...]] = {
    # (start, end) in minutes from midnight
    "day": ((7 * 60, 19 * 60),),
    "extended": ((6 * 60, 22 * 60),),
    "night": ((22 * 60, 30 * 60),),
    "split": ((6 * 60, 14 * 60), (15 * 60, 23 * 60)),
}


def _weighted_priority(rng: random.Random, weights: Sequence[float]) -> Priority:
    values = list(Priority)
    return rng.choices(values, weights=list(weights), k=1)[0]


def generate(config: ScenarioConfig) -> Scenario:
    """Generate a scenario deterministically from *config*.

    The RNG is a single ``random.Random(config.seed)`` consumed in a fixed order,
    so the output is a pure function of the config. Nothing reads the clock or
    the process environment.
    """
    rng = random.Random(config.seed)

    # ---- locations ----
    span = 100.0 / max(0.05, config.geographic_density)
    coordinates: dict[str, tuple[float, float]] = {
        DEPOT_ID: (span / 2.0, span / 2.0)
    }
    for i in range(config.num_locations):
        coordinates[format_id("LOC", i + 1, width=2)] = (
            round(rng.uniform(0.0, span), 3),
            round(rng.uniform(0.0, span), 3),
        )
    work_sites = [loc for loc in coordinates if loc != DEPOT_ID]
    travel = TravelMatrix.from_coordinates(
        coordinates,
        default_speed_kmh=config.travel_speed_kmh,
        congestion_factor=config.congestion_factor,
    )

    # ---- resources ----
    scarcity = config.resource_scarcity
    min_caps = max(1, int(round(len(CAPABILITY_POOL) * scarcity)))
    resources: list[Resource] = []
    total_resources = config.num_teams + config.num_vehicles
    for i in range(total_resources):
        is_vehicle = i >= config.num_teams
        index = i + 1
        caps = set(rng.sample(CAPABILITY_POOL, min_caps))
        if is_vehicle:
            # vehicles add transport capability, and are faster
            caps.update({"vehicle", "transport"})
        capacity: dict[str, float] = {}
        if is_vehicle:
            for dim in CAPACITY_POOL:
                capacity[dim] = float(rng.choice([200, 500, 1000, 2000]))
        else:
            capacity["seats"] = float(rng.choice([2, 3, 4]))

        shift_start, shift_end = rng.choice(SHIFTS[config.shift_pattern])
        # Stagger starts so resources are not all available at the same instant,
        # and clamp into the horizon. A "night" roster starting after the horizon
        # would make the whole scenario unservable, so the shift is forced to fit.
        shift_start = max(0, min(shift_start + rng.randrange(-30, 31), config.horizon_minutes - 60))
        shift_end = min(
            config.horizon_minutes,
            max(shift_end + rng.randrange(-30, 31), shift_start + 120),
        )

        resources.append(
            Resource(
                id=format_id("V" if is_vehicle else "T", index, width=3),
                kind=ResourceKind.VEHICLE if is_vehicle else ResourceKind.TEAM,
                capabilities=frozenset(caps),
                home_location=DEPOT_ID,
                shift_start=shift_start,
                shift_end=shift_end,
                capacity=capacity,
                operating_cost_per_hour=round(rng.uniform(15.0, 70.0), 2),
                speed_factor=round(rng.uniform(1.2, 2.0) if is_vehicle else rng.uniform(0.6, 1.1), 3),
                max_work_minutes=int((shift_end - shift_start) * config.max_work_fraction),
                fixed_dispatch_cost=round(rng.uniform(3.0, 18.0), 2),
            )
        )

    # ---- maintenance windows ----
    maintenance_blocks: dict[str, list[Interval]] = {r.id: [] for r in resources}
    for _ in range(config.maintenance_count):
        resource = rng.choice(resources)
        start = rng.randrange(resource.shift_start, max(resource.shift_start + 1, resource.shift_end - 60))
        duration = rng.choice([30, 60, 90, 120])
        end = min(resource.shift_end, start + duration)
        if end > start:
            maintenance_blocks[resource.id].append(Interval(start, end))
    for resource in resources:
        if maintenance_blocks[resource.id]:
            resource.unavailable = tuple(sorted(maintenance_blocks[resource.id]))

    # ---- tasks ----
    tasks: list[Task] = []
    weights = list(config.priority_weights)
    tightness = config.deadline_tightness
    # The release window has to sit inside the rostered hours, otherwise tasks
    # are released while every crew is off shift and the scenario is trivially
    # unservable rather than interesting. earliest_shift is when the first crew
    # comes on; releases are drawn from there.
    earliest_shift = min(r.shift_start for r in resources)
    # a deadline beyond the horizon is invalid, so the rostered end is clamped
    latest_shift = min(max(r.shift_end for r in resources), config.horizon_minutes)
    release_low = max(0, earliest_shift)
    release_high = max(release_low + 1, min(latest_shift - 30, config.horizon_minutes - 60))

    for i in range(config.num_tasks):
        location = rng.choice(work_sites)
        duration = rng.choice(list(config.duration_choices))
        # The release is drawn so that the task's *earliest possible finish*
        # (release + duration) always lands inside [release_low, latest_shift].
        # That single invariant - asserted below - is what guarantees the
        # generated instance is servable in principle, instead of leaving a
        # whole class of tasks that no plan could ever complete.
        release = rng.randrange(release_low, max(release_low + 1, release_high))
        # Fit the task inside the roster: both the release and the duration are
        # adjusted so that release + duration <= latest_shift. Without this a
        # "stress" instance can contain tasks that no plan could ever finish,
        # which tests infeasibility handling rather than solver behaviour.
        if release + duration > latest_shift:
            duration = max(1, latest_shift - release)
        if duration > config.duration_choices[-1]:
            duration = config.duration_choices[-1]
        if release + duration > latest_shift:
            release = max(release_low, latest_shift - duration)
        # nominal travel allowance from the fastest resource, used to size slack
        fastest = max((r.speed_factor for r in resources), default=1.0)
        nominal = travel.travel_from_speed(DEPOT_ID, location, fastest)
        # slack shrinks as tightness rises: 0.0 -> generous, 1.0 -> tight
        slack = int(round(nominal * (3.0 * (1.0 - tightness) + 0.15)))
        # deadline is clamped into [earliest_finish, latest_shift] and never
        # beyond the horizon, which is what Scenario.__post_init__ requires
        earliest_finish = release + duration
        deadline = min(release + duration + max(0, slack), latest_shift)
        deadline = max(deadline, earliest_finish)
        assert release + duration <= deadline <= config.horizon_minutes

        required = set(rng.sample(CAPABILITY_POOL, rng.choice([1, 1, 2])))
        required_capacity: list[CapacityDemand] = []
        if rng.random() < 0.3:
            dim = rng.choice([d for d in CAPACITY_POOL if d != "seats"])
            required_capacity.append(
                CapacityDemand(dim, float(rng.choice([50, 100, 250, 500])))
            )

        tasks.append(
            Task(
                id=format_id("T", i + 1),
                location=location,
                release_time=release,
                deadline=deadline,
                duration=duration,
                priority=_weighted_priority(rng, weights),
                required_capabilities=frozenset(required),
                required_capacity=tuple(required_capacity),
            )
        )

    # ---- dependencies (DAG, so only earlier tasks are referenced) ----
    # A precedence edge pushes the child's earliest start to the parent's
    # earliest *finish*. If that would push the child past its own deadline, no
    # plan can satisfy the edge, so the edge is only created when the child's
    # slack genuinely absorbs it. Skipping it in that case is the honest choice:
    # generating a contradictory edge would make the instance infeasible for a
    # reason that has nothing to do with the solvers.
    if config.dependency_rate > 0:
        for i in range(config.num_tasks):
            if i < 2 or rng.random() > config.dependency_rate:
                continue
            parent_index = rng.randrange(0, i)
            parent = tasks[parent_index]
            child = tasks[i]
            parent_finish = parent.release_time + parent.duration
            # the child must be able to start after the parent finishes and
            # still finish by its own deadline and by the horizon
            if parent_finish + child.duration > min(child.deadline, latest_shift):
                continue
            # and the edge must not push the child before its release time in a
            # way that contradicts the task's own time window
            if parent_finish > child.release_time:
                child.release_time = parent_finish
            child.dependencies = (parent.id,)

    # ---- assemble ----
    weights_obj = ObjectiveWeights.from_dict(
        {**ObjectiveWeights().to_dict(), **dict(config.weight_overrides)}
    )
    constraints = ScenarioConstraints.from_dict(
        {**ScenarioConstraints().to_dict(), **dict(config.constraint_overrides)}
    )
    if not constraints.allow_unassigned:
        constraints = replace(
            constraints, allow_unassigned=True, max_unassigned_fraction=0.2
        )

    scenario = Scenario(
        id=f"SCN-{config.name}-{config.seed}",
        name=config.name,
        tasks=tuple(tasks),
        resources=tuple(resources),
        travel=travel,
        objective_weights=weights_obj,
        constraints=constraints,
        horizon=Interval(0, config.horizon_minutes),
        depot=DEPOT_ID,
        description=config.description,
        tags=config.tags,
        metadata={
            "generator": "orion.data.scenario_generator",
            "generator_version": "1.0.0",
            "config": config.to_dict(),
            "seed": config.seed,
            "locations": len(coordinates),
        },
    )
    return scenario


def generate_suite(
    levels: Sequence[str] = ("small", "medium", "large", "stress"),
    *,
    seed: int = 20260101,
    **overrides: Any,
) -> list[Scenario]:
    """Generate one scenario per difficulty level, all from the same base seed."""
    return [
        generate(difficulty_config(level, seed=seed, **overrides)) for level in levels
    ]


def scaling_configs(
    sizes: Sequence[int],
    *,
    seed: int = 20260101,
    scarcity: float = 0.55,
    tightness: float = 0.5,
) -> list[ScenarioConfig]:
    """Configs for the scalability study.

    Resources scale with tasks at a fixed ratio, which is the honest way to run
    a scalability study: holding resources fixed would measure scarcity, not
    size. Scarcity and tightness are held constant so size is the only variable.
    """
    out: list[ScenarioConfig] = []
    for n in sizes:
        teams = max(2, round(n * 0.18))
        vehicles = max(1, round(n * 0.10))
        out.append(
            ScenarioConfig(
                name=f"scale-{n}",
                num_tasks=n,
                num_teams=teams,
                num_vehicles=vehicles,
                num_locations=max(6, min(40, round(n ** 0.6) + 5)),
                deadline_tightness=tightness,
                resource_scarcity=scarcity,
                seed=seed,
                tags=("scalability", f"n={n}"),
                description=f"Scalability instance with {n} tasks",
            )
        )
    return out


__all__ = [
    "ScenarioConfig",
    "generate",
    "generate_suite",
    "difficulty_config",
    "difficulty_config as difficulty",
    "scaling_configs",
    "DIFFICULTY_PRESETS",
    "SHIFTS",
    "CAPABILITY_POOL",
    "CAPACITY_POOL",
]
