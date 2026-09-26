"""Hand-checkable micro-instances used to validate the optimization formulation.

These are deliberately tiny - 2 to 6 tasks, 1 to 3 resources, a short horizon -
so that :mod:`orion.optimization.reference` can enumerate them exhaustively and
ORION's solvers can be compared against a proven optimum. They are constructed
explicitly (no randomness in the interesting parts) so a human can work out the
right answer on paper and a failing test means something real.

The distance/time relation is ``minutes = km * 2`` throughout, i.e. a constant
30 km/h, so the numbers stay easy to check by hand.
"""

from __future__ import annotations

import random
from typing import Sequence

from orion.domain.entities import (
    Priority,
    Resource,
    ResourceKind,
    Scenario,
    ScenarioConstraints,
    Task,
)
from orion.domain.travel import TravelMatrix
from orion.domain.time_model import Interval

#: km/h implied by the fixed minutes = km * 2 relation used by every fixture.
FIXTURE_SPEED_KMH = 30.0

#: Deterministic capability names, chosen so fixtures can vary the match.
ALL_SKILLS = ("electrical", "plumbing", "network", "hvac", "crane_heavy")


def grid_travel(size: int, *, step_km: float = 8.0, jitter: float = 0.0, seed: int = 0) -> TravelMatrix:
    """A symmetric distance matrix on a line of *size* locations, 0 = depot.

    With ``jitter=0`` this is a straight line, so a route's distance is simply
    the sum of the gaps it crosses - the easiest possible thing to verify by
    hand.
    """
    rnd = random.Random(seed)
    positions = [0.0]
    for index in range(1, size):
        positions.append(positions[-1] + step_km + (rnd.uniform(-jitter, jitter) if jitter else 0.0))
    locations = tuple(["DEPOT"] + [f"L{i:02d}" for i in range(1, size)])
    distance = [[round(abs(positions[i] - positions[j]), 3) for j in range(size)] for i in range(size)]
    minutes = [[int(round(d / FIXTURE_SPEED_KMH * 60.0)) for d in row] for row in distance]
    return TravelMatrix(locations=locations, distance_km=distance, travel_minutes=minutes)


def tiny_scenario(
    scenario_id: str = "TINY-1",
    *,
    task_count: int = 4,
    resource_count: int = 2,
    horizon: int = 600,
    release: int = 0,
    duration: int = 45,
    deadline_slack: int = 120,
    skills: Sequence[str] = ALL_SKILLS,
    seed: int = 0,
    enforce_deadlines: bool = True,
) -> Scenario:
    """A small, fully specified scenario for solver cross-checks.

    Every task needs one capability drawn from *skills*; every resource can do
    all of them unless *narrow* is passed. Deadlines are uniform so that a
    paper solution is checkable.
    """
    travel = grid_travel(task_count + 1, seed=seed)
    tasks = []
    for index in range(task_count):
        tasks.append(
            Task(
                id=f"T{index:02d}",
                location=f"L{index + 1:02d}",
                release_time=release,
                deadline=release + deadline_slack + index * 30,
                duration=duration,
                priority=Priority.MEDIUM,
                required_capabilities=(skills[index % len(skills)],),
            )
        )
    resources = []
    for index in range(resource_count):
        resources.append(
            Resource(
                id=f"R{index}",
                kind=ResourceKind.TEAM,
                capabilities=tuple(skills),
                home_location="DEPOT",
                shift_start=0,
                shift_end=horizon,
                operating_cost_per_hour=40.0,
                speed_factor=1.0,
            )
        )
    return Scenario(
        id=scenario_id,
        name=f"tiny {task_count}t/{resource_count}r",
        tasks=tuple(tasks),
        resources=tuple(resources),
        travel=travel,
        horizon=Interval(0, horizon),
        depot="DEPOT",
        constraints=ScenarioConstraints(enforce_deadlines=enforce_deadlines),
    )


def narrow_skill_scenario() -> Scenario:
    """Three tasks, three resources, exactly one capable match each.

    Hand-checkable: the only feasible plans assign T00-R0, T01-R1, T02-R2. If a
    solver's answer differs, capability filtering is broken.
    """
    travel = grid_travel(4)
    tasks = (
        Task(id="T00", location="L01", release_time=0, deadline=300,

             duration=45, priority=Priority.HIGH, required_capabilities=("electrical",)),
        Task(id="T01", location="L02", release_time=0, deadline=300,

             duration=45, priority=Priority.HIGH, required_capabilities=("plumbing",)),
        Task(id="T02", location="L03", release_time=0, deadline=300,

             duration=45, priority=Priority.HIGH, required_capabilities=("network",)),
    )
    resources = (
        Resource(id="R0", kind=ResourceKind.TEAM,

                 capabilities=("electrical",), home_location="DEPOT", shift_start=0,
                 shift_end=600, operating_cost_per_hour=40.0, speed_factor=1.0),
        Resource(id="R1", kind=ResourceKind.TEAM,

                 capabilities=("plumbing",), home_location="DEPOT", shift_start=0,
                 shift_end=600, operating_cost_per_hour=40.0, speed_factor=1.0),
        Resource(id="R2", kind=ResourceKind.TEAM,

                 capabilities=("network",), home_location="DEPOT", shift_start=0,
                 shift_end=600, operating_cost_per_hour=40.0, speed_factor=1.0),
    )
    return Scenario(
        id="TINY-SKILL", name="one capable resource per task", tasks=tasks,
        resources=resources, travel=travel, horizon=Interval(0, 600), depot="DEPOT",
    )


def infeasible_scenario() -> Scenario:
    """No resource is capable of the single task: the plan must be INFEASIBLE.

    Deliberately *not* silently repaired by widening the deadline - the point is
    to prove ORION reports infeasibility rather than degrading quietly.
    """
    travel = grid_travel(2)
    tasks = (
        Task(id="T00", location="L01", release_time=0,

             deadline=240, duration=60, priority=Priority.CRITICAL,
             required_capabilities=("underwater_welding",)),
    )
    resources = (
        Resource(id="R0", kind=ResourceKind.TEAM,

                 capabilities=("electrical", "plumbing"), home_location="DEPOT",
                 shift_start=0, shift_end=480, operating_cost_per_hour=50.0, speed_factor=1.0),
    )
    return Scenario(
        id="TINY-INFEASIBLE", name="no capable resource", tasks=tasks, resources=resources,
        travel=travel, horizon=Interval(0, 480), depot="DEPOT",
    )


def all_tasks_exhausted_scenario() -> Scenario:
    """One resource that can do everything but has no time for all of the tasks.

    Forces the optimiser to choose *which* tasks to drop, which is the decision
    the objective and the lateness penalty exist to make.
    """
    travel = grid_travel(4)
    tasks = tuple(
        Task(id=f"T{index:02d}", location=f"L{index + 1:02d}",

             release_time=0, deadline=180, duration=60,
             priority=Priority.LOW if index == 0 else Priority.CRITICAL,
             required_capabilities=("electrical",))
        for index in range(3)
    )
    resources = (
        Resource(id="R0", kind=ResourceKind.TEAM,

                 capabilities=("electrical",), home_location="DEPOT", shift_start=0,
                 shift_end=300, operating_cost_per_hour=40.0, speed_factor=1.0),
    )
    return Scenario(
        id="TINY-OVERLOAD", name="more demand than one resource can serve",
        tasks=tasks, resources=resources, travel=travel, horizon=Interval(0, 300),
        depot="DEPOT",
    )


#: The fixture set the verification suite and the tests both iterate over.
TINY_FIXTURES: dict[str, object] = {
    "single_task_single_resource": tiny_scenario("TINY-A", task_count=1, resource_count=1),
    "three_tasks_two_resources": tiny_scenario("TINY-B", task_count=3, resource_count=2),
    "five_tasks_two_resources": tiny_scenario("TINY-C", task_count=5, resource_count=2),
    "four_tasks_three_resources": tiny_scenario("TINY-D", task_count=4, resource_count=3),
    "one_capable_resource_per_task": narrow_skill_scenario(),
    "demand_exceeds_capacity": all_tasks_exhausted_scenario(),
    "no_capable_resource": infeasible_scenario(),
}
