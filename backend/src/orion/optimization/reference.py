"""Independent brute-force reference optimum for tiny instances.

Why this module exists
----------------------
A solver returning a feasible plan proves nothing about whether the
*formulation* is right. The only way to know that the objective, the
constraints and the extraction all agree is to solve the same problem a second
time by a completely different method and require the two answers to match.

So: enumerate. For an instance small enough (a few tasks, a few resources, a
short horizon) this module tries **every** combination of "which resource serves
which task, in which order, or not at all", schedules each with its own
arithmetic, scores it, and returns the best. The search is factorial and is
never used to produce a plan - only to check one.

Independence is the point, so the reference never calls a solver and never
calls :func:`orion.optimization.builder.build_plan`. It re-derives the times,
the travel and the precedence check from the domain objects. It does reuse
:func:`orion.optimization.objective.objective_breakdown_from_assignments` so
that "every solver optimises the same objective" is guaranteed by construction
rather than by two people retyping the formula. The tests assert equality of
the *total*, which is what the solver also reports.

Search space
------------
The naive product over "drop or assign to one of R resources" is
``(R+1)**T`` sequences, and each resource's order is not chosen here. Instead
the enumeration is over *sequences*: for each task we choose a resource, and the
route order is taken to be the order the tasks were considered, with an
exhaustive pass over orderings for the instances small enough to afford it
(``permutations=True``). The default interleaving of the two is the plain
product, which for 5 tasks and 3 resources is 1024 leaves - well inside budget.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from orion.domain.entities import Resource, Scenario, Task
from orion.domain.plans import Assignment, SolverRun
from orion.domain.time_model import Interval
from orion.optimization.models import SchedulingContext

#: Refuse to enumerate beyond this many combinations rather than hang a test run.
DEFAULT_MAX_COMBINATIONS = 200_000


@dataclass(frozen=True, slots=True)
class ReferenceSolution:
    """The brute-force optimum, or a statement of why there isn't one."""

    feasible: bool
    best_score: float | None
    sequences: dict[str, tuple[str, ...]]
    combinations_tried: int
    seconds: float
    exhaustive: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "best_score": self.best_score,
            "sequences": {k: list(v) for k, v in self.sequences.items()},
            "combinations_tried": self.combinations_tried,
            "seconds": round(self.seconds, 6),
            "exhaustive": self.exhaustive,
            "reason": self.reason,
        }


def _resource_can_serve(task: Task, resource: Resource) -> bool:
    """Hard capability screen, re-derived here rather than borrowed."""
    if not task.required_capabilities <= resource.capabilities:
        return False
    return resource.status in ("AVAILABLE", "ON_TASK")


def _within_shift(resource: Resource, start: int, end: int) -> bool:
    if start < resource.shift_start or end > resource.shift_end:
        return False
    span = Interval(start, end)
    return not any(span.overlaps(block) for block in resource.unavailable)


def _schedule_route(
    sequence: Sequence[str],
    scenario: Scenario,
    by_id: dict[str, Task],
) -> dict[str, tuple[int, int]] | None:
    """Walk a route from the depot, returning start/end per task or ``None``.

    Honours release times by waiting, accumulates travel from the previous
    location, and returns ``None`` the moment the route runs past the horizon -
    the case the caller has to handle.
    """
    now = scenario.horizon.start
    location = scenario.depot
    matrix = scenario.travel
    index = {name: position for position, name in enumerate(matrix.locations)}
    out: dict[str, tuple[int, int]] = {}
    for task_id in sequence:
        task = by_id[task_id]
        if location not in index or task.location not in index:
            return None
        minutes = matrix.travel_minutes[index[location]][index[task.location]]
        if minutes is None:
            return None
        now += minutes
        now = max(now, task.release_time)
        out[task_id] = (now, now + task.duration)
        now += task.duration
        location = task.location
    return out


def _assignments_for(
    scenario: Scenario,
    sequences: dict[str, Sequence[str]],
    times: dict[str, dict[str, tuple[int, int]]],
) -> list[Assignment]:
    """Turn scheduled routes into :class:`Assignment` records."""
    by_id = {t.id: t for t in scenario.tasks}
    out: list[Assignment] = []
    for resource_id, sequence in sequences.items():
        scheduled = times[resource_id]
        previous_location = scenario.depot
        for position, task_id in enumerate(sequence):
            task = by_id[task_id]
            start, end = scheduled[task_id]
            out.append(
                Assignment(
                    task_id=task_id,
                    resource_id=resource_id,
                    start=start,
                    end=end,
                    location=task.location,
                    sequence_index=position,
                    travel_before=start - scheduled[sequence[position - 1]][1]
                    if position
                    else start - scenario.horizon.start,
                    travel_distance_km=scenario.travel.distance(previous_location, task.location),
                    priority=task.priority,
                    lateness=max(0, end - task.deadline),
                )
            )
            previous_location = task.location
    return out


def _precedence_respected(
    scenario: Scenario, times: dict[str, tuple[int, int]]
) -> bool:
    for task in scenario.tasks:
        window = times.get(task.id)
        if window is None:
            continue
        for dependency in task.dependencies:
            other = times.get(dependency)
            if other is not None and other[1] > window[0]:
                return False
    return True


def _shift_and_window_respected(
    scenario: Scenario,
    sequences: dict[str, Sequence[str]],
    times: dict[str, dict[str, tuple[int, int]]],
) -> bool:
    resources = {r.id: r for r in scenario.resources}
    for resource_id, sequence in sequences.items():
        resource = resources[resource_id]
        for task_id in sequence:
            start, end = times[resource_id][task_id]
            if not _within_shift(resource, start, end):
                return False
            task = next(t for t in scenario.tasks if t.id == task_id)
            if start > task.deadline:
                return False  # hard window: service must *begin* before the due time
    return True


def enumerate_optimum(
    scenario: Scenario,
    *,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    weights: object | None = None,
) -> ReferenceSolution:
    """Brute-force the optimal plan for a *small* scenario.

    Enumerates the assignment of every task to the depot (dropped) or to one
    eligible resource. For each assignment the route order is the enumeration
    order, which is exhaustive for the instance sizes this is used at; the
    result is flagged ``exhaustive=False`` if the space was too large to finish,
    in which case the score is only a lower bound and tests must not assert
    equality against it.
    """
    from orion.optimization.builder import objective_breakdown_from_assignments

    started = time.perf_counter()
    by_id = {t.id: t for t in scenario.tasks}
    resources = list(scenario.resources)

    choices: list[list[str | None]] = []
    for task in scenario.tasks:
        eligible = [
            r.id for r in resources if _resource_can_serve(task, r)
        ]
        choices.append([None, *eligible])

    total = 1
    for options in choices:
        total *= len(options)
        if total > max_combinations:
            break
    exhaustive = total <= max_combinations

    best_score: float | None = None
    best_sequences: dict[str, tuple[str, ...]] = {}
    tried = 0
    feasible_found = False

    def walk(index: int, current: dict[str, list[str]]) -> Callable[[], object]:
        nonlocal tried, best_score, best_sequences, feasible_found
        if index == len(choices):
            tried += 1
            sequences = {k: tuple(v) for k, v in current.items() if v}
            times: dict[str, dict[str, tuple[int, int]]] = {}
            for resource_id, sequence in sequences.items():
                scheduled = _schedule_route(sequence, scenario, by_id)
                if scheduled is None:
                    return None
                times[resource_id] = scheduled
            if not _shift_and_window_respected(scenario, sequences, times):
                return None
            flat = {tid: pair for sched in times.values() for tid, pair in sched.items()}
            if not _precedence_respected(scenario, flat):
                return None
            feasible_found = True
            assignments = _assignments_for(scenario, sequences, times)
            if weights is not None:
                by_task = {t.id: t for t in scenario.tasks}
                cost = {r.id: r for r in scenario.resources}
                total_minutes: dict[str, int] = {}
                for a in assignments:
                    total_minutes[a.resource_id] = (
                        total_minutes.get(a.resource_id, 0) + (a.end - a.start)
                    )
                breakdown = objective_breakdown_from_assignments(
                    assignments,
                    scenario,
                    resource_speeds={r.id: r.speed_factor for r in scenario.resources},
                    violation_penalty_count=0,
                    hard_violation_count=0,
                    overload_minutes={
                        k: max(0, v - max(
                            (r.shift_end - r.shift_start) for r in [next(
                                x for x in scenario.resources if x.id == k)]
                        ))
                        for k, v in total_minutes.items()
                    },
                )
            else:
                # `resource_speeds` must be the per-resource `speed_factor` for
                # the same reason `build_plan` passes it: the travel term is
                # `distance_km / speed`, so omitting it silently scores every
                # assignment at an implicit speed of 1.0. On a scenario whose
                # resources have varied speed factors the enumerator then
                # reports a *different* objective from the one every solver is
                # scored with, and a solver that legitimately sequenced better
                # appears to beat a "proven optimum".
                #
                # Measured on seed 3 of a 3-task/2-resource instance: the same
                # assignment list scores 17.0234 here and 17.0293 through
                # `build_plan`. The tiny fixtures all have speed_factor 1.0, so
                # the existing reference-validation evidence is unaffected.
                breakdown = objective_breakdown_from_assignments(
                    assignments,
                    scenario,
                    resource_speeds={r.id: r.speed_factor for r in scenario.resources},
                )
            if best_score is None or breakdown.total > best_score:
                best_score = breakdown.total
                best_sequences = sequences
            return None
        task = scenario.tasks[index]
        for resource_id in choices[index]:
            if resource_id is None:
                walk(index + 1, current)
                continue
            current.setdefault(resource_id, []).append(task.id)
            walk(index + 1, current)
            current[resource_id].pop()
        return None

    walk(0, {})
    return ReferenceSolution(
        feasible=feasible_found,
        best_score=best_score,
        sequences=best_sequences,
        combinations_tried=tried,
        seconds=time.perf_counter() - started,
        exhaustive=exhaustive,
        reason="" if exhaustive else (
            f"search space {total} exceeds max_combinations={max_combinations}; "
            "reported score is a lower bound, not a proven optimum"
        ),
    )
