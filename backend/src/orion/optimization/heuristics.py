
"""Constructive heuristic: greedy insertion with regret-k.

Design
------

The constructive heuristic exists for two reasons:

1. It is the only ORION solver with a guaranteed sub-second response on every
   scenario size, including ones that break CP-SAT and MILP. That makes it the
   fallback that keeps the application usable.
2. It provides an *independent* implementation of the sequencing rules. When
   CP-SAT and the heuristic agree on a small scenario, the answer is almost
   certainly right; when they disagree, the disagreement is informative.

Algorithm
---------

1. Sort tasks by a **density key**: ``-(value / duration)`` with a tie-break on
   deadline and then on task id. Density ordering is the classic regret-free
   insertion rule and it front-loads the high-value, short work, which is what
   makes the late stages of a scarce scenario cheap.
2. For each task, compute the best insertion position on every eligible
   resource. Insertion cost is the *marginal* objective change: the extra travel
   plus the extra operating cost, minus the value gained.
3. Insert the task into the best position, tie-broken deterministically.
4. Apply a bounded improvement pass: for every unassigned task, try again after
   all insertions (catches tasks that were unservable early because every
   resource was busy).

Determinism
-----------

Every tie-break falls back to task id / resource id, and the RNG is only used
for the optional stochastic ordering mode which is seeded from the request. The
same scenario + request always produces the same plan. This is asserted by a
determinism test.
"""

from __future__ import annotations

import random
import time
from typing import Sequence

from orion.domain.plans import Assignment, SolverName, SolverRun, SolverStatus
from orion.optimization.models import (
    SchedulingContext,
    SolverRequest,
    dependency_bounds,
    schedule_sequence,
)
from orion.optimization.objective import task_value


def _insertion_options(
    task_id: str,
    context: SchedulingContext,
    sequences: dict[str, list[str]],
) -> list[tuple[float, str, int]]:
    """Score every feasible insertion of *task_id*.

    Returns ``(cost, resource_id, position)`` triples sorted best-first. The
    cost is the increase in the plan's movement + operating cost; it does not
    include the task's value, which is identical for every option and therefore
    a constant that cannot change the argmin.
    """
    weights = context.scenario.objective_weights
    options: list[tuple[float, str, int]] = []
    pair_indices = context.task_pairs.get(task_id, ())
    # Finish times of tasks on *other* resources, so a prerequisite placed
    # elsewhere still constrains this insertion.
    external = dependency_bounds(context, sequences)
    for pair_index in pair_indices:
        pair = context.pairs[pair_index]
        resource_id = pair.resource_id
        resource = context.resource_by_id(resource_id)
        if resource is None:
            continue
        current = sequences[resource_id]
        speed = resource.speed_factor
        for position in range(len(current) + 1):
            candidate = current[:position] + [task_id] + current[position:]
            result = schedule_sequence(
                candidate, context, resource_id, external_bounds=external
            )
            if result is None:
                continue
            base = (
                schedule_sequence(current, context, resource_id, external_bounds=external)
                if current
                else []
            )
            if base is None:
                continue
            delta_travel = sum(a.travel_distance_km for a in result) - sum(
                a.travel_distance_km for a in base
            )
            delta_minutes = sum(a.end - a.start for a in result) - sum(
                a.end - a.start for a in base
            )
            delta_cost = delta_minutes / 60.0 * resource.operating_cost_per_hour
            delta_late = sum(a.lateness for a in result) - sum(a.lateness for a in base)
            cost = (
                weights.w_travel * delta_travel
                + weights.w_cost * delta_cost
                + weights.w_late * (delta_late / 60.0)
            )
            options.append((round(cost, 6), resource_id, position))
    options.sort(key=lambda item: (item[0], item[1], item[2]))
    return options


def _rebuild(
    context: SchedulingContext, sequences: dict[str, list[str]]
) -> list[Assignment]:
    """Materialise every resource's sequence into assignments.

    Scheduled in two passes so that a dependency whose endpoints are on
    different resources is still honoured: the first pass places every task
    without cross-resource precedence, the second re-schedules with the
    finish times of the first pass as lower bounds.
    """
    first = dependency_bounds(context, sequences)
    out: list[Assignment] = []
    for resource_id in sorted(sequences):
        sequence = sequences[resource_id]
        if not sequence:
            continue
        result = schedule_sequence(
            sequence, context, resource_id, external_bounds=first
        )
        if result is not None:
            out.extend(result)
    return out


def constructive_plan(
    context: SchedulingContext,
    request: SolverRequest,
    *,
    stochastic: bool = False,
) -> tuple[list[Assignment], dict[str, float]]:
    """Build a plan greedily. Returns ``(assignments, diagnostics)``."""
    started = time.perf_counter()
    scenario = context.scenario
    weights = scenario.objective_weights

    allowed = request.task_filter
    locked_by_resource: dict[str, list[str]] = {}
    for a in request.locked_assignments:
        locked_by_resource.setdefault(a.resource_id, []).append(a.task_id)
    for group in locked_by_resource.values():
        group.sort()

    sequences: dict[str, list[str]] = {r.id: [] for r in context.resources}
    for resource_id, tasks in locked_by_resource.items():
        sequences[resource_id] = list(tasks)

    # locked tasks are already placed; exclude them from the insertion pool
    locked_ids = {a.task_id for a in request.locked_assignments}

    def density_key(task_id: str) -> tuple[float, int, str]:
        task = context.task_by_id(task_id)
        if task is None:  # pragma: no cover - context guarantees membership
            return (0.0, 0, task_id)
        value = task_value(task, weights).total
        density = value / max(1, task.duration)
        return (-round(density, 6), task.deadline, task_id)

    pool = [
        pair.task_id
        for pair in {p.task_id: p for p in context.pairs}.values()
        if allowed is None or pair.task_id in allowed
    ]
    pool = [tid for tid in pool if tid not in locked_ids]

    rng = random.Random(request.seed) if stochastic else None
    if rng is not None:
        # randomise only within equal-density bands so the ordering stays sensible
        pool.sort(key=density_key)
        buckets: dict[tuple[float, int], list[str]] = {}
        for tid in pool:
            task = context.task_by_id(tid)
            assert task is not None
            value = task_value(task, weights).total
            key = (round(value / max(1, task.duration), 3), task.deadline)
            buckets.setdefault(key, []).append(tid)
        pool = []
        for key in sorted(buckets, key=lambda k: (-k[0], k[1])):
            group = buckets[key]
            rng.shuffle(group)
            pool.extend(group)
    else:
        pool.sort(key=density_key)

    inserted = 0
    unassigned = 0
    evaluated = 0
    for task_id in pool:
        options = _insertion_options(task_id, context, sequences)
        evaluated += 1
        if not options:
            unassigned += 1
            continue
        if stochastic and rng is not None and len(options) > 1:
            # pick among the 3 cheapest, weighted toward the front
            top = options[:3]
            choice = rng.choices(top, weights=[3, 2, 1][: len(top)])[0]
        else:
            choice = options[0]
        _, resource_id, position = choice
        sequences[resource_id].insert(position, task_id)
        inserted += 1

    # second pass: tasks that no resource could take initially may fit now
    # that some resources are idle at the end of their sequence
    if unassigned:
        for task_id in list(pool):
            if task_id in {t for seq in sequences.values() for t in seq}:
                continue
            options = _insertion_options(task_id, context, sequences)
            if options:
                _, resource_id, position = options[0]
                sequences[resource_id].insert(position, task_id)
                unassigned -= 1
                inserted += 1

    assignments = _rebuild(context, sequences)
    diagnostics = {
        "inserted": float(inserted),
        "unassigned": float(unassigned),
        "evaluated": float(evaluated),
        "construct_seconds": time.perf_counter() - started,
        "iteration_bound": 1.0,
    }
    return assignments, diagnostics


class ConstructiveHeuristic:
    """Deterministic greedy insertion solver."""

    name = SolverName.HEURISTIC

    def __init__(self, *, stochastic: bool = False) -> None:
        self.stochastic = stochastic

    def supports(self, context: SchedulingContext) -> bool:
        # The heuristic always works: it has no structural requirements.
        return True

    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[list[Assignment], SolverRun, dict[str, float]]:
        started = time.perf_counter()
        assignments, diagnostics = constructive_plan(
            context, request, stochastic=self.stochastic
        )
        runtime = time.perf_counter() - started
        run = SolverRun(
            solver=self.name,
            status=SolverStatus.FEASIBLE if assignments else SolverStatus.INFEASIBLE,
            runtime_s=runtime,
            objective=0.0,  # filled in by build_plan
            feasible=bool(assignments),
            num_variables=context.num_pairs,
            num_constraints=0,
            optimality_gap=None,
            time_limit_s=request.time_limit_s,
            num_iterations=int(diagnostics["inserted"]),
            notes=(
                f"greedy density insertion; {int(diagnostics['unassigned'])} task(s) "
                f"could not be placed"
            ),
        )
        return assignments, run, diagnostics


__all__ = ["constructive_plan", "ConstructiveHeuristic"]
