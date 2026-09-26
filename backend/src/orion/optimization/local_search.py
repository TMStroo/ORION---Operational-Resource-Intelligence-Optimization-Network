
"""Local search: adaptive large-neighbourhood destroy-and-repair (ALNS).

Motivation
----------

The constructive heuristic makes one pass of greedy decisions and stops. Its
weakness is that an early bad decision (a valuable task placed far from its
neighbours, a resource loaded with short tasks that block a long urgent one) is
never revisited. Local search fixes exactly that, and it is the solver that
bridges the gap between "instantly feasible" and "near-optimal".

Algorithm
---------

Each iteration:

1. **Destroy.** Remove a subset of the current solution using one of several
   operators, chosen adaptively.
2. **Repair.** Re-insert the removed tasks with the same greedy insertion used
   by the constructive heuristic.
3. **Accept** if the new solution is better, or with a small simulated-annealing
   probability when it is slightly worse (this is what lets the search escape a
   local optimum).
4. **Update operator weights** by the usual ALNS score/rho adaptation.

Destroy operators
-----------------

``random``       remove *k* random tasks
``related``      remove *k* tasks geographically near a random seed task; this is
                 the operator that exposes bad clustering decisions
``worst``        remove the *k* tasks with the highest marginal cost in the
                 current solution
``resource``     empty one resource entirely, forcing its work elsewhere
``deadline``     remove the *k* latest-finishing tasks, which is the pressure
                 that fixes lateness

Determinism
-----------

The RNG is seeded from the request, every operator choice falls back to a sorted
iteration order, and the acceptance test is deterministic given the RNG stream.
Same scenario + same seed + same budget = same plan. Asserted by a test.

Budget
------

The time budget is honoured strictly: the loop checks the clock every iteration
and returns the best solution found so far. A run that exceeds its budget is
reported as ``TIME_LIMIT`` with the number of completed iterations recorded, so
the quality/runtime trade-off can be plotted honestly.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Callable, Sequence

from orion.domain.plans import Assignment, SolverName, SolverRun, SolverStatus
from orion.optimization.builder import build_plan
from orion.optimization.heuristics import _insertion_options
from orion.optimization.models import (
    SchedulingContext,
    SolverRequest,
    dependency_bounds,
    schedule_sequence,
)
from orion.optimization.objective import task_value

DestroyFn = Callable[[dict[str, list[str]], SchedulingContext, int, random.Random], list[str]]


def _tasks_of(sequences: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for resource_id in sorted(sequences):
        out.extend(sequences[resource_id])
    return out


def _positions(sequences: dict[str, list[str]], task_id: str) -> tuple[str, int] | None:
    for resource_id in sorted(sequences):
        seq = sequences[resource_id]
        if task_id in seq:
            return resource_id, seq.index(task_id)
    return None


def _remove(sequences: dict[str, list[str]], task_ids: Sequence[str]) -> None:
    for task_id in task_ids:
        found = _positions(sequences, task_id)
        if found is not None:
            resource_id, position = found
            sequences[resource_id].pop(position)


def destroy_random(
    sequences: dict[str, list[str]], context: SchedulingContext, k: int, rng: random.Random
) -> list[str]:
    pool = _tasks_of(sequences)
    if not pool:
        return []
    k = min(k, len(pool))
    return rng.sample(sorted(pool), k)


def destroy_related(
    sequences: dict[str, list[str]], context: SchedulingContext, k: int, rng: random.Random
) -> list[str]:
    """Remove tasks clustered around a random seed task."""
    pool = _tasks_of(sequences)
    if not pool:
        return []
    seed_task = rng.choice(sorted(pool))
    seed_loc = context.task_by_id(seed_task)
    if seed_loc is None:
        return destroy_random(sequences, context, k, rng)
    travel = context.scenario.travel
    scored = sorted(
        (
            (travel.distance(seed_loc.location, context.task_by_id(tid).location), tid)  # type: ignore[union-attr]
            for tid in pool
            if context.task_by_id(tid) is not None
        )
    )
    return [tid for _, tid in scored[: min(k, len(scored))]]


def destroy_worst(
    sequences: dict[str, list[str]], context: SchedulingContext, k: int, rng: random.Random
) -> list[str]:
    """Remove the tasks whose removal frees the most costly time."""
    weights = context.scenario.objective_weights
    scored: list[tuple[float, str]] = []
    external = dependency_bounds(context, sequences)
    for resource_id in sorted(sequences):
        resource = context.resource_by_id(resource_id)
        if resource is None:
            continue
        result = (
            schedule_sequence(
                sequences[resource_id], context, resource_id, external_bounds=external
            )
            or []
        )
        for a in result:
            marginal = (
                weights.w_cost * resource.operating_cost_per_hour * (a.duration / 60.0)
                + weights.w_travel * a.travel_distance_km
                + weights.w_late * (a.lateness / 60.0)
            )
            scored.append((-marginal, a.task_id))
    scored.sort()
    return [tid for _, tid in scored[: min(k, len(scored))]]


def destroy_resource(
    sequences: dict[str, list[str]], context: SchedulingContext, k: int, rng: random.Random
) -> list[str]:
    """Empty one resource, forcing its work to be redistributed."""
    loaded = [rid for rid in sorted(sequences) if sequences[rid]]
    if not loaded:
        return []
    # pick a resource with a middling load: emptying the biggest or smallest is
    # rarely informative
    target = loaded[len(loaded) // 2] if len(loaded) > 2 else rng.choice(loaded)
    return list(sequences[target])


def destroy_deadline(
    sequences: dict[str, list[str]], context: SchedulingContext, k: int, rng: random.Random
) -> list[str]:
    """Remove the latest-finishing tasks, which is the lateness pressure."""
    ends: list[tuple[int, str]] = []
    external = dependency_bounds(context, sequences)
    for resource_id in sorted(sequences):
        for a in (
            schedule_sequence(
                sequences[resource_id], context, resource_id, external_bounds=external
            )
            or []
        ):
            ends.append((a.end, a.task_id))
    ends.sort(reverse=True)
    return [tid for _, tid in ends[: min(k, len(ends))]]


DESTRUCTION_OPERATORS: dict[str, DestroyFn] = {
    "random": destroy_random,
    "related": destroy_related,
    "worst": destroy_worst,
    "resource": destroy_resource,
    "deadline": destroy_deadline,
}


def _materialise(context: SchedulingContext, sequences: dict[str, list[str]]) -> list[Assignment]:
    """Two-pass materialisation so cross-resource precedence is honoured.

    See :func:`orion.optimization.heuristics._rebuild` for why a single pass is
    not sufficient.
    """
    first = dependency_bounds(context, sequences)
    out: list[Assignment] = []
    for resource_id in sorted(sequences):
        if not sequences[resource_id]:
            continue
        result = schedule_sequence(
            sequences[resource_id], context, resource_id, external_bounds=first
        )
        if result is not None:
            out.extend(result)
    return out


def _score(assignments: Sequence[Assignment], context: SchedulingContext) -> float:
    """Cheap surrogate objective for the inner loop.

    The full :func:`build_plan` is too expensive to call every iteration (it
    re-validates the whole plan). This surrogate sums the same terms that
    dominate the objective - service value, lateness, travel, cost - and the
    final best solution is always re-scored through ``build_plan`` so the
    reported number comes from the single scoring implementation.
    """
    weights = context.scenario.objective_weights
    tasks = {t.id: t for t in context.tasks}
    score = 0.0
    for a in assignments:
        task = tasks.get(a.task_id)
        if task is None:
            continue
        score += task_value(task, weights).total
        score -= weights.w_late * (a.lateness / 60.0)
        score -= weights.w_travel * a.travel_distance_km / max(1e-9, context.speed_of(a.resource_id))
        resource = context.resource_by_id(a.resource_id)
        if resource is not None:
            score -= weights.w_cost * resource.operating_cost_per_hour * (a.duration / 60.0)
    return score


class LocalSearchSolver:
    """ALNS destroy-and-repair over route sequences."""

    name = SolverName.LOCAL_SEARCH

    def __init__(
        self,
        *,
        destroy_fraction: float = 0.15,
        max_iterations: int = 400,
        temperature: float = 1.0,
        cooling: float = 0.995,
        seed: int = 0,
    ) -> None:
        if not 0.0 < destroy_fraction < 1.0:
            raise ValueError(f"destroy_fraction must be in (0, 1), got {destroy_fraction}")
        self.destroy_fraction = destroy_fraction
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.cooling = cooling
        self.seed = seed

    @property
    def available(self) -> bool:
        return True

    def supports(self, context: SchedulingContext) -> bool:
        return context.num_pairs > 0

    def initial_solution(
        self, context: SchedulingContext, request: SolverRequest
    ) -> dict[str, list[str]]:
        """Seed the search with the constructive heuristic's solution."""
        from orion.optimization.heuristics import constructive_plan

        assignments, _ = constructive_plan(context, request)
        sequences: dict[str, list[str]] = {r.id: [] for r in context.resources}
        by_resource: dict[str, list[str]] = {}
        for a in assignments:
            by_resource.setdefault(a.resource_id, []).append(a)
        for resource_id, group in by_resource.items():
            sequences[resource_id] = [a.task_id for a in sorted(group, key=lambda x: x.start)]
        # keep any task the heuristic could not place in the pool
        placed = {a.task_id for a in assignments}
        leftovers = [t.id for t in context.tasks if t.id not in placed]
        for task_id in leftovers:
            options = _insertion_options(task_id, context, sequences)
            if options:
                _, resource_id, position = options[0]
                sequences[resource_id].insert(position, task_id)
        return sequences

    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[list[Assignment], SolverRun, dict[str, object]]:
        started = time.perf_counter()
        deadline = (
            started + request.time_limit_s if request.time_limit_s else None
        )
        rng = random.Random(request.seed or self.seed)

        current = self.initial_solution(context, request)
        current_assignments = _materialise(context, current)
        current_score = _score(current_assignments, context)

        best = {rid: list(seq) for rid, seq in current.items()}
        best_assignments = list(current_assignments)
        best_score = current_score

        operator_names = sorted(DESTRUCTION_OPERATORS)
        weights = {name: 1.0 for name in operator_names}
        scores = {name: 0.0 for name in operator_names}
        uses = {name: 0 for name in operator_names}

        temperature = self.temperature
        iterations = 0
        improvements = 0
        hit_budget = False
        total_tasks = max(1, len(context.tasks))
        k_base = max(1, int(self.destroy_fraction * total_tasks))

        while iterations < self.max_iterations:
            if deadline is not None and time.perf_counter() >= deadline:
                hit_budget = True
                break
            iterations += 1

            # adaptive operator choice
            name = max(operator_names, key=lambda n: (weights[n], n))
            k = max(1, min(k_base, int(k_base * (0.5 + rng.random()))))
            removed = DESTRUCTION_OPERATORS[name](current, context, k, rng)
            if not removed:
                continue
            uses[name] += 1

            trial = {rid: list(seq) for rid, seq in current.items()}
            _remove(trial, removed)
            reinserted = 0
            # re-insert in density order so the repair is sensible
            pending = sorted(
                removed,
                key=lambda tid: (
                    -task_value(
                        context.task_by_id(tid), context.scenario.objective_weights
                    ).total,
                    context.task_by_id(tid).duration if context.task_by_id(tid) else 0,
                    tid,
                ),
            )
            for task_id in pending:
                options = _insertion_options(task_id, context, trial)
                if options:
                    _, resource_id, position = options[0]
                    trial[resource_id].insert(position, task_id)
                    reinserted += 1

            trial_assignments = _materialise(context, trial)
            trial_score = _score(trial_assignments, context)

            delta = trial_score - current_score
            accept = False
            if delta > 1e-9:
                accept = True
                scores[name] += delta
            elif delta > -1e-9:
                accept = True  # equal score: accept to enable drift
            else:
                probability = math.exp(max(-20.0, delta / max(1e-9, temperature)))
                if rng.random() < probability:
                    accept = True
            if accept:
                current = trial
                current_score = trial_score
                current_assignments = trial_assignments

            if trial_score > best_score + 1e-9:
                best = {rid: list(seq) for rid, seq in trial.items()}
                best_score = trial_score
                best_assignments = list(trial_assignments)
                improvements += 1

            # ALNS weight adaptation
            if iterations % 25 == 0:
                for op in operator_names:
                    if uses[op]:
                        weights[op] = 0.8 * weights[op] + 0.2 * (scores[op] / max(1e-9, uses[op]))
                        if weights[op] <= 0:
                            weights[op] = 1e-3
                temperature *= self.cooling

        runtime = time.perf_counter() - started
        final = build_plan(context, best_assignments, SolverRun(
            solver=self.name,
            status=SolverStatus.FEASIBLE,
            runtime_s=runtime,
            objective=best_score,
            feasible=True,
        ), strategy=self.name)
        # the plan-level metrics are authoritative; report the exact score
        run = SolverRun(
            solver=self.name,
            status=SolverStatus.TIME_LIMIT if hit_budget else SolverStatus.FEASIBLE,
            runtime_s=runtime,
            objective=final.score,
            feasible=bool(best_assignments),
            num_variables=context.num_pairs,
            num_constraints=len(DESTRUCTION_OPERATORS),
            optimality_gap=None,
            time_limit_s=request.time_limit_s,
            num_iterations=iterations,
            notes=(
                f"ALNS: {iterations} iteration(s), {improvements} improvement(s), "
                f"k_base={k_base}, temp={temperature:.3f}, "
                f"budget={'exhausted' if hit_budget else 'iteration cap'}"
            ),
        )
        return best_assignments, run, {
            "iterations": iterations,
            "improvements": improvements,
            "operator_usage": dict(uses),
            "hit_budget": hit_budget,
            "surrogate_score": best_score,
        }


__all__ = ["LocalSearchSolver", "DESTRUCTION_OPERATORS"]
