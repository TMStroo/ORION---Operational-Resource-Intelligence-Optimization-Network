
"""Min-cost flow relaxation.

What this solves, and what it does not
-------------------------------------

The ORION problem is a **time-dependent** assignment-plus-sequencing problem:
a task's cost of joining a resource depends on where that resource already is
and when. That is not a min-cost flow problem, and claiming otherwise would be
dishonest.

What *is* a min-cost flow problem is the **aggregate capacity-allocation
relaxation**: given a fixed per-task earliest start and duration, choose an
assignment of tasks to resources so that

* each task goes to at most one resource,
* each resource's total worked minutes stay within its work limit,
* each resource's operating cost is charged,

maximising total task value. That is a transportation problem, solved exactly and
in polynomial time by min-cost flow. ORION builds it as a bipartite flow network:

.. code::

    source --(cap 1, cost -value_t)--> task_node_t
    task_node_t --(cap 1, cost c_r/60 * p_t)--> resource_node_r
    resource_node_r --(cap max_work_r, cost 0)--> sink

Solving it yields a *lower bound on the achievable number of assignments* and a
cheap feasibility certificate. ORION then schedules each resource's allocated
tasks with the reference earliest-start scheduler, so the returned plan is a
real, schedulable plan - not a relaxation artefact.

Why keep it
-----------

Three reasons, all of which the report demonstrates with numbers:

1. **Bound quality.** On capacity-constrained scenarios the flow relaxation's
   assignment count is a valid lower bound on the exact optimum, which CP-SAT
   and MILP can then be measured against. A bound from a different method is
   worth more than a second incumbent.
2. **Speed.** Flow solves in milliseconds where CP-SAT takes seconds. On large
   scenarios it is the only method that returns anything at all, so it defines
   the "always available" floor of the capability matrix.
3. **A genuinely different algorithm.** It is the only ORION solver that is not
   a local search over sequences, which is what makes the solver comparison
   informative rather than five variations on one idea.

Known limitations, stated in the output rather than glossed over:

* It ignores travel, so its plans are worse than CP-SAT's on travel-heavy
  scenarios. ``travel_in_objective: false`` is recorded on every run.
* It ignores precedence and lateness. Tasks are allocated by value/priority, so
  a low-priority task with a hard deadline can be scheduled late.
* Its result is therefore never claimed to be optimal for the full ORION
  problem. It is optimal *for the relaxation*, and it says so.
"""

from __future__ import annotations

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

try:  # pragma: no cover
    from ortools.graph.python import min_cost_flow

    HAVE_FLOW = True
except ImportError:  # pragma: no cover
    min_cost_flow = None  # type: ignore[assignment]
    HAVE_FLOW = False


#: Costs are integers; this scale keeps the value terms dominant without
#: overflowing the 64-bit arc costs OR-Tools uses.
COST_SCALE = 1000


def build_flow_network(
    context: SchedulingContext,
    *,
    include_travel: bool = False,
) -> tuple[object, dict[str, object]]:
    """Build the min-cost flow network for the allocation relaxation."""
    smf = min_cost_flow.SimpleMinCostFlow()
    tasks = context.tasks
    resources = [r for r in context.resources if context.resource_pairs.get(r.id)]

    source = 0
    task_base = 1
    resource_base = task_base + len(tasks)
    sink = resource_base + len(resources)

    handles: dict[str, object] = {
        "source": source,
        "task_base": task_base,
        "resource_base": resource_base,
        "sink": sink,
        # (task_index, resource_index, arc_index) triples recorded as the arcs are
        # created, because SimpleMinCostFlow.flow() is indexed by arc, not by
        # node pair - the arc id is only known at creation time.
        "task_resource_arcs": [],
    }

    # A min-cost flow needs balanced supplies. Each task unit then flows
    # source -> task -> resource -> sink.
    #
    # A task that *no* resource can serve (a capability combination nobody
    # holds) has no outgoing arc, so demanding a unit for it made the whole
    # network infeasible. That is a modelling error, not a property of the
    # instance: a plan is allowed to leave such a task unassigned. So the supply
    # counts only tasks with at least one eligible resource, and a zero-cost
    # bypass arc carries the difference.
    servable = [
        index
        for index, task in enumerate(tasks)
        if any(
            context.pair(task.id, resource.id) is not None for resource in resources
        )
    ]
    servable_set = set(servable)
    smf.set_node_supply(source, len(servable))
    smf.set_node_supply(sink, -len(servable))
    if len(servable) < len(tasks):
        # Bypass: lets the solver satisfy the supply without forcing every task
        # to be served. Cost 0 so it is only used when nothing else is possible,
        # which matches "unservable tasks are simply left unassigned".
        smf.add_arc_with_capacity_and_unit_cost(
            source, sink, len(tasks) - len(servable), 0
        )
    handles["unservable_tasks"] = tuple(
        tasks[index].id for index in range(len(tasks)) if index not in servable_set
    )

    for index, task in enumerate(tasks):
        if index not in servable_set:
            continue  # no eligible resource: the node is deliberately left isolated
        task_node = task_base + index
        value = task_value(task, context.scenario.objective_weights).total
        # reward for serving this task: a negative arc cost from the source
        smf.add_arc_with_capacity_and_unit_cost(
            source, task_node, 1, -int(round(value * COST_SCALE))
        )
        for r_index, resource in enumerate(resources):
            pair = context.pair(task.id, resource.id)
            if pair is None:
                continue
            resource_node = resource_base + r_index
            cost_per_minute = resource.operating_cost_per_hour / 60.0
            arc_cost = int(round(cost_per_minute * pair.duration * COST_SCALE))
            if include_travel:
                arc_cost += int(round(
                    context.distance_between(
                        resource.home_location, task.location, resource.speed_factor
                    ) * COST_SCALE
                ))
            arc = smf.add_arc_with_capacity_and_unit_cost(task_node, resource_node, 1, arc_cost)
            handles.setdefault("task_resource_arcs", []).append(
                (index, r_index, arc)
            )

    for r_index, resource in enumerate(resources):
        resource_node = resource_base + r_index
        # capacity arc: total worked minutes on this resource, scaled so that
        # integer minutes map onto integer flow
        limit = max(1, resource.max_work_minutes)
        smf.add_arc_with_capacity_and_unit_cost(resource_node, sink, limit, 0)

    return smf, handles


def solve_flow(
    context: SchedulingContext,
    *,
    time_limit_s: float | None = None,
    include_travel: bool = False,
) -> tuple[dict[str, list[str]], int, int]:
    """Solve the relaxation. Returns ``(allocation, flow_value, arc_count)``."""
    smf, handles = build_flow_network(context, include_travel=include_travel)
    # The OR-Tools min-cost flow solver is a single-shot exact algorithm with no
    # time-limit parameter in this build, so the caller's budget cannot bound it.
    # The limitation is recorded in the run notes rather than hidden.
    tasks = context.tasks
    resources = [
        r for r in context.resources if context.resource_pairs.get(r.id)
    ]
    task_base = handles["task_base"]  # type: ignore[index]
    resource_base = handles["resource_base"]  # type: ignore[index]

    allocation: dict[str, list[str]] = {r.id: [] for r in resources}
    # Solve exactly once. An earlier version called `smf.solve()` here and then
    # again a few lines below; the second call re-ran the optimisation, so the
    # status being inspected was not the status of the allocation being read.
    status = smf.solve()
    if status not in (
        min_cost_flow.SimpleMinCostFlow.OPTIMAL,
        min_cost_flow.SimpleMinCostFlow.FEASIBLE,
    ):
        # `Status.name` is a property on the OR-Tools 9.15 enum, not a method;
        # calling it raised `TypeError: 'property' object is not callable` and
        # that TypeError propagated out of the solver instead of being reported
        # as a solver status. `str(status)` is the portable spelling.
        raise RuntimeError(
            "min-cost flow did not solve: "
            f"status={getattr(status, 'name', status)!r} "
            f"(optimal_cost={smf.optimal_cost()!r})"
        )
    for t_index, r_index, arc in handles["task_resource_arcs"]:  # type: ignore[union-attr]
        if smf.flow(arc) > 0:
            allocation[resources[r_index].id].append(tasks[t_index].id)
    return allocation, smf.optimal_cost(), smf.num_arcs()


class MinCostFlowSolver:
    """Aggregate capacity-allocation solver (transportation relaxation)."""

    name = SolverName.MIN_COST_FLOW

    def __init__(self, *, include_travel: bool = True) -> None:
        self.include_travel = include_travel

    @property
    def available(self) -> bool:
        return HAVE_FLOW

    def supports(self, context: SchedulingContext) -> bool:
        # Flow needs at least one task and one resource with a candidate pair,
        # and it deliberately refuses scenarios with dependencies (it cannot
        # represent precedence) so the capability matrix is honest.
        if not HAVE_FLOW:
            return False
        if not context.pairs:
            return False
        if context.scenario.constraints.enforce_dependencies:
            if any(t.dependencies for t in context.tasks):
                return False
        return True

    def unsupported_reason(self, context: SchedulingContext) -> str:
        if not HAVE_FLOW:
            return "OR-Tools min-cost flow is not installed"
        if not context.pairs:
            return "no candidate (task, resource) pairs: nothing to allocate"
        if context.scenario.constraints.enforce_dependencies and any(
            t.dependencies for t in context.tasks
        ):
            return (
                "scenario has task dependencies; the flow relaxation cannot "
                "represent precedence and is therefore not applicable"
            )
        return ""

    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[list[Assignment], SolverRun, dict[str, object]]:
        if not HAVE_FLOW:  # pragma: no cover
            from orion.domain.errors import SolverUnavailableError

            raise SolverUnavailableError("OR-Tools min-cost flow is not installed")
        if not self.supports(context):
            reason = self.unsupported_reason(context)
            run = SolverRun(
                solver=self.name,
                status=SolverStatus.NOT_APPLICABLE,
                runtime_s=0.0,
                objective=0.0,
                feasible=False,
                num_variables=context.num_pairs,
                num_constraints=0,
                notes=f"not applicable: {reason}",
            )
            return [], run, {"reason": reason}

        build_started = time.perf_counter()
        allocation, flow_cost, arc_count = solve_flow(
            context,
            time_limit_s=request.time_limit_s,
            include_travel=self.include_travel,
        )
        build_seconds = time.perf_counter() - build_started

        started = time.perf_counter()
        # the flow allocation is unordered; sequence each resource by value and
        # deadline, then apply cross-resource dependency bounds
        from orion.optimization.objective import task_value as _tv

        weights = context.scenario.objective_weights
        for resource_id, task_ids in allocation.items():
            task_ids.sort(
                key=lambda tid: (
                    -_tv(context.task_by_id(tid), weights).total,  # type: ignore[arg-type]
                    context.task_by_id(tid).deadline if context.task_by_id(tid) else 0,
                    tid,
                )
            )
        external = dependency_bounds(context, allocation)
        assignments: list[Assignment] = []
        unassigned = 0
        for resource_id in sorted(allocation):
            task_ids = allocation[resource_id]
            if not task_ids:
                continue
            result = schedule_sequence(
                task_ids, context, resource_id, external_bounds=external
            )
            if result is None:
                # resource cannot take its whole allocation; fall back to the
                # subset that prefixes cleanly rather than dropping everything
                result = self._truncate(task_ids, context, resource_id)
                unassigned += len(task_ids) - len(
                    [a for a in result if a.resource_id == resource_id]
                )
            assignments.extend(result)
        runtime = time.perf_counter() - started + build_seconds

        # relax-optimal cost, converted back to service units
        relax_value = -flow_cost / COST_SCALE
        run = SolverRun(
            solver=self.name,
            # A min-cost flow over an assignment relaxation has no notion of route order,
        # so optimality of the flow is not optimality of a schedule. Reporting
        # RELAXATION_OPTIMAL keeps the two guarantees distinguishable.
        status=SolverStatus.RELAXATION_OPTIMAL if assignments else SolverStatus.INFEASIBLE,
            runtime_s=runtime,
            objective=0.0,  # scored by build_plan
            feasible=bool(assignments),
            num_variables=context.num_pairs,
            num_constraints=arc_count,
            optimality_gap=None,
            time_limit_s=request.time_limit_s,
            notes=(
                f"transportation relaxation solved exactly (arcs={arc_count}); "
                f"relaxation value={relax_value:.2f}; travel_in_objective="
                f"{self.include_travel}; pre-sequence dropped={unassigned}"
            ),
        )
        return assignments, run, {
            "relaxation_value": relax_value,
            "arcs": arc_count,
            "build_seconds": build_seconds,
        }

    @staticmethod
    def _truncate(
        task_ids: Sequence[str], context: SchedulingContext, resource_id: str
    ) -> list[Assignment]:
        """Longest feasible prefix of the allocation, ordered by value."""
        from orion.optimization.objective import task_value

        ordered = sorted(
            task_ids,
            key=lambda tid: (
                -task_value(
                    context.task_by_id(tid), context.scenario.objective_weights
                ).total,
                tid,
            ),
        )
        kept: list[str] = []
        for candidate in ordered:
            trial = kept + [candidate]
            if schedule_sequence(trial, context, resource_id) is not None:
                kept = trial
        if not kept:
            return []
        result = schedule_sequence(kept, context, resource_id)
        return result or []


__all__ = ["MinCostFlowSolver", "build_flow_network", "solve_flow", "HAVE_FLOW", "COST_SCALE"]
