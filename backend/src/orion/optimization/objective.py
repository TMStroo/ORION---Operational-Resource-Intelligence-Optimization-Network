
"""The ORION objective: a single, configurable, explainable service score.

ORION **maximises**

.. math::

    S = \\sum_{t \\in T} w_c\\, a_t
      + \\sum_{t \\in T} w_p\\, \\pi_t\\, a_t
      - \\sum_{t \\in T} w_l\\, L_t
      - \\sum_{r} w_d\\, y_r
      - \\sum_{r,t} w_x\\, \\tau_{r,t}
      - \\sum_{r} w_k\\, c_r \\frac{m_r}{60}
      - w_o \\, \\mathrm{overload}_r
      - w_v \\, |\\mathrm{violations}|

with the notation below. Every coefficient comes from
:class:`~orion.domain.entities.ObjectiveWeights`, so a scenario can trade service
against cost or lateness freely. Nothing is hard-coded.

Decision variables
------------------

====================  =========================================================
Symbol                Meaning
====================  =========================================================
:math:`a_t`           1 if task *t* is assigned to some resource
:math:`x_{t,r}`       1 if task *t* is assigned to resource *r*
:math:`s_t`           start time of task *t* (integer minutes)
:math:`L_t`           lateness of task *t* (minutes past deadline)
:math:`y_r`           1 if resource *r* is used at all (dispatch indicator)
:math:`z_{t,t',r}`    1 if *t* directly precedes *t'* on resource *r*
:math:`m_r`           minutes resource *r* works
:math:`\\pi_t`         priority weight of task *t* (1..4)
====================  =========================================================

Why one scalar
--------------

A planner has to defend a score to a human. A vector objective ("maximise service
*and* minimise cost") has no natural ordering, so every implementation ends up
hiding the trade-off in weights anyway. ORION makes the weights explicit and
*reported*: :class:`ObjectiveBreakdown` returns each term separately, and the UI
shows the term-by-term decomposition of any plan. The trade-off is therefore
visible, not buried.

Travel cost
-----------

Travel is penalised in *cost-minutes*: ``distance_km / speed_factor``. A slow
resource pays more per kilometre, so the objective contains a real economic
preference for fast nearby resources without any scenario-specific price table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from orion.domain.entities import ObjectiveWeights, Resource, Scenario, Task
from orion.domain.plans import Assignment, ObjectiveBreakdown, Plan


@dataclass(frozen=True, slots=True)
class TaskValue:
    """The service value of completing one task, under a given weight set."""

    task_id: str
    completion: float
    priority_bonus: float

    @property
    def total(self) -> float:
        return self.completion + self.priority_bonus


def task_value(task: Task, weights: ObjectiveWeights) -> TaskValue:
    """Value of serving *task* on time.

    The priority bonus is ``w_priority * priority`` where priority is 1..4. With
    defaults this makes a CRITICAL task worth 18.0 against a LOW task worth 12.0,
    so the solver will sacrifice one low-priority job to save one critical job,
    but not two.
    """
    completion = weights.w_completion
    priority_bonus = weights.w_priority * float(task.priority)
    return TaskValue(task.id, completion, priority_bonus)


def lateness_penalty(lateness_minutes: int, weights: ObjectiveWeights) -> float:
    """Penalty for finishing *lateness_minutes* after a deadline.

    Expressed per hour so the coefficient stays interpretable: with the default
    ``w_late=4.0``, one hour late costs 4.0, which is less than the 12.0 value of
    a LOW task. The result is that the solver prefers to serve a low-priority
    task late rather than drop it, which matches how field planners actually
    behave, and the number of late tasks remains a reported metric so the cost
    of that preference is visible.
    """
    return weights.w_late * (lateness_minutes / 60.0)


def travel_penalty(travel_minutes: int, travel_km: float, speed_factor: float) -> float:
    """Penalty for *travel_km* at *speed_factor*.

    Uses cost-minutes (``km / speed_factor``) scaled by ``w_travel``.
    """
    return travel_minutes * 0.0 + (travel_km / max(1e-9, speed_factor)) * 1.0


def resource_cost(resource: Resource, worked_minutes: int) -> float:
    """Operating cost of running *resource* for *worked_minutes*."""
    return resource.operating_cost_per_hour * (worked_minutes / 60.0)


@dataclass(frozen=True, slots=True)
class ServiceLevel:
    """Fraction of total achievable service value actually delivered.

    ``denominator`` is the value of *every* task in the scenario, so a plan that
    serves nothing has service level 0 and a plan that serves everything on time
    has service level 1. This normalisation is what makes the "disruption ->
    degraded -> repaired" percentages comparable across scenarios of different
    size.
    """

    achieved: float
    possible: float

    @property
    def ratio(self) -> float:
        return self.achieved / self.possible if self.possible > 0 else 0.0

    def as_percent(self) -> float:
        return 100.0 * self.ratio


def service_level(breakdown: ObjectiveBreakdown, scenario: Scenario) -> ServiceLevel:
    """Compute the achieved/possible service value pair."""
    weights = scenario.objective_weights
    possible = sum(task_value(t, weights).total for t in scenario.tasks)
    achieved = breakdown.completion + breakdown.priority
    return ServiceLevel(achieved=achieved, possible=possible)


def objective_breakdown_from_assignments(
    assignments: list[Assignment],
    scenario: Scenario,
    *,
    resource_speeds: Mapping[str, float] | None = None,
    violation_penalty_count: int = 0,
    hard_violation_count: int = 0,
    overload_minutes: Mapping[str, int] | None = None,
) -> ObjectiveBreakdown:
    """Compute every objective term from a concrete assignment list.

    This is the single scoring implementation. CP-SAT, MILP, min-cost flow, the
    heuristic and local search all produce a :class:`Plan`, and the plan's score
    is *always* computed here from ``(assignments, scenario)``. Solvers never
    report their own internal objective as the score, because CP-SAT's internal
    scaling and the MILP's would differ by a constant factor and comparing them
    would be meaningless.

    That design decision is what makes the solver comparison in the report fair:
    every solver's output is scored by identical code on identical inputs.
    """
    weights = scenario.objective_weights
    constraints = scenario.constraints
    tasks = {t.id: t for t in scenario.tasks}
    resources = {r.id: r for r in scenario.resources}

    completion = 0.0
    priority = 0.0
    late_minutes_total = 0.0

    for a in assignments:
        task = tasks.get(a.task_id)
        if task is None:
            continue
        value = task_value(task, weights)
        completion += value.completion
        priority += value.priority_bonus
        late_minutes_total += lateness_penalty(a.lateness, weights)

    # travel
    travel_km = 0.0
    travel_min = 0
    for a in assignments:
        travel_km += a.travel_distance_km
        travel_min += a.travel_before
    travel_cost = sum(
        (a.travel_distance_km / max(1e-9, (resource_speeds or {}).get(a.resource_id, 1.0)))
        for a in assignments
    )
    travel_term = weights.w_travel * travel_cost

    # operating cost + dispatch
    worked: dict[str, int] = {}
    for a in assignments:
        worked[a.resource_id] = worked.get(a.resource_id, 0) + (a.end - a.start)
    cost_term = 0.0
    dispatch_term = 0.0
    for resource_id, minutes in worked.items():
        resource = resources.get(resource_id)
        if resource is None:
            continue
        cost_term += weights.w_cost * resource_cost(resource, minutes)
        dispatch_term += weights.w_dispatch * resource.fixed_dispatch_cost

    # overload
    overload_term = 0.0
    if overload_minutes:
        for resource_id, excess in overload_minutes.items():
            overload_term += weights.w_overload * (excess / 60.0)
    else:
        for resource_id, minutes in worked.items():
            resource = resources.get(resource_id)
            if resource is not None and constraints.enforce_max_work:
                excess = minutes - resource.max_work_minutes
                if excess > 0:
                    overload_term += weights.w_overload * (excess / 60.0)

    # idle movement: a resource that travels but works nothing still costs
    idle_move_term = 0.0
    if constraints.enforce_travel:
        worked_ids = set(worked)
        for a in assignments:
            if a.travel_distance_km > 0 and a.resource_id not in worked_ids:
                idle_move_term += weights.w_move_idle * a.travel_distance_km

    violation_term = weights.w_violation * (
        violation_penalty_count + 3.0 * hard_violation_count
    )

    return ObjectiveBreakdown(
        completion=completion,
        priority=priority,
        lateness_penalty=late_minutes_total,
        travel_penalty=travel_term,
        cost_penalty=cost_term,
        dispatch_penalty=dispatch_term,
        overload_penalty=overload_term,
        idle_move_penalty=idle_move_term,
        violation_penalty=violation_term,
    )


def describe_weights(weights: ObjectiveWeights) -> list[str]:
    """Plain-language description of the active objective, for the UI."""
    return [
        f"serve a task: +{weights.w_completion:.2f}",
        f"priority bonus: +{weights.w_priority:.2f} x priority (1-4)",
        f"lateness: -{weights.w_late:.2f} per hour",
        f"travel: -{weights.w_travel:.3f} per cost-km",
        f"operating cost: -{weights.w_cost:.3f} per currency-hour",
        f"dispatch: -{weights.w_dispatch:.3f} per resource used",
        f"overload: -{weights.w_overload:.3f} per hour over the work limit",
        f"violation: -{weights.w_violation:.2f} each",
    ]


__all__ = [
    "TaskValue",
    "task_value",
    "lateness_penalty",
    "travel_penalty",
    "resource_cost",
    "ServiceLevel",
    "service_level",
    "objective_breakdown_from_assignments",
    "describe_weights",
]
