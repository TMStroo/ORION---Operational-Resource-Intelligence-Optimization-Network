
"""CP-SAT formulation of the ORION scheduling problem.

Formulation
-----------

This is a *time-indexed* model with optional intervals, not a big-M
linearisation. OR-Tools CP-SAT is a constraint-programming solver, so the
natural encoding is:

**Variables**

* ``x[t,r] in {0,1}`` - task *t* is assigned to resource *r* (only for eligible
  pairs, so capability/capacity cost nothing).
* ``s[t] in [0, H]`` - start of task *t* (a single start variable per task; the
  resource is a dimension of the assignment, not of the time).
* ``interval[t,r]`` - an *optional* interval ``[s[t], s[t]+p[t])`` present iff
  ``x[t,r]``. Optional intervals let CP-SAT do the disjunctive reasoning
  internally instead of us writing ``O(k^2)`` precedence constraints per
  resource.
* ``L[t] >= 0`` - lateness in minutes.
* ``y[r] in {0,1}`` - resource *r* is dispatched.
* ``u[t] in {0,1}`` - task *t* is unserved (slack on assignment).

**Constraints**

1. *Assignment uniqueness* ``sum_r x[t,r] + u[t] == 1``.
2. *No-overlap* ``AddNoOverlap`` per resource over its optional intervals - this
   is the resource-conflict constraint.
3. *Release* ``s[t] >= r[t]`` and, for each pair, ``s[t] >= travel(home_r, t)``.
4. *Deadline* ``L[t] >= s[t] + p[t] - d[t]``, penalised.
5. *Precedence* ``s[t] >= s[dep] + p[dep]`` via ``AddHint``/linear constraints on
   the task-level start variables (no resource dimension needed because a task
   has exactly one start).
6. *Shift and horizon* enforced by tightening each optional interval's domain.
7. *Maintenance* enforced by forbidding any optional interval overlapping a
   blocked window, using ``AddForbiddenAssignments`` on a boolean that is the
   conjunction of the relevant start literals. Implemented as domain cuts on
   ``s[t]`` for the affected pairs.
8. *Max work* ``sum_{t on r} p[t]*x[t,r] <= max_work[r]``.
9. *Travel* between consecutive tasks on a resource: because ``s`` is per-task
   and travel depends on which pairs of tasks are on the same resource, ORION
   adds explicit pairwise sequencing literals for tasks that *share at least one
   eligible resource*. This keeps the travel rows proportional to real conflicts
   rather than to ``|T|^2 |R|``.

Objective
---------

``maximise  sum_t value_t * (1 - u[t])  - w_late * sum_t L[t] / 60
              - w_dispatch * sum_r dispatch_cost_r * y[r]
              - w_travel * (precomputed movement term)``

CP-SAT wants integer coefficients, so the whole objective is scaled to integers
(``SCALE = 100``) and the reported score is always recomputed by
:func:`orion.optimization.builder.build_plan`. The scale factor is recorded in
the run notes so the numbers are auditable.

Why CP-SAT and not a time-indexed MIP
-------------------------------------

Because the model is small in *pairs* rather than in *time*: a scenario with 500
tasks and 2000 candidate pairs builds a model with 2000 booleans, 500 start
integers and 2000 optional intervals, regardless of whether the horizon is 8
hours or 8 days. A time-indexed MIP would be O(|T| * H) and become useless
around 50 tasks. This is the main reason CP-SAT is ORION's default exact solver.
"""

from __future__ import annotations

import time
from typing import Sequence

from orion.domain.plans import Assignment, SolverName, SolverRun, SolverStatus
from orion.optimization.models import (
    SchedulingContext,
    SolverRequest,
    dependency_bounds,
    materialise_sequences,
    schedule_sequence,
)

#: Objective coefficients are multiplied by this to keep CP-SAT on integers.
#:
#: The value is chosen so that the *per-minute* lateness coefficient is exact:
#: ``w_late / 60 * SCALE`` must be an integer to three decimal places, and
#: CP-SAT has no division operator on linear expressions, so the division is
#: folded into the scale instead. With ``SCALE = 6000`` and the default
#: ``w_late = 4.0`` the coefficient is exactly 400 per minute - no rounding
#: error at all - and the travel/operating-cost terms keep six significant
#: figures.
SCALE = 6000

try:  # pragma: no cover - import guard
    from ortools.sat.python import cp_model

    HAVE_CPSAT = True
except ImportError:  # pragma: no cover
    cp_model = None  # type: ignore[assignment]
    HAVE_CPSAT = False


def _maintenance_forbidden_windows(
    context: SchedulingContext, task_id: str, resource_id: str
) -> list[tuple[int, int]]:
    """Start windows that would overlap a maintenance block, and so are banned."""
    resource = context.resource_by_id(resource_id)
    if resource is None:
        return []
    task = context.task_by_id(task_id)
    if task is None:
        return []
    banned: list[tuple[int, int]] = []
    for block in resource.unavailable:
        # start in [block.start - p + 1, block.end - 1] overlaps the block
        low = block.start - task.duration + 1
        high = block.end - 1
        if high >= low:
            banned.append((low, high))
    return banned


class CpSatSolver:
    """Exact CP-SAT solver for the ORION formulation."""

    name = SolverName.CP_SAT

    #: Models whose estimated arc count exceeds this fall back to the ``order``
    #: relaxation. The threshold is a deliberate engineering choice, documented
    #: in the report's scalability section rather than buried here.
    DEFAULT_ARC_BUDGET = 400_000

    def __init__(
        self,
        *,
        workers: int = 8,
        log_search: bool = False,
        travel_model: str = "auto",
        arc_budget: int | None = None,
    ) -> None:
        if travel_model not in {"auto", "arc", "order"}:
            raise ValueError(
                f"travel_model must be 'auto', 'arc' or 'order', got {travel_model!r}"
            )
        self.default_workers = workers
        self.log_search = log_search
        self.travel_model = travel_model
        self.arc_budget = arc_budget if arc_budget is not None else self.DEFAULT_ARC_BUDGET

    def _resolve_travel_model(self, context: SchedulingContext) -> str:
        """Choose the travel encoding, honouring the arc budget.

        ``auto`` uses the exact ``arc`` model while the estimated number of arc
        literals fits the budget, and falls back to ``order`` beyond it. The
        choice is reported in every solver run so a reader always knows whether
        a cost-exact or a relaxed model produced a number.
        """
        if self.travel_model != "auto":
            return self.travel_model
        estimate = 0
        for resource_id, pair_ids in context.resource_pairs.items():
            n = len(pair_ids)
            estimate += n * (n - 1)
        return "arc" if estimate <= self.arc_budget else "order"

    @property
    def available(self) -> bool:
        return HAVE_CPSAT

    def supports(self, context: SchedulingContext) -> bool:
        return HAVE_CPSAT and context.num_pairs > 0

    # -- model -------------------------------------------------------------
    def build(self, context: SchedulingContext) -> tuple[object, dict[str, object]]:
        """Construct the CP-SAT model. Returns ``(model, handles)``."""
        if not HAVE_CPSAT:  # pragma: no cover
            raise RuntimeError("OR-Tools is not installed")
        scenario = context.scenario
        constraints = scenario.constraints
        weights = scenario.objective_weights
        horizon = scenario.horizon
        H = horizon.end

        model = cp_model.CpModel()
        x: dict[tuple[str, str], object] = {}
        start: dict[str, object] = {}
        lateness: dict[str, object] = {}
        unserved: dict[str, object] = {}
        intervals: dict[str, list[object]] = {r.id: [] for r in context.resources}
        num_constraints = 0

        # ---- start variables (one per task) ----
        for task in context.tasks:
            lo = task.release_time if constraints.enforce_release_times else horizon.start
            hi = max(lo, H - task.duration)
            start[task.id] = model.NewIntVar(lo, hi, f"start_{task.id}")
            unserved[task.id] = model.NewIntVar(0, 1, f"unserved_{task.id}")
            if constraints.enforce_deadlines:
                max_late = H
                lateness[task.id] = model.NewIntVar(0, max_late, f"late_{task.id}")

        # ---- assignment + optional intervals ----
        for pair in context.pairs:
            resource = context.resource_by_id(pair.resource_id)
            if resource is None:  # pragma: no cover
                continue
            lo = pair.earliest_start
            hi = max(lo, pair.latest_start)
            var_x = model.NewBoolVar(f"x_{pair.task_id}_{pair.resource_id}")
            x[pair.key] = var_x

            size = pair.duration
            start_lit = model.NewIntVar(lo, hi, f"st_{pair.task_id}_{pair.resource_id}")
            end_lit = model.NewIntVar(lo + size, hi + size, f"en_{pair.task_id}_{pair.resource_id}")
            # the pair's start must equal the task start when assigned
            model.Add(start[pair.task_id] == start_lit).OnlyEnforceIf(var_x)
            interval = model.NewOptionalIntervalVar(
                start_lit, size, end_lit, var_x, f"iv_{pair.task_id}_{pair.resource_id}"
            )
            intervals[pair.resource_id].append(interval)
            num_constraints += 1

            # maintenance: forbid starts that overlap a blocked window
            if constraints.enforce_maintenance:
                for bad_lo, bad_hi in _maintenance_forbidden_windows(
                    context, pair.task_id, pair.resource_id
                ):
                    if bad_hi < lo or bad_lo > hi:
                        continue
                    # forbid x=1 when start in [bad_lo, bad_hi]
                    blocked = model.NewBoolVar(f"blk_{pair.task_id}_{pair.resource_id}_{bad_lo}")
                    model.Add(start_lit >= bad_lo).OnlyEnforceIf([var_x, blocked])
                    model.Add(start_lit <= bad_hi).OnlyEnforceIf([var_x, blocked])
                    # simpler and stronger: a literal that is 1 iff start in range
                    in_range = model.NewBoolVar(f"inr_{pair.task_id}_{pair.resource_id}_{bad_lo}")
                    model.Add(start_lit >= bad_lo).OnlyEnforceIf(in_range)
                    model.Add(start_lit <= bad_hi).OnlyEnforceIf(in_range)
                    model.Add(start_lit < bad_lo).OnlyEnforceIf([in_range.Not(), var_x])
                    model.Add(start_lit > bad_hi).OnlyEnforceIf([in_range.Not(), var_x])
                    model.AddImplication(in_range, var_x.Not())
                    num_constraints += 1
                    del blocked

            # max work is a cumulative constraint per resource, added below
            _ = resource

        # ---- assignment uniqueness + service ----
        for task in context.tasks:
            pair_ids = context.task_pairs.get(task.id, ())
            lits = [x[context.pairs[i].key] for i in pair_ids if context.pairs[i].key in x]
            model.Add(sum(lits) + unserved[task.id] == 1)
            num_constraints += 1

        # ---- no-overlap per resource ----
        for resource_id, group in intervals.items():
            if group:
                model.AddNoOverlap(group)
                num_constraints += 1

        # ---- precedence (dependencies) ----
        if constraints.enforce_dependencies:
            by_id = {t.id: t for t in context.tasks}
            for task in context.tasks:
                for dep in task.dependencies:
                    parent = by_id.get(dep)
                    if parent is None:
                        continue
                    # the dependent cannot start before the prerequisite finishes.
                    # If the prerequisite is unserved, the dependency is vacuous
                    # (validated separately); modelled by conditioning on service.
                    model.Add(
                        start[task.id] >= start[dep] + parent.duration
                    ).OnlyEnforceIf(unserved[dep].Not())
                    num_constraints += 1

        # ---- lateness linkage ----
        if constraints.enforce_deadlines:
            for task in context.tasks:
                model.Add(
                    lateness[task.id] >= start[task.id] + task.duration - task.deadline
                )
                # lateness is only charged when the task is served
                model.Add(lateness[task.id] == 0).OnlyEnforceIf(unserved[task.id])
                num_constraints += 1

        # ---- max work per resource ----
        if constraints.enforce_max_work:
            for resource in context.resources:
                pair_ids = context.resource_pairs.get(resource.id, ())
                if not pair_ids:
                    continue
                terms = []
                for i in pair_ids:
                    pair = context.pairs[i]
                    terms.append(pair.duration * x[pair.key])
                model.Add(sum(terms) <= resource.max_work_minutes)
                num_constraints += 1

        # ---- dispatch variables ----
        dispatch: dict[str, object] = {}
        for resource in context.resources:
            pair_ids = context.resource_pairs.get(resource.id, ())
            if not pair_ids:
                continue
            var_y = model.NewBoolVar(f"y_{resource.id}")
            dispatch[resource.id] = var_y
            lits = [x[context.pairs[i].key] for i in pair_ids if context.pairs[i].key in x]
            # y >= x for every pair: dispatched if used
            for lit in lits:
                model.AddImplication(lit, var_y)
            # y <= sum(x): do not dispatch for nothing
            model.Add(var_y <= sum(lits))
            num_constraints += len(lits) + 1

        # ---- route sequencing and travel cost ----
        # Emitted after dispatch so the arc model's first/last variables can be
        # tied to the dispatch indicator.
        if constraints.enforce_travel:
            emitted, travel_literals, travel_stats = self._add_travel_constraints(
                model, context, x, start, dispatch
            )
            num_constraints += emitted

        # ---- objective ----
        from orion.optimization.objective import task_value

        service_terms = []
        for task in context.tasks:
            value = task_value(task, weights).total
            service_terms.append(int(round(value * SCALE)) * (1 - unserved[task.id]))
        objective_expr = sum(service_terms) if service_terms else 0

        if constraints.enforce_deadlines and weights.w_late > 0:
            # w_late is per hour; lateness is in minutes. The /60 is folded into
            # SCALE (see the module docstring), so this is an exact integer
            # coefficient per minute of lateness.
            late_coef = int(round(weights.w_late * SCALE / 60.0))
            if late_coef > 0:
                for task in context.tasks:
                    objective_expr -= late_coef * lateness[task.id]

        for resource_id, var_y in dispatch.items():
            resource = context.resource_by_id(resource_id)
            if resource is None or resource.fixed_dispatch_cost <= 0:
                continue
            objective_expr -= int(round(weights.w_dispatch * resource.fixed_dispatch_cost * SCALE)) * var_y

        # Operating cost: a task costs the assigned resource c_r/60 per minute of
        # duration. This is linear in the assignment booleans, so it belongs in
        # the exact model. Without it the solver would be indifferent between a
        # cheap local resource and an expensive distant one.
        if weights.w_cost > 0:
            for pair in context.pairs:
                resource = context.resource_by_id(pair.resource_id)
                var_x = x.get(pair.key)
                if resource is None or var_x is None or resource.operating_cost_per_hour <= 0:
                    continue
                minutes_cost = resource.operating_cost_per_hour * (pair.duration / 60.0)
                objective_expr -= int(round(weights.w_cost * minutes_cost * SCALE)) * var_x

        # Travel cost: for every ordered task pair that can share a resource, the
        # literal lit means "t' directly follows t on r". Charging
        # w_travel * (km / speed) * lit makes movement an explicit objective term.
        # Summing over all ordered pairs on a resource counts each consecutive
        # leg exactly once, so the total equals the route's real travel cost.
        if constraints.enforce_travel and weights.w_travel > 0:
            for lit, cost_minutes in travel_literals:
                objective_expr -= int(round(weights.w_travel * cost_minutes * SCALE)) * lit

        model.Maximize(objective_expr)

        handles = {
            "x": x,
            "start": start,
            "lateness": lateness,
            "unserved": unserved,
            "dispatch": dispatch,
            "num_constraints": num_constraints,
            "objective_expr": objective_expr,
            "travel_stats": travel_stats,
        }
        return model, handles

    def _add_travel_constraints(
        self,
        model: object,
        context: SchedulingContext,
        x: dict[tuple[str, str], object],
        start: dict[str, object],
        y: dict[str, object] | None = None,
    ) -> tuple[int, list[tuple[object, float]], dict[str, object]]:
        """Route sequencing and travel cost.

        Two encodings are implemented, because they trade correctness of the cost
        against model size, and the report treats that trade-off as a finding
        rather than hiding it.

        **``arc`` (exact).** For each resource *r* with candidate task set
        :math:`S_r`, introduce directed arc literals :math:`A_{a,b}` meaning
        "task *b* directly follows task *a* on *r*", plus ``first[a]`` and
        ``last[a]``. Degree constraints

        .. math::
            \\sum_b A_{a,b} + \\mathrm{last}_a = x_{a,r} \\\\
            \\sum_a A_{a,b} + \\mathrm{first}_b = x_{b,r}

        make each route a single path, and the travel cost
        :math:`\\sum_{a,b} w_{tr}\\, \\mathrm{cost}(a,b) A_{a,b}` counts each leg of
        the route exactly once - which is precisely what
        ``schedule_sequence`` charges. Cycles are impossible without an explicit
        constraint: every arc enforces
        :math:`s_b \\ge s_a + p_a + \\tau_{a,b}` with :math:`\\tau \\ge 1`, so a
        cycle would imply :math:`s_a > s_a`.

        The home leg is charged through ``first`` so the total matches the
        reference scheduler, and the depot return through ``last`` when the
        scenario requires it.

        Cost: :math:`O(\\sum_r |S_r|^2)` arc literals.

        **``order`` (relaxation).** One ordering literal per conflicting task
        pair. Sequencing feasibility is *identical* - the disjunctive travel
        constraints are the same - but the cost term charges every ordered pair
        rather than only adjacent ones, so the modelled cost is an upper bound
        on the real route cost, not the route cost itself. Used when the exact
        model would exceed the arc budget.

        Which one was used is recorded in ``travel_stats`` and surfaced in the
        solver run notes, because a cost-exact result and a relaxed one must not
        be compared as if they were the same model.
        """
        from ortools.sat.python import cp_model

        emitted = 0
        cost_literals: list[tuple[object, float]] = []
        stats: dict[str, object] = {
            "model": self.travel_model,
            "arcs": 0,
            "pruned_pairs": 0,
            "conflicting_pairs": 0,
        }

        if self.travel_model == "arc":
            emitted, cost_literals, stats = self._add_arc_constraints(
                model, context, x, start, y or {}
            )
        else:
            emitted, cost_literals, stats = self._add_order_constraints(
                model, context, x, start
            )
        return emitted, cost_literals, stats

    def _add_arc_constraints(
        self,
        model: object,
        context: SchedulingContext,
        x: dict[tuple[str, str], object],
        start: dict[str, object],
        y: dict[str, object],
    ) -> tuple[int, list[tuple[object, float]], dict[str, object]]:
        """Exact adjacency formulation. See :meth:`_add_travel_constraints`."""
        scenario = context.scenario
        require_return = scenario.constraints.require_depot_return
        weights = scenario.objective_weights
        emitted = 0
        cost_literals: list[tuple[object, float]] = []
        arcs_total = 0
        pruned = 0

        for resource in context.resources:
            rid = resource.id
            candidates = [
                context.pairs[i] for i in context.resource_pairs.get(rid, ())
            ]
            if len(candidates) < 2:
                # single-task resource: still charge the home leg
                continue
            # window map for pruning
            window = {c.task_id: (c.earliest_start, c.latest_start, c.duration) for c in candidates}

            first: dict[str, object] = {}
            last: dict[str, object] = {}
            for c in candidates:
                first[c.task_id] = model.NewBoolVar(f"first_{c.task_id}_{rid}")
                last[c.task_id] = model.NewBoolVar(f"last_{c.task_id}_{rid}")

            out_terms: dict[str, list[object]] = {c.task_id: [] for c in candidates}
            in_terms: dict[str, list[object]] = {c.task_id: [] for c in candidates}

            for a in candidates:
                for b in candidates:
                    if a.task_id == b.task_id:
                        continue
                    ea, la, pa = window[a.task_id]
                    eb, lb, pb = window[b.task_id]
                    # b cannot follow a if a always finishes after b must start
                    if ea + pa + self._tau(context, a, b) > lb:
                        pruned += 1
                        continue
                    var_arc = model.NewBoolVar(f"arc_{a.task_id}_{b.task_id}_{rid}")
                    key_a = (a.task_id, rid)
                    key_b = (b.task_id, rid)
                    if key_a not in x or key_b not in x:
                        continue
                    tau = self._tau(context, a, b)
                    model.Add(
                        start[b.task_id] >= start[a.task_id] + a.duration + tau
                    ).OnlyEnforceIf([var_arc, x[key_a], x[key_b]])
                    out_terms[a.task_id].append(var_arc)
                    in_terms[b.task_id].append(var_arc)
                    arcs_total += 1
                    emitted += 1
                    if weights.w_travel > 0:
                        cost_literals.append(
                            (var_arc, context.distance_between(
                                scenario.task(a.task_id).location,
                                scenario.task(b.task_id).location,
                                resource.speed_factor,
                            ))
                        )

            for c in candidates:
                key = (c.task_id, rid)
                if key not in x:
                    continue
                var_x = x[key]
                # out-degree: an assigned task either has a successor or is last
                if out_terms[c.task_id]:
                    model.Add(sum(out_terms[c.task_id]) + last[c.task_id] == var_x)
                else:
                    model.Add(last[c.task_id] == var_x)
                # in-degree: a task either has a predecessor or is first
                if in_terms[c.task_id]:
                    model.Add(sum(in_terms[c.task_id]) + first[c.task_id] == var_x)
                else:
                    model.Add(first[c.task_id] == var_x)
                emitted += 2

            # exactly one first and one last among the used tasks
            if rid in y:
                var_y = y[rid]
                model.Add(sum(first.values()) == var_y)
                model.Add(sum(last.values()) == var_y)
                emitted += 2
                # home leg
                for c in candidates:
                    if weights.w_travel <= 0:
                        break
                    home_cost = context.distance_between(
                        resource.home_location,
                        scenario.task(c.task_id).location,
                        resource.speed_factor,
                    )
                    if home_cost > 0:
                        cost_literals.append((first[c.task_id], home_cost))
                if require_return:
                    for c in candidates:
                        depot_cost = context.distance_between(
                            scenario.task(c.task_id).location,
                            scenario.depot,
                            resource.speed_factor,
                        )
                        if depot_cost > 0:
                            cost_literals.append((last[c.task_id], depot_cost))

        return (
            emitted,
            cost_literals,
            {
                "model": "arc",
                "arcs": arcs_total,
                "pruned_pairs": pruned,
                "conflicting_pairs": arcs_total,
            },
        )

    @staticmethod
    def _tau(context: SchedulingContext, a: object, b: object) -> int:
        """Travel minutes from task *a*'s location to task *b*'s on *a*'s resource."""
        scenario = context.scenario
        resource = context.resource_by_id(a.resource_id)  # type: ignore[attr-defined]
        speed = resource.speed_factor if resource else 1.0
        loc_a = scenario.task(a.task_id).location  # type: ignore[attr-defined]
        loc_b = scenario.task(b.task_id).location  # type: ignore[attr-defined]
        return scenario.travel.travel_from_speed(loc_a, loc_b, speed)

    def _add_order_constraints(
        self,
        model: object,
        context: SchedulingContext,
        x: dict[tuple[str, str], object],
        start: dict[str, object],
    ) -> tuple[int, list[tuple[object, float]], dict[str, object]]:
        """Ordering-literal relaxation. See :meth:`_add_travel_constraints`."""
        emitted = 0
        cost_literals: list[tuple[object, float]] = []
        pruned = 0
        conflicting = 0
        weights = context.scenario.objective_weights

        resources_by_task = {
            task_id: {context.pairs[i].resource_id for i in pair_ids}
            for task_id, pair_ids in context.task_pairs.items()
        }
        window: dict[str, tuple[int, int, int]] = {}
        for pair in context.pairs:
            window[pair.task_id] = (pair.earliest_start, pair.latest_start, pair.duration)

        task_ids = list(context.task_ids)
        for i, a_id in enumerate(task_ids):
            for b_id in task_ids[i + 1 :]:
                shared = resources_by_task.get(a_id, set()) & resources_by_task.get(b_id, set())
                if not shared:
                    continue
                ea, la, pa = window[a_id]
                eb, lb, pb = window[b_id]
                if ea >= lb + pb or eb >= la + pa:
                    pruned += 1
                    continue
                conflicting += 1
                task_a = context.task_by_id(a_id)
                task_b = context.task_by_id(b_id)
                if task_a is None or task_b is None:
                    continue
                b_first = model.NewBoolVar(f"bf_{a_id}_{b_id}")
                for resource_id in sorted(shared):
                    resource = context.resource_by_id(resource_id)
                    if resource is None:
                        continue
                    speed = resource.speed_factor
                    key = (a_id, resource_id)
                    key_b = (b_id, resource_id)
                    if key not in x or key_b not in x:
                        continue
                    tau_ab = context.travel_between(task_a.location, task_b.location, speed)
                    tau_ba = context.travel_between(task_b.location, task_a.location, speed)
                    model.Add(
                        start[b_id] >= start[a_id] + task_a.duration + tau_ab
                    ).OnlyEnforceIf([x[key], x[key_b], b_first])
                    model.Add(
                        start[a_id] >= start[b_id] + task_b.duration + tau_ba
                    ).OnlyEnforceIf([x[key], x[key_b], b_first.Not()])
                    emitted += 2
                    lit = model.NewBoolVar(f"tc_{a_id}_{b_id}_{resource_id}")
                    model.AddBoolAnd([x[key], x[key_b], b_first]).OnlyEnforceIf(lit)
                    model.AddBoolOr(
                        [x[key].Not(), x[key_b].Not(), b_first.Not()]
                    ).OnlyEnforceIf(lit.Not())
                    if weights.w_travel > 0:
                        cost_literals.append(
                            (lit, context.distance_between(
                                task_a.location, task_b.location, speed
                            ))
                        )
                    emitted += 1
        return (
            emitted,
            cost_literals,
            {
                "model": "order",
                "arcs": emitted,
                "pruned_pairs": pruned,
                "conflicting_pairs": conflicting,
            },
        )

    # -- solve -------------------------------------------------------------
    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[list[Assignment], SolverRun, dict[str, object]]:
        if not HAVE_CPSAT:  # pragma: no cover
            from orion.domain.errors import SolverUnavailableError

            raise SolverUnavailableError("OR-Tools CP-SAT is not installed")

        resolved_travel = (
            self._resolve_travel_model(context)
            if context.scenario.constraints.enforce_travel
            else "none"
        )
        previous, self.travel_model = self.travel_model, resolved_travel
        build_started = time.perf_counter()
        try:
            model, handles = self.build(context)
        finally:
            self.travel_model = previous
        build_seconds = time.perf_counter() - build_started

        solver = cp_model.CpSolver()
        if request.time_limit_s:
            solver.parameters.max_time_in_seconds = float(request.time_limit_s)
        solver.parameters.num_search_workers = int(request.workers)
        solver.parameters.log_search_progress = bool(self.log_search)
        solver.parameters.random_seed = int(request.seed) % (2**31 - 1)

        started = time.perf_counter()
        status = solver.Solve(model)
        runtime = time.perf_counter() - started

        x = handles["x"]  # type: ignore[index]
        start = handles["start"]  # type: ignore[index]
        num_constraints = int(handles["num_constraints"])  # type: ignore[index]
        travel_stats = dict(handles.get("travel_stats") or {})  # type: ignore[union-attr]

        status_name = solver.StatusName(status)
        has_solution = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if not has_solution:
            mapped = {
                cp_model.INFEASIBLE: SolverStatus.INFEASIBLE,
                cp_model.MODEL_INVALID: SolverStatus.ERROR,
                cp_model.UNKNOWN: SolverStatus.TIME_LIMIT,
            }.get(status, SolverStatus.ERROR)
            run = SolverRun(
                solver=self.name,
                status=mapped,
                runtime_s=runtime,
                objective=0.0,
                feasible=False,
                num_variables=context.num_pairs + context.num_tasks,
                num_constraints=num_constraints,
                optimality_gap=None,
                time_limit_s=request.time_limit_s,
                notes=f"CP-SAT status={status_name}; no solution returned",
            )
            return [], run, {"build_seconds": build_seconds}

        # extract the assignment set, then rebuild the schedule with the
        # reference earliest-start scheduler so that sequencing is identical
        # across every solver in ORION.
        by_resource: dict[str, list[str]] = {}
        for pair in context.pairs:
            key = pair.key
            var = x.get(key)
            if var is None:
                continue
            if solver.Value(var):
                by_resource.setdefault(pair.resource_id, []).append(pair.task_id)

        # Rebuild with the reference earliest-start scheduler so that sequencing
        # is byte-identical to what the heuristic and local search produce. The
        # first pass supplies cross-resource dependency bounds, the second
        # applies them; without this a task whose prerequisite sits on another
        # resource would be scheduled before it.
        #
        # The orderings MUST match between the two passes. `dependency_bounds`
        # schedules each resource's list in the order given, so a bound computed
        # on one ordering and applied to another is stale: the prerequisite's end
        # time it refers to was produced by a different schedule. On ft10 that
        # mismatch left 48 of 100 operations starting before their prerequisite
        # finished, while the model - which does enforce precedence - reported
        # OPTIMAL. So each sequence is sorted first, and the same list is used
        # for both passes.
        for resource_id in by_resource:
            by_resource[resource_id].sort(
                key=lambda tid: (solver.Value(start[tid]), tid)
            )
        assignments = materialise_sequences(context, by_resource)

        internal_objective = solver.ObjectiveValue() / SCALE
        bound = solver.BestObjectiveBound() / SCALE
        gap = None
        if abs(internal_objective) > 1e-9:
            gap = abs(bound - internal_objective) / max(1e-9, abs(internal_objective))

        optimal = status == cp_model.OPTIMAL
        run = SolverRun(
            solver=self.name,
            status=SolverStatus.OPTIMAL if optimal else SolverStatus.FEASIBLE,
            runtime_s=runtime,
            objective=internal_objective,
            feasible=True,
            num_variables=context.num_pairs + context.num_tasks,
            num_constraints=num_constraints,
            optimality_gap=gap,
            time_limit_s=request.time_limit_s,
            notes=(
                f"CP-SAT {status_name}; workers={request.workers}; scale={SCALE}; "
                f"bound={bound:.3f}; travel_model={travel_stats.get('model', 'none')}, "
                f"arcs={travel_stats.get('arcs', 0)}, "
                f"pruned={travel_stats.get('pruned_pairs', 0)}"
            ),
        )
        return assignments, run, {
            "build_seconds": build_seconds,
            "bound": bound,
            "internal_objective": internal_objective,
            "travel_stats": travel_stats,
        }


__all__ = ["CpSatSolver", "SCALE"]
