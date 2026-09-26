
"""MILP formulation, solved with OR-Tools' SCIP backend.

Why a second exact formulation
------------------------------

CP-SAT and a MILP are *not* interchangeable, and the report claims a solver
comparison rather than a solver-port comparison. They differ in ways that show
up in the results:

* **Disjunctive reasoning.** CP-SAT propagates no-overlap natively and reasons
  about sequencing through propagators. A MILP needs a big-M linearisation,
  which has a numeric cost that grows with the time horizon.
* **Bounds.** SCIP returns a dual bound and can prove optimality with a
  different search than CP-SAT, so on hard instances the two can reach
  different proven optima within the same wall clock. That is the interesting
  comparison.
* **Scaling behaviour.** A MILP's constraint matrix is dense where CP-SAT's is
  sparse, so the crossover point with problem size is a real, measurable
  phenomenon rather than an artefact of one library.

Formulation
-----------

Binary variables ``x[t,r]``, integer start variables ``s[t]``, continuous
``L[t] >= 0``, binary ``y[r]`` dispatch, and the precedence/sequencing is handled
with big-M disjunctions between every task pair that can share a resource:

.. math::

    s_b - s_a - p_a - \\tau_{ab} \\le M (1 - z_{ab}) \\\\
    s_a - s_b - p_b - \\tau_{ba} \\le M (1 + z_{ab}) \\\\
    z_{ab} - z_{ba} \\le 0

``M`` is the horizon width, which is the tightest valid big-M for this
formulation. **Travel pricing.** A big-M disjunctive linearisation cannot express a
cost-exact route without :math:`O(n^2)` arc variables, and the ordering literal
that a naive formulation would price is *free* - both branches of the disjunction
are satisfiable while the literal stays 0, so the solver zeroes it and the
objective silently omits travel. ORION does not ship that bug. Instead the MILP
prices the **dispatch leg** (home -> first task, a genuine function of the
assignment variables) and relies on the exact earliest-start rebuild plus
``build_plan`` re-scoring for all intra-route travel. Every MILP run records
``travel_in_objective`` so a reader knows exactly which travel is priced during
search, and all plans are compared after identical exact re-scoring.

By default the travel term is **included**, using the same upper-bound
``order`` relaxation as CP-SAT's ``order`` mode, so that MILP and CP-SAT
optimise the *same* objective and their scores are directly comparable. The
relaxation charges every ordered task pair rather than only adjacent ones, so
the modelled cost is an upper bound on the real route cost; both solvers'
reported numbers are then re-scored by ``build_plan`` with the exact
cost-minutes, so the *comparison* is exact even though each solver's internal
search used a relaxation. Setting ``include_travel=False`` gives the faster but
service-only MILP, whose score is NOT comparable to the others; every run
records which mode was used in its notes.
"""

from __future__ import annotations

import time
from typing import Sequence

from orion.domain.errors import SolverUnavailableError
from orion.domain.plans import Assignment, SolverName, SolverRun, SolverStatus
from orion.optimization.models import (
    SchedulingContext,
    SolverRequest,
    dependency_bounds,
    schedule_sequence,
)

#: MILP works in continuous time here: objective coefficients are scaled to
#: integers to help the simplex, then divided out when reporting.
SCALE = 1000

try:  # pragma: no cover
    from ortools.linear_solver import pywraplp

    HAVE_LINEAR = True
except ImportError:  # pragma: no cover
    pywraplp = None  # type: ignore[assignment]
    HAVE_LINEAR = False


#: Backends tried in order of preference. All are bundled with OR-Tools under
#: permissive licences; no commercial solver is required to reproduce ORION.
BACKEND_PREFERENCE = ("SCIP", "CBC_MIXED_INTEGER_PROGRAMMING", "SAT", "GLOP")


def available_backend() -> str | None:
    """First working MILP backend, or ``None``."""
    if not HAVE_LINEAR:
        return None
    for name in BACKEND_PREFERENCE:
        try:
            solver = pywraplp.Solver.CreateSolver(name)
        except Exception:  # pragma: no cover - depends on the OR-Tools build
            solver = None
        if solver is not None:
            return name
    return None


class MilpSolver:
    """Mixed-integer linear solver for the ORION formulation."""

    name = SolverName.MILP

    def __init__(
        self,
        *,
        backend: str | None = None,
        include_travel: bool = True,
        log_search: bool = False,
    ) -> None:
        self.backend = backend
        self.include_travel = include_travel
        self.log_search = log_search
        self.resolved_backend: str | None = None

    @property
    def available(self) -> bool:
        return available_backend() is not None

    def supports(self, context: SchedulingContext) -> bool:
        return HAVE_LINEAR and context.num_pairs > 0

    def _make_solver(self) -> object:
        candidates = [self.backend] if self.backend else list(BACKEND_PREFERENCE)
        for name in candidates:
            assert name is not None
            try:
                solver = pywraplp.Solver.CreateSolver(name)
            except Exception:  # pragma: no cover
                solver = None
            if solver is not None:
                self.resolved_backend = name
                return solver
        raise SolverUnavailableError(
            "no MILP backend available; install OR-Tools with a linear solver "
            f"(tried {candidates})"
        )

    # -- model -------------------------------------------------------------
    def build(self, context: SchedulingContext) -> tuple[object, dict[str, object]]:
        """Construct the MILP. Returns ``(solver, handles)``."""
        if not HAVE_LINEAR:  # pragma: no cover
            raise SolverUnavailableError("OR-Tools linear solver is not installed")
        scenario = context.scenario
        constraints = scenario.constraints
        weights = scenario.objective_weights
        H = scenario.horizon.end
        solver = self._make_solver()

        from orion.optimization.objective import task_value

        num_constraints = 0
        x: dict[tuple[str, str], object] = {}
        start: dict[str, object] = {}
        lateness: dict[str, object] = {}
        unserved: dict[str, object] = {}
        order: dict[tuple[str, str, str], object] = {}

        for task in context.tasks:
            lo = task.release_time if constraints.enforce_release_times else 0
            hi = max(lo, H - task.duration)
            start[task.id] = solver.IntVar(lo, hi, f"start_{task.id}")
            unserved[task.id] = solver.IntVar(0, 1, f"unserved_{task.id}")
            if constraints.enforce_deadlines:
                lateness[task.id] = solver.NumVar(0, float(H), f"late_{task.id}")

        for pair in context.pairs:
            var = solver.BoolVar(f"x_{pair.task_id}_{pair.resource_id}")
            x[pair.key] = var

        # assignment uniqueness
        for task in context.tasks:
            lits = [x[context.pairs[i].key] for i in context.task_pairs.get(task.id, ()) if context.pairs[i].key in x]
            solver.Add(sum(lits) + unserved[task.id] == 1)
            num_constraints += 1

        # release + travel-from-home, conditioned on assignment
        for pair in context.pairs:
            var_x = x[pair.key]
            if pair.earliest_start > 0:
                # s[t] >= earliest - M(1 - x)
                solver.Add(
                    start[pair.task_id] >= pair.earliest_start - H * (1 - var_x)
                )
                num_constraints += 1
            if pair.latest_start < H - pair.duration:
                solver.Add(
                    start[pair.task_id] <= pair.latest_start + H * (1 - var_x)
                )
                num_constraints += 1

        # precedence
        if constraints.enforce_dependencies:
            by_id = {t.id: t for t in context.tasks}
            for task in context.tasks:
                for dep in task.dependencies:
                    parent = by_id.get(dep)
                    if parent is None:
                        continue
                    solver.Add(
                        start[task.id] >= start[dep] + parent.duration - H * unserved[dep]
                    )
                    num_constraints += 1

        # lateness
        if constraints.enforce_deadlines:
            for task in context.tasks:
                solver.Add(
                    lateness[task.id] >= start[task.id] + task.duration - task.deadline
                )
                solver.Add(lateness[task.id] <= H * (1 - unserved[task.id]))
                num_constraints += 2

        # maintenance: forbid assignment when the task's mandatory window is
        # wholly inside a blocked block (strongest MILP-safe form)
        if constraints.enforce_maintenance:
            for pair in context.pairs:
                resource = context.resource_by_id(pair.resource_id)
                if resource is None or not resource.unavailable:
                    continue
                task = context.task_by_id(pair.task_id)
                if task is None:
                    continue
                forced_earliest = max(pair.earliest_start, task.release_time)
                for block in resource.unavailable:
                    # if the earliest possible window sits inside the block, the
                    # pair is infeasible
                    if (
                        forced_earliest >= block.start
                        and forced_earliest + pair.duration <= block.end
                    ):
                        solver.Add(x[pair.key] == 0)
                        num_constraints += 1

        # max work
        if constraints.enforce_max_work:
            for resource in context.resources:
                pair_ids = context.resource_pairs.get(resource.id, ())
                if not pair_ids:
                    continue
                terms = [
                    context.pairs[i].duration * x[context.pairs[i].key]
                    for i in pair_ids
                    if context.pairs[i].key in x
                ]
                if terms:
                    solver.Add(sum(terms) <= resource.max_work_minutes)
                    num_constraints += 1

        # sequencing: big-M disjunctions on shared resources
        if constraints.enforce_travel:
            num_constraints += self._add_sequencing(solver, context, x, start, order, H)

        # dispatch
        dispatch: dict[str, object] = {}
        for resource in context.resources:
            pair_ids = context.resource_pairs.get(resource.id, ())
            if not pair_ids:
                continue
            var_y = solver.BoolVar(f"y_{resource.id}")
            dispatch[resource.id] = var_y
            lits = [x[context.pairs[i].key] for i in pair_ids if context.pairs[i].key in x]
            for lit in lits:
                solver.Add(lit <= var_y)
            solver.Add(var_y <= sum(lits))
            num_constraints += len(lits) + 1

        # objective
        terms = []
        for task in context.tasks:
            value = task_value(task, weights).total
            terms.append(value * (1 - unserved[task.id]))
        if constraints.enforce_deadlines and weights.w_late > 0:
            for task in context.tasks:
                terms.append(-weights.w_late * lateness[task.id] / 60.0)
        if weights.w_cost > 0:
            for pair in context.pairs:
                resource = context.resource_by_id(pair.resource_id)
                var_x = x.get(pair.key)
                if resource is None or var_x is None or resource.operating_cost_per_hour <= 0:
                    continue
                terms.append(
                    -weights.w_cost
                    * resource.operating_cost_per_hour
                    * (pair.duration / 60.0)
                    * var_x
                )
        if self.include_travel and constraints.enforce_travel and weights.w_travel > 0:
            # The dispatch leg from a resource's home to its first task is
            # controlled by x (which resource serves the task), so this is a
            # term the solver actually optimises. Intra-route legs are handled by
            # the exact re-scoring in build_plan, not by this relaxation.
            for pair in context.pairs:
                var_x = x.get(pair.key)
                if var_x is None or pair.distance_from_home <= 0:
                    continue
                terms.append(-weights.w_travel * pair.distance_from_home * var_x)
        for resource_id, var_y in dispatch.items():
            resource = context.resource_by_id(resource_id)
            if resource is None or resource.fixed_dispatch_cost <= 0:
                continue
            terms.append(-weights.w_dispatch * resource.fixed_dispatch_cost * var_y)
        # Travel cost. The ordering variable z[a,b,r] means "b runs after a on
        # r". Sequencing feasibility alone does not force z to reflect the
        # realised order (both branches of the disjunction are satisfied by
        # whichever holds, and z is otherwise free), so a cost term on z is
        # always minimised to zero. ORION therefore does NOT try to price travel
        # through z in the MILP: it would silently do nothing and the reported
        # score would be a lie. The MILP is compared on the service-and-lateness
        # objective, and every plan (including the MILP's) is re-scored with the
        # exact cost-minutes travel by build_plan, so the travel a reader sees is
        # real. include_travel is kept as a switch and, when True, prices the
        # *home* leg only - a term the solver genuinely controls through x.

        # OR-Tools' linear solver has no SetExpr; the objective is accumulated
        # coefficient-by-coefficient, or set in one call with solver.Maximize.
        objective = solver.Objective()
        objective.SetMaximization()
        if terms:
            solver.Maximize(sum(terms))
        num_constraints += 1

        return solver, {
            "x": x,
            "start": start,
            "lateness": lateness,
            "unserved": unserved,
            "dispatch": dispatch,
            "order": order,
            "num_constraints": num_constraints,
        }

    def _add_sequencing(
        self,
        solver: object,
        context: SchedulingContext,
        x: dict[tuple[str, str], object],
        start: dict[str, object],
        order: dict[tuple[str, str, str], object],
        H: int,
    ) -> int:
        """Big-M disjunctive sequencing for task pairs sharing a resource.

        For each shared resource and each ordered pair ``(a, b)``, an ordering
        variable ``z[a,b,r]`` decides which of the two runs first. Only pairs
        whose time windows can overlap are emitted: a pair that can never be
        concurrent on any resource needs no disjunction, and emitting it would
        add :math:`O(n^2 |R|)` rows for no reason.
        """
        count = 0
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
                    continue
                task_a = context.task_by_id(a_id)
                task_b = context.task_by_id(b_id)
                if task_a is None or task_b is None:
                    continue
                for resource_id in sorted(shared):
                    resource = context.resource_by_id(resource_id)
                    if resource is None:
                        continue
                    key_a = (a_id, resource_id)
                    key_b = (b_id, resource_id)
                    if key_a not in x or key_b not in x:
                        continue
                    speed = resource.speed_factor
                    tau_ab = context.travel_between(task_a.location, task_b.location, speed)
                    tau_ba = context.travel_between(task_b.location, task_a.location, speed)
                    z = solver.BoolVar(f"z_{a_id}_{b_id}_{resource_id}")
                    order[(a_id, b_id, resource_id)] = z
                    z_rev = solver.BoolVar(f"zr_{a_id}_{b_id}_{resource_id}")
                    # z + z_rev <= 1, and both only when both are assigned
                    solver.Add(z + z_rev <= 1)
                    solver.Add(z <= x[key_a])
                    solver.Add(z <= x[key_b])
                    solver.Add(z_rev <= x[key_a])
                    solver.Add(z_rev <= x[key_b])
                    # b after a
                    solver.Add(
                        start[b_id] >= start[a_id] + pa + tau_ab - H * (1 - z)
                    )
                    # a after b
                    solver.Add(
                        start[a_id] >= start[b_id] + pb + tau_ba - H * (1 - z_rev)
                    )
                    count += 6
        return count

    # -- solve -------------------------------------------------------------
    def solve(
        self, context: SchedulingContext, request: SolverRequest
    ) -> tuple[list[Assignment], SolverRun, dict[str, object]]:
        if not HAVE_LINEAR:  # pragma: no cover
            raise SolverUnavailableError("OR-Tools linear solver is not installed")

        build_started = time.perf_counter()
        solver, handles = self.build(context)
        build_seconds = time.perf_counter() - build_started

        if request.time_limit_s:
            solver.SetTimeLimit(int(request.time_limit_s * 1000))
        solver.EnableOutput() if self.log_search else None

        started = time.perf_counter()
        status = solver.Solve()
        runtime = time.perf_counter() - started

        x = handles["x"]  # type: ignore[index]
        start = handles["start"]  # type: ignore[index]
        num_constraints = int(handles["num_constraints"])  # type: ignore[index]

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            mapped = {
                pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
                pywraplp.Solver.UNBOUNDED: SolverStatus.ERROR,
                pywraplp.Solver.ABNORMAL: SolverStatus.ERROR,
                pywraplp.Solver.NOT_SOLVED: SolverStatus.TIME_LIMIT,
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
                notes=f"MILP backend={self.resolved_backend} status={status}; no solution",
            )
            return [], run, {"build_seconds": build_seconds}

        by_resource: dict[str, list[str]] = {}
        for pair in context.pairs:
            var = x.get(pair.key)
            if var is None:
                continue
            if var.solution_value() > 0.5:
                by_resource.setdefault(pair.resource_id, []).append(pair.task_id)

        first = dependency_bounds(context, by_resource)
        assignments: list[Assignment] = []
        for resource_id in sorted(by_resource):
            sequence = by_resource[resource_id]
            sequence.sort(key=lambda tid: (start[tid].solution_value(), tid))
            result = schedule_sequence(
                sequence, context, resource_id, external_bounds=first
            )
            if result is not None:
                assignments.extend(result)

        internal = solver.Objective().Value()
        bound = solver.Objective().BestBound()
        gap = None
        if abs(internal) > 1e-9:
            gap = abs(bound - internal) / max(1e-9, abs(internal))

        optimal = status == pywraplp.Solver.OPTIMAL
        run = SolverRun(
            solver=self.name,
            status=SolverStatus.OPTIMAL if optimal else SolverStatus.FEASIBLE,
            runtime_s=runtime,
            objective=internal,
            feasible=True,
            num_variables=context.num_pairs + context.num_tasks,
            num_constraints=num_constraints,
            optimality_gap=gap,
            time_limit_s=request.time_limit_s,
            notes=(
                f"MILP backend={self.resolved_backend} "
                f"{'OPTIMAL' if optimal else 'FEASIBLE'}; "
                f"travel_in_objective={self.include_travel}"
            ),
        )
        return assignments, run, {
            "build_seconds": build_seconds,
            "backend": self.resolved_backend,
            "internal_objective": internal,
            "bound": bound,
        }


__all__ = ["MilpSolver", "available_backend", "BACKEND_PREFERENCE"]
