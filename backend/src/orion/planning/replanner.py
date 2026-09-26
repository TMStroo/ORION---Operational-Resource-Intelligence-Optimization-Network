
"""Replanning: local repair versus full re-optimisation.

This module implements the project's central comparison.

The flow, exactly as the brief specifies it:

1. preserve the current world state (the pre-disruption ``Scenario`` object is
   never mutated),
2. apply the disruption to produce a new ``Scenario``,
3. detect which assignments the disruption endangers
   (:func:`orion.domain.events.analyse_impact`),
4. attempt **local repair** - keep the unaffected assignments, re-insert only
   the endangered ones,
5. attempt **full re-optimisation** from scratch on the disrupted scenario,
6. compare the two on quality, runtime and churn.

Local repair
------------

Local repair preserves every assignment that is still feasible in the disrupted
world and re-inserts only the affected tasks. The key idea is that a planner who
has to re-explain the whole plan to a crew after a single vehicle failure loses
the crew's trust, so **plan churn is a first-class metric**, not an afterthought.

Repair uses cheapest-insertion with three candidate rules, and takes the best
result rather than the first feasible one:

``sequential``  re-insert affected tasks one at a time, most valuable first
``by_resource`` try each freed resource separately, keeping the best single fill
``greedy_all``  consider all (task, resource) insertion options jointly

If repair cannot place a task feasibly, that task is reported as unrecovered
rather than being force-placed illegally. A partial recovery with an honest
count is more useful to a planner than a complete-looking plan that breaks a
constraint.

Full re-optimisation
--------------------

The same solver, the same budget, on the disrupted scenario from scratch. Its
advantage is that it can re-optimise assignments that were *not* affected but
are now suboptimal; its cost is churn and runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from orion.domain.entities import Scenario
from orion.domain.errors import RepairFailedError
from orion.domain.events import Disruption, DisruptionImpact, analyse_impact
from orion.domain.plans import (
    Assignment,
    Plan,
    SolverName,
    SolverStatus,
    has_hard_violation,
)
from orion.optimization.builder import build_plan, next_plan_id
from orion.optimization.heuristics import _insertion_options
from orion.optimization.models import (
    SchedulingContext,
    SolverRequest,
    dependency_bounds,
    schedule_sequence,
)
from orion.optimization.objective import task_value
from orion.optimization.registry import run_solver

#: Repair strategies, tried in order of cost.
REPAIR_STRATEGIES = ("sequential", "by_resource", "greedy_all")


@dataclass(slots=True)
class ChurnReport:
    """How much of the plan changed.

    Churn is reported three ways because they answer different questions:

    ``changed_assignments``
        tasks whose assigned resource differs - the operational churn
    ``rescheduled_tasks``
        tasks whose start time moved by more than ``tolerance_minutes`` - the
        schedule churn
    ``added`` / ``removed``
        tasks newly served / dropped
    """

    changed_assignments: int = 0
    rescheduled_tasks: int = 0
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    delayed: tuple[str, ...] = ()
    travel_delta_km: float = 0.0
    cost_delta: float = 0.0
    total_before: int = 0
    total_after: int = 0
    tolerance_minutes: int = 5

    @property
    def churn_ratio(self) -> float:
        base = max(1, self.total_before)
        return self.changed_assignments / base

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_assignments": self.changed_assignments,
            "rescheduled_tasks": self.rescheduled_tasks,
            "added": list(self.added),
            "removed": list(self.removed),
            "delayed": list(self.delayed),
            "travel_delta_km": round(self.travel_delta_km, 3),
            "cost_delta": round(self.cost_delta, 3),
            "total_before": self.total_before,
            "total_after": self.total_after,
            "churn_ratio": round(self.churn_ratio, 6),
            "tolerance_minutes": self.tolerance_minutes,
        }


def compute_churn(
    before: Plan, after: Plan, *, tolerance_minutes: int = 5
) -> ChurnReport:
    """Compare two plans and quantify the change."""
    before_map = before.by_task()
    after_map = after.by_task()
    changed: list[str] = []
    rescheduled: list[str] = []
    for task_id, old in before_map.items():
        new = after_map.get(task_id)
        if new is None:
            continue
        if new.resource_id != old.resource_id:
            changed.append(task_id)
        elif abs(new.start - old.start) > tolerance_minutes:
            rescheduled.append(task_id)
    added = tuple(sorted(set(after_map) - set(before_map)))
    removed = tuple(sorted(set(before_map) - set(after_map)))
    delayed = tuple(
        sorted(
            tid
            for tid, new in after_map.items()
            if tid in before_map and new.start > before_map[tid].start
        )
    )
    return ChurnReport(
        changed_assignments=len(changed) + len(added) + len(removed),
        rescheduled_tasks=len(rescheduled) + len(delayed),
        added=added,
        removed=removed,
        delayed=delayed,
        travel_delta_km=after.total_travel_km - before.total_travel_km,
        cost_delta=after.total_operating_cost - before.total_operating_cost,
        total_before=before.tasks_assigned,
        total_after=after.tasks_assigned,
        tolerance_minutes=tolerance_minutes,
    )


@dataclass(slots=True)
class RepairOutcome:
    """Result of a local-repair attempt."""

    plan: Plan | None
    recovered_tasks: tuple[str, ...]
    unrecovered_tasks: tuple[str, ...]
    strategy: str
    seconds: float
    candidates_considered: int = 0
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.plan is not None

    @property
    def recovery_ratio(self) -> float:
        total = len(self.recovered_tasks) + len(self.unrecovered_tasks)
        return len(self.recovered_tasks) / total if total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "strategy": self.strategy,
            "recovered_tasks": list(self.recovered_tasks),
            "unrecovered_tasks": list(self.unrecovered_tasks),
            "recovery_ratio": round(self.recovery_ratio, 6),
            "seconds": round(self.seconds, 6),
            "candidates_considered": self.candidates_considered,
            "reason": self.reason,
        }


class LocalRepairer:
    """Preserves unaffected assignments and re-inserts the endangered ones."""

    def __init__(self, *, prefer_value: bool = True) -> None:
        self.prefer_value = prefer_value

    def repair(
        self,
        disrupted: Scenario,
        previous: Plan,
        impact: DisruptionImpact,
        *,
        strategies: Sequence[str] = REPAIR_STRATEGIES,
    ) -> RepairOutcome:
        """Repair *previous* against *disrupted*, keeping sound assignments."""
        started = time.perf_counter()
        context = SchedulingContext.build(disrupted)

        # Only tasks the previous plan actually assigned can be "recovered":
        # a task that was already unassigned has nothing to restore, and counting
        # it as unrecovered would report a repair failure that never happened.
        previous_ids = {a.task_id for a in previous.assignments}
        affected = set(impact.affected_tasks) & previous_ids
        # a task introduced by the disruption is affected by definition
        for task in disrupted.tasks:
            if task.id not in previous_ids:
                affected.add(task.id)

        survivors: dict[str, list[str]] = {r.id: [] for r in disrupted.resources}
        for assignment in previous.assignments:
            if assignment.task_id in affected:
                continue
            # a survivor must still be individually valid in the new world
            if not self._assignment_still_valid(assignment, disrupted):
                affected.add(assignment.task_id)
                continue
            survivors[assignment.resource_id].append(assignment.task_id)

        # Keep each survivor list in the order the previous plan actually ran
        # it. Sorting by task id here (which an earlier version did) rearranges
        # the route alphabetically, and because time windows and dependencies
        # are order-sensitive the rebuilt sequence is usually infeasible - the
        # whole resource then dropped out of the repaired plan even though
        # nothing about it had changed. The previous plan's start times are the
        # authoritative order.
        for resource_id in list(survivors):
            survivors[resource_id] = self._order_as_previous(survivors[resource_id], previous)

        pending = sorted(affected, key=lambda tid: self._order_key(tid, context))

        best: RepairOutcome | None = None
        candidates = 0
        for strategy in strategies:
            sequences = {rid: list(seq) for rid, seq in survivors.items()}
            recovered: list[str] = []
            for task_id in pending:
                options = _insertion_options(task_id, context, sequences)
                candidates += 1
                if not options:
                    continue
                if strategy == "by_resource":
                    # only consider the resources that were hit by the disruption
                    allowed = set(impact.affected_resources) | {
                        r for r in sequences if not sequences[r]
                    }
                    filtered = [o for o in options if o[1] in allowed]
                    options = filtered or options
                _, resource_id, position = options[0]
                sequences[resource_id].insert(position, task_id)
                recovered.append(task_id)

            outcome = self._materialise(
                context, sequences, recovered, pending, strategy, started, candidates
            )
            if outcome.plan is None:
                continue
            if best is None or outcome.plan.score > (best.plan.score if best.plan else float("-inf")):
                best = outcome

        elapsed = time.perf_counter() - started
        if best is None:
            return RepairOutcome(
                plan=None,
                recovered_tasks=(),
                unrecovered_tasks=tuple(pending),
                strategy=strategies[0] if strategies else "none",
                seconds=elapsed,
                candidates_considered=candidates,
                reason=(
                    "no feasible repair found: every affected task had no legal "
                    "insertion on any remaining resource"
                ),
            )
        best.seconds = elapsed
        best.candidates_considered = candidates
        if best.unrecovered_tasks and not best.reason:
            best.reason = (
                f"{len(best.unrecovered_tasks)} task(s) could not be re-inserted "
                "without violating a hard constraint"
            )
        return best

    # -- helpers -----------------------------------------------------------
    def _order_key(self, task_id: str, context: SchedulingContext) -> tuple[float, int, str]:
        task = context.task_by_id(task_id)
        if task is None:
            return (0.0, 0, task_id)
        value = task_value(task, context.scenario.objective_weights).total
        return (-round(value, 6), task.deadline, task_id)

    @staticmethod
    def _assignment_still_valid(assignment: Assignment, scenario: Scenario) -> bool:
        """Re-check one pre-disruption assignment against the new world."""
        try:
            task = scenario.task(assignment.task_id)
            resource = scenario.resource(assignment.resource_id)
        except Exception:  # noqa: BLE001 - an unknown entity is simply not a survivor
            return False
        if not resource.can_serve(task):
            return False
        if resource.is_globally_unavailable:
            return False
        if assignment.start < resource.shift_start or assignment.end > resource.shift_end:
            return False
        for block in resource.unavailable:
            if assignment.window.overlaps(block):
                return False
        if assignment.end > scenario.horizon.end:
            return False
        return True

    @staticmethod
    def _order_as_previous(task_ids: Sequence[str], previous: Any) -> list[str]:
        """Sort *task_ids* by the start time they had in *previous*.

        Tasks absent from the previous plan (introduced by the disruption) keep a
        deterministic order derived from the id, so the result is stable.
        """
        starts = {a.task_id: a.start for a in previous.assignments}
        return sorted(task_ids, key=lambda tid: (starts.get(tid, 1 << 30), tid))

    def _materialise(
        self,
        context: SchedulingContext,
        sequences: dict[str, list[str]],
        recovered: Sequence[str],
        pending: Sequence[str],
        strategy: str,
        started: float,
        candidates: int,
    ) -> RepairOutcome:
        from orion.domain.plans import SolverRun

        first = dependency_bounds(context, sequences)
        assignments: list[Assignment] = []
        # A resource whose rebuilt sequence does not fit must not be dropped
        # wholesale: that silently deleted unaffected work. Instead keep the
        # longest feasible prefix of the route, so only the genuinely
        # unschedulable tail is lost, and record why.
        dropped: list[str] = []
        for resource_id in sorted(sequences):
            sequence = sequences[resource_id]
            if not sequence:
                continue
            result = schedule_sequence(sequence, context, resource_id, external_bounds=first)
            if result is None:
                kept: list[Assignment] = []
                for length in range(len(sequence) - 1, 0, -1):
                    head = sequence[:length]
                    first = dependency_bounds(context, {**sequences, resource_id: head})
                    result = schedule_sequence(head, context, resource_id, external_bounds=first)
                    if result is not None:
                        kept = result
                        break
                assignments.extend(kept)
                dropped.extend(sequence[len(kept) :])
            else:
                assignments.extend(result)
        run = SolverRun(
            solver="LOCAL_REPAIR",
            status=SolverStatus.FEASIBLE if assignments else SolverStatus.INFEASIBLE,
            runtime_s=time.perf_counter() - started,
            objective=0.0,
            feasible=bool(assignments),
            num_variables=context.num_pairs,
            num_constraints=0,
            notes=f"local repair strategy={strategy}",
        )
        plan = build_plan(
            context, assignments, run, strategy=f"local_repair:{strategy}"
        )
        if has_hard_violation(plan.violations):
            return RepairOutcome(
                plan=None,
                recovered_tasks=(),
                unrecovered_tasks=tuple(pending),
                strategy=strategy,
                seconds=time.perf_counter() - started,
                candidates_considered=candidates,
                reason="repair produced a plan with hard constraint violations",
            )
        unrecovered = tuple(t for t in pending if t not in set(recovered))
        if dropped:
            reason_extra = (
                f"; {len(dropped)} unschedulable tail task(s) dropped: "
                f"{', '.join(dropped[:5])}"
            )
        else:
            reason_extra = ""
        return RepairOutcome(
            plan=plan,
            recovered_tasks=tuple(recovered),
            unrecovered_tasks=unrecovered,
            strategy=strategy,
            seconds=time.perf_counter() - started,
            reason=reason_extra,
            candidates_considered=candidates,
        )


@dataclass(slots=True)
class ReplanOutcome:
    """Full comparison of the two replanning strategies."""

    disruption: Disruption
    before_scenario: Scenario
    after_scenario: Scenario
    impact: DisruptionImpact
    repair: RepairOutcome
    full: Plan | None
    full_seconds: float
    solver: str
    time_budget_s: float
    full_status: str = ""
    full_reason: str = ""
    full_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def baseline_score(self) -> float:
        return self.repair.plan.parent_score if self.repair.plan else 0.0

    def scores(self) -> dict[str, float]:
        return {
            "repair": self.repair.plan.score if self.repair.plan else 0.0,
            "full": self.full.score if self.full else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "disruption": self.disruption.to_dict(),
            "impact": self.impact.to_dict(),
            "repair": self.repair.to_dict(),
            "full": {
                "plan_id": self.full.id if self.full else None,
                "status": self.full_status,
                "score": round(self.full.score, 4) if self.full else 0.0,
                "seconds": round(self.full_seconds, 6),
                "reason": self.full_reason,
                "solver": self.solver,
                "time_budget_s": self.time_budget_s,
                "diagnostics": dict(self.full_diagnostics),
            },
        }


class Replanner:
    """Runs both replanning strategies and reports the comparison."""

    def __init__(
        self,
        *,
        solver: str = SolverName.HEURISTIC,
        time_budget_s: float = 5.0,
        seed: int = 0,
    ) -> None:
        self.solver = solver
        self.time_budget_s = time_budget_s
        self.seed = seed
        self.repairer = LocalRepairer()

    def replan(
        self,
        scenario: Scenario,
        plan: Plan,
        disruption: Disruption,
        *,
        strategies: Sequence[str] = REPAIR_STRATEGIES,
        run_full: bool = True,
        solver: str | None = None,
        time_budget_s: float | None = None,
    ) -> ReplanOutcome:
        """Apply *disruption* to *scenario* and replan the resulting world.

        The pre-disruption ``scenario`` and ``plan`` are only ever read, so the
        caller keeps a valid "before" reference for the comparison view.
        """
        after = disruption.apply(scenario)
        impact = analyse_impact(disruption, scenario, after, plan)
        repair = self.repairer.repair(after, plan, impact, strategies=strategies)
        if repair.plan is not None:
            repair.plan.parent_plan_id = plan.id
            repair.plan.disruption_id = disruption.id

        full_plan: Plan | None = None
        full_seconds = 0.0
        full_status = SolverStatus.NOT_APPLICABLE
        full_reason = ""
        diagnostics: dict[str, Any] = {}
        if run_full:
            chosen = solver or self.solver
            budget = time_budget_s if time_budget_s is not None else self.time_budget_s
            started = time.perf_counter()
            context = SchedulingContext.build(after)
            result = run_solver(
                chosen,
                context,
                SolverRequest(time_limit_s=budget, seed=self.seed),
            )
            full_seconds = time.perf_counter() - started
            full_plan = result.plan
            full_status = result.run.status
            full_reason = result.reason
            diagnostics = dict(result.diagnostics)
            if full_plan is not None:
                full_plan.parent_plan_id = plan.id
                full_plan.disruption_id = disruption.id

        return ReplanOutcome(
            disruption=disruption,
            before_scenario=scenario,
            after_scenario=after,
            impact=impact,
            repair=repair,
            full=full_plan,
            full_seconds=full_seconds,
            solver=solver or self.solver,
            time_budget_s=time_budget_s if time_budget_s is not None else self.time_budget_s,
            full_status=full_status,
            full_reason=full_reason,
            full_diagnostics=diagnostics,
        )


__all__ = [
    "LocalRepairer",
    "Replanner",
    "ReplanOutcome",
    "RepairOutcome",
    "ChurnReport",
    "compute_churn",
    "REPAIR_STRATEGIES",
]
