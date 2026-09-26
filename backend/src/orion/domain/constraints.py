
"""Constraint catalogue.

The optimisation models in :mod:`orion.optimization` are generated from this
catalogue rather than written inline. Each entry states, in one place:

* the constraint's math;
* its type (hard/soft);
* which solver builders support it;
* the failure mode it exists to prevent.

Having a single catalogue is what makes the "number of constraints" figure in a
solver report honest: it is the count of *emitted rows*, computed from the same
list the builders iterate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ConstraintType(str):
    HARD = "HARD"
    SOFT = "SOFT"
    MIXED = "MIXED"


class SolverSupport(str):
    ALL = "ALL"
    CP_SAT_ONLY = "CP_SAT_ONLY"
    MILP_ONLY = "MILP_ONLY"
    EXACT_ONLY = "EXACT_ONLY"
    NOT_FOR_FLOW = "NOT_FOR_FLOW"


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """Documentation + metadata for one constraint family."""

    key: str
    name: str
    constraint_type: str
    support: str
    math: str
    rationale: str
    scenario_flag: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "key": self.key,
            "name": self.name,
            "type": self.constraint_type,
            "support": self.support,
            "math": self.math,
            "rationale": self.rationale,
            "scenario_flag": self.scenario_flag,
        }


CONSTRAINT_CATALOGUE: tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        key="capability",
        name="Skill / capability matching",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="x[t,r] = 0  for all r where C_t ⊄ K_r",
        rationale=(
            "A team without the required skill must never be assigned. Enforced by "
            "restricting the candidate pair set, so it costs no variables."
        ),
        scenario_flag="enforce_capability",
    ),
    ConstraintSpec(
        key="capacity",
        name="Resource capacity",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="d[t,k] <= cap[r,k]  for all assigned (t,r) and demand dimension k",
        rationale=(
            "A vehicle carrying 500 kg cannot take a 900 kg load. Enforced by candidate "
            "filtering plus a per-assignment check in validate_plan."
        ),
        scenario_flag="enforce_capacity",
    ),
    ConstraintSpec(
        key="assignment_uniqueness",
        name="Each task assigned at most once",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="sum_r x[t,r] <= 1   for all tasks t",
        rationale=(
            "Prevents a plan from 'solving' a scenario by duplicating effort. This is the "
            "constraint that makes the completion count meaningful."
        ),
    ),
    ConstraintSpec(
        key="resource_conflict",
        name="No-overlap on a resource",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.EXACT_ONLY,
        math="s[t] + p[t] <= s[t']   for consecutive tasks t, t' on resource r",
        rationale=(
            "The core scheduling constraint. Exact solvers model it with interval / "
            "disjunctive variables; the heuristic enforces it by construction (a "
            "sequence is a list)."
        ),
    ),
    ConstraintSpec(
        key="release_time",
        name="Release time / time window",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="s[t] >= r[t]",
        rationale="Work cannot start before the task is released or the crew is on shift.",
        scenario_flag="enforce_release_times",
    ),
    ConstraintSpec(
        key="deadline",
        name="Deadline (soft by default)",
        constraint_type=ConstraintType.SOFT,
        support=SolverSupport.ALL,
        math="L[t] >= s[t] + p[t] - d[t],  L[t] >= 0,  penalised by w_late",
        rationale=(
            "Soft because a hard deadline turns an entire scenario class infeasible and "
            "hides the planning behaviour that matters. A scenario can make a specific "
            "task's deadline hard via max_lateness."
        ),
        scenario_flag="enforce_deadlines",
    ),
    ConstraintSpec(
        key="dependencies",
        name="Task precedence",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.EXACT_ONLY,
        math="s[t] >= s[t'] + p[t']   for each prerequisite t' of t",
        rationale=(
            "A repair cannot start before the inspection that authorises it. Acyclicity "
            "is validated at scenario construction, so this is a DAG constraint."
        ),
        scenario_flag="enforce_dependencies",
    ),
    ConstraintSpec(
        key="maintenance",
        name="Maintenance / downtime windows",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="x[t,r] = 0  for t whose window intersects any block in U_r",
        rationale=(
            "Modelled as forbidden intervals rather than a separate variable: a resource "
            "in maintenance simply has a smaller feasible window."
        ),
        scenario_flag="enforce_maintenance",
    ),
    ConstraintSpec(
        key="shift",
        name="Shift bounds",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.ALL,
        math="shift_start[r] <= s[t] and s[t]+p[t] <= shift_end[r]",
        rationale="Nobody works outside their rostered hours.",
        scenario_flag="enforce_shift",
    ),
    ConstraintSpec(
        key="max_work",
        name="Maximum working time",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.EXACT_ONLY,
        math="sum_{t on r} p[t] <= max_work[r]",
        rationale=(
            "Fatigue limit. A cumulative knapsack constraint; only the exact solvers emit "
            "it, which is a real modelling difference the report discusses."
        ),
        scenario_flag="enforce_max_work",
    ),
    ConstraintSpec(
        key="travel",
        name="Travel time between consecutive tasks",
        constraint_type=ConstraintType.HARD,
        support=SolverSupport.EXACT_ONLY,
        math="s[t'] >= s[t] + p[t] + tau(r, loc(t), loc(t'))",
        rationale=(
            "The sequencing constraint. Modelled with arc variables z[t,t',r] in CP-SAT and "
            "with a big-M disjunctive linearisation in MILP."
        ),
        scenario_flag="enforce_travel",
    ),
    ConstraintSpec(
        key="dispatch",
        name="Dispatched / idle resources",
        constraint_type=ConstraintType.SOFT,
        support=SolverSupport.ALL,
        math="penalise  y[r] = 1  when resource r is used at all",
        rationale=(
            "Sending a crew out has a fixed cost, so a marginal task far away may not be "
            "worth the dispatch. This is what stops the solver from using every resource."
        ),
    ),
    ConstraintSpec(
        key="cost",
        name="Operating cost",
        constraint_type=ConstraintType.SOFT,
        support=SolverSupport.ALL,
        math="penalise  operating_cost[r] * (worked_minutes[r] / 60) + distance-based travel",
        rationale="Cheap resources should be preferred when service is otherwise equal.",
    ),
    ConstraintSpec(
        key="unassigned",
        name="Unassigned-task allowance",
        constraint_type=ConstraintType.MIXED,
        support=SolverSupport.ALL,
        math="sum_{t in T} u[t] <= ceil(max_unassigned_fraction * |T|)",
        rationale=(
            "Real planners cannot serve everything. Bounding how many tasks may be dropped "
            "is what stops the objective from buying service with feasibility."
        ),
    ),
)

CONSTRAINT_INDEX: Mapping[str, ConstraintSpec] = {c.key: c for c in CONSTRAINT_CATALOGUE}

#: Constraints the min-cost-flow relaxation cannot represent. The capability
#: matrix in the README and the report is generated from this, not typed by hand.
FLOW_INCOMPATIBLE = frozenset(
    spec.key for spec in CONSTRAINT_CATALOGUE if spec.support == SolverSupport.EXACT_ONLY
)


def active_constraints(flags: Mapping[str, bool]) -> tuple[ConstraintSpec, ...]:
    """Return the catalogue entries enabled by a scenario's constraint flags."""
    active: list[ConstraintSpec] = []
    for spec in CONSTRAINT_CATALOGUE:
        if spec.scenario_flag is None:
            active.append(spec)
            continue
        if flags.get(spec.scenario_flag, True):
            active.append(spec)
    return tuple(active)


def count_active(flags: Mapping[str, bool]) -> int:
    return len(active_constraints(flags))


__all__ = [
    "ConstraintType",
    "SolverSupport",
    "ConstraintSpec",
    "CONSTRAINT_CATALOGUE",
    "CONSTRAINT_INDEX",
    "FLOW_INCOMPATIBLE",
    "active_constraints",
    "count_active",
]
