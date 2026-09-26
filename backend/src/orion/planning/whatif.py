
"""What-if analysis.

A what-if question ("what if vehicle availability falls by 20%?") is answered by
building a *modified* scenario, planning it under identical conditions, and
comparing against the baseline. The modification is explicit and inspectable -
ORION does not apply a hidden transformation, it produces a new scenario whose
difference from the baseline is listed in the result.

Every operator validates its input and raises on nonsense (a negative
percentage, a resource that does not exist), because a what-if that silently
does nothing is worse than an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from orion.domain.entities import Resource, Scenario
from orion.domain.errors import UnknownEntityError, ValidationError
from orion.domain.plans import Plan, SolverStatus
from orion.planning.comparison import PlanComparison, compare_plans
from orion.planning.planner import Planner


def op_availability_drop(scenario: Scenario, fraction: float, *, resource_ids: Sequence[str] | None = None) -> tuple[Scenario, str]:
    """Take a fraction of resources out of service for the whole horizon."""
    if not 0.0 <= fraction < 1.0:
        raise ValidationError(
            f"availability drop fraction must be in [0, 1), got {fraction}"
        )
    resources = list(scenario.resources)
    if resource_ids:
        known = {r.id for r in resources}
        for rid in resource_ids:
            if rid not in known:
                raise UnknownEntityError(f"scenario {scenario.id} has no resource {rid!r}")
        targets = [r for r in resources if r.id in set(resource_ids)]
    else:
        # deterministic selection: lowest-cost-capable first, then by id
        targets = sorted(resources, key=lambda r: (r.operating_cost_per_hour, r.id))[
            : max(1, int(round(len(resources) * fraction)))
        ]
    if not targets:
        raise ValidationError("availability drop selected no resources")
    target_ids = {r.id for r in targets}
    from orion.domain.time_model import Interval

    def _block(r: Resource) -> Resource:
        # Block exactly the rostered window, not the whole horizon: a block
        # longer than the shift is rejected by Resource validation, and blocking
        # outside the shift would be meaningless anyway.
        return replace(
            r,
            unavailable=tuple(
                sorted(set(r.unavailable) | {Interval(r.shift_start, r.shift_end)})
            ),
        )

    new_resources = tuple(
        _block(r) if r.id in target_ids else r for r in resources
    )
    out = _replace_resources(scenario, new_resources)
    return out, (
        f"removed {len(targets)} of {len(resources)} resources from service: "
        f"{', '.join(sorted(target_ids))}"
    )


def _replace_resources(scenario: Scenario, resources: Sequence[Resource]) -> Scenario:
    from dataclasses import replace as _replace

    return _replace(scenario, resources=tuple(resources))


def op_demand_increase(scenario: Scenario, fraction: float, *, seed: int = 0) -> tuple[Scenario, str]:
    """Add new urgent tasks, scaling the existing task set by *fraction*."""
    if fraction <= 0:
        raise ValidationError(f"demand increase fraction must be > 0, got {fraction}")
    import random

    rng = random.Random(seed or scenario.metadata.get("seed", 0) or 0)
    from orion.domain.capacity import Priority
    from orion.domain.entities import Task
    from orion.domain.ids import format_id

    existing_ids = {t.id for t in scenario.tasks}
    locations = [loc for loc in scenario.travel.locations if loc != scenario.depot]
    extra_count = max(1, int(round(len(scenario.tasks) * fraction)))
    new_tasks = []
    next_index = len(scenario.tasks) + 1
    for _ in range(extra_count):
        index = next_index
        next_index += 1
        task_id = format_id("T", index)
        while task_id in existing_ids:
            task_id = format_id("T", next_index)
            next_index += 1
        existing_ids.add(task_id)
        template = rng.choice(scenario.tasks)
        duration = template.duration
        release = min(template.release_time, scenario.horizon.end - duration - 1)
        new_tasks.append(
            Task(
                id=task_id,
                location=rng.choice(locations) if locations else scenario.depot,
                release_time=release,
                deadline=min(release + duration + 90, scenario.horizon.end),
                duration=duration,
                priority=Priority.HIGH,
                required_capabilities=template.required_capabilities,
                required_capacity=template.required_capacity,
            )
        )
    out = _replace_tasks(scenario, new_tasks)
    return out, f"added {len(new_tasks)} new high-priority task(s)"


def _replace_tasks(scenario: Scenario, tasks: Sequence[Any]) -> Scenario:
    from dataclasses import replace as _replace

    return _replace(scenario, tasks=tuple(scenario.tasks) + tuple(tasks))


def op_deadline_tighten(scenario: Scenario, fraction: float) -> tuple[Scenario, str]:
    """Pull every deadline earlier by *fraction* of its slack."""
    if not 0.0 < fraction <= 1.0:
        raise ValidationError(
            f"deadline tightening fraction must be in (0, 1], got {fraction}"
        )
    from dataclasses import replace as _replace

    new_tasks = []
    tightened = 0
    for task in scenario.tasks:
        slack = max(0, task.deadline - task.release_time - task.duration)
        reduction = int(round(slack * fraction))
        new_deadline = max(task.release_time + task.duration, task.deadline - reduction)
        if new_deadline != task.deadline:
            tightened += 1
        new_tasks.append(replace(task, deadline=new_deadline))
    out = _replace(scenario, tasks=tuple(new_tasks))
    return out, f"tightened {tightened}/{len(scenario.tasks)} deadlines by {fraction * 100:.0f}% of their slack"


def op_resource_outage(scenario: Scenario, resource_id: str) -> tuple[Scenario, str]:
    """Take one named resource out of service."""
    from dataclasses import replace as _replace
    from orion.domain.time_model import Interval

    for resource in scenario.resources:
        if resource.id != resource_id:
            continue
        blocked = replace(
            resource,
            unavailable=tuple(
                sorted(
                    set(resource.unavailable)
                    | {Interval(resource.shift_start, resource.shift_end)}
                )
            ),
        )
        resources = tuple(blocked if r.id == resource_id else r for r in scenario.resources)
        out = _replace(scenario, resources=resources)
        return out, f"resource {resource_id} unavailable for the whole horizon"
    raise UnknownEntityError(f"scenario {scenario.id} has no resource {resource_id!r}")


def op_travel_increase(scenario: Scenario, factor: float) -> tuple[Scenario, str]:
    """Increase all travel times by *factor*."""
    if factor < 1.0:
        raise ValidationError(
            f"travel increase factor must be >= 1.0, got {factor}"
        )
    out = scenario.with_travel(scenario.travel.scaled(factor))
    return out, f"all travel times multiplied by {factor:.2f}"


def op_capacity_reduction(scenario: Scenario, factor: float, *, resource_ids: Sequence[str] | None = None) -> tuple[Scenario, str]:
    """Reduce every resource's capacity in every dimension by *factor*."""
    if not 0.0 <= factor <= 1.0:
        raise ValidationError(f"capacity factor must be in [0, 1], got {factor}")
    from dataclasses import replace as _replace

    resources = []
    for resource in scenario.resources:
        if resource_ids and resource.id not in set(resource_ids):
            resources.append(resource)
            continue
        new_capacity = {k: v * factor for k, v in resource.capacity.items()}
        resources.append(replace(resource, capacity=new_capacity))
    out = _replace(scenario, resources=tuple(resources))
    n = len(resource_ids) if resource_ids else len(scenario.resources)
    return out, f"capacity of {n} resource(s) reduced to {factor * 100:.0f}%"


#: Registry of what-if operators, used by the API and the CLI.
WHAT_IF_OPERATORS: dict[str, Callable[..., tuple[Scenario, str]]] = {
    "availability_drop": op_availability_drop,
    "demand_increase": op_demand_increase,
    "deadline_tighten": op_deadline_tighten,
    "resource_outage": op_resource_outage,
    "travel_increase": op_travel_increase,
    "capacity_reduction": op_capacity_reduction,
}

WHAT_IF_DESCRIPTIONS: Mapping[str, str] = {
    "availability_drop": "What if resource availability falls by X%?",
    "demand_increase": "What if demand rises by X%?",
    "deadline_tighten": "What if deadlines become X% tighter?",
    "resource_outage": "What if one named resource becomes unavailable?",
    "travel_increase": "What if travel times increase by factor X?",
    "capacity_reduction": "What if resource capacity falls to X%?",
}


@dataclass(slots=True)
class WhatIfResult:
    """One what-if question, answered against a baseline plan."""

    operator: str
    question: str
    parameter: float | str
    modification: str
    baseline_plan: Plan
    scenario_plan: Plan
    comparison: PlanComparison
    seconds: float
    status: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "question": self.question,
            "parameter": self.parameter,
            "modification": self.modification,
            "seconds": round(self.seconds, 6),
            "status": self.status,
            "comparison": self.comparison.to_dict(),
        }

    def render(self) -> str:
        return (
            f"{self.question}\n"
            f"  {self.modification}\n"
            f"  status={self.status} in {self.seconds * 1000:.0f} ms\n"
            f"{_indent(self.comparison.render(), '  ')}"
        )


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class WhatIfEngine:
    """Runs what-if questions against a baseline plan."""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def available_operators(self) -> list[dict[str, str]]:
        return [
            {"operator": name, "question": desc}
            for name, desc in sorted(WHAT_IF_DESCRIPTIONS.items())
        ]

    def run(
        self,
        scenario: Scenario,
        baseline: Plan,
        operator: str,
        parameter: float | str,
        *,
        solver: str | None = None,
        time_budget_s: float | None = None,
        seed: int = 0,
    ) -> WhatIfResult:
        """Apply *operator* and plan the modified scenario."""
        if operator not in WHAT_IF_OPERATORS:
            raise ValidationError(
                f"unknown what-if operator {operator!r}; available: "
                f"{sorted(WHAT_IF_OPERATORS)}"
            )
        started = time.perf_counter()
        if operator == "resource_outage":
            if not isinstance(parameter, str) or not parameter:
                raise ValidationError(
                    "resource_outage requires a resource id string parameter"
                )
            modified, description = WHAT_IF_OPERATORS[operator](scenario, parameter)
            question = f"What if {parameter} becomes unavailable?"
        else:
            try:
                value = float(parameter)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"operator {operator!r} requires a numeric parameter, got "
                    f"{parameter!r}"
                ) from exc
            if operator == "demand_increase":
                modified, description = WHAT_IF_OPERATORS[operator](
                    scenario, value, seed=seed
                )
            elif operator == "availability_drop":
                modified, description = WHAT_IF_OPERATORS[operator](scenario, value)
            else:
                modified, description = WHAT_IF_OPERATORS[operator](scenario, value)
            # Every operator takes a *fraction* (0.15 == 15 %), but the question
            # templates are written in percent. Substituting the raw fraction
            # produced "deadlines become 0.15% tighter", so scale here. The one
            # exception is travel_increase, whose parameter is a multiplier
            # (1.25 == "by factor 1.25"), not a percentage.
            shown = f"{value:.2f}".rstrip("0").rstrip(".") if operator == "travel_increase" else f"{value * 100:g}"
            question = WHAT_IF_DESCRIPTIONS[operator].replace("X", shown)

        result = self.planner.plan(
            modified,
            solver=solver,
            time_budget_s=time_budget_s,
            seed=seed,
            explain=False,
        )
        comparison = compare_plans(baseline, result.plan, modified)
        return WhatIfResult(
            operator=operator,
            question=question,
            parameter=parameter,
            modification=description,
            baseline_plan=baseline,
            scenario_plan=result.plan,
            comparison=comparison,
            seconds=time.perf_counter() - started,
            status=result.plan.status,
        )

    def run_many(
        self,
        scenario: Scenario,
        baseline: Plan,
        questions: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> list[WhatIfResult]:
        out: list[WhatIfResult] = []
        for question in questions:
            out.append(
                self.run(
                    scenario,
                    baseline,
                    str(question["operator"]),
                    question.get("parameter", 0.0),  # type: ignore[arg-type]
                    **kwargs,
                )
            )
        return out


__all__ = [
    "WhatIfEngine",
    "WhatIfResult",
    "WHAT_IF_OPERATORS",
    "WHAT_IF_DESCRIPTIONS",
    "op_availability_drop",
    "op_demand_increase",
    "op_deadline_tighten",
    "op_resource_outage",
    "op_travel_increase",
    "op_capacity_reduction",
]
