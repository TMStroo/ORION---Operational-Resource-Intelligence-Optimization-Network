"""Reproducible disruption generation.

Disruptions are part of the *experiment design*, not of the domain model, so
they live here rather than in ``domain/``. Generation is seeded and
deterministic: the same (scenario, rate, severity, seed) always yields the same
event list, which is what makes the disruption stress test reproducible.

Severity maps to concrete, documented magnitudes rather than to a vague label,
because "high severity" has to mean something a reader can check:

======== ======= ================== ================================
severity count    vehicle failure    deadline / capacity shift
======== ======= ================== ================================
low      1        out for ~20% of     deadline +5%
                 the remaining day
medium   1        out for ~40%        deadline +10%
high     1        out for the rest    deadline +20%
                 of the horizon
======== ======= ================== ================================

Every disruption also gets a second, independent kind so the suite is not a
single-mode test: resource unavailability (blocked shift), maintenance
(blocked window) and demand surge (new urgent tasks).
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from orion.domain.entities import ResourceKind, Scenario
from orion.domain.events import Disruption

#: severity -> (share of horizon removed, deadline inflation, count multiplier)
SEVERITY_TABLE: dict[str, tuple[float, float, int]] = {
    "low": (0.2, 0.05, 1),
    "medium": (0.4, 0.10, 2),
    "high": (0.70, 0.20, 3),
}

DISRUPTION_KINDS = (
    "VEHICLE_FAILURE",
    "RESOURCE_UNAVAILABLE",
    "MAINTENANCE",
    "DEMAND_SURGE",
    "TRAVEL_INCREASE",
)


def severity_params(severity: str) -> tuple[float, float, int]:
    try:
        return SEVERITY_TABLE[severity]
    except KeyError:
        raise ValueError(
            f"unknown severity {severity!r}; expected one of {sorted(SEVERITY_TABLE)}"
        ) from None


def generate_disruptions(
    scenario: Scenario,
    *,
    rate: float,
    severity: str = "medium",
    seed: int = 0,
    kinds: Sequence[str] = DISRUPTION_KINDS,
    horizon_fraction: float = 0.25,
) -> tuple[Disruption, ...]:
    """Generate a deterministic list of disruptions for *scenario*.

    ``rate`` is the *share of resources* hit (plus a demand surge when the
    random draw fires), so a rate of 0.2 on 10 resources yields roughly two
    affected resources - the parameter is interpretable rather than a
    probability over an unbounded event stream.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be within [0, 1], got {rate}")
    share, deadline_inflation, multiplier = severity_params(severity)
    rng = random.Random((seed * 1_000_003) ^ hash(scenario.id) & 0xFFFFFFFF)
    if rate == 0.0:
        return ()

    resources = list(scenario.resources)
    if not resources:
        return ()
    count = max(1, round(len(resources) * rate * multiplier / 2))
    count = min(count, len(resources))
    chosen = rng.sample(resources, count)

    earliest = scenario.horizon.start + int((scenario.horizon.end - scenario.horizon.start) * horizon_fraction)
    span = scenario.horizon.end - scenario.horizon.start
    out: list[Disruption] = []

    for index, resource in enumerate(chosen):
        kind = kinds[index % len(kinds)]
        when = earliest + rng.randrange(0, max(1, scenario.horizon.end - earliest))
        if kind == "DEMAND_SURGE":
            new_tasks = max(1, round(len(scenario.tasks) * 0.2 * multiplier))
            out.append(
                Disruption(
                    id=f"{scenario.id}-D{index + 1:02d}",
                    type="NEW_TASK",
                    timestamp=when,
                    magnitude=float(new_tasks),
                    duration=None,
                    description=(
                        f"{new_tasks} new urgent task(s) arrive at "
                        f"{when // 60:02d}:{when % 60:02d}"
                    ),
                    severity=severity.upper(),
                    payload={"count": new_tasks, "priority": "HIGH", "kind": kind},
                )
            )
            continue
        if kind == "TRAVEL_INCREASE":
            out.append(
                Disruption(
                    id=f"{scenario.id}-D{index + 1:02d}",
                    type="TRAVEL_INCREASE",
                    timestamp=when,
                    target_id=None,
                    magnitude=1.0 + deadline_inflation * 2,
                    duration=span,
                    description=(
                        f"congestion multiplies all travel times by "
                        f"{1.0 + deadline_inflation * 2:.2f} from "
                        f"{when // 60:02d}:{when % 60:02d}"
                    ),
                    severity=severity.upper(),
                    payload={"factor": 1.0 + deadline_inflation * 2, "kind": kind},
                )
            )
            continue

        if kind == "MAINTENANCE":
            duration = int(span * share * 0.5)
            out.append(
                Disruption(
                    id=f"{scenario.id}-D{index + 1:02d}",
                    type="MAINTENANCE",
                    timestamp=when,
                    target_id=resource.id,
                    magnitude=1.0,
                    duration=max(1, duration),
                    description=(
                        f"{resource.id} enters maintenance for {max(1, duration)} min from "
                        f"{when // 60:02d}:{when % 60:02d}"
                    ),
                    severity=severity.upper(),
                    payload={"kind": kind},
                )
            )
            continue

        if kind == "RESOURCE_UNAVAILABLE":
            out.append(
                Disruption(
                    id=f"{scenario.id}-D{index + 1:02d}",
                    type="RESOURCE_UNAVAILABLE",
                    timestamp=when,
                    target_id=resource.id,
                    magnitude=1.0,
                    duration=None,
                    description=(
                        f"{resource.id} ({resource.kind.value}) is unavailable from "
                        f"{when // 60:02d}:{when % 60:02d} onward"
                    ),
                    severity=severity.upper(),
                    payload={"kind": kind},
                )
            )
            continue

        # default: VEHICLE_FAILURE
        duration = int(span * share)
        out.append(
            Disruption(
                id=f"{scenario.id}-D{index + 1:02d}",
                type="VEHICLE_FAILURE",
                timestamp=when,
                target_id=resource.id,
                magnitude=1.0,
                duration=max(1, min(duration, scenario.horizon.end - when)),
                description=(
                    f"{resource.id} fails at {when // 60:02d}:{when % 60:02d} and is out "
                    f"of service for {max(1, min(duration, scenario.horizon.end - when))} min"
                ),
                severity=severity.upper(),
                payload={"kind": kind, "resource_kind": resource.kind.value},
            )
        )
    return tuple(out)
