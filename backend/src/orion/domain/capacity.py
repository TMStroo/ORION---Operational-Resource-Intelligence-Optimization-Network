
"""Capability, priority and capacity vocabulary.

Capabilities are free-form strings so a scenario can describe any operational
domain ("electrician", "crane_heavy", "refrigerated") without ORION carrying a
built-in taxonomy. Priority, however, is a fixed vocabulary because it enters
the objective function, and silently accepting an arbitrary number would make
objective weights uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Iterable


class Priority(IntEnum):
    """Task priority. Higher value means more important.

    The numeric gaps are meaningful: the default objective weights a priority
    class roughly 10x the one below it, so the gaps are deliberately uniform.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()


def parse_priority(value: object) -> Priority:
    """Accept ``Priority``, an int 1-4, or a case-insensitive name."""
    if isinstance(value, Priority):
        return value
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"invalid priority {value!r}")
    if isinstance(value, int):
        try:
            return Priority(value)
        except ValueError as exc:
            raise ValueError(
                f"invalid priority {value!r}, expected one of "
                f"{[p.value for p in Priority]}"
            ) from exc
    if isinstance(value, str):
        key = value.strip().upper()
        if key in Priority.__members__:
            return Priority[key]
    raise ValueError(f"invalid priority {value!r}")


def normalize_capabilities(values: Iterable[str] | None) -> frozenset[str]:
    """Normalise a capability collection to a lowercase frozenset.

    Normalisation is case-folding plus whitespace trimming only. ORION never
    invents synonyms: "crane" and "crane_heavy" are different capabilities, and
    a scenario that wants them merged must say so.
    """
    if values is None:
        return frozenset()
    return frozenset(v.strip().lower() for v in values if v and v.strip())


def missing_capabilities(required: frozenset[str], available: frozenset[str]) -> frozenset[str]:
    """Return the capabilities in *required* that *available* does not cover."""
    return required - available


def has_all_capabilities(required: frozenset[str], available: frozenset[str]) -> bool:
    return required <= available


@dataclass(frozen=True, slots=True)
class CapacityDemand:
    """A resource capacity requirement attached to a task.

    ``amount`` is in abstract capacity units (litres, kilograms, crew seats,
    concurrent circuits - the scenario decides the physical meaning).
    """

    dimension: str
    amount: float

    def __post_init__(self) -> None:
        if not self.dimension.strip():
            raise ValueError("capacity dimension must be a non-empty string")
        if self.amount < 0:
            raise ValueError(f"capacity amount {self.amount} must be >= 0")

    def describe(self) -> str:
        return f"{self.amount:g} {self.dimension}"


DEFAULT_CAPACITY_DIMENSION: Final[str] = "units"

__all__ = [
    "Priority",
    "parse_priority",
    "normalize_capabilities",
    "missing_capabilities",
    "has_all_capabilities",
    "CapacityDemand",
    "DEFAULT_CAPACITY_DIMENSION",
]
