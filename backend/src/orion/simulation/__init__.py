"""Simulation layer: discrete-event execution and disruption injection."""

from __future__ import annotations

from orion.simulation.disruptions import generate_disruptions, severity_params
from orion.simulation.engine import (
    ReplanCallback,
    ResourceState,
    SimulationEngine,
    SimulationResult,
    TaskState,
    WorldState,
)
from orion.simulation.events import (
    EVENT_LABELS,
    EventQueue,
    EventType,
    SimEvent,
)

__all__ = [
    "SimulationEngine",
    "generate_disruptions",
    "severity_params",
    "SimulationResult",
    "WorldState",
    "ResourceState",
    "TaskState",
    "ReplanCallback",
    "SimEvent",
    "EventType",
    "EventQueue",
    "EVENT_LABELS",
]
