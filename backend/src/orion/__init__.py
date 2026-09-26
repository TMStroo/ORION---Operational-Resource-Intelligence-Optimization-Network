"""ORION - Operational Resource Intelligence & Optimization Network.

ORION is a decision-support platform for allocating limited operational
resources to time-sensitive tasks under constraints, and for recovering an
existing plan when the operating environment changes.

The package is organised in explicit domain modules:

``orion.domain``
    Entities, constraints, plans and events. Pure data, no solver coupling.
``orion.optimization``
    Explicit mathematical formulations plus five solution methods.
``orion.simulation``
    Discrete-event execution of a plan and the disruption framework.
``orion.planning``
    Planner, local repair, full replanning and plan comparison.
``orion.data``
    Scenario generation/loading and public benchmark adapters.
``orion.evaluation``
    Metrics, scalability studies and failure analysis.
``orion.experiments``
    Immutable experiment store with full provenance.
``orion.api``
    FastAPI application and typed schemas.
``orion.cli``
    Command line interface covering every major workflow.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "ORION_NAME",
    "SCHEMA_VERSION",
]

__version__ = "0.1.0"

ORION_NAME = "Operational Resource Intelligence & Optimization Network"

#: Version of the persisted domain schema. Bumping this invalidates stored
#: experiment manifests, which is the point: provenance must never silently
#: mix incompatible artefacts.
SCHEMA_VERSION = "1.0.0"
