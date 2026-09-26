"""Typed error hierarchy.

ORION never converts a failure into a successful-looking result. Every
condition listed in the project failure-mode inventory has either a specific
exception here or an explicit solver/simulation status.
"""

from __future__ import annotations


class OrionError(Exception):
    """Base class for every error raised by ORION."""


# --------------------------------------------------------------------------
# Domain / validation errors
# --------------------------------------------------------------------------
class DomainError(OrionError):
    """Base class for invalid domain objects."""


class ValidationError(DomainError):
    """A domain object failed structural validation."""


class DuplicateEntityError(DomainError):
    """Two entities share an identifier inside the same collection."""


class UnknownEntityError(DomainError):
    """A reference points at an entity that does not exist."""


class InfeasibleScenarioError(DomainError):
    """The scenario is structurally impossible to satisfy."""


class TravelGraphError(DomainError):
    """The travel graph is disconnected, empty, or contains invalid edges."""


# --------------------------------------------------------------------------
# Solver errors
# --------------------------------------------------------------------------
class SolverError(OrionError):
    """Base class for solver failures."""


class SolverUnavailableError(SolverError):
    """A requested solver backend is not installed or not supported here."""


class SolverBuildError(SolverError):
    """The optimisation model could not be constructed for this scenario."""


class TimeBudgetError(SolverError):
    """The time budget is missing, negative, or otherwise unusable."""


# --------------------------------------------------------------------------
# Planning / replanning errors
# --------------------------------------------------------------------------
class PlanningError(OrionError):
    """Base class for planning failures."""


class NoPlanError(PlanningError):
    """An operation requiring an existing plan was invoked without one."""


class RepairFailedError(PlanningError):
    """Local repair could not produce a feasible plan."""


# --------------------------------------------------------------------------
# Simulation errors
# --------------------------------------------------------------------------
class SimulationError(OrionError):
    """Base class for simulation failures."""


class SimulationStateError(SimulationError):
    """An event referenced a world state that does not exist."""


# --------------------------------------------------------------------------
# Experiment / provenance errors
# --------------------------------------------------------------------------
class ExperimentError(OrionError):
    """Base class for experiment-store errors."""


class ExperimentExistsError(ExperimentError):
    """An experiment with this id already exists. Experiments are immutable."""


class ExperimentNotFoundError(ExperimentError):
    """No experiment with the requested id exists."""


class ProvenanceError(ExperimentError):
    """Required provenance metadata is missing or incomplete."""


# --------------------------------------------------------------------------
# Configuration errors
# --------------------------------------------------------------------------
class ConfigError(OrionError):
    """A configuration file is missing, malformed, or semantically invalid."""


# --------------------------------------------------------------------------
# Export / reporting errors
# --------------------------------------------------------------------------
class ExportError(OrionError):
    """An export or report could not be produced."""


__all__ = [
    "OrionError",
    "DomainError",
    "ValidationError",
    "DuplicateEntityError",
    "UnknownEntityError",
    "InfeasibleScenarioError",
    "TravelGraphError",
    "SolverError",
    "SolverUnavailableError",
    "SolverBuildError",
    "TimeBudgetError",
    "PlanningError",
    "NoPlanError",
    "RepairFailedError",
    "SimulationError",
    "SimulationStateError",
    "ExperimentError",
    "ExperimentExistsError",
    "ExperimentNotFoundError",
    "ProvenanceError",
    "ConfigError",
    "ExportError",
]
