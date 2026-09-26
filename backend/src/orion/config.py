"""Typed, validated experiment configuration.

Every experiment in ORION is config-driven: nothing that changes a number is a
Python literal. This module is the only place a config is interpreted, and it
**fails loudly** on anything malformed - an unknown solver, a negative time
budget, a scenario level that does not exist, an objective weight that would
invert the objective. A silent default is how benchmark numbers become
unreproducible, so every field is required unless a default is genuinely
neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from orion.domain.entities import ObjectiveWeights
from orion.domain.errors import ConfigError
from orion.optimization.registry import ALL_SOLVERS, SOLVER_INDEX

DIFFICULTY_LEVELS = ("small", "medium", "large", "stress")
SPLITS = ("dev", "val", "eval")

#: How the objective weights were chosen, recorded in every manifest so a
#: reader can tell tuned weights from hand-set ones.
WEIGHT_PROVENANCE = {
    "default": "calibrated defaults from ObjectiveWeights (see its calibration rule)",
    "constraint_pressure": "deliberately swept; see the constraint-pressure experiment",
    "reported": "the scenario's own weights, used for headline numbers",
}


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """How to build a synthetic operational scenario."""

    level: str = "medium"
    seed: int = 2026
    maintenance_count: int = 2
    scarcity: float = 1.0
    deadline_tightness: float = 0.5
    dependency_rate: float = 0.15

    def validate(self) -> None:
        if self.level not in DIFFICULTY_LEVELS:
            raise ConfigError(
                f"scenario.level must be one of {DIFFICULTY_LEVELS}, got {self.level!r}"
            )
        if self.maintenance_count < 0:
            raise ConfigError("scenario.maintenance_count must be >= 0")
        for name in ("scarcity", "deadline_tightness", "dependency_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"scenario.{name} must be within [0, 1], got {value}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError(f"scenario.seed must be a non-negative int, got {self.seed!r}")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """One public benchmark suite entry."""

    name: str
    dataset: str  # "solomon" | "jobshop" | "synthetic"
    instances: tuple[str, ...] = ()
    split: str = "dev"
    fleet_multiplier: float = 1.0

    def validate(self) -> None:
        if self.dataset not in {"solomon", "jobshop", "synthetic"}:
            raise ConfigError(
                f"benchmark.dataset must be solomon|jobshop|synthetic, got {self.dataset!r}"
            )
        if self.split not in SPLITS:
            raise ConfigError(f"benchmark.split must be one of {SPLITS}, got {self.split!r}")
        if self.fleet_multiplier <= 0:
            raise ConfigError("benchmark.fleet_multiplier must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "instances": list(self.instances),
            "split": self.split,
            "fleet_multiplier": self.fleet_multiplier,
        }


@dataclass(frozen=True, slots=True)
class DisruptionConfig:
    """Disruption generation for the stress tests."""

    rate: float = 0.0
    severity: str = "medium"
    seed: int = 7
    #: Generator-local labels, deliberately the five kinds the stress suite
    #: sweeps. These are mapped onto the domain's DisruptionType vocabulary by
    #: `orion.simulation.disruptions`, which is the only place that knows both.
    kinds: tuple[str, ...] = (
        "VEHICLE_FAILURE",
        "RESOURCE_UNAVAILABLE",
        "MAINTENANCE",
        "DEMAND_SURGE",
        "TRAVEL_INCREASE",
    )
    #: How many disruptions of each generated set are actually applied. The
    #: suite applies a deterministic, evenly-spread sample rather than a flat
    #: prefix slice, so raising `rate` increases coverage instead of only
    #: reordering which disruptions get measured.
    sample_per_rate: int = 4

    def validate(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ConfigError(f"disruption.rate must be within [0, 1], got {self.rate}")
        if self.severity not in ("low", "medium", "high"):
            raise ConfigError(
                f"disruption.severity must be low|medium|high, got {self.severity!r}"
            )
        if not self.kinds:
            raise ConfigError("disruption.kinds must not be empty")
        if self.sample_per_rate < 1:
            raise ConfigError(
                f"disruption.sample_per_rate must be >= 1, got {self.sample_per_rate}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "severity": self.severity,
            "seed": self.seed,
            "kinds": list(self.kinds),
            "sample_per_rate": self.sample_per_rate,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, self-describing experiment."""

    kind: str
    description: str
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    benchmarks: tuple[BenchmarkConfig, ...] = ()
    solvers: tuple[str, ...] = ("LOCAL_SEARCH",)
    time_budgets: tuple[float, ...] = (5.0,)
    disruption: DisruptionConfig = field(default_factory=DisruptionConfig)
    objective_weights: ObjectiveWeights | None = None
    weight_provenance: str = "default"
    seed: int = 2026
    output_dir: str = "experiments"
    results_dir: str = "results"
    figures_dir: str = "docs/figures"
    cache_dir: str = "data/cache"
    cache_results: bool = True
    database_url: str | None = None

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        self.scenario.validate()
        self.disruption.validate()
        for b in self.benchmarks:
            b.validate()
        if not self.solvers:
            raise ConfigError("at least one solver must be configured")
        unknown = [s for s in self.solvers if s not in SOLVER_INDEX]
        if unknown:
            raise ConfigError(
                f"unknown solver(s) {unknown}; available: {sorted(ALL_SOLVERS)}"
            )
        if not self.time_budgets:
            raise ConfigError("at least one time budget must be configured")
        for budget in self.time_budgets:
            if budget <= 0:
                raise ConfigError(f"time budgets must be > 0, got {budget}")
        if self.weight_provenance not in WEIGHT_PROVENANCE:
            raise ConfigError(
                f"weight_provenance must be one of {sorted(WEIGHT_PROVENANCE)}"
            )
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")

    def solver_names(self) -> tuple[str, ...]:
        """Validated solver names, in config order."""
        return tuple(self.solvers)

    def weights(self) -> ObjectiveWeights:
        return self.objective_weights or ObjectiveWeights()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "scenario": self.scenario.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "solvers": list(self.solvers),
            "time_budgets": list(self.time_budgets),
            "disruption": self.disruption.to_dict(),
            "objective_weights": self.weights().to_dict(),
            "weight_provenance": self.weight_provenance,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "results_dir": self.results_dir,
            "figures_dir": self.figures_dir,
            "cache_dir": self.cache_dir,
            "cache_results": self.cache_results,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config from a parsed YAML mapping, with clear errors."""
        known = {f.name for f in fields(cls)}
        unexpected = set(data) - known
        if unexpected:
            raise ConfigError(
                f"unknown config key(s) {sorted(unexpected)}; allowed: {sorted(known)}"
            )
        kwargs: dict[str, Any] = {}
        if "scenario" in data:
            kwargs["scenario"] = _sub(ScenarioConfig, data["scenario"], "scenario")
        if "disruption" in data:
            kwargs["disruption"] = _sub(DisruptionConfig, data["disruption"], "disruption")
        if "benchmarks" in data:
            kwargs["benchmarks"] = tuple(
                _sub(BenchmarkConfig, b, f"benchmarks[{i}]")
                for i, b in enumerate(data["benchmarks"])
            )
        for key in ("solvers", "time_budgets"):
            if key in data and data[key] is not None:
                kwargs[key] = tuple(data[key])
        weights = data.get("objective_weights")
        if weights:
            allowed = {f.name for f in fields(ObjectiveWeights)}
            bad = set(weights) - allowed
            if bad:
                raise ConfigError(f"unknown objective weight(s) {sorted(bad)}; allowed: {sorted(allowed)}")
            kwargs["objective_weights"] = ObjectiveWeights(**{k: float(v) for k, v in weights.items()})
        for key, value in data.items():
            if key in ("scenario", "disruption", "benchmarks", "solvers", "time_budgets", "objective_weights"):
                continue
            if key not in known:
                continue  # already rejected above
            kwargs[key] = value
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        import yaml

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p}: invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        return cls.from_dict(data)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        return replace(self, **kwargs)


def _sub(cls: type, data: Any, label: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{label} must be a mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    bad = set(data) - allowed
    if bad:
        raise ConfigError(f"unknown key(s) {sorted(bad)} in {label}; allowed: {sorted(allowed)}")
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items()})
