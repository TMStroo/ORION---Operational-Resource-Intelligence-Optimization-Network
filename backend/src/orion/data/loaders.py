"""Scenario loading: from disk, from a generator preset, or from a benchmark.

This is the single entry point the CLI, the API and the tests all use, so the
"is this a path or a preset name?" decision is made once. The loaders return
plain :class:`~orion.domain.entities.Scenario` objects; no benchmark-specific
field ever escapes :mod:`orion.data.benchmark_adapters`.

Three input shapes are supported:

1. a path to a ``.json`` scenario written by ``orion scenario create``
2. a difficulty preset name (``small`` / ``medium`` / ``large`` / ``stress``)
3. a benchmark reference, ``<dataset>:<instance>`` (e.g. ``solomon:c101``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion.data.benchmark_adapters import (
    DataSource,
    JOBSHOP_SOURCE,
    SOLOMON_SOURCE,
    jobshop_to_scenario,
    load_jobshop,
    load_solomon,
    solomon_to_scenario,
)
from orion.data.scenario_generator import DIFFICULTY_PRESETS, difficulty_config, generate
from orion.domain.entities import Scenario
from orion.domain.errors import ConfigError, ExportError, ValidationError

#: Bumped whenever an adapter's field mapping changes, so an old experiment can
#: be interpreted with the adapter version that produced it.
ADAPTER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    """A scenario plus everything needed to cite where it came from."""

    scenario: Scenario
    origin: str  # "file" | "generated" | "benchmark"
    source: str
    provenance: dict[str, Any]
    adapter_version: str = ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "source": self.source,
            "adapter_version": self.adapter_version,
            "provenance": self.provenance,
            "scenario": self.scenario.to_dict(),
        }


def _file_provenance(path: Path) -> dict[str, Any]:
    import hashlib

    blob = path.read_bytes()
    return {
        "local_source_file": str(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def load_scenario(spec: str | Path, *, cache_dir: str = "data/cache") -> LoadedScenario:
    """Load a scenario from a path, a preset name, or ``dataset:instance``."""
    if spec is None:
        raise ValidationError("no scenario specified")
    text = str(spec)

    if ":" in text and not Path(text).exists():
        dataset, _, instance = text.partition(":")
        if dataset in {"solomon", "jobshop"}:
            return load_benchmark(dataset, instance, cache_dir=cache_dir)
        raise ConfigError(
            f"unknown benchmark dataset {dataset!r}; expected solomon or jobshop"
        )

    path = Path(text)
    if path.exists():
        if path.suffix.lower() != ".json":
            raise ConfigError(f"scenario file must be .json, got {path.suffix!r}: {path}")
        try:
            scenario = Scenario.from_json(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
        return LoadedScenario(
            scenario=scenario,
            origin="file",
            source=str(path),
            provenance={"local_source_file": str(path), **_file_provenance(path)},
        )

    if text.lower() in DIFFICULTY_PRESETS:
        scenario = generate(difficulty_config(text.lower(), seed=2026))
        return LoadedScenario(
            scenario=scenario,
            origin="generated",
            source=f"generator:{text.lower()}",
            provenance={
                "generator": "orion.data.scenario_generator",
                "preset": text.lower(),
                "seed": 2026,
                "adapter_version": ADAPTER_VERSION,
            },
        )

    raise ConfigError(
        f"cannot interpret {spec!r}: not a file, not one of {sorted(DIFFICULTY_PRESETS)}, "
        f"and not a 'dataset:instance' benchmark reference"
    )


def load_benchmark(dataset: str, instance: str, *, cache_dir: str = "data/cache") -> LoadedScenario:
    """Load one public benchmark instance as an ORION scenario."""
    name = instance.lower()
    if dataset == "solomon":
        instances = load_solomon(cache_dir, (name,))
        if name not in instances:
            raise ConfigError(f"unknown Solomon instance {instance!r}")
        scenario = solomon_to_scenario(instances[name])
        source: DataSource = SOLOMON_SOURCE
    elif dataset == "jobshop":
        instances = load_jobshop(cache_dir, (name,))
        if name not in instances:
            raise ConfigError(f"unknown job-shop instance {instance!r}")
        scenario = jobshop_to_scenario(instances[name])
        source = JOBSHOP_SOURCE
    else:
        raise ConfigError(f"unknown dataset {dataset!r}; expected solomon or jobshop")

    return LoadedScenario(
        scenario=scenario,
        origin="benchmark",
        source=f"{dataset}:{name}",
        provenance={**source.to_dict(), "instance": name, "adapter_version": ADAPTER_VERSION},
    )


def scenario_provenance(loaded: LoadedScenario) -> dict[str, Any]:
    """Provenance block suitable for embedding in an experiment manifest."""
    return {
        "origin": loaded.origin,
        "source": loaded.source,
        "adapter_version": loaded.adapter_version,
        **{k: v for k, v in loaded.provenance.items() if k not in {"limitations"}},
        "limitations": loaded.provenance.get("limitations", ""),
    }


def save_scenario(scenario: Scenario, path: Path | str) -> Path:
    """Write a scenario to disk, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(scenario.to_json(indent=2), encoding="utf-8")
    except (OSError, ExportError) as exc:
        raise ExportError(f"could not write scenario to {target}: {exc}") from exc
    return target
