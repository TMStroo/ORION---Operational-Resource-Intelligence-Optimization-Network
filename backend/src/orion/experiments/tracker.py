"""Immutable, provenance-complete experiment store.

Design rules, all enforced in code rather than by convention:

* An experiment ID is a *content* address. Re-using one raises
  :class:`ExperimentError` instead of overwriting, so a completed run can never
  be silently replaced by a different one.
* The directory is created with ``O_EXCL`` semantics on the manifest, so two
  concurrent runs cannot claim the same ID.
* Every run records git commit, Python version, package versions, seed, the
  full scenario/benchmark config, solver, time budget and objective weights.
  Anything needed to re-run the experiment is in the manifest or in files
  beside it - never in a developer's shell history.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "orion.experiment/1"


class ExperimentError(RuntimeError):
    """Raised when an experiment ID is re-used or a manifest is malformed."""


def _git_commit(root: Path) -> str:
    """Best-effort git commit; ``unknown`` outside a checkout (e.g. a wheel)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _package_versions() -> dict[str, str]:
    """Pinned versions of everything that can change a numeric result."""
    names = ("ortools", "numpy", "scipy", "pandas", "fastapi", "pydantic", "matplotlib")
    out: dict[str, str] = {}
    for name in names:
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:  # noqa: BLE001 - a missing optional dep is not fatal
            out[name] = "not-installed"
    return out


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, no NaN.

    ``allow_nan=False`` is deliberate. A NaN in a manifest would make the hash
    non-reproducible across platforms and would smuggle a broken result into the
    report; failing here is the correct behaviour.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One (scenario, solver, budget) measurement inside an experiment."""

    benchmark: str
    instance: str
    solver: str
    time_budget_s: float
    seed: int
    status: str
    runtime_s: float
    objective: float
    service_level: float
    tasks_assigned: int
    late_tasks: int
    travel_km: float
    operating_cost: float
    violations: int
    model_variables: int
    model_constraints: int
    optimality_gap: float | None = None
    split: str = "dev"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def csv_row(self) -> dict[str, Any]:
        d = self.to_dict()
        d["optimality_gap"] = "" if self.optimality_gap is None else f"{self.optimality_gap:.6f}"
        return d


@dataclass(slots=True)
class Manifest:
    """Everything needed to re-run and audit an experiment."""

    experiment_id: str
    created_utc: str
    kind: str
    description: str
    git_commit: str
    python_version: str
    platform: str
    package_versions: dict[str, str]
    seeds: list[int]
    config: dict[str, Any]
    objective_weights: dict[str, float]
    config_hash: str
    schema_hash: str
    dataset_checksums: dict[str, str] = field(default_factory=dict)
    splits: dict[str, str] = field(default_factory=dict)
    runs: list[RunRecord] = field(default_factory=list)
    status: str = "running"
    completed_utc: str | None = None
    #: Set only by `ExperimentStore.supersede`. A superseded experiment is kept
    #: on disk as the record of a run that happened, but is no longer evidence:
    #: figures and reports must filter on this.
    superseded_utc: str | None = None
    superseded_reason: str | None = None
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        payload["runs"] = [r.to_dict() for r in self.runs]
        return payload


def environment_fingerprint(root: Path) -> dict[str, Any]:
    """Host details that can change timings (never correctness)."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
        "package_versions": _package_versions(),
        "git_commit": _git_commit(root),
    }
    try:  # best effort: only present on Linux
        info["cpu_model"] = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    return info


class ExperimentStore:
    """Filesystem-backed experiment store rooted at ``root/experiments``."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- lookup -----------------------------------------------------------
    def path_for(self, experiment_id: str) -> Path:
        if not experiment_id or "/" in experiment_id or "\\" in experiment_id or experiment_id in {".", ".."}:
            raise ExperimentError(f"invalid experiment id: {experiment_id!r}")
        return self.root / experiment_id

    def exists(self, experiment_id: str) -> bool:
        return (self.path_for(experiment_id) / "manifest.json").exists()

    def list_ids(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
        )

    def load(self, experiment_id: str) -> Manifest:
        path = self.path_for(experiment_id) / "manifest.json"
        if not path.exists():
            raise ExperimentError(f"unknown experiment: {experiment_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = [RunRecord(**r) for r in data.pop("runs", [])]
        data.pop("schema_version", None)
        return Manifest(runs=runs, **data)

    # -- creation ---------------------------------------------------------
    def create(
        self,
        *,
        kind: str,
        description: str,
        config: Mapping[str, Any],
        objective_weights: Mapping[str, float],
        seeds: Sequence[int],
        experiment_id: str | None = None,
        dataset_checksums: Mapping[str, str] | None = None,
        splits: Mapping[str, str] | None = None,
        root: Path | str | None = None,
    ) -> tuple[Manifest, Path]:
        """Create a new experiment directory, refusing to reuse an ID."""
        base = Path(root).resolve() if root is not None else self.root
        config_dict = dict(config)
        exp_id = experiment_id or self._derive_id(kind, config_dict, list(seeds), dataset_checksums or {})
        directory = base / exp_id
        if (directory / "manifest.json").exists():
            raise ExperimentError(
                f"experiment {exp_id!r} already exists; experiments are immutable. "
                f"Use a new id (e.g. add a config hash) rather than overwriting it."
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "figures").mkdir(exist_ok=True)
        (directory / "logs").mkdir(exist_ok=True)

        env = environment_fingerprint(base)
        manifest = Manifest(
            experiment_id=exp_id,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            kind=kind,
            description=description,
            git_commit=env["git_commit"],
            python_version=env["python_version"],
            platform=env["platform"],
            package_versions=env["package_versions"],
            seeds=list(seeds),
            config=config_dict,
            objective_weights=dict(objective_weights),
            config_hash=content_hash(config_dict),
            schema_hash=content_hash(sorted(_schema_of(config_dict))),
            dataset_checksums=dict(dataset_checksums or {}),
            splits=dict(splits or {}),
        )
        self._write_manifest(directory, manifest)
        (directory / "manifest.partial.json").unlink(missing_ok=True)
        return manifest, directory

    @staticmethod
    def _derive_id(
        kind: str,
        config: Mapping[str, Any],
        seeds: Sequence[int],
        checksums: Mapping[str, str],
    ) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = content_hash({"config": config, "seeds": list(seeds), "checksums": dict(checksums)})[:10]
        return f"{kind}-{stamp}-{digest}"

    @staticmethod
    def _write_manifest(directory: Path, manifest: Manifest) -> None:
        """Atomically publish the manifest (temp file + replace)."""
        tmp = directory / "manifest.partial.json"
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, directory / "manifest.json")

    # -- writing ----------------------------------------------------------
    def add_run(self, directory: Path, record: RunRecord) -> None:
        """Append a run and re-publish the manifest atomically."""
        manifest = self.load(directory.name)
        manifest.runs.append(record)
        self._write_manifest(directory, manifest)

    def write_artifacts(
        self,
        directory: Path,
        *,
        metrics: Mapping[str, Any] | None = None,
        timings: Mapping[str, Any] | None = None,
        tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Write metrics.json, timings.json and any named CSV tables."""
        if metrics is not None:
            (directory / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False, default=str), encoding="utf-8"
            )
        if timings is not None:
            (directory / "timings.json").write_text(
                json.dumps(timings, indent=2, allow_nan=False, default=str), encoding="utf-8"
            )
        for name, rows in (tables or {}).items():
            self.write_csv(directory / f"{name}.csv", rows)

    @staticmethod
    def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        import csv

        rows = list(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def supersede(self, experiment_id: str, reason: str, *, replaced_by: str | None = None) -> Path:
        """Mark a completed experiment as no longer live evidence.

        An experiment directory is never deleted or rewritten - the record of
        what was actually run stays on disk. What changes is its status, so any
        report or figure generator that filters on status automatically stops
        quoting it. This is how a run that turned out to rest on a bug is
        retired without erasing the evidence that it was run.
        """
        directory = self.path_for(experiment_id)
        manifest = self.load(experiment_id)
        if manifest.status in ("superseded",):
            return directory  # already retired; keep it idempotent
        manifest.status = "superseded"
        superseded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.superseded_utc = superseded_utc
        manifest.superseded_reason = reason
        manifest.superseded_by = replaced_by
        self._write_manifest(directory, manifest)
        note = {
            "status": "superseded",
            "reason": reason,
            "replaced_by": replaced_by,
            "superseded_utc": superseded_utc,
            "note": (
                "This directory is retained as a historical record of a run that "
                "was executed. It must not be quoted as evidence."
            ),
        }
        (directory / "SUPERSEDED.json").write_text(
            json.dumps(note, indent=2, allow_nan=False, default=str), encoding="utf-8"
        )
        return directory

    def live(self) -> list[str]:
        """Ids of experiments that are still valid evidence."""
        return [
            eid
            for eid in self.list_ids()
            if self.load(eid).status not in ("superseded", "failed")
        ]

    def finish(
        self,
        directory: Path,
        *,
        status: str,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        """Seal the experiment. A failed run is recorded as failed, not hidden."""
        manifest = self.load(directory.name)
        manifest.status = status
        manifest.completed_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_manifest(directory, manifest)
        if summary is not None:
            (directory / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False, default=str), encoding="utf-8"
            )


def _schema_of(config: Mapping[str, Any]) -> Iterable[str]:
    """Flatten a config into dotted key paths, for the schema hash.

    Hashing the *shape* rather than the values means two experiments that
    differ only in a seed still share a schema hash, so a schema change (a new
    key, a removed field) is distinguishable from a parameter change.
    """
    stack: list[tuple[str, Any]] = [("", config)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, Mapping):
            for k, v in node.items():
                stack.append((f"{prefix}.{k}" if prefix else str(k), v))
        elif isinstance(node, (list, tuple)):
            stack.append((f"{prefix}[]", type(node).__name__))
            for i, v in enumerate(node):
                stack.append((f"{prefix}[{i}]", v))
        else:
            yield f"{prefix}:{type(node).__name__}"


class ExperimentTimer:
    """Collects stage timings so performance claims are measured, not guessed."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def record(self, stage: str, seconds: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + float(seconds)

    def time(self, stage: str) -> "_StageTimer":
        return _StageTimer(self, stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages_seconds": dict(self.stages),
            "total_seconds": sum(self.stages.values()),
            "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


class _StageTimer:
    def __init__(self, owner: ExperimentTimer, stage: str) -> None:
        self._owner = owner
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._owner.record(self._stage, time.perf_counter() - self._start)
