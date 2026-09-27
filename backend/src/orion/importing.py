"""Reading back what :mod:`orion.exporting` wrote.

An export that cannot be reloaded is a screenshot, not evidence. Every loader
here goes through the domain entity's own ``from_dict`` rather than rebuilding
fields by hand, so an export and a reload cannot drift apart as the domain grows.

Each loader returns ``(entity, provenance)``. The provenance block travels with
the artifact and is returned to the caller instead of being discarded, because a
reloaded scenario with no record of the seed that produced it is not
reproducible.

Round-trip contract, pinned by ``tests/test_exports.py``:

* ``load_scenario`` returns a scenario whose ``to_dict()`` equals the exported
  ``data`` block;
* ``load_plan`` returns a plan whose ``to_dict()`` equals the exported ``data``
  block;
* nothing is dropped silently - ids, assignments, timestamps, solver status,
  objective and provenance all survive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from orion.domain.entities import Scenario
from orion.domain.plans import Plan
from orion.exporting import ExportError

__all__ = [
    "ExportError",
    "read_artifact",
    "read_inline_artifact",
    "load_scenario",
    "load_plan",
    "read_json",
    "read_csv",
    "csv_provenance",
    "read_manifest",
]


def read_json(path: Path | str) -> dict[str, Any]:
    """Read one exported artifact, rejecting anything that is not one.

    An export is ``{"provenance": ..., "data": ...}``. A bare payload is a
    different file that happens to end in ``.json``; loading it as an export
    would report a missing provenance block as an empty one.
    """
    target = Path(path)
    if not target.exists():
        raise ExportError(f"no such export: {target}")
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{target.name} is not valid JSON: {exc}") from exc
    if not isinstance(body, dict) or "data" not in body:
        raise ExportError(
            f"{target.name} is not an ORION export: expected a 'data' key, "
            f"found {sorted(body) if isinstance(body, dict) else type(body).__name__}"
        )
    return body


def read_artifact(path: Path | str) -> tuple[Any, dict[str, Any]]:
    """The ``(data, provenance)`` pair every export carries."""
    body = read_json(path)
    provenance = body.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ExportError(f"{Path(path).name}: provenance is not an object")
    return body["data"], provenance


def load_scenario(path: Path | str) -> tuple[Scenario, dict[str, Any]]:
    """Rebuild a scenario from its export.

    The domain's own ``from_dict`` does the work, so the reloaded scenario is
    the same class with the same defaults rather than a hand-assembled copy that
    could quietly differ.
    """
    data, provenance = read_artifact(path)
    if not isinstance(data, dict):
        raise ExportError(f"{Path(path).name}: scenario payload is not an object")
    try:
        return Scenario.from_dict(data), provenance
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        raise ExportError(f"{Path(path).name}: cannot rebuild scenario: {exc}") from exc


def load_plan(path: Path | str) -> tuple[Plan, dict[str, Any]]:
    """Rebuild a plan from its export, including assignments and solver runs."""
    data, provenance = read_artifact(path)
    if not isinstance(data, dict):
        raise ExportError(f"{Path(path).name}: plan payload is not an object")
    try:
        return Plan.from_dict(data), provenance
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        raise ExportError(f"{Path(path).name}: cannot rebuild plan: {exc}") from exc


def read_csv(path: Path | str) -> list[dict[str, str]]:
    """Read an exported CSV's data rows.

    ``write_csv`` puts the provenance block on a leading ``# provenance:``
    comment rather than in a sidecar, so the file opens with a comment line
    that ``csv.DictReader`` would otherwise treat as the header row. The
    comment is skipped here and returned separately by :func:`csv_provenance`,
    so a caller never has to know which convention the writer used.
    """
    target = Path(path)
    if not target.exists():
        raise ExportError(f"no such export: {target}")
    with target.open(newline="", encoding="utf-8") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        return [dict(row) for row in csv.DictReader(lines)]


def csv_provenance(path: Path | str) -> dict[str, Any]:
    """The provenance comment a CSV was written with, or an empty mapping."""
    target = Path(path)
    if not target.exists():
        raise ExportError(f"no such export: {target}")
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# provenance: "):
                try:
                    return json.loads(line[len("# provenance: "):])
                except ValueError as exc:
                    raise ExportError(
                        f"{target.name}: provenance comment is not valid JSON: {exc}"
                    ) from exc
    return {}


def read_manifest(path: Path | str) -> dict[str, Any]:
    """Read an experiment manifest, which is a plain JSON document."""
    target = Path(path)
    if not target.exists():
        raise ExportError(f"no such manifest: {target}")
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{target.name} is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ExportError(f"{target.name}: manifest is not an object")
    return body


def provenance_of(path: Path | str) -> Mapping[str, Any]:
    """The provenance block of an export, without loading its payload."""
    if Path(path).suffix == ".json":
        return read_json(path).get("provenance") or {}
    sidecar = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    if sidecar.exists():
        return read_json(sidecar).get("provenance") or {}
    return {}


def read_inline_artifact(body: Any) -> tuple[Any, dict[str, Any]]:
    """The ``(data, provenance)`` pair from an export held in memory.

    The API receives an exported artifact as a request body rather than a file,
    so it cannot use :func:`read_artifact`. This is the same validation against
    the same rules: a payload without a ``data`` key is not an export and is
    rejected, rather than being handed to ``from_dict`` and producing a
    confusing downstream error.
    """
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ExportError(f"artifact is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ExportError(
            f"artifact is not an ORION export: expected an object, "
            f"found {type(body).__name__}"
        )
    if "data" not in body:
        raise ExportError(
            f"artifact is not an ORION export: expected a 'data' key, "
            f"found {sorted(body)}"
        )
    provenance = body.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ExportError("provenance is not an object")
    return body["data"], provenance
