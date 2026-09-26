"""Structured logging and the audit log.

Two distinct concerns, deliberately separate:

* **Logs** answer "what is the system doing right now, and why did it fail?"
  They go to stderr as JSON lines, with scenario/plan/run correlation ids.
* **The audit log** answers "who or what changed operational state, in what
  order, and was the result accepted?" It is an append-only record persisted
  with the run, because a plan that was never approved must be
  distinguishable from a plan that was.

Neither contains secrets: the formatters only emit the fields listed in
:data:`SAFE_FIELDS`, and no configuration value is logged unless it is
explicitly allow-listed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

#: Fields safe to include in a log line. Anything else must be passed through
#: :func:`redact` first, so an accidental ``password=`` never reaches a log.
SAFE_FIELDS = frozenset(
    {
        "scenario_id", "plan_id", "experiment_id", "run_id", "solver", "status",
        "runtime_s", "objective", "service_level", "tasks_assigned", "late_tasks",
        "travel_km", "violations", "disruption_id", "disruption_type", "strategy",
        "severity", "affected_tasks", "repaired_tasks", "recovery_pct", "churn",
        "time_budget_s", "seed", "level", "suite", "stage", "seconds", "events",
        "error", "error_type", "message", "count", "rows", "exit_code",
    }
)
_REDACTED = "***redacted***"
_SENSITIVE_HINTS = ("password", "secret", "token", "api_key", "apikey", "credential", "passwd")


def redact(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unknown fields and mask anything that looks sensitive."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = key.lower()
        if any(hint in lowered for hint in _SENSITIVE_HINTS):
            out[key] = _REDACTED
        elif key in SAFE_FIELDS:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so logs are greppable and machine-parsable."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, Mapping):
            payload.update(redact(extra))
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Error"
            payload["error"] = str(record.exc_info[1])
        return json.dumps(payload, default=str)


_LOGGER_CONFIGURED = False


def configure_logging(level: str | int | None = None, stream: TextIO | None = None) -> None:
    """Install the JSON handler on the root ORION logger exactly once."""
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("orion")
    if _LOGGER_CONFIGURED:
        if level is not None:
            logger.setLevel(_coerce_level(level))
        return
    resolved = _coerce_level(level if level is not None else os.environ.get("ORION_LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(resolved)
    logger.propagate = False
    _LOGGER_CONFIGURED = True


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"orion.{name}")


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"fields": redact(fields)})


class AuditAction(str, Enum):
    """Meaningful state changes, recorded in order."""

    SCENARIO_CREATED = "scenario_created"
    SCENARIO_VALIDATED = "scenario_validated"
    PLAN_GENERATED = "plan_generated"
    OPTIMIZATION_RUN = "optimization_run"
    SIMULATION_STARTED = "simulation_started"
    DISRUPTION_INJECTED = "disruption_injected"
    LOCAL_REPAIR_ATTEMPTED = "local_repair_attempted"
    FULL_REOPT_ATTEMPTED = "full_reoptimization_attempted"
    PLAN_COMPARED = "plan_compared"
    PLAN_ACCEPTED = "plan_accepted"
    WHAT_IF_RUN = "what_if_run"
    SCENARIO_EXPORTED = "scenario_exported"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_FINISHED = "experiment_finished"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    sequence: int
    timestamp_utc: str
    action: str
    actor: str
    scenario_id: str
    plan_id: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    outcome: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "action": self.action,
            "actor": self.actor,
            "scenario_id": self.scenario_id,
            "plan_id": self.plan_id,
            "detail": redact(self.detail),
            "outcome": self.outcome,
        }


class AuditLog:
    """Thread-safe, append-only audit trail for one scenario or run."""

    def __init__(self, path: Path | str | None = None, *, actor: str = "system") -> None:
        self.path = Path(path) if path else None
        self.actor = actor
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._log = get_logger("audit")

    def record(
        self,
        action: AuditAction | str,
        scenario_id: str,
        *,
        plan_id: str | None = None,
        outcome: str = "ok",
        **detail: Any,
    ) -> AuditEntry:
        with self._lock:
            entry = AuditEntry(
                sequence=len(self._entries) + 1,
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                action=action.value if isinstance(action, AuditAction) else str(action),
                actor=self.actor,
                scenario_id=scenario_id,
                plan_id=plan_id,
                detail=detail,
                outcome=outcome,
            )
            self._entries.append(entry)
        self._log.info(f"audit {entry.action}", extra={"fields": {"scenario_id": scenario_id, "plan_id": plan_id, "message": entry.action}})
        if self.path:
            self._append_file(entry)
        return entry

    def _append_file(self, entry: AuditEntry) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), default=str) + "\n")

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def export(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"entries": self.to_dicts()}, indent=2, default=str), encoding="utf-8"
        )
        return target


class StageTimer:
    """Context manager that logs and records how long a stage took."""

    def __init__(self, stage: str, logger: logging.Logger | None = None, **fields: Any) -> None:
        self.stage = stage
        self.logger = logger or get_logger("perf")
        self.fields = fields
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
        self.logger.info(
            f"stage {self.stage} done",
            extra={"fields": {"stage": self.stage, "seconds": round(self.seconds, 6), **self.fields}},
        )
