"""Read every number the documentation quotes from one place.

The README, the technical report and the consistency checker must never
disagree. The only reliable way to guarantee that is to make them all read the
same loader, so no document is ever allowed to contain a hand-typed result.

This module reads the *live* experiments only. A superseded run is preserved on
disk and is deliberately invisible here: quoting from one would mean quoting
from a result the project has already declared invalid.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Statuses that assert a plan exists. Anything else has no result to quote.
CLAIMED_RESULT = frozenset({"OPTIMAL", "FEASIBLE", "RELAXATION_OPTIMAL"})
SUCCESS = CLAIMED_RESULT | {"TIME_LIMIT"}


@dataclass(frozen=True)
class Suite:
    """One live experiment run."""

    kind: str
    experiment_id: str
    directory: Path
    manifest: dict[str, Any]
    rows: list[dict[str, str]]

    @property
    def git_commit(self) -> str:
        return str(self.manifest.get("git_commit", ""))[:8]

    @property
    def created_utc(self) -> str:
        return str(self.manifest.get("created_utc", ""))

    def claimed(self) -> list[dict[str, str]]:
        """Rows that assert a plan, i.e. rows a document may quote."""
        return [r for r in self.rows if r.get("status") in CLAIMED_RESULT]

    def find(self, **match: str) -> dict[str, str] | None:
        """First row matching every key/value pair."""
        for row in self.rows:
            if all(row.get(k) == v for k, v in match.items()):
                return row
        return None

    def all(self, **match: str) -> list[dict[str, str]]:
        return [r for r in self.rows if all(r.get(k) == v for k, v in match.items())]


@dataclass
class Evidence:
    """Everything the documentation is allowed to say."""

    root: Path
    suites: dict[str, Suite] = field(default_factory=dict)
    superseded: list[dict[str, Any]] = field(default_factory=list)
    verify: dict[str, Any] = field(default_factory=dict)
    demo: dict[str, Any] = field(default_factory=dict)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, project_root: Path | str) -> "Evidence":
        root = Path(project_root)
        evidence = cls(root=root)

        experiments = root / "experiments"
        if experiments.is_dir():
            for directory in sorted(experiments.iterdir()):
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                kind = str(manifest.get("kind", directory.name)).removeprefix("benchmark-")
                if manifest.get("status") == "superseded":
                    evidence.superseded.append(
                        {
                            "experiment_id": directory.name,
                            "kind": kind,
                            "reason": manifest.get("superseded_reason", ""),
                            "created_utc": manifest.get("created_utc", ""),
                        }
                    )
                    continue
                results = directory / "results.csv"
                rows: list[dict[str, str]] = []
                if results.is_file():
                    rows = list(csv.DictReader(io.StringIO(results.read_text(encoding="utf-8"))))
                # Newest run per suite wins; ties broken by directory name.
                existing = evidence.suites.get(kind)
                if existing is None or directory.name > existing.experiment_id:
                    evidence.suites[kind] = Suite(kind, directory.name, directory, manifest, rows)

        verify_path = root / "results" / "verify" / "verify.json"
        if verify_path.is_file():
            evidence.verify = json.loads(verify_path.read_text(encoding="utf-8"))
        demo_path = root / "results" / "demo" / "demo_summary.json"
        if demo_path.is_file():
            evidence.demo = json.loads(demo_path.read_text(encoding="utf-8"))
        return evidence

    def get(self, kind: str) -> Suite:
        if kind not in self.suites:
            raise KeyError(
                f"no live {kind} experiment; have {sorted(self.suites)}"
            )
        return self.suites[kind]

    def has(self, kind: str) -> bool:
        return kind in self.suites

    # -- solver table ----------------------------------------------------

    def solver_table(self) -> list[dict[str, Any]]:
        """The public solver comparison, one row per (dataset, instance, solver, budget).

        `Method` is not decoration. A relaxation and an exact solver may both
        report a large objective; reading the objective column without the method
        column is what makes MIN_COST_FLOW look like a peer of CP-SAT.
        """
        if not self.has("solver_comparison"):
            return []
        method = {
            "CP_SAT": "exact (constraint programming)",
            "MILP": "relaxation (big-M precedence)",
            "MIN_COST_FLOW": "relaxation (transportation)",
            "HEURISTIC": "heuristic (greedy insertion)",
            "LOCAL_SEARCH": "heuristic (ALNS improvement)",
        }
        out: list[dict[str, Any]] = []
        for row in self.get("solver_comparison").rows:
            solver = row.get("solver", "")
            status = row.get("status", "")
            out.append(
                {
                    "dataset": row.get("dataset", ""),
                    "instance": row.get("instance", ""),
                    "budget_s": row.get("time_budget_s", ""),
                    "solver": solver,
                    "method": method.get(solver, "unknown"),
                    "proven_optimal": status == "OPTIMAL",
                    "status": status,
                    "objective": _num(row.get("objective")),
                    "tasks_assigned": int(row.get("tasks_assigned") or 0),
                    "late_tasks": int(row.get("late_tasks") or 0),
                    "travel_km": _num(row.get("travel_km")),
                    "runtime_s": _num(row.get("runtime_s")),
                }
            )
        return out

    # -- headline figures ------------------------------------------------

    def scalability(self) -> list[dict[str, Any]]:
        if not self.has("scalability"):
            return []
        out = []
        for row in self.get("scalability").rows:
            out.append(
                {
                    "tasks": int(row.get("task_count") or 0),
                    "budget_s": float(row.get("time_budget_s") or 0),
                    "solver": row.get("solver", ""),
                    "status": row.get("status", ""),
                    "assigned": int(row.get("tasks_assigned") or 0),
                    "objective": _num(row.get("objective")),
                    "runtime_s": _num(row.get("runtime_s")),
                    "gap_to_best": _num(row.get("gap_to_best")),
                }
            )
        return out

    def timeout_rate(self) -> dict[str, dict[str, Any]]:
        """Share of runs each solver ended in TIME_LIMIT, by size."""
        rates: dict[str, dict[str, Any]] = {}
        for row in self.scalability():
            entry = rates.setdefault(row["solver"], {"timeouts": 0, "runs": 0})
            entry["runs"] += 1
            if row["status"] == "TIME_LIMIT":
                entry["timeouts"] += 1
        for solver, entry in rates.items():
            entry["rate"] = entry["timeouts"] / entry["runs"] if entry["runs"] else 0.0
        return rates

    def public_benchmarks(self) -> list[dict[str, Any]]:
        """Solomon and job-shop rows only."""
        return [r for r in self.solver_table() if r["dataset"] in ("solomon", "jobshop")]

    def disruption_recovery(self) -> dict[str, Any]:
        """Recovery by strategy and severity, averaged over rates."""
        if not self.has("disruption_stress"):
            return {}
        buckets: dict[tuple[str, str], list[float]] = {}
        runtimes: dict[str, list[float]] = {}
        churn: dict[str, list[float]] = {}
        for row in self.get("disruption_stress").rows:
            strategy = row.get("strategy", "")
            severity = row.get("severity", "")
            recovery = _num(row.get("recovery_pct"))
            if recovery is None:
                continue
            buckets.setdefault((strategy, severity), []).append(recovery)
            if strategy != "none":
                seconds = _num(row.get("replan_seconds"))
                if seconds is not None:
                    runtimes.setdefault(strategy, []).append(seconds)
                fraction = _num(row.get("churn_fraction"))
                if fraction is not None:
                    churn.setdefault(strategy, []).append(fraction)
        summary: dict[str, Any] = {"by_strategy_severity": {}, "runtime_s": {}, "churn": {}}
        for (strategy, severity), values in sorted(buckets.items()):
            summary["by_strategy_severity"][f"{strategy}/{severity}"] = _mean(values)
        for strategy, values in sorted(runtimes.items()):
            summary["runtime_s"][strategy] = _mean(values)
        for strategy, values in sorted(churn.items()):
            summary["churn"][strategy] = _mean(values)
        return summary

    def constraint_pressure(self) -> list[dict[str, Any]]:
        if not self.has("constraint_pressure"):
            return []
        return [
            {
                "scarcity": row.get("scarcity", ""),
                "deadline_tightness": row.get("deadline_tightness", ""),
                "feasible": row.get("feasible", ""),
                "assigned": int(row.get("tasks_assigned") or 0),
                "late_tasks": int(row.get("late_tasks") or 0),
                "objective": _num(row.get("objective")),
                "runtime_s": _num(row.get("runtime_s")),
            }
            for row in self.get("constraint_pressure").rows
        ]

    def reference_validation(self) -> dict[str, Any]:
        """Brute-force agreement, read from `orion verify` output.

        The lists are empty when the property holds, so the counts are what the
        documentation may quote. `shortfall` holds the expected gaps of the
        relaxations and heuristics - a non-empty list is the honest record of
        which solver did not reach the proven optimum, and hiding it would make
        the comparison look better than it is.
        """
        data = self.verify
        references = data.get("references") or {}
        shortfall = data.get("shortfall") or []
        worst = None
        if shortfall:
            values = [
                float(s.get("gap_pct", s.get("gap", 0.0)))
                for s in shortfall
                if isinstance(s, dict)
            ]
            values = [v for v in values if v == v]
            if values:
                worst = min(values)
        return {
            "fixtures": len(references),
            "proven_mismatches": len(data.get("proven_mismatches") or []),
            "beats_reference": len(data.get("beats_reference") or []),
            "expected_shortfalls": len(shortfall),
            "worst_shortfall_pct": worst,
            "seeds": data.get("seeds") or [],
            "heuristic_summary": data.get("heuristic_summary") or {},
            "seed_spread": data.get("seed_spread") or {},
            "available": bool(data),
        }

    # -- integrity -------------------------------------------------------

    def integrity_problems(self) -> list[str]:
        """Every inconsistency found in the live set. Empty means clean.

        These are the same properties the test suite asserts, run over whatever
        is on disk so the documentation checker and the tests cannot drift.
        """
        problems: list[str] = []
        for kind, suite in sorted(self.suites.items()):
            for index, row in enumerate(suite.rows):
                where = f"{suite.experiment_id} row {index}"
                status = row.get("status")
                if status is None:
                    continue
                objective = _num(row.get("objective"))
                if status in CLAIMED_RESULT and objective == 0.0:
                    problems.append(f"{where}: claims {status} with objective 0.0")
                if status not in SUCCESS and objective not in (0.0, None):
                    problems.append(f"{where}: {status} carries objective {objective}")
                gap = _num(row.get("gap_to_best"))
                if gap is not None and gap < -1e-9:
                    problems.append(f"{where}: negative gap {gap}")
                if status == "OPTIMAL" and row.get("solver") in ("MILP", "MIN_COST_FLOW"):
                    problems.append(f"{where}: relaxation solver claims OPTIMAL")
        return problems

    def live_suites(self) -> list[str]:
        return sorted(self.suites)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed  # drop NaN


def _mean(values: list[float]) -> float | None:
    real = [v for v in values if v == v]
    return sum(real) / len(real) if real else None
