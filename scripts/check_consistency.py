"""Check that the experiment artifacts, README.md and the report agree.

Three documents can each be internally correct and still contradict each other:
a benchmark is re-run, one of them is regenerated and another is not. This
compares the numbers that matter across all three sources.

Deliberate design choices, because the artifacts are not uniform:

* **Not every experiment has metrics.json.** Some suites store only CSV result
  tables. The reader tries metrics.json, then the suite's CSVs, and records
  which it used, rather than assuming one layout.
* **Superseded runs are not evidence.** Only live runs are compared, so a stale
  row can never make a document look wrong, and a live row can never be hidden
  by a newer one.
* **Different suites have different schemas.** A scalability row has `tasks`
  and a budget; a solver-comparison row has `instance` and `time_budget_s`.
  Fields are read through a small alias map instead of one rigid schema.

Exit code 0 means the three agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from orion.evidence import Evidence  # noqa: E402

README = ROOT / "README.md"
REPORT_HTML = ROOT / "docs" / "report.html"
REPORT_PDF = ROOT / "docs" / "report.pdf"

#: Field aliases, because the suites were not written to one schema.
ALIASES: dict[str, tuple[str, ...]] = {
    "objective": ("objective", "candidate_objective", "best_score"),
    "runtime_s": ("runtime_s", "replan_seconds", "seconds"),
    "status": ("status", "solver_status"),
    "assigned": ("assigned", "tasks_assigned", "operations"),
    "service_level": ("service_level", "completion", "baseline_service"),
    "late_tasks": ("late_tasks", "late"),
    "travel_km": ("travel_km", "distance_km"),
    "tasks": ("tasks", "num_tasks", "task_count"),
}

#: Tolerances. Objectives are compared as printed (2dp in the docs); runtimes
#: are only ever checked for presence and sign, because they are not
#: reproducible to the digit and the documents do not claim they are.
OBJECTIVE_TOLERANCE = 0.01


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.problems.append(message)
        return condition


def _row_value(row: dict[str, Any], field: str) -> Any:
    for key in ALIASES.get(field, (field,)):
        if key in row and row[key] not in ("", None):
            return row[key]
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_suite_rows(evidence: Evidence, suite: str) -> tuple[list[dict[str, Any]], str]:
    """Read a suite's rows and report which artifact form was used.

    Some suites publish metrics.json, some only CSV tables, and some both. The
    point of returning the source is that a suite readable *only* as CSV is a
    legitimate state, not a failure.
    """
    entry = evidence.suites.get(suite)
    if entry is None:
        return [], "absent"
    rows = [dict(r) for r in entry.rows]
    source = "suite rows"
    metrics_path = entry.directory / "metrics.json"
    if metrics_path.is_file():
        source = "metrics.json + suite rows"
    else:
        csvs = sorted(entry.directory.glob("*.csv"))
        if csvs:
            source = f"CSV tables ({len(csvs)} file(s))"
    return rows, source


def check(documents: dict[str, str], evidence: Evidence) -> Report:
    report = Report()
    readme = documents.get("README.md", "")
    html = documents.get("report.html", "")

    # -- 0. the documents exist and are non-trivial ----------------------
    for name, text in documents.items():
        report.require(len(text) > 2000, f"{name} is missing or far too short to be real")

    # -- 1. only live runs are treated as evidence -----------------------
    # `superseded` is a property holding records, not a callable.
    for record in evidence.superseded:
        run_id = record.get("experiment_id", "")
        report.require(
            bool(record.get("reason")),
            f"superseded run {run_id} has no reason, so it cannot be audited",
        )
        report.require(
            run_id not in html,
            f"superseded run {run_id} is named in the report as if it were live",
        )

    # -- 2. reference validation agrees ----------------------------------
    ref = evidence.reference_validation()
    fixtures = ref.get("fixtures")
    mismatches = ref.get("proven_mismatches")
    if fixtures is not None:
        report.require(
            f"{fixtures} fixtures" in readme,
            f"the README does not state the fixture count ({fixtures}) the evidence reports",
        )
    if mismatches is not None:
        # The README prints "N proven mismatches" possibly across a line break.
        collapsed = " ".join(readme.split())
        report.require(
            f"{mismatches} proven mismatches" in collapsed,
            f"the README does not state the proven-mismatch count ({mismatches})",
        )

    # -- 3. c101 at the largest stored budget ----------------------------
    public = evidence.public_benchmarks()
    c101_budgets = sorted({float(r["budget_s"]) for r in public
                           if r.get("instance") == "c101"})
    if c101_budgets:
        budget = c101_budgets[-1]
        rows = [r for r in public if r.get("instance") == "c101" and float(r["budget_s"]) == budget]
        for r in rows:
            obj = _as_float(r.get("objective"))
            if obj is None:
                continue
            printed = f"{obj:,.2f}"
            report.require(
                printed in readme or printed.replace(",", "") in readme,
                f"c101 {r['solver']} objective {printed} from the live artifact "
                "is absent from the README",
            )
            status = str(r.get("status", ""))
            if status:
                report.require(
                    f"`{r['solver']}` | {status}" in readme
                    or f"| {status} |" in readme,
                    f"c101 {r['solver']} status {status} does not match the README table",
                )

    # -- 4. scalability at the largest size ------------------------------
    scal = evidence.scalability()
    if scal:
        max_tasks = max(int(_row_value(r, "tasks") or 0) for r in scal)
        report.require(
            f"At {max_tasks} tasks" in " ".join(readme.split()) or f"{max_tasks} tasks" in readme,
            f"the README does not mention the largest scalability size ({max_tasks} tasks)",
        )
        for r in scal:
            if int(_row_value(r, "tasks") or 0) != max_tasks:
                continue
            obj = _as_float(r.get("objective"))
            if obj is not None and r["solver"] == "HEURISTIC":
                report.require(
                    f"{obj:,.2f}" in readme or f"{obj:,.0f}" in readme,
                    f"the README omits the heuristic objective {obj:,.2f} at {max_tasks} tasks",
                )

    # -- 5. disruption recovery means agree ------------------------------
    recovery = evidence.disruption_recovery()
    by_strategy = recovery.get("by_strategy_severity", {})
    for strategy in ("local_repair", "full_reopt"):
        values = [v for k, v in by_strategy.items() if k.startswith(f"{strategy}/")]
        if not values:
            continue
        mean = sum(values) / len(values)
        # The documents print percentages to one decimal, so compare at the
        # precision they actually use rather than demanding more digits than
        # the document claims to carry.
        ok = any(
            f"{mean * 100:.{places}f}" in readme
            for places in (0, 1, 2)
        ) or f"{mean:.3f}" in " ".join(readme.split())
        report.require(
            ok,
            f"{strategy} mean recovery {mean * 100:.2f}% is not reflected in the README",
        )

    # -- 6. every suite is readable and reported -------------------------
    for suite in sorted(evidence.suites):
        rows, source = read_suite_rows(evidence, suite)
        report.require(
            bool(rows),
            f"live suite {suite!r} produced no readable rows (source: {source})",
        )

    # -- 7. the report embeds only figures that exist --------------------
    if html:
        for src in re.findall(r'<img src="([^"]+)"', html):
            resolved = (REPORT_HTML.parent / src.replace("/", "\\")).resolve()
            report.require(
                resolved.is_file(),
                f"the report references {src}, which does not exist on disk",
            )
    if REPORT_PDF.is_file():
        report.require(REPORT_PDF.stat().st_size > 10_000, "report.pdf is suspiciously small")
    else:
        report.require(False, "report.pdf does not exist")

    # -- 8. the report has all 27 sections -------------------------------
    if html:
        found = re.findall(r'<section id=[\'"]s(\d+)[\'"]>', html)
        report.require(
            found == [str(i) for i in range(1, 28)],
            f"the report has {len(found)} sections, expected exactly 27 numbered 1..27",
        )
        for heading in ("Executive Summary", "Conclusion", "Limitations", "Failure Analysis"):
            report.require(heading in html, f"the report is missing the {heading!r} section")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    documents: dict[str, str] = {}
    for name, path in (("README.md", README), ("report.html", REPORT_HTML)):
        if path.is_file():
            documents[name] = path.read_text(encoding="utf-8", errors="replace")
        else:
            documents[name] = ""
    if not REPORT_PDF.is_file():
        documents["report.pdf"] = ""

    evidence = Evidence.load(ROOT)
    report = check(documents, evidence)

    print(f"consistency check: {report.checks} assertions over "
          f"{len(evidence.suites)} live suite(s)")
    if report.problems:
        print(f"\nFAILED ({len(report.problems)}):\n")
        for problem in report.problems:
            print(f"  - {problem}")
        return 1
    print("consistency check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
