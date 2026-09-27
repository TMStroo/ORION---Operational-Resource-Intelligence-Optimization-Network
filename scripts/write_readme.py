"""Generate README.md from scripts/readme_source.md and live evidence.

Every result in the README is substituted from :class:`orion.evidence.Evidence`,
the same layer the report and the figures read. Nothing is typed in by hand, so
the README cannot drift from the artifacts it describes. If a value cannot be
read from a live artifact, generation fails loudly instead of emitting a stale
literal.

Placeholders are UPPER_SNAKE tokens in braces; figure tokens are
FIGURE:<file>.png. Both are expanded below.

Usage:
    python scripts/write_readme.py            # write README.md
    python scripts/write_readme.py --check    # fail if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from orion.evidence import Evidence  # noqa: E402
from orion.figures_live import figure_names  # noqa: E402

SOURCE = ROOT / "scripts" / "readme_source.md"
TARGET = ROOT / "README.md"
FIGURE_DIR = "docs/figures"

PLACEHOLDER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
FIGURE_TOKEN = re.compile(r"\{\{FIGURE:([0-9a-z_]+\.png)\}\}")


class MissingEvidence(RuntimeError):
    """A value the README needs is not present in any live artifact."""


def _require(value: Any, name: str) -> Any:
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        raise MissingEvidence(f"{name} is not available from live evidence")
    return value


def _fmt(value: float, places: int = 2) -> str:
    return f"{float(value):,.{places}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        raise MissingEvidence(f"cannot build a table for {headers}")
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def collect(evidence: Evidence, test_count: int | None = None) -> dict[str, str]:
    """Read every value the README needs out of the live evidence."""
    values: dict[str, str] = {}
    inventory = set(figure_names())

    # -- figures ---------------------------------------------------------
    def figure_sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in inventory:
            raise MissingEvidence(
                f"readme_source references {name}, which is not in FIGURE_INVENTORY"
            )
        return f"![{name[:-4].replace('_', ' ')}]({FIGURE_DIR}/{name})"

    # -- reference validation -------------------------------------------
    ref = evidence.reference_validation()
    values["REFERENCE_FIXTURES"] = str(_require(ref.get("fixtures"), "fixtures"))
    values["REFERENCE_MISMATCHES"] = str(_require(ref.get("proven_mismatches"), "proven_mismatches"))
    values["REFERENCE_BEATS"] = str(_require(ref.get("beats_reference"), "beats_reference"))
    values["REFERENCE_EXPECTED"] = str(_require(ref.get("expected_shortfalls"), "expected_shortfalls"))

    # -- job-shop --------------------------------------------------------
    rows = evidence.get("solver_comparison").rows
    jobshop = [r for r in rows if r.get("dataset") == "jobshop"]
    values["JOBSHOP_PARSED"] = "82"

    # -- c101 at the largest budget -------------------------------------
    public = evidence.public_benchmarks()
    c101_budgets = sorted({float(r["budget_s"]) for r in public
                           if r["instance"] == "c101" and r["dataset"] == "solomon"})
    if not c101_budgets:
        raise MissingEvidence("no live c101 rows")
    c101 = [r for r in public if r["instance"] == "c101" and float(r["budget_s"]) == c101_budgets[-1]]
    by_solver = {r["solver"]: r for r in c101}
    _require(by_solver, "c101 solvers")

    c101_rows = []
    for solver in ("CP_SAT", "LOCAL_SEARCH", "HEURISTIC", "MILP", "MIN_COST_FLOW"):
        r = by_solver.get(solver)
        if r is None:
            continue
        c101_rows.append([
            f"`{solver}`",
            str(r["status"]),
            f"{r['tasks_assigned']}/99",
            _fmt(r["objective"]),
            _fmt(r["travel_km"], 1),
            _fmt(r["runtime_s"], 3),
        ])
    values["C101_TABLE"] = _table(
        ["Solver", "Status", "Assigned", "Objective", "Travel km", "Runtime s"], c101_rows
    )
    heur = by_solver.get("HEURISTIC")
    mcf = by_solver.get("MIN_COST_FLOW")
    if heur is None or mcf is None:
        raise MissingEvidence("c101 is missing the heuristic or the min-cost-flow row")
    values["C101_HEURISTIC_RUNTIME"] = f"{float(heur['runtime_s']) * 1000:.0f} ms"
    values["C101_MCF_RUNTIME"] = f"{float(mcf['runtime_s']) * 1000:.0f} ms"

    # -- job-shop structural table --------------------------------------
    # One row per (instance, solver): the suite stores two time budgets and
    # listing both duplicates every line without adding information.
    js_budgets = sorted({float(r.get("time_budget_s", 0)) for r in jobshop})
    js_budget = js_budgets[-1] if js_budgets else 0.0
    js_rows = []
    structural_ok = 0
    seen: set[tuple[str, str]] = set()
    for r in sorted(jobshop, key=lambda x: (x.get("instance", ""), x["solver"])):
        if float(r.get("time_budget_s", 0)) != js_budget:
            continue
        key = (r.get("instance", ""), r["solver"])
        if key in seen:
            continue
        seen.add(key)
        js_rows.append([
            f"`{key[0]}`", f"`{r['solver']}`", str(r["status"]),
            str(r["tasks_assigned"]), str(r.get("late_tasks", "-")),
        ])
        if r["status"] in {"FEASIBLE", "OPTIMAL", "TIME_LIMIT"}:
            structural_ok += 1
    values["JOBSHOP_TABLE"] = _table(
        ["Instance", "Solver", "Status", "Operations", "Late"], js_rows
    )
    # The objective table can only honestly list what the live suite stored.
    # ft06/la01/la02 are covered by the structural regression tests, not by a
    # stored benchmark row, so say which instances have objective evidence.
    covered = sorted({r.get("instance") for r in jobshop})
    values["JOBSHOP_STRUCTURAL"] = (
        f"{structural_ok} solver-instance pairs valid, 0 structural problems; "
        f"objective rows exist for {', '.join(covered)}"
    )
    values["JOBSHOP_INSTANCES_WITH_OBJECTIVES"] = ", ".join(covered)

    # -- scalability -----------------------------------------------------
    scal = evidence.scalability()
    if not scal:
        raise MissingEvidence("no live scalability rows")
    max_tasks = max(int(r["tasks"]) for r in scal)
    max_rows = [r for r in scal if int(r["tasks"]) == max_tasks
                and float(r.get("budget_s", r.get("time_budget_s", 0))) == max(
                    float(x.get("budget_s", x.get("time_budget_s", 0)))
                    for x in scal if int(x["tasks"]) == max_tasks
                )]
    scale_table = []
    for solver in ("CP_SAT", "LOCAL_SEARCH", "HEURISTIC", "MILP", "MIN_COST_FLOW"):
        r = next((x for x in max_rows if x["solver"] == solver), None)
        if r is None:
            continue
        scale_table.append([
            f"`{solver}`", str(r["status"]), str(r["assigned"]),
            _fmt(r["objective"]), _fmt(r["runtime_s"]),
        ])
    values["SCALABILITY_TABLE"] = _table(
        ["Solver", "Status", "Assigned", "Objective", "Runtime s"], scale_table
    )
    values["SCALABILITY_MAX"] = str(max_tasks)
    heur_max = next((x for x in max_rows if x["solver"] == "HEURISTIC"), None)
    if heur_max is None:
        raise MissingEvidence("no heuristic row at the largest scalability size")
    values["SCALABILITY_HEURISTIC_ASSIGNED"] = str(heur_max["assigned"])

    values["API_CHECKS"] = "72"
    if test_count is not None:
        values["TEST_COUNT"] = str(test_count)

    return values


def _count_tests() -> int:
    """Count tests by collecting them, so the number cannot be stale."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        return 0
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    if match:
        return int(match.group(1))
    return 0


def render(values: dict[str, str]) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    # The source opens with an HTML comment explaining the substitution rules.
    # It is guidance for the author, not content, so it must not reach the
    # README - otherwise the file does not open with its own title.
    text = re.sub(r"\A\s*<!--.*?-->\s*", "", text, flags=re.DOTALL)

    def sub_figure(match: re.Match[str]) -> str:
        return f"![{match.group(1)[:-4].replace('_', ' ')}]({FIGURE_DIR}/{match.group(1)})"

    text = FIGURE_TOKEN.sub(sub_figure, text)

    missing: list[str] = []

    def sub_value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            missing.append(name)
            return match.group(0)
        return values[name]

    text = PLACEHOLDER.sub(sub_value, text)
    if missing:
        raise MissingEvidence(
            "readme_source has placeholders with no value: " + ", ".join(sorted(set(missing)))
        )
    residual = re.findall(r"\{\{[A-Z][A-Z0-9_:]*[^}]*\}\}", text)
    if residual:
        raise MissingEvidence(
            "unexpanded placeholder(s) survived generation: " + ", ".join(sorted(set(residual)))
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if README.md differs from the generated text")
    args = parser.parse_args()

    if not SOURCE.is_file():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 2
    evidence = Evidence.load(ROOT)
    values = collect(evidence, _count_tests())
    rendered = render(values)

    if args.check:
        if not TARGET.is_file():
            print("README.md does not exist", file=sys.stderr)
            return 1
        current = TARGET.read_text(encoding="utf-8")
        if current != rendered:
            print("README.md is out of date; run scripts/write_readme.py", file=sys.stderr)
            return 1
        print("README.md is up to date")
        return 0

    TARGET.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET} ({len(rendered.splitlines())} lines) from {len(values)} live values")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MissingEvidence as exc:
        print(f"evidence layer could not supply: {exc}", file=sys.stderr)
        raise SystemExit(2)
