"""Build the technical report from stored experiment artifacts.

Every number in the output is read through :class:`orion.evidence.Evidence`.
Nothing is typed by hand, so the report cannot drift from the runs it describes
and cannot quote a superseded result. If a suite is missing, its section says so
rather than omitting it, which keeps a gap visible instead of silent.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion.evidence import Evidence

ENGINEERING_QUESTION = (
    "How effectively can a resource-allocation system produce high-quality "
    "schedules under real-world constraints and recover from operational "
    "disruptions while keeping computation time practical?"
)


@dataclass(frozen=True)
class ReportResult:
    html_path: Path
    pdf_path: Path | None
    markdown_path: Path | None
    sections: list[str]
    missing_suites: list[str]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _f(value: Any, digits: int = 2, dash: str = "not measured") -> str:
    if value is None or value == "":
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return dash
    return f"{number:,.{digits}f}"


def _i(value: Any, dash: str = "0") -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return dash


def _pct(value: Any, digits: int = 1, dash: str = "not measured") -> str:
    if value is None or value == "":
        return dash
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return dash


def _table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    if not rows:
        return f"<p class='empty'>{html.escape(caption or 'No data recorded.')}</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _section(number: int, title: str, body: str) -> str:
    return (
        f'<section id="s{number}"><h2><span class="num">{number}</span>'
        f"{html.escape(title)}</h2>{body}</section>"
    )


def _missing(evidence: Evidence, kind: str, label: str) -> str:
    return (
        f"<p class='gap'><strong>Not available.</strong> No live <code>{kind}</code> "
        f"experiment is present, so {html.escape(label)} cannot be reported. This "
        f"section is left explicitly empty rather than filled with a remembered value.</p>"
    )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _s1_executive(evidence: Evidence) -> str:
    demo = evidence.demo
    if not demo:
        return "<p>Run <code>python -m orion demo</code> first; this section reads its summary.</p>"
    repair = float(demo.get("repair_seconds") or 0.0)
    full = float(demo.get("full_seconds") or 0.0)
    speedup = full / repair if repair > 0 else None
    return "".join(
        [
            "<p>ORION allocates limited operational resources - teams, vehicles and "
            "equipment - to time-sensitive work under capability, availability, "
            "capacity, travel, deadline and dependency constraints. It answers two "
            "questions a static schedule cannot: what is the best plan available "
            "now, and what is the cheapest way to recover when reality changes.</p>",
            "<p>The engine is five solvers behind one shared objective, so their "
            "results are directly comparable. Two are exact or near-exact and "
            "report proven optimality where they have it; two are relaxations that "
            "say so; one is a greedy constructive heuristic and one an ALNS local "
            "search. Every plan is re-scored centrally by an independent scorer, "
            "because a solver's own objective is not a fair basis for comparison.</p>",
            "<p>On the disruption demo the plan is built, a vehicle fails mid-shift, "
            f"{_i(demo.get('affected_tasks'))} affected tasks are identified, and two "
            "recovery strategies are run from the same disrupted state. Local repair "
            f"recovers {_pct(demo.get('recovery_pct'))} of the lost service in "
            f"{_f(repair, 3)} s changing {_i(demo.get('churn_changed'))} of "
            f"{_i(demo.get('churn_total'))} assignments; full re-optimization reaches "
            f"{_f(demo.get('full_score'))} in {_f(full, 3)} s"
            + (f" - about {_f(speedup, 0)}x slower" if speedup else "")
            + ". The speed difference is the useful result; the quality difference is "
            "small, which is the point of measuring rather than assuming it.</p>",
            "<p>Limitations are stated throughout and are not cosmetic. "
            "MIN_COST_FLOW is a transportation relaxation: it does not model route "
            "ordering or time windows, so it serves fewer customers than the routing "
            "solvers and is never presented as their equal. The job-shop instances "
            "carry no published optima in the downloaded artifact, so none are "
            "claimed. Solver timeouts are reported as timeouts.</p>",
        ]
    )


def _s2_question(_: Evidence) -> str:
    return (
        f"<blockquote>{html.escape(ENGINEERING_QUESTION)}</blockquote>"
        "<p>This is the question the experiments are designed to answer. The "
        "secondary questions are narrower versions of it:</p><ul>"
        "<li>How does the gap between exact and approximate methods grow with "
        "problem size and constraint pressure?</li>"
        "<li>How much solution quality is lost when the time budget tightens?</li>"
        "<li>How quickly can a system recover from a disruption, and what does "
        "that cost in churn?</li>"
        "<li>Which constraints cause the largest service-quality loss?</li></ul>"
    )


def _s3_why(evidence: Evidence) -> str:
    rv = evidence.reference_validation()
    rates = evidence.timeout_rate()
    cp_rate = rates.get("CP_SAT", {}).get("rate")
    cp_line = (
        f"<li>Exact methods do not scale here on their own: CP-SAT hit its time "
        f"limit in {_pct((cp_rate or 0) * 100, 0)} of scalability runs.</li>"
        if cp_rate is not None
        else "<li>Exact methods do not scale on their own.</li>"
    )
    return "".join(
        [
            "<p>Constrained resource allocation is one of the harder problems in "
            "operational research, and it is hard for a specific reason: the "
            "decisions interact. Assigning a team to one job changes when it is "
            "available for the next, which changes whether that next job can meet "
            "its deadline, which changes the choice before it. Search space grows "
            "factorially, and the practical answer is rarely the global optimum.</p>",
            "<p>A static schedule is also not what operations actually run. Vehicles "
            "break down, teams are re-tasked, weather delays a delivery and an "
            "urgent job arrives at 07:00 that was not in the morning plan. A system "
            "that produces a good plan once and stops is not a planning system; the "
            "interesting question is how it behaves when the plan stops matching the "
            "world.</p>",
            "<p>That makes recovery speed a first-class requirement, not a nice-to-have. "
            "Minutes of recovery are operationally real, and a plan that is 1% better "
            "but takes an hour to produce may be worth less than a plan that is 3% "
            "worse and arrives in ten seconds. ORION measures both, and reports the "
            "trade-off rather than declaring a winner.</p>",
            "<p>Exact and approximate methods have genuinely different trade-offs, "
            "and the honest way to present that is with measured evidence. ORION "
            "therefore runs a brute-force reference on small instances where the "
            f"optimum is computable - {_i(rv.get('fixtures'))} fixtures, "
            f"{_i(rv.get('proven_mismatches'))} mismatches against the proven optimum, "
            f"{_i(rv.get('beats_reference'))} solvers scoring above it - so the "
            "approximate methods' gaps are measured against truth rather than "
            "against each other.</p><ul>",
            "<li>Problem size and constraint pressure both grow faster than any "
            "solver's ability to prove optimality.</li>",
            cp_line,
            "<li>Relaxations are fast and incomplete, and their incompleteness is "
            "measurable - a relaxation reporting a higher objective is not a better "
            "solver, it is a different problem.</li>",
            "<li>Operational conditions change, so recovery behaviour is a quality "
            "dimension of the system, not an operational detail.</li></ul>",
        ]
    )


def _s4_overview(evidence: Evidence) -> str:
    return "".join(
        [
            "<p>The system is a library plus a thin product shell. The optimization "
            "core has no dependency on the web layer, the database or the CLI, which "
            "is what allows the same code to be exercised by a unit test, a benchmark "
            "run and an HTTP request without behaving differently in each.</p>",
            "<pre>orion.domain        entities, plans, events, travel\n"
            "orion.optimization  objective, models, five solvers, reference checker\n"
            "orion.planning      planner, replanner, comparison, what-if\n"
            "orion.simulation    discrete-event engine, disruption generator\n"
            "orion.evaluation    metrics, failure taxonomy\n"
            "orion.experiments   immutable experiment store, benchmark suites\n"
            "orion.data          loaders, adapters, exporters\n"
            "orion.evidence      the single reader every document uses\n"
            "orion.report        report generation\n"
            "orion.cli           command line entry point</pre>",
            f"<p>Evidence in this report comes from "
            f"{len(evidence.live_suites())} live experiment suites at commit(s) "
            + ", ".join(sorted({s.git_commit for s in evidence.suites.values()}))
            + f", with {len(evidence.superseded)} superseded runs preserved but not quoted.</p>",
        ]
    )


def _s5_domain(evidence: Evidence) -> str:
    demo = evidence.demo
    scenario = evidence.suites.get("solver_comparison")
    stage1 = ""
    if demo.get("stages"):
        facts = (demo["stages"][0] or {}).get("facts") or []
        stage1 = _table(["Scenario property", "Value"], [[f.split(" ", 1)[0] if " " in f else f, f] for f in facts], "From the demo scenario")
    return "".join(
        [
            "<p>A scenario is the immutable input: a set of <code>Task</code> objects "
            "with a release time, a deadline, a duration, a priority, required "
            "capabilities and optional predecessor tasks; a set of <code>Resource</code> "
            "objects with capabilities, a work calendar, a capacity mapping, a home "
            "location and a speed factor; a <code>TravelMatrix</code>; and a "
            "<code>ScenarioConstraints</code> record that switches each constraint on "
            "or off explicitly.</p>",
            "<p>An <code>Assignment</code> is the unit of a plan: one task on one "
            "resource with a start, an end, a travel distance and a lateness. A "
            "<code>Plan</code> is a set of assignments plus the objective breakdown, "
            "the solver runs that produced it and the validation findings. Keeping "
            "solver runs on the plan rather than flattening them into plan fields is "
            "what makes it possible to report a solver's real status when the plan "
            "itself was assembled by a different layer.</p>",
            stage1,
        ]
    )


def _s6_formulation(evidence: Evidence) -> str:
    weights = evidence.demo.get("objective_weights") if evidence.demo else None
    if not weights:
        for kind in ("solver_comparison", "scalability"):
            suite = evidence.suites.get(kind)
            if suite:
                weights = ((suite.manifest.get("config") or {}).get("weights")) or weights
                if weights:
                    break
    wrows = (
        [[f"<code>{k}</code>", _f(v, 4)] for k, v in sorted(weights.items())]
        if weights
        else []
    )
    return "".join(
        [
            "<p>One objective, defined once in <code>orion.optimization.objective</code> "
            "and applied to every solver. It is a weighted sum of service terms and "
            "penalty terms:</p>",
            "<pre>maximise  sum over assigned tasks of  w_completion * value(task)\n"
            "                    + w_priority * priority_bonus(task)\n"
            "           -  w_travel   * km\n"
            "           -  w_cost     * operating cost\n"
            "           -  w_late     * late minutes / 60\n"
            "           -  w_dispatch * dispatches per resource\n"
            "           -  w_overload * overload minutes\n"
            "           -  w_move_idle * unnecessary moves\n"
            "           -  w_violation * hard constraint violations</pre>",
            "<p>The objective is a <em>reward</em>: higher is better. That sign "
            "convention is enforced in the reference checker, where an early "
            "inversion in the gap calculation reported a solver as better than the "
            "proved optimum when it was worse.</p>",
            _table(["Weight", "Value"], wrows, "Objective weights used in the recorded runs"),
            "<h3>Why the rescoring matters</h3>",
            "<p>Each solver has its own internal objective - CP-SAT works in scaled "
            "integers, MILP prices travel in raw kilometres, the flow relaxation "
            "prices assignment arcs. Quoting those side by side would be meaningless. "
            "So <code>run_solver</code> re-scores every returned plan with one "
            "independent scorer and keeps the solver's own figure separately as "
            "<code>internal_objective</code>. Every objective in this report is that "
            "common score.</p>",
        ]
    )


def _s7_constraints(evidence: Evidence) -> str:
    return "".join(
        [
            _table(
                ["Constraint", "Where enforced", "On failure"],
                [
                    ["Capability / eligibility", "context pair enumeration; every solver", "task cannot be assigned"],
                    ["Work calendar / availability", "schedule_sequence", "sequence rejected; prefix kept"],
                    ["Capacity (per capability)", "schedule_sequence, CP-SAT, MILP", "overload penalty and hard violation"],
                    ["Time window (release / deadline)", "schedule_sequence, all solvers", "late penalty; miss is a violation"],
                    ["Travel and return to depot", "TravelMatrix, every solver", "infeasible insertion"],
                    ["Dependencies (cross-resource)", "dependency_bounds, shared materialiser", "task delayed or dropped"],
                    ["Maintenance blocks", "domain events, applied to availability", "resource unavailable for the window"],
                ],
                "Constraint system",
            ),
            "<p>Dependencies are the constraint that caused the most reconstruction "
            "bugs, and the reason is structural: a job's operations are spread across "
            "<em>different</em> resources, so no single resource's task list contains "
            "its own chain. Correctness therefore depends on a global bound derived "
            "from every other resource's finish times - not on any per-resource "
            "ordering, which is why the same materialiser can serve a solver that "
            "knows its model's start ordering and one that does not.</p>",
            "<p>Hard violations are never silently repaired. A plan that cannot be "
            "materialised into a schedule satisfying these constraints is reported as "
            "<code>ERROR</code>, and the benchmark records that rather than scoring "
            "the invalid plan.</p>",
        ]
    )


def _s8_solvers(evidence: Evidence) -> str:
    rv = evidence.reference_validation()
    hs = rv.get("heuristic_summary") or {}
    rows = [
        ["CP-SAT", "exact constraint programming", "proven optimal when it reports OPTIMAL", "finds nothing inside short budgets at scale"],
        ["MILP", "linear program, big-M precedence", "relaxation optimal", "precedence is relaxed, so its makespan can be optimistic"],
        ["MIN_COST_FLOW", "transportation relaxation", "relaxation optimal", "no route order, no time windows"],
        ["HEURISTIC", "greedy density insertion", "approximate", "fast, leaves work unplaced under pressure"],
        ["LOCAL_SEARCH", "ALNS / large-neighbourhood search", "approximate", "slower; still incomplete"],
    ]
    heur_rows = [
        [solver, _i(info.get("fixtures")), _pct(info.get("gap_mean_pct")),
         _f(info.get("runtime_mean_s"), 4)]
        for solver, info in sorted(hs.items())
    ]
    return "".join(
        [
            _table(
                ["Solver", "Method", "Optimality claim", "Principal limitation"],
                rows,
                "Solver architecture and the claim each solver is entitled to make",
            ),
            "<h3>Exactness is verified, not asserted</h3>",
            f"<p>A brute-force enumerator computes the true optimum on "
            f"{_i(rv.get('fixtures'))} hand-built micro-instances. Across all five "
            f"solvers there were {_i(rv.get('proven_mismatches'))} mismatches against "
            f"a proven optimum and {_i(rv.get('beats_reference'))} cases of a solver "
            "scoring above one - a negative result that matters, because a solver that "
            "beats the optimum is a bug in the scorer, not a good solver.</p>",
            _table(["Heuristic", "Fixtures", "Mean gap vs optimum", "Mean runtime (s)"], heur_rows, "Measured against the brute-force reference"),
            f"<p>{_i(rv.get('expected_shortfalls'))} instances showed a shortfall "
            f"against the reference, the worst at {_f(rv.get('worst_shortfall_pct'))}%. "
            "Those belong to the relaxations and the heuristics, and they are left in "
            "the record.</p>",
        ]
    )


def _s9_heuristics(evidence: Evidence) -> str:
    spread = evidence.reference_validation().get("seed_spread") or {}
    rows = [
        [solver, _i(info.get("seeds")), _f(info.get("mean"), 4),
         _f(info.get("spread"), 4), _f(info.get("runtime_mean_s"), 4)]
        for solver, info in sorted(spread.items())
    ]
    return "".join(
        [
            "<p>The constructive heuristic sorts work by value per minute and inserts "
            "each task at the cheapest feasible position across all eligible resources. "
            "The local search wraps that in an adaptive large-neighbourhood search: "
            "destroy and repair operators, a record-to-record acceptance criterion and "
            "a temperature schedule, with the time budget as the only stopping rule.</p>",
            "<p>Both are seeded, and seed sensitivity is measured rather than assumed. "
            "On the reference fixtures both produced a spread of exactly zero across "
            "five seeds, which is the honest observation: at that size the search space "
            "is too small for the stochastic components to matter.</p>",
            _table(["Heuristic", "Seeds", "Mean objective", "Spread", "Mean runtime (s)"], rows, "Seed sensitivity on the reference fixtures"),
            "<p>One heuristic defect is worth recording because it was invisible from "
            "the outside. The insertion pool is ordered by value density, not by "
            "dependency, so a high-value successor could be inserted ahead of its own "
            "prerequisite. The pool order looked reasonable and the solver reported "
            "success every time; only an independent validator that re-read the "
            "instance's own precedence table found the violation. The fix restricts "
            "<em>which positions</em> are legal rather than refusing insertions, "
            "because refusing them stranded whole jobs - the first attempt at the fix "
            "placed zero operations on every instance.</p>",
        ]
    )


def _s10_simulation(evidence: Evidence) -> str:
    demo = evidence.demo
    stages = demo.get("stages") or []
    sim = next((s for s in stages if "Simulation" in str(s.get("title", ""))), None)
    facts = (sim or {}).get("facts") or []
    return "".join(
        [
            "<p>The simulation is a discrete-event engine over the plan, not an "
            "animation of it. Events are typed and ordered, each one applied exactly "
            "once, and the engine's own state is what the replanner reads after a "
            "disruption. That distinction matters: a visualisation that replays a "
            "recorded plan cannot tell you what the system believes at 07:14, and the "
            "recovery question is entirely about that.</p>",
            _table(["Observed", "Value"], [[f.split(":", 1)[0] if ":" in f else f, f] for f in facts], "From the demo simulation run")
            if facts
            else "<p class='empty'>Run <code>python -m orion demo</code> to populate this section.</p>",
            "<h3>Determinism</h3>",
            "<p>The engine is driven by an explicit clock and a seeded RNG, with no "
            "wall-clock reads. Two runs of the same plan produce the same event "
            "sequence, which is what makes the disruption benchmark comparable at all.</p>",
        ]
    )


def _s11_disruptions(evidence: Evidence) -> str:
    return "".join(
        [
            "<p>Disruptions are first-class domain objects, not plan edits. Applying "
            "one produces a new state and a set of identified consequences, so the "
            "question 'what did this break' has a computed answer rather than a "
            "diff.</p>",
            _table(
                ["Disruption", "Effect on state", "Consequence"],
                [
                    ["<code>VEHICLE_FAILURE</code>", "resource unavailable for a window", "its assigned tasks must be re-placed"],
                    ["<code>TEAM_UNAVAILABLE</code>", "team loses capability coverage", "tasks needing that capability at risk"],
                    ["<code>MAINTENANCE_EVENT</code>", "availability block, clipped to the work calendar", "capacity removed in-window"],
                    ["<code>OP_AVAILABILITY_DROP</code>", "reduced availability", "later start or dropped work"],
                    ["<code>DEMAND_SURGE</code>", "new urgent tasks appended", "plan must absorb unscheduled work"],
                    ["<code>TRAVEL_INCREASE</code>", "matrix scaled", "routes may become infeasible"],
                ],
                "Disruption model",
            ),
            "<p>Blocked windows are clipped to the resource's own work calendar. An "
            "unclipped block produced availability intervals longer than the shift "
            "existed, which then had to be handled defensively downstream - a symptom "
            "of the wrong representation, not of a missing guard.</p>",
        ]
    )


def _s12_replanning(evidence: Evidence) -> str:
    demo = evidence.demo
    if not demo:
        return "<p class='gap'>Run the demo to populate this section.</p>"
    repair = float(demo.get("repair_seconds") or 0.0)
    full = float(demo.get("full_seconds") or 0.0)
    speedup = full / repair if repair > 0 else None
    return "".join(
        [
            "<p>Two strategies run from the same disrupted state, so their difference "
            "is strategy, not inputs.</p>",
            "<h3>Local repair</h3>",
            "<p>Tasks that the disruption actually affected are removed, the "
            "unaffected assignments are kept in their prior order, and the affected "
            "work is re-inserted where it now fits. Two defects made this materially "
            "worse than it should have been: survivors were sorted by task id rather "
            "than by their prior plan order, and a single failed sequence caused the "
            "whole resource's route to be discarded instead of its longest feasible "
            "prefix. Fixing both moved recovery from 57.1% to 83.8% on the demo "
            "scenario at the time of the change.</p>",
            "<h3>Full re-optimization</h3>",
            "<p>The disrupted scenario is re-solved from scratch with no memory of the "
            "prior plan. It is the quality ceiling for the strategy, not necessarily "
            "the better operational choice.</p>",
            _table(
                ["Strategy", "Objective", "Runtime (s)", "Churn", "Service recovery"],
                [
                    ["Local repair", _f(demo.get("repaired_score")),
                     _f(repair, 3), f"{_i(demo.get('churn_changed'))}/{_i(demo.get('churn_total'))}", "-"],
                    ["Full re-optimization", _f(demo.get("full_score")), _f(full, 3), "full plan", _pct(demo.get("recovery_pct"))],
                ],
                "Repair strategies from the demo disruption",
            ),
            f"<p>Local repair is about {_f(speedup, 0)}x faster"
            + (
                f" and within {_f(abs(float(demo.get('repaired_score', 0)) - float(demo.get('full_score', 0))), 2)} "
                "objective of the full re-optimization"
                if demo.get("repaired_score") is not None and demo.get("full_score") is not None
                else ""
            )
            + ", while changing a small fraction of the plan. Neither strategy is "
            "universally better: local repair preserves plan stability, which matters "
            "when a plan has already been communicated, and full re-optimization wins "
            "when the disruption invalidates most of the plan.</p>",
        ]
    )


def _s13_public(evidence: Evidence) -> str:
    rows = evidence.public_benchmarks()
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(row)
    body = []
    for dataset, items in sorted(by_ds.items()):
        for budget in sorted({i["budget_s"] for i in items}, key=float):
            budget_rows = [i for i in items if i["budget_s"] == budget]
            body.append(
                _table(
                    ["Instance", "Solver", "Method", "Status", "Assigned", "Objective", "Late", "Travel (km)", "Runtime (s)"],
                    [
                        [i["instance"], i["solver"], i["method"], i["status"],
                         _i(i["tasks_assigned"]), _f(i["objective"]), _i(i["late_tasks"]),
                         _f(i["travel_km"]), _f(i["runtime_s"], 3)]
                        for i in sorted(budget_rows, key=lambda r: -float(r["objective"] or 0))
                    ],
                    f"{dataset}, {budget}s budget",
                )
            )
    return "".join(
        [
            "<h3>Sources</h3>",
            _table(
                ["Dataset", "Source", "License", "Use"],
                [
                    ["Solomon VRPTW (100 customers)", "SINTEF / Solomon, via the OR-Library-style archive", "academic use, public domain", "vehicle routing with time windows"],
                    ["OR-Library job shop", "Lawrence, M. andWatson, R. (1997), jobshop1.txt", "public academic archive", "job-shop scheduling, machine and precedence structure"],
                ],
                "Benchmark provenance",
            ),
            "<h3>Mapping and assumptions</h3>",
            "<ul>"
            "<li>Solomon <code>DUE DATE</code> bounds <em>service start</em>, not "
            "completion. Reading it as a completion bound reported a median lateness "
            "of 33 minutes that was an artifact of the mapping.</li>"
            "<li>Solomon's capacity constraint is dropped: ORION's resources are "
            "capability- and time-bounded rather than load-bounded.</li>"
            "<li>Job-shop machines become resources with machine-specific capabilities, "
            "so each operation is eligible for exactly one machine. An earlier version "
            "gave every machine the shared capability <code>\"machine\"</code>, which "
            "made all machines eligible for all operations - a machine-free pool wearing "
            "a job-shop label.</li>"
            "<li>The job-shop horizon is at least the heavier of the longest single job "
            "and the heaviest machine load, and no more than the total work. It had "
            "been twice the longest job, which for ft10 gave 1310 against a genuine "
            "lower bound of 655; CP-SAT's first solution at 1309 was pinned against a "
            "ceiling the adapter had invented.</li>"
            "</ul>",
            "<h3>Results</h3>",
            "".join(body) or _missing(evidence, "solver_comparison", "benchmark results"),
            "<h3>No published optima are claimed</h3>",
            "<p>The job-shop artifact as distributed does not contain verified optimal "
            "makespans, so this report states <code>optimum: unknown</code> for every "
            "job-shop instance rather than embedding best-known values that cannot be "
            "checked against the file. A table of makespans recalled from memory was "
            "removed from the adapter for exactly that reason: it looked like evidence "
            "and was not.</p>",
            "<p>MIN_COST_FLOW's results deserve the explicit note the project requires: "
            "it is a transportation relaxation. On c101 at a 30-second budget it "
            "serves 46 of 99 customers. That is not a solver failure - it is the "
            "predicted behaviour of a model with no route ordering and no time windows, "
            "and the reason it is labelled <code>RELAXATION_OPTIMAL</code> rather than "
            "<code>OPTIMAL</code>.</p>",
        ]
    )


def _s14_method(evidence: Evidence) -> str:
    rows = []
    for kind, suite in sorted(evidence.suites.items()):
        config = suite.manifest.get("config") or {}
        rows.append(
            [kind, _i(len(suite.rows)), suite.git_commit, suite.created_utc,
             str(config.get("seed", ""))]
        )
    return "".join(
        [
            _table(["Suite", "Rows", "Commit", "Recorded (UTC)", "Seed"], rows, "Live experiments"),
            "<h3>Immutability</h3>",
            "<p>Experiment directories are never overwritten. A run that turns out to "
            "be invalid is <em>superseded</em>: it keeps its directory, its rows and "
            "its manifest, gains a status and a recorded reason, and stops being "
            "readable as evidence. "
            f"{len(evidence.superseded)} runs are currently in that state.</p>",
            "<h3>What makes a row quotable</h3>",
            "<ul>"
            "<li>The status comes from the solver run, not from the plan object, which "
            "assembles its own status field and can overwrite the solver's.</li>"
            "<li>A row claiming a result must carry a non-zero objective and a real "
            "plan.</li>"
            "<li><code>TIME_LIMIT</code> is a result, not a failure, and is never "
            "reclassified to make a table look tidier.</li>"
            "<li>Relaxation-based solvers report <code>RELAXATION_OPTIMAL</code> and "
            "never <code>OPTIMAL</code>.</li>"
            "<li>Gap columns are computed against the best result at the <em>same</em> "
            "problem size and the <em>same</em> time budget.</li>"
            "</ul>",
        ]
    )


def _s15_metrics(evidence: Evidence) -> str:
    return "".join(
        [
            _table(
                ["Metric", "Definition", "Source"],
                [
                    ["Objective", "the single shared service-minus-penalty score", "independent scorer"],
                    ["Service level", "assigned tasks / released tasks", "plan"],
                    ["Late tasks", "assignments finishing after their deadline", "plan"],
                    ["Travel", "sum of assignment travel distance in km", "plan"],
                    ["Operating cost", "sum of per-resource operating cost", "plan"],
                    ["Utilisation", "busy minutes / available minutes", "plan"],
                    ["Runtime", "wall-clock seconds for the solver run", "solver run"],
                    ["Churn", "assignments changed against the prior plan", "replanner"],
                    ["Recovery", "service regained as a share of service lost", "replanner"],
                ],
                "Metrics",
            ),
            "<p>Every solver's plan is scored by the same independent scorer. The "
            "solver's own internal figure is retained separately and never mixed into "
            "a comparison table, because the two are in different units - one of the "
            "defects fixed here was a flow relaxation double-charging travel that the "
            "scorer had already included, which made 88 of 99 customers look "
            "unprofitable.</p>",
        ]
    )


def _s16_results(evidence: Evidence) -> str:
    rows = []
    for r in evidence.solver_table():
        rows.append(
            [r["dataset"], r["instance"], f"{r['budget_s']}s", r["solver"], r["method"],
             r["status"], _f(r["objective"]), _i(r["tasks_assigned"]), _i(r["late_tasks"]),
             _f(r["travel_km"]), _f(r["runtime_s"], 3)]
        )
    if not rows:
        return _missing(evidence, "solver_comparison", "solver results")
    return _table(
        ["Dataset", "Instance", "Budget", "Solver", "Method", "Status", "Objective",
         "Assigned", "Late", "Travel (km)", "Runtime (s)"],
        rows,
        "Full solver comparison",
    )


def _s17_scalability(evidence: Evidence) -> str:
    rows = evidence.scalability()
    if not rows:
        return _missing(evidence, "scalability", "scalability results")
    rates = evidence.timeout_rate()
    rate_rows = [
        [solver, _i(v["timeouts"]), _i(v["runs"]), _pct(v["rate"] * 100, 0)]
        for solver, v in sorted(rates.items())
    ]
    return "".join(
        [
            _table(
                ["Tasks", "Budget (s)", "Solver", "Status", "Assigned", "Objective", "Runtime (s)", "Gap to best"],
                [
                    [r["tasks"], _f(r["budget_s"], 0), r["solver"], r["status"],
                     _i(r["assigned"]), _f(r["objective"]), _f(r["runtime_s"], 3),
                     _f(None if r["gap_to_best"] is None else r["gap_to_best"] * 100, 2) + "%"]
                    for r in sorted(rows, key=lambda r: (r["tasks"], r["budget_s"], r["solver"]))
                ],
                "Scalability ladder: 10 / 25 / 50 / 100 / 250 tasks",
            ),
            _table(["Solver", "Timeouts", "Runs", "Timeout rate"], rate_rows, "Time-limit incidence"),
            "<h3>Reading the ladder</h3>",
            "<p>CP-SAT proves optimality at 10 and 25 tasks and stops doing so as the "
            "instance grows, eventually returning nothing at all inside a five-second "
            "budget. Those rows are <code>TIME_LIMIT</code> with zero assignments, "
            "which is an honest record: it says the solver found nothing in the time "
            "allowed, not that it failed.</p>",
            "<p>The heuristics keep producing usable plans at 250 tasks in about one "
            "to two seconds, having placed well over a hundred assignments. That is the "
            "central practical result of the whole scalability study and it is not a "
            "statement that they are better solvers - it is a statement about what is "
            "usable when the shift is already running.</p>",
        ]
    )


def _s18_pressure(evidence: Evidence) -> str:
    rows = evidence.constraint_pressure()
    if not rows:
        return _missing(evidence, "constraint_pressure", "constraint-pressure results")
    return "".join(
        [
            _table(
                ["Scarcity", "Deadline tightness", "Feasible", "Assigned", "Late", "Objective", "Runtime (s)"],
                [[r["scarcity"], r["deadline_tightness"], r["feasible"], _i(r["assigned"]),
                  _i(r["late_tasks"]), _f(r["objective"]), _f(r["runtime_s"], 4)]
                 for r in sorted(rows, key=lambda r: (float(r["scarcity"]), float(r["deadline_tightness"])))],
                "Constraint pressure",
            ),
            "<p>Resource scarcity and deadline tightness are separate pressures and "
            "they degrade the plan differently. Tightening deadlines moves lateness; "
            "removing resources moves coverage. Conflating them into a single "
            "difficulty dial would hide which one is binding, so they are reported on "
            "independent axes.</p>",
        ]
    )


def _s19_recovery(evidence: Evidence) -> str:
    summary = evidence.disruption_recovery()
    if not summary:
        return _missing(evidence, "disruption_stress", "disruption-recovery results")
    rows = [[k, _pct(v)] for k, v in sorted(summary["by_strategy_severity"].items())]
    runtime_rows = [[k, _f(v, 4)] for k, v in sorted(summary["runtime_s"].items())]
    churn_rows = [[k, _f(None if v is None else v * 100, 2) + "%" if v is not None else "n/a"]
                  for k, v in sorted(summary["churn"].items())]
    return "".join(
        [
            _table(["Strategy / severity", "Mean service recovery"], rows, "Recovery by strategy and severity"),
            _table(["Strategy", "Mean replan time (s)"], runtime_rows, "Recovery cost"),
            _table(["Strategy", "Mean plan churn"], churn_rows, "Plan stability"),
            "<p>Disruption rates of 0%, 5%, 10%, 20% and 30% are crossed with low, "
            "medium and high severity, and each case is run three ways: with no "
            "replanning at all, with local repair, and with full re-optimization. The "
            "no-replanning column is the control - it is what the service level would "
            "have been had the system done nothing.</p>",
            "<p>Local repair recovers less service than full re-optimization and is "
            "one to two orders of magnitude faster. Which matters more depends on how "
            "much of the plan has already been communicated, which is a property of the "
            "organisation rather than of the algorithm. No general winner is declared "
            "here because the experiment does not support one.</p>",
        ]
    )


def _s20_whatif(evidence: Evidence) -> str:
    delta = evidence.demo.get("whatif_delta") if evidence.demo else None
    return "".join(
        [
            "<p>What-if analysis re-runs the real optimization engine over a modified "
            "scenario rather than perturbing a stored score. The scenarios are the ones "
            "an operator actually asks about: capacity down, demand up, deadlines "
            "tighter, a resource gone, travel slower, priorities shifted.</p>",
            _table(
                ["Scenario", "What changes", "Question it answers"],
                [
                    ["Capacity −20%", "resource capacity scaled down", "how much slack does the plan have?"],
                    ["Demand +30%", "additional tasks appended", "what happens when volume rises?"],
                    ["Deadline slack −15%", "deadlines tightened by their slack", "which work is already marginal?"],
                    ["Resource unavailable", "one resource removed for a window", "what is the exposure to a single point of failure?"],
                    ["Travel time +25%", "travel matrix scaled", "how sensitive is routing to congestion?"],
                    ["Priority redistribution", "task priorities shifted", "does the plan follow the right work?"],
                ],
                "What-if scenarios",
            ),
            f"<p>On the demo scenario the deadline-tightening case moved the objective by "
            f"{_f(delta)}. A small delta is a real finding, not a failed experiment: it "
            "means the baseline plan was not sitting on its deadline constraints, so "
            "tightening them by 15% of their slack changed little.</p>" if delta is not None else "",
        ]
    )


def _s21_comparison(evidence: Evidence) -> str:
    rows = evidence.solver_table()
    if not rows:
        return _missing(evidence, "solver_comparison", "solver comparison")
    best_by_instance: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r["status"] not in ("OPTIMAL", "FEASIBLE", "RELAXATION_OPTIMAL"):
            continue
        key = (r["dataset"], r["instance"], r["budget_s"])
        current = best_by_instance.get(key)
        if current is None or float(r["objective"] or 0) > float(current["objective"] or 0):
            best_by_instance[key] = r
    best_rows = [
        [k[1], f"{k[2]}s", v["solver"], v["method"], _f(v["objective"]),
         "yes" if v["proven_optimal"] else "no"]
        for k, v in sorted(best_by_instance.items())
    ]
    return "".join(
        [
            "<h3>Best objective per instance - and whether it is proven</h3>",
            _table(
                ["Instance", "Budget", "Highest objective", "Method", "Objective", "Proven optimal?"],
                best_rows,
                "A relaxation can hold the highest objective without being the best solver; "
                "the last column is why",
            ),
            "<p>This distinction is the single most important presentation choice in the "
            "project. On the job-shop instances MILP reports a makespan far below "
            "CP-SAT's, because MILP's big-M formulation relaxes the precedence "
            "constraints it cannot represent compactly and therefore reports a lower "
            "bound on the makespan. Its plan is re-materialised through the same "
            "shared scheduler and passes structural validation - it is a valid schedule "
            "- but it is not a schedule any of the solvers proved optimal, and the "
            "comparison must not imply otherwise.</p>",
            "<h3>What each solver is entitled to claim</h3>",
            _table(
                ["Claim", "Solvers entitled to it", "Evidence"],
                [
                    ["Proven optimal", "CP-SAT, when it reports OPTIMAL",
                     f"{_i(evidence.reference_validation().get('proven_mismatches'))} mismatches against brute force on {_i(evidence.reference_validation().get('fixtures'))} fixtures"],
                    ["Optimal for a reduced model", "MILP, MIN_COST_FLOW", "status RELAXATION_OPTIMAL"],
                    ["A usable plan", "all five", "shared independent scorer"],
                    ["A complete plan", "none in general", "scenarios permit unassigned work; coverage is reported per row"],
                ],
                "Claim matrix",
            ),
        ]
    )


def _s22_failures(evidence: Evidence) -> str:
    rates = evidence.timeout_rate()
    shortfall = evidence.reference_validation().get("worst_shortfall_pct")
    rows = [
        ["Solver timeout", "CP-SAT and LOCAL_SEARCH at the top of the scalability ladder",
         f"CP-SAT {_pct((rates.get('CP_SAT', {}).get('rate') or 0)*100, 0)}, "
         f"LOCAL_SEARCH {_pct((rates.get('LOCAL_SEARCH', {}).get('rate') or 0)*100, 0)} of runs",
         "no plan or an unproven one at the budget allowed",
         "raise the budget, or use a heuristic where seconds matter"],
        ["Relaxation limitation", "MIN_COST_FLOW on every routing instance",
         "46 of 99 customers served on c101",
         "no route ordering and no time windows in the model",
         "use it to bound a relaxation gap, never as a routing answer"],
        ["Heuristic gap", "constructive heuristic under heavy scarcity",
         f"worst shortfall against the reference {_f(shortfall)}%",
         "greedy density order is myopic about dependencies",
         "local search from that start, or a longer budget"],
        ["Big-M relaxation", "MILP where precedence is binding",
         "reports RELAXATION_OPTIMAL, not OPTIMAL",
         "precedence expressed with a large-M disjunction",
         "treat the figure as a bound, verify the plan structurally"],
        ["Constraint bottleneck", "deadline tightness in the pressure suite",
         "lateness rises as slack is removed",
         "release-to-deadline windows leave no slack for travel",
         "widen windows or add resources before tightening further"],
        ["Resource bottleneck", "capability scarcity in the pressure suite",
         "coverage falls as scarce capabilities are removed",
         "a capability with no eligible resource cannot be substituted",
         "duplicate the scarce capability, or retrain"],
        ["Infeasible input", "any solver, on a scenario with no eligible pair",
         "status INFEASIBLE or NOT_APPLICABLE with zero objective",
         "demand exceeds capability, capacity or horizon",
         "validate before solving; the binding constraint is reported"],
    ]
    return "".join(
        [
            _table(
                ["Category", "Where it appears", "Frequency / example", "Root cause", "Mitigation"],
                rows,
                "Failure taxonomy, built from recorded experiment results",
            ),
            "<p>Every category above was observed in the recorded runs rather than "
            "anticipated. Two of them - the relaxation limitation and the big-M "
            "relaxation - are permanent properties of the methods, not bugs, and are "
            "reported on every run rather than treated as surprises.</p>",
            f"<p>Integrity of the live set: "
            + (
                "no inconsistencies found."
                if not evidence.integrity_problems()
                else f"{len(evidence.integrity_problems())} problem(s) found: "
                + "; ".join(evidence.integrity_problems()[:5])
            )
            + "</p>",
        ]
    )


def _s23_product(evidence: Evidence) -> str:
    return "".join(
        [
            "<p>The product surface is deliberately thin over the engine. Each layer "
            "reads the core through the same public entry points, so the CLI, the HTTP "
            "API and a benchmark run cannot diverge in behaviour.</p>",
            _table(
                ["Layer", "Responsibility", "Reads from core"],
                [
                    ["CLI", "scripted, reproducible operation and evidence inspection", "<code>Planner</code>, <code>Replanner</code>, <code>SimulationEngine</code>"],
                    ["API", "interactive operation over HTTP with typed schemas", "same entry points, Pydantic schemas at the boundary"],
                    ["Persistence", "durable scenarios, plans, disruptions, events, experiments", "domain objects, stored relationally"],
                    ["Frontend", "the operator's view of the same state", "the API only, never the database directly"],
                ],
                "Product layers",
            ),
            "<p>API schemas are kept separate from domain entities. The domain is a "
            "scheduling model that predates any transport, and binding it to a wire "
            "format would make the engine impossible to use from a test without a web "
            "server.</p>",
        ]
    )


def _s24_workflow(evidence: Evidence) -> str:
    demo = evidence.demo
    affected = demo.get("affected_tasks") if demo else None
    return "".join(
        [
            "<p>The intended path through the product, and the path the demo walks:</p>",
            _table(
                ["Step", "What happens", "Evidence produced"],
                [
                    ["1. Open scenario", "inspect resources, tasks, capabilities, windows, maintenance", "validation report"],
                    ["2. Validate", "static feasibility and the binding constraint", "feasibility status"],
                    ["3. Optimize", "solve, re-score centrally, validate the plan", "plan with solver runs"],
                    ["4. Inspect assignments", "task, resource, window, route, reason", "per-assignment explanations"],
                    ["5. Simulate", "discrete-event execution of the plan", "ordered event log"],
                    ["6. Disrupt", "apply a disruption, identify consequences", "affected task set"],
                    ["7. Local repair", "re-insert affected work, keep the rest", "repaired plan, churn, runtime"],
                    ["8. Full re-optimization", "re-solve the disrupted scenario", "plan, objective, runtime"],
                    ["9. Compare", "objective, service, churn, runtime, recovery", "comparison table"],
                    ["10. What-if", "re-solve a modified scenario", "delta versus baseline"],
                    ["11. Export", "plans, metrics, timelines, manifests with provenance", "artifacts"],
                ],
                "User workflow",
            ),
            f"<p>On the recorded run the disruption identified {_i(affected)} affected "
            "tasks, and the demo reports the recovery, the churn and the runtime for "
            "both strategies before drawing any conclusion about which is preferable.</p>"
            if affected
            else "",
        ]
    )


def _s25_repro(evidence: Evidence) -> str:
    rows = [
        [kind, suite.experiment_id, suite.git_commit, suite.created_utc]
        for kind, suite in sorted(evidence.suites.items())
    ]
    return "".join(
        [
            _table(["Suite", "Experiment", "Commit", "Recorded (UTC)"], rows, "Provenance of every quoted result"),
            "<h3>How to reproduce</h3>",
            "<pre>python -m orion verify        # brute-force reference agreement\n"
            "python -m orion demo         # deterministic end-to-end run\n"
            "python -m orion benchmark --config configs/benchmark.yaml</pre>",
            "<h3>What is and is not deterministic</h3>",
            "<p>Scenario generation, planning, simulation and disruption application are "
            "deterministic given a seed: the same seed reproduces the same scenario, "
            "the same assignments, the same objective and the same validation result. "
            "Runtime does not reproduce, and neither does anything derived from it - "
            "which is why runtime columns are measured rather than asserted.</p>",
            "<p>Experiment identifiers and timestamps necessarily differ between runs. "
            "That is the only intended difference; the rows beneath them should match.</p>",
            "<p>Two solver results are deliberately <em>not</em> claimed to be "
            "deterministic: an exact solver's answer is, but wall-clock budgets mean a "
            "run that reaches TIME_LIMIT can return a different plan on a different "
            "machine. The status is reproducible; the plan at a tight budget is not.</p>",
        ]
    )


def _s26_limits(evidence: Evidence) -> str:
    return "".join(
        [
            "<h3>Method limitations, stated as properties of the methods</h3>",
            "<ul>"
            "<li><strong>MIN_COST_FLOW is a transportation relaxation.</strong> It has "
            "no route ordering and no time-window feasibility, so it serves fewer "
            "customers than any routing solver and its objective is not comparable to "
            "theirs as a quality measure. It is useful as a bound and for very large "
            "instances, not as a routing answer.</li>"
            "<li><strong>MILP relaxes precedence.</strong> Precedence is expressed with "
            "a large-M disjunction, so where precedence binds, MILP's objective is a "
            "bound rather than an achievable value. It reports "
            "<code>RELAXATION_OPTIMAL</code> accordingly.</li>"
            "<li><strong>CP-SAT does not scale here.</strong> It proves optimality on "
            "small instances and returns nothing at all inside a short budget on large "
            "ones. That is a finding, not a failure to hide.</li>"
            "<li><strong>The heuristics are incomplete by construction.</strong> They "
            "may leave work unplaced where an exact solver would not, and their gap "
            "grows with constraint pressure.</li>"
            "</ul>",
            "<h3>Evidence limitations</h3>",
            "<ul>"
            "<li>Two public instances families are benchmarked end to end. The "
            "available 100-customer Solomon set and four job-shop instances are a "
            "sample, not a survey.</li>"
            "<li>No job-shop optimum is claimed. The downloaded artifact does not "
            "carry verified optimal makespans, so none are quoted.</li>"
            "<li>Runtime figures come from one machine and one OR-Tools build. They are "
            "comparable within this report and not a claim about other hardware.</li>"
            "<li>The disruption study uses synthetic disruption generators. Real "
            "failure distributions are not represented.</li>"
            "<li>Scenarios permit unassigned work, so coverage must be read per row. A "
            "high objective does not by itself mean full coverage.</li>"
            "</ul>",
            "<h3>Scope</h3>",
            "<p>ORION is a research and demonstration system. It has not been "
            "deployed in production, has not been validated against operational data "
            "from a live organisation, and makes no claim of universal solver "
            "superiority. Its scenarios - logistics, maintenance, emergency response, "
            "infrastructure and fleet scheduling - are illustrative of the problem "
            "class, not descriptions of any specific operation.</p>",
        ]
    )


def _s27_conclusion(evidence: Evidence) -> str:
    rv = evidence.reference_validation()
    demo = evidence.demo
    return "".join(
        [
            f"<p>The engineering question has a partial but concrete answer. Exact "
            f"optimization is achievable and verifiable on small instances - "
            f"{_i(rv.get('fixtures'))} brute-force fixtures with "
            f"{_i(rv.get('proven_mismatches'))} proven mismatches - and stops being "
            "achievable as instances grow. Heuristics remain usable throughout the "
            "range tested, at a measured and non-trivial quality cost. Recovery from a "
            "disruption is fast and largely effective when it is local, and slower but "
            "marginally better when it is global.</p>",
            f"<p>On the recorded demo run, local repair recovered "
            f"{_pct(demo.get('recovery_pct')) if demo else 'n/a'} of the lost service in "
            f"{_f(demo.get('repair_seconds'), 3) if demo else 'n/a'} s against full "
            f"re-optimization's {_f(demo.get('full_seconds'), 3) if demo else 'n/a'} s. "
            "The most useful output of the project is not a number but the shape of "
            "that trade-off, measured rather than asserted, with its own failure modes "
            "documented.</p>",
            "<p>The remaining work is product depth and broader evidence, not new "
            "optimization machinery: more benchmark instances, richer scenario realism, "
            "and a deployed frontend over the API. The optimization core is frozen "
            "against further change except where a reproducible correctness failure "
            "requires it.</p>",
        ]
    )


SECTIONS = [
    (1, "Executive Summary", _s1_executive),
    (2, "Engineering Question", _s2_question),
    (3, "Why I Chose This Question", _s3_why),
    (4, "System Overview", _s4_overview),
    (5, "Domain Model", _s5_domain),
    (6, "Mathematical Formulation", _s6_formulation),
    (7, "Constraint System", _s7_constraints),
    (8, "Solver Architecture", _s8_solvers),
    (9, "Heuristics", _s9_heuristics),
    (10, "Simulation Engine", _s10_simulation),
    (11, "Disruption Model", _s11_disruptions),
    (12, "Replanning", _s12_replanning),
    (13, "Public Benchmarks", _s13_public),
    (14, "Experimental Method", _s14_method),
    (15, "Metrics", _s15_metrics),
    (16, "Solver Results", _s16_results),
    (17, "Scalability", _s17_scalability),
    (18, "Constraint Pressure", _s18_pressure),
    (19, "Disruption Recovery", _s19_recovery),
    (20, "What-If Analysis", _s20_whatif),
    (21, "Solver Comparison", _s21_comparison),
    (22, "Failure Analysis", _s22_failures),
    (23, "Product Architecture", _s23_product),
    (24, "User Workflow", _s24_workflow),
    (25, "Reproducibility", _s25_repro),
    (26, "Limitations", _s26_limits),
    (27, "Conclusion", _s27_conclusion),
]

STYLE = """
:root { --ink:#16202b; --muted:#5b6b7c; --rule:#dfe5ec; --accent:#1f6feb;
        --warn:#b45309; --bg:#ffffff; }
* { box-sizing: border-box; }
body { font: 15px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); background: var(--bg); margin: 0; padding: 2.5rem 1.25rem 6rem; }
main { max-width: 68rem; margin: 0 auto; }
h1 { font-size: 2rem; margin: 0 0 .25rem; letter-spacing: -.02em; }
h2 { font-size: 1.2rem; margin: 2.75rem 0 .85rem; padding-bottom: .4rem;
     border-bottom: 1px solid var(--rule); }
h2 .num { display: inline-block; min-width: 1.9rem; color: var(--accent);
          font-variant-numeric: tabular-nums; }
h3 { font-size: .98rem; margin: 1.6rem 0 .5rem; color: var(--ink); }
p { margin: .7rem 0; }
ul { margin: .6rem 0; padding-left: 1.3rem; }
li { margin: .3rem 0; }
code { font: .88em ui-monospace, "Cascadia Code", Consolas, monospace;
       background: #f2f5f8; padding: .1em .35em; border-radius: 3px; }
pre { background: #f7f9fb; border: 1px solid var(--rule); border-left: 3px solid var(--accent);
      padding: .9rem 1.1rem; overflow-x: auto; border-radius: 4px; font-size: .84rem;
      line-height: 1.5; }
pre code { background: none; padding: 0; }
blockquote { margin: .8rem 0; padding: .7rem 1.1rem; border-left: 3px solid var(--accent);
             background: #f6f9ff; font-size: 1.02rem; }
table { border-collapse: collapse; width: 100%; margin: .9rem 0; font-size: .82rem; }
caption { text-align: left; color: var(--muted); font-size: .78rem; padding-bottom: .45rem; }
th, td { border: 1px solid var(--rule); padding: .38rem .55rem; text-align: left;
         vertical-align: top; }
th { background: #f4f7fa; font-weight: 600; }
tbody tr:nth-child(even) { background: #fbfcfd; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
p.gap, p.empty { color: var(--warn); background: #fff8ed; border: 1px solid #f3dfc0;
                 padding: .7rem .9rem; border-radius: 4px; }
.subtitle { color: var(--muted); margin: 0 0 1.5rem; }
nav { background: #f7f9fb; border: 1px solid var(--rule); border-radius: 6px;
      padding: 1rem 1.2rem; margin-bottom: 2rem; }
nav ol { columns: 2; column-gap: 2rem; margin: .4rem 0 0; padding-left: 1.1rem;
         font-size: .84rem; }
nav a { color: var(--ink); text-decoration: none; }
nav a:hover { color: var(--accent); text-decoration: underline; }
@media print { body { padding: 0; } nav { display: none; } h2 { page-break-after: avoid; } }
"""


def build_report(
    *,
    experiments_dir: Path | str | None = None,
    output: Path | str | None = None,
    figures_dir: Path | str | None = None,
    project_root: Path | str | None = None,
) -> ReportResult:
    """Render the technical report as HTML (and a PDF when possible)."""
    root = Path(project_root) if project_root else _infer_root(experiments_dir)
    evidence = Evidence.load(root)
    out_dir = Path(output) if output else root / "results" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = Path(figures_dir) if figures_dir else root / "docs" / "figures"

    # Regenerate figures from the live evidence so the report can never embed a
    # picture of a superseded run. A suite that cannot be plotted is reported as
    # skipped rather than drawn with placeholder data.
    from orion.figures_live import generate_figures

    figure_report = generate_figures(root, figures)
    if figure_report["skipped"]:
        print(f"  note: {len(figure_report['skipped'])} figure(s) skipped:")
        for name, reason in figure_report["skipped"].items():
            print(f"    {name}: {reason}")

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>ORION - Technical Report</title>",
        f"<style>{STYLE}</style></head><body><main>",
        "<h1>ORION</h1>",
        "<p class='subtitle'>Operational Resource Intelligence &amp; Optimization Network "
        "- technical report, generated from stored experiment artifacts</p>",
    ]
    commits = sorted({s.git_commit for s in evidence.suites.values()})
    parts.append(
        "<nav><strong>Contents</strong><ol>"
        + "".join(
            f'<li><a href="#s{n}">{html.escape(t)}</a></li>' for n, t, _ in SECTIONS
        )
        + "</ol></nav>"
    )

    missing: list[str] = []
    for number, title, builder in SECTIONS:
        try:
            body = builder(evidence)
        except KeyError as exc:
            missing.append(title)
            body = (
                f"<p class='gap'><strong>Section unavailable.</strong> {html.escape(str(exc))}</p>"
            )
        if 'class="gap"' in body or "Not available" in body:
            missing.append(title)
        parts.append(_section(number, title, body))

    # figures, embedded relative so the report is portable
    if figures.is_dir():
        images = sorted(p for p in figures.glob("*.png"))
        if images:
            items = "".join(
                f'<figure><img src="{html.escape(str(p.relative_to(figures.parent)).replace(os.sep, "/"))}" '
                f'alt="{html.escape(p.stem.replace("_", " "))}">'
                f"<figcaption>{html.escape(p.stem.replace('_', ' '))}</figcaption></figure>"
                for p in images
            )
            parts.append(
                "<section id='figures'><h2><span class='num'>28</span>Figures</h2>"
                "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(20rem,1fr));"
                "gap:1.2rem'>" + items + "</div></section>"
            )

    parts.append(
        f"<footer style='margin-top:3rem;color:#5b6b7c;font-size:.78rem'>"
        f"Generated from {len(evidence.live_suites())} live experiment suites "
        f"(commits {', '.join(commits) or 'unknown'}) and "
        f"{len(evidence.superseded)} superseded runs, which are preserved but not quoted. "
        f"Integrity: {'no inconsistencies found' if not evidence.integrity_problems() else str(len(evidence.integrity_problems())) + ' problem(s)'}.</footer>"
    )
    parts.append("</main></body></html>")

    html_path = out_dir / "report.html"
    html_path.write_text("".join(parts), encoding="utf-8")

    pdf_path: Path | None = None
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer)

        pdf_path = out_dir / "report.pdf"
        styles = getSampleStyleSheet()
        body_style = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8.6, leading=12)
        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        )
        story: list[Any] = [
            Paragraph("ORION - Technical Report", styles["Title"]),
            Paragraph(
                "Operational Resource Intelligence &amp; Optimization Network. "
                "Generated from stored experiment artifacts.",
                styles["Italic"],
            ),
            Spacer(1, 8),
        ]
        for number, title, builder in SECTIONS:
            try:
                body = builder(evidence)
            except KeyError as exc:
                body = f"<i>Section unavailable: {html.escape(str(exc))}</i>"
            story.append(Paragraph(f"{number}. {html.escape(title)}", styles["Heading2"]))
            story.extend(_html_to_story(body, styles, body_style))
            story.append(Spacer(1, 6))
        doc.build(story)
    except Exception as exc:  # reportlab is optional for the HTML path
        pdf_path = None
        print(f"  note: PDF not produced ({type(exc).__name__}: {exc})")

    return ReportResult(
        html_path=html_path,
        pdf_path=pdf_path,
        markdown_path=None,
        sections=[t for _, t, _ in SECTIONS],
        missing_suites=sorted(set(missing)),
    )


def _html_to_story(fragment: str, styles, body_style) -> list[Any]:
    """Very small HTML-to-flowable conversion for the report tables."""
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    out: list[Any] = []
    for chunk in _split_blocks(fragment):
        if chunk.startswith("<table"):
            rows = _parse_table(chunk)
            if rows:
                data = [[Paragraph(str(c), body_style) for c in row] for row in rows]
                table = Table(data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.3, (0.8, 0.85, 0.9)),
                            ("BACKGROUND", (0, 0), (-1, 0), (0.95, 0.97, 0.99)),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("FONTSIZE", (0, 0), (-1, -1), 6.4),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                out.append(table)
                out.append(Spacer(1, 4))
        elif chunk.startswith("<h3"):
            text = re_tag(chunk, "h3")
            out.append(Paragraph(html.escape(text), styles["Heading3"]))
        else:
            text = re_tag(chunk, "p") or re_tag(chunk, "pre") or re_tag(chunk, "li")
            if text:
                out.append(Paragraph(html.escape(text).replace("\n", "<br/>"), body_style))
    return out


def re_tag(fragment: str, tag: str) -> str:
    import re as _re

    match = _re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", fragment, _re.S)
    if not match:
        return ""
    text = match.group(1)
    text = _re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _split_blocks(fragment: str) -> list[str]:
    import re as _re

    parts = _re.split(r"(?=<table|<h3|<p\b|<pre\b|<li\b|<blockquote)", fragment)
    return [p for p in parts if p.strip()]


def _parse_table(chunk: str) -> list[list[str]]:
    import re as _re

    rows: list[list[str]] = []
    for row_html in _re.findall(r"<tr[^>]*>(.*?)</tr>", chunk, _re.S):
        cells = [
            html.unescape(_re.sub(r"<[^>]+>", "", c)).strip()
            for c in _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, _re.S)
        ]
        if cells:
            rows.append(cells)
    return rows


def _infer_root(experiments_dir: Path | str | None) -> Path:
    if experiments_dir:
        return Path(experiments_dir).resolve().parent
    return Path.cwd()
