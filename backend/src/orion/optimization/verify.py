"""Cross-checks every solver against an independent brute-force optimum.

This is the evidence that ORION's objective and constraints are *correct* rather
than merely self-consistent. For each hand-built micro-instance in
:mod:`orion.optimization.tiny` the reference enumerator solves the problem
exhaustively; every solver is then run on the identical scenario with an
identical time budget and its plan is re-scored by the same objective the
reference used.

What a result means
-------------------
``matches_reference``
    The solver's independently-scored objective equals the proven optimum. This
    is only ever asserted for a solver that claims ``OPTIMAL``; a heuristic that
    matches is reported as a match but is never described as proven.

``gap_pct``
    ``(solver - reference) / |reference| * 100``. ORION's objective is a
    *reward* - higher is better - so a shortfall is **negative** and beating the
    enumeration is **positive**. Only the positive direction is a bug: a solver
    cannot exceed a proven optimum, so a positive gap means either the
    enumeration or the scoring is wrong, and it is flagged for investigation.

Two guarantees are separated on purpose. CP-SAT is required to match the proven
optimum on every fixture. MILP and the min-cost flow are *relaxations* - they
prove optimality of a weaker model - so they are expected to fall short and are
reported, not asserted against. The heuristics are expected to fall short too,
and the size of the shortfall is the heuristic-validation signal.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

from orion.domain.entities import Scenario
from orion.domain.plans import PROVEN_OPTIMAL_STATUSES, SolverName, SolverStatus
from orion.optimization.models import SchedulingContext
from orion.optimization.reference import ReferenceSolution, enumerate_optimum
from orion.optimization.registry import (
    ALL_SOLVERS,
    SolverRequest,
    make_solver,
)
from orion.optimization.tiny import TINY_FIXTURES

#: Solvers whose formulation is a relaxation, so a shortfall is expected.
RELAXATION_SOLVERS = frozenset({SolverName.MILP, SolverName.MIN_COST_FLOW})

#: Solvers that are expected to be provably optimal on these fixtures.
MUST_MATCH_SOLVERS = frozenset({SolverName.CP_SAT})

#: Solvers measured for heuristic quality rather than asserted.
HEURISTIC_SOLVERS = frozenset({SolverName.HEURISTIC, SolverName.LOCAL_SEARCH})


@dataclass(frozen=True, slots=True)
class SolverFixtureResult:
    solver: str
    fixture: str
    status: str
    objective: float
    reference_objective: float | None
    gap_pct: float | None
    matches_reference: bool
    hard_violations: int
    runtime_s: float
    tasks_assigned: int
    late_tasks: int
    travel_km: float
    operating_cost: float
    expected_to_match: bool
    gap_is_absolute: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    results: tuple[SolverFixtureResult, ...]
    references: Mapping[str, ReferenceSolution]
    seconds: float
    seeds: tuple[int, ...] = (0,)

    # -- aggregate views -------------------------------------------------
    def by_solver(self) -> dict[str, list[SolverFixtureResult]]:
        out: dict[str, list[SolverFixtureResult]] = {}
        for row in self.results:
            out.setdefault(row.solver, []).append(row)
        return out

    def proven_mismatches(self) -> list[SolverFixtureResult]:
        """Exact solvers that failed to reach the proven optimum."""
        return [
            row
            for row in self.results
            if row.expected_to_match and not row.matches_reference
        ]

    def beats_reference(self) -> list[SolverFixtureResult]:
        """Any solver scoring *above* the proven optimum - always a bug.

        A solver that exceeds an exhaustively-proven optimum means one of the
        two is wrong, so this is surfaced rather than tolerated.
        """
        return [
            row
            for row in self.results
            if (row.gap_pct or 0.0) > 1e-6 and not row.gap_is_absolute
        ]

    def shortfall(self) -> list[SolverFixtureResult]:
        """Rows that did not reach the optimum, worst gap first.

        Expected for relaxations and heuristics; a zero-length list for CP-SAT
        means the exact solver was validated on every fixture.
        """
        rows = [
            row
            for row in self.results
            if row.gap_pct is not None and row.gap_pct < -1e-6 and not row.gap_is_absolute
        ]
        return sorted(rows, key=lambda r: r.gap_pct)

    def heuristic_summary(self) -> dict[str, dict[str, float]]:
        """mean/median/best/worst gap and runtime per heuristic solver."""
        summary: dict[str, dict[str, float]] = {}
        for solver, rows in self.by_solver().items():
            if solver not in HEURISTIC_SOLVERS:
                continue
            gaps = [r.gap_pct for r in rows if r.gap_pct is not None]
            runtimes = [r.runtime_s for r in rows]
            if not gaps:
                continue
            summary[solver] = {
                "fixtures": len(gaps),
                "gap_mean_pct": statistics.fmean(gaps),
                "gap_median_pct": statistics.median(gaps),
                "gap_best_pct": min(gaps),
                "gap_worst_pct": max(gaps),
                "runtime_mean_s": statistics.fmean(runtimes),
                "runtime_median_s": statistics.median(runtimes),
                "matches_count": float(sum(1 for r in rows if r.matches_reference)),
            }
        return summary

    def to_dict(self) -> dict[str, object]:
        return {
            "seconds": round(self.seconds, 4),
            "seeds": list(self.seeds),
            "proven_mismatches": [asdict(r) for r in self.proven_mismatches()],
            "beats_reference": [asdict(r) for r in self.beats_reference()],
            "shortfall": [asdict(r) for r in self.shortfall()],
            "heuristic_summary": self.heuristic_summary(),
            "rows": [asdict(r) for r in self.results],
            "references": {k: v.to_dict() for k, v in self.references.items()},
        }


def verify_fixture(
    fixture_name: str,
    scenario: Scenario,
    *,
    solvers: Sequence[str] = ALL_SOLVERS,
    time_limit_s: float = 20.0,
    seed: int = 0,
    max_combinations: int | None = None,
) -> tuple[list[SolverFixtureResult], ReferenceSolution]:
    """Run every solver on one scenario and compare against brute force."""
    kwargs: dict[str, object] = {}
    if max_combinations is not None:
        kwargs["max_combinations"] = max_combinations
    reference = enumerate_optimum(scenario, **kwargs)  # type: ignore[arg-type]
    context = SchedulingContext.build(scenario)
    rows: list[SolverFixtureResult] = []
    for name in solvers:
        solver = make_solver(name)
        request = SolverRequest(time_limit_s=time_limit_s, seed=seed)
        started = time.perf_counter()
        assignments, run, _diagnostics = solver.solve(context, request)
        elapsed = time.perf_counter() - started
        from orion.optimization.builder import build_plan

        plan = build_plan(context, assignments, run, strategy=name)
        objective = plan.objective.total
        hard = sum(1 for v in plan.violations if v.severity >= 1)
        gap: float | None = None
        matches = False
        if reference.best_score is not None and reference.exhaustive:
            if abs(reference.best_score) <= 1e-9:
                # A degenerate reference (e.g. the infeasible fixture, whose best
                # plan serves nothing and therefore scores 0) has no meaningful
                # percentage gap. Report the absolute difference and treat exact
                # equality as a match, rather than dividing by zero or inventing
                # a percentage.
                gap = objective - reference.best_score
                matches = abs(gap) <= 1e-6
            else:
                gap = (objective - reference.best_score) / abs(reference.best_score) * 100.0
                matches = abs(gap) <= 1e-6
        rows.append(
            SolverFixtureResult(
                solver=name,
                fixture=fixture_name,
                status=run.status,
                objective=round(objective, 6),
                reference_objective=reference.best_score,
                gap_pct=None if gap is None else round(gap, 6),
                gap_is_absolute=reference.best_score is not None and abs(reference.best_score) <= 1e-9,
                matches_reference=matches,
                hard_violations=hard,
                runtime_s=round(elapsed, 6),
                tasks_assigned=plan.tasks_assigned,
                late_tasks=plan.late_tasks,
                travel_km=round(plan.total_travel_km, 3),
                operating_cost=round(plan.total_operating_cost, 3),
                expected_to_match=name in MUST_MATCH_SOLVERS,
                note=_note(name, run.status, reference, hard),
            )
        )
    return rows, reference


def _note(solver: str, status: str, reference: ReferenceSolution, hard: int) -> str:
    if solver in RELAXATION_SOLVERS and status == SolverStatus.RELAXATION_OPTIMAL:
        return "relaxation: optimality proven for a weaker model, shortfall expected"
    if solver in HEURISTIC_SOLVERS:
        return "heuristic: no optimality claim, gap is the quality signal"
    if hard:
        return f"{hard} hard violation(s) in the returned plan"
    return ""


def verify_all(
    fixtures: Mapping[str, Scenario] | None = None,
    *,
    solvers: Sequence[str] = ALL_SOLVERS,
    time_limit_s: float = 20.0,
    seed: int = 0,
) -> VerificationReport:
    """Cross-check every solver against brute force on every tiny fixture."""
    started = time.perf_counter()
    chosen = dict(fixtures if fixtures is not None else TINY_FIXTURES)
    rows: list[SolverFixtureResult] = []
    references: dict[str, ReferenceSolution] = {}
    for name, scenario in chosen.items():
        fixture_rows, reference = verify_fixture(
            name, scenario, solvers=solvers, time_limit_s=time_limit_s, seed=seed
        )
        rows.extend(fixture_rows)
        references[name] = reference
    return VerificationReport(
        results=tuple(rows),
        references=references,
        seconds=time.perf_counter() - started,
        seeds=(seed,),
    )


def multi_seed_heuristic_quality(
    scenario: Scenario,
    *,
    solvers: Iterable[str] = tuple(sorted(HEURISTIC_SOLVERS)),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    time_limit_s: float = 20.0,
) -> dict[str, dict[str, float]]:
    """Spread of heuristic quality across seeds on one scenario.

    A single run says nothing about a stochastic solver. This reports the
    distribution the brief asks for: mean, median, best and worst objective, plus
    runtimes, and the spread (worst - best) so a solver whose output is
    effectively seed-independent can be told apart from an unstable one.
    """
    from orion.optimization.builder import build_plan

    context = SchedulingContext.build(scenario)
    out: dict[str, dict[str, float]] = {}
    for name in solvers:
        solver = make_solver(name)
        objectives: list[float] = []
        runtimes: list[float] = []
        for seed in seeds:
            started = time.perf_counter()
            assignments, run, _ = solver.solve(
                context, SolverRequest(time_limit_s=time_limit_s, seed=seed)
            )
            runtimes.append(time.perf_counter() - started)
            plan = build_plan(context, assignments, run, strategy=name)
            objectives.append(plan.objective.total)
        out[name] = {
            "seeds": len(seeds),
            "mean": statistics.fmean(objectives),
            "median": statistics.median(objectives),
            "best": max(objectives),
            "worst": min(objectives),
            "spread": max(objectives) - min(objectives),
            "runtime_mean_s": statistics.fmean(runtimes),
            "runtime_median_s": statistics.median(runtimes),
            "runtime_max_s": max(runtimes),
        }
    return out
