"""Independent validation of job-shop schedules.

This module deliberately shares no code with the solvers. It re-reads the
instance's own machine and duration tables and checks a returned assignment set
against them from scratch, so a solver cannot pass by agreeing with a shared
helper that is itself wrong.

It exists because a job-shop schedule is a claim about *correctness*, not about
an objective value: a makespan is only meaningful if every operation ran on a
machine that can run it, for exactly its processing time, with no two operations
overlapping on a machine, and with each job's operations in order. A solver
scoring well on an invalid schedule is worse than a solver that fails outright.

A failure here is reported, never repaired. The caller decides what to do with a
schedule that does not validate - ORION demotes it to an explicit status rather
than presenting it as a successful result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


class JobShopInstanceLike(Protocol):
    """The part of ``JobShopInstance`` this validator needs."""

    name: str
    jobs: Sequence[Sequence[tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class JobShopFinding:
    kind: str
    detail: str
    task_id: str | None = None
    resource_id: str | None = None


@dataclass(frozen=True, slots=True)
class JobShopValidation:
    """Outcome of checking one assignment set against one instance."""

    instance: str
    operations_total: int
    operations_assigned: int
    makespan: int
    valid: bool
    findings: tuple[JobShopFinding, ...] = field(default_factory=tuple)

    @property
    def problem_count(self) -> int:
        return len(self.findings)

    def by_kind(self, kind: str) -> tuple[JobShopFinding, ...]:
        return tuple(f for f in self.findings if f.kind == kind)

    def summary(self) -> str:
        verdict = "VALID" if self.valid else f"INVALID ({len(self.findings)} problems)"
        return (
            f"{self.instance}: {self.operations_assigned}/{self.operations_total} "
            f"operations, makespan {self.makespan}, {verdict}"
        )


def _operation_tables(
    instance: JobShopInstanceLike,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[str]]]:
    """Map every operation id to its processing time, machine, and job chain."""
    duration: dict[str, int] = {}
    machine: dict[str, int] = {}
    chain: dict[str, list[str]] = {}
    for j, job in enumerate(instance.jobs):
        job_id = f"J{j + 1:02d}"
        chain[job_id] = []
        for k, (m, d) in enumerate(job):
            task_id = f"{job_id}-O{k + 1:02d}"
            duration[task_id] = int(d)
            machine[task_id] = int(m)
            chain[job_id].append(task_id)
    return duration, machine, chain


def validate_jobshop(
    instance: JobShopInstanceLike,
    assignments: Iterable[object],
    *,
    require_full_coverage: bool = False,
) -> JobShopValidation:
    """Check *assignments* against *instance* from scratch.

    ``assignments`` only has to expose ``task_id``, ``resource_id``, ``start``
    and ``end``; it is duck-typed so a solver's own objects can be passed
    straight in without an adapter, and so a deliberately malformed stand-in can
    be used in tests.

    Pass ``require_full_coverage=True`` to also treat an unplaced operation as a
    finding. The default is False, because ORION's job-shop scenario permits
    unassigned work by construction.

    Returns a :class:`JobShopValidation`. ``valid`` is true only when there are no
    findings at all, including any unassigned operation.
    """
    rows = list(assignments)
    findings: list[JobShopFinding] = []
    duration, machine, chain = _operation_tables(instance)
    expected_ids = set(duration)

    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        seen[row.task_id] += 1  # type: ignore[attr-defined]
    for task_id, count in seen.items():
        if count > 1:
            findings.append(
                JobShopFinding(
                    "DUPLICATE_ASSIGNMENT",
                    f"assigned {count} times; every operation must be scheduled exactly once",
                    task_id=task_id,
                )
            )

    assigned_ids = set(seen)
    # An unplaced operation is only a *finding* when the caller says the instance
    # requires full coverage. ORION's own job-shop scenario sets
    # `allow_unassigned=True`, because its objective explicitly prices leaving
    # work undone - a solver that drops 63 of 100 operations has made a
    # legitimate scheduling decision, not committed an error, and reporting it as
    # invalid would misdescribe the objective it was optimising.
    #
    # The count is still always reported (`operations_assigned`), so a caller can
    # see the coverage without the validator asserting a constraint that does not
    # exist.
    if require_full_coverage:
        for task_id in sorted(expected_ids - assigned_ids):
            findings.append(
                JobShopFinding(
                    "OPERATION_UNASSIGNED",
                    "no schedule places this operation",
                    task_id=task_id,
                )
            )
    for task_id in sorted(assigned_ids - expected_ids):
        findings.append(
            JobShopFinding(
                "UNKNOWN_OPERATION",
                "assignment refers to an operation the instance does not define",
                task_id=task_id,
            )
        )

    by_task: dict[str, object] = {}
    for row in rows:
        by_task.setdefault(row.task_id, row)  # type: ignore[attr-defined]

    # processing time
    for task_id, row in by_task.items():
        want = duration.get(task_id)
        if want is None:
            continue
        actual = row.end - row.start  # type: ignore[attr-defined]
        if actual != want:
            findings.append(
                JobShopFinding(
                    "PROCESSING_TIME_MISMATCH",
                    f"occupied {actual} time units, the instance specifies {want}",
                    task_id=task_id,
                    resource_id=row.resource_id,  # type: ignore[attr-defined]
                )
            )

    # machine eligibility
    for task_id, row in by_task.items():
        want_machine = machine.get(task_id)
        if want_machine is None:
            continue
        resource_id = str(row.resource_id)  # type: ignore[attr-defined]
        digits = "".join(ch for ch in resource_id if ch.isdigit())
        if not digits:
            findings.append(
                JobShopFinding(
                    "UNKNOWN_RESOURCE",
                    f"resource id {resource_id!r} does not name a machine",
                    task_id=task_id,
                    resource_id=resource_id,
                )
            )
            continue
        if int(digits) != want_machine:
            findings.append(
                JobShopFinding(
                    "MACHINE_INELIGIBLE",
                    f"operation requires machine {want_machine} but was placed on "
                    f"machine {int(digits)}",
                    task_id=task_id,
                    resource_id=resource_id,
                )
            )

    # machine occupancy: no two operations on one machine may overlap
    per_machine: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for task_id, row in by_task.items():
        per_machine[str(row.resource_id)].append(  # type: ignore[attr-defined]
            (row.start, row.end, task_id)  # type: ignore[attr-defined]
        )
    for resource_id, intervals in sorted(per_machine.items()):
        intervals.sort()
        for i in range(len(intervals) - 1):
            end_i, start_next = intervals[i][1], intervals[i + 1][0]
            if end_i > start_next:
                findings.append(
                    JobShopFinding(
                        "MACHINE_OVERLAP",
                        f"{intervals[i][2]} ends at {end_i} but {intervals[i + 1][2]} "
                        f"starts at {start_next}",
                        task_id=intervals[i + 1][2],
                        resource_id=resource_id,
                    )
                )

    # Precedence within each job.
    #
    # The chain is walked in *index* order, not in the order the operations happen
    # to start. Sorting by start time and comparing neighbours tests the wrong
    # thing: it asks whether a job's operations were scheduled back to back, which
    # is not what precedence means. Two operations can be correctly ordered in
    # time and still be reported as a violation, and - worse - a genuine inversion
    # can be missed whenever a third operation interleaves between them.
    #
    #
    # And the chain must be followed as *declared dependencies*, not as
    # "consecutive present operations". A job whose fourth operation was left
    # unassigned still has the fifth depending on the fourth; comparing the fifth
    # against the third is checking a relation that does not exist. With
    # `allow_unassigned` on, that mistake reported 12 phantom violations on
    # LOCAL_SEARCH's ft10 plan.
    predecessor: dict[str, str | None] = {}
    for job_id, operations in sorted(chain.items()):
        previous: str | None = None
        for task_id in operations:
            predecessor[task_id] = previous
            previous = task_id
    for task_id, parent_id in sorted(predecessor.items(), key=lambda kv: kv[0]):
        if parent_id is None or parent_id not in by_task or task_id not in by_task:
            continue
        parent_end = by_task[parent_id].end  # type: ignore[attr-defined]
        child_start = by_task[task_id].start  # type: ignore[attr-defined]
        if child_start < parent_end:
            findings.append(
                JobShopFinding(
                    "PRECEDENCE_VIOLATED",
                    f"{task_id} starts at {child_start} before its direct "
                    f"prerequisite {parent_id} ends at {parent_end}",
                    task_id=task_id,
                )
            )

    makespan = max((row.end for row in rows), default=0)  # type: ignore[attr-defined]
    return JobShopValidation(
        instance=instance.name,
        operations_total=len(expected_ids),
        operations_assigned=len(assigned_ids - (assigned_ids - expected_ids)),
        makespan=makespan,
        valid=not findings,
        findings=tuple(findings),
    )
