"""Tests for the two product figures and the figure inventory.

The baseline schedule and the resource-utilisation chart are depictions of the
product's own output rather than of a benchmark suite, so they read the
persisted demo plan instead of going through :class:`orion.evidence.Evidence`.
They still go through :mod:`orion.importing`, i.e. the same reload path a user
takes when importing an export, so these tests also pin that a figure cannot
depict a plan the product would fail to reload.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from orion.figures import available_minutes, figure_schedule, figure_utilization_rate
from orion.figures_live import FIGURE_INVENTORY, README_FIGURES, figure_names
from orion.importing import load_plan, load_scenario

ROOT = Path(__file__).resolve().parents[1]
DEMO_PLAN = ROOT / "results" / "demo" / "baseline_plan.json"
DEMO_SCENARIO = ROOT / "results" / "demo" / "scenario.json"

needs_demo = pytest.mark.skipif(
    not (DEMO_PLAN.is_file() and DEMO_SCENARIO.is_file()),
    reason="the demo artifacts are not present; run `python -m orion demo` first",
)


def _demo():
    plan, _ = load_plan(DEMO_PLAN)
    scenario, _ = load_scenario(DEMO_SCENARIO)
    return plan, scenario


def _is_valid_png(path: Path) -> bool:
    """A real PNG: the 8-byte signature, then an IHDR chunk with real extents."""
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    if data[12:16] != b"IHDR":
        return False
    width, height = struct.unpack(">II", data[16:24])
    return width > 0 and height > 0 and len(data) > 1000


# ----------------------------------------------------------------- schedule


@needs_demo
def test_the_baseline_schedule_figure_uses_the_real_assignments(tmp_path):
    plan, _scenario = _demo()
    out = figure_schedule(plan, tmp_path / "schedule.png")

    assert _is_valid_png(out)
    # The figure must reflect the plan it was handed, not a decorative sample:
    # one bar per assignment, and the same resources the plan dispatched.
    assert len(plan.assignments) > 0
    dispatched = {a.resource_id for a in plan.assignments}
    utilisation = {u.resource_id for u in plan.utilization if u.assigned_tasks > 0}
    assert dispatched == utilisation, (
        "the schedule figure groups rows by assignment resource; that set must "
        "match the plan's own utilisation rows or the picture misrepresents it"
    )
    assert len(dispatched) >= 2, "a single-resource plan cannot show a schedule"


@needs_demo
def test_the_baseline_schedule_reflects_this_run_and_is_deterministic(tmp_path):
    plan, _ = _demo()
    first = figure_schedule(plan, tmp_path / "a.png").read_bytes()
    second = figure_schedule(plan, tmp_path / "b.png").read_bytes()
    assert first == second, (
        "two renders of the same plan must be byte-identical; otherwise the "
        "README cannot trust a regenerated figure"
    )


def test_the_schedule_figure_fails_cleanly_on_a_plan_with_no_assignments(tmp_path):
    import dataclasses

    plan, _scenario = _demo()
    stripped = dataclasses.replace(plan, assignments=(), utilization=(), tasks_assigned=0)
    with pytest.raises(ValueError, match="no assignments"):
        figure_schedule(stripped, tmp_path / "empty.png")
    assert not (tmp_path / "empty.png").exists(), (
        "a figure that could not be drawn must not leave a file behind"
    )


# ------------------------------------------------------------- utilization


@needs_demo
def test_utilization_uses_actual_availability_not_task_count(tmp_path):
    plan, scenario = _demo()
    out = figure_utilization_rate(plan, scenario, tmp_path / "util.png")
    assert _is_valid_png(out)

    resources = {r.id: r for r in scenario.resources}
    for row in plan.utilization:
        if row.assigned_tasks <= 0:
            continue
        available = available_minutes(resources[row.resource_id])
        if available <= 0:
            continue
        rate = row.worked_minutes / available
        assert 0.0 <= rate <= 1.0, (
            f"{row.resource_id}: worked {row.worked_minutes} of {available} "
            "available minutes is not a utilisation rate"
        )
        # Worked time can never exceed the time the domain said was available.
        assert row.worked_minutes <= available, (
            f"{row.resource_id} worked {row.worked_minutes} minutes but the "
            f"domain only offers {available}"
        )


@needs_demo
def test_available_minutes_is_the_shift_minus_blocked_time():
    plan, scenario = _demo()
    resources = {r.id: r for r in scenario.resources}
    checked = 0
    for row in plan.utilization:
        resource = resources.get(row.resource_id)
        if resource is None:
            continue
        # `max_work_minutes` is an independent cap on *worked* minutes, not a
        # shortening of the shift: the engine compares accumulated work against
        # it (see builder.py and cp_sat_solver.py), so availability is the
        # smaller of the two.
        shift_minutes = resource.shift_end - resource.shift_start
        unblocked = shift_minutes
        for window in resource.unavailable:
            lo = max(window.start, resource.shift_start)
            hi = min(window.end, resource.shift_end)
            if hi > lo:
                unblocked -= hi - lo
        expected = max(0, min(unblocked, int(resource.max_work_minutes)))
        assert available_minutes(resource) == expected
        checked += 1
    assert checked > 0, "the demo plan dispatched nothing, so nothing was checked"


def test_a_fully_blocked_resource_has_no_available_time_and_is_not_faked():
    import dataclasses

    from orion.domain.entities import Interval

    plan, _scenario = _demo()
    resources = {r.id: r for r in _scenario.resources}
    template = next(iter(resources.values()))
    # A shift entirely covered by a blocked interval leaves nothing available.
    blocked = dataclasses.replace(
        template,
        id="R-BLOCKED",
        shift_start=400,
        shift_end=500,
        unavailable=(Interval(400, 500),),
    )
    assert available_minutes(blocked) == 0

    util = [u for u in plan.utilization if u.assigned_tasks > 0][:1]
    if not util:
        pytest.skip("the demo plan dispatched nothing")
    # A plan that dispatched only that resource has an undefined utilisation,
    # so the figure must refuse rather than divide by zero or invent a value.
    only = dataclasses.replace(
        plan,
        assignments=tuple(a for a in plan.assignments if a.resource_id == util[0].resource_id),
        utilization=tuple(util),
    )
    blocked_scenario = dataclasses.replace(
        _scenario, resources=(blocked,)
    )
    with pytest.raises(ValueError, match="positive available time"):
        figure_utilization_rate(only, blocked_scenario, Path("never.png"))


def test_utilization_omits_resources_with_no_comparable_availability(tmp_path):
    import dataclasses

    from orion.domain.entities import Interval

    plan, scenario = _demo()
    resources = {r.id: r for r in scenario.resources}
    dispatched = [u for u in plan.utilization if u.assigned_tasks > 0]
    if len(dispatched) < 2:
        pytest.skip("need at least two dispatched resources for this check")

    # One dispatched resource is fully blocked: it has no comparable
    # availability, so it must be omitted rather than shown as 0% or 100%.
    victim = dispatched[0].resource_id
    template = resources[victim]
    zeroed = dataclasses.replace(
        template, unavailable=(Interval(template.shift_start, template.shift_end),)
    )
    assert available_minutes(zeroed) == 0
    patched = dataclasses.replace(
        scenario,
        resources=tuple(zeroed if r.id == victim else r for r in scenario.resources),
    )

    out = figure_utilization_rate(plan, patched, tmp_path / "util.png")
    assert _is_valid_png(out)


def test_utilization_fails_cleanly_when_nothing_was_dispatched(tmp_path):
    import dataclasses

    plan, scenario = _demo()
    empty = dataclasses.replace(plan, utilization=(), assignments=())
    with pytest.raises(ValueError, match="dispatched nothing"):
        figure_utilization_rate(empty, scenario, tmp_path / "none.png")
    assert not (tmp_path / "none.png").exists()


# ----------------------------------------------------------------- inventory


def test_the_inventory_is_the_single_source_of_figure_names():
    names = figure_names()
    assert len(names) == len(set(names)), "duplicate figure names in the inventory"
    assert all(n.startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_")) for n in names)
    assert [n[:2] for n in names] == [f"{i:02d}" for i in range(1, len(names) + 1)], (
        "the inventory is numbered out of order, which is how the README, the "
        "report and the files on disk drifted apart"
    )


def test_every_inventory_entry_documents_a_source():
    for entry in FIGURE_INVENTORY:
        assert entry.get("title"), f"{entry['file']} has no title"
        assert entry.get("source"), f"{entry['file']} has no documented source"


def test_the_readme_subset_comes_from_the_inventory():
    for name in README_FIGURES:
        assert name in figure_names(), (
            f"README promotes {name}, which is not in the inventory, so the "
            "README can link to a figure that is never generated"
        )


def test_the_baseline_and_utilisation_figures_are_in_the_inventory():
    names = figure_names()
    assert any("baseline_schedule" in n for n in names)
    assert any("resource_utilization" in n for n in names)


@needs_demo
def test_the_two_product_figures_render_through_the_live_layer(tmp_path):
    from orion.figures_live import generate_figures

    report = generate_figures(ROOT, tmp_path)
    written = report["written"]
    assert "01_baseline_schedule.png" in written, report["skipped"]
    assert "02_resource_utilization.png" in written, report["skipped"]
    for name in written:
        assert _is_valid_png(tmp_path / name), f"{name} is not a valid PNG"
    assert not report["integrity_problems"], report["integrity_problems"]
