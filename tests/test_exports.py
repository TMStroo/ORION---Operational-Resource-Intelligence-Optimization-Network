"""Export and re-import.

An export that cannot be reloaded is a screenshot, not evidence. These tests
pin the round trip rather than the file format: every artifact is written by
the real exporter, read back by the real loader, and the reloaded entity is
compared against the payload it came from.

What must survive, and is checked explicitly because each was a real loss at
some point:

* ids, so a reloaded plan is still the same plan;
* assignments, in order, with their times and travel;
* timestamps;
* solver status and the scored objective;
* provenance - the seed, git commit and objective weights.

A test that only asserted "the file exists" would have passed while the
comparison deltas, the solver runtimes and the availability windows were all
being dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from orion.data.scenario_generator import ScenarioConfig, generate  # noqa: E402
from orion.exporting import (  # noqa: E402
    ExportError,
    export_plan,
    export_scenario,
    plan_metric_rows_csv,
)
from orion.importing import load_plan, load_scenario, read_csv  # noqa: E402
from orion.planning.planner import Planner  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    """A real scenario and a real plan, solved once and reused."""
    config = ScenarioConfig(name="exports", num_tasks=18, num_teams=4, num_vehicles=3, seed=21)
    scenario = generate(config)
    outcome = Planner(default_solver="HEURISTIC", default_time_budget=5.0).plan(scenario)
    return scenario, outcome.plan


# ------------------------------------------------------- scenario round trip


def test_scenario_export_reload_preserves_domain_state(built, tmp_path):
    scenario, _plan = built
    path = export_scenario(scenario, tmp_path / "scenario.json", seed=21)
    reloaded, provenance = load_scenario(path)
    assert reloaded.to_dict() == scenario.to_dict()
    assert reloaded.id == scenario.id
    assert len(reloaded.tasks) == len(scenario.tasks)
    assert len(reloaded.resources) == len(scenario.resources)


def test_scenario_export_keeps_its_provenance(built, tmp_path):
    scenario, _plan = built
    path = export_scenario(scenario, tmp_path / "scenario.json", seed=21)
    _reloaded, provenance = load_scenario(path)
    assert provenance["seed"] == 21
    assert provenance["kind"] == "scenario"
    assert provenance["git_commit"], "an export without a commit is not reproducible"
    assert provenance["export_schema"]


# ----------------------------------------------------------- plan round trip


def test_plan_export_reload_preserves_domain_state(built, tmp_path):
    """The reloaded plan equals the plan that was written.

    Compared on ``to_dict``, not on the raw floats: ``to_dict`` rounds each
    metric to a fixed precision so an artifact is readable and diffable, and
    the round trip has to be faithful to the serialized form. Comparing raw
    floats would fail on representation alone, and comparing nothing would let
    a dropped field pass.
    """
    _scenario, plan = built
    path = export_plan(plan, tmp_path / "plan.json")
    reloaded, _provenance = load_plan(path)
    assert reloaded.to_dict() == plan.to_dict()


def test_plan_export_keeps_ids_assignments_status_and_objective(built, tmp_path):
    _scenario, plan = built
    path = export_plan(plan, tmp_path / "plan.json")
    reloaded, _provenance = load_plan(path)

    assert reloaded.id == plan.id
    assert reloaded.scenario_id == plan.scenario_id
    assert reloaded.status == plan.status
    assert reloaded.objective.total == pytest.approx(plan.objective.total)
    assert reloaded.created_at == plan.created_at

    assert len(reloaded.assignments) == len(plan.assignments)
    for original, copy in zip(plan.assignments, reloaded.assignments):
        assert copy.task_id == original.task_id
        assert copy.resource_id == original.resource_id
        assert (copy.start, copy.end) == (original.start, original.end)
        assert copy.location == original.location
        assert copy.lateness == original.lateness
        assert copy.travel_before == original.travel_before
        assert copy.travel_distance_km == pytest.approx(original.travel_distance_km)


def test_plan_export_keeps_every_solver_run(built, tmp_path):
    _scenario, plan = built
    path = export_plan(plan, tmp_path / "plan.json")
    reloaded, _provenance = load_plan(path)

    assert len(reloaded.solver_runs) == len(plan.solver_runs)
    for original, copy in zip(plan.solver_runs, reloaded.solver_runs):
        assert copy.status == original.status
        assert copy.solver == original.solver
        # A runtime of 0.0 is the exact defect this round trip has to catch:
        # the objective used to be written back as a hard-coded zero.
        # to_dict rounds objective to 4dp and runtime to 6dp; the reloaded
        # values must agree at that precision, not at full float width.
        assert copy.objective == pytest.approx(original.objective, abs=1e-4)
        assert copy.runtime_s == pytest.approx(original.runtime_s, abs=1e-6)


def test_plan_export_carries_no_nan(built, tmp_path):
    _scenario, plan = built
    path = export_plan(plan, tmp_path / "plan.json")
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


# ------------------------------------------------------------------ metrics


def test_metrics_csv_is_readable_and_agrees_with_the_plan(built, tmp_path):
    scenario, plan = built
    path = plan_metric_rows_csv(plan, scenario, tmp_path / "metrics.csv", seed=21)
    rows = read_csv(path)
    assert rows, "the metrics CSV has no data rows"
    # The column names come from the exporter; read them off the first row
    # rather than guessing, so a column rename fails here instead of silently
    # making this assertion vacuous.
    columns = set(rows[0])
    value_column = next(
        (c for c in ("value", "metric_value", "amount") if c in columns), None
    )
    assert value_column, f"no value column among {sorted(columns)}"
    assert any(row[value_column] not in ("", None) for row in rows), (
        f"every {value_column} is empty"
    )


def test_metrics_csv_starts_with_its_provenance(built, tmp_path):
    scenario, plan = built
    path = plan_metric_rows_csv(plan, scenario, tmp_path / "metrics.csv", seed=21)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# provenance: ")
    assert '"seed": 21' in first or '"seed":21' in first


# ------------------------------------------------------- report output paths


def test_the_report_is_written_as_files_not_as_a_directory(tmp_path):
    """`report.html` must be a file.

    Pointing the builder at the directory that already holds the figures once
    created a nested `docs/report.html/report.html` instead of failing, so the
    report committed to the repository was a directory containing a report.
    """
    from orion.report.build import build_report

    out = tmp_path / "docs"
    out.mkdir()
    result = build_report(
        experiments_dir=REPO_ROOT / "experiments",
        output=out,
        figures_dir=tmp_path / "figs",
    )
    assert result.html_path.is_file(), f"{result.html_path} is not a file"
    assert result.html_path.name == "report.html"
    assert result.html_path.read_text(encoding="utf-8").lstrip().startswith("<")


# ------------------------------------------------------------- bad artifacts


def test_loading_a_non_export_json_fails_loudly(tmp_path):
    """A bare payload is not an export, and must not be reported as one."""
    path = tmp_path / "not-an-export.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(ExportError, match="not an ORION export"):
        load_scenario(path)


def test_loading_a_missing_file_fails_loudly(tmp_path):
    with pytest.raises(ExportError, match="no such export"):
        load_plan(tmp_path / "absent.json")


def test_loading_a_corrupt_plan_reports_why(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        '{"provenance": {}, "data": {"id": "PLAN-1", "assignments": "not-a-list"}}',
        encoding="utf-8",
    )
    with pytest.raises(ExportError, match="cannot rebuild plan"):
        load_plan(path)


def test_the_pdf_embeds_the_figures_that_the_html_embeds(tmp_path):
    """The PDF must carry the plots, not only the prose.

    It did not once. The HTML embedded all eleven figures inline, but the PDF
    path looped over the sections and never consulted the figure map, so the
    published PDF was twenty-seven sections of numbers and not one chart. A
    reader comparing the two documents was looking at different reports.
    """
    pytest.importorskip("pypdf")
    from orion.report import build_report
    from orion.figures_live import FIGURE_INVENTORY

    result = build_report(
        experiments_dir=REPO_ROOT / "experiments",
        output=tmp_path,
        figures_dir=REPO_ROOT / "docs" / "figures",
        project_root=REPO_ROOT,
    )
    assert result.pdf_path is not None, "the report produced no PDF"
    assert result.pdf_path.is_file()

    import pypdf

    embedded = 0
    for page in pypdf.PdfReader(str(result.pdf_path)).pages:
        resources = page.get("/Resources")
        xobjects = resources.get("/XObject") if resources else None
        if xobjects:
            embedded += len(xobjects)
    assert embedded >= len(FIGURE_INVENTORY), (
        f"the PDF embeds {embedded} images but the inventory claims "
        f"{len(FIGURE_INVENTORY)}"
    )
