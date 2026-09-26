"""Adapters from public benchmark formats to the ORION domain model.

The adapters live in ``data/``, not in ``domain/``, on purpose. A benchmark
instance is *not* an operational scenario: a Solomon customer has no priority, no
required capability and no operating cost, and a job-shop operation has no
location to travel to. Mapping them onto :class:`Scenario` is a modelling
decision that belongs to the data layer, so a reader can see exactly which
benchmark field became which operational field (and which fields had no
counterpart and were defaulted).

What each benchmark measures in ORION
-------------------------------------
``solomon_vrptw``
    Routing feasibility and distance quality. ORION's objective is *service
    weighted* while Solomon's published numbers minimise (vehicles, distance)
    hierarchically, so ORION does **not** claim its scores are comparable to
    published Solomon best-known values. What is comparable is the distance
    column: on an identical instance and objective, ORION's route distance is
    directly checkable.
``job_shop``
    Sequencing quality under a machine-capacity bottleneck. ORION's machines are
    modelled as resources with a capability, so makespan is a natural read-out.
"""

from __future__ import annotations

import hashlib
import io
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from orion.domain.entities import (
    ObjectiveWeights,
    Priority,
    Resource,
    ResourceKind,
    Scenario,
    ScenarioConstraints,
    Task,
    TravelMatrix,
)
from orion.domain.time_model import Interval

# --------------------------------------------------------------------------
# provenance: every external source is registered here, never hard-coded
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataSource:
    """Provenance record for an external dataset or benchmark."""

    name: str
    source: str
    url: str
    license: str
    purpose: str
    sha256: str
    download: str
    preprocessing: str
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "url": self.url,
            "license": self.license,
            "purpose": self.purpose,
            "sha256": self.sha256,
            "download": self.download,
            "preprocessing": self.preprocessing,
            "limitations": self.limitations,
        }


SOLOMON_URL = "https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip"
SOLOMON_SHA256 = "8a0a72cbe6b7f8f9988ace4ebde0378ec34943acaaac47f2c408915e41887747"

SOLOMON_SOURCE = DataSource(
    name="Solomon VRPTW (100-customer, SINTEF mirror)",
    source="SINTEF / Solomon (1987), distributed by the SINTEF VRPTW project page",
    url=SOLOMON_URL,
    license=(
        "Free for academic and non-commercial use as distributed by SINTEF. ORION treats "
        "the instances as read-only research inputs and redistributes no instance files "
        "in the repository; a downloader fetches them on demand."
    ),
    purpose=(
        "Public, citable routing instances that map onto ORION's vehicle-routing and "
        "time-window constraint classes. They anchor the solver comparison to an "
        "external yardstick rather than to ORION's own generator."
    ),
    sha256=SOLOMON_SHA256,
    download=(
        "python -m orion benchmark --config configs/benchmark.yaml  (or "
        "`python -m orion data fetch --dataset solomon`); cached under data/cache/"
    ),
    preprocessing=(
        "Parse the fixed-width Solomon text format: depot row 0 becomes the single depot "
        "location, each customer becomes one Task with the customer time window as its "
        "[release, deadline] and the SERVICE TIME column as its duration. Distances are "
        "Euclidean; travel minutes = distance / speed, which is why the instances scale "
        "the time axis by the depot horizon. Customer demand is dropped because ORION's "
        "resources are capacity-unconstrained in this mapping - see limitations."
    ),
    limitations=(
        "Solomon instances have no priority, no capability requirement and no cost, so "
        "ORION assigns a uniform priority and a single capability. They therefore test "
        "routing and time-window handling, NOT priority weighting, scarcity or cost. "
        "Published Solomon best-known values use a hierarchical (vehicles, distance) "
        "objective and are not directly comparable to ORION's service-weighted score."
    ),
)


def _fetch(url: str, cache_dir: Path, expected_sha256: str | None) -> bytes:
    """Download *url* with a cache, verifying the checksum.

    A checksum mismatch is fatal. Silently using a corrupt benchmark would
    invalidate every number derived from it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1] or "download.bin"
    target = cache_dir / name
    if target.exists():
        blob = target.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "orion-research/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
            blob = response.read()
        target.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"checksum mismatch for {url}: expected {expected_sha256}, got {digest}. "
            f"Refusing to use an unverifiable benchmark."
        )
    return blob


# --------------------------------------------------------------------------
# Solomon VRPTW
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolomonNode:
    index: int
    x: float
    y: float
    demand: int
    ready: float
    due: float
    service: float


@dataclass(frozen=True, slots=True)
class SolomonInstance:
    name: str
    vehicle_count: int
    vehicle_capacity: int
    nodes: tuple[SolomonNode, ...]

    @property
    def depot(self) -> SolomonNode:
        return self.nodes[0]

    @property
    def customers(self) -> tuple[SolomonNode, ...]:
        return self.nodes[1:]


def parse_solomon(text: str) -> SolomonInstance:
    """Parse one Solomon instance from its fixed-width text format."""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) < 5:
        raise ValueError("Solomon instance is truncated")
    name = lines[0]
    # find the vehicle line: the numeric line after the 'VEHICLE' header
    idx = lines.index("VEHICLE")
    vehicle_count, vehicle_capacity = (int(v) for v in lines[idx + 2].split()[:2])
    # the customer table starts after the 'CUSTOMER' header and its column line
    start = lines.index("CUSTOMER") + 3
    nodes: list[SolomonNode] = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 7:
            break
        nodes.append(
            SolomonNode(
                index=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=int(float(parts[3])),
                ready=float(parts[4]),
                due=float(parts[5]),
                service=float(parts[6]),
            )
        )
    if not nodes:
        raise ValueError(f"{name}: no customer rows parsed")
    return SolomonInstance(name, vehicle_count, vehicle_capacity, tuple(nodes))


def load_solomon(cache_dir: Path | str = "data/cache", names: Sequence[str] | None = None) -> dict[str, SolomonInstance]:
    """Download (once) and parse the requested Solomon instances."""
    blob = _fetch(SOLOMON_URL, Path(cache_dir), SOLOMON_SHA256)
    archive = zipfile.ZipFile(io.BytesIO(blob))
    wanted = tuple(n.lower() for n in names) if names else None
    out: dict[str, SolomonInstance] = {}
    for entry in archive.namelist():
        path = Path(entry)
        # ``Path(entry).stem`` already drops the extension, so testing the stem
        # for ".txt" is always False. Test the suffix of the full entry name.
        if path.suffix.lower() != ".txt":
            continue
        stem = path.stem.lower()
        if wanted and stem not in wanted:
            continue
        text = archive.read(entry).decode("utf-8", errors="replace")
        out[stem] = parse_solomon(text)
    if not out:
        raise ValueError(f"no Solomon instances matched {names!r}")
    return out


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def loc_id_for(node_index: int) -> str:
    """Solomon node index -> ORION location id (node 0 is the shared depot)."""
    return "DEPOT" if node_index == 0 else f"N{node_index:03d}"


def solomon_to_scenario(
    instance: SolomonInstance,
    *,
    fleet_size: int | None = None,
    speed_kmh: float = 60.0,
    weights: ObjectiveWeights | None = None,
    name_prefix: str = "solomon",
) -> Scenario:
    """Map a Solomon instance onto an operational scenario.

    Mapping, and the defaults that have no benchmark counterpart:

    =============================  =========================================
    Solomon field                  ORION field
    =============================  =========================================
    depot (node 0)                  ``TravelMatrix`` location ``DEPOT``; every
                                   resource starts and ends there
    customer i                      one ``Task`` at location ``N<i>``
    READY TIME / DUE DATE           ``release_time`` / ``deadline``
                                   (due date + service time; see note)
    SERVICE TIME                    ``duration`` in minutes
    K (vehicle count)               number of ``Resource`` objects
    - (none)                        ``priority = MEDIUM`` for every customer
    - (none)                        ``required_capabilities = ("routing",)``
    - (none)                        ``operating_cost_per_hour = 0``
    - (Q capacity)                  not modelled - see ``SOLOMON_SOURCE``
    =============================  =========================================

    **Time axis.** Solomon defines travel time = Euclidean distance, i.e. the
    numeric coordinate of the instance is simultaneously a distance and a
    duration. ORION's ``TravelMatrix.from_coordinates`` works in km/h, so this
    adapter calls :meth:`TravelMatrix.scaled` semantics indirectly: it sets
    ``congestion_factor`` to ``60.0 / speed_kmh`` so that

        travel_minutes = euclidean_km / speed_kmh * (60 / speed_kmh)

    is *not* what we want. Instead the adapter builds the matrix with
    ``default_speed_kmh=60`` (1 km per minute) and divides the resulting minute
    travel times by 60 via ``scaled(1/60)``, which reproduces Solomon's identity
    ``travel_time == distance`` exactly while keeping integer minutes. With
    ``speed_kmh=60`` - the default - ORION's distances are directly comparable
    to published Solomon distances.
    """
    nodes = instance.customers
    depot = instance.depot
    # Solomon's schedule ends at the last due date, but a plan also has to get
    # a vehicle from its final customer back to the depot. Reserve that leg so
    # the last customer is not declared unreachable by ORION's own travel model.
    max_leg = max(
        _euclid((n.x, n.y), (depot.x, depot.y)) for n in nodes
    )
    # The last obligation is: start the final customer at its due date, serve it
    # for its service time, then drive back to the depot.
    horizon_end = int(max(n.due for n in nodes) + max(n.service for n in nodes) + max_leg) + 1

    coordinates: dict[str, object] = {"DEPOT": (depot.x, depot.y)}
    tasks: list[Task] = []
    for node in nodes:
        loc = loc_id_for(node.index)
        coordinates[loc] = (node.x, node.y)
        tasks.append(
            Task(
                id=f"T-{node.index:03d}",
                location=loc,
                release_time=int(node.ready),
                # Solomon's DUE DATE bounds the *start of service*, not the end.
                # ORION's ``deadline`` bounds completion, so the due date must be
                # widened by the service time. Using the raw due date as a
                # completion deadline makes every 90-minute job in C101
                # "late" by construction, which is a mapping artefact and not a
                # property of the instance.
                deadline=int(node.due) + max(1, int(node.service)),
                duration=max(1, int(node.service)),
                priority=Priority.MEDIUM,
                required_capabilities=("routing",),
            )
        )

    # 1 km per minute, then rescale so integer minutes equal Solomon's distances
    base = TravelMatrix.from_coordinates(coordinates, default_speed_kmh=60.0, min_travel_minutes=0)
    travel = base.scaled(speed_kmh / 60.0)

    fleet = fleet_size or instance.vehicle_count
    resources = tuple(
        Resource(
            id=f"V-{i + 1:02d}",
            kind=ResourceKind.VEHICLE,
            capabilities=("routing",),
            home_location="DEPOT",
            shift_start=0,
            shift_end=horizon_end,
            operating_cost_per_hour=0.0,
            fixed_dispatch_cost=0.0,
            max_work_minutes=horizon_end,
        )
        for i in range(fleet)
    )

    return Scenario(
        id=f"{name_prefix}-{instance.name.lower()}",
        name=f"{name_prefix}:{instance.name}",
        horizon=Interval(0, horizon_end),
        tasks=tuple(tasks),
        resources=resources,
        travel=travel,
        objective_weights=weights or ObjectiveWeights(),
        constraints=ScenarioConstraints(allow_late=True, allow_unassigned=True),
    )


# --------------------------------------------------------------------------
# job shop (OR-Library jobshop1 format), a second public benchmark
# --------------------------------------------------------------------------

JOBSHOP_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"
JOBSHOP_SOURCE = DataSource(
    name="OR-Library jobshop1 (Fisher & Thompson; Taillard; Lawrence; Adams et al.)",
    source="OR-Library, Brunel University London (J.E. Beasley, 1990)",
    url=JOBSHOP_URL,
    license="Free for academic and non-commercial use, as distributed by OR-Library.",
    purpose=(
        "A machine-capacity-bottleneck scheduling benchmark. It stresses the part of "
        "ORION's model the routing instances never touch: many tasks competing for a "
        "small set of single-capacity resources, with precedence chains - where "
        "makespan, not distance, is the quality signal."
    ),
    sha256="recorded in the experiment manifest at download time",
    download="python -m orion data fetch --dataset jobshop",
    preprocessing=(
        "Each operation becomes a Task at one shared location (no travel), with "
        "duration = processing time, release 0 and a horizon-wide deadline. The "
        "precedence chain inside a job becomes the Task dependency graph. Each machine "
        "becomes a Resource with capacity 1 and the skill of that machine class."
    ),
    limitations=(
        "Job-shop has no time windows, no geography and no priorities, so it exercises "
        "precedence and capacity only. ORION reports makespan; it does not claim parity "
        "with published job-shop optima, because ORION's objective is service-weighted "
        "and permits leaving work unassigned."
    ),
)


@dataclass(frozen=True, slots=True)
class JobShopInstance:
    name: str
    optimum: int
    jobs: tuple[tuple[tuple[int, int], ...], ...]  # (machine, duration) per job

    @property
    def machines(self) -> tuple[int, ...]:
        return tuple(sorted({m for job in self.jobs for m, _ in job}))

    @property
    def num_operations(self) -> int:
        return sum(len(job) for job in self.jobs)


def parse_jobshop(text: str) -> list[JobShopInstance]:
    """Parse the OR-Library ``jobshop1.txt`` 82-instance collection.

    The file is *line structured*, not a flat token stream, and it opens with a
    long prose preamble that contains numbers ("a set of 82 JSP test instances",
    "J. Adams, E. Balas and D. Zawack (1988)"). Scanning tokens from the start
    therefore mis-reads prose as a record header and fails with
    ``invalid literal for int()``.

    The real layout, repeated 82 times, is::

        instance abz5
         Adams, Balas, and Zawack 10x10 instance (Table 1, instance 5)
         10 10                     <- num_jobs, num_machines
         4 88 8 68 ...             <- one line per job: (machine, duration) pairs

    So records are located by the literal ``instance`` marker, and everything
    between markers is read by line.
    """
    instances: list[JobShopInstance] = []
    lines = text.splitlines()

    def _is_marker(line: str) -> bool:
        parts = line.split()
        return len(parts) == 2 and parts[0].lower() == "instance"

    marker_indices = [i for i, line in enumerate(lines) if _is_marker(line)]
    for start, stop in zip(marker_indices, marker_indices[1:] + [len(lines)]):
        name = lines[start].split()[1].strip()
        body = lines[start + 1 : stop]
        # The first two content lines are the description and the
        # "num_jobs num_machines" header; the separator rules ('+' / blank)
        # are interleaved, so they are skipped rather than assumed absent.
        content = [ln for ln in body if ln.strip() and set(ln.strip()) != {"+"}]
        if len(content) < 2:
            continue
        header = content[1].split()
        if len(header) < 2:
            continue
        try:
            num_jobs, _num_machines = int(header[0]), int(header[1])
        except ValueError:
            continue
        job_lines = content[2 : 2 + num_jobs]
        jobs: list[tuple[tuple[int, int], ...]] = []
        for job_line in job_lines:
            tokens = job_line.split()
            pairs = [
                (int(tokens[k]), int(tokens[k + 1]))
                for k in range(0, len(tokens) - 1, 2)
            ]
            if pairs:
                jobs.append(tuple(pairs))
        if jobs:
            instances.append(
                JobShopInstance(name=name, optimum=_optimum_for(name), jobs=tuple(jobs))
            )
    return instances


def load_jobshop(
    cache_dir: Path | str = "data/cache",
    names: Sequence[str] | None = None,
) -> dict[str, JobShopInstance]:
    """Load OR-Library job-shop instances, downloading once and caching.

    Keyless and free, like the Solomon source. The cached file is
    gitignored; its checksum is recorded in every experiment manifest that uses
    it, so a result can always be traced to the exact bytes it came from.
    """
    cache = Path(cache_dir)
    target = cache / "jobshop1.txt"
    if not target.exists():
        blob = _fetch(JOBSHOP_URL, cache, None)
        target.write_bytes(blob)
    text = target.read_text(encoding="utf-8", errors="replace")
    wanted = {n.strip().lower() for n in names} if names else None
    out: dict[str, JobShopInstance] = {}
    for instance in parse_jobshop(text):
        key = instance.name.strip().lower()
        if wanted is not None and key not in wanted:
            continue
        out[key] = instance
    if wanted:
        missing = sorted(wanted - set(out))
        if missing:
            raise KeyError(
                f"job-shop instances not found in jobshop1.txt: {missing}; "
                f"available: {sorted(i.name.lower() for i in parse_jobshop(text))[:20]}..."
            )
    return out


def _optimum_for(name: str) -> int:
    """Published optimum for *name*, or 0 when unknown.

    OR-Library's ``jobshop1.txt`` ships instance data and provenance notes but no
    optimal makespans, so there is nothing in the downloaded artifact to verify a
    number against. Rather than embed best-known values that no experiment
    artifact backs up, ORION reports 0 ("unknown") and the report says the
    published optimum was not available from the source used.

    A downstream comparison must therefore treat `optimum == 0` as "no
    reference", never as "optimum is zero".
    """
    del name  # no verified source available in the downloaded artifact
    return 0



def _machine_capability(machine: int) -> str:
    """Unique capability name for machine *machine*.

    Job-shop resources are interchangeable in every respect except which
    operations they may run, so the machine index is the only thing that has to
    distinguish them.
    """
    return f"machine-{int(machine):02d}"


def jobshop_to_scenario(
    instance: JobShopInstance,
    *,
    weights: ObjectiveWeights | None = None,
    name_prefix: str = "jobshop",
) -> Scenario:
    """Map a job-shop instance onto an operational scenario.

    Mapping:

    ==========================  ===========================================
    job-shop field              ORION field
    ==========================  ===========================================
    operation (machine, dur)    ``Task`` (duration=dur, location=``SITE``)
    operations of one job       ``Task.dependencies`` in chain order
    machine index               ``Resource`` with ``capacity=1``,
                                capability ``"machine"``
    - (none)                    deadline = horizon end (no time windows)
    ==========================  ===========================================

    ``deadline`` spans the whole horizon and priorities are uniform, so this
    instance tests **precedence and capacity**, not time windows or priority.
    """
    machines = instance.machines
    # The horizon has to be a *valid* upper bound on the makespan, or the adapter
    # manufactures infeasibility. A schedule cannot finish before the longest job
    # (its operations are sequential) nor before the busiest machine finishes its
    # whole load, so the horizon is at least that maximum. The previous formula -
    # twice the longest job - gave 1310 for ft10, while the instance's own bounds
    # already reach 655, and CP-SAT's first valid solution came in at 1309: pinned
    # against a ceiling the adapter had invented, and the reason every other
    # solver's schedule ran past the end of its shift.
    #
    # A simple serial schedule is always feasible (one machine at a time), giving
    # sum of all processing times as a hard upper bound. The horizon is placed
    # between the two so a plan is never truncated but the model still has to work.
    total_work = sum(d for job in instance.jobs for _, d in job)
    lower_bound = max(
        max(sum(d for _, d in job) for job in instance.jobs),
        max(
            (
                sum(d for job in instance.jobs for (m, d) in job if m == machine)
                for machine in machines
            ),
            default=0,
        ),
    )
    horizon_end = max(lower_bound * 2, total_work) or 100
    tasks: list[Task] = []
    for j, job in enumerate(instance.jobs):
        previous: str | None = None
        for k, (machine, duration) in enumerate(job):
            task_id = f"J{j + 1:02d}-O{k + 1:02d}"
            # Each operation can only run on its own machine, so the requirement is
            # that machine's unique capability. Giving every machine the shared
            # capability "machine" would make all 10 machines eligible for every
            # operation, which silently turns a job shop into a machine-free pool
            # and destroys the benchmark.
            tasks.append(
                Task(
                    id=task_id,
                    location="SITE",
                    release_time=0,
                    deadline=horizon_end,
                    duration=max(1, int(duration)),
                    priority=Priority.MEDIUM,
                    required_capabilities=(_machine_capability(machine),),
                    dependencies=(previous,) if previous else (),
                )
            )
            previous = task_id
    resources = tuple(
        Resource(
            id=f"M-{m:02d}",
            kind=ResourceKind.EQUIPMENT,
            capabilities=(_machine_capability(m),),
            home_location="SITE",
            shift_start=0,
            shift_end=horizon_end,
            # `capacity` is a mapping of resource name to amount, not a scalar.
            # A job-shop machine processes one operation at a time, so it has
            # capacity 1 under its own name.
            capacity={_machine_capability(m): 1.0},
            operating_cost_per_hour=0.0,
            fixed_dispatch_cost=0.0,
            max_work_minutes=horizon_end,
        )
        for m in machines
    )
    travel = TravelMatrix.from_coordinates({"SITE": (0.0, 0.0)}, default_speed_kmh=60.0, min_travel_minutes=0)
    return Scenario(
        id=f"{name_prefix}-{instance.name.lower()}",
        name=f"{name_prefix}:{instance.name}",
        horizon=Interval(0, horizon_end),
        tasks=tuple(tasks),
        resources=resources,
        travel=travel,
        objective_weights=weights or ObjectiveWeights(),
        constraints=ScenarioConstraints(allow_late=True, allow_unassigned=True),
    )


def all_data_sources() -> list[DataSource]:
    """Every external input ORION uses, for the data-provenance report."""
    return [SOLOMON_SOURCE, JOBSHOP_SOURCE]
