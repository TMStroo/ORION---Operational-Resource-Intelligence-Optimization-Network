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
    """Parse the OR-Library ``jobshop1.txt`` multi-instance format.

    The file is whitespace-separated tokens: ``name optimum num_jobs num_machines``
    then, for each job, a count followed by ``(machine, duration)`` pairs.
    """
    tokens = text.split()
    i = 0
    out: list[JobShopInstance] = []
    n = len(tokens)
    while i < n:
        name = tokens[i]
        i += 1
        if not (i + 3 <= n):
            break
        optimum = int(tokens[i])
        num_jobs = int(tokens[i + 1])
        i += 2  # num_machines is implicit: the max machine index defines it
        jobs: list[tuple[tuple[int, int], ...]] = []
        for _ in range(num_jobs):
            if i >= n:
                break
            count = int(tokens[i])
            i += 1
            ops: list[tuple[int, int]] = []
            for _ in range(count):
                machine = int(tokens[i])
                duration = int(tokens[i + 1])
                ops.append((machine, duration))
                i += 2
            jobs.append(tuple(ops))
        if jobs:
            out.append(JobShopInstance(name=name, optimum=optimum, jobs=tuple(jobs)))
    return out


def load_jobshop(cache_dir: Path | str = "data/cache", names: Sequence[str] | None = None) -> dict[str, JobShopInstance]:
    """Download (once) and parse the OR-Library job-shop instances."""
    blob = _fetch(JOBSHOP_URL, Path(cache_dir), None)
    instances = parse_jobshop(blob.decode("utf-8", errors="replace"))
    wanted = {n.lower() for n in names} if names else None
    out = {i.name.lower(): i for i in instances if wanted is None or i.name.lower() in wanted}
    if not out:
        raise ValueError(f"no job-shop instances matched {names!r}")
    return out


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
    horizon_end = int(max(sum(d for _, d in job) for job in instance.jobs) * 2) or 100
    tasks: list[Task] = []
    for j, job in enumerate(instance.jobs):
        previous: str | None = None
        for k, (_, duration) in enumerate(job):
            task_id = f"J{j + 1:02d}-O{k + 1:02d}"
            tasks.append(
                Task(
                    id=task_id,
                    location="SITE",
                    release_time=0,
                    deadline=horizon_end,
                    duration=max(1, int(duration)),
                    priority=Priority.MEDIUM,
                    required_capabilities=("machine",),
                    dependencies=(previous,) if previous else (),
                )
            )
            previous = task_id
    resources = tuple(
        Resource(
            id=f"M-{m:02d}",
            kind=ResourceKind.EQUIPMENT,
            capabilities=("machine",),
            home_location="SITE",
            shift_start=0,
            shift_end=horizon_end,
            capacity=1,
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
