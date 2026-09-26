
"""Travel graph and travel-time model.

ORION models travel as a **complete directed metric with explicit per-resource
speed**, not as a shortest-path computation over a road network. This is a
deliberate modelling decision, documented so a reviewer can challenge it:

* The optimisation formulations in :mod:`orion.optimization` need the travel
  time between every (task, task) and (resource, task) pair as a coefficient.
  A routing solver over a sparse road graph would require path-internal nodes
  and arc variables, which triples the model size and makes the comparison
  against the constructive heuristic unfair (the heuristic would not be
  optimising the same thing).
* The decision ORION needs to support - "how much does moving from A to B cost
  *this* resource" - is fully captured by a distance matrix plus a speed.
* Real road networks are still supported: :class:`TravelMatrix.from_networkx`
  consumes any weighted graph and produces the same matrix, so a scenario built
  from an OpenStreetMap extract drops in without touching the solvers.

Travel is Euclidean in the base units, scaled by a per-resource speed factor and
inflated by a congestion factor a scenario may set. The congestion multiplier is
how ORION models "traffic got worse" as a disruption: it is a single scalar, so
the effect on the whole plan is measurable and the replan comparison is
interpretable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

from orion.domain.errors import TravelGraphError, ValidationError


class Coordinate(Protocol):
    """Anything with ``x``/``y`` attributes, or a 2-tuple."""

    x: float
    y: float


def _xy(node: object) -> tuple[float, float]:
    """Extract an (x, y) pair from a tuple or an object with x/y attributes."""
    if isinstance(node, (tuple, list)) and len(node) == 2:
        return float(node[0]), float(node[1])
    x = getattr(node, "x", None)
    y = getattr(node, "y", None)
    if x is None or y is None:
        raise TravelGraphError(
            f"node {node!r} has no coordinates; expected (x, y) or .x/.y attributes"
        )
    return float(x), float(y)


@dataclass(frozen=True, slots=True)
class TravelMatrix:
    """Travel time in minutes and distance in km between named locations.

    The matrix is symmetric because the base model is Euclidean distance. The
    class still exposes directed ``travel_time(a, b)`` accessors so a scenario
    with genuinely asymmetric roads (one-way streets, ferries) can subclass
    without changing any solver.
    """

    locations: tuple[str, ...]
    distance_km: tuple[tuple[float, ...], ...]
    travel_minutes: tuple[tuple[int, ...], ...]
    #: Multiplier applied to every leg, used for "traffic got worse" events.
    congestion_factor: float = 1.0
    #: index lookup for O(1) access
    _index: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_coordinates(
        cls,
        coordinates: Mapping[str, object],
        *,
        default_speed_kmh: float = 45.0,
        congestion_factor: float = 1.0,
        min_travel_minutes: int = 1,
    ) -> TravelMatrix:
        """Build a matrix from ``{location_id: (x, y)}`` using Euclidean distance.

        ``default_speed_kmh`` converts kilometres into minutes. The 1-minute
        floor exists because a zero-minute leg creates no incentive for the
        solver to sequence two co-located tasks in a particular order, which in
        turn makes local repair unable to distinguish "same site" from
        "adjacent site".
        """
        if not coordinates:
            raise TravelGraphError("travel matrix requires at least one location")
        if default_speed_kmh <= 0:
            raise ValueError(f"speed must be > 0, got {default_speed_kmh}")
        if congestion_factor <= 0:
            raise ValueError(f"congestion factor must be > 0, got {congestion_factor}")
        if min_travel_minutes < 0:
            raise ValueError("min_travel_minutes must be >= 0")

        locations = tuple(sorted(coordinates))
        index = {loc: i for i, loc in enumerate(locations)}
        n = len(locations)
        dist = [[0.0] * n for _ in range(n)]
        minutes = [[0] * n for _ in range(n)]

        for i, a in enumerate(locations):
            ax, ay = _xy(coordinates[a])
            for j in range(i + 1, n):
                b = locations[j]
                bx, by = _xy(coordinates[b])
                d = math.hypot(bx - ax, by - ay)
                dist[i][j] = dist[j][i] = d
                t = max(min_travel_minutes, int(round(d / default_speed_kmh * 60.0)))
                minutes[i][j] = minutes[j][i] = t
        return cls(
            locations=locations,
            distance_km=tuple(tuple(row) for row in dist),
            travel_minutes=tuple(tuple(row) for row in minutes),
            congestion_factor=float(congestion_factor),
            _index=index,
        )

    @classmethod
    def from_networkx(
        cls,
        graph: object,
        *,
        weight: str = "distance_km",
        default_speed_kmh: float = 45.0,
        congestion_factor: float = 1.0,
    ) -> TravelMatrix:
        """Build a matrix from a ``networkx`` weighted graph.

        Uses network shortest paths by default. This is the path real road
        networks enter ORION, and it is covered by a test that checks the
        resulting matrix against a hand-computed three-node chain.
        """
        import networkx as nx  # imported lazily: only network-backed scenarios need it

        if not isinstance(graph, nx.Graph):
            raise TravelGraphError("from_networkx requires a networkx.Graph")
        locations = tuple(sorted(graph.nodes()))
        index = {loc: i for i, loc in enumerate(locations)}
        n = len(locations)
        dist = [[0.0] * n for _ in range(n)]
        minutes = [[0] * n for _ in range(n)]

        for i, a in enumerate(locations):
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    graph, a, weight=weight, cutoff=None
                )
            except nx.NetworkXNoPath as exc:  # pragma: no cover - guarded by validation
                raise TravelGraphError(
                    f"location {a!r} is unreachable; travel graph is disconnected"
                ) from exc
            for b, length in lengths.items():
                j = index[b]
                dist[i][j] = float(length)
                minutes[i][j] = max(1, int(round(float(length) / default_speed_kmh * 60.0)))
        # symmetrise: shortest-path distances are symmetric for undirected graphs
        for i in range(n):
            for j in range(i + 1, n):
                d = min(dist[i][j], dist[j][i])
                t = min(minutes[i][j], minutes[j][i])
                dist[i][j] = dist[j][i] = d
                minutes[i][j] = minutes[j][i] = t
        return cls(
            locations=locations,
            distance_km=tuple(tuple(row) for row in dist),
            travel_minutes=tuple(tuple(row) for row in minutes),
            congestion_factor=float(congestion_factor),
            _index=index,
        )

    # -- accessors ---------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.locations)
        if len(self.distance_km) != n or len(self.travel_minutes) != n:
            raise ValidationError("travel matrix must be square over its locations")
        for row in self.distance_km:
            if len(row) != n:
                raise ValidationError("travel distance matrix is not square")
        for row in self.travel_minutes:
            if len(row) != n:
                raise ValidationError("travel time matrix is not square")
        for i, row in enumerate(self.travel_minutes):
            for j, t in enumerate(row):
                if t < 0:
                    raise ValidationError(
                        f"negative travel time {t} between "
                        f"{self.locations[i]} and {self.locations[j]}"
                    )
        object.__setattr__(self, "_index", {loc: i for i, loc in enumerate(self.locations)})

    @property
    def size(self) -> int:
        return len(self.locations)

    def index(self, location: str) -> int:
        try:
            return self._index[location]
        except KeyError as exc:
            raise TravelGraphError(
                f"unknown location {location!r}; known locations: "
                f"{list(self.locations)[:8]}{'...' if len(self.locations) > 8 else ''}"
            ) from exc

    def has(self, location: str) -> bool:
        return location in self._index

    def distance(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.distance_km[self.index(a)][self.index(b)]

    def base_travel_minutes(self, a: str, b: str) -> int:
        if a == b:
            return 0
        return self.travel_minutes[self.index(a)][self.index(b)]

    def travel_minutes_scaled(self, a: str, b: str) -> int:
        """Travel minutes including the current congestion factor."""
        if a == b:
            return 0
        raw = self.travel_minutes[self.index(a)][self.index(b)]
        return max(1, int(round(raw * self.congestion_factor)))

    def scaled(self, factor: float) -> TravelMatrix:
        """Return a copy with a different congestion factor."""
        if factor <= 0:
            raise ValueError(f"congestion factor must be > 0, got {factor}")
        return TravelMatrix(
            locations=self.locations,
            distance_km=self.distance_km,
            travel_minutes=self.travel_minutes,
            congestion_factor=float(factor),
            _index=dict(self._index),
        )

    def travel_from_speed(
        self, a: str, b: str, speed_factor: float, *, congestion: float | None = None
    ) -> int:
        """Travel minutes for a resource moving at ``speed_factor`` x base speed.

        ``speed_factor`` of 2.0 halves travel time (fast vehicle). The result is
        floored at 1 minute for any non-identical pair, matching
        :meth:`from_coordinates`.
        """
        if a == b:
            return 0
        if speed_factor <= 0:
            raise ValueError(f"speed factor must be > 0, got {speed_factor}")
        base = self.travel_minutes[self.index(a)][self.index(b)]
        factor = self.congestion_factor if congestion is None else congestion
        return max(1, int(round(base * factor / speed_factor)))

    def distance_from_speed(self, a: str, b: str, speed_factor: float) -> float:
        """Distance-equivalent metres used as the cost proxy for movement.

        Travel *cost* in ORION is expressed as ``distance / speed_factor`` in
        "cost-minutes", so a slow resource pays more per kilometre. This keeps
        the objective free of scenario-specific price tables while still making
        "prefer the nearby fast resource" an economic decision.
        """
        if speed_factor <= 0:
            raise ValueError(f"speed factor must be > 0, got {speed_factor}")
        return self.distance(a, b) / speed_factor

    def total_route(self, sequence: Iterable[str], speed_factor: float = 1.0) -> tuple[int, float]:
        """Return ``(minutes, km)`` for visiting *sequence* in order."""
        seq = list(sequence)
        if len(seq) < 2:
            return 0, 0.0
        minutes = sum(
            self.travel_from_speed(seq[i], seq[i + 1], speed_factor) for i in range(len(seq) - 1)
        )
        km = sum(self.distance(seq[i], seq[i + 1]) for i in range(len(seq) - 1))
        return minutes, round(km, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "locations": list(self.locations),
            "congestion_factor": self.congestion_factor,
            "distance_km": [list(row) for row in self.distance_km],
            "travel_minutes": [list(row) for row in self.travel_minutes],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TravelMatrix:
        return cls(
            locations=tuple(str(loc) for loc in data["locations"]),  # type: ignore[index]
            distance_km=tuple(tuple(float(v) for v in row) for row in data["distance_km"]),  # type: ignore[index]
            travel_minutes=tuple(tuple(int(v) for v in row) for row in data["travel_minutes"]),  # type: ignore[index]
            congestion_factor=float(data.get("congestion_factor", 1.0)),  # type: ignore[arg-type]
        )


__all__ = ["TravelMatrix", "Coordinate"]
