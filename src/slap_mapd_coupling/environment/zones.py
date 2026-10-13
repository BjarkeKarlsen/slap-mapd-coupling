"""Zones of the section-based controller and their halo.

The zone partition only has to be a disjoint cover of the movable vertices.
How the zones are drawn is a design choice made here. A zone is a block of
consecutive aisles, and a movable vertex belongs to the zone of its aisle,
cross-aisle vertices included. The number of zones ``p`` is a sweep
parameter and has no default.

Pure functions only, with no Ray and no torch. A hand-built graph can pass
its own zones and use ``check_partition`` and ``halo`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph


def _movable(graph: WarehouseGraph) -> frozenset[VertexId]:
    return frozenset(v for v, vertex in graph.vertices.items() if vertex.role.movable)


def zones_by_aisle(
    graph: WarehouseGraph,
    aisle_index: Mapping[VertexId, int],
    num_zones: int,
) -> tuple[frozenset[VertexId], ...]:
    """Split the movable vertices into ``num_zones`` blocks of consecutive aisles.

    The aisles that hold a movable vertex are sorted and cut into blocks that
    are as equal as possible, with the earlier blocks taking the extra aisle.
    Seven aisles into three zones give 3, 2 and 2 aisles. A movable vertex
    joins the zone of its aisle. Raises ValueError if ``num_zones`` is below 1
    or above the number of aisles.
    """
    movable = _movable(graph)
    aisles = sorted({aisle_index[v] for v in movable})
    if num_zones < 1:
        raise ValueError(f"num_zones must be at least 1, got {num_zones}.")
    if num_zones > len(aisles):
        raise ValueError(f"num_zones is {num_zones} but there are only {len(aisles)} aisles.")

    base, extra = divmod(len(aisles), num_zones)
    zone_of_aisle: dict[int, int] = {}
    start = 0
    for zone in range(num_zones):
        size = base + (1 if zone < extra else 0)
        for aisle in aisles[start : start + size]:
            zone_of_aisle[aisle] = zone
        start += size

    members: list[set[VertexId]] = [set() for _ in range(num_zones)]
    for v in movable:
        members[zone_of_aisle[aisle_index[v]]].add(v)
    return tuple(frozenset(m) for m in members)


def halo(
    graph: WarehouseGraph,
    zone: frozenset[VertexId],
    d_obs: float,
) -> frozenset[VertexId]:
    """The halo of a zone: movable vertices outside it within cost ``d_obs`` of it.

    A vertex ``v`` is in the halo when ``d_G(w, v) <= d_obs`` for some ``w``
    in the zone. With one zone holding every movable vertex the halo is empty.
    """
    outside = _movable(graph) - zone
    reached: set[VertexId] = set()
    for v in outside:
        for w in zone:
            try:
                within = graph.distance(w, v) <= d_obs
            except KeyError:
                within = False
            if within:
                reached.add(v)
                break
    return frozenset(reached)


def check_partition(graph: WarehouseGraph, zones: Sequence[frozenset[VertexId]]) -> None:
    """Raise ValueError unless the zones are a disjoint cover of the movable vertices."""
    seen: set[VertexId] = set()
    for index, zone in enumerate(zones):
        overlap = seen & zone
        if overlap:
            raise ValueError(f"Zone {index} overlaps an earlier zone on {sorted(overlap)}.")
        seen |= zone
    movable = _movable(graph)
    if seen != movable:
        missing = sorted(movable - seen)
        extra = sorted(seen - movable)
        raise ValueError(
            f"Zones are not a cover of the movable vertices. Missing {missing}, not movable {extra}."
        )


def zone_of(zones: Sequence[frozenset[VertexId]]) -> dict[VertexId, int]:
    """The index of the zone that owns each vertex."""
    return {v: index for index, zone in enumerate(zones) for v in zone}
