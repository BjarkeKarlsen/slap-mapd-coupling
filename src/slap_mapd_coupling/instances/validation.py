"""Connectivity and well-formedness checks for generated warehouse instances (Ma et al. 2017)."""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph


class ConnectivityReport(BaseModel):
    """(a) reachability: every vertex ever assigned as a task target must
    be reachable from every occupiable vertex -- weaker than blanket
    strong connectivity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    num_pairs_checked: NonNegativeInt
    unreachable_pairs: tuple[tuple[VertexId, VertexId], ...]  # (source, target)

    @property
    def failure_rate(self) -> float:
        if self.num_pairs_checked == 0:
            return 0.0
        return len(self.unreachable_pairs) / self.num_pairs_checked


class WellFormednessReport(BaseModel):
    """(b) well-formedness (Ma et al. 2017), adapted for a directed graph:
    every pair of endpoints needs a path in BOTH directions, each
    avoiding every third endpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    num_endpoints: NonNegativeInt  # |V_ep u V_str u V_del|
    required_endpoints: NonNegativeInt  # m (fleet size)
    endpoint_count_ok: bool
    blocked_pairs: tuple[tuple[VertexId, VertexId, VertexId], ...]  # (v, w, blocking_endpoint)

    @property
    def failure_rate(self) -> float:
        """Fraction of ordered endpoint pairs that are blocked.

        0.0 if endpoint_count_ok is False (the pairwise check isn't
        meaningful below m endpoints) or fewer than 2 endpoints exist.
        """
        if not self.endpoint_count_ok or self.num_endpoints < 2:
            return 0.0
        total_pairs = self.num_endpoints * (self.num_endpoints - 1)
        return len(self.blocked_pairs) / total_pairs


class InstanceGenerationReport(BaseModel):
    """One generation attempt's outcome -- what a batch-generation run
    accumulates across seeds to report a discard rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    accepted: bool
    connectivity: ConnectivityReport
    well_formedness: WellFormednessReport | None  # None if not requested


def _reachable_avoiding(
    graph: WarehouseGraph, start: VertexId, goal: VertexId, avoid: set[VertexId]
) -> bool:
    """BFS from start to goal over N+(.), never entering a vertex in `avoid`."""
    if start == goal:
        return True
    visited = {start}
    queue: deque[VertexId] = deque([start])
    while queue:
        v = queue.popleft()
        for w in graph.out_neighbours(v):
            if w == goal:
                return True
            if w not in avoid and w not in visited:
                visited.add(w)
                queue.append(w)
    return False


def check_connectivity(
    graph: WarehouseGraph, target_vertices: set[VertexId] | None = None
) -> ConnectivityReport:
    """(a): for every v in V_mov and every w in target_vertices (default:
    V_str | V_del), d_G(v, w) must be finite. NOT blanket strong
    connectivity -- only checks (any occupiable vertex, any target vertex).
    """
    if target_vertices is None:
        target_vertices = {
            v for v, vertex in graph.vertices.items() if vertex.role.storage or vertex.role.delivery
        }
    movable = [v for v, vertex in graph.vertices.items() if vertex.role.movable]

    unreachable: list[tuple[VertexId, VertexId]] = []
    num_pairs_checked = 0
    for v in movable:
        for w in target_vertices:
            num_pairs_checked += 1
            try:
                graph.distance(v, w)
            except KeyError:
                unreachable.append((v, w))

    return ConnectivityReport(
        ok=not unreachable,
        num_pairs_checked=num_pairs_checked,
        unreachable_pairs=tuple(unreachable),
    )


def check_well_formedness(graph: WarehouseGraph, fleet_size: int) -> WellFormednessReport:
    """(b): endpoints := V_ep | V_str | V_del. Requires |endpoints| >=
    fleet_size AND, for every ordered pair of distinct endpoints (v, w):
    a v->w path AND a w->v path, each avoiding every other endpoint.
    """
    endpoints = sorted(
        v
        for v, vertex in graph.vertices.items()
        if vertex.role.endpoint or vertex.role.storage or vertex.role.delivery
    )
    num_endpoints = len(endpoints)
    endpoint_count_ok = num_endpoints >= fleet_size

    blocked_pairs: list[tuple[VertexId, VertexId, VertexId]] = []
    if endpoint_count_ok:
        endpoint_set = set(endpoints)
        for v in endpoints:
            for w in endpoints:
                if v == w:
                    continue
                others = endpoint_set - {v, w}
                if others and not _reachable_avoiding(graph, v, w, others):
                    # Report a single blocking endpoint: prefer one whose
                    # individual removal already breaks every path (the
                    # case the thesis's own figure illustrates); fall back
                    # to an arbitrary member of `others` if the block only
                    # arises from several endpoints jointly.
                    blocker = next(
                        (b for b in others if not _reachable_avoiding(graph, v, w, {b})),
                        next(iter(others)),
                    )
                    blocked_pairs.append((v, w, blocker))

    ok = endpoint_count_ok and not blocked_pairs
    return WellFormednessReport(
        ok=ok,
        num_endpoints=num_endpoints,
        required_endpoints=fleet_size,
        endpoint_count_ok=endpoint_count_ok,
        blocked_pairs=tuple(blocked_pairs),
    )
