"""Connectivity and well-formedness checks for generated warehouse instances (Ma et al. 2017)."""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

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
    """(b) well-formedness (Ma et al. 2017, sec:pf:agents), adapted for a
    directed graph: there are at least m non-task endpoints, and every pair of
    endpoints needs a path in BOTH directions, each avoiding every third
    endpoint. A non-task endpoint is a vertex of V_ep in none of V_str, V_del and
    V_rcv. The pairwise check takes the endpoints to be V_ep u V_str u V_del u V_rcv,
    since a put-away task is picked up at a receiving vertex (app:wellformed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    num_endpoints: NonNegativeInt  # |V_ep u V_str u V_del u V_rcv|, for the pairwise check
    num_non_task_endpoints: NonNegativeInt  # |V_ep minus (V_str u V_del u V_rcv)|
    required_endpoints: NonNegativeInt  # m (fleet size), the least number of non-task endpoints
    endpoint_count_ok: bool  # num_non_task_endpoints >= required_endpoints
    blocked_pairs: tuple[tuple[VertexId, VertexId, VertexId], ...]  # (v, w, blocking_endpoint)

    @property
    def failure_rate(self) -> float:
        """Fraction of ordered endpoint pairs that are blocked.

        0.0 if endpoint_count_ok is False (the pairwise check is skipped
        when there are too few non-task endpoints) or fewer than 2
        endpoints exist.
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
    # Topologies sampled for this seed. attempts - 1 were discarded.
    topology_attempts: PositiveInt = 1


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
    V_str | V_del | V_rcv), d_G(v, w) must be finite. NOT blanket strong
    connectivity -- only checks (any occupiable vertex, any target vertex).
    """
    if target_vertices is None:
        target_vertices = {
            v
            for v, vertex in graph.vertices.items()
            if vertex.role.storage or vertex.role.delivery or vertex.role.receiving
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
    """(b): requires at least fleet_size non-task endpoints, the vertices of
    V_ep in neither V_str nor V_del, AND, with endpoints := V_ep | V_str |
    V_del | V_rcv, for every ordered pair of distinct endpoints (v, w): a v->w path
    AND a w->v path, each avoiding every other endpoint (sec:pf:agents).

    Token Passing parks an agent on a non-task endpoint where no other path
    ends. Fewer than m of them can leave none free (Ma et al. 2017, Property 2),
    and the environment places the agents on V_ep, so it needs m of those.
    """
    endpoints = sorted(v for v, vertex in graph.vertices.items() if vertex.role.is_task_endpoint)
    num_endpoints = len(endpoints)
    num_non_task_endpoints = sum(
        1
        for vertex in graph.vertices.values()
        if vertex.role.endpoint
        and not vertex.role.storage
        and not vertex.role.delivery
        and not vertex.role.receiving
    )
    endpoint_count_ok = num_non_task_endpoints >= fleet_size

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
        num_non_task_endpoints=num_non_task_endpoints,
        required_endpoints=fleet_size,
        endpoint_count_ok=endpoint_count_ok,
        blocked_pairs=tuple(blocked_pairs),
    )
