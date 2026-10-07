"""F_cng: F_dem's ranking extended with a congestion-exposure penalty
(sec:method:storage, eq:storagegreedy).

score(k,v) = rho_hat_t(k) * d_G(v, g*) + beta * sum_{e in path(g*,v)} mu_hat_t(e) * w_hat_t(e)

where g* is v's nearest delivery point and path(g*,v) is the shortest
path used to reach v from it. SKUs are still placed in the SAME outer
order F_dem uses (rho_hat_t descending -- "F_cng extends F_dem's
ranking," not a different structure); what changes is each SKU's OWN
vertex ranking, now scored per (SKU, vertex) pair instead of by access
distance alone, since score(k,v) genuinely depends on both. This
realises D + beta*K from eq:storageobjective greedily, same as F_dem
realises D alone, and reuses demand.py's greedy_fill unchanged (its
per-SKU vertex_order_by_sku parameter exists specifically for this).

Two points flagged, not literally specified:
- "g*" (the "nearest delivery point" the congestion term's path starts
  from) is taken to be the SAME point F_dem's access_distance already
  picks (nearest by d_G(v,g) ascending -- the direction score(k,v)'s own
  first term uses), even though the congestion term's own path is
  computed in the OTHER direction (g* to v, "the shortest path used to
  reach v from" it, as literally written). Reusing one "nearest" point
  for both terms is the coherent reading; picking two different nearest
  points via two different distance directions would be a second,
  unstated degree of freedom.
- WarehouseGraph.distance only returns a distance value, not the edges
  along a shortest path, so this module computes its own Dijkstra with
  predecessor tracking (_shortest_path_edges) to get eq:storagegreedy's
  path(v) as an explicit edge sequence.
"""

from __future__ import annotations

import heapq

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, StorageState
from slap_mapd_coupling.storage.base import DemandEstimate, EdgeKey, TrafficEstimate
from slap_mapd_coupling.storage.demand import delivery_vertices, greedy_fill, storage_vertices
from slap_mapd_coupling.storage.registry import register_storage_rule
from slap_mapd_coupling.storage.relocation import apply_relocation_cap


def nearest_delivery_point(
    graph: WarehouseGraph, vertex: VertexId, delivery: list[VertexId]
) -> VertexId:
    """g* = argmin_{g in V_del} d_G(v, g), ties broken by vertex id -- the
    same direction/tie-break F_dem's access_distance already uses."""
    return min(delivery, key=lambda g: (graph.distance(vertex, g), g))


def _shortest_path_edges(
    graph: WarehouseGraph, source: VertexId, target: VertexId
) -> list[EdgeKey]:
    """The edge sequence of one shortest (cost-weighted) path from source
    to target, via Dijkstra with predecessor tracking. WarehouseGraph's
    own `distance` only returns the distance value, not the path itself,
    which eq:storagegreedy's path(v) congestion sum needs."""
    if source == target:
        return []
    edge_cost = {(e.source, e.target): e.cost for e in graph.edges}
    dist: dict[VertexId, float] = {source: 0.0}
    predecessor: dict[VertexId, VertexId] = {}
    frontier: list[tuple[float, VertexId]] = [(0.0, source)]
    while frontier:
        d, v = heapq.heappop(frontier)
        if d > dist.get(v, float("inf")):
            continue
        if v == target:
            break
        for w in graph.out_neighbours(v):
            nd = d + edge_cost[(v, w)]
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                predecessor[w] = v
                heapq.heappush(frontier, (nd, w))

    if target not in dist:
        raise KeyError(f"{target} is unreachable from {source}.")

    vertices = [target]
    while vertices[-1] != source:
        vertices.append(predecessor[vertices[-1]])
    vertices.reverse()
    return list(zip(vertices, vertices[1:]))


def congestion_exposure(
    graph: WarehouseGraph,
    vertex: VertexId,
    nearest_delivery: VertexId,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
) -> float:
    """sum_{e in path(g*,v)} mu_hat_t(e) * w_hat_t(e)."""
    path = _shortest_path_edges(graph, nearest_delivery, vertex)
    return sum(traversal_estimate.get(e, 0.0) * waiting_estimate.get(e, 0.0) for e in path)


def score(
    graph: WarehouseGraph,
    sku_id: SkuId,
    vertex: VertexId,
    delivery: list[VertexId],
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
    congestion_weight: float,
) -> float:
    """score(k,v) (eq:storagegreedy)."""
    nearest = nearest_delivery_point(graph, vertex, delivery)
    access_term = demand_estimate.get(sku_id, 0.0) * graph.distance(vertex, nearest)
    exposure_term = congestion_weight * congestion_exposure(
        graph, vertex, nearest, traversal_estimate, waiting_estimate
    )
    return access_term + exposure_term


def ranked_storage_vertices_for(
    graph: WarehouseGraph,
    sku_id: SkuId,
    storage: list[VertexId],
    delivery: list[VertexId],
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
    congestion_weight: float,
) -> list[VertexId]:
    """V_str sorted by score(k, v) ascending for this one SKU, ties by
    vertex id."""
    return sorted(
        storage,
        key=lambda v: (
            score(
                graph,
                sku_id,
                v,
                delivery,
                demand_estimate,
                traversal_estimate,
                waiting_estimate,
                congestion_weight,
            ),
            v,
        ),
    )


@register_storage_rule("congestion")
def f_cng(
    x_prev: StorageState,
    graph: WarehouseGraph,
    reassignment_cap: int | None,
    congestion_weight: float | None,
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
) -> StorageState:
    if reassignment_cap is None:
        raise ValueError(
            "storage_mode='congestion' requires reassignment_cap (nu); "
            "ExperimentConfig already enforces this, so this indicates a caller bug."
        )
    if congestion_weight is None:
        raise ValueError(
            "storage_mode='congestion' requires congestion_weight (beta); "
            "ExperimentConfig already enforces this, so this indicates a caller bug."
        )

    storage = storage_vertices(graph)
    delivery = delivery_vertices(graph)
    sku_order = sorted(x_prev.skus, key=lambda k: (-demand_estimate.get(k, 0.0), k))
    vertex_order_by_sku = {
        sku_id: ranked_storage_vertices_for(
            graph,
            sku_id,
            storage,
            delivery,
            demand_estimate,
            traversal_estimate,
            waiting_estimate,
            congestion_weight,
        )
        for sku_id in x_prev.skus
    }
    x_target, priority = greedy_fill(x_prev, vertex_order_by_sku, sku_order)
    return apply_relocation_cap(x_prev, x_target, priority, reassignment_cap)
