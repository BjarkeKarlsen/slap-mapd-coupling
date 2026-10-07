"""F_dem: demand-only slotting heuristic (sec:method:storage).

Sorts K by rho_hat_t(k) descending, sorts V_str by
min_{g in V_del} d_G(v,g) ascending, and places SKUs into vertices in
that joint order, skipping a vertex once it is at capacity
(eq:feasiblestorage). This is argmin_x D(x; rho_hat_t) from
eq:storageobjective with beta=chi=0, solved greedily rather than exactly
-- ties in either sort are broken by id, since eq:storageupdate must be
a genuine function (sec:pf:storage), and the SKU-major/vertex-minor
iteration order this produces is also the "descending marginal score"
order storage/relocation.py's cap uses (see its own docstring for why
that reuse, rather than a separate scoring mechanism, was the agreed
resolution).

If the fleet's total capacity can't fit every unit of every SKU under
this particular greedy ordering (a different ordering might have),
whatever doesn't fit is simply not placed -- an accepted consequence of
"solved greedily rather than exactly," not a bug to work around here.
"""

from __future__ import annotations

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, StorageState
from slap_mapd_coupling.storage.base import DemandEstimate, TrafficEstimate
from slap_mapd_coupling.storage.registry import register_storage_rule
from slap_mapd_coupling.storage.relocation import apply_relocation_cap


def storage_vertices(graph: WarehouseGraph) -> list[VertexId]:
    return [v for v, vertex in graph.vertices.items() if vertex.role.storage]


def delivery_vertices(graph: WarehouseGraph) -> list[VertexId]:
    return [v for v, vertex in graph.vertices.items() if vertex.role.delivery]


def access_distance(graph: WarehouseGraph, vertex: VertexId, delivery: list[VertexId]) -> float:
    """min_{g in V_del} d_G(v, g): how far a storage vertex sits from the
    nearest delivery point."""
    return min(graph.distance(vertex, g) for g in delivery)


def ranked_storage_vertices(graph: WarehouseGraph) -> list[VertexId]:
    """V_str sorted by access_distance ascending, ties by vertex id."""
    delivery = delivery_vertices(graph)
    return sorted(storage_vertices(graph), key=lambda v: (access_distance(graph, v, delivery), v))


def greedy_fill(
    x_prev: StorageState,
    vertex_order_by_sku: dict[SkuId, list[VertexId]],
    sku_order: list[SkuId],
) -> tuple[StorageState, list[tuple[SkuId, VertexId]]]:
    """The shared greedy bin-packing loop: place each SKU's total units
    (conserved from x_prev) into that SKU's own ranked vertex order
    (`vertex_order_by_sku[sku_id]`), skipping any vertex once it's full.
    `sku_order` fixes which SKU goes first. F_dem and F_cng differ only
    in how they compute `sku_order` and each SKU's vertex ranking (F_dem
    uses the SAME ranking for every SKU; F_cng's score(k,v) makes it
    genuinely per-SKU) -- this loop itself is identical, so F_cng reuses
    it too, rather than duplicating the bin-packing logic.

    Returns the resulting (uncapped) target StorageState, plus the
    ordered list of (sku, vertex) cells that gained units relative to
    x_prev -- storage/relocation.py's `priority` argument, in exactly
    the order this fill placed them.
    """
    remaining_capacity = dict(x_prev.capacities)
    new_counts: dict[SkuId, dict[VertexId, int]] = {k: {} for k in x_prev.skus}
    placements: list[tuple[SkuId, VertexId]] = []

    for sku_id in sku_order:
        remaining_units = sum(x_prev.counts.get(sku_id, {}).values())
        unit_capacity = x_prev.skus[sku_id].unit_capacity
        for vertex in vertex_order_by_sku[sku_id]:
            if remaining_units <= 0:
                break
            fit = int(remaining_capacity[vertex] // unit_capacity)
            if fit <= 0:
                continue
            place = min(fit, remaining_units)
            new_counts[sku_id][vertex] = new_counts[sku_id].get(vertex, 0) + place
            remaining_capacity[vertex] -= place * unit_capacity
            remaining_units -= place
            placements.append((sku_id, vertex))

    x_target = StorageState(skus=x_prev.skus, capacities=x_prev.capacities, counts=new_counts)
    priority = [
        (sku_id, vertex)
        for sku_id, vertex in placements
        if x_target.units(sku_id, vertex) > x_prev.units(sku_id, vertex)
    ]
    return x_target, priority


@register_storage_rule("demand")
def f_dem(
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
            "storage_mode='demand' requires reassignment_cap (nu); "
            "ExperimentConfig already enforces this, so this indicates a caller bug."
        )
    sku_order = sorted(x_prev.skus, key=lambda k: (-demand_estimate.get(k, 0.0), k))
    storage_order = ranked_storage_vertices(graph)
    vertex_order_by_sku = {sku_id: storage_order for sku_id in x_prev.skus}
    x_target, priority = greedy_fill(x_prev, vertex_order_by_sku, sku_order)
    return apply_relocation_cap(x_prev, x_target, priority, reassignment_cap)
