"""Put-away rules of the arrival stock model (thesis wording pending).

Under the arrival stock model (config.stock_model == "arrival") an order line takes its units
by the pickup rule when it is released, so picks empty the shelves. Every Delta steps new stock arrives at the
receiving vertices, and the storage update becomes the put-away of that stock: the storage rule
decides which shelves the arriving units go to. Nothing already on a shelf moves, so a SKU can
sit on several shelves at once, old stock on one and new stock on another.

How much arrives does not depend on the rule (`arrival_amounts`). Each SKU is filled up to
max(floor, ceil((1 + epsilon) * its units ordered in the last period)), plus the units lines
still wait for, less what is on hand. Where it goes is the rule's choice, through one shared fill loop
(`_put_away`) that only differs in the order the SKUs are placed in and each SKU's order of
shelves:

- "fixed" (dedicated storage, F_fix). SKUs by id. A SKU's units go to its home shelf, and once
  that is full to the other storage vertices by access distance min_{g in V_del} d_G(v, g),
  ties by vertex id. Units placed off the home shelf are overflow.
- "demand" (F_dem). SKUs by rho_hat_t(k) descending, ties by id, each into the storage vertices
  by access distance, ties by id, the same order for every SKU.
- "congestion" (F_cng). SKUs as in "demand", each into its own order of storage vertices by the
  congestion-aware score with weight beta (storage/congestion.py).

A shelf takes as many units as fit in its free capacity. Units that fit nowhere are not placed
and are counted as unplaced.

When robots carry the put-away (config.stock_putaway == "robots") every rule judges a shelf by
the trip distance d^trip(v) = min_{g in V_rcv} d_G(g, v) + min_{g in V_del} d_G(v, g), the
trip of a unit through it (sec:method:storage). "fixed" sends overflow and "demand" sends every
SKU to the shelves in order of d^trip, ties by vertex id, and the congestion-aware score of
"congestion" is rho_hat_t(k) * d^trip(v) plus beta times the traffic over both legs of the
trip. With instant put-away the rules are as before.

Choices flagged, not fixed by the thesis yet:
- The units of SKU k ordered in the last period are counted as round(rho_hat^unit_t(k) * Delta),
  the exact count, since the estimate is per step. Taking the ceiling of the float product
  (1 + epsilon) * rho_hat^unit_t(k) * Delta directly could round an exact count up by one unit.
- The number of units of SKU k a shelf v takes is int(free(v) // b_k + 1e-9), the tolerance
  keeping a floating-point remainder from losing a unit that fits.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Protocol, runtime_checkable

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, StorageState
from slap_mapd_coupling.storage.base import DemandEstimate, TrafficEstimate
from slap_mapd_coupling.storage.congestion import (
    _shortest_path_edges,
    nearest_delivery_point,
    ranked_storage_vertices_for,
)
from slap_mapd_coupling.storage.demand import (
    delivery_vertices,
    ranked_storage_vertices,
    storage_vertices,
)


class PutAwayResult(NamedTuple):
    """What one arrival leaves behind: the storage state after the put-away and its measures."""

    storage: StorageState
    arrived_units: int  # every unit that arrived, placed or not
    unplaced_units: int  # units that fit on no shelf
    overflow_units: int  # "fixed" only: units placed off the SKU's home shelf


@runtime_checkable
class PutAwayRule(Protocol):
    """The storage update under the arrival stock model: where the arriving units go.

    A plain callable, like storage/base.py::StorageRule. `amounts` are the units arriving per
    SKU (`arrival_amounts`), the same under every rule. `home` maps each SKU to its home shelf,
    which only "fixed" uses, and `congestion_weight` is beta, which only "congestion" uses.
    `receiving` is V_rcv, None for the delivery vertices. `trip` ranks the shelves by the trip
    distance (robots carry the put-away), False by the access distance as before.
    The returned storage satisfies the capacity constraint, which StorageState enforces.
    """

    def __call__(
        self,
        x_prev: StorageState,
        graph: WarehouseGraph,
        amounts: dict[SkuId, int],
        home: dict[SkuId, VertexId],
        demand_estimate: DemandEstimate,
        traversal_estimate: TrafficEstimate,
        waiting_estimate: TrafficEstimate,
        congestion_weight: float | None,
        *,
        receiving: list[VertexId] | None = None,
        trip: bool = False,
    ) -> PutAwayResult: ...


_REGISTRY: dict[str, PutAwayRule] = {}


def register_putaway_rule(name: str) -> Callable[[PutAwayRule], PutAwayRule]:
    """Decorator: @register_putaway_rule("fixed")."""

    def decorator(rule: PutAwayRule) -> PutAwayRule:
        if name in _REGISTRY:
            raise ValueError(f"Put-away rule {name!r} is already registered.")
        _REGISTRY[name] = rule
        return rule

    return decorator


def get_putaway_rule(name: str) -> PutAwayRule:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown put-away rule {name!r}; registered: {registered_putaway_rules()}.")
    return _REGISTRY[name]


def registered_putaway_rules() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def arrival_amounts(
    x_prev: StorageState,
    unit_estimate: DemandEstimate,
    epoch_length: int,
    margin: float,
    floor: int,
    waiting_by_sku: dict[SkuId, int],
) -> dict[SkuId, int]:
    """The arrivals, the units of every SKU that arrive at an epoch, the same under every rule.

    SKU k is filled up to max(floor, ceil((1 + epsilon) * n_k)), n_k being the units of k
    ordered in the period that ends now (rho_hat^unit), plus the units that lines still wait for
    (`waiting_by_sku`). What is on hand counts towards that, and nothing is taken away, so a SKU
    with more than enough gets none. Under robot put-away `x_prev` holds the stock on its way
    too, which counts as on hand.
    """
    amounts: dict[SkuId, int] = {}
    for sku_id in x_prev.skus:
        ordered = round(unit_estimate.get(sku_id, 0.0) * epoch_length)
        target = max(floor, math.ceil((1 + margin) * ordered))
        on_hand = sum(x_prev.counts.get(sku_id, {}).values())
        amounts[sku_id] = max(0, target + waiting_by_sku.get(sku_id, 0) - on_hand)
    return amounts


def receiving_distance(graph: WarehouseGraph, vertex: VertexId, receiving: list[VertexId]) -> float:
    """min_{g in V_rcv} d_G(g, v): how far a unit travels from the nearest receiving vertex to
    the storage vertex v."""
    return min(graph.distance(g, vertex) for g in receiving)


def nearest_receiving_point(
    graph: WarehouseGraph, vertex: VertexId, receiving: list[VertexId]
) -> VertexId:
    """argmin_{g in V_rcv} d_G(g, v), ties by vertex id: where a put-away to v picks up."""
    return min(receiving, key=lambda g: (graph.distance(g, vertex), g))


def trip_distance(
    graph: WarehouseGraph, vertex: VertexId, receiving: list[VertexId], delivery: list[VertexId]
) -> float:
    """d^trip(v), the trip distance: from the nearest receiving vertex to v, then on to the
    nearest delivery vertex. Both legs are in timesteps, so they add with no weight."""
    return receiving_distance(graph, vertex, receiving) + min(
        graph.distance(vertex, g) for g in delivery
    )


def ranked_by_trip(graph: WarehouseGraph, receiving: list[VertexId]) -> list[VertexId]:
    """V_str sorted by d^trip ascending, ties by vertex id."""
    delivery = delivery_vertices(graph)
    return sorted(
        storage_vertices(graph), key=lambda v: (trip_distance(graph, v, receiving, delivery), v)
    )


def trip_exposure(
    graph: WarehouseGraph,
    vertex: VertexId,
    receiving: list[VertexId],
    delivery: list[VertexId],
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
) -> float:
    """sum mu_hat_t(e) * w_hat_t(e) over the edges of the trip through v, the shortest path
    from the nearest receiving vertex to v and on to the nearest delivery vertex."""
    start = nearest_receiving_point(graph, vertex, receiving)
    end = nearest_delivery_point(graph, vertex, delivery)
    path = _shortest_path_edges(graph, start, vertex) + _shortest_path_edges(graph, vertex, end)
    return sum(traversal_estimate.get(e, 0.0) * waiting_estimate.get(e, 0.0) for e in path)


def _put_away(
    x_prev: StorageState,
    amounts: dict[SkuId, int],
    sku_order: list[SkuId],
    shelf_order: Callable[[SkuId], list[VertexId]],
    home: dict[SkuId, VertexId] | None = None,
) -> PutAwayResult:
    """The fill loop every rule shares. Each SKU in `sku_order` places its arriving units on
    its shelves in `shelf_order(k)`, as many on each as fit in the free capacity left. `home`
    is given by "fixed" only, to count the overflow. Stock already on a shelf stays."""
    free = {v: cap - x_prev.used_capacity(v) for v, cap in x_prev.capacities.items()}
    counts = {sku_id: dict(per_vertex) for sku_id, per_vertex in x_prev.counts.items()}
    unplaced = 0
    overflow = 0
    for sku_id in sku_order:
        units = amounts.get(sku_id, 0)
        if units <= 0:
            continue
        unit_capacity = x_prev.skus[sku_id].unit_capacity
        for vertex in shelf_order(sku_id):
            fit = min(units, int(free[vertex] // unit_capacity + 1e-9))
            if fit <= 0:
                continue
            per_vertex = counts.setdefault(sku_id, {})
            per_vertex[vertex] = per_vertex.get(vertex, 0) + fit
            free[vertex] -= fit * unit_capacity
            if home is not None and vertex != home[sku_id]:
                overflow += fit
            units -= fit
            if units == 0:
                break
        unplaced += units
    return PutAwayResult(
        storage=StorageState(skus=x_prev.skus, capacities=x_prev.capacities, counts=counts),
        arrived_units=sum(amounts.values()),
        unplaced_units=unplaced,
        overflow_units=overflow,
    )


def _shelf_ranking(
    graph: WarehouseGraph, receiving: list[VertexId] | None, trip: bool
) -> list[VertexId]:
    """The shelves of "fixed" and "demand": by trip distance under robot put-away, by access
    distance otherwise."""
    if not trip:
        return ranked_storage_vertices(graph)
    return ranked_by_trip(graph, delivery_vertices(graph) if receiving is None else receiving)


def _by_demand(x_prev: StorageState, demand_estimate: DemandEstimate) -> list[SkuId]:
    """SKUs by rho_hat_t(k) descending, ties by id, the order of F_dem and F_cng."""
    return sorted(x_prev.skus, key=lambda k: (-demand_estimate.get(k, 0.0), k))


@register_putaway_rule("fixed")
def putaway_fixed(
    x_prev: StorageState,
    graph: WarehouseGraph,
    amounts: dict[SkuId, int],
    home: dict[SkuId, VertexId],
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
    congestion_weight: float | None,
    *,
    receiving: list[VertexId] | None = None,
    trip: bool = False,
) -> PutAwayResult:
    """Dedicated storage: the home shelf first, then the nearest shelves with room."""
    by_access = _shelf_ranking(graph, receiving, trip)

    def shelf_order(sku_id: SkuId) -> list[VertexId]:
        if sku_id not in home:
            raise ValueError(f"SKU {sku_id!r} has stock arriving and no home shelf.")
        return [home[sku_id]] + [v for v in by_access if v != home[sku_id]]

    return _put_away(x_prev, amounts, sorted(x_prev.skus), shelf_order, home)


@register_putaway_rule("demand")
def putaway_demand(
    x_prev: StorageState,
    graph: WarehouseGraph,
    amounts: dict[SkuId, int],
    home: dict[SkuId, VertexId],
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
    congestion_weight: float | None,
    *,
    receiving: list[VertexId] | None = None,
    trip: bool = False,
) -> PutAwayResult:
    """The most demanded SKUs first, each into the nearest shelves with room."""
    by_access = _shelf_ranking(graph, receiving, trip)
    return _put_away(x_prev, amounts, _by_demand(x_prev, demand_estimate), lambda _sku: by_access)


@register_putaway_rule("congestion")
def putaway_congestion(
    x_prev: StorageState,
    graph: WarehouseGraph,
    amounts: dict[SkuId, int],
    home: dict[SkuId, VertexId],
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
    congestion_weight: float | None,
    *,
    receiving: list[VertexId] | None = None,
    trip: bool = False,
) -> PutAwayResult:
    """The most demanded SKUs first, each into its shelves by the congestion-aware score."""
    if congestion_weight is None:
        raise ValueError(
            "the congestion put-away rule requires congestion_weight (beta); "
            "ExperimentConfig already enforces this, so this indicates a caller bug."
        )
    storage = storage_vertices(graph)
    delivery = delivery_vertices(graph)

    if trip:
        # The score rho_hat_t(k) * d^trip(v) + beta * the traffic over the trip. The trip of a
        # shelf does not depend on the SKU, so both parts are worked out once per shelf.
        trip_receiving = delivery if receiving is None else receiving
        weight = congestion_weight
        distance = {v: trip_distance(graph, v, trip_receiving, delivery) for v in storage}
        exposure = {
            v: trip_exposure(
                graph, v, trip_receiving, delivery, traversal_estimate, waiting_estimate
            )
            for v in storage
        }

        def trip_order(sku_id: SkuId) -> list[VertexId]:
            rho = demand_estimate.get(sku_id, 0.0)
            return sorted(storage, key=lambda v: (rho * distance[v] + weight * exposure[v], v))

        return _put_away(x_prev, amounts, _by_demand(x_prev, demand_estimate), trip_order)

    def shelf_order(sku_id: SkuId) -> list[VertexId]:
        return ranked_storage_vertices_for(
            graph,
            sku_id,
            storage,
            delivery,
            demand_estimate,
            traversal_estimate,
            waiting_estimate,
            congestion_weight,
        )

    return _put_away(x_prev, amounts, _by_demand(x_prev, demand_estimate), shelf_order)
