"""Put-away rules of the arrival stock model (thesis wording pending).

Under the arrival stock model (config.stock_model == "arrival") an order line takes its units
by the pickup rule when it is released, so picks empty the shelves. Every Delta steps new stock arrives at the
receiving vertices, and the storage update becomes the put-away of that stock: the storage rule
decides which shelves the arriving units go to. Nothing already on a shelf moves, so a SKU can
sit on several shelves at once, old stock on one and new stock on another.

How much arrives does not depend on the rule. Under the top-up rule (`arrival_amounts`) each
SKU is filled up to max(floor, ceil((1 + epsilon) * its units ordered in the last period)), plus
the units lines still wait for, less what is on hand. Under the refill rule (`refill_amounts`)
stock is replaced a refill at a time: a SKU gets one refill of Q units for every Q units ordered
since its last refill, and the remainder carries over. Where it goes is the rule's choice, through one shared fill loop
(`_put_away`) that only differs in the order the SKUs are placed in and each SKU's order of
shelves:

- "fixed" (dedicated storage, F_fix). SKUs by id. A SKU's units go to its home shelf, and once
  that is full to the other storage vertices by access distance min_{g in V_del} d_G(v, g),
  ties by vertex id. Units placed off the home shelf are overflow.
- "demand" (F_dem). SKUs by rho_hat_t(k) descending, ties by id, each into the storage vertices
  by access distance, ties by id, the same order for every SKU.
- "congestion" (F_cng). SKUs as in "demand", each into its own order of storage vertices by the
  congestion-aware score with weight beta (storage/congestion.py).
- "random". SKUs by id, each into the storage vertices in a uniformly random order drawn
  afresh for every SKU, so each shelf it fills is drawn uniformly among the shelves with room.
  It uses no demand and no distance, only the room left, and is a check on "demand": whether its
  gain comes from where it puts stock or only from spreading it. The draws come from `rng`, a
  stream the environment seeds from the run seed.

A shelf takes as many units as fit in its free capacity. Units that fit nowhere are not placed
and are counted as unplaced. Under the refill rule a refill is placed whole, on the first shelf
in the order with room for all Q of its units, and is never split over shelves. A refill that
fits on no shelf is unplaced.

When robots carry the put-away (config.stock_putaway == "robots") every rule judges a shelf by
the trip distance d^trip(v) = min_{g in V_rcv} d_G(g, v) + min_{g in V_del} d_G(v, g), the
trip of a unit through it (sec:method:storage). "fixed" sends overflow and "demand" sends every
SKU to the shelves in order of d^trip, ties by vertex id, and the congestion-aware score of
"congestion" is rho_hat_t(k) * d^trip(v) plus beta times the traffic over both legs of the
trip. With instant put-away the rules are as before.

Under the refill rule with robot put-away the two legs of the trip no longer happen equally
often: a SKU is picked about rho_hat_t(k) times per step but put away only rho_hat^unit_t(k) / Q
times. So every rule weighs each leg by how often it happens, and SKU k on shelf v scores
rho_hat_t(k) * d^pick(v) + (rho_hat^unit_t(k) / Q) * d^put(v) (`leg_weighted_distance`). d^pick(v)
is the mean distance from v to the delivery vertices, since an order picks its delivery vertex
uniformly, and d^put(v) the distance from the nearest receiving vertex. "fixed" orders its
overflow and "demand" its shelves by this score, each SKU on its own, and "congestion" adds
beta times the traffic over the trip. A SKU with no orders in the window scores 0 everywhere,
so ties go by d^trip, then vertex id.

Choices flagged, not fixed by the thesis yet:
- The units of SKU k ordered in the last period are counted as round(rho_hat^unit_t(k) * Delta),
  the exact count, since the estimate is per step. Taking the ceiling of the float product
  (1 + epsilon) * rho_hat^unit_t(k) * Delta directly could round an exact count up by one unit.
- The number of units of SKU k a shelf v takes is int(free(v) // b_k + 1e-9), the tolerance
  keeping a floating-point remainder from losing a unit that fits.
- Under the refill rule the pick leg d^pick(v) is the mean distance to the delivery vertices,
  not the nearest one that d^trip uses, since an order picks its delivery vertex uniformly.
  The congestion term still follows the trip through the nearest delivery vertex.
"""

from __future__ import annotations

import math
import random
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
    distance (robots carry the put-away), False by the access distance as before. `rng` is the
    run's put-away stream, which only "random" uses. Every rule takes it, so the environment
    can pass it to any of them. `refill_units` is Q under the refill rule, None under the top-up
    rule, and `unit_estimate` is rho_hat^unit, which the refill rule weighs the put-away leg by.
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
        rng: random.Random | None = None,
        refill_units: int | None = None,
        unit_estimate: DemandEstimate | None = None,
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


def refill_amounts(
    ordered: dict[SkuId, int], refill_units: int
) -> tuple[dict[SkuId, int], dict[SkuId, int]]:
    """The arrivals under the refill rule, and what carries over. `ordered` holds each SKU's
    units ordered since its last refill. SKU k gets floor(ordered / Q) refills of Q units, so
    its arriving units are a multiple of Q, and the remainder counts towards the next refill."""
    amounts = {sku_id: (count // refill_units) * refill_units for sku_id, count in ordered.items()}
    left = {sku_id: count % refill_units for sku_id, count in ordered.items()}
    return amounts, left


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


def pick_distance(graph: WarehouseGraph, vertex: VertexId, delivery: list[VertexId]) -> float:
    """d^pick(v), the mean d_G(v, g) over the delivery vertices g, since an order picks its
    delivery vertex uniformly. The same distance the starting placement orders shelves by."""
    return sum(graph.distance(vertex, g) for g in delivery) / len(delivery)


def leg_weighted_distance(
    picks: float, refills: float, pick_leg: float, putaway_leg: float
) -> float:
    """The distance of a SKU on a shelf with each leg weighed by how often it happens: `picks`
    per step on the pick leg d^pick(v), `refills` per step on the put-away leg d^put(v)."""
    return picks * pick_leg + refills * putaway_leg


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
    whole: int | None = None,
) -> PutAwayResult:
    """The fill loop every rule shares. Each SKU in `sku_order` places its arriving units on
    its shelves in `shelf_order(k)`, as many on each as fit in the free capacity left. `home`
    is given by "fixed" only, to count the overflow. Stock already on a shelf stays. With
    `whole`, the refill size Q, the units come in refills of Q and each refill goes whole to the
    first shelf in the order with room for it."""
    free = {v: cap - x_prev.used_capacity(v) for v, cap in x_prev.capacities.items()}
    counts = {sku_id: dict(per_vertex) for sku_id, per_vertex in x_prev.counts.items()}
    unplaced = 0
    overflow = 0
    for sku_id in sku_order:
        units = amounts.get(sku_id, 0)
        if units <= 0:
            continue
        unit_capacity = x_prev.skus[sku_id].unit_capacity
        if whole is not None:
            order = shelf_order(sku_id)
            for _ in range(units // whole):
                vertex = next(
                    (v for v in order if int(free[v] // unit_capacity + 1e-9) >= whole), None
                )
                if vertex is None:
                    unplaced += whole
                    continue
                per_vertex = counts.setdefault(sku_id, {})
                per_vertex[vertex] = per_vertex.get(vertex, 0) + whole
                free[vertex] -= whole * unit_capacity
                if home is not None and vertex != home[sku_id]:
                    overflow += whole
            unplaced += units % whole
            continue
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


def _refill_scores(
    graph: WarehouseGraph,
    receiving: list[VertexId] | None,
    demand_estimate: DemandEstimate,
    unit_estimate: DemandEstimate,
    refill_units: int,
    extra: dict[VertexId, float] | None = None,
) -> Callable[[SkuId], list[VertexId]]:
    """Each SKU's shelves under the refill rule with robot put-away, by the leg-weighted
    distance plus `extra` (the congestion term), ties by d^trip, then vertex id."""
    storage = storage_vertices(graph)
    delivery = delivery_vertices(graph)
    receiving = delivery if receiving is None else receiving
    pick_leg = {v: pick_distance(graph, v, delivery) for v in storage}
    putaway_leg = {v: receiving_distance(graph, v, receiving) for v in storage}
    trip = {v: trip_distance(graph, v, receiving, delivery) for v in storage}
    added = extra or {}

    def order(sku_id: SkuId) -> list[VertexId]:
        picks = demand_estimate.get(sku_id, 0.0)
        refills = unit_estimate.get(sku_id, 0.0) / refill_units
        return sorted(
            storage,
            key=lambda v: (
                leg_weighted_distance(picks, refills, pick_leg[v], putaway_leg[v])
                + added.get(v, 0.0),
                trip[v],
                v,
            ),
        )

    return order


def _refill_order(
    graph: WarehouseGraph,
    receiving: list[VertexId] | None,
    trip: bool,
    demand_estimate: DemandEstimate,
    unit_estimate: DemandEstimate | None,
    refill_units: int | None,
) -> Callable[[SkuId], list[VertexId]] | None:
    """The per-SKU shelf order of "fixed" and "demand" under the refill rule with robot
    put-away, or None when the shelves keep one order for every SKU."""
    if refill_units is None or not trip:
        return None
    return _refill_scores(graph, receiving, demand_estimate, unit_estimate or {}, refill_units)


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
    rng: random.Random | None = None,
    refill_units: int | None = None,
    unit_estimate: DemandEstimate | None = None,
) -> PutAwayResult:
    """Dedicated storage: the home shelf first, then the nearest shelves with room."""
    refill_order = _refill_order(
        graph, receiving, trip, demand_estimate, unit_estimate, refill_units
    )
    by_access = _shelf_ranking(graph, receiving, trip) if refill_order is None else []

    def shelf_order(sku_id: SkuId) -> list[VertexId]:
        if sku_id not in home:
            raise ValueError(f"SKU {sku_id!r} has stock arriving and no home shelf.")
        rest = by_access if refill_order is None else refill_order(sku_id)
        return [home[sku_id]] + [v for v in rest if v != home[sku_id]]

    return _put_away(x_prev, amounts, sorted(x_prev.skus), shelf_order, home, refill_units)


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
    rng: random.Random | None = None,
    refill_units: int | None = None,
    unit_estimate: DemandEstimate | None = None,
) -> PutAwayResult:
    """The most demanded SKUs first, each into the nearest shelves with room."""
    refill_order = _refill_order(
        graph, receiving, trip, demand_estimate, unit_estimate, refill_units
    )
    if refill_order is None:
        by_access = _shelf_ranking(graph, receiving, trip)
        refill_order = lambda _sku: by_access  # noqa: E731
    return _put_away(
        x_prev,
        amounts,
        _by_demand(x_prev, demand_estimate),
        refill_order,
        whole=refill_units,
    )


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
    rng: random.Random | None = None,
    refill_units: int | None = None,
    unit_estimate: DemandEstimate | None = None,
) -> PutAwayResult:
    """The most demanded SKUs first, each into its shelves by the congestion-aware score."""
    if congestion_weight is None:
        raise ValueError(
            "the congestion put-away rule requires congestion_weight (beta); "
            "ExperimentConfig already enforces this, so this indicates a caller bug."
        )
    storage = storage_vertices(graph)
    delivery = delivery_vertices(graph)

    if trip and refill_units is not None:
        # The leg-weighted distance plus beta * the traffic over the trip.
        trip_receiving = delivery if receiving is None else receiving
        exposure = {
            v: congestion_weight
            * trip_exposure(
                graph, v, trip_receiving, delivery, traversal_estimate, waiting_estimate
            )
            for v in storage
        }
        order = _refill_scores(
            graph, receiving, demand_estimate, unit_estimate or {}, refill_units, exposure
        )
        return _put_away(
            x_prev, amounts, _by_demand(x_prev, demand_estimate), order, whole=refill_units
        )

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

    return _put_away(
        x_prev, amounts, _by_demand(x_prev, demand_estimate), shelf_order, whole=refill_units
    )


@register_putaway_rule("random")
def putaway_random(
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
    rng: random.Random | None = None,
    refill_units: int | None = None,
    unit_estimate: DemandEstimate | None = None,
) -> PutAwayResult:
    """SKUs by id, each into the shelves in a random order: the first shelf of a uniformly
    random order that has room is uniform among the shelves with room."""
    if rng is None:
        raise ValueError(
            "the random put-away rule requires rng, the run's put-away stream, which the "
            "environment passes; a caller that does not indicates a bug."
        )
    storage = storage_vertices(graph)
    orders = {sku_id: rng.sample(storage, len(storage)) for sku_id in sorted(x_prev.skus)}
    return _put_away(
        x_prev, amounts, sorted(x_prev.skus), lambda sku_id: orders[sku_id], whole=refill_units
    )
