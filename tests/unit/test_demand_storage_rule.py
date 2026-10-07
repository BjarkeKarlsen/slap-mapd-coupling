"""Unit tests for slap_mapd_coupling.storage.demand (F_dem)."""

import pytest

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.storage.demand import (
    access_distance,
    f_dem,
    greedy_fill,
    ranked_storage_vertices,
)


def _line_graph() -> WarehouseGraph:
    # 0 (delivery) - 1 - 2 - 3, all storage vertices except 0.
    vertices = {
        0: Vertex(id=0, role=VertexRole(movable=True, delivery=True)),
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, storage=True)),
        3: Vertex(id=3, role=VertexRole(movable=True, storage=True)),
    }
    edges = tuple(
        Edge(source=a, target=b, cost=1.0) for i in range(3) for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _skus() -> dict[str, SkuType]:
    return {
        "hot": SkuType(sku_id="hot", unit_capacity=1.0),
        "cold": SkuType(sku_id="cold", unit_capacity=1.0),
    }


def _state(counts: dict[str, dict[int, int]], capacities: dict[int, float]) -> StorageState:
    return StorageState(skus=_skus(), capacities=capacities, counts=counts)


def test_access_distance_matches_hop_distance_on_a_line():
    graph = _line_graph()
    assert access_distance(graph, 1, [0]) == 1.0
    assert access_distance(graph, 3, [0]) == 3.0


def test_ranked_storage_vertices_ascending_by_access_distance():
    graph = _line_graph()
    assert ranked_storage_vertices(graph) == [1, 2, 3]


def test_greedy_fill_prefers_closer_vertices_for_higher_ranked_sku():
    graph = _line_graph()
    capacities = {1: 2.0, 2: 2.0, 3: 2.0}
    x_prev = _state({"hot": {3: 2}, "cold": {1: 2}}, capacities)
    storage_order = ranked_storage_vertices(graph)
    vertex_order_by_sku = {"hot": storage_order, "cold": storage_order}

    x_target, priority = greedy_fill(x_prev, vertex_order_by_sku, sku_order=["hot", "cold"])

    assert x_target.units("hot", 1) == 2  # hot (placed first) gets the closest vertex
    assert x_target.units("cold", 1) == 0
    assert x_target.units("cold", 2) == 2  # cold gets the next-closest remaining vertex
    assert ("hot", 1) in priority
    assert ("cold", 2) in priority
    assert ("hot", 3) not in priority  # hot's x_prev cell isn't a NEW arrival anywhere


def test_greedy_fill_skips_a_full_vertex():
    graph = _line_graph()
    capacities = {1: 1.0, 2: 2.0, 3: 2.0}
    x_prev = _state({"hot": {1: 1}, "cold": {2: 1}}, capacities)
    storage_order = ranked_storage_vertices(graph)
    vertex_order_by_sku = {"hot": storage_order, "cold": storage_order}

    x_target, _ = greedy_fill(x_prev, vertex_order_by_sku, sku_order=["hot", "cold"])

    assert x_target.units("hot", 1) == 1  # fills vertex 1 to capacity
    assert x_target.units("cold", 1) == 0  # no room left at 1, skipped
    assert x_target.units("cold", 2) == 1  # falls through to vertex 2


def test_f_dem_raises_without_reassignment_cap():
    graph = _line_graph()
    x_prev = _state({"hot": {3: 2}}, {1: 2.0, 2: 2.0, 3: 2.0})
    with pytest.raises(ValueError, match="reassignment_cap"):
        f_dem(x_prev, graph, None, None, {"hot": 10.0}, {}, {})


def test_f_dem_respects_the_relocation_cap():
    graph = _line_graph()
    capacities = {1: 5.0, 2: 5.0, 3: 5.0}
    x_prev = _state({"hot": {3: 5}}, capacities)

    result = f_dem(
        x_prev,
        graph,
        reassignment_cap=2,
        congestion_weight=None,
        demand_estimate={"hot": 10.0},
        traversal_estimate={},
        waiting_estimate={},
    )

    # Greedy target wants all 5 units at vertex 1; only 2 can relocate.
    assert result.units("hot", 1) == 2
    assert result.units("hot", 3) == 3
    assert result.units("hot", 1) + result.units("hot", 3) == 5


def test_f_dem_registered_under_demand():
    from slap_mapd_coupling.storage.registry import get_storage_rule

    graph = _line_graph()
    x_prev = _state({"hot": {3: 2}}, {1: 2.0, 2: 2.0, 3: 2.0})
    rule = get_storage_rule("demand")
    result = rule(x_prev, graph, 10, None, {"hot": 5.0}, {}, {})
    assert result.units("hot", 1) == 2
