"""Unit tests for slap_mapd_coupling.storage.congestion (F_cng)."""

import pytest

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.storage.congestion import (
    congestion_exposure,
    f_cng,
    nearest_delivery_point,
    score,
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
    return {"hot": SkuType(sku_id="hot", unit_capacity=1.0)}


def _state(counts: dict[str, dict[int, int]], capacities: dict[int, float]) -> StorageState:
    return StorageState(skus=_skus(), capacities=capacities, counts=counts)


def test_nearest_delivery_point_single_candidate():
    graph = _line_graph()
    assert nearest_delivery_point(graph, 3, [0]) == 0


def test_congestion_exposure_zero_with_no_traffic_data():
    graph = _line_graph()
    assert congestion_exposure(graph, 3, 0, {}, {}) == 0.0


def test_congestion_exposure_sums_along_the_path():
    graph = _line_graph()
    # path(0 -> 3) = (0,1), (1,2), (2,3)
    traversal = {(0, 1): 2.0, (1, 2): 3.0, (2, 3): 1.0}
    waiting = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    exposure = congestion_exposure(graph, 3, 0, traversal, waiting)
    assert exposure == pytest.approx(2.0 + 3.0 + 1.0)


def test_score_increases_with_congestion_weight():
    graph = _line_graph()
    traversal = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    waiting = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    low_beta = score(graph, "hot", 3, [0], {"hot": 1.0}, traversal, waiting, congestion_weight=0.0)
    high_beta = score(graph, "hot", 3, [0], {"hot": 1.0}, traversal, waiting, congestion_weight=5.0)
    assert high_beta > low_beta


def test_f_cng_raises_without_congestion_weight():
    graph = _line_graph()
    x_prev = _state({"hot": {3: 2}}, {1: 2.0, 2: 2.0, 3: 2.0})
    with pytest.raises(ValueError, match="congestion_weight"):
        f_cng(
            x_prev,
            graph,
            reassignment_cap=10,
            congestion_weight=None,
            demand_estimate={},
            traversal_estimate={},
            waiting_estimate={},
        )


def test_f_cng_raises_without_reassignment_cap():
    graph = _line_graph()
    x_prev = _state({"hot": {3: 2}}, {1: 2.0, 2: 2.0, 3: 2.0})
    with pytest.raises(ValueError, match="reassignment_cap"):
        f_cng(
            x_prev,
            graph,
            reassignment_cap=None,
            congestion_weight=0.5,
            demand_estimate={},
            traversal_estimate={},
            waiting_estimate={},
        )


def _branching_graph() -> WarehouseGraph:
    # Delivery point 0 with two one-hop storage vertices on separate
    # branches, so their paths from 0 don't overlap: (0,1) and (0,2).
    vertices = {
        0: Vertex(id=0, role=VertexRole(movable=True, delivery=True)),
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, storage=True)),
    }
    edges = (
        Edge(source=0, target=1, cost=1.0),
        Edge(source=1, target=0, cost=1.0),
        Edge(source=0, target=2, cost=1.0),
        Edge(source=2, target=0, cost=1.0),
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_f_cng_prefers_the_less_congested_branch_when_distances_tie():
    graph = _branching_graph()
    capacities = {1: 5.0, 2: 5.0}
    x_prev = StorageState(skus=_skus(), capacities=capacities, counts={"hot": {1: 3}})
    # (0,1) is heavily congested, (0,2) is clean; both vertices are
    # equidistant (1 hop) from the delivery point, so pure access
    # distance can't distinguish them -- only the congestion term can.
    traversal = {(0, 1): 100.0}
    waiting = {(0, 1): 100.0}

    result = f_cng(
        x_prev,
        graph,
        reassignment_cap=10,
        congestion_weight=10.0,
        demand_estimate={"hot": 1.0},
        traversal_estimate=traversal,
        waiting_estimate=waiting,
    )
    assert result.units("hot", 2) == 3  # relocated to the uncongested branch
    assert result.units("hot", 1) == 0


def test_f_cng_ties_broken_by_vertex_id_with_zero_congestion_weight():
    graph = _branching_graph()
    capacities = {1: 5.0, 2: 5.0}
    x_prev = StorageState(skus=_skus(), capacities=capacities, counts={"hot": {2: 3}})
    traversal = {(0, 1): 100.0}
    waiting = {(0, 1): 100.0}

    result = f_cng(
        x_prev,
        graph,
        reassignment_cap=10,
        congestion_weight=0.0,  # congestion term is switched off entirely
        demand_estimate={"hot": 1.0},
        traversal_estimate=traversal,
        waiting_estimate=waiting,
    )
    assert result.units("hot", 1) == 3  # pure distance tie -> lower vertex id wins
    assert result.units("hot", 2) == 0


def test_f_cng_registered_under_congestion():
    from slap_mapd_coupling.storage.registry import get_storage_rule

    graph = _line_graph()
    x_prev = _state({"hot": {3: 2}}, {1: 2.0, 2: 2.0, 3: 2.0})
    rule = get_storage_rule("congestion")
    result = rule(x_prev, graph, 10, 0.5, {"hot": 5.0}, {}, {})
    assert result.units("hot", 1) == 2
