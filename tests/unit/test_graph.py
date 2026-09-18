"""Unit tests for slap_mapd_coupling.core.graph."""

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.graph import Action, Edge, Vertex, VertexRole, WarehouseGraph


def test_vertex_role_is_not_exclusive():
    role = VertexRole(movable=True, storage=True, delivery=True)
    assert role.storage is True
    assert role.delivery is True


def test_vertex_role_non_movable_role_rejected():
    with pytest.raises(ValidationError):
        VertexRole(movable=False, storage=True)


def test_edge_rejects_self_loop():
    with pytest.raises(ValidationError, match="self-loop"):
        Edge(source=1, target=1, cost=1.0)


def test_edge_rejects_non_positive_cost():
    with pytest.raises(ValidationError):
        Edge(source=1, target=2, cost=0.0)
    with pytest.raises(ValidationError):
        Edge(source=1, target=2, cost=-1.0)


def test_warehouse_graph_rejects_edge_to_unknown_vertex():
    vertices = {1: Vertex(id=1), 2: Vertex(id=2)}
    edges = (Edge(source=1, target=3, cost=1.0),)
    with pytest.raises(ValidationError):
        WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_warehouse_graph_rejects_edge_touching_non_movable_vertex():
    vertices = {1: Vertex(id=1), 2: Vertex(id=2, role=VertexRole(movable=False))}
    edges = (Edge(source=1, target=2, cost=1.0),)
    with pytest.raises(ValidationError):
        WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _diamond_graph() -> WarehouseGraph:
    # 1 -> 2 -> 4 (cost 1+1=2) is shorter than 1 -> 3 -> 4 (cost 5+1=6).
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 4)}
    edges = (
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=4, cost=1.0),
        Edge(source=1, target=3, cost=5.0),
        Edge(source=3, target=4, cost=1.0),
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_distance_matches_hand_computed_shortest_path():
    graph = _diamond_graph()
    assert graph.distance(1, 4) == pytest.approx(2.0)


def test_distance_unreachable_raises_keyerror():
    vertices = {1: Vertex(id=1), 2: Vertex(id=2), 5: Vertex(id=5)}
    edges = (Edge(source=1, target=2, cost=1.0),)
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    with pytest.raises(KeyError):
        graph.distance(1, 5)


def test_legal_actions_include_wait_and_all_successors():
    vertices = {1: Vertex(id=1), 2: Vertex(id=2), 3: Vertex(id=3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=1, target=3, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    assert set(graph.legal_actions(1)) == {
        Action(kind="wait"),
        Action(kind="move", target=2),
        Action(kind="move", target=3),
    }


def test_action_wait_rejects_target():
    with pytest.raises(ValidationError):
        Action(kind="wait", target=5)
