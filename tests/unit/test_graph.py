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


def test_within_is_the_field_of_view_worked_out_once():
    """within(v, depth) is every movable vertex within depth of v, in vertex order, and a second
    call returns the kept tuple."""
    from pathlib import Path

    from slap_mapd_coupling.training.spec import build_instance, load_spec

    root = Path(__file__).resolve().parents[2]
    graph = build_instance(load_spec(root / "configs" / "fleet_8" / "decentralised.yaml")).graph
    for depth in (0.0, 2.0, 5.0):
        for v, vertex in graph.vertices.items():
            if not vertex.role.movable:
                continue
            expected = tuple(
                w
                for w, item in graph.vertices.items()
                if item.role.movable and graph.distance(v, w) <= depth
            )
            assert graph.within(v, depth) == expected
            assert graph.within(v, depth) is graph.within(v, depth)


def test_distance_table_holds_the_distances_and_window_rows_are_kept():
    """Every entry of distance_table() is distance(v, w), NaN where distance raises, and
    window_rows gives each vertex's table row and window position, kept per window."""
    import math
    from pathlib import Path

    from slap_mapd_coupling.training.spec import build_instance, load_spec

    root = Path(__file__).resolve().parents[2]
    graph = build_instance(load_spec(root / "configs" / "fleet_8" / "decentralised.yaml")).graph
    table, row = graph.distance_table(), graph.vertex_rows()
    for v in graph.vertices:
        for w in graph.vertices:
            try:
                assert table[row[v], row[w]] == graph.distance(v, w)
            except KeyError:
                assert math.isnan(table[row[v], row[w]])
    window = graph.within(next(iter(row)), 3.0)
    rows, index = graph.window_rows(window)
    assert rows.tolist() == [row[v] for v in window]
    assert index == {v: i for i, v in enumerate(window)}
    assert graph.window_rows(window)[0] is rows


def _python_dijkstra(graph: WarehouseGraph) -> dict[tuple[int, int], float]:
    """The all-pairs distances as the graph computed them before scipy: Dijkstra in Python from
    every movable vertex, one entry per reachable pair."""
    import heapq

    distances: dict[tuple[int, int], float] = {}
    out: dict[int, list[Edge]] = {v: [] for v in graph.vertices}
    for edge in graph.edges:
        out[edge.source].append(edge)
    for source, vertex in graph.vertices.items():
        if not vertex.role.movable:
            continue
        best = {source: 0.0}
        frontier = [(0.0, source)]
        while frontier:
            d, current = heapq.heappop(frontier)
            if d > best.get(current, float("inf")):
                continue
            for edge in out[current]:
                candidate = d + edge.cost
                if candidate < best.get(edge.target, float("inf")):
                    best[edge.target] = candidate
                    heapq.heappush(frontier, (candidate, edge.target))
        distances.update({(source, target): value for target, value in best.items()})
    return distances


@pytest.mark.parametrize("fleet", [8, 16, 32, 64])
def test_the_distance_table_equals_dijkstra_in_python_on_the_fleet_warehouses(fleet):
    """The table scipy fills holds exactly the distances of the Python Dijkstra it replaced, and
    distance raises KeyError for every pair that has none, on the warehouse of every fleet."""
    import math
    from pathlib import Path

    from slap_mapd_coupling.training.spec import build_instance, load_spec

    root = Path(__file__).resolve().parents[2]
    config = root / "configs" / f"fleet_{fleet}" / "centralised.yaml"
    graph = build_instance(load_spec(config)).graph
    expected = _python_dijkstra(graph)
    table, row = graph.distance_table(), graph.vertex_rows()
    found = {
        (v, w): table[row[v], row[w]]
        for v in graph.vertices
        for w in graph.vertices
        if not math.isnan(table[row[v], row[w]])
    }
    assert found == expected
    for v, w in expected:
        assert graph.distance(v, w) == expected[(v, w)]


def test_distance_raises_for_a_pair_without_a_path_and_an_unknown_vertex():
    vertices = {1: Vertex(id=1), 2: Vertex(id=2), 3: Vertex(id=3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    assert graph.distance(1, 2) == 1.0
    assert type(graph.distance(1, 2)) is float
    with pytest.raises(KeyError):
        graph.distance(1, 3)
    with pytest.raises(KeyError):
        graph.distance(1, 99)
