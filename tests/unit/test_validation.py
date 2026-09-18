"""Unit tests for slap_mapd_coupling.instances.validation."""

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness


def test_check_connectivity_passes_on_fully_reachable_instance():
    vertices = {
        1: Vertex(id=1),
        2: Vertex(id=2, role=VertexRole(movable=True, storage=True)),
    }
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    report = check_connectivity(graph)
    assert report.ok is True
    assert report.unreachable_pairs == ()


def test_check_connectivity_fails_on_constructed_unreachable_instance():
    vertices = {
        1: Vertex(id=1),
        2: Vertex(id=2),
        3: Vertex(id=3, role=VertexRole(movable=True, storage=True)),  # isolated: no incoming edge
    }
    edges = (Edge(source=1, target=2, cost=1.0),)
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    report = check_connectivity(graph)
    assert report.ok is False
    assert any(pair[1] == 3 for pair in report.unreachable_pairs)


def _line_graph(ids: list[int]) -> WarehouseGraph:
    vertices = {vid: Vertex(id=vid, role=VertexRole(movable=True, endpoint=True)) for vid in ids}
    edges = []
    for i in range(1, len(ids)):
        edges.append(Edge(source=ids[i - 1], target=ids[i], cost=1.0))
        edges.append(Edge(source=ids[i], target=ids[i - 1], cost=1.0))
    return WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=0.0)


def test_well_formedness_fails_on_too_few_endpoints():
    graph = _line_graph([1])
    report = check_well_formedness(graph, fleet_size=2)
    assert report.ok is False
    assert report.endpoint_count_ok is False


def test_well_formedness_fails_on_blocking_third_endpoint():
    graph = _line_graph([1, 2, 3])  # e1=1, e2=2 (middle, blocks), e3=3
    report = check_well_formedness(graph, fleet_size=3)
    assert report.ok is False
    assert report.endpoint_count_ok is True
    assert (1, 3, 2) in report.blocked_pairs or (3, 1, 2) in report.blocked_pairs


def test_well_formedness_passes_on_figure_a_analogue():
    # A triangle: every pair of endpoints reaches the other directly,
    # without needing to pass through the third.
    vertices = {i: Vertex(id=i, role=VertexRole(movable=True, endpoint=True)) for i in (1, 2, 3)}
    edges = []
    for a, b in [(1, 2), (2, 3), (3, 1)]:
        edges.append(Edge(source=a, target=b, cost=1.0))
        edges.append(Edge(source=b, target=a, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=0.0)
    report = check_well_formedness(graph, fleet_size=2)
    assert report.ok is True
