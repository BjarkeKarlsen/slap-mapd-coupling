"""Unit tests for slap_mapd_coupling.viz.layout."""

from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.viz.layout import spring_layout


def test_spring_layout_covers_every_vertex():
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=3, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)

    positions = spring_layout(graph)

    assert set(positions.keys()) == set(graph.vertices.keys())
