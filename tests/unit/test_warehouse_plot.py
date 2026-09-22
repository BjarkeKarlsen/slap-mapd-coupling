"""Unit tests for slap_mapd_coupling.viz.warehouse_plot."""

from matplotlib.axes import Axes

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.viz.warehouse_plot import (
    assign_vertex_names,
    plot_agents,
    plot_graph,
    plot_node_names,
    plot_storage_capacity_list,
    plot_storage_contents,
    plot_storage_heatmap,
    plot_traffic,
)


def _small_graph() -> WarehouseGraph:
    vertices = {
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, delivery=True, endpoint=True)),
        3: Vertex(id=3),
    }
    edges = (
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=1, cost=1.0),
        Edge(source=2, target=3, cost=1.0, one_way=True),
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_plot_graph_returns_axes_and_creates_its_own_figure():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}

    ax = plot_graph(graph, positions)

    assert isinstance(ax, Axes)


def test_plot_graph_handles_a_vertex_with_a_secondary_role():
    # vertex 2 has both delivery and endpoint set -- exercises the
    # secondary-role outline-ring branch, not just the primary marker.
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}

    ax = plot_graph(graph, positions)

    assert isinstance(ax, Axes)


def test_plot_agents_overlay_does_not_raise():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    fleet = FleetState(agents={1: AgentState(agent_id=1, location=1)})

    result = plot_agents(ax, fleet, positions)

    assert result is ax


def test_plot_storage_heatmap_overlay_does_not_raise():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    storage = StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={1: 10.0},
        counts={"tea": {1: 4}},
    )

    result = plot_storage_heatmap(ax, graph, positions, storage, "tea")

    assert result is ax


def test_plot_storage_heatmap_with_zero_units_is_a_noop():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    storage = StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={1: 10.0},
        counts={},
    )

    result = plot_storage_heatmap(ax, graph, positions, storage, "tea")

    assert result is ax


def test_plot_traffic_overlay_does_not_raise():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)

    result = plot_traffic(ax, graph, positions, {(1, 2): 5, (2, 3): 0})

    assert result is ax


def test_plot_traffic_with_no_traversals_is_a_noop():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)

    result = plot_traffic(ax, graph, positions, {})

    assert result is ax


def _mixed_role_graph() -> WarehouseGraph:
    vertices = {
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, storage=True, delivery=True)),
        3: Vertex(id=3, role=VertexRole(movable=True, delivery=True, endpoint=True)),
        4: Vertex(id=4, role=VertexRole(movable=True, endpoint=True)),
        5: Vertex(id=5, role=VertexRole(movable=True, delivery=True)),
        6: Vertex(id=6, role=VertexRole(movable=True)),
    }
    return WarehouseGraph(vertices=vertices, edges=(), wait_cost=0.0)


def test_assign_vertex_names_covers_storage_delivery_endpoint_and_plain():
    graph = _mixed_role_graph()

    names = assign_vertex_names(graph)

    # 1: storage-only -> A. 2: storage+delivery -> still gets a storage
    # letter (B), not a D# name -- StorageState capacity is unconditional,
    # independent of theme.ROLE_PRIORITY's marker-shape priority (which
    # would draw vertex 2 with a delivery marker). 3: delivery+endpoint ->
    # D1 only (not also an E#). 4: endpoint-only -> E1. 5: delivery-only
    # -> D2. 6: plain -> no name at all.
    assert names == {1: "A", 2: "B", 3: "D1", 4: "E1", 5: "D2"}
    assert 6 not in names


def test_plot_storage_contents_overlay_does_not_raise():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    storage = StorageState(
        skus={
            "tea": SkuType(sku_id="tea", unit_capacity=1.0),
            "mugs": SkuType(sku_id="mugs", unit_capacity=1.0),
        },
        capacities={1: 10.0},
        counts={"tea": {1: 4}, "mugs": {1: 2}},
    )

    result = plot_storage_contents(ax, positions, storage, {1: "A"})

    assert result is ax


def test_plot_storage_contents_with_zero_units_only_draws_the_letter():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    storage = StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={1: 10.0},
        counts={},
    )

    result = plot_storage_contents(ax, positions, storage, {1: "A"})

    assert result is ax


def test_plot_storage_capacity_list_overlay_does_not_raise():
    graph = _small_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    ax = plot_graph(graph, positions)
    storage = StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={1: 10.0},
        counts={"tea": {1: 4}},
    )

    result = plot_storage_capacity_list(ax, storage, {1: "A"})

    assert result is ax


def test_plot_node_names_overlay_does_not_raise():
    graph = _mixed_role_graph()
    positions = {v: (float(v), 0.0) for v in graph.vertices}
    ax = plot_graph(graph, positions)
    names = assign_vertex_names(graph)
    non_storage_names = {v: n for v, n in names.items() if not graph.vertices[v].role.storage}

    result = plot_node_names(ax, graph, positions, non_storage_names)

    assert result is ax
