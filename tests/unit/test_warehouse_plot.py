"""Unit tests for slap_mapd_coupling.viz.warehouse_plot."""

from matplotlib.axes import Axes

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.viz.warehouse_plot import (
    assign_delivery_endpoint_names,
    assign_storage_letters,
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


def _two_storage_graph() -> WarehouseGraph:
    vertices = {
        3: Vertex(id=3, role=VertexRole(movable=True, storage=True)),
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True)),
    }
    return WarehouseGraph(vertices=vertices, edges=(), wait_cost=0.0)


def test_assign_storage_letters_names_storage_vertices_in_id_order():
    graph = _two_storage_graph()

    letters = assign_storage_letters(graph)

    assert letters == {1: "A", 3: "B"}


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


def _delivery_endpoint_graph() -> WarehouseGraph:
    vertices = {
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, delivery=True, endpoint=True)),
        3: Vertex(id=3, role=VertexRole(movable=True, endpoint=True)),
        4: Vertex(id=4, role=VertexRole(movable=True, delivery=True)),
        5: Vertex(id=5, role=VertexRole(movable=True)),
    }
    return WarehouseGraph(vertices=vertices, edges=(), wait_cost=0.0)


def test_assign_delivery_endpoint_names_skips_storage_and_plain_vertices():
    graph = _delivery_endpoint_graph()

    names = assign_delivery_endpoint_names(graph)

    # vertex 1 is storage (assign_storage_letters' job, not this one's),
    # vertex 5 is plain -- neither gets a name here. Vertex 2 has both
    # delivery and endpoint set and is named D1, matching delivery's
    # priority over endpoint in theme.ROLE_PRIORITY.
    assert names == {2: "D1", 3: "E1", 4: "D2"}


def test_plot_node_names_overlay_does_not_raise():
    graph = _delivery_endpoint_graph()
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0), 4: (3.0, 0.0), 5: (4.0, 0.0)}
    ax = plot_graph(graph, positions)
    names = assign_delivery_endpoint_names(graph)

    result = plot_node_names(ax, graph, positions, names)

    assert result is ax
