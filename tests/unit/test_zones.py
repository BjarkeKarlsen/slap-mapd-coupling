"""Unit tests for slap_mapd_coupling.environment.zones."""

import pytest

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.environment.zones import (
    check_partition,
    halo,
    zone_of,
    zones_by_aisle,
)
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance


def _line_graph(costs: list[float], movable: bool = True) -> WarehouseGraph:
    """Vertices 0..n joined by two-way edges with the given costs."""
    vertices = {i: Vertex(id=i, role=VertexRole(movable=movable)) for i in range(len(costs) + 1)}
    edges = []
    for i, cost in enumerate(costs):
        edges.append(Edge(source=i, target=i + 1, cost=cost))
        edges.append(Edge(source=i + 1, target=i, cost=cost))
    return WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=1.0)


def _aisle_line(num_aisles: int, per_aisle: int = 2) -> tuple[WarehouseGraph, dict[int, int]]:
    """A line of vertices, ``per_aisle`` of them in each aisle."""
    graph = _line_graph([1.0] * (num_aisles * per_aisle - 1))
    return graph, {v: v // per_aisle for v in graph.vertices}


def _aisles_per_zone(zones, aisle_index) -> list[int]:
    return [len({aisle_index[v] for v in zone}) for zone in zones]


def _params(seed: int) -> GeneratorParams:
    return GeneratorParams(
        max_num_aisles=7,
        max_aisle_length=7,
        max_num_cross_aisles=6,
        one_way_fraction=0.0,
        random_edge_costs=True,
        wait_cost=5.0,
        num_storage_vertices=4,
        num_delivery_vertices=4,
        num_endpoints=2,
        min_internal_role_vertices=2,
        seed=seed,
    )


def _movable(graph: WarehouseGraph) -> frozenset[int]:
    return frozenset(v for v, vertex in graph.vertices.items() if vertex.role.movable)


# --- zones_by_aisle ----------------------------------------------------------


def test_one_zone_is_the_movable_vertices_with_an_empty_halo():
    graph, aisle_index = _aisle_line(5)
    (zone,) = zones_by_aisle(graph, aisle_index, 1)
    assert zone == _movable(graph)
    assert halo(graph, zone, d_obs=100.0) == frozenset()


def test_one_zone_per_aisle_when_num_zones_equals_the_number_of_aisles():
    graph, aisle_index = _aisle_line(5)
    zones = zones_by_aisle(graph, aisle_index, 5)
    assert zones == tuple(frozenset({2 * a, 2 * a + 1}) for a in range(5))


def test_seven_aisles_into_three_zones_gives_3_2_2():
    graph, aisle_index = _aisle_line(7)
    zones = zones_by_aisle(graph, aisle_index, 3)
    assert _aisles_per_zone(zones, aisle_index) == [3, 2, 2]


def test_seven_aisles_into_two_zones_gives_4_3():
    graph, aisle_index = _aisle_line(7)
    zones = zones_by_aisle(graph, aisle_index, 2)
    assert _aisles_per_zone(zones, aisle_index) == [4, 3]


def test_zones_are_blocks_of_consecutive_aisles_left_to_right():
    graph, aisle_index = _aisle_line(7)
    zones = zones_by_aisle(graph, aisle_index, 3)
    blocks = [sorted({aisle_index[v] for v in zone}) for zone in zones]
    assert blocks == [[0, 1, 2], [3, 4], [5, 6]]


def test_num_zones_above_the_number_of_aisles_raises():
    graph, aisle_index = _aisle_line(4)
    with pytest.raises(ValueError):
        zones_by_aisle(graph, aisle_index, 5)


@pytest.mark.parametrize("num_zones", [0, -1])
def test_num_zones_below_one_raises(num_zones):
    graph, aisle_index = _aisle_line(4)
    with pytest.raises(ValueError):
        zones_by_aisle(graph, aisle_index, num_zones)


def test_num_zones_has_no_default():
    graph, aisle_index = _aisle_line(4)
    with pytest.raises(TypeError):
        zones_by_aisle(graph, aisle_index)  # type: ignore[call-arg]


def test_non_movable_vertices_belong_to_no_zone():
    graph = WarehouseGraph(
        vertices={
            0: Vertex(id=0),
            1: Vertex(id=1),
            2: Vertex(id=2, role=VertexRole(movable=False)),
        },
        edges=(Edge(source=0, target=1, cost=1.0), Edge(source=1, target=0, cost=1.0)),
        wait_cost=1.0,
    )
    (zone,) = zones_by_aisle(graph, {0: 0, 1: 0, 2: 1}, 1)
    assert zone == frozenset({0, 1})
    check_partition(graph, (zone,))


# --- generated instances -----------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_partition_holds_on_generated_instances(seed):
    instance = generate_instance(_params(seed))
    for num_zones in range(1, instance.num_aisles + 1):
        zones = zones_by_aisle(instance.graph, instance.aisle_index, num_zones)
        assert len(zones) == num_zones
        check_partition(instance.graph, zones)


@pytest.mark.parametrize("seed", range(6))
def test_cross_aisle_vertices_belong_to_the_zone_of_their_aisle(seed):
    instance = generate_instance(_params(seed))
    zones = zones_by_aisle(instance.graph, instance.aisle_index, min(2, instance.num_aisles))
    owner = zone_of(zones)
    for v, aisle in instance.aisle_index.items():
        # Two vertices of one aisle, cross-aisle or not, share a zone.
        same_aisle = [w for w, a in instance.aisle_index.items() if a == aisle]
        assert {owner[w] for w in same_aisle} == {owner[v]}
    # Cross-aisle vertices exist, so the check above covers them.
    assert instance.cross_aisle_rows


def test_zones_are_deterministic_for_a_fixed_seed():
    first = generate_instance(_params(3))
    second = generate_instance(_params(3))
    assert zones_by_aisle(first.graph, first.aisle_index, 2) == zones_by_aisle(
        second.graph, second.aisle_index, 2
    )


# --- check_partition and zone_of ---------------------------------------------


def test_check_partition_raises_on_an_overlap():
    graph = _line_graph([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        check_partition(graph, (frozenset({0, 1, 2}), frozenset({2, 3})))


def test_check_partition_raises_on_a_gap():
    graph = _line_graph([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        check_partition(graph, (frozenset({0, 1}), frozenset({3})))


def test_zone_of_gives_the_index_of_the_owning_zone():
    zones = (frozenset({0, 1}), frozenset({2, 3}))
    assert zone_of(zones) == {0: 0, 1: 0, 2: 1, 3: 1}


# --- halo --------------------------------------------------------------------


def test_halo_contains_the_neighbour_across_the_boundary_when_the_edge_is_cheap_enough():
    graph = _line_graph([1.0, 2.0, 1.0])
    assert halo(graph, frozenset({0, 1}), d_obs=2.0) == frozenset({2})


def test_halo_excludes_the_neighbour_when_the_edge_costs_more_than_d_obs():
    graph = _line_graph([1.0, 2.0, 1.0])
    assert halo(graph, frozenset({0, 1}), d_obs=1.5) == frozenset()


def test_halo_reaches_further_vertices_by_travel_cost():
    graph = _line_graph([1.0, 2.0, 1.0])
    assert halo(graph, frozenset({0, 1}), d_obs=3.0) == frozenset({2, 3})


def test_halo_never_contains_a_vertex_of_the_zone():
    graph = _line_graph([1.0, 1.0, 1.0])
    zone = frozenset({1, 2})
    assert halo(graph, zone, d_obs=10.0).isdisjoint(zone)
