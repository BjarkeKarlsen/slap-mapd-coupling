"""Unit tests for slap_mapd_coupling.instances.generator."""

from slap_mapd_coupling.core.graph import Edge, WarehouseGraph
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_instance,
    generate_warehouse_graph,
)


def _params(one_way_fraction: float, seed: int = 0) -> GeneratorParams:
    return GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=1,
        one_way_fraction=one_way_fraction,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=2,
        seed=seed,
    )


def _has_reverse(graph: WarehouseGraph, edge: Edge) -> bool:
    return any(e.source == edge.target and e.target == edge.source for e in graph.edges)


def test_generator_produces_at_least_one_one_way_edge():
    graph = generate_warehouse_graph(_params(one_way_fraction=1.0))
    assert any(not _has_reverse(graph, e) for e in graph.edges)


def test_generator_zero_one_way_fraction_is_fully_bidirectional():
    graph = generate_warehouse_graph(_params(one_way_fraction=0.0))
    assert all(_has_reverse(graph, e) for e in graph.edges)


def test_generator_is_deterministic_given_seed():
    params = _params(one_way_fraction=0.3, seed=42)
    g1 = generate_warehouse_graph(params)
    g2 = generate_warehouse_graph(params)
    assert g1.vertices == g2.vertices
    assert g1.edges == g2.edges
    assert g1.wait_cost == g2.wait_cost


def test_generate_instance_positions_match_graph_vertices():
    instance = generate_instance(_params(one_way_fraction=0.1))
    assert set(instance.positions.keys()) == set(instance.graph.vertices.keys())


def test_generate_warehouse_graph_unchanged_by_refactor():
    params = _params(one_way_fraction=0.2, seed=7)
    direct = generate_warehouse_graph(params)
    via_instance = generate_instance(params).graph
    assert direct.vertices == via_instance.vertices
    assert direct.edges == via_instance.edges
    assert direct.wait_cost == via_instance.wait_cost


def _storage_vertices(graph: WarehouseGraph) -> set[int]:
    return {v for v, vertex in graph.vertices.items() if vertex.role.storage}


def test_storage_vertices_are_a_seeded_random_subset_of_aisle_cells():
    # Issue #75: storage placement used to be "the first num_storage_vertices
    # aisle cells in generation order", identical across every seed. It's now
    # a seeded random sample, so different seeds give different placements
    # (same count, still all real aisle cells) -- and the same seed still
    # gives the same placement, per test_generator_is_deterministic_given_seed.
    graph_a = generate_warehouse_graph(_params(one_way_fraction=0.0, seed=1))
    graph_b = generate_warehouse_graph(_params(one_way_fraction=0.0, seed=2))
    storage_a = _storage_vertices(graph_a)
    storage_b = _storage_vertices(graph_b)

    assert len(storage_a) == 6
    assert len(storage_b) == 6
    assert storage_a != storage_b
