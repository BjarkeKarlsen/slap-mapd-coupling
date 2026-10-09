"""Unit tests for slap_mapd_coupling.instances.generator."""

import pytest

from slap_mapd_coupling.core.graph import Edge, WarehouseGraph
from slap_mapd_coupling.instances import generator
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    InstanceGenerationError,
    generate_and_validate,
    generate_instance,
    generate_warehouse_graph,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness

FLEET_SIZE = 2


def _params(one_way_fraction: float = 0.0, seed: int = 0, **overrides) -> GeneratorParams:
    values = dict(
        max_num_aisles=4,
        max_aisle_length=5,
        max_num_cross_aisles=2,
        one_way_fraction=one_way_fraction,
        wait_cost=0.0,
        num_storage_vertices=4,
        num_delivery_vertices=2,
        num_endpoints=FLEET_SIZE,
        seed=seed,
    )
    values.update(overrides)
    return GeneratorParams(**values)


def _large_params(seed: int) -> GeneratorParams:
    """The sparse setting of example 00, where many topologies admit no roles."""
    return GeneratorParams(
        max_num_aisles=7,
        max_aisle_length=7,
        max_num_cross_aisles=6,
        one_way_fraction=0.0,
        random_edge_costs=True,
        wait_cost=5.0,
        num_storage_vertices=4,
        num_delivery_vertices=4,
        num_endpoints=FLEET_SIZE,
        min_internal_role_vertices=2,
        seed=seed,
    )


def _has_reverse(graph: WarehouseGraph, edge: Edge) -> bool:
    return any(e.source == edge.target and e.target == edge.source for e in graph.edges)


def test_generator_produces_at_least_one_one_way_edge():
    graph = generate_warehouse_graph(_params(one_way_fraction=0.2))
    assert any(not _has_reverse(graph, e) for e in graph.edges)


def test_generator_zero_one_way_fraction_is_fully_bidirectional():
    graph = generate_warehouse_graph(_params(one_way_fraction=0.0))
    assert all(_has_reverse(graph, e) for e in graph.edges)


def test_generator_is_deterministic_given_seed():
    params = _params(one_way_fraction=0.2, seed=42)
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


def test_storage_vertices_are_a_seeded_random_subset_of_the_graph():
    # Issue #75: storage placement differs between seeds, with the same
    # count, and is the same for the same seed.
    storage_a = _storage_vertices(generate_warehouse_graph(_params(seed=1)))
    storage_b = _storage_vertices(generate_warehouse_graph(_params(seed=2)))
    assert len(storage_a) == len(storage_b) == 4
    assert storage_a != storage_b


# --- Every seed gives a well-formed instance (B5) ---------------------------


def _assert_usable(instance) -> None:
    assert check_connectivity(instance.graph).ok
    assert check_well_formedness(instance.graph, FLEET_SIZE).ok


def test_a_seed_whose_first_topology_has_no_role_assignment_still_gives_an_instance():
    # Seed 664 used to raise: its first topology is nearly a tree, so no
    # placement of eight roles leaves the plain vertices connected.
    instance = generate_instance(_large_params(seed=664))
    assert instance.topology_attempts > 1
    _assert_usable(instance)


def test_every_seed_in_a_range_gives_a_well_formed_instance():
    seeds = range(40)
    instances = [generate_instance(_large_params(seed=s)) for s in seeds]
    for instance in instances:
        _assert_usable(instance)
    # The range includes seeds that needed a second topology and seeds that
    # did not, so both paths are exercised.
    attempts = {instance.topology_attempts for instance in instances}
    assert 1 in attempts
    assert max(attempts) > 1


def test_a_seed_that_needed_retries_is_reproducible():
    # Training and evaluation seeds each map to one fixed instance, however
    # many topologies were discarded on the way.
    params = _large_params(seed=664)
    first = generate_instance(params)
    second = generate_instance(params)
    assert first.topology_attempts == second.topology_attempts
    assert first.graph.vertices == second.graph.vertices
    assert first.graph.edges == second.graph.edges
    assert first.positions == second.positions


def test_different_seeds_do_not_share_a_topology_stream():
    # Retries of seed s must not reproduce the topology of another seed.
    shapes = {
        (i.aisle_lengths, i.cross_aisle_rows)
        for i in (generate_instance(_large_params(seed=s)) for s in range(10))
    }
    assert len(shapes) > 1


def test_report_carries_the_number_of_topologies_sampled():
    params = _large_params(seed=664)
    graph, report = generate_and_validate(params, fleet_size=FLEET_SIZE)
    assert report.accepted
    assert report.topology_attempts == generate_instance(params).topology_attempts
    assert graph.vertices == generate_instance(params).graph.vertices


def test_infeasible_parameters_raise_after_the_attempt_limit(monkeypatch):
    # One aisle and no cross-aisle is a path. Plain vertices stay connected
    # only if roles sit at its two ends, so six roles can never be placed.
    monkeypatch.setattr(generator, "_MAX_TOPOLOGY_ATTEMPTS", 3)
    monkeypatch.setattr(generator, "_MAX_ROLE_ATTEMPTS", 50)
    params = GeneratorParams(
        max_num_aisles=1,
        max_aisle_length=6,
        max_num_cross_aisles=0,
        require_irregular_shape=False,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=2,
        num_delivery_vertices=2,
        num_endpoints=2,
        seed=0,
    )
    with pytest.raises(InstanceGenerationError, match="none of 3 sampled topologies"):
        generate_instance(params)


def test_fixed_size_with_no_irregular_shape_raises_without_resampling():
    # 7 aisles of length 7 is the largest graph, which has only one shape.
    # The size is fixed, so this is a parameter error and not a bad seed.
    params = GeneratorParams(
        max_num_aisles=7,
        max_aisle_length=7,
        max_num_cross_aisles=6,
        num_transit_vertices=56,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=4,
        num_delivery_vertices=4,
        num_endpoints=2,
        seed=0,
    )
    with pytest.raises(InstanceGenerationError, match="Unequal aisle lengths"):
        generate_instance(params)
