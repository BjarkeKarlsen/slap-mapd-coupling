"""Unit tests for slap_mapd_coupling.environment.spaces.claim_mask."""

import random

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.environment.spaces import (
    action_for_slot,
    claim_mask,
    legality_mask,
    max_out_degree,
)
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.resolution.conflict_resolution import resolve_conflicts

NEG_INF = float("-inf")


def _graph() -> WarehouseGraph:
    """A line 1 <-> 2 <-> 3 with a spur 2 -> 4. Slots at 2 are 1, 3, 4."""
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 4)}
    pairs = ((1, 2), (2, 1), (2, 3), (3, 2), (2, 4))
    edges = tuple(Edge(source=a, target=b, cost=1.0) for a, b in pairs)
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _slot(graph: WarehouseGraph, vertex: int, target: int) -> int:
    return graph.legal_actions(vertex).index(Action(kind="move", target=target))


def test_nothing_chosen_and_no_neighbours_occupied_leaves_every_slot_open():
    graph = _graph()
    mask = claim_mask(graph, 1, {1: 2}, {})
    assert mask == (0.0,) * (max_out_degree(graph) + 1)


def test_vertex_chosen_by_an_earlier_agent_is_blocked():
    graph = _graph()
    mask = claim_mask(graph, 2, {1: 1, 2: 2}, {1: 3})
    assert mask[_slot(graph, 2, 3)] == 0.0  # 3 is empty and unclaimed
    mask = claim_mask(graph, 2, {1: 1, 2: 2}, {1: 4})
    assert mask[_slot(graph, 2, 4)] == NEG_INF


def test_vertex_holding_an_agent_that_has_not_chosen_is_blocked():
    graph = _graph()
    mask = claim_mask(graph, 2, {1: 1, 2: 2, 3: 3}, {})
    assert mask[_slot(graph, 2, 1)] == NEG_INF
    assert mask[_slot(graph, 2, 3)] == NEG_INF
    assert mask[_slot(graph, 2, 4)] == 0.0


def test_swap_with_an_earlier_agent_is_blocked():
    graph = _graph()
    mask = claim_mask(graph, 1, {1: 1, 2: 2}, {2: 1})
    assert mask[_slot(graph, 1, 2)] == NEG_INF


def test_following_an_earlier_agent_that_leaves_is_allowed():
    graph = _graph()
    mask = claim_mask(graph, 1, {1: 1, 2: 2}, {2: 3})
    assert mask[_slot(graph, 1, 2)] == 0.0


def test_an_earlier_agent_that_waits_blocks_its_own_vertex():
    graph = _graph()
    mask = claim_mask(graph, 1, {1: 1, 2: 2}, {2: 2})
    assert mask[_slot(graph, 1, 2)] == NEG_INF


def test_wait_is_never_switched_off():
    graph = _graph()
    mask = claim_mask(graph, 2, {1: 1, 2: 2, 3: 3}, {})
    assert mask[0] == 0.0


def test_slots_beyond_the_vertex_degree_stay_open_for_the_legality_mask_to_close():
    graph = _graph()
    mask = claim_mask(graph, 1, {1: 1, 2: 2}, {})
    assert mask[2:] == (0.0,) * (max_out_degree(graph) - 1)
    assert legality_mask(graph, 1)[2:] == (NEG_INF,) * (max_out_degree(graph) - 1)


def test_an_agent_blocked_from_waiting_raises():
    graph = _graph()
    with pytest.raises(RuntimeError, match="cannot wait"):
        claim_mask(graph, 2, {1: 1, 2: 2}, {1: 2})


def _random_instance(seed: int) -> WarehouseGraph:
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.3,
        wait_cost=0.5,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=4,
        seed=seed,
    )
    return generate_warehouse_graph(params)


@pytest.mark.parametrize("seed", range(200))
def test_choices_under_the_claim_mask_are_never_overridden(seed):
    graph = _random_instance(seed % 5)
    rng = random.Random(seed)
    movable = sorted(v for v, vertex in graph.vertices.items() if vertex.role.movable)
    num_agents = rng.randint(2, 8)
    locations = dict(enumerate(rng.sample(movable, num_agents), start=1))
    before = FleetState(
        agents={a: AgentState(agent_id=a, location=v) for a, v in locations.items()}
    )
    priority = list(locations)
    rng.shuffle(priority)
    d_max = max_out_degree(graph)

    chosen: dict[int, int] = {}
    proposed: dict[int, Action] = {}
    for agent_id in priority:
        legal = legality_mask(graph, locations[agent_id], d_max)
        claim = claim_mask(graph, agent_id, locations, chosen, d_max)
        open_slots = [k for k in range(d_max + 1) if legal[k] + claim[k] == 0.0]
        action = action_for_slot(graph, locations[agent_id], rng.choice(open_slots))
        proposed[agent_id] = action
        chosen[agent_id] = locations[agent_id] if action.kind == "wait" else action.target

    result = resolve_conflicts(graph, before, proposed, priority)
    assert result.overridden == frozenset()
