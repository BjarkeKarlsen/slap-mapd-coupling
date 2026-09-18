"""Unit tests for slap_mapd_coupling.core.agents."""

from slap_mapd_coupling.core.agents import (
    AgentState,
    FleetState,
    has_swap_conflict,
    has_vertex_conflict,
    is_collision_free,
    is_legal_transition,
)
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph


def _small_graph() -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0),)
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_is_legal_transition_wait_and_move():
    graph = _small_graph()
    assert is_legal_transition(graph, 1, 1) is True
    assert is_legal_transition(graph, 1, 2) is True
    assert is_legal_transition(graph, 1, 3) is False


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


def test_vertex_conflict_detected():
    before = _fleet({1: 10, 2: 20})
    after = _fleet({1: 30, 2: 30})
    assert has_vertex_conflict(before, after) is not None


def test_swap_conflict_detected():
    before = _fleet({1: 10, 2: 20})
    after = _fleet({1: 20, 2: 10})
    assert has_swap_conflict(before, after) in {(1, 2), (2, 1)}


def test_no_conflict_on_disjoint_moves():
    before = _fleet({1: 10, 2: 20})
    after = _fleet({1: 30, 2: 40})
    assert is_collision_free(before, after) is True
