"""Unit tests for slap_mapd_coupling.resolution.conflict_resolution."""

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState, is_collision_free
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.resolution.conflict_resolution import (
    priority_permutation,
    resolve_conflicts,
)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


def _wait() -> Action:
    return Action(kind="wait")


def _move(target: int) -> Action:
    return Action(kind="move", target=target)


def test_priority_permutation_is_deterministic_given_seed():
    ids = [1, 2, 3, 4, 5]
    assert priority_permutation(42, ids) == priority_permutation(42, ids)


def test_priority_permutation_is_a_permutation():
    ids = [1, 2, 3, 4, 5]
    assert set(priority_permutation(7, ids)) == set(ids)


def test_no_conflict_all_accepted():
    # Two disjoint vertices, no shared graph needed for a no-op transition check.
    vertices = {i: Vertex(id=i) for i in (10, 20, 30, 40)}
    edges = (
        Edge(source=10, target=30, cost=1.0),
        Edge(source=20, target=40, cost=1.0),
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 10, 2: 20})
    proposed = {1: _move(30), 2: _move(40)}
    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))

    assert result.fleet.locations() == {1: 30, 2: 40}
    assert result.overridden == frozenset()
    assert is_collision_free(before, result.fleet)


def test_vertex_conflict_overrides_lower_priority_agent_to_wait():
    vertices = {i: Vertex(id=i) for i in (1, 2, 9)}
    edges = (Edge(source=1, target=9, cost=1.0), Edge(source=2, target=9, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2})
    proposed = {1: _move(9), 2: _move(9)}

    # Agent 1 has priority: it gets the vertex, agent 2 is overridden.
    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))
    assert result.fleet.locations() == {1: 9, 2: 2}
    assert result.overridden == frozenset({2})
    assert is_collision_free(before, result.fleet)

    # Reversed priority: agent 2 gets the vertex, agent 1 is overridden.
    result = resolve_conflicts(graph, before, proposed, priority=(2, 1))
    assert result.fleet.locations() == {1: 1, 2: 9}
    assert result.overridden == frozenset({1})
    assert is_collision_free(before, result.fleet)


def test_swap_conflict_forces_both_agents_to_wait():
    # Classic head-on swap: whichever agent is finalised first would move
    # into the other's still-occupied vertex if only eq:vertexconflict /
    # eq:swapconflict against *finalised* successors were checked. The
    # current-vertex reservation (see module docstring) blocks that, so
    # both agents end up waiting regardless of priority order -- not just
    # the lower-priority one.
    vertices = {i: Vertex(id=i) for i in (1, 2)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2})
    proposed = {1: _move(2), 2: _move(1)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))
    assert result.fleet.locations() == {1: 1, 2: 2}
    assert result.overridden == frozenset({1, 2})
    assert is_collision_free(before, result.fleet)

    # Symmetric under reversed priority.
    result = resolve_conflicts(graph, before, proposed, priority=(2, 1))
    assert result.fleet.locations() == {1: 1, 2: 2}
    assert result.overridden == frozenset({1, 2})
    assert is_collision_free(before, result.fleet)


def test_follow_move_into_already_vacated_vertex_succeeds():
    # Agent 1 (higher priority) moves out of v1 first; agent 2 can then
    # legally move into v1 once agent 1 is finalised as having vacated it.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=3, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 3})
    proposed = {1: _move(2), 2: _move(1)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))
    assert result.fleet.locations() == {1: 2, 2: 1}
    assert result.overridden == frozenset()
    assert is_collision_free(before, result.fleet)


def test_move_into_not_yet_processed_agents_vertex_is_blocked():
    # Agent 1 wants to move onto agent 2's current vertex, but agent 2
    # (lower priority) hasn't been processed yet -- unknown whether it
    # will vacate, so the move is blocked (agent 1 waits) even though
    # agent 2's own proposal happens to move it elsewhere.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=3, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2})
    proposed = {1: _move(2), 2: _move(3)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))
    assert result.fleet.locations() == {1: 1, 2: 3}
    assert result.overridden == frozenset({1})
    assert is_collision_free(before, result.fleet)


def test_operator_is_a_pure_function_of_inputs():
    vertices = {i: Vertex(id=i) for i in (1, 2, 9)}
    edges = (Edge(source=1, target=9, cost=1.0), Edge(source=2, target=9, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2})
    proposed = {1: _move(9), 2: _move(9)}
    priority = priority_permutation(seed=123, agent_ids=[1, 2])

    first = resolve_conflicts(graph, before, proposed, priority)
    second = resolve_conflicts(graph, before, proposed, priority)
    assert first.fleet.locations() == second.fleet.locations()
    assert first.overridden == second.overridden


def test_rejects_illegal_proposed_action():
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0),)
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1})
    proposed = {1: _move(3)}  # 1 -> 3 is not an edge

    with pytest.raises(ValueError):
        resolve_conflicts(graph, before, proposed, priority=(1,))


def test_rejects_priority_not_matching_fleet():
    vertices = {i: Vertex(id=i) for i in (1, 2)}
    graph = WarehouseGraph(vertices=vertices, edges=(), wait_cost=0.0)
    before = _fleet({1: 1})

    with pytest.raises(ValueError):
        resolve_conflicts(graph, before, {1: _wait()}, priority=(1, 2))


def test_three_agents_all_targeting_one_vertex_only_top_priority_moves():
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 9)}
    edges = tuple(Edge(source=s, target=9, cost=1.0) for s in (1, 2, 3))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2, 3: 3})
    proposed = {1: _move(9), 2: _move(9), 3: _move(9)}

    result = resolve_conflicts(graph, before, proposed, priority=(2, 3, 1))
    assert result.fleet.locations() == {1: 1, 2: 9, 3: 3}
    assert result.overridden == frozenset({1, 3})
    assert is_collision_free(before, result.fleet)
