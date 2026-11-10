"""Unit tests for slap_mapd_coupling.resolution.conflict_resolution."""

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState, is_collision_free
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.resolution.conflict_resolution import (
    move_blocked,
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


def test_a_robot_follows_a_later_robot_that_leaves():
    # [A81]: agent 1 comes first in the order and wants agent 2's vertex. Agent 2 moves on, so
    # agent 2 is handled first and agent 1 follows it in the same step, with no override.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3)}
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=3, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2})
    proposed = {1: _move(2), 2: _move(3)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2))
    assert result.fleet.locations() == {1: 2, 2: 3}
    assert result.overridden == frozenset()
    assert is_collision_free(before, result.fleet)


def test_a_whole_line_moves_up_together():
    vertices = {i: Vertex(id=i) for i in range(1, 6)}
    edges = tuple(Edge(source=i, target=i + 1, cost=1.0) for i in range(1, 5))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2, 3: 3, 4: 4})
    proposed = {aid: _move(aid + 1) for aid in (1, 2, 3, 4)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2, 3, 4))
    assert result.fleet.locations() == {1: 2, 2: 3, 3: 4, 4: 5}
    assert result.overridden == frozenset()


def test_a_robot_behind_a_robot_that_is_stopped_waits():
    # Agent 2 is stopped by agent 3, which takes vertex 9 first, so agent 1 behind it waits too.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 9)}
    edges = (
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=9, cost=1.0),
        Edge(source=3, target=9, cost=1.0),
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2, 3: 3})
    proposed = {1: _move(2), 2: _move(9), 3: _move(9)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 3, 2))
    assert result.fleet.locations() == {1: 1, 2: 2, 3: 9}
    assert result.overridden == frozenset({1, 2})
    assert is_collision_free(before, result.fleet)


def test_the_priority_order_decides_who_takes_a_vertex_that_is_left():
    # Agents 1 and 3 both want the vertex agent 2 leaves. Both wait for agent 2, and then agent
    # 1, earlier in the order, takes it, although agent 3 comes right after agent 2.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 4)}
    edges = (
        Edge(source=1, target=2, cost=1.0),
        Edge(source=3, target=2, cost=1.0),
        Edge(source=2, target=4, cost=1.0),
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2, 3: 3})
    proposed = {1: _move(2), 2: _move(4), 3: _move(2)}

    result = resolve_conflicts(graph, before, proposed, priority=(1, 2, 3))
    assert result.fleet.locations() == {1: 2, 2: 4, 3: 3}
    assert result.overridden == frozenset({3})


def test_a_rotation_and_the_robots_behind_it_wait():
    # Agents 1, 2, 3 each want the next one's vertex, a cycle, and agent 4 wants agent 1's.
    vertices = {i: Vertex(id=i) for i in (1, 2, 3, 4)}
    edges = (
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=3, cost=1.0),
        Edge(source=3, target=1, cost=1.0),
        Edge(source=4, target=1, cost=1.0),
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before = _fleet({1: 1, 2: 2, 3: 3, 4: 4})
    proposed = {1: _move(2), 2: _move(3), 3: _move(1), 4: _move(1)}

    result = resolve_conflicts(graph, before, proposed, priority=(4, 3, 2, 1))
    assert result.fleet.locations() == {1: 1, 2: 2, 3: 3, 4: 4}
    assert result.overridden == frozenset({1, 2, 3, 4})


def test_a_joint_action_that_passes_the_claim_mask_is_accepted_whole():
    # Agents choose in priority order with move_blocked as the claim mask. Whatever they choose,
    # conflict resolution overrides none of them.
    vertices = {i: Vertex(id=i) for i in range(1, 7)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for a in range(1, 7)
        for b in range(1, 7)
        if a != b and abs(a - b) <= 2
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    before_loc = {1: 1, 2: 3, 3: 5}
    before = _fleet(before_loc)
    priority = (2, 3, 1)
    for wanted in ((3, 5, 2), (5, 4, 1), (2, 6, 3), (4, 6, 2)):
        chosen: dict[int, int] = {}
        proposed: dict[int, Action] = {}
        for agent_id, target in zip(priority, wanted):
            start = before_loc[agent_id]
            ok = target != start and not move_blocked(agent_id, start, target, before_loc, chosen)
            proposed[agent_id] = _move(target) if ok else _wait()
            chosen[agent_id] = target if ok else start
        result = resolve_conflicts(graph, before, proposed, priority)
        assert result.overridden == frozenset()
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


def test_move_blocked_is_the_rule_of_the_claim_mask():
    before_loc = {1: 1, 2: 2, 3: 3}
    # vertex rule
    assert move_blocked(2, 2, 9, before_loc, {1: 9})
    # swap rule
    assert move_blocked(2, 2, 1, before_loc, {1: 2})
    # an agent that has not chosen stands on the target
    assert move_blocked(2, 2, 3, before_loc, {1: 1})
    # following an agent that has left the target is allowed
    assert not move_blocked(2, 2, 1, before_loc, {1: 9})
