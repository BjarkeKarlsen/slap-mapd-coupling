"""Unit tests for slap_mapd_coupling.environment.observation."""

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import (
    build_observation,
    communication_neighbours,
    local_subgraph,
    occupancy_fraction,
    role_flags,
)


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=1.0,
        seed=0,
        horizon=50,
        wait_cost=0.0,
        observation_depth=2,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_local_subgraph_respects_depth():
    graph = _line_graph(6)
    assert set(local_subgraph(graph, 2, depth=1)) == {1, 2, 3}
    assert set(local_subgraph(graph, 2, depth=2)) == {0, 1, 2, 3, 4}


def test_role_flags_multi_hot_for_overlapping_roles():
    role = VertexRole(movable=True, storage=True, endpoint=True)
    assert role_flags(role) == (True, False, True, False)


def test_role_flags_transit_when_no_other_role():
    role = VertexRole(movable=True)
    assert role_flags(role) == (False, False, False, True)


def test_occupancy_fraction_matches_worked_example():
    fleet = _fleet({1: 10, 2: 11, 3: 12, 4: 99})
    # window = {10,11,12,13}, excl={1}; agents 2 (at 11) and 3 (at 12) are
    # inside, agent 4 (at 99) is not, giving 2/(4-1) = 2/3.
    fraction = occupancy_fraction(fleet, window=[10, 11, 12, 13], excl={1})
    assert fraction == pytest.approx(2 / 3)


def test_occupancy_fraction_empty_window_uses_max_one_denominator():
    fleet = _fleet({1: 10})
    assert occupancy_fraction(fleet, window=[], excl={1}) == 0.0


def test_communication_neighbours_within_radius():
    graph = _line_graph(10)
    fleet = _fleet({1: 0, 2: 2, 3: 8})
    assert set(communication_neighbours(graph, fleet, agent_id=1, communication_radius=2)) == {2}


def test_build_observation_raises_without_observation_depth():
    # ExperimentConfig itself already forbids controller="decentralised"
    # without observation_depth; model_copy bypasses that validator (same
    # trick core/tasks.py's own docstring warns about) to simulate a
    # caller bug reaching build_observation's own defensive check.
    graph = _line_graph(5)
    fleet = _fleet({1: 2})
    config = _config().model_copy(update={"observation_depth": None})
    with pytest.raises(ValueError, match="observation_depth"):
        build_observation(graph, fleet, [], agent_id=1, t=0, config=config)


def test_build_observation_visible_vertices_and_features():
    graph = _line_graph(5)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=1)
    obs = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    assert set(obs.visible_vertices) == {1, 2, 3}
    assert obs.features[3].occupied is True  # agent 2 sits there
    assert obs.features[1].occupied is False


def test_build_observation_eta_uses_current_goal_before_pickup():
    graph = _line_graph(5)
    fleet = _fleet({1: 0})
    task = Task(task_id=1, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    config = _config(observation_depth=4)

    obs = build_observation(graph, fleet, [task], agent_id=1, t=0, config=config)
    assert obs.features[0].eta == 4  # d_G(0, pickup=4)
    assert obs.features[4].eta == 0  # the goal itself


def test_build_observation_congestion_none_when_radius_unset():
    graph = _line_graph(5)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=2, congestion_radius=None)
    obs = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert obs.congestion is None


def test_build_observation_congestion_computed_when_radius_set():
    graph = _line_graph(5)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=2, congestion_sensitive=True, congestion_radius=1)
    obs = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    # window = {1,2,3}, excl={1}; agent 2 (at 3) is inside -> 1/(3-1)=0.5
    assert obs.congestion == pytest.approx(0.5)


def test_build_observation_no_messages_when_communication_disabled():
    graph = _line_graph(5)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=2, communication=False)
    obs = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert obs.messages == ()


def test_build_observation_messages_within_communication_radius():
    graph = _line_graph(5)
    fleet = _fleet({1: 0, 2: 1, 3: 4})
    task2 = Task(task_id=2, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1").assign(
        agent_id=2, t=0
    )
    config = _config(observation_depth=4, communication=True, communication_radius=1)

    obs = build_observation(graph, fleet, [task2], agent_id=1, t=0, config=config)
    senders = {m.sender for m in obs.messages}
    assert senders == {2}  # agent 3 (at vertex 4) is out of range
    message = next(m for m in obs.messages if m.sender == 2)
    assert message.distance == 1  # d_G(0, 1)
    assert message.eta == 3  # d_G(1, pickup=4)
