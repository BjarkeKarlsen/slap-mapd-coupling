"""Unit tests for slap_mapd_coupling.models.local_subgraph_encoding."""

import numpy as np
import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.environment.spaces import legality_mask, max_out_degree
from slap_mapd_coupling.models.gnn_encoder import MESSAGE_FEATURE_DIM, NODE_FEATURE_DIM
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_observation,
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


def test_encode_observation_shapes_match_config():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=3)

    encoded = encode_observation(
        graph, agent_location=2, observation=observation, mask=mask, config=encoding_config
    )

    assert encoded.node_features.shape == (8, NODE_FEATURE_DIM)
    assert encoded.node_mask.shape == (8,)
    assert encoded.adjacency.shape == (8, 8)
    assert encoded.message_features.shape == (3, MESSAGE_FEATURE_DIM)
    assert encoded.message_mask.shape == (3,)
    assert encoded.action_mask.shape == (d_max + 1,)


def test_encode_observation_own_index_points_at_agent_vertex():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    mask = legality_mask(graph, vertex=2)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(
        graph, agent_location=2, observation=observation, mask=mask, config=encoding_config
    )

    own_features = observation.features[2]
    np.testing.assert_allclose(
        encoded.node_features[encoded.own_index],
        [
            float(own_features.storage),
            float(own_features.delivery),
            float(own_features.endpoint),
            float(own_features.transit),
            own_features.eta,
            float(own_features.occupied),
        ],
    )
    assert encoded.node_mask[encoded.own_index] == 1.0


def test_encode_observation_node_mask_marks_padding():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert len(observation.visible_vertices) == 3  # {1, 2, 3}
    mask = legality_mask(graph, vertex=2)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(
        graph, agent_location=2, observation=observation, mask=mask, config=encoding_config
    )

    assert encoded.node_mask.sum() == 3
    assert encoded.node_mask[3:].sum() == 0


def test_encode_observation_adjacency_matches_line_graph_neighbours():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    mask = legality_mask(graph, vertex=2)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(
        graph, agent_location=2, observation=observation, mask=mask, config=encoding_config
    )

    # visible = {1, 2, 3}; edges 1-2 and 2-3, no edge 1-3.
    index = {v: i for i, v in enumerate(observation.visible_vertices)}
    assert encoded.adjacency[index[1], index[2]] == 1.0
    assert encoded.adjacency[index[2], index[1]] == 1.0
    assert encoded.adjacency[index[2], index[3]] == 1.0
    assert encoded.adjacency[index[1], index[3]] == 0.0
    assert encoded.adjacency[7, 0] == 0.0  # padded rows stay zero


def test_encode_observation_truncates_nodes_keeping_own_vertex_and_closest_eta():
    graph = _line_graph(11)
    fleet = _fleet({1: 5})
    task = Task(task_id=1, release_time=0, pickup_vertex=10, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    config = _config(observation_depth=5)
    observation = build_observation(graph, fleet, [task], agent_id=1, t=0, config=config)
    assert len(observation.visible_vertices) == 11  # {0..10}
    mask = legality_mask(graph, vertex=5)

    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=4, max_messages=1)
    encoded = encode_observation(
        graph, agent_location=5, observation=observation, mask=mask, config=encoding_config
    )

    assert encoded.node_mask.sum() == 4
    assert encoded.node_mask[encoded.own_index] == 1.0
    # target is vertex 10 (eta 0 there); closest-by-eta among the rest are
    # 10, 9, 8 (eta 0, 1, 2) -- the three smallest etas other than the
    # agent's own vertex (5, eta 5, which is force-kept regardless).
    kept_etas = sorted(
        float(encoded.node_features[i][4]) for i in range(4) if i != encoded.own_index
    )
    assert kept_etas == [0.0, 1.0, 2.0]


def test_encode_observation_truncates_messages_keeping_closest_senders():
    graph = _line_graph(10)
    fleet = _fleet({1: 0, 2: 2, 3: 4, 4: 6})
    config = _config(observation_depth=9, communication=True, communication_radius=9)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert len(observation.messages) == 3  # distances 2, 4, 6
    mask = legality_mask(graph, vertex=0)

    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=12, max_messages=2)
    encoded = encode_observation(
        graph, agent_location=0, observation=observation, mask=mask, config=encoding_config
    )

    assert encoded.message_mask.sum() == 2
    kept_distances = sorted(
        float(encoded.message_features[i][0]) for i in range(2) if encoded.message_mask[i] == 1.0
    )
    assert kept_distances == [2.0, 4.0]  # the two closest senders, not the one at distance 6


def test_encode_observation_congestion_and_action_mask():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=2, congestion_sensitive=True, congestion_radius=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    mask = legality_mask(graph, vertex=2)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(
        graph, agent_location=2, observation=observation, mask=mask, config=encoding_config
    )

    assert encoded.congestion == pytest.approx(observation.congestion)
    assert set(np.unique(encoded.action_mask)).issubset({0.0, 1.0})
    assert encoded.action_mask[0] == 1.0  # wait is always legal
