"""Unit tests for slap_mapd_coupling.models.gnn_encoder."""

import torch

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.models.gnn_encoder import (
    MESSAGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GNNEncoder,
    GNNEncoderConfig,
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


def test_output_dim_with_zero_rounds_uses_raw_node_features():
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    assert encoder.output_dim == NODE_FEATURE_DIM + MESSAGE_FEATURE_DIM + 1


def test_output_dim_with_rounds_uses_hidden_width():
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    assert encoder.output_dim == 8 + MESSAGE_FEATURE_DIM + 1


def test_forward_output_shape_matches_output_dim():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    z = encoder(graph, agent_location=2, observation=observation)
    assert z.shape == (encoder.output_dim,)


def test_zero_rounds_returns_raw_features_concatenated_with_message_and_congestion():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3, congestion_sensitive=True, congestion_radius=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    z = encoder(graph, agent_location=2, observation=observation)

    own_features = observation.features[2]
    expected_node_part = torch.tensor(
        [
            float(own_features.storage),
            float(own_features.delivery),
            float(own_features.endpoint),
            float(own_features.transit),
            own_features.eta,
            float(own_features.occupied),
        ]
    )
    assert torch.allclose(z[:NODE_FEATURE_DIM], expected_node_part)
    assert torch.allclose(z[NODE_FEATURE_DIM:-1], torch.zeros(MESSAGE_FEATURE_DIM))  # no messages
    assert z[-1].item() == observation.congestion


def test_congestion_falls_back_to_zero_when_not_configured():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=3, congestion_radius=None)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert observation.congestion is None

    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    z = encoder(graph, agent_location=2, observation=observation)
    assert z[-1].item() == 0.0


def test_gradients_flow_through_message_passing_rounds():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    z = encoder(graph, agent_location=2, observation=observation)
    z.sum().backward()

    assert all(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())


def test_forward_is_deterministic_for_fixed_weights_and_input():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()
    with torch.no_grad():
        first = encoder(graph, agent_location=2, observation=observation)
        second = encoder(graph, agent_location=2, observation=observation)
    assert torch.equal(first, second)
