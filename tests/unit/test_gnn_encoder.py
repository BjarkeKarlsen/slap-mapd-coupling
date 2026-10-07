"""Unit tests for slap_mapd_coupling.models.gnn_encoder."""

import torch

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
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


def _padded_tensors_from_observation(
    graph: WarehouseGraph, agent_location: int, observation
) -> tuple[torch.Tensor, ...]:
    """Builds the exact (unpadded, N=len(visible)) tensors `forward_padded`
    expects, straight from an unbatched Observation -- used only to check
    forward_padded agrees with forward on the same input (the full
    truncation/padding policy is #28's models/local_subgraph_encoding.py,
    not tested here)."""
    visible = list(observation.visible_vertices)
    n = len(visible)
    index = {v: i for i, v in enumerate(visible)}

    node_features = torch.zeros(1, n, NODE_FEATURE_DIM)
    for i, v in enumerate(visible):
        f = observation.features[v]
        node_features[0, i] = torch.tensor(
            [
                float(f.storage),
                float(f.delivery),
                float(f.endpoint),
                float(f.transit),
                f.eta,
                float(f.occupied),
            ]
        )
    node_mask = torch.ones(1, n)

    adjacency = torch.zeros(1, n, n)
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            adjacency[0, index[edge.source], index[edge.target]] = 1.0
            adjacency[0, index[edge.target], index[edge.source]] = 1.0

    own_index = torch.tensor([index[agent_location]], dtype=torch.long)

    if observation.messages:
        m = len(observation.messages)
        message_features = torch.zeros(1, m, MESSAGE_FEATURE_DIM)
        for i, message in enumerate(observation.messages):
            message_features[0, i] = torch.tensor([message.distance, message.eta])
        message_mask = torch.ones(1, m)
    else:
        message_features = torch.zeros(1, 1, MESSAGE_FEATURE_DIM)
        message_mask = torch.zeros(1, 1)

    congestion = torch.tensor(
        [observation.congestion if observation.congestion is not None else 0.0]
    )

    return (
        node_features,
        node_mask,
        adjacency,
        own_index,
        message_features,
        message_mask,
        congestion,
    )


def test_forward_padded_matches_unbatched_forward_on_same_observation():
    """forward_padded is a vectorised reformulation of the identical
    eq:msgpass computation (see module docstring) -- the two must agree
    numerically on a batch-of-one built from the same Observation."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()

    with torch.no_grad():
        unbatched = encoder(graph, agent_location=2, observation=observation)
        batched = encoder.forward_padded(
            *_padded_tensors_from_observation(graph, agent_location=2, observation=observation)
        )

    assert torch.allclose(unbatched, batched[0], atol=1e-6)


def test_forward_padded_matches_unbatched_forward_with_messages_and_zero_rounds():
    graph = _line_graph(6)
    fleet = _fleet({1: 0, 2: 1, 3: 4})
    task2 = Task(task_id=2, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1").assign(
        agent_id=2, t=0
    )
    config = _config(observation_depth=4, communication=True, communication_radius=1)
    observation = build_observation(graph, fleet, [task2], agent_id=1, t=0, config=config)
    assert observation.messages  # sanity: this scenario actually exercises messages

    torch.manual_seed(1)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    encoder.eval()

    with torch.no_grad():
        unbatched = encoder(graph, agent_location=0, observation=observation)
        batched = encoder.forward_padded(
            *_padded_tensors_from_observation(graph, agent_location=0, observation=observation)
        )

    assert torch.allclose(unbatched, batched[0], atol=1e-6)


def test_forward_padded_ignores_padded_nodes_and_messages():
    """Appending masked-out (all-zero) padding rows/columns must not change
    the result -- this is the whole point of node_mask/message_mask."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()

    tensors = _padded_tensors_from_observation(graph, agent_location=2, observation=observation)
    node_features, node_mask, adjacency, own_index, message_features, message_mask, congestion = (
        tensors
    )

    pad_nodes = 3
    n = node_features.shape[1]
    padded_node_features = torch.cat(
        [node_features, torch.randn(1, pad_nodes, NODE_FEATURE_DIM)], dim=1
    )
    padded_node_mask = torch.cat([node_mask, torch.zeros(1, pad_nodes)], dim=1)
    padded_adjacency = torch.zeros(1, n + pad_nodes, n + pad_nodes)
    padded_adjacency[:, :n, :n] = adjacency

    pad_messages = 2
    padded_message_features = torch.cat(
        [message_features, torch.randn(1, pad_messages, MESSAGE_FEATURE_DIM)], dim=1
    )
    padded_message_mask = torch.cat([message_mask, torch.zeros(1, pad_messages)], dim=1)

    with torch.no_grad():
        unpadded = encoder.forward_padded(
            node_features,
            node_mask,
            adjacency,
            own_index,
            message_features,
            message_mask,
            congestion,
        )
        padded = encoder.forward_padded(
            padded_node_features,
            padded_node_mask,
            padded_adjacency,
            own_index,
            padded_message_features,
            padded_message_mask,
            congestion,
        )

    assert torch.allclose(unpadded, padded, atol=1e-6)


def test_forward_padded_output_shape_is_batched():
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=1, hidden_width=8))
    batch_size, n, m = 3, 4, 2
    node_features = torch.randn(batch_size, n, NODE_FEATURE_DIM)
    node_mask = torch.ones(batch_size, n)
    adjacency = torch.zeros(batch_size, n, n)
    own_index = torch.zeros(batch_size, dtype=torch.long)
    message_features = torch.randn(batch_size, m, MESSAGE_FEATURE_DIM)
    message_mask = torch.ones(batch_size, m)
    congestion = torch.zeros(batch_size)

    z = encoder.forward_padded(
        node_features, node_mask, adjacency, own_index, message_features, message_mask, congestion
    )
    assert z.shape == (batch_size, encoder.output_dim)
