"""Unit tests for slap_mapd_coupling.models.gnn_encoder."""

import pytest
import torch

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.environment.spaces import legality_mask
from slap_mapd_coupling.models.gnn_encoder import (
    MESSAGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GNNEncoder,
    GNNEncoderConfig,
)
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
    config = _config(observation_depth=3, congestion_sensitive=True)
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
    assert z[-1].item() == torch.tensor(observation.congestion).item()  # float32


def test_congestion_is_read_even_when_not_sensitive():
    """delta_i(t) is always one of o_i(t)'s three pieces (sec:pf:observations);
    congestion_sensitive only decides whether the policy reacts to it."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=3, congestion_sensitive=False)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert observation.congestion > 0.0

    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    z = encoder(graph, agent_location=2, observation=observation)
    assert z[-1].item() == pytest.approx(observation.congestion)


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
    """The unpadded tensors `forward_padded` expects, built straight from an
    unbatched Observation (N = len(visible), E = number of undirected pairs)."""
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

    pairs = sorted(
        {
            (min(index[e.source], index[e.target]), max(index[e.source], index[e.target]))
            for e in graph.edges
            if e.source in index and e.target in index
        }
    )
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(1, len(pairs), 2)
    edge_mask = torch.ones(1, len(pairs))

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

    congestion = torch.tensor([observation.congestion])

    return (
        node_features,
        node_mask,
        edge_index,
        edge_mask,
        own_index,
        message_features,
        message_mask,
        congestion,
    )


def test_forward_padded_matches_unbatched_forward_on_same_observation():
    """The message-passing step over an edge list equals the unbatched
    forward on a batch of one built from the same Observation."""
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
    config = _config(observation_depth=4, communication=True)
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
    """Padded vertices, edges and messages must not change the result."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()

    tensors = _padded_tensors_from_observation(graph, agent_location=2, observation=observation)
    (
        node_features,
        node_mask,
        edge_index,
        edge_mask,
        own_index,
        message_features,
        message_mask,
        congestion,
    ) = tensors

    pad_nodes = 3
    padded_node_features = torch.cat(
        [node_features, torch.randn(1, pad_nodes, NODE_FEATURE_DIM)], dim=1
    )
    padded_node_mask = torch.cat([node_mask, torch.zeros(1, pad_nodes)], dim=1)
    pad_edges = 4
    # padded edges point at real vertices on purpose, only the mask silences them
    padded_edge_index = torch.cat([edge_index, torch.tensor([[[1, 2]] * pad_edges])], dim=1)
    padded_edge_mask = torch.cat([edge_mask, torch.zeros(1, pad_edges)], dim=1)

    pad_messages = 2
    padded_message_features = torch.cat(
        [message_features, torch.randn(1, pad_messages, MESSAGE_FEATURE_DIM)], dim=1
    )
    padded_message_mask = torch.cat([message_mask, torch.zeros(1, pad_messages)], dim=1)

    with torch.no_grad():
        unpadded = encoder.forward_padded(
            node_features,
            node_mask,
            edge_index,
            edge_mask,
            own_index,
            message_features,
            message_mask,
            congestion,
        )
        padded = encoder.forward_padded(
            padded_node_features,
            padded_node_mask,
            padded_edge_index,
            padded_edge_mask,
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
    edge_index = torch.zeros(batch_size, 3, 2, dtype=torch.long)
    edge_mask = torch.zeros(batch_size, 3)
    own_index = torch.zeros(batch_size, dtype=torch.long)
    message_features = torch.randn(batch_size, m, MESSAGE_FEATURE_DIM)
    message_mask = torch.ones(batch_size, m)
    congestion = torch.zeros(batch_size)

    z = encoder.forward_padded(
        node_features,
        node_mask,
        edge_index,
        edge_mask,
        own_index,
        message_features,
        message_mask,
        congestion,
    )
    assert z.shape == (batch_size, encoder.output_dim)


def _dense_forward_padded(
    encoder: GNNEncoder,
    node_features: torch.Tensor,
    node_mask: torch.Tensor,
    adjacency: torch.Tensor,
    own_index: torch.Tensor,
    message_features: torch.Tensor,
    message_mask: torch.Tensor,
    congestion: torch.Tensor,
) -> torch.Tensor:
    """The reference: the dense N x N formula forward_padded used before the
    edge list, phi_r on every vertex pair, weighted by `adjacency`."""
    h = node_features
    batch_size, max_nodes, _ = h.shape
    for phi_r, psi_r in zip(encoder.phi, encoder.psi):
        own = h.unsqueeze(2).expand(batch_size, max_nodes, max_nodes, -1)
        other = h.unsqueeze(1).expand(batch_size, max_nodes, max_nodes, -1)
        pairwise_messages = phi_r(torch.cat([own, other], dim=-1))
        weight = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        summed = (pairwise_messages * weight.unsqueeze(-1)).sum(dim=2)
        degree = weight.sum(dim=2, keepdim=True).clamp(min=1.0)
        h = psi_r(torch.cat([h, summed / degree], dim=-1))
    own_embedding = h[torch.arange(batch_size), own_index]
    message_count = message_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    aggregated = (message_features * message_mask.unsqueeze(-1)).sum(dim=1) / message_count
    return torch.cat([own_embedding, aggregated, congestion.unsqueeze(-1)], dim=-1)


def _adjacency_from_edges(
    edge_index: torch.Tensor, edge_mask: torch.Tensor, max_nodes: int
) -> torch.Tensor:
    batch_size = edge_index.shape[0]
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    for b in range(batch_size):
        for (a, c), valid in zip(edge_index[b].tolist(), edge_mask[b].tolist()):
            if valid:
                adjacency[b, a, c] = 1.0
                adjacency[b, c, a] = 1.0
    return adjacency


def _random_padded_batch(generator: torch.Generator, batch_size: int, n: int, e_max: int):
    """Random graphs among the valid vertices of each sample, with padded
    vertices and padded edge rows."""
    node_features = torch.randn(batch_size, n, NODE_FEATURE_DIM, generator=generator)
    node_mask = torch.ones(batch_size, n)
    edge_index = torch.zeros(batch_size, e_max, 2, dtype=torch.long)
    edge_mask = torch.zeros(batch_size, e_max)
    for b in range(batch_size):
        num_valid = int(torch.randint(2, n + 1, (1,), generator=generator))
        node_mask[b, num_valid:] = 0.0
        pairs = [(a, c) for a in range(num_valid) for c in range(a + 1, num_valid)]
        order = torch.randperm(len(pairs), generator=generator).tolist()
        chosen = [pairs[i] for i in order[: int(torch.randint(0, e_max + 1, (1,)))]]
        for k, (a, c) in enumerate(chosen[:e_max]):
            # either order, since a pair is undirected
            edge_index[b, k] = torch.tensor([a, c] if k % 2 == 0 else [c, a])
            edge_mask[b, k] = 1.0
    own_index = torch.zeros(batch_size, dtype=torch.long)
    message_features = torch.randn(batch_size, 2, MESSAGE_FEATURE_DIM, generator=generator)
    message_mask = torch.tensor([[1.0, 0.0]] * batch_size)
    congestion = torch.rand(batch_size, generator=generator)
    return (
        node_features,
        node_mask,
        edge_index,
        edge_mask,
        own_index,
        message_features,
        message_mask,
        congestion,
    )


@pytest.mark.parametrize("num_rounds", [1, 2, 3])
def test_forward_padded_matches_the_dense_pair_formula_on_random_graphs(num_rounds):
    generator = torch.Generator().manual_seed(7)
    torch.manual_seed(num_rounds)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=num_rounds, hidden_width=8))
    encoder.eval()
    for _ in range(5):
        tensors = _random_padded_batch(generator, batch_size=4, n=7, e_max=9)
        node_features, node_mask, edge_index, edge_mask, *rest = tensors
        adjacency = _adjacency_from_edges(edge_index, edge_mask, max_nodes=7)
        with torch.no_grad():
            sparse = encoder.forward_padded(*tensors)
            dense = _dense_forward_padded(encoder, node_features, node_mask, adjacency, *rest)
        assert torch.allclose(sparse, dense, atol=1e-5)


def test_forward_padded_edge_touching_a_padded_vertex_contributes_nothing():
    generator = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()
    tensors = _random_padded_batch(generator, batch_size=3, n=6, e_max=6)
    node_features, node_mask, edge_index, edge_mask, *rest = tensors
    node_mask[:, -1] = 0.0  # the last vertex is padding in every sample
    edge_mask = edge_mask * (edge_index < 5).all(dim=-1).float()

    stray_index = edge_index.clone()
    stray_mask = edge_mask.clone()
    free = (edge_mask == 0).nonzero()
    for b, k in free.tolist():
        stray_index[b, k] = torch.tensor([0, 5])  # valid mask, padded endpoint
        stray_mask[b, k] = 1.0

    with torch.no_grad():
        clean = encoder.forward_padded(node_features, node_mask, edge_index, edge_mask, *rest)
        stray = encoder.forward_padded(node_features, node_mask, stray_index, stray_mask, *rest)
    assert torch.allclose(clean, stray, atol=1e-6)


def test_forward_padded_vertex_without_neighbours_gets_a_zero_message():
    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()
    node_features = torch.randn(1, 3, NODE_FEATURE_DIM)
    edge_index = torch.zeros(1, 4, 2, dtype=torch.long)
    edge_mask = torch.zeros(1, 4)  # no edges at all
    own_index = torch.tensor([1])
    message_features = torch.zeros(1, 1, MESSAGE_FEATURE_DIM)
    message_mask = torch.zeros(1, 1)
    congestion = torch.zeros(1)

    with torch.no_grad():
        z = encoder.forward_padded(
            node_features,
            torch.ones(1, 3),
            edge_index,
            edge_mask,
            own_index,
            message_features,
            message_mask,
            congestion,
        )
        h = node_features[0, 1].unsqueeze(0)
        for psi_r in encoder.psi:
            h = psi_r(torch.cat([h, torch.zeros(1, 8)], dim=-1))
    assert torch.allclose(z[0, :8], h[0], atol=1e-6)


def test_forward_padded_zero_rounds_returns_the_raw_node_features():
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    node_features = torch.randn(2, 3, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[[0, 1]], [[1, 2]]])
    z = encoder.forward_padded(
        node_features,
        torch.ones(2, 3),
        edge_index,
        torch.ones(2, 1),
        torch.tensor([2, 0]),
        torch.zeros(2, 1, MESSAGE_FEATURE_DIM),
        torch.zeros(2, 1),
        torch.zeros(2),
    )
    assert torch.equal(z[0, :NODE_FEATURE_DIM], node_features[0, 2])
    assert torch.equal(z[1, :NODE_FEATURE_DIM], node_features[1, 0])


def test_forward_padded_edge_listed_once_counts_for_both_endpoints():
    """A pair is listed once, in either order, and both of its vertices
    receive a message from the other."""
    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=1, hidden_width=8))
    encoder.eval()
    node_features = torch.randn(2, 2, NODE_FEATURE_DIM)
    node_features[1] = node_features[0]
    node_mask = torch.ones(2, 2)
    edge_index = torch.tensor([[[0, 1]], [[1, 0]]])  # the same pair, listed in either order
    no_messages = (torch.zeros(2, 1, MESSAGE_FEATURE_DIM), torch.zeros(2, 1), torch.zeros(2))

    with torch.no_grad():
        for own in (0, 1):
            own_index = torch.tensor([own, own])
            linked = encoder.forward_padded(
                node_features, node_mask, edge_index, torch.ones(2, 1), own_index, *no_messages
            )
            alone = encoder.forward_padded(
                node_features, node_mask, edge_index, torch.zeros(2, 1), own_index, *no_messages
            )
            assert not torch.allclose(linked[0], alone[0])  # the neighbour was heard
            assert torch.allclose(linked[0], linked[1], atol=1e-6)  # listing order is irrelevant


def test_forward_padded_on_an_encoded_observation_matches_the_unbatched_forward():
    """The full path, with node and edge padding from encode_observation."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    mask = legality_mask(graph, vertex=2)
    encoded = encode_observation(
        graph, 2, observation, mask, LocalSubgraphEncodingConfig(max_local_nodes=9, max_messages=2)
    )

    torch.manual_seed(0)
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    encoder.eval()
    with torch.no_grad():
        unbatched = encoder(graph, agent_location=2, observation=observation)
        batched = encoder.forward_padded(
            torch.as_tensor(encoded.node_features).unsqueeze(0),
            torch.as_tensor(encoded.node_mask).unsqueeze(0),
            torch.as_tensor(encoded.edge_index).unsqueeze(0),
            torch.as_tensor(encoded.edge_mask).unsqueeze(0),
            torch.tensor([encoded.own_index]),
            torch.as_tensor(encoded.message_features).unsqueeze(0),
            torch.as_tensor(encoded.message_mask).unsqueeze(0),
            torch.tensor([encoded.congestion]),
        )
    assert torch.allclose(unbatched, batched[0], atol=1e-6)
