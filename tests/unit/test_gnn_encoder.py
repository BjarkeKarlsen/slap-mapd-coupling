"""Unit tests for slap_mapd_coupling.models.gnn_encoder."""

import numpy as np
import pytest
import torch

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import (
    build_observation,
    communication_neighbours,
)
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.models.gnn_encoder import (
    NODE_FEATURE_DIM,
    SENDER_KEYS,
    GNNEncoder,
    GNNEncoderConfig,
)
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
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




def _encoder(num_rounds: int, communication: bool, hidden_width: int = 8) -> GNNEncoder:
    config = GNNEncoderConfig(
        num_rounds=num_rounds, hidden_width=hidden_width, communication=communication
    )
    return GNNEncoder(config)


def _own_vector(observation, vertex: int) -> torch.Tensor:
    f = observation.features[vertex]
    return torch.tensor(
        [
            float(f.storage),
            float(f.delivery),
            float(f.endpoint),
            float(f.transit),
            f.eta,
            float(f.occupied),
        ]
    )


@pytest.mark.parametrize(
    "num_rounds, communication, expected",
    [
        (0, False, NODE_FEATURE_DIM + 1),
        (0, True, 2 * NODE_FEATURE_DIM + 1),
        (2, False, 8 + 1),
        (2, True, 2 * 8 + 1),
    ],
)
def test_output_dim_is_the_embedding_plus_the_messages_when_on_plus_congestion(
    num_rounds, communication, expected
):
    encoder = _encoder(num_rounds, communication)
    assert encoder.output_dim == expected


def test_a_message_has_the_width_of_the_encoders_own_embedding():
    assert _encoder(0, True).embedding_dim == NODE_FEATURE_DIM
    assert _encoder(2, True).embedding_dim == 8


def test_forward_output_shape_matches_output_dim():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3, communication=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    sender = (4, build_observation(graph, fleet, [], agent_id=2, t=0, config=config))

    for communication, senders in ((False, ()), (True, (sender,)), (True, ())):
        encoder = _encoder(2, communication)
        z = encoder(graph, 2, observation, senders)
        assert z.shape == (encoder.output_dim,)


def test_forward_rejects_senders_when_communication_is_off():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    other = build_observation(graph, fleet, [], agent_id=2, t=0, config=config)

    with pytest.raises(ValueError, match="communication"):
        _encoder(1, False)(graph, 2, observation, [(4, other)])


def test_zero_rounds_returns_raw_features_then_the_mean_of_the_senders_raw_features():
    """At L = 0 a message is the sender's own raw vertex features, since the
    embedding is not propagated (sec:method:model)."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3, 3: 4})
    config = _config(observation_depth=3, congestion_sensitive=True, communication=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    other_2 = build_observation(graph, fleet, [], agent_id=2, t=0, config=config)
    other_3 = build_observation(graph, fleet, [], agent_id=3, t=0, config=config)

    encoder = _encoder(0, True)
    z = encoder(graph, 2, observation, [(3, other_2), (4, other_3)])

    expected_mean = (_own_vector(other_2, 3) + _own_vector(other_3, 4)) / 2
    assert torch.allclose(z[:NODE_FEATURE_DIM], _own_vector(observation, 2))
    assert torch.allclose(z[NODE_FEATURE_DIM:-1], expected_mean)
    assert z[-1].item() == torch.tensor(observation.congestion).item()  # float32


def test_zero_rounds_with_communication_off_has_no_message_term():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3, congestion_sensitive=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    z = _encoder(0, False)(graph, 2, observation)

    assert z.shape == (NODE_FEATURE_DIM + 1,)
    assert torch.allclose(z[:-1], _own_vector(observation, 2))
    assert z[-1].item() == torch.tensor(observation.congestion).item()  # float32


def test_a_receiver_with_no_senders_gets_a_zero_message():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=3, communication=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    encoder = _encoder(2, True)
    z = encoder(graph, 2, observation)

    assert torch.equal(z[8:16], torch.zeros(8))


def test_congestion_is_read_even_when_not_sensitive():
    """delta_i(t) is always one of o_i(t)'s three pieces (sec:pf:observations);
    congestion_sensitive only decides whether the policy reacts to it."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=3, congestion_sensitive=False)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert observation.congestion > 0.0

    z = _encoder(0, False)(graph, 2, observation)
    assert z[-1].item() == pytest.approx(observation.congestion)


def test_gradients_flow_through_message_passing_rounds():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)

    encoder = _encoder(2, False)
    z = encoder(graph, 2, observation)
    z.sum().backward()

    assert all(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())


def test_forward_is_deterministic_for_fixed_weights_and_input():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3, communication=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    sender = (4, build_observation(graph, fleet, [], agent_id=2, t=0, config=config))

    torch.manual_seed(0)
    encoder = _encoder(2, True)
    encoder.eval()
    with torch.no_grad():
        first = encoder(graph, 2, observation, [sender])
        second = encoder(graph, 2, observation, [sender])
    assert torch.equal(first, second)


# A scenario with five robots on a line, where robot 2 serves a task, so the
# views differ in their distance labels. Robots 1 and 5 hear two senders,
# robots 2 and 3 hear three and robot 4, far from the others, hears nobody.
_LOCATIONS = {1: 0, 2: 2, 3: 3, 4: 9, 5: 5}
_MAX_MESSAGES = 3


def _scenario(communication: bool):
    graph = _line_graph(10)
    fleet = _fleet(_LOCATIONS)
    task = Task(task_id=2, release_time=0, pickup_vertex=9, delivery_vertex=0, sku="sku-1").assign(
        agent_id=2, t=0
    )
    config = _config(observation_depth=3, communication=communication)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=9, max_messages=_MAX_MESSAGES)
    encoded = encode_all_observations(
        graph, fleet, [task], 0, config, encoding_config, max_out_degree(graph)
    )
    return graph, fleet, [task], config, encoded


def _as_tensors(items) -> dict[str, torch.Tensor]:
    """The encoded observations stacked into one batch, one row per robot."""
    rows = [item.as_rllib_dict()["observations"] for item in items]
    return {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}


def _forward_padded(encoder: GNNEncoder, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    senders = {key: tensors[key] for key in SENDER_KEYS if key in tensors}
    return encoder.forward_padded(
        tensors["node_features"],
        tensors["node_mask"],
        tensors["edge_index"],
        tensors["edge_mask"],
        tensors["own_index"],
        tensors["congestion"],
        **senders,
    )


def _heard_senders(scenario, agent_id: int):
    """The vertex and the own observation of each robot `agent_id` hears,
    built from scratch."""
    graph, fleet, tasks, config, _ = scenario
    locations = fleet.locations()
    senders = []
    if config.communication:
        heard = communication_neighbours(graph, fleet, agent_id, config.observation_depth)
        for other in heard[:_MAX_MESSAGES]:
            view = build_observation(graph, fleet, tasks, other, 0, config)
            senders.append((locations[other], view))
    return senders


def _oracle(encoder: GNNEncoder, scenario, agent_id: int) -> torch.Tensor:
    """The unbatched forward of one robot, with the observations of the
    robots it hears built from scratch."""
    graph, fleet, tasks, config, _ = scenario
    observation = build_observation(graph, fleet, tasks, agent_id, 0, config)
    return encoder(
        graph, fleet.locations()[agent_id], observation, _heard_senders(scenario, agent_id)
    )


@pytest.mark.parametrize("num_rounds", [0, 2])
@pytest.mark.parametrize("communication", [False, True])
def test_forward_padded_matches_the_unbatched_forward_for_every_robot(num_rounds, communication):
    """The padded path, with node, edge and sender padding from
    encode_all_observations, equals the unbatched forward of each robot."""
    scenario = _scenario(communication)
    encoded = scenario[-1]

    torch.manual_seed(1)
    encoder = _encoder(num_rounds, communication)
    encoder.eval()
    with torch.no_grad():
        batched = _forward_padded(encoder, _as_tensors(encoded.values()))
        for row, agent_id in enumerate(encoded):
            assert torch.allclose(_oracle(encoder, scenario, agent_id), batched[row], atol=1e-5)


def test_the_scenario_has_robots_with_no_two_and_three_senders():
    encoded = _scenario(communication=True)[-1]
    counts = {agent_id: int(item.sender_mask.sum()) for agent_id, item in encoded.items()}
    assert counts == {1: 2, 2: 3, 3: 3, 4: 0, 5: 2}


def test_a_batch_row_does_not_depend_on_the_other_rows():
    scenario = _scenario(communication=True)
    encoded = scenario[-1]
    torch.manual_seed(2)
    encoder = _encoder(2, True)
    encoder.eval()
    with torch.no_grad():
        together = _forward_padded(encoder, _as_tensors(encoded.values()))
        alone = _forward_padded(encoder, _as_tensors([encoded[3]]))
    assert torch.allclose(together[2], alone[0], atol=1e-5)


def _random_views(generator: torch.Generator, rows: int, n: int, e_max: int):
    """Random padded views, with random graphs among the valid vertices of
    each row, padded vertices and padded edge rows. The own vertex is 0."""
    node_features = torch.randn(rows, n, NODE_FEATURE_DIM, generator=generator)
    node_mask = torch.ones(rows, n)
    edge_index = torch.zeros(rows, e_max, 2, dtype=torch.long)
    edge_mask = torch.zeros(rows, e_max)
    for b in range(rows):
        num_valid = int(torch.randint(2, n + 1, (1,), generator=generator))
        node_mask[b, num_valid:] = 0.0
        pairs = [(a, c) for a in range(num_valid) for c in range(a + 1, num_valid)]
        order = torch.randperm(len(pairs), generator=generator).tolist()
        chosen = [pairs[i] for i in order[: int(torch.randint(0, e_max + 1, (1,)))]]
        for k, (a, c) in enumerate(chosen[:e_max]):
            # either order, since a pair is undirected
            edge_index[b, k] = torch.tensor([a, c] if k % 2 == 0 else [c, a])
            edge_mask[b, k] = 1.0
    own_index = torch.zeros(rows, dtype=torch.long)
    return node_features, node_mask, edge_index, edge_mask, own_index


def _random_batch(generator, batch_size: int, n: int, e_max: int, m: int, communication: bool):
    """A random padded batch as the keyword arguments of forward_padded, and
    the same rows as separate views for the dense reference."""
    own = _random_views(generator, batch_size, n, e_max)
    batch = dict(
        node_features=own[0],
        node_mask=own[1],
        edge_index=own[2],
        edge_mask=own[3],
        own_index=own[4],
        congestion=torch.rand(batch_size, generator=generator),
    )
    if communication:
        views = _random_views(generator, batch_size * m, n, e_max)
        mask = (torch.rand(batch_size, m, generator=generator) < 0.6).float()
        batch.update(
            sender_node_features=views[0].reshape(batch_size, m, n, NODE_FEATURE_DIM),
            sender_node_mask=views[1].reshape(batch_size, m, n),
            sender_edge_index=views[2].reshape(batch_size, m, e_max, 2),
            sender_edge_mask=views[3].reshape(batch_size, m, e_max),
            sender_own_index=views[4].reshape(batch_size, m),
            sender_mask=mask,
        )
    return batch


def _dense_embedding(encoder: GNNEncoder, node_features, node_mask, edge_index, edge_mask, own):
    """The reference: the dense N x N formula forward_padded used before the
    edge list, phi_r on every vertex pair, weighted by an adjacency matrix."""
    h = node_features
    batch_size, max_nodes, _ = h.shape
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes)
    for b in range(batch_size):
        for (a, c), valid in zip(edge_index[b].tolist(), edge_mask[b].tolist()):
            if valid:
                adjacency[b, a, c] = 1.0
                adjacency[b, c, a] = 1.0
    for phi_r, psi_r in zip(encoder.phi, encoder.psi):
        own_h = h.unsqueeze(2).expand(batch_size, max_nodes, max_nodes, -1)
        other_h = h.unsqueeze(1).expand(batch_size, max_nodes, max_nodes, -1)
        pairwise_messages = phi_r(torch.cat([own_h, other_h], dim=-1))
        weight = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        summed = (pairwise_messages * weight.unsqueeze(-1)).sum(dim=2)
        degree = weight.sum(dim=2, keepdim=True).clamp(min=1.0)
        h = psi_r(torch.cat([h, summed / degree], dim=-1))
    return h[torch.arange(batch_size), own]


def _dense_forward_padded(encoder: GNNEncoder, batch: dict) -> torch.Tensor:
    """The dense reference for z_i, with the mean over the masked senders."""
    own = _dense_embedding(
        encoder,
        batch["node_features"],
        batch["node_mask"],
        batch["edge_index"],
        batch["edge_mask"],
        batch["own_index"],
    )
    parts = [own]
    if encoder.config.communication:
        batch_size, m = batch["sender_mask"].shape
        embedded = _dense_embedding(
            encoder,
            batch["sender_node_features"].flatten(0, 1),
            batch["sender_node_mask"].flatten(0, 1),
            batch["sender_edge_index"].flatten(0, 1),
            batch["sender_edge_mask"].flatten(0, 1),
            batch["sender_own_index"].flatten(),
        ).reshape(batch_size, m, -1)
        count = batch["sender_mask"].sum(dim=1, keepdim=True).clamp(min=1.0)
        parts.append((embedded * batch["sender_mask"].unsqueeze(-1)).sum(dim=1) / count)
    parts.append(batch["congestion"].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


@pytest.mark.parametrize("communication", [False, True])
@pytest.mark.parametrize("num_rounds", [1, 2, 3])
def test_forward_padded_matches_the_dense_pair_formula_on_random_graphs(num_rounds, communication):
    generator = torch.Generator().manual_seed(7)
    torch.manual_seed(num_rounds)
    encoder = _encoder(num_rounds, communication)
    encoder.eval()
    for _ in range(5):
        batch = _random_batch(generator, 4, n=7, e_max=9, m=3, communication=communication)
        with torch.no_grad():
            sparse = encoder.forward_padded(**batch)
            dense = _dense_forward_padded(encoder, batch)
        assert torch.allclose(sparse, dense, atol=1e-5)


def _with_padded_vertices_and_edges(batch: dict, pad_nodes: int, pad_edges: int) -> dict:
    """Extra padded vertices and edge rows on the own views and on every sender
    view. The padded edges point at real vertices on purpose, so only the mask
    silences them."""
    padded = dict(batch)
    for prefix, axis in (("", 1), ("sender_", 2)):
        node_features = batch[prefix + "node_features"]
        node_mask = batch[prefix + "node_mask"]
        edge_index = batch[prefix + "edge_index"]
        edge_mask = batch[prefix + "edge_mask"]
        node_shape = (*node_features.shape[:axis], pad_nodes, NODE_FEATURE_DIM)
        mask_shape = (*node_mask.shape[:axis], pad_nodes)
        edge_shape = (*edge_index.shape[:axis], pad_edges, 2)
        padded[prefix + "node_features"] = torch.cat(
            [node_features, torch.randn(node_shape)], dim=axis
        )
        padded[prefix + "node_mask"] = torch.cat([node_mask, torch.zeros(mask_shape)], dim=axis)
        padded[prefix + "edge_index"] = torch.cat(
            [edge_index, torch.ones(edge_shape, dtype=torch.long)], dim=axis
        )
        padded[prefix + "edge_mask"] = torch.cat(
            [edge_mask, torch.zeros(edge_shape[:-1])], dim=axis
        )
    return padded


def _with_garbage_senders(batch: dict, count: int) -> dict:
    """`count` extra sender slots with random content that sender_mask silences."""
    padded = dict(batch)
    for key in SENDER_KEYS:
        value = batch[key]
        shape = (value.shape[0], count, *value.shape[2:])
        if value.dtype == torch.long:
            extra = torch.randint(0, 2, shape)
        else:
            extra = torch.randn(shape).abs()
        padded[key] = torch.cat([value, extra], dim=1)
    padded["sender_mask"] = torch.cat(
        [batch["sender_mask"], torch.zeros(batch["sender_mask"].shape[0], count)], dim=1
    )
    return padded


def test_padded_vertices_edges_and_senders_do_not_change_z():
    generator = torch.Generator().manual_seed(5)
    torch.manual_seed(0)
    encoder = _encoder(2, True)
    encoder.eval()
    batch = _random_batch(generator, 3, n=6, e_max=8, m=2, communication=True)

    padded = _with_garbage_senders(_with_padded_vertices_and_edges(batch, 3, 4), 2)

    with torch.no_grad():
        plain = encoder.forward_padded(**batch)
        with_padding = encoder.forward_padded(**padded)
    assert torch.allclose(plain, with_padding, atol=1e-5)


def _swap_sender_slots(batch: dict, first: int, second: int) -> dict:
    swapped = dict(batch)
    for key in SENDER_KEYS:
        value = batch[key].clone()
        value[:, [first, second]] = value[:, [second, first]]
        swapped[key] = value
    return swapped


def test_the_order_of_the_senders_does_not_change_z():
    generator = torch.Generator().manual_seed(11)
    torch.manual_seed(0)
    encoder = _encoder(2, True)
    encoder.eval()
    batch = _random_batch(generator, 4, n=6, e_max=8, m=3, communication=True)

    with torch.no_grad():
        original = encoder.forward_padded(**batch)
        swapped = encoder.forward_padded(**_swap_sender_slots(batch, 0, 2))
    assert torch.allclose(original, swapped, atol=1e-5)


def test_a_message_depends_only_on_its_sender():
    """Two receivers with different views of their own that hear the same
    sender get the same message, and it is that sender's own embedding."""
    generator = torch.Generator().manual_seed(13)
    torch.manual_seed(0)
    encoder = _encoder(2, True)
    encoder.eval()
    n, e_max = 6, 8
    receivers = _random_views(generator, 2, n, e_max)
    sender = _random_views(generator, 1, n, e_max)

    def repeat(tensor):
        """[1, ...] -> [2, 1, ...], the same sender block for both receivers."""
        return tensor.unsqueeze(0).expand(2, *tensor.shape).clone()

    batch = dict(
        node_features=receivers[0],
        node_mask=receivers[1],
        edge_index=receivers[2],
        edge_mask=receivers[3],
        own_index=receivers[4],
        congestion=torch.zeros(2),
        sender_node_features=repeat(sender[0]),
        sender_node_mask=repeat(sender[1]),
        sender_edge_index=repeat(sender[2]),
        sender_edge_mask=repeat(sender[3]),
        sender_own_index=repeat(sender[4]),
        sender_mask=torch.ones(2, 1),
    )

    with torch.no_grad():
        z = encoder.forward_padded(**batch)
        # the sender's own embedding, from its own view as a receiver with nobody to hear
        alone = dict(
            node_features=sender[0],
            node_mask=sender[1],
            edge_index=sender[2],
            edge_mask=sender[3],
            own_index=sender[4],
            congestion=torch.zeros(1),
        )
        own = _embedding_of(encoder, alone)
    message = z[:, encoder.embedding_dim : 2 * encoder.embedding_dim]
    assert not torch.allclose(z[0, : encoder.embedding_dim], z[1, : encoder.embedding_dim])
    assert torch.allclose(message[0], message[1], atol=1e-6)
    assert torch.allclose(message[0], own[0], atol=1e-6)


def _embedding_of(encoder: GNNEncoder, views: dict) -> torch.Tensor:
    """h^(L) at the own vertex of each view, read from a communication-off
    encoder that shares the weights."""
    off = GNNEncoder(
        GNNEncoderConfig(
            num_rounds=encoder.config.num_rounds,
            hidden_width=encoder.config.hidden_width,
            communication=False,
        )
    )
    off.load_state_dict(encoder.state_dict())
    return off.forward_padded(**views)[:, :-1]


def test_a_loss_on_the_message_reaches_the_encoder_through_the_sender_path():
    """The receiver's loss reaches the sender's embedding through the shared
    weights (sec:method:model, trained end to end). The loss reads only the
    message part of z, so a gradient can only come through the sender path."""
    generator = torch.Generator().manual_seed(17)
    torch.manual_seed(0)
    encoder = _encoder(2, True)
    batch = _random_batch(generator, 4, n=6, e_max=8, m=2, communication=True)
    batch["sender_mask"] = torch.ones(4, 2)

    z = encoder.forward_padded(**batch)
    d = encoder.embedding_dim
    z[:, d : 2 * d].pow(2).sum().backward()

    assert all(p.grad is not None and torch.any(p.grad != 0) for p in encoder.parameters())


def test_no_gradient_reaches_the_encoder_from_the_message_without_senders():
    generator = torch.Generator().manual_seed(19)
    torch.manual_seed(0)
    encoder = _encoder(2, True)
    batch = _random_batch(generator, 4, n=6, e_max=8, m=2, communication=True)
    batch["sender_mask"] = torch.zeros(4, 2)

    z = encoder.forward_padded(**batch)
    d = encoder.embedding_dim
    z[:, d : 2 * d].pow(2).sum().backward()

    assert all(p.grad is None or not torch.any(p.grad != 0) for p in encoder.parameters())


def test_forward_padded_requires_the_sender_tensors_exactly_when_communication_is_on():
    generator = torch.Generator().manual_seed(23)
    off_batch = _random_batch(generator, 2, n=5, e_max=6, m=2, communication=False)
    on_batch = _random_batch(generator, 2, n=5, e_max=6, m=2, communication=True)

    with pytest.raises(ValueError, match="sender"):
        _encoder(1, True).forward_padded(**off_batch)
    with pytest.raises(ValueError, match="sender"):
        _encoder(1, False).forward_padded(**on_batch)


def test_forward_padded_output_shape_is_batched():
    generator = torch.Generator().manual_seed(29)
    for communication in (False, True):
        encoder = _encoder(1, communication)
        batch = _random_batch(generator, 3, n=4, e_max=3, m=2, communication=communication)
        assert encoder.forward_padded(**batch).shape == (3, encoder.output_dim)


def test_forward_padded_edge_touching_a_padded_vertex_contributes_nothing():
    generator = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    encoder = _encoder(2, False)
    encoder.eval()
    batch = _random_batch(generator, 3, n=6, e_max=6, m=1, communication=False)
    batch["node_mask"][:, -1] = 0.0  # the last vertex is padding in every sample
    batch["edge_mask"] = batch["edge_mask"] * (batch["edge_index"] < 5).all(dim=-1).float()

    stray = dict(batch)
    stray["edge_index"] = batch["edge_index"].clone()
    stray["edge_mask"] = batch["edge_mask"].clone()
    for b, k in (batch["edge_mask"] == 0).nonzero().tolist():
        stray["edge_index"][b, k] = torch.tensor([0, 5])  # valid mask, padded endpoint
        stray["edge_mask"][b, k] = 1.0

    with torch.no_grad():
        clean = encoder.forward_padded(**batch)
        with_stray = encoder.forward_padded(**stray)
    assert torch.allclose(clean, with_stray, atol=1e-6)


def test_forward_padded_vertex_without_neighbours_gets_a_zero_message():
    torch.manual_seed(0)
    encoder = _encoder(2, False)
    encoder.eval()
    node_features = torch.randn(1, 3, NODE_FEATURE_DIM)
    edge_index = torch.zeros(1, 4, 2, dtype=torch.long)
    edge_mask = torch.zeros(1, 4)  # no edges at all

    with torch.no_grad():
        z = encoder.forward_padded(
            node_features,
            torch.ones(1, 3),
            edge_index,
            edge_mask,
            torch.tensor([1]),
            torch.zeros(1),
        )
        h = node_features[0, 1].unsqueeze(0)
        for psi_r in encoder.psi:
            h = psi_r(torch.cat([h, torch.zeros(1, 8)], dim=-1))
    assert torch.allclose(z[0, :8], h[0], atol=1e-6)


def test_forward_padded_zero_rounds_returns_the_raw_node_features():
    encoder = _encoder(0, False)
    node_features = torch.randn(2, 3, NODE_FEATURE_DIM)
    edge_index = torch.tensor([[[0, 1]], [[1, 2]]])
    z = encoder.forward_padded(
        node_features,
        torch.ones(2, 3),
        edge_index,
        torch.ones(2, 1),
        torch.tensor([2, 0]),
        torch.zeros(2),
    )
    assert torch.equal(z[0, :NODE_FEATURE_DIM], node_features[0, 2])
    assert torch.equal(z[1, :NODE_FEATURE_DIM], node_features[1, 0])


def test_forward_padded_zero_rounds_messages_are_the_senders_raw_features():
    encoder = _encoder(0, True)
    node_features = torch.randn(1, 3, NODE_FEATURE_DIM)
    sender_features = torch.randn(1, 2, 3, NODE_FEATURE_DIM)
    z = encoder.forward_padded(
        node_features,
        torch.ones(1, 3),
        torch.zeros(1, 1, 2, dtype=torch.long),
        torch.zeros(1, 1),
        torch.tensor([0]),
        torch.zeros(1),
        sender_node_features=sender_features,
        sender_node_mask=torch.ones(1, 2, 3),
        sender_edge_index=torch.zeros(1, 2, 1, 2, dtype=torch.long),
        sender_edge_mask=torch.zeros(1, 2, 1),
        sender_own_index=torch.tensor([[1, 2]]),
        sender_mask=torch.ones(1, 2),
    )
    expected = (sender_features[0, 0, 1] + sender_features[0, 1, 2]) / 2
    assert torch.allclose(z[0, NODE_FEATURE_DIM : 2 * NODE_FEATURE_DIM], expected)


def test_forward_padded_edge_listed_once_counts_for_both_endpoints():
    """A pair is listed once, in either order, and both of its vertices
    receive a message from the other."""
    torch.manual_seed(0)
    encoder = _encoder(1, False)
    encoder.eval()
    node_features = torch.randn(2, 2, NODE_FEATURE_DIM)
    node_features[1] = node_features[0]
    node_mask = torch.ones(2, 2)
    edge_index = torch.tensor([[[0, 1]], [[1, 0]]])  # the same pair, listed in either order

    with torch.no_grad():
        for own in (0, 1):
            own_index = torch.tensor([own, own])
            args = (node_features, node_mask, edge_index)
            linked = encoder.forward_padded(*args, torch.ones(2, 1), own_index, torch.zeros(2))
            alone = encoder.forward_padded(*args, torch.zeros(2, 1), own_index, torch.zeros(2))
            assert not torch.allclose(linked[0], alone[0])  # the neighbour was heard
            assert torch.allclose(linked[0], linked[1], atol=1e-6)  # listing order is irrelevant


def _encode_padded(encoder: GNNEncoder, tensors: dict[str, torch.Tensor]):
    senders = {key: tensors[key] for key in SENDER_KEYS if key in tensors}
    return encoder.encode_padded(
        tensors["node_features"],
        tensors["node_mask"],
        tensors["edge_index"],
        tensors["edge_mask"],
        tensors["own_index"],
        tensors["congestion"],
        **senders,
    )


@pytest.mark.parametrize("communication", [False, True])
def test_encode_padded_returns_what_forward_padded_returns_and_the_per_vertex_embeddings(
    communication,
):
    generator = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    encoder = _encoder(2, communication)
    encoder.eval()
    batch = _random_batch(generator, 4, n=7, e_max=9, m=3, communication=communication)

    with torch.no_grad():
        out = encoder.encode_padded(**batch)
        z = encoder.forward_padded(**batch)

    assert torch.equal(out.z, z)
    assert out.node_embeddings.shape == (4, 7, encoder.embedding_dim)
    if communication:
        assert out.messages is not None
        assert out.messages.shape == (4, 3, encoder.embedding_dim)
    else:
        assert out.messages is None
    # z holds the embedding at the robot's own vertex, so the two agree
    own = out.node_embeddings[torch.arange(4), batch["own_index"]]
    assert torch.equal(z[:, : encoder.embedding_dim], own)


@pytest.mark.parametrize("num_rounds", [0, 2])
@pytest.mark.parametrize("communication", [False, True])
def test_encode_padded_move_inputs_match_the_unbatched_embed_moves(num_rounds, communication):
    """h_w gathered at move_target_index, and the message gathered at
    move_sender_index and masked, equal what the unbatched oracle computes
    for each move slot from scratch."""
    scenario = _scenario(communication)
    graph, fleet, tasks, config, encoded = scenario
    torch.manual_seed(4)
    encoder = _encoder(num_rounds, communication)
    encoder.eval()
    with torch.no_grad():
        tensors = _as_tensors(encoded.values())
        out = _encode_padded(encoder, tensors)

    found_a_sender = False
    for row, agent_id in enumerate(encoded):
        observation = build_observation(graph, fleet, tasks, agent_id, 0, config)
        with torch.no_grad():
            h_w, xibar_w = encoder.embed_moves(
                graph,
                fleet.locations()[agent_id],
                observation,
                _heard_senders(scenario, agent_id),
            )
        slots = int(tensors["move_target_mask"][row].sum())
        assert h_w.shape == (slots, encoder.embedding_dim)
        gathered_h = out.node_embeddings[row, tensors["move_target_index"][row]][:slots]
        assert torch.allclose(gathered_h, h_w, atol=1e-5)

        if not communication:
            assert xibar_w is None and out.messages is None
            continue
        mask = tensors["move_sender_mask"][row]
        gathered = out.messages[row, tensors["move_sender_index"][row]] * mask.unsqueeze(-1)
        assert torch.allclose(gathered[:slots], xibar_w, atol=1e-5)
        assert not gathered[slots:].any()
        found_a_sender = found_a_sender or bool(mask.any())
    assert found_a_sender


def test_embed_moves_rejects_senders_when_communication_is_off():
    scenario = _scenario(communication=True)
    graph, fleet, tasks, config, _ = scenario
    observation = build_observation(graph, fleet, tasks, 1, 0, config)
    encoder = _encoder(2, communication=False)

    with pytest.raises(ValueError, match="communication off"):
        encoder.embed_moves(graph, 0, observation, _heard_senders(scenario, 1))
