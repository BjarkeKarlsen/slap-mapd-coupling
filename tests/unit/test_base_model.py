"""Unit tests for slap_mapd_coupling.models.base_model."""

import numpy as np
import pytest
import torch
from torch import nn

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.environment.observation import (
    build_observation,
    communication_neighbours,
    local_subgraph,
)
from slap_mapd_coupling.models.base_model import (
    MOVE_SENDER_KEYS,
    GNNPolicyValueModel,
    PolicyValueHead,
    PolicyValueHeadConfig,
)
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS, GNNEncoder, GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
)

_ENCODER_KEYS = ("node_features", "node_mask", "edge_index", "edge_mask", "own_index", "congestion")
_SLOT_KEYS = ("move_target_index", *MOVE_SENDER_KEYS)


def _star_graph() -> WarehouseGraph:
    """Vertex 0 in the middle, leaves 1, 2 and 3 at costs 1, 2 and 3, so the
    centre has three move slots and a leaf has one."""
    vertices = {i: Vertex(id=i) for i in range(4)}
    edges = tuple(
        Edge(source=a, target=b, cost=float(leaf))
        for leaf in (3, 1, 2)
        for a, b in ((0, leaf), (leaf, 0))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


# robot 1 in the centre hears robot 2 (distance 1) and robot 3 (distance 3). Robot 2 on
# vertex 1 hears robot 1, and robot 3 on vertex 3 hears robot 1 only.
_LOCATIONS = {1: 0, 2: 1, 3: 3}


def _fleet() -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in _LOCATIONS.items()}
    )


def _config(communication: bool) -> ExperimentConfig:
    return ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=communication,
        num_agents=3,
        arrival_rate=1.0,
        seed=0,
        horizon=50,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )


def _zone_pool(graph: WarehouseGraph, location: int) -> tuple[int, ...]:
    """The field of view's vertices in the zone of `location`, for the zones
    {0, 1} and {2, 3}. The rest of the field of view is the halo."""
    return tuple(v for v in local_subgraph(graph, location, 3) if (v < 2) == (location < 2))


def _batch(
    communication: bool, max_messages: int = 2, pooling: str = "none"
) -> dict[str, torch.Tensor]:
    """The three robots' encoded observations stacked into one batch. With
    pooling on, each robot pools over its zone (_zone_pool)."""
    graph = _star_graph()
    pool_vertices = None
    if pooling != "none":
        pool_vertices = {aid: _zone_pool(graph, loc) for aid, loc in _LOCATIONS.items()}
    encoded = encode_all_observations(
        graph,
        _fleet(),
        [],
        0,
        _config(communication),
        LocalSubgraphEncodingConfig(max_local_nodes=6, max_messages=max_messages),
        d_max=3,
        pooling=pooling,
        pool_vertices=pool_vertices,
    )
    rows = [item.as_rllib_dict()["observations"] for item in encoded.values()]
    return {key: torch.as_tensor(np.stack([row[key] for row in rows])) for key in rows[0]}


def _randomise(module: nn.Module, seed: int = 0) -> None:
    """Weights of order one half, so that different inputs give clearly different logits."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=generator))


def _model(
    num_rounds: int, communication: bool, hidden_width: int = 8, pooling: str = "none"
) -> GNNPolicyValueModel:
    encoder = GNNEncoder(
        GNNEncoderConfig(
            num_rounds=num_rounds,
            hidden_width=hidden_width,
            communication=communication,
            pooling=pooling,
        )
    )
    model = GNNPolicyValueModel(encoder, PolicyValueHeadConfig(hidden_width=hidden_width))
    _randomise(model)
    model.eval()
    return model


def _call(model: GNNPolicyValueModel, tensors: dict[str, torch.Tensor]):
    """The model on a batch."""
    return model(**tensors)


def _encode(model: GNNPolicyValueModel, tensors: dict[str, torch.Tensor]):
    senders = {key: tensors[key] for key in SENDER_KEYS if key in tensors}
    return model.encoder.encode_padded(
        *(tensors[key] for key in _ENCODER_KEYS), **senders, pool_mask=tensors.get("pool_mask")
    )


def _permuted(tensors: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """The batch with the move slots relabelled, slot k taking the old slot perm[k]."""
    return {key: value[:, perm] if key in _SLOT_KEYS else value for key, value in tensors.items()}


def _random_head_inputs(communication: bool, batch_size: int = 3, num_slots: int = 4):
    generator = torch.Generator().manual_seed(5)
    inputs = dict(
        z=torch.randn(batch_size, 11, generator=generator),
        node_embeddings=torch.randn(batch_size, 6, 5, generator=generator),
        move_target_index=torch.randint(0, 6, (batch_size, num_slots), generator=generator),
    )
    if communication:
        inputs.update(
            messages=torch.randn(batch_size, 2, 5, generator=generator),
            move_sender_index=torch.randint(0, 2, (batch_size, num_slots), generator=generator),
            move_sender_mask=(torch.rand(batch_size, num_slots, generator=generator) < 0.5).float(),
        )
    return inputs


@pytest.mark.parametrize("communication", [False, True])
def test_policy_value_head_output_shapes(communication):
    head = PolicyValueHead(11, 5, communication, PolicyValueHeadConfig(hidden_width=8))
    logits, value = head(**_random_head_inputs(communication))
    assert logits.shape == (3, 5)  # wait and four move slots
    assert value.shape == (3,)


def test_policy_value_head_gradients_flow():
    head = PolicyValueHead(11, 5, True, PolicyValueHeadConfig(hidden_width=4))
    _randomise(head)
    logits, value = head(**_random_head_inputs(True))
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters())


def test_the_three_networks_share_no_trunk():
    head = PolicyValueHead(11, 5, True, PolicyValueHeadConfig(hidden_width=4))
    inputs = _random_head_inputs(True)
    assert not hasattr(head, "trunk")
    parameter_ids = [{id(p) for p in net.parameters()} for net in (head.score, head.score_wait)]
    parameter_ids.append({id(p) for p in head.value_net.parameters()})
    assert all(a.isdisjoint(b) for i, a in enumerate(parameter_ids) for b in parameter_ids[:i])

    head.value(inputs["z"]).sum().backward()
    assert all(p.grad is None for net in (head.score, head.score_wait) for p in net.parameters())
    head.zero_grad()
    head.policy(**inputs).sum().backward()
    assert all(p.grad is None for p in head.value_net.parameters())


def test_the_move_score_input_is_z_h_w_and_the_message_unless_communication_is_off():
    on = PolicyValueHead(11, 5, True, PolicyValueHeadConfig(hidden_width=4))
    off = PolicyValueHead(11, 5, False, PolicyValueHeadConfig(hidden_width=4))
    assert on.score[0].in_features == 11 + 5 + 5
    assert off.score[0].in_features == 11 + 5
    assert on.score_wait[0].in_features == off.score_wait[0].in_features == 11


def test_the_head_takes_message_inputs_exactly_when_communication_is_on():
    off = PolicyValueHead(11, 5, False, PolicyValueHeadConfig(hidden_width=4))
    on = PolicyValueHead(11, 5, True, PolicyValueHeadConfig(hidden_width=4))
    with pytest.raises(ValueError, match="communication off"):
        off.policy(**_random_head_inputs(True))
    with pytest.raises(ValueError, match="communication is on"):
        on.policy(**_random_head_inputs(False))


@pytest.mark.parametrize("communication", [False, True])
def test_gnn_policy_value_model_end_to_end_shapes(communication):
    logits, value = _call(_model(2, communication), _batch(communication))
    assert logits.shape == (3, 4)  # wait and d_max = 3 move slots, one row per robot
    assert value.shape == (3,)


def test_gnn_policy_value_model_logits_are_unmasked_raw_values():
    """The heads must NOT apply the action mask themselves, because
    ActionMaskingTorchRLModule owns that. Slots that do not exist at a vertex
    get a finite score from padding."""
    tensors = _batch(False)
    logits, _ = _call(_model(0, False), tensors)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("pooling", ["none", "mean"])
@pytest.mark.parametrize("num_rounds", [0, 2])
@pytest.mark.parametrize("communication", [False, True])
def test_padded_logits_equal_an_unbatched_oracle_that_scores_one_neighbour_at_a_time(
    num_rounds, communication, pooling
):
    graph = _star_graph()
    config = _config(communication)
    fleet = _fleet()
    model = _model(num_rounds, communication, pooling=pooling)
    with torch.no_grad():
        logits, _ = _call(model, _batch(communication, pooling=pooling))
        for row, agent_id in enumerate(_LOCATIONS):
            location = _LOCATIONS[agent_id]
            observation = build_observation(graph, fleet, [], agent_id, 0, config)
            senders = []
            if communication:
                heard = communication_neighbours(graph, fleet, agent_id, 3)[:2]
                senders = [
                    (_LOCATIONS[j], build_observation(graph, fleet, [], j, 0, config))
                    for j in heard
                ]
            pool_vertices = _zone_pool(graph, location) if pooling == "mean" else None
            z = model.encoder(graph, location, observation, senders, pool_vertices)
            h_w, xibar_w = model.encoder.embed_moves(graph, location, observation, senders)

            assert torch.allclose(model.head.score_wait(z)[0], logits[row, 0], atol=1e-5)
            for k in range(h_w.shape[0]):
                score_input = [z, h_w[k]] + ([xibar_w[k]] if communication else [])
                score = model.head.score(torch.cat(score_input))[0]
                assert torch.allclose(score, logits[row, 1 + k], atol=1e-5)


@pytest.mark.parametrize("num_rounds", [0, 2])
@pytest.mark.parametrize("communication", [False, True])
def test_relabelling_the_move_slots_permutes_the_move_logits_the_same_way(
    num_rounds, communication
):
    """The thesis claim that vertex numbering does not affect what the policy
    learns, for the head: permute the slot arrays and the move logits permute,
    while the wait logit stays."""
    model = _model(num_rounds, communication)
    tensors = _batch(communication)
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        logits, _ = _call(model, tensors)
        relabelled, _ = _call(model, _permuted(tensors, perm))

    # the permutation has to change something, or the test would pass for any head
    assert (logits[:, 1:] - logits[:, 1:][:, perm]).abs().max() > 1e-3
    assert torch.allclose(relabelled[:, 1:], logits[:, 1:][:, perm], atol=1e-5)
    assert torch.equal(relabelled[:, 0], logits[:, 0])


@pytest.mark.parametrize("communication", [False, True])
def test_two_slots_with_equal_inputs_get_equal_logits(communication):
    """A score depends only on (z_i, h_w, xibar_w), not on the slot number."""
    model = _model(2, communication)
    tensors = _batch(communication)
    for key in _SLOT_KEYS:
        if key in tensors:
            tensors[key][:, 1] = tensors[key][:, 0]
    with torch.no_grad():
        logits, _ = _call(model, tensors)
    assert torch.allclose(logits[:, 1], logits[:, 2], atol=1e-6)


class _Capture(nn.Module):
    """Stands in for the move score network and records what it reads."""

    def __init__(self):
        super().__init__()
        self.inputs = torch.empty(0)

    def forward(self, x):
        self.inputs = x
        return x[..., :1]


def _xibar_w(model: GNNPolicyValueModel, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """The last D columns of the move score input, [B, K, D], read through the head."""
    capture = _Capture()
    model.head.score = capture
    out = _encode(model, tensors)
    model.head.policy(
        out.z,
        out.node_embeddings,
        tensors["move_target_index"],
        out.messages,
        tensors["move_sender_index"],
        tensors["move_sender_mask"],
    )
    return capture.inputs[..., -model.encoder.embedding_dim :]


def test_xibar_is_zero_for_an_empty_vertex_and_for_a_robot_that_is_not_heard():
    model = _model(2, True)
    tensors = _batch(True, max_messages=1)  # robot 1 hears only robot 2, the closest
    with torch.no_grad():
        xibar = _xibar_w(model, tensors)
    # robot 1 stands in the centre. The move to vertex 1 has robot 2 as its sender. Vertex 2 is
    # empty, and robot 3 on vertex 3 is not among the one sender robot 1 hears.
    assert tensors["move_sender_mask"][0].tolist() == [1.0, 0.0, 0.0]
    assert not xibar[0, 1:].any()


def test_xibar_equals_the_senders_own_embedding_otherwise():
    model = _model(2, True)
    tensors = _batch(True)
    with torch.no_grad():
        xibar = _xibar_w(model, tensors)
        out = _encode(model, tensors)
    # the first slot of robot 1 leads to vertex 1, where robot 2 stands. Robot 2 is row 1 of the
    # batch, and its own view puts it at own_index.
    own_embedding_of_robot_2 = out.node_embeddings[1, tensors["own_index"][1]]
    assert own_embedding_of_robot_2.abs().sum() > 0
    assert torch.allclose(xibar[0, 0], own_embedding_of_robot_2, atol=1e-5)
    own_embedding_of_robot_3 = out.node_embeddings[2, tensors["own_index"][2]]
    assert torch.allclose(xibar[0, 2], own_embedding_of_robot_3, atol=1e-5)


def test_xibar_is_zero_for_a_padded_sender_whose_own_embedding_is_not():
    model = _model(2, True)
    tensors = _batch(True)  # robot 3 hears one sender, so its second sender block is padding
    assert tensors["sender_mask"][2].tolist() == [1.0, 0.0]
    tensors["move_sender_index"][2, 0] = 1  # point slot 1 at the padded block
    tensors["move_sender_mask"][2, 0] = 0.0
    with torch.no_grad():
        out = _encode(model, tensors)
        xibar = _xibar_w(model, tensors)
    assert out.messages[2, 1].abs().sum() > 0  # the padded block does hold an embedding
    assert not xibar[2, 0].any()


def _move_logits(model, tensors, detach_h: bool = False, detach_xibar: bool = False):
    """Logits from a head that reads z_i detached, so a gradient can only reach the
    encoder through h_w or through the message of xibar_w."""
    out = _encode(model, tensors)
    node_embeddings = out.node_embeddings.detach() if detach_h else out.node_embeddings
    senders = {}
    if model.head.communication:
        senders = dict(
            messages=out.messages.detach() if detach_xibar else out.messages,
            move_sender_index=tensors["move_sender_index"],
            move_sender_mask=tensors["move_sender_mask"],
        )
    return model.head.policy(
        out.z.detach(), node_embeddings, tensors["move_target_index"], **senders
    )


def _encoder_got_a_gradient(model) -> bool:
    return any(p.grad is not None and torch.any(p.grad != 0) for p in model.encoder.parameters())


@pytest.mark.parametrize("communication", [False, True])
def test_a_loss_on_one_move_logit_reaches_the_encoder_through_h_w(communication):
    model = _model(2, communication)
    logits = _move_logits(model, _batch(communication), detach_xibar=True)
    model.zero_grad()
    logits[0, 1].backward()
    assert _encoder_got_a_gradient(model)


def test_a_loss_on_one_move_logit_reaches_the_encoder_through_the_message_of_xibar_w():
    model = _model(2, True)
    logits = _move_logits(model, _batch(True), detach_h=True)
    model.zero_grad()
    logits[0, 1].backward()  # the first slot of robot 1 leads to the vertex robot 2 stands on
    assert _encoder_got_a_gradient(model)


def test_no_gradient_reaches_the_encoder_when_h_w_and_xibar_w_are_both_detached():
    model = _model(2, True)
    logits = _move_logits(model, _batch(True), detach_h=True, detach_xibar=True)
    model.zero_grad()
    logits[0, 1].backward()
    assert not _encoder_got_a_gradient(model)


def _logits_through_the_pooled_term(model, tensors, detach_pooled: bool = False):
    """Logits from a head whose inputs are all detached except the pooled
    term of z_i, which sits just before the congestion feature."""
    out = _encode(model, tensors)
    start = out.z.shape[1] - 1 - model.encoder.embedding_dim
    pooled = out.z[:, start:-1]
    z = torch.cat(
        [
            out.z[:, :start].detach(),
            pooled.detach() if detach_pooled else pooled,
            out.z[:, -1:].detach(),
        ],
        dim=-1,
    )
    senders = {}
    if model.head.communication:
        senders = {key: tensors[key] for key in MOVE_SENDER_KEYS}
        senders["messages"] = out.messages.detach()
    return model.head.policy(
        z, out.node_embeddings.detach(), tensors["move_target_index"], **senders
    )


@pytest.mark.parametrize("communication", [False, True])
def test_a_loss_on_the_logits_reaches_the_encoder_through_the_pooled_term(communication):
    model = _model(2, communication, pooling="mean")
    tensors = _batch(communication, pooling="mean")

    logits = _logits_through_the_pooled_term(model, tensors)
    model.zero_grad()
    logits.sum().backward()
    assert _encoder_got_a_gradient(model)

    # the check above is not vacuous, since with the pooled term detached as well
    # nothing reaches the encoder
    logits = _logits_through_the_pooled_term(model, tensors, detach_pooled=True)
    model.zero_grad()
    logits.sum().backward()
    assert not _encoder_got_a_gradient(model)


@pytest.mark.parametrize("communication", [False, True])
def test_the_value_does_not_depend_on_the_slot_arrays(communication):
    model = _model(2, communication)
    tensors = _batch(communication)
    zeroed = {k: torch.zeros_like(v) if k in _SLOT_KEYS else v for k, v in tensors.items()}
    with torch.no_grad():
        _, value = _call(model, tensors)
        _, relabelled = _call(model, _permuted(tensors, torch.tensor([2, 0, 1])))
        _, without_slots = _call(model, zeroed)
    assert torch.equal(value, relabelled)
    assert torch.equal(value, without_slots)
