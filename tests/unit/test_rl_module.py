"""Unit tests for slap_mapd_coupling.models.rl_module."""

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.spaces import action_space, max_out_degree
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.models.base_model import MOVE_SENDER_KEYS
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
    encoded_observation_space,
)
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _build_module(
    graph: WarehouseGraph,
    enc_config: LocalSubgraphEncodingConfig,
    communication,
    pooling: str = "none",
):
    d_max = max_out_degree(graph)
    base_space = encoded_observation_space(enc_config, d_max, communication, pooling=pooling)
    obs_space = masked_observation_space(base_space, d_max + 1)
    spec = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=obs_space,
        action_space=action_space(graph),
        model_config={
            "gnn_encoder_config": {
                "num_rounds": 2,
                "hidden_width": 8,
                "communication": communication,
                "pooling": pooling,
            },
            "policy_value_head_config": {"hidden_width": 8},
        },
    )
    return spec.build(), d_max


@pytest.mark.parametrize("communication", [False, True])
def test_gnn_action_masking_module_builds_and_masks_illegal_logits(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config, communication)

    # vertex 0 is an endpoint of the line graph with degree 1
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=1)}
    )
    config = ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=communication,
        num_agents=2,
        arrival_rate=1.0,
        seed=0,
        horizon=10,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        move_cost=0.0,
        congestion_reward_weight=0.1,
    )
    encoded = encode_all_observations(graph, fleet, [], 0, config, enc_config, d_max)
    rllib_obs = encoded[1].as_rllib_dict()
    assert (set(SENDER_KEYS) <= set(rllib_obs["observations"])) == communication

    def to_batch(value):
        if isinstance(value, dict):
            return {k: to_batch(v) for k, v in value.items()}
        return torch.as_tensor(value).unsqueeze(0)

    batch = {Columns.OBS: to_batch(rllib_obs)}
    out = module.forward_inference(batch)
    logits = out[Columns.ACTION_DIST_INPUTS][0]

    # vertex 0 (an endpoint of the line graph) has out-degree 1: only
    # wait (slot 0) and one move slot (slot 1) are legal; every other
    # slot must be masked to effectively -inf.
    assert logits[0] > -1e30
    assert logits[1] > -1e30
    assert all(logits[i] < -1e30 for i in range(2, d_max + 1))


def _stack(rows):
    """One batch with a row per observation, nested like the observations."""
    if isinstance(rows[0], dict):
        return {key: _stack([row[key] for row in rows]) for key in rows[0]}
    return torch.as_tensor(np.stack(rows))


def _two_robot_batch(graph, enc_config, communication, d_max, pooling="none"):
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=2), 2: AgentState(agent_id=2, location=3)}
    )
    config = ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=communication,
        num_agents=2,
        arrival_rate=1.0,
        seed=0,
        horizon=10,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        move_cost=0.0,
        congestion_reward_weight=0.1,
    )
    encoded = encode_all_observations(
        graph, fleet, [], 0, config, enc_config, d_max, pooling=pooling
    )
    return {Columns.OBS: _stack([item.as_rllib_dict() for item in encoded.values()])}


@pytest.mark.parametrize("communication", [False, True])
def test_the_actor_gets_the_slot_inputs_and_the_critic_only_z(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config, communication)
    batch = _two_robot_batch(graph, enc_config, communication, d_max)

    with torch.no_grad():
        out = module.encoder({Columns.OBS: batch[Columns.OBS]["observations"]})[ENCODER_OUT]

    actor = out[ACTOR]
    assert isinstance(actor, dict)
    assert torch.equal(actor["z"], out[CRITIC])
    assert actor["node_embeddings"].shape[:2] == (2, 8)
    assert actor["move_target_index"].shape == (2, d_max)
    assert (set(MOVE_SENDER_KEYS) | {"messages"} <= set(actor)) == communication


@pytest.mark.parametrize("pooling", ["none", "mean"])
@pytest.mark.parametrize("communication", [False, True])
def test_every_forward_mode_and_the_value_run_with_the_dict_actor_output(communication, pooling):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config, communication, pooling)

    def fresh_batch():
        # the masking module strips the action mask off the batch it is given
        return _two_robot_batch(graph, enc_config, communication, d_max, pooling)

    with torch.no_grad():
        inference = module.forward_inference(fresh_batch())[Columns.ACTION_DIST_INPUTS]
        exploration = module.forward_exploration(fresh_batch())[Columns.ACTION_DIST_INPUTS]
        values = module.compute_values(fresh_batch())

    assert inference.shape == exploration.shape == (2, d_max + 1)
    assert torch.equal(inference, exploration)
    assert values.shape == (2,)


@pytest.mark.parametrize("communication", [False, True])
def test_the_value_is_the_same_when_computed_in_chunks(communication, monkeypatch):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config, communication)

    with torch.no_grad():
        whole = module.compute_values(_two_robot_batch(graph, enc_config, communication, d_max))
        monkeypatch.setattr("slap_mapd_coupling.environment.action_masking.VALUE_CHUNK_ROWS", 1)
        chunked = module.compute_values(_two_robot_batch(graph, enc_config, communication, d_max))

    assert chunked.shape == whole.shape == (2,)
    assert torch.allclose(chunked, whole, atol=1e-6)


def test_the_module_passes_the_pool_mask_to_the_encoder():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config, False, "mean")
    batch = _two_robot_batch(graph, enc_config, False, d_max, "mean")
    observations = batch[Columns.OBS]["observations"]
    gnn = module.encoder.gnn
    keys = ("node_features", "node_mask", "edge_index", "edge_mask", "own_index", "congestion")

    def z_of(obs):
        with torch.no_grad():
            z = module.encoder({Columns.OBS: obs})[ENCODER_OUT][CRITIC]
            expected = gnn.encode_padded(*(obs[k] for k in keys), pool_mask=obs["pool_mask"]).z
        assert torch.equal(z, expected)
        return z

    z = z_of(observations)
    assert z.shape == (2, gnn.output_dim)
    own_vertex_only = dict(observations)
    own_vertex_only["pool_mask"] = torch.nn.functional.one_hot(
        observations["own_index"], num_classes=8
    ).float()
    assert not torch.allclose(z, z_of(own_vertex_only))
