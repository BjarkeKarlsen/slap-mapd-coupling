"""Unit tests for slap_mapd_coupling.models.rl_module."""

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.spaces import action_space, legality_mask, max_out_degree
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_observation,
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


def _build_module(graph: WarehouseGraph, enc_config: LocalSubgraphEncodingConfig):
    d_max = max_out_degree(graph)
    base_space = encoded_observation_space(enc_config)
    obs_space = masked_observation_space(base_space, d_max + 1)
    spec = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=obs_space,
        action_space=action_space(graph),
        model_config={
            "gnn_encoder_config": {"num_rounds": 2, "hidden_width": 8},
            "policy_value_head_config": {"hidden_width": 8, "num_actions": d_max + 1},
        },
    )
    return spec.build(), d_max


def test_gnn_action_masking_module_builds_and_masks_illegal_logits():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module, d_max = _build_module(graph, enc_config)

    fleet = FleetState(agents={1: AgentState(agent_id=1, location=0)})  # endpoint, degree 1
    config = ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=1,
        arrival_rate=1.0,
        seed=0,
        horizon=10,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    mask = legality_mask(graph, vertex=0, d_max=d_max)
    encoded = encode_observation(graph, 0, observation, mask, enc_config)
    rllib_obs = encoded.as_rllib_dict()

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
