"""Pads and masks an Observation (environment/observation.py) into the
fixed-shape arrays GNNEncoder.forward_padded and RLlib's batched pipeline
need. The field of view has a variable vertex count, so everything is
padded to max_local_nodes.

The thesis does not say what to do when the field of view exceeds the cap
(issue #66), so the truncation is a flagged choice. Nodes keep the
agent's own vertex first, then the rest by ascending eta_i(v,t). An edge
to a dropped vertex is left out of the edge list. Messages keep the
max_messages closest senders by distance.

The edge list has one row per undirected pair among the selected
vertices, padded to E_max = max_local_nodes * d_max. That bound holds
because every selected pair is a directed edge, and no selected vertex
has more than d_max outgoing edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.action_masking import to_action_mask
from slap_mapd_coupling.environment.observation import Message, Observation
from slap_mapd_coupling.environment.spaces import Mask
from slap_mapd_coupling.models.gnn_encoder import MESSAGE_FEATURE_DIM, NODE_FEATURE_DIM


class LocalSubgraphEncodingConfig(BaseModel):
    """max_local_nodes/max_messages: the padding caps. Local to this module,
    like GNNEncoderConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_local_nodes: PositiveInt
    max_messages: NonNegativeInt


@dataclass(frozen=True)
class EncodedObservation:
    """The padded arrays for one agent at one timestep, without a batch
    dimension."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    edge_index: np.ndarray  # [E_max, 2] int64, one row per undirected pair, pad rows are (0, 0)
    edge_mask: np.ndarray  # [E_max], 1.0 valid / 0.0 pad
    own_index: int
    message_features: np.ndarray  # [max_messages, MESSAGE_FEATURE_DIM]
    message_mask: np.ndarray  # [max_messages]
    congestion: float
    action_mask: np.ndarray  # [num_actions], {0.0, 1.0} -- to_action_mask(mask)

    def as_rllib_dict(self) -> dict[str, Any]:
        """{"action_mask": ..., "observations": {...}}, the Dict shape that
        masked_observation_space() and encoded_observation_space() declare."""
        return {
            "action_mask": self.action_mask,
            "observations": {
                "node_features": self.node_features,
                "node_mask": self.node_mask,
                "edge_index": self.edge_index,
                "edge_mask": self.edge_mask,
                "own_index": self.own_index,
                "message_features": self.message_features,
                "message_mask": self.message_mask,
                "congestion": np.float32(self.congestion),
            },
        }


def max_edges(config: LocalSubgraphEncodingConfig, d_max: int) -> int:
    """E_max, the padded length of the edge list."""
    return config.max_local_nodes * d_max


def encoded_observation_space(config: LocalSubgraphEncodingConfig, d_max: int) -> gym.spaces.Dict:
    """The "observations" half of masked_observation_space() for one agent.
    `d_max` is max_out_degree(graph), which sets the edge list's size."""
    num_edges = max_edges(config, d_max)
    return gym.spaces.Dict(
        {
            "node_features": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(config.max_local_nodes, NODE_FEATURE_DIM),
                dtype=np.float32,
            ),
            "node_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(config.max_local_nodes,), dtype=np.float32
            ),
            "edge_index": gym.spaces.Box(
                low=0,
                high=config.max_local_nodes - 1,
                shape=(num_edges, 2),
                dtype=np.int64,
            ),
            "edge_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(num_edges,), dtype=np.float32),
            "own_index": gym.spaces.Discrete(config.max_local_nodes),
            "message_features": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(config.max_messages, MESSAGE_FEATURE_DIM),
                dtype=np.float32,
            ),
            "message_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(config.max_messages,), dtype=np.float32
            ),
            "congestion": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        }
    )


def _select_nodes(
    observation: Observation, agent_location: VertexId, max_local_nodes: int
) -> list[VertexId]:
    """l_i(t) first, then the remaining visible vertices by ascending eta."""
    visible = list(observation.visible_vertices)
    if len(visible) <= max_local_nodes:
        return visible
    others = sorted(
        (v for v in visible if v != agent_location),
        key=lambda v: observation.features[v].eta,
    )
    return [agent_location] + others[: max_local_nodes - 1]


def _select_messages(observation: Observation, max_messages: int) -> tuple[Message, ...]:
    """The max_messages closest senders by distance."""
    if len(observation.messages) <= max_messages:
        return observation.messages
    return tuple(sorted(observation.messages, key=lambda m: m.distance)[:max_messages])


def encode_observation(
    graph: WarehouseGraph,
    agent_location: VertexId,
    observation: Observation,
    mask: Mask,
    config: LocalSubgraphEncodingConfig,
) -> EncodedObservation:
    """o_i(t) and its action mask -> the fixed-shape EncodedObservation.
    `mask` is the caller's own legality_mask result. Its length is
    d_max + 1, which sets the edge list's size."""
    selected = _select_nodes(observation, agent_location, config.max_local_nodes)
    index = {v: i for i, v in enumerate(selected)}
    n = len(selected)

    node_features = np.zeros((config.max_local_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    for i, v in enumerate(selected):
        f = observation.features[v]
        node_features[i] = (
            float(f.storage),
            float(f.delivery),
            float(f.endpoint),
            float(f.transit),
            f.eta,
            float(f.occupied),
        )
    node_mask = np.zeros(config.max_local_nodes, dtype=np.float32)
    node_mask[:n] = 1.0

    num_edges = max_edges(config, len(mask) - 1)
    edge_index = np.zeros((num_edges, 2), dtype=np.int64)
    edge_mask = np.zeros(num_edges, dtype=np.float32)
    pairs: set[tuple[int, int]] = set()
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            a, b = index[edge.source], index[edge.target]
            pair = (min(a, b), max(a, b))
            if pair not in pairs:
                edge_index[len(pairs)] = pair
                edge_mask[len(pairs)] = 1.0
                pairs.add(pair)

    own_index = index[agent_location]

    selected_messages = _select_messages(observation, config.max_messages)
    message_features = np.zeros((config.max_messages, MESSAGE_FEATURE_DIM), dtype=np.float32)
    message_mask = np.zeros(config.max_messages, dtype=np.float32)
    for i, message in enumerate(selected_messages):
        message_features[i] = (message.distance, message.eta)
        message_mask[i] = 1.0

    return EncodedObservation(
        node_features=node_features,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        own_index=own_index,
        message_features=message_features,
        message_mask=message_mask,
        congestion=observation.congestion,
        action_mask=to_action_mask(mask),
    )
