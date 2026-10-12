"""Pads and masks an Observation (environment/observation.py) into the
fixed-shape arrays GNNEncoder.forward_padded and RLlib's batched pipeline
need. The field of view has a variable vertex count, so everything is
padded to max_local_nodes.

The thesis does not say what to do when the field of view exceeds the cap
(issue #66), so the truncation is a flagged choice. Nodes keep the
agent's own vertex first, then the rest by ascending eta_i(v,t). An edge
to a dropped vertex is left out of the edge list. A receiver keeps its
max_messages closest senders by d_G, ties by agent id.

The edge list has one row per undirected pair among the selected
vertices, padded to E_max = max_local_nodes * d_max. That bound holds
because every selected pair is a directed edge, and no selected vertex
has more than d_max outgoing edges.

A message is the sender's own embedding (sec:method:model), so with
communication on a receiver's sample carries the padded views of its
senders, in the layout of its own view with an extra axis M. The network
embeds them with the current weights. With communication off the sender
keys are absent. `encode_all_observations` builds the sample of every robot
for one timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.action_masking import to_action_mask
from slap_mapd_coupling.environment.observation import (
    Observation,
    build_observation,
    communication_neighbours,
)
from slap_mapd_coupling.environment.spaces import Mask, legality_mask
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM


class LocalSubgraphEncodingConfig(BaseModel):
    """max_local_nodes/max_messages: the padding caps. Local to this module,
    like GNNEncoderConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_local_nodes: PositiveInt
    max_messages: NonNegativeInt


def check_communication_caps(config: LocalSubgraphEncodingConfig, communication: bool) -> None:
    """Rejects communication with room for no sender, which would silently
    switch the messages off while the encoder still expects them."""
    if communication and config.max_messages == 0:
        raise ValueError(
            "communication is on but max_messages is 0, so no sender fits in an "
            "observation. Set max_messages to at least 1, or switch communication off."
        )


@dataclass(frozen=True)
class EncodedView:
    """One robot's padded field of view, without a batch dimension. It is
    the same layout for the robot itself and for each of its senders."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    edge_index: np.ndarray  # [E_max, 2] int64, one row per undirected pair, pad rows are (0, 0)
    edge_mask: np.ndarray  # [E_max], 1.0 valid / 0.0 pad
    own_index: int


@dataclass(frozen=True)
class EncodedObservation:
    """The padded arrays for one agent at one timestep, without a batch
    dimension. The sender arrays are None with communication off."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    edge_index: np.ndarray  # [E_max, 2] int64, one row per undirected pair, pad rows are (0, 0)
    edge_mask: np.ndarray  # [E_max], 1.0 valid / 0.0 pad
    own_index: int
    congestion: float
    action_mask: np.ndarray  # [num_actions], {0.0, 1.0}, to_action_mask(mask)
    sender_node_features: Optional[np.ndarray] = None  # [max_messages, N, NODE_FEATURE_DIM]
    sender_node_mask: Optional[np.ndarray] = None  # [max_messages, N]
    sender_edge_index: Optional[np.ndarray] = None  # [max_messages, E_max, 2] int64
    sender_edge_mask: Optional[np.ndarray] = None  # [max_messages, E_max]
    sender_own_index: Optional[np.ndarray] = None  # [max_messages] int64
    sender_mask: Optional[np.ndarray] = None  # [max_messages], 1.0 real sender / 0.0 pad

    def as_rllib_dict(self) -> dict[str, Any]:
        """{"action_mask": ..., "observations": {...}}, the Dict shape that
        masked_observation_space() and encoded_observation_space() declare."""
        observations: dict[str, Any] = {
            "node_features": self.node_features,
            "node_mask": self.node_mask,
            "edge_index": self.edge_index,
            "edge_mask": self.edge_mask,
            "own_index": self.own_index,
            "congestion": np.float32(self.congestion),
        }
        if self.sender_mask is not None:
            observations.update(
                sender_node_features=self.sender_node_features,
                sender_node_mask=self.sender_node_mask,
                sender_edge_index=self.sender_edge_index,
                sender_edge_mask=self.sender_edge_mask,
                sender_own_index=self.sender_own_index,
                sender_mask=self.sender_mask,
            )
        return {"action_mask": self.action_mask, "observations": observations}


def max_edges(config: LocalSubgraphEncodingConfig, d_max: int) -> int:
    """E_max, the padded length of the edge list."""
    return config.max_local_nodes * d_max


def encoded_observation_space(
    config: LocalSubgraphEncodingConfig, d_max: int, communication: bool
) -> gym.spaces.Dict:
    """The "observations" half of masked_observation_space() for one agent.
    `d_max` is max_out_degree(graph), which sets the edge list's size. The
    sender keys are present only with `communication` on."""
    check_communication_caps(config, communication)
    num_nodes = config.max_local_nodes
    num_edges = max_edges(config, d_max)
    spaces: dict[str, gym.Space] = {
        "node_features": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_nodes, NODE_FEATURE_DIM), dtype=np.float32
        ),
        "node_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(num_nodes,), dtype=np.float32),
        "edge_index": gym.spaces.Box(
            low=0, high=num_nodes - 1, shape=(num_edges, 2), dtype=np.int64
        ),
        "edge_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(num_edges,), dtype=np.float32),
        "own_index": gym.spaces.Discrete(num_nodes),
        "congestion": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
    }
    if communication:
        m = config.max_messages
        spaces.update(
            sender_node_features=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(m, num_nodes, NODE_FEATURE_DIM), dtype=np.float32
            ),
            sender_node_mask=gym.spaces.Box(
                low=0.0, high=1.0, shape=(m, num_nodes), dtype=np.float32
            ),
            sender_edge_index=gym.spaces.Box(
                low=0, high=num_nodes - 1, shape=(m, num_edges, 2), dtype=np.int64
            ),
            sender_edge_mask=gym.spaces.Box(
                low=0.0, high=1.0, shape=(m, num_edges), dtype=np.float32
            ),
            sender_own_index=gym.spaces.Box(low=0, high=num_nodes - 1, shape=(m,), dtype=np.int64),
            sender_mask=gym.spaces.Box(low=0.0, high=1.0, shape=(m,), dtype=np.float32),
        )
    return gym.spaces.Dict(spaces)


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


def encode_view(
    graph: WarehouseGraph,
    location: VertexId,
    observation: Observation,
    d_max: int,
    config: LocalSubgraphEncodingConfig,
) -> EncodedView:
    """The field of view in `observation` as padded arrays. Used for the
    robot itself and for each sender. `d_max` is max_out_degree(graph)."""
    selected = _select_nodes(observation, location, config.max_local_nodes)
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

    num_edges = max_edges(config, d_max)
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

    return EncodedView(
        node_features=node_features,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        own_index=index[location],
    )


def _pad_senders(
    sender_views: Sequence[EncodedView], max_messages: int, view: EncodedView
) -> dict[str, np.ndarray]:
    """The six sender arrays, with the views of `sender_views` first and
    zero padding after them. `view` only supplies the shapes."""
    if len(sender_views) > max_messages:
        raise ValueError(f"{len(sender_views)} sender views exceed max_messages={max_messages}")
    arrays = {
        "sender_node_features": np.zeros((max_messages, *view.node_features.shape), np.float32),
        "sender_node_mask": np.zeros((max_messages, *view.node_mask.shape), np.float32),
        "sender_edge_index": np.zeros((max_messages, *view.edge_index.shape), np.int64),
        "sender_edge_mask": np.zeros((max_messages, *view.edge_mask.shape), np.float32),
        "sender_own_index": np.zeros(max_messages, np.int64),
        "sender_mask": np.zeros(max_messages, np.float32),
    }
    for k, sender in enumerate(sender_views):
        arrays["sender_node_features"][k] = sender.node_features
        arrays["sender_node_mask"][k] = sender.node_mask
        arrays["sender_edge_index"][k] = sender.edge_index
        arrays["sender_edge_mask"][k] = sender.edge_mask
        arrays["sender_own_index"][k] = sender.own_index
        arrays["sender_mask"][k] = 1.0
    return arrays


def _assemble(
    view: EncodedView,
    observation: Observation,
    mask: Mask,
    sender_views: Optional[Sequence[EncodedView]],
    config: LocalSubgraphEncodingConfig,
) -> EncodedObservation:
    senders = (
        _pad_senders(sender_views, config.max_messages, view) if sender_views is not None else {}
    )
    return EncodedObservation(
        node_features=view.node_features,
        node_mask=view.node_mask,
        edge_index=view.edge_index,
        edge_mask=view.edge_mask,
        own_index=view.own_index,
        congestion=observation.congestion,
        action_mask=to_action_mask(mask),
        **senders,
    )


def encode_observation(
    graph: WarehouseGraph,
    agent_location: VertexId,
    observation: Observation,
    mask: Mask,
    d_max: int,
    config: LocalSubgraphEncodingConfig,
    sender_views: Optional[Sequence[EncodedView]] = None,
) -> EncodedObservation:
    """o_i(t) and its action mask -> the fixed-shape EncodedObservation.
    `mask` is the caller's own legality_mask result and `d_max` is
    max_out_degree(graph). `sender_views` holds the views of the senders
    the agent hears, closest first, at most max_messages of them. None
    means communication is off and leaves the sender arrays out."""
    view = encode_view(graph, agent_location, observation, d_max, config)
    return _assemble(view, observation, mask, sender_views, config)


def encode_all_observations(
    graph: WarehouseGraph,
    fleet: FleetState,
    tasks: Sequence[Task],
    t: int,
    experiment_config: ExperimentConfig,
    encoding_config: LocalSubgraphEncodingConfig,
    d_max: int,
) -> dict[AgentId, EncodedObservation]:
    """The EncodedObservation of every robot at timestep t, keyed by agent
    id in ascending order. `d_max` is max_out_degree(graph).

    Each robot's view is encoded once and reused as a sender block by every
    robot that hears it. With experiment_config.communication on, a robot
    hears the max_messages closest agents of the communication graph (ties
    by agent id), closest first. The controller, the training env and any
    later env all build their observations here, so a message is the same
    wherever it is read."""
    communication = experiment_config.communication
    check_communication_caps(encoding_config, communication)
    if experiment_config.observation_depth is None:
        raise ValueError("encode_all_observations requires experiment_config.observation_depth")

    locations = fleet.locations()
    agent_ids = sorted(locations)
    observations = {
        agent_id: build_observation(graph, fleet, tasks, agent_id, t, experiment_config)
        for agent_id in agent_ids
    }
    views = {
        agent_id: encode_view(
            graph, locations[agent_id], observations[agent_id], d_max, encoding_config
        )
        for agent_id in agent_ids
    }

    encoded: dict[AgentId, EncodedObservation] = {}
    for agent_id in agent_ids:
        sender_views: Optional[list[EncodedView]] = None
        if communication:
            heard = communication_neighbours(
                graph, fleet, agent_id, experiment_config.observation_depth
            )
            sender_views = [views[j] for j in heard[: encoding_config.max_messages]]
        mask = legality_mask(graph, locations[agent_id], d_max=d_max)
        encoded[agent_id] = _assemble(
            views[agent_id], observations[agent_id], mask, sender_views, encoding_config
        )
    return encoded
