"""Pads/masks a structured Observation (#25, environment/observation.py)
into the fixed-shape tensors GNNEncoder.forward_padded (#27) and RLlib's
batched training pipeline need.

G_i^(d)(t) has a variable vertex count (eq:localsubgraph); RLlib's
tensor-batched pipeline needs a fixed shape. Per the user-approved
option for #28, this pads/masks to a fixed max_local_nodes cap.

Truncation policy, flagged rather than picked silently since the thesis
text never addresses what to do when |G_i^(d)(t)| exceeds a fixed cap
(a real gap: eq:localsubgraph defines the local subgraph itself, not a
bound on its size, and eq:observation/eq:msgpass are written as if the
whole thing were always carried):
  - Nodes: always keep the agent's own vertex l_i(t) (own_index must be
    valid), then fill remaining slots by ascending eta_i(v,t)
    (eq:potential) -- i.e. keep the vertices closest to the agent's
    current goal first, the ones most likely to matter for the decision.
    Any edge to a truncated-away vertex is simply absent from the
    resulting (dense) adjacency, matching what "not in G_i^(d)(t) after
    truncation" should mean.
  - Messages: keep the max_messages closest senders by distance
    (Message.distance), for the same reasoning.
This is the natural "keep what's closest" choice, not an arbitrary one,
but it is a choice -- documented here so it's easy to revisit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.action_masking import to_action_mask
from slap_mapd_coupling.environment.observation import Message, Observation
from slap_mapd_coupling.environment.spaces import Mask
from slap_mapd_coupling.models.gnn_encoder import MESSAGE_FEATURE_DIM, NODE_FEATURE_DIM


class LocalSubgraphEncodingConfig(BaseModel):
    """max_local_nodes/max_messages: the padding caps. Model architecture
    hyperparameters, local to this module -- same "not ExperimentConfig"
    pattern as GNNEncoderConfig (see models/gnn_encoder.py's docstring)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_local_nodes: PositiveInt
    max_messages: NonNegativeInt


@dataclass(frozen=True)
class EncodedObservation:
    """The padded/masked tensors for one agent at one timestep, unbatched
    (no leading batch dim -- callers stack these themselves)."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    adjacency: np.ndarray  # [max_local_nodes, max_local_nodes]
    own_index: int
    message_features: np.ndarray  # [max_messages, MESSAGE_FEATURE_DIM]
    message_mask: np.ndarray  # [max_messages]
    congestion: float
    action_mask: np.ndarray  # [num_actions], {0.0, 1.0} -- to_action_mask(mask)


def _select_nodes(
    observation: Observation, agent_location: VertexId, max_local_nodes: int
) -> list[VertexId]:
    """l_i(t) always first (guarantees own_index is valid), then the
    remaining visible vertices by ascending eta -- see module docstring."""
    visible = list(observation.visible_vertices)
    if len(visible) <= max_local_nodes:
        return visible
    others = sorted(
        (v for v in visible if v != agent_location),
        key=lambda v: observation.features[v].eta,
    )
    return [agent_location] + others[: max_local_nodes - 1]


def _select_messages(observation: Observation, max_messages: int) -> tuple[Message, ...]:
    """The max_messages closest senders by distance -- see module docstring."""
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
    """o_i(t) (#25) + its action mask (environment/spaces.py::legality_mask,
    eq:mask) -> the fixed-shape EncodedObservation GNNEncoder.forward_padded
    (#27) and RLlib both need. `mask` is the caller's own legality_mask
    result, not recomputed here (same "pass it, don't recompute per call"
    convention legality_mask's own docstring already uses for d_max)."""
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

    adjacency = np.zeros((config.max_local_nodes, config.max_local_nodes), dtype=np.float32)
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            adjacency[index[edge.source], index[edge.target]] = 1.0
            adjacency[index[edge.target], index[edge.source]] = 1.0

    own_index = index[agent_location]

    selected_messages = _select_messages(observation, config.max_messages)
    message_features = np.zeros((config.max_messages, MESSAGE_FEATURE_DIM), dtype=np.float32)
    message_mask = np.zeros(config.max_messages, dtype=np.float32)
    for i, message in enumerate(selected_messages):
        message_features[i] = (message.distance, message.eta)
        message_mask[i] = 1.0

    congestion = observation.congestion if observation.congestion is not None else 0.0

    return EncodedObservation(
        node_features=node_features,
        node_mask=node_mask,
        adjacency=adjacency,
        own_index=own_index,
        message_features=message_features,
        message_mask=message_mask,
        congestion=congestion,
        action_mask=to_action_mask(mask),
    )
