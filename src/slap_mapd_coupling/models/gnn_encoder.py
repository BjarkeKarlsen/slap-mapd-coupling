"""Multi-round message-passing encoder over the field of view G_i^(d_obs)(t)
(sec:method:model, the message-passing step, the message and the readout).

phi_r is a per-neighbour MLP over concat(h_v, h_w), averaged over the
neighbour set, and psi_r is an MLP over concat(h_v, m_v). The thesis
leaves their internal form open. L and the hidden width (tab:modelparams)
live in GNNEncoderConfig, local to this module like GeneratorParams in
instances/generator.py.

The message of a sender is its own final-layer embedding at its own vertex,
computed from its own field of view with the shared weights and sent as it
is. The receiver takes the mean over its senders. With communication on, a
receiver's input therefore carries the padded views of its senders, and the
embeddings are recomputed with the current weights, so the receiver's loss
reaches the sender's embedding through the shared encoder (issue #91).
`forward_padded` embeds the receivers and their senders in one pass.

The move score (sec:method:model) also needs h^(L) at every vertex of the
field of view, not only at the robot's own, and the message of the robot
standing on a neighbour. `encode_padded` returns them next to z_i.

`forward` is the unbatched oracle, one Observation (and its senders'
Observations) in, one z_i out, and `embed_moves` gives its h_w and message
per move slot. `forward_padded` and `encode_padded` are the batched paths
over the padded tensors of models/local_subgraph_encoding.py. Tests check
that both paths agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import nn

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.observation import Observation, VertexFeatures

NODE_FEATURE_DIM = 6  # storage, delivery, endpoint, transit, eta, occupied (the vertex features)

# The keys of the senders' padded views in an encoded observation. They are
# present exactly when GNNEncoderConfig.communication is on.
SENDER_KEYS = (
    "sender_node_features",
    "sender_node_mask",
    "sender_edge_index",
    "sender_edge_mask",
    "sender_own_index",
    "sender_mask",
)

# Whether the readout carries the pooled mean. The observation then holds
# the key "pool_mask", which marks the vertices to pool over.
Pooling = Literal["none", "mean"]


class GNNEncoderConfig(BaseModel):
    """L, the hidden width (tab:modelparams) and the communication switch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_rounds: NonNegativeInt  # L, the message-passing step. L=0 is a valid, real ablation
    hidden_width: PositiveInt  # phi_r/psi_r layer width, tab:modelparams
    # No default. It must equal ExperimentConfig.communication, which
    # training/config.py::build_ppo_config checks.
    communication: bool


def _node_feature_vector(features: VertexFeatures) -> list[float]:
    return [
        float(features.storage),
        float(features.delivery),
        float(features.endpoint),
        float(features.transit),
        features.eta,
        float(features.occupied),
    ]


def _local_undirected_neighbours(
    graph: WarehouseGraph, visible: Sequence[VertexId]
) -> dict[VertexId, set[VertexId]]:
    """The neighbour set of the message-passing step. An edge in either
    direction counts, restricted to the visible vertices."""
    visible_set = set(visible)
    neighbours: dict[VertexId, set[VertexId]] = {v: set() for v in visible}
    for edge in graph.edges:
        if edge.source in visible_set and edge.target in visible_set:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    return neighbours


@dataclass(frozen=True)
class EncoderOutput:
    """What GNNEncoder.encode_padded returns. B is the batch size, N
    max_local_nodes, M max_messages and D the embedding width."""

    z: torch.Tensor  # [B, output_dim], the readout
    node_embeddings: torch.Tensor  # [B, N, D], h^(L) at every vertex of each field of view
    messages: Optional[torch.Tensor]  # [B, M, D], the sender blocks' embeddings, None if off


class GNNEncoder(nn.Module):
    """z_i (the readout), via L rounds of the message-passing step over
    the field of view. L = 0 skips the rounds, so the embedding is the raw
    vertex features. Communication has its own switch, so at L = 0 with
    communication on a message is the sender's raw features. Weights are
    shared across all agents (sec:pf:controllers).
    """

    def __init__(self, config: GNNEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.phi = nn.ModuleList()
        self.psi = nn.ModuleList()
        prev_dim = NODE_FEATURE_DIM
        for _ in range(config.num_rounds):
            self.phi.append(nn.Sequential(nn.Linear(prev_dim * 2, config.hidden_width), nn.ReLU()))
            self.psi.append(
                nn.Sequential(
                    nn.Linear(prev_dim + config.hidden_width, config.hidden_width), nn.ReLU()
                )
            )
            prev_dim = config.hidden_width
        self._node_embedding_dim = prev_dim

    @property
    def embedding_dim(self) -> int:
        """The width of h^(L) at a vertex, and so of a message. It is
        NODE_FEATURE_DIM when L=0 and hidden_width otherwise."""
        return self._node_embedding_dim

    @property
    def output_dim(self) -> int:
        """z_i's dimension: the own embedding, the mean of the messages
        (only with communication on) and the congestion scalar."""
        message_dim = self._node_embedding_dim if self.config.communication else 0
        return self._node_embedding_dim + message_dim + 1

    def _embed_vertices(
        self, graph: WarehouseGraph, observation: Observation
    ) -> tuple[torch.Tensor, dict[VertexId, int]]:
        """h^(L) at every visible vertex of `observation`, [V, D], and the
        row of each vertex."""
        visible = observation.visible_vertices
        index = {v: i for i, v in enumerate(visible)}
        h = torch.tensor(
            [_node_feature_vector(observation.features[v]) for v in visible], dtype=torch.float32
        )

        if self.config.num_rounds > 0:
            neighbours = _local_undirected_neighbours(graph, visible)
            for phi_r, psi_r in zip(self.phi, self.psi):
                messages = torch.zeros(len(visible), self.config.hidden_width)
                for i, v in enumerate(visible):
                    neighbour_ids = neighbours[v]
                    if not neighbour_ids:
                        continue
                    own = h[i].unsqueeze(0).expand(len(neighbour_ids), -1)
                    neighbour_h = torch.stack([h[index[w]] for w in neighbour_ids])
                    pairwise = torch.cat([own, neighbour_h], dim=-1)
                    messages[i] = phi_r(pairwise).mean(dim=0)
                h = psi_r(torch.cat([h, messages], dim=-1))

        return h, index

    def _embed_view(
        self, graph: WarehouseGraph, location: VertexId, observation: Observation
    ) -> torch.Tensor:
        """h^(L) at `location`, from the field of view in `observation`."""
        h, index = self._embed_vertices(graph, observation)
        return h[index[location]]

    def forward(
        self,
        graph: WarehouseGraph,
        agent_location: VertexId,
        observation: Observation,
        senders: Sequence[tuple[VertexId, Observation]] = (),
    ) -> torch.Tensor:
        """z_i for one agent. `senders` holds the vertex and the own
        Observation of each agent it hears, and must stay empty with
        communication off."""
        if senders and not self.config.communication:
            raise ValueError("senders were given but the encoder has communication off")
        parts = [self._embed_view(graph, agent_location, observation)]

        if self.config.communication:
            if senders:
                messages = torch.stack(
                    [self._embed_view(graph, location, view) for location, view in senders]
                )
                parts.append(messages.mean(dim=0))
            else:
                parts.append(torch.zeros(self._node_embedding_dim))

        # the readout always includes delta_i(t); Observation.congestion is
        # always a concrete scalar (environment/observation.py), whether or
        # not the policy is meant to condition on it (the "sensitivity"
        # toggle governs training/interpretation, not tensor shape).
        parts.append(torch.tensor([observation.congestion], dtype=torch.float32))
        return torch.cat(parts, dim=-1)

    def embed_moves(
        self,
        graph: WarehouseGraph,
        agent_location: VertexId,
        observation: Observation,
        senders: Sequence[tuple[VertexId, Observation]] = (),
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The inputs of the move score besides z_i, for each outgoing
        neighbour w of `agent_location` in slot order. Returns h_w^(L) from
        the agent's own field of view, [K, D], and the message of the sender
        standing on w, [K, D], zero where nobody it hears stands there. The
        message is None with communication off. `senders` is as in forward."""
        if senders and not self.config.communication:
            raise ValueError("senders were given but the encoder has communication off")
        h, index = self._embed_vertices(graph, observation)
        targets = graph.out_neighbours(agent_location)
        h_w = h[[index[w] for w in targets]]
        if not self.config.communication:
            return h_w, None

        sent = {location: self._embed_view(graph, location, view) for location, view in senders}
        zero = torch.zeros(self._node_embedding_dim)
        return h_w, torch.stack([sent.get(w, zero) for w in targets])

    def _embed_padded(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """h^(L) at every vertex of each of R padded views, [R, N, D]. Each
        edge gives both directions, phi_r runs once per directed edge, and
        the results are summed per receiving vertex and divided by its masked
        degree (minimum 1). This equals the mean over each vertex's neighbour
        set in `_embed_view`. Padded edges and vertices carry zero weight."""
        h = node_features
        num_views, max_nodes, _ = h.shape

        if self.config.num_rounds > 0:
            offsets = (torch.arange(num_views, device=h.device) * max_nodes).unsqueeze(1)
            first = edge_index[..., 0]  # [R,E]
            second = edge_index[..., 1]  # [R,E]
            edge_weight = edge_mask * node_mask.gather(1, first) * node_mask.gather(1, second)
            weight = edge_weight.reshape(-1)
            receivers = torch.cat([(first + offsets).reshape(-1), (second + offsets).reshape(-1)])
            sources = torch.cat([(second + offsets).reshape(-1), (first + offsets).reshape(-1)])
            weight = torch.cat([weight, weight])  # [2*R*E], one entry per directed edge
            degree = torch.zeros(num_views * max_nodes, dtype=h.dtype, device=h.device)
            degree.index_add_(0, receivers, weight)
            degree = degree.clamp(min=1.0).unsqueeze(-1)  # [R*N,1]

            for phi_r, psi_r in zip(self.phi, self.psi):
                flat = h.reshape(num_views * max_nodes, -1)
                edge_messages = phi_r(torch.cat([flat[receivers], flat[sources]], dim=-1))
                summed = torch.zeros(
                    num_views * max_nodes,
                    edge_messages.shape[-1],
                    dtype=edge_messages.dtype,
                    device=h.device,
                )
                summed.index_add_(0, receivers, edge_messages * weight.unsqueeze(-1))
                m = (summed / degree).reshape(num_views, max_nodes, -1)
                h = psi_r(torch.cat([h, m], dim=-1))

        return h

    def encode_padded(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        own_index: torch.Tensor,
        congestion: torch.Tensor,
        sender_node_features: Optional[torch.Tensor] = None,
        sender_node_mask: Optional[torch.Tensor] = None,
        sender_edge_index: Optional[torch.Tensor] = None,
        sender_edge_mask: Optional[torch.Tensor] = None,
        sender_own_index: Optional[torch.Tensor] = None,
        sender_mask: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        """Batched z_i (the readout), h^(L) at every vertex and the messages,
        over padded tensors. B is the batch size, N max_local_nodes, E the
        edge list length, M max_messages.

        node_features [B,N,NODE_FEATURE_DIM], node_mask [B,N], edge_index
        [B,E,2] (int64, one row per undirected pair, padded rows point at
        vertex 0), edge_mask [B,E] (1 valid / 0 pad), own_index [B] (int64),
        congestion [B].

        The six sender tensors are the senders' own views in the same layout
        with an extra axis M ([B,M,N,NODE_FEATURE_DIM], [B,M,N], [B,M,E,2],
        [B,M,E], [B,M], and sender_mask [B,M] with 1 for a real sender). They
        are given exactly when communication is on. The B receivers and the
        B*M senders share one embedding pass. A message is the sender's
        embedding as it is, and the receiver takes the mean over its masked
        senders, zero when there are none. `messages` holds the embedding of
        every sender block, padded ones included, so the caller masks them.
        """
        sender_tensors = (
            sender_node_features,
            sender_node_mask,
            sender_edge_index,
            sender_edge_mask,
            sender_own_index,
            sender_mask,
        )
        batch_size = node_features.shape[0]
        rows = torch.arange(batch_size, device=node_features.device)

        if not self.config.communication:
            if any(tensor is not None for tensor in sender_tensors):
                raise ValueError("sender tensors were given but the encoder has communication off")
            node_embeddings = self._embed_padded(node_features, node_mask, edge_index, edge_mask)
            own_embedding = node_embeddings[rows, own_index]
            z = torch.cat([own_embedding, congestion.unsqueeze(-1)], dim=-1)
            return EncoderOutput(z=z, node_embeddings=node_embeddings, messages=None)

        if (
            sender_node_features is None
            or sender_node_mask is None
            or sender_edge_index is None
            or sender_edge_mask is None
            or sender_own_index is None
            or sender_mask is None
        ):
            raise ValueError(f"communication is on, so all of {SENDER_KEYS} are required")

        num_senders = sender_mask.shape[1]
        embeddings = self._embed_padded(
            torch.cat([node_features, sender_node_features.flatten(0, 1)]),
            torch.cat([node_mask, sender_node_mask.flatten(0, 1)]),
            torch.cat([edge_index, sender_edge_index.flatten(0, 1)]),
            torch.cat([edge_mask, sender_edge_mask.flatten(0, 1)]),
        )
        node_embeddings = embeddings[:batch_size]  # [B,N,D]
        own_embedding = node_embeddings[rows, own_index]
        sender_rows = torch.arange(embeddings.shape[0] - batch_size, device=embeddings.device)
        messages = embeddings[batch_size:][sender_rows, sender_own_index.flatten()]
        messages = messages.reshape(batch_size, num_senders, -1)  # [B,M,D]

        count = sender_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        mean_message = (messages * sender_mask.unsqueeze(-1)).sum(dim=1) / count  # [B,D]

        z = torch.cat([own_embedding, mean_message, congestion.unsqueeze(-1)], dim=-1)
        return EncoderOutput(z=z, node_embeddings=node_embeddings, messages=messages)

    def forward_padded(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        own_index: torch.Tensor,
        congestion: torch.Tensor,
        sender_node_features: Optional[torch.Tensor] = None,
        sender_node_mask: Optional[torch.Tensor] = None,
        sender_edge_index: Optional[torch.Tensor] = None,
        sender_edge_mask: Optional[torch.Tensor] = None,
        sender_own_index: Optional[torch.Tensor] = None,
        sender_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched z_i (the readout), the `z` of `encode_padded`, which
        documents the arguments."""
        return self.encode_padded(
            node_features,
            node_mask,
            edge_index,
            edge_mask,
            own_index,
            congestion,
            sender_node_features,
            sender_node_mask,
            sender_edge_index,
            sender_edge_mask,
            sender_own_index,
            sender_mask,
        ).z
