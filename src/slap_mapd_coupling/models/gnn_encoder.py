"""Multi-round message-passing encoder over the field of view G_i^(d_obs)(t)
(sec:method:model, the message-passing step and the readout).

phi_r is a per-neighbour MLP over concat(h_v, h_w), averaged over the
neighbour set, and psi_r is an MLP over concat(h_v, m_v). The thesis
leaves their internal form open. AGG for incoming messages is the mean.
L and the hidden width (tab:modelparams) live in GNNEncoderConfig, local
to this module like GeneratorParams in instances/generator.py.

`forward` is the unbatched oracle, one Observation in, one z_i out.
`forward_padded` is the batched path over the padded tensors of
models/local_subgraph_encoding.py. It passes messages over an undirected
edge list, so its cost grows with the number of edges (issue #86). Tests
check that both paths agree.
"""

from __future__ import annotations

from typing import Sequence

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import nn

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.observation import Observation, VertexFeatures

NODE_FEATURE_DIM = 6  # storage, delivery, endpoint, transit, eta, occupied (the vertex features)
MESSAGE_FEATURE_DIM = 2  # hand-built (distance, eta) messages, until #91's learned message


class GNNEncoderConfig(BaseModel):
    """L and hidden width (tab:modelparams)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_rounds: NonNegativeInt  # L, the message-passing step -- L=0 is a valid, real ablation
    hidden_width: PositiveInt  # phi_r/psi_r layer width, tab:modelparams


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


class GNNEncoder(nn.Module):
    """z_i (the readout), via L rounds of the message-passing step over
    the field of view. L = 0 gives the raw node features. Weights are
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
    def output_dim(self) -> int:
        """z_i's dimension: the final node embedding width (NODE_FEATURE_DIM
        itself when L=0, hidden_width otherwise) plus the message
        aggregate and the congestion scalar."""
        return self._node_embedding_dim + MESSAGE_FEATURE_DIM + 1

    def forward(
        self, graph: WarehouseGraph, agent_location: VertexId, observation: Observation
    ) -> torch.Tensor:
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

        own_embedding = h[index[agent_location]]

        if observation.messages:
            message_tensor = torch.tensor(
                [[m.distance, m.eta] for m in observation.messages], dtype=torch.float32
            )
            aggregated = message_tensor.mean(dim=0)
        else:
            aggregated = torch.zeros(MESSAGE_FEATURE_DIM)

        # the readout always includes delta_i(t); Observation.congestion is
        # always a concrete scalar (environment/observation.py), whether or
        # not the policy is meant to condition on it (the "sensitivity"
        # toggle governs training/interpretation, not tensor shape).
        congestion_tensor = torch.tensor([observation.congestion], dtype=torch.float32)

        return torch.cat([own_embedding, aggregated, congestion_tensor], dim=-1)

    def forward_padded(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        own_index: torch.Tensor,
        message_features: torch.Tensor,
        message_mask: torch.Tensor,
        congestion: torch.Tensor,
    ) -> torch.Tensor:
        """Batched z_i (the readout) over padded tensors. B is the batch
        size, N max_local_nodes, E the edge list length, M max_messages.

        node_features [B,N,NODE_FEATURE_DIM], node_mask [B,N], edge_index
        [B,E,2] (int64, one row per undirected pair, padded rows point at
        vertex 0), edge_mask [B,E] (1 valid / 0 pad), own_index [B] (int64),
        message_features [B,M,MESSAGE_FEATURE_DIM], message_mask [B,M],
        congestion [B].

        The batch is flattened to B*N vertices. Each edge gives both
        directions, phi_r runs once per directed edge, and the results are
        summed per receiving vertex and divided by its masked degree
        (minimum 1). This equals `forward`'s mean over each vertex's
        neighbour set. Padded edges and vertices carry zero weight.
        """
        h = node_features
        batch_size, max_nodes, _ = h.shape

        if self.config.num_rounds > 0:
            offsets = (torch.arange(batch_size, device=h.device) * max_nodes).unsqueeze(1)
            first = edge_index[..., 0]  # [B,E]
            second = edge_index[..., 1]  # [B,E]
            edge_weight = edge_mask * node_mask.gather(1, first) * node_mask.gather(1, second)
            weight = edge_weight.reshape(-1)
            receivers = torch.cat([(first + offsets).reshape(-1), (second + offsets).reshape(-1)])
            senders = torch.cat([(second + offsets).reshape(-1), (first + offsets).reshape(-1)])
            weight = torch.cat([weight, weight])  # [2*B*E], one entry per directed edge
            degree = torch.zeros(batch_size * max_nodes, dtype=h.dtype, device=h.device)
            degree.index_add_(0, receivers, weight)
            degree = degree.clamp(min=1.0).unsqueeze(-1)  # [B*N,1]

            for phi_r, psi_r in zip(self.phi, self.psi):
                flat = h.reshape(batch_size * max_nodes, -1)
                edge_messages = phi_r(torch.cat([flat[receivers], flat[senders]], dim=-1))
                summed = torch.zeros(
                    batch_size * max_nodes,
                    edge_messages.shape[-1],
                    dtype=edge_messages.dtype,
                    device=h.device,
                )
                summed.index_add_(0, receivers, edge_messages * weight.unsqueeze(-1))
                m = (summed / degree).reshape(batch_size, max_nodes, -1)
                h = psi_r(torch.cat([h, m], dim=-1))

        own_embedding = h[torch.arange(batch_size), own_index]  # [B, node_dim]

        message_weight = message_mask.unsqueeze(-1)  # [B,M,1]
        message_count = message_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        aggregated = (message_features * message_weight).sum(
            dim=1
        ) / message_count  # [B,MESSAGE_FEATURE_DIM]

        return torch.cat([own_embedding, aggregated, congestion.unsqueeze(-1)], dim=-1)
