"""Multi-round message-passing encoder over the local observation graph
G_i^(d_obs)(t) (sec:method:model, the message-passing step / the readout).

h_v^(0) = f(v,t) (the vertex features); for r = 1..L:
  m_v^(r) = phi_r(h_v^(r-1), {h_w^(r-1) : w in G_i^(d_obs)(t), (w,v) or (v,w) in E})
  h_v^(r) = psi_r(h_v^(r-1), m_v^(r))
z_i = h_{l_i(t)}^(L) || AGG(messages from a_j : (a_i,a_j) in E^A_t) || delta_i(t)

phi_r/psi_r's internal form isn't pinned down beyond this signature.
Realised here as a standard MPNN pair: phi_r is a per-neighbour-pair MLP
(concat(h_v, h_w)) mean-pooled over the neighbour set, and psi_r is a
second MLP over concat(h_v, m_v). AGG for incoming messages is mean, as
the readout's own parenthetical suggests ("e.g. mean").

L and hidden width (tab:modelparams) are TBD-by-sweep, so they live in
GNNEncoderConfig here, not ExperimentConfig, the same "parameters local
to the module they configure" pattern instances/generator.py's own
GeneratorParams follows.

L=0 collapses the message-passing step to raw node features, with no
propagation within the field of view. Communication has its own switch
(ExperimentConfig.communication): the message mean at L=0 is unchanged,
so the structural ablation and the communication ablation are separate
(4.Implementation.tex:379-382).

`forward` is the unbatched path: one Observation in, one z_i out.
`forward_padded` is the batched entry point over already-padded/masked
tensors (models/local_subgraph_encoding.py), needed because
G_i^(d_obs)(t) has a variable vertex count RLlib's fixed-shape pipeline
can't carry directly. Both paths share the same phi/psi weights and
compute the same message-passing step, just vectorised; tests check
they agree on the same input.
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
    """L and hidden width (tab:modelparams). See module docstring for why
    this lives here rather than on ExperimentConfig."""

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
    """{v: {w : (w,v) in E or (v,w) in E, w in G_i^(d_obs)(t)}} -- the
    message-passing step's neighbour set is explicitly "or," so a
    one-way edge in either direction still counts, restricted to the
    visible vertex set."""
    visible_set = set(visible)
    neighbours: dict[VertexId, set[VertexId]] = {v: set() for v in visible}
    for edge in graph.edges:
        if edge.source in visible_set and edge.target in visible_set:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    return neighbours


class GNNEncoder(nn.Module):
    """z_i (the readout), via L rounds of the message-passing step over
    G_i^(d_obs)(t). See module docstring for phi_r/psi_r's realisation and
    the L=0 ablation. Weights are shared across all agents (parameter
    sharing, sec:pf:controllers) -- one GNNEncoder instance, called once
    per agent per decision, not one instance per agent.
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
        # None specifically when observation_depth (d_obs) isn't configured
        # (environment/observation.py) -- falls back to 0.0 here, since
        # the encoder needs a concrete scalar regardless of whether the
        # policy is meant to condition on it (the "sensitivity" toggle
        # governs training/interpretation, not tensor shape).
        congestion = observation.congestion if observation.congestion is not None else 0.0
        congestion_tensor = torch.tensor([congestion], dtype=torch.float32)

        return torch.cat([own_embedding, aggregated, congestion_tensor], dim=-1)

    def forward_padded(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
        own_index: torch.Tensor,
        message_features: torch.Tensor,
        message_mask: torch.Tensor,
        congestion: torch.Tensor,
    ) -> torch.Tensor:
        """Batched z_i (the readout) over already-padded/masked tensors --
        see class docstring. Shapes, with B the batch size, N
        max_local_nodes, M max_messages (models/local_subgraph_encoding.py):

        node_features [B,N,NODE_FEATURE_DIM], node_mask [B,N] (1 valid /
        0 pad), adjacency [B,N,N] (1 undirected neighbour / 0 otherwise,
        already 0 on any padded row/col), own_index [B] (long, which row
        is l_i(t)), message_features [B,M,MESSAGE_FEATURE_DIM],
        message_mask [B,M], congestion [B].

        Mathematically identical to `forward`'s per-vertex mean
        aggregation: phi_r is evaluated for every (v,w) pair densely,
        weighted by `adjacency` (zeroing out non-neighbours and padding
        in one step) and averaged by each node's own (masked) degree --
        the same neighbour-set mean `forward` computes sparsely via a
        Python loop, just vectorised.
        """
        h = node_features
        batch_size, max_nodes, _ = h.shape

        if self.config.num_rounds > 0:
            for phi_r, psi_r in zip(self.phi, self.psi):
                own = h.unsqueeze(2).expand(batch_size, max_nodes, max_nodes, -1)
                other = h.unsqueeze(1).expand(batch_size, max_nodes, max_nodes, -1)
                pairwise = torch.cat([own, other], dim=-1)
                pairwise_messages = phi_r(pairwise)  # [B,N,N,hidden]

                weight = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # [B,N,N]
                weighted = pairwise_messages * weight.unsqueeze(-1)
                summed = weighted.sum(dim=2)  # sum over w -> [B,N,hidden]
                degree = weight.sum(dim=2, keepdim=True).clamp(min=1.0)
                m = summed / degree

                h = psi_r(torch.cat([h, m], dim=-1))

        own_embedding = h[torch.arange(batch_size), own_index]  # [B, node_dim]

        message_weight = message_mask.unsqueeze(-1)  # [B,M,1]
        message_count = message_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1]
        aggregated = (message_features * message_weight).sum(
            dim=1
        ) / message_count  # [B,MESSAGE_FEATURE_DIM]

        return torch.cat([own_embedding, aggregated, congestion.unsqueeze(-1)], dim=-1)
