"""Multi-round message-passing encoder over the local observation graph
G_i^(d)(t) (sec:method:model, eq:msgpass/eq:readout).

h_v^(0) = f(v,t) (eq:features); for q = 1..Q:
  m_v^(q) = phi_q(h_v^(q-1), {h_w^(q-1) : w in G_i^(d)(t), (w,v) or (v,w) in E})
  h_v^(q) = psi_q(h_v^(q-1), m_v^(q))
z_i = h_{l_i(t)}^(Q) || AGG(messages from a_j : (a_i,a_j) in E^A_t) || delta_i(t)

This is the "one level further" step from Knippenberg (2021)'s single GCN
pass to explicit multi-round message passing, so information travels more
than one hop within G_i^(d)(t) per decision.

Flagged, not literally specified: phi_q/psi_q's internal form is never
pinned down beyond their (input) -> (output) signature, which
eq:msgpass/eq:readout already fixes. Realised here as the standard MPNN
pair matching that exact signature: phi_q is a per-neighbour-pair MLP
(concat(h_v, h_w)) mean-pooled over the neighbour set (the conventional,
order-invariant "message" step for a set-valued second argument --
GraphSAGE's own realisation of the same signature), and psi_q is a
second MLP over concat(h_v, m_v) (the conventional "update" step). AGG
for incoming messages is mean, exactly as eq:readout's own parenthetical
suggests ("e.g. mean"). Both are the standard textbook choices for these
two roles, not an arbitrary architecture pick.

Q and hidden width (tab:modelparams) are TBD-by-sweep, so they're a named
GNNEncoderConfig here -- not ExperimentConfig, and not hard-coded.
ExperimentConfig is specifically "the five independent variables of the
study" (storage mode, controller, congestion sensitivity, communication,
load) plus what one concrete episode needs; Q/hidden width are model
architecture hyperparameters used only by this encoder, the same
"parameters local to the module they configure" pattern
instances/generator.py's own GeneratorParams already follows rather than
folding into an unrelated shared config.

Q=0 is a real, supported ablation (tab:modelparams: "TBD (sweep, incl.
Q=0)"), collapsing z_i to raw node/message features with no propagation
at all -- ablating both local structural aggregation and inter-agent
communication together. The thesis explicitly notes that isolating the
two would need two independent round-counts instead of one Q, calling
that an "open refinement, not yet adopted" -- this encoder does NOT
implement that split, since doing so would be inventing a decision the
thesis text explicitly says hasn't been made.
"""

from __future__ import annotations

from typing import Sequence

import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch import nn

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.observation import Observation, VertexFeatures

NODE_FEATURE_DIM = 6  # storage, delivery, endpoint, transit, eta, occupied (eq:features)
MESSAGE_FEATURE_DIM = 2  # (distance, eta) per eq:observation's message content


class GNNEncoderConfig(BaseModel):
    """Q and hidden width (tab:modelparams). See module docstring for why
    this lives here rather than on ExperimentConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_rounds: NonNegativeInt  # Q, eq:msgpass -- Q=0 is a valid, real ablation
    hidden_width: PositiveInt  # phi_q/psi_q layer width, tab:modelparams


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
    """{v: {w : (w,v) in E or (v,w) in E, w in G_i^(d)(t)}} -- eq:msgpass's
    neighbour set is explicitly "or," so a one-way edge in either
    direction still counts, restricted to the visible vertex set."""
    visible_set = set(visible)
    neighbours: dict[VertexId, set[VertexId]] = {v: set() for v in visible}
    for edge in graph.edges:
        if edge.source in visible_set and edge.target in visible_set:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    return neighbours


class GNNEncoder(nn.Module):
    """z_i (eq:readout), via Q rounds of message passing (eq:msgpass) over
    G_i^(d)(t). See module docstring for phi_q/psi_q's realisation and
    the Q=0 ablation. Weights are shared across all agents (parameter
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
        itself when Q=0, hidden_width otherwise) plus the message
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
            for phi_q, psi_q in zip(self.phi, self.psi):
                messages = torch.zeros(len(visible), self.config.hidden_width)
                for i, v in enumerate(visible):
                    neighbour_ids = neighbours[v]
                    if not neighbour_ids:
                        continue
                    own = h[i].unsqueeze(0).expand(len(neighbour_ids), -1)
                    neighbour_h = torch.stack([h[index[w]] for w in neighbour_ids])
                    pairwise = torch.cat([own, neighbour_h], dim=-1)
                    messages[i] = phi_q(pairwise).mean(dim=0)
                h = psi_q(torch.cat([h, messages], dim=-1))

        own_embedding = h[index[agent_location]]

        if observation.messages:
            message_tensor = torch.tensor(
                [[m.distance, m.eta] for m in observation.messages], dtype=torch.float32
            )
            aggregated = message_tensor.mean(dim=0)
        else:
            aggregated = torch.zeros(MESSAGE_FEATURE_DIM)

        # eq:readout always includes delta_i(t); Observation.congestion is
        # None specifically when congestion_radius isn't configured
        # (environment/observation.py) -- falls back to 0.0 here, since
        # the encoder needs a concrete scalar regardless of whether the
        # policy is meant to condition on it (the "sensitivity" toggle
        # governs training/interpretation, not tensor shape).
        congestion = observation.congestion if observation.congestion is not None else 0.0
        congestion_tensor = torch.tensor([congestion], dtype=torch.float32)

        return torch.cat([own_embedding, aggregated, congestion_tensor], dim=-1)
