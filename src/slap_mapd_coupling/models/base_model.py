"""The policy and value heads (sec:method:model), over z_i (the readout), the
embeddings h^(L) at the vertices of the field of view and, with
communication on, the messages received.

The policy scores each move by where it leads (the move score). One small
network `score` reads z_i, h_w^(L) and the message of the robot standing on
w, and is shared by every move slot. A second network `score_wait` reads
z_i alone and scores waiting. The value is a third network over z_i. They
share no trunk. With communication off the message is left out of the
move score.

Slot k is the k-th outgoing neighbour by vertex number, and the
observation holds the vertex it leads to (models/local_subgraph_encoding.py).
The head returns raw logits for slots 0 (wait) to d_max, so slots that do
not exist at a vertex also get a score. The action mask is applied
afterwards by ActionMaskingTorchRLModule (environment/action_masking.py),
so masking here would apply it twice. This module is plain PyTorch with no
RLlib dependency. The RLlib wiring is in models/rl_module.py.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn

from slap_mapd_coupling.models.gnn_encoder import GNNEncoder

# The keys that name the sender standing on each move slot. They are in an
# encoded observation exactly when communication is on.
MOVE_SENDER_KEYS = ("move_sender_index", "move_sender_mask")


class PolicyValueHeadConfig(BaseModel):
    """Head architecture hyperparameters, local to this module -- same
    "not ExperimentConfig" pattern as GNNEncoderConfig. The number of move
    slots is not here, it is the length of the slot arrays."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_width: PositiveInt


def _network(input_dim: int, hidden_width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 1))


class PolicyValueHead(nn.Module):
    """The move score, the wait score and the value, three networks with no
    shared trunk. `input_dim` is the width of z_i, `embedding_dim` the width of
    h^(L) and of a message (GNNEncoder.embedding_dim), and `communication`
    must equal the encoder's switch."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        communication: bool,
        config: PolicyValueHeadConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.communication = communication
        message_dim = embedding_dim if communication else 0
        self.score = _network(input_dim + embedding_dim + message_dim, config.hidden_width)
        self.score_wait = _network(input_dim, config.hidden_width)
        self.value_net = _network(input_dim, config.hidden_width)

    def policy(
        self,
        z: torch.Tensor,
        node_embeddings: torch.Tensor,
        move_target_index: torch.Tensor,
        messages: Optional[torch.Tensor] = None,
        move_sender_index: Optional[torch.Tensor] = None,
        move_sender_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw logits [B, d_max + 1], wait first. B is the batch size, N
        max_local_nodes, M max_messages, K = d_max and D the embedding width.

        z [B, input_dim], node_embeddings [B, N, D], move_target_index [B, K]
        (int64). With communication on, messages [B, M, D], move_sender_index
        [B, K] (int64) and move_sender_mask [B, K] are required, and the
        message of slot k is the sender's embedding times the mask, so a
        padded sender never leaks in. They must be None with it off."""
        rows = torch.arange(z.shape[0], device=z.device).unsqueeze(1)  # [B,1]
        h_w = node_embeddings[rows, move_target_index]  # [B,K,D]
        parts = [z.unsqueeze(1).expand(-1, h_w.shape[1], -1), h_w]

        sender_inputs = (messages, move_sender_index, move_sender_mask)
        if not self.communication:
            if any(item is not None for item in sender_inputs):
                raise ValueError("message inputs were given but the head has communication off")
        else:
            if messages is None or move_sender_index is None or move_sender_mask is None:
                raise ValueError(f"communication is on, so all of {MOVE_SENDER_KEYS} are required")
            xibar_w = messages[rows, move_sender_index] * move_sender_mask.unsqueeze(-1)
            parts.append(xibar_w)

        scores = self.score(torch.cat(parts, dim=-1)).squeeze(-1)  # [B,K]
        return torch.cat([self.score_wait(z), scores], dim=-1)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        """The value [B], from z_i alone."""
        value: torch.Tensor = self.value_net(z).squeeze(-1)
        return value

    def forward(
        self,
        z: torch.Tensor,
        node_embeddings: torch.Tensor,
        move_target_index: torch.Tensor,
        messages: Optional[torch.Tensor] = None,
        move_sender_index: Optional[torch.Tensor] = None,
        move_sender_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits [B, d_max + 1], value [B]). The logits are raw, see the
        module docstring."""
        logits = self.policy(
            z, node_embeddings, move_target_index, messages, move_sender_index, move_sender_mask
        )
        return logits, self.value(z)


class GNNPolicyValueModel(nn.Module):
    """GNNEncoder.encode_padded followed by PolicyValueHead
    (sec:method:model), on the padded tensors of
    models/local_subgraph_encoding.py. Batched only."""

    def __init__(self, encoder: GNNEncoder, head_config: PolicyValueHeadConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = PolicyValueHead(
            encoder.output_dim, encoder.embedding_dim, encoder.config.communication, head_config
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        own_index: torch.Tensor,
        congestion: torch.Tensor,
        move_target_index: torch.Tensor,
        **sender_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`sender_blocks` are the GNNEncoder.forward_padded sender tensors and
        MOVE_SENDER_KEYS, given exactly when communication is on."""
        move_sender = {
            key: sender_blocks.pop(key) for key in MOVE_SENDER_KEYS if key in sender_blocks
        }
        out = self.encoder.encode_padded(
            node_features,
            node_mask,
            edge_index,
            edge_mask,
            own_index,
            congestion,
            **sender_blocks,
        )
        logits, value = self.head(
            out.z, out.node_embeddings, move_target_index, out.messages, **move_sender
        )
        return logits, value
