"""The policy and value heads (sec:method:model): fully connected layers
with ReLU activations over z_i (the readout), which is built from o_i(t)
and, with communication on, the messages received.

The heads return raw logits and a scalar value. The action mask is applied
afterwards by ActionMaskingTorchRLModule (environment/action_masking.py),
so masking here would apply it twice. This module is plain PyTorch with no
RLlib dependency. The RLlib wiring is in models/rl_module.py.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch import nn

from slap_mapd_coupling.models.gnn_encoder import GNNEncoder


class PolicyValueHeadConfig(BaseModel):
    """Head architecture hyperparameters, local to this module -- same
    "not ExperimentConfig" pattern as GNNEncoderConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_width: PositiveInt
    num_actions: PositiveInt  # d_max + 1 (the action mask), environment/spaces.py::action_space


class PolicyValueHead(nn.Module):
    """FC+ReLU trunk over z_i, then a policy output (num_actions raw logits)
    and a value output (scalar). The shared trunk is a choice, not a thesis
    fact."""

    def __init__(self, input_dim: int, config: PolicyValueHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(nn.Linear(input_dim, config.hidden_width), nn.ReLU())
        self.policy_head = nn.Linear(config.hidden_width, config.num_actions)
        self.value_head = nn.Linear(config.hidden_width, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [..., input_dim] -> (logits [..., num_actions], value [...]).
        logits are raw/unmasked -- see module docstring."""
        h = self.trunk(z)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class GNNPolicyValueModel(nn.Module):
    """GNNEncoder.forward_padded followed by PolicyValueHead
    (sec:method:model), on the padded tensors of
    models/local_subgraph_encoding.py. Batched only."""

    def __init__(self, encoder: GNNEncoder, head_config: PolicyValueHeadConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = PolicyValueHead(encoder.output_dim, head_config)

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        own_index: torch.Tensor,
        congestion: torch.Tensor,
        **sender_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`sender_blocks` are the GNNEncoder.forward_padded sender tensors,
        given exactly when communication is on."""
        z = self.encoder.forward_padded(
            node_features,
            node_mask,
            edge_index,
            edge_mask,
            own_index,
            congestion,
            **sender_blocks,
        )
        logits, value = self.head(z)
        return logits, value
