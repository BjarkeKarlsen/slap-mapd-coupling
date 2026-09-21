"""pi_theta(.|o_i(t)) and V_theta(o_i(t)) (sec:method:model, issue #28):
fully connected layers with ReLU activations over z_i (eq:readout).

Scoped to the issue's literal text -- "fully connected layers with ReLU
activations over z_i ... masked per eq:mask before the softmax" -- as
the policy/value heads over the encoder's output. Masking itself is
NOT applied here: #26's ActionMaskingTorchRLModule already owns
"masked ... before the softmax" as RLlib infrastructure (it adds -inf
to raw action logits before they reach the action distribution).
Re-masking here would double-apply eq:mask, so this module hands back
raw (unmasked) logits plus a scalar value -- exactly the pair #26
expects to receive.

The issue's own text calls this a "TorchModelV2 wrapper," which is
stale relative to RLlib's actual current API (see models/gnn_encoder.py
and environment/action_masking.py, which already established this: the
new API stack uses RLModule, not TorchModelV2). Registering
GNNEncoder + PolicyValueHead as the real RLlib RLModule/Catalog that
#26 wraps -- so DefaultPPOTorchRLModule builds this architecture
instead of RLlib's own default MLP-on-flat-observation -- is #29/#30's
job (decentralised controller wrapper / training config), not this
one. This module is deliberately RLlib-agnostic: a plain, independently
testable PyTorch nn.Module operating on encode_observation()'s tensors
(models/local_subgraph_encoding.py), the same "pure module now, RLlib
wiring deferred" split GNNEncoder.forward/forward_padded already uses.
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
    num_actions: PositiveInt  # d_max + 1 (eq:mask), environment/spaces.py::action_space


class PolicyValueHead(nn.Module):
    """FC+ReLU trunk over z_i, then separate policy (num_actions raw
    logits) and value (scalar) outputs -- the conventional actor-critic
    realisation of "fully connected layers with ReLU activations over
    z_i" (issue #28); a shared trunk isn't itself pinned down further by
    the thesis text, so this is a defensible, not a forced, choice."""

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
    """The full pi_theta/V_theta pipeline, o_i(t) -> (raw logits, value)
    (sec:method:model): GNNEncoder.forward_padded (#27) -> PolicyValueHead.
    Batched only (mirrors forward_padded); RLlib wiring is #29/#30's job,
    see module docstring."""

    def __init__(self, encoder: GNNEncoder, head_config: PolicyValueHeadConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = PolicyValueHead(encoder.output_dim, head_config)

    def forward(
        self,
        node_features: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
        own_index: torch.Tensor,
        message_features: torch.Tensor,
        message_mask: torch.Tensor,
        congestion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder.forward_padded(
            node_features,
            node_mask,
            adjacency,
            own_index,
            message_features,
            message_mask,
            congestion,
        )
        logits, value = self.head(z)
        return logits, value
