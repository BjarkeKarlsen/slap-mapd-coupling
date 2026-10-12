"""The GNN architecture registered as an RLlib RLModule (sec:method:model,
sec:method:training).

DefaultPPORLModule needs `setup()` to build `self.encoder`, `self.pi` and
`self.vf`, with `self.encoder(batch)` returning
`{ENCODER_OUT: {ACTOR: t, CRITIC: t}}`. GNNEncoder.forward_padded gives one
shared z_i for both roles (the readout), so ACTOR and CRITIC are the same
tensor.

GNNActionMaskingRLModule composes this with ActionMaskingTorchRLModule by
multiple inheritance. Python's method resolution order puts
GNNPPOTorchRLModule where the masking module expects
DefaultPPOTorchRLModule, so no masking logic is repeated here.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.utils.annotations import override
from torch import nn

from slap_mapd_coupling.environment.action_masking import ActionMaskingTorchRLModule
from slap_mapd_coupling.models.base_model import PolicyValueHead, PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoder, GNNEncoderConfig


class _GNNSharedEncoder(nn.Module):
    """Wraps GNNEncoder.forward_padded to satisfy DefaultPPORLModule's
    `self.encoder(batch) -> {ENCODER_OUT: {ACTOR: z, CRITIC: z}}` contract
    -- see module docstring for why ACTOR and CRITIC are the same z_i."""

    def __init__(self, encoder: GNNEncoder) -> None:
        super().__init__()
        self.gnn = encoder

    def forward(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        z = self.gnn.forward_padded(
            obs["node_features"],
            obs["node_mask"],
            obs["edge_index"],
            obs["edge_mask"],
            obs["own_index"],
            obs["message_features"],
            obs["message_mask"],
            obs["congestion"],
        )
        return {ENCODER_OUT: {ACTOR: z, CRITIC: z}}


class _PiHead(nn.Module):
    """self.pi(actor_embedding) -> raw policy logits, delegating to
    PolicyValueHead's shared trunk + policy output (masking is applied
    afterwards by ActionMaskingTorchRLModule, not here -- see module
    docstring)."""

    def __init__(self, head: PolicyValueHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        logits: torch.Tensor
        logits, _ = self.head(z)
        return logits


class _VfHead(nn.Module):
    """self.vf(critic_embedding) -> value tensor; DefaultPPORLModule
    squeezes the last dim itself, so this returns [..., 1] like RLlib's
    own default vf head does."""

    def __init__(self, head: PolicyValueHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        value: torch.Tensor
        _, value = self.head(z)
        return value.unsqueeze(-1)


class GNNPPOTorchRLModule(DefaultPPOTorchRLModule):
    """DefaultPPOTorchRLModule with GNNEncoder + PolicyValueHead (#27/#28)
    in place of RLlib's Catalog-built default network -- see module
    docstring. `model_config` must carry "gnn_encoder_config" and
    "policy_value_head_config" (plain dicts, matching GNNEncoderConfig/
    PolicyValueHeadConfig's own fields) -- RLlib passes model_config
    through as a dict, not the pydantic model itself.
    """

    @override(DefaultPPOTorchRLModule)
    def setup(self) -> None:
        # Deliberately does NOT call super().setup(): that builds
        # Catalog-based networks this module replaces outright, not
        # extends.
        model_config = self.model_config or {}
        if not isinstance(model_config, dict):
            raise TypeError(
                f"GNNPPOTorchRLModule requires a plain dict model_config (with "
                f"'gnn_encoder_config'/'policy_value_head_config' keys), got "
                f"{type(model_config).__name__}."
            )
        encoder_config = GNNEncoderConfig(**model_config["gnn_encoder_config"])
        head_config = PolicyValueHeadConfig(**model_config["policy_value_head_config"])

        gnn_encoder = GNNEncoder(encoder_config)
        self.encoder = _GNNSharedEncoder(gnn_encoder)
        head = PolicyValueHead(gnn_encoder.output_dim, head_config)
        self.pi = _PiHead(head)
        self.vf = _VfHead(head)


class GNNActionMaskingRLModule(ActionMaskingTorchRLModule, GNNPPOTorchRLModule):
    """The actual RLModule class passed to RLlib's RLModuleSpec for the
    decentralised regime's training (training/config.py, #30): #26's
    masking wrapper composed with the GNN architecture (#27/#28) instead
    of RLlib's Catalog-built default -- see module docstring for how."""
