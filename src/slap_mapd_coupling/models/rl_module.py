"""The GNN architecture (#27/#28) registered as an actual RLlib RLModule
(sec:method:model, sec:method:training, issue #30).

#28's own module docstring deferred exactly this wiring here: "Registering
GNNEncoder + PolicyValueHead as the real RLlib RLModule/Catalog that #26
wraps -- so DefaultPPOTorchRLModule builds this architecture instead of
RLlib's own default MLP-on-flat-observation -- is #29/#30's job." Skipping
#29 per the user's instruction, this module does that registration
directly for #30's PPO training path.

DefaultPPORLModule's contract (ray.rllib.algorithms.ppo.default_ppo_rl_module,
inspected directly against the installed ray version rather than assumed
from memory, matching the same "verify against the actual installed
package" discipline #26/#27 already established): `setup()` must build
`self.encoder`, `self.pi`, `self.vf`; `self.encoder(batch)` must return
`{ENCODER_OUT: {ACTOR: t, CRITIC: t}}`; `self.pi(actor_embedding)` must
return raw action-dist-input logits; `self.vf(critic_embedding)` must
return a value tensor RLlib squeezes on its last dim (see
`ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module`).
GNNEncoder.forward_padded already produces one shared z_i (eq:readout)
for both roles -- unlike RLlib's own actor/critic-encoder split, there
is no separate value encoder here, matching eq:readout's single z_i
feeding both pi_theta and V_theta (sec:method:model): ACTOR and CRITIC
are the same tensor.

GNNActionMaskingRLModule below composes this with #26's
ActionMaskingTorchRLModule via multiple inheritance rather than
re-deriving its forward/masking logic: verified directly (not assumed)
that Python's C3 linearization places GNNPPOTorchRLModule in the MRO
exactly where ActionMaskingTorchRLModule expects DefaultPPOTorchRLModule,
so every method ActionMaskingTorchRLModule already implements --
setup(), _forward_inference/_forward_exploration/_forward_train,
compute_values, mask application -- works completely unchanged, only
now dispatching to self.encoder/self.pi/self.vf as GNNPPOTorchRLModule.
setup() builds them, instead of RLlib's Catalog-built default network.
No masking logic is duplicated here.
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
            obs["adjacency"],
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
