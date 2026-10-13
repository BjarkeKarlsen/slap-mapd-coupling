"""The GNN architecture registered as an RLlib RLModule (sec:method:model,
sec:method:training).

DefaultPPORLModule needs `setup()` to build `self.encoder`, `self.pi` and
`self.vf`, with `self.encoder(batch)` returning
`{ENCODER_OUT: {ACTOR: a, CRITIC: c}}`. The critic reads z_i (the readout),
so CRITIC is that tensor. The move score also reads h^(L) at the vertex each
move leads to and the message of the robot standing there, so ACTOR is a dict
holding z_i and those tensors, which DefaultPPORLModule hands to `self.pi`
unchanged. Everything the head needs arrives through the observation, so the
module never looks at the graph.

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
from slap_mapd_coupling.models.base_model import (
    MOVE_SENDER_KEYS,
    PolicyValueHead,
    PolicyValueHeadConfig,
)
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS, GNNEncoder, GNNEncoderConfig


class _GNNSharedEncoder(nn.Module):
    """Wraps GNNEncoder.encode_padded to satisfy DefaultPPORLModule's
    `self.encoder(batch) -> {ENCODER_OUT: {ACTOR: ..., CRITIC: ...}}`
    contract, see the module docstring for what each role gets."""

    def __init__(self, encoder: GNNEncoder) -> None:
        super().__init__()
        self.gnn = encoder

    def forward(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        # the sender keys are in the observation exactly when communication is on
        communication = self.gnn.config.communication
        sender_blocks = {key: obs[key] for key in SENDER_KEYS} if communication else {}
        out = self.gnn.encode_padded(
            obs["node_features"],
            obs["node_mask"],
            obs["edge_index"],
            obs["edge_mask"],
            obs["own_index"],
            obs["congestion"],
            **sender_blocks,
        )
        actor: Dict[str, Any] = {
            "z": out.z,
            "node_embeddings": out.node_embeddings,
            "move_target_index": obs["move_target_index"],
        }
        if communication:
            actor["messages"] = out.messages
            actor.update({key: obs[key] for key in MOVE_SENDER_KEYS})
        return {ENCODER_OUT: {ACTOR: actor, CRITIC: out.z}}


class _PiHead(nn.Module):
    """self.pi(actor_inputs) -> raw policy logits, the move score and the
    wait score of PolicyValueHead (masking is applied afterwards by
    ActionMaskingTorchRLModule, not here, see the module docstring)."""

    def __init__(self, head: PolicyValueHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, actor: Dict[str, Any]) -> torch.Tensor:
        return self.head.policy(**actor)


class _VfHead(nn.Module):
    """self.vf(critic_embedding) -> value tensor; DefaultPPORLModule
    squeezes the last dim itself, so this returns [..., 1] like RLlib's
    own default vf head does."""

    def __init__(self, head: PolicyValueHead) -> None:
        super().__init__()
        self.head = head

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head.value(z).unsqueeze(-1)


class GNNPPOTorchRLModule(DefaultPPOTorchRLModule):
    """DefaultPPOTorchRLModule with GNNEncoder + PolicyValueHead (#27/#28)
    in place of RLlib's Catalog-built default network, see the module
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
        head = PolicyValueHead(
            gnn_encoder.output_dim,
            gnn_encoder.embedding_dim,
            encoder_config.communication,
            head_config,
        )
        self.pi = _PiHead(head)
        self.vf = _VfHead(head)


class GNNActionMaskingRLModule(ActionMaskingTorchRLModule, GNNPPOTorchRLModule):
    """The actual RLModule class passed to RLlib's RLModuleSpec for the
    decentralised regime's training (training/config.py, #30): #26's
    masking wrapper composed with the GNN architecture (#27/#28) instead
    of RLlib's Catalog-built default -- see module docstring for how."""
