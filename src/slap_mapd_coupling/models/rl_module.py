"""The GNN architecture registered as an RLlib RLModule (sec:method:model,
sec:method:training).

DefaultPPORLModule needs `setup()` to build `self.encoder`, `self.pi` and
`self.vf`, with `self.encoder(batch)` returning
`{ENCODER_OUT: {ACTOR: a, CRITIC: c}}`. The critic reads z_i (the readout),
so CRITIC is that tensor. The move score also reads h^(L) at the vertex each
move leads to and the message of the robot standing there, so ACTOR is a dict
holding z_i and those tensors, which DefaultPPORLModule hands to `self.pi`
unchanged. Everything the head needs arrives through the observation and the
window tables of the model config (models/window_tables.py), so the module
never looks at the graph.

GNNActionMaskingRLModule composes this with ActionMaskingTorchRLModule by
multiple inheritance. Python's method resolution order puts
GNNPPOTorchRLModule where the masking module expects
DefaultPPOTorchRLModule, so no masking logic is repeated here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.utils.annotations import override
from torch import nn

from slap_mapd_coupling.environment.action_masking import (
    MESSAGE_TABLE,
    ActionMaskingTorchRLModule,
)
from slap_mapd_coupling.models.base_model import (
    MOVE_SENDER_KEYS,
    PolicyValueHead,
    PolicyValueHeadConfig,
)
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS, GNNEncoder, GNNEncoderConfig
from slap_mapd_coupling.models.window_tables import (
    StoredWindowLookup,
    WindowLookup,
    window_tables_from_config,
)


class _GNNSharedEncoder(nn.Module):
    """Wraps GNNEncoder.encode_padded to satisfy DefaultPPORLModule's
    `self.encoder(batch) -> {ENCODER_OUT: {ACTOR: ..., CRITIC: ...}}`
    contract, see the module docstring for what each role gets."""

    def __init__(self, encoder: GNNEncoder, lookup: Optional[WindowLookup] = None) -> None:
        super().__init__()
        self.gnn = encoder
        self.lookup = lookup

    def forward(self, batch: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        # A compact sample is turned back into the padded arrays first, see
        # models/window_tables.py. Without tables the samples are padded already.
        if self.lookup is not None:
            obs = self.lookup(obs)
            if self.gnn.config.communication:
                return self._forward_keyed(obs, batch.get(MESSAGE_TABLE))
        # the sender keys are in the observation exactly when communication is
        # on, and pool_mask exactly when pooling is on
        communication = self.gnn.config.communication
        sender_blocks = {key: obs[key] for key in SENDER_KEYS} if communication else {}
        pooling = self.gnn.config.pooling != "none"
        out = self.gnn.encode_padded(
            obs["node_features"],
            obs["node_mask"],
            obs["edge_index"],
            obs["edge_mask"],
            obs["own_index"],
            obs["congestion"],
            **sender_blocks,
            pool_mask=obs["pool_mask"] if pooling else None,
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

    def _forward_keyed(
        self, obs: Dict[str, Any], message_table: Optional[tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """forward for compact samples with communication on, where the senders
        are row keys (GNNEncoder.encode_keyed)."""
        out = self.gnn.encode_keyed(
            obs,
            obs["row_key"],
            obs["sender_row_key"],
            obs.get("context"),
            obs.get("context_row_key"),
            message_table,
        )
        actor: Dict[str, Any] = {
            "z": out.z,
            "node_embeddings": out.node_embeddings,
            "move_target_index": obs["move_target_index"],
            "messages": out.messages,
        }
        actor.update({key: obs[key] for key in MOVE_SENDER_KEYS})
        return {ENCODER_OUT: {ACTOR: actor, CRITIC: out.z}}

    def own_embeddings(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Each compact row's own embedding [B, D], the message its robot sends."""
        assert self.lookup is not None
        views = self.lookup.views(obs["vertex"], obs["eta"], obs["flags"])
        return self.gnn.own_embeddings(
            views["node_features"],
            views["node_mask"],
            views["edge_index"],
            views["edge_mask"],
            views["own_index"],
        )


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
    through as a dict, not the pydantic model itself. With "window_tables"
    (WindowTables.as_arrays()) the module reads compact samples, which is
    what the adapter of training/env.py stores, with "window_store" (a pool of
    training/warehouses.py) compact samples keyed by warehouse, and without
    either padded ones.
    """

    def __init__(self, *args: Any, catalog_class: Any = None, **kwargs: Any) -> None:
        # No Catalog, since setup() builds every network. DefaultPPOTorchRLModule.__init__ only
        # swaps a missing catalog_class for PPOCatalog, which cannot read the graph observation
        # and logs "Didn't create a Catalog object" for every module built, so it is skipped.
        # A checkpoint that names PPOCatalog loads the same way.
        super(DefaultPPOTorchRLModule, self).__init__(*args, catalog_class=None, **kwargs)

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
        tables = window_tables_from_config(model_config)
        lookup: Optional[WindowLookup] = None
        if tables is not None:
            lookup = WindowLookup(tables, encoder_config.pooling, encoder_config.communication)
        elif "window_store" in model_config:
            # A warehouse per episode: the tables are read from the pool on disk.
            from slap_mapd_coupling.training.warehouses import WarehousePool

            lookup = StoredWindowLookup(
                WarehousePool.from_config(model_config["window_store"]),
                encoder_config.pooling,
                encoder_config.communication,
            )
        self.encoder = _GNNSharedEncoder(gnn_encoder, lookup)
        head = PolicyValueHead(
            gnn_encoder.output_dim,
            gnn_encoder.embedding_dim,
            encoder_config.communication,
            head_config,
        )
        self.pi = _PiHead(head)
        self.vf = _VfHead(head)

    def message_table(
        self, obs: Dict[str, Any], chunk_rows: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """The value pass's message table for compact samples with communication
        on, None otherwise. It embeds every row once, `chunk_rows` rows at a time
        and without gradients, and returns the sorted row keys with the matching
        own embeddings, on the module's device. A chunk of the value pass then
        finds every sender in it, wherever the sender's row is in the batch."""
        encoder = self.encoder
        if encoder.lookup is None or not encoder.gnn.config.communication:
            return None
        device = next(self.parameters()).device
        rows = len(obs["row_key"])
        parts = []
        with torch.no_grad():
            for start in range(0, rows, chunk_rows):
                chunk = {
                    key: obs[key][start : start + chunk_rows].to(device)
                    for key in ("vertex", "eta", "flags")
                }
                parts.append(encoder.own_embeddings(chunk))
        keys, order = torch.sort(obs["row_key"].to(device))
        return keys, torch.cat(parts)[order]


class GNNActionMaskingRLModule(ActionMaskingTorchRLModule, GNNPPOTorchRLModule):
    """The RLModule class passed to RLlib's RLModuleSpec for every learned arm
    (training/config.py): #26's masking wrapper composed with the GNN
    architecture (#27/#28) instead of RLlib's Catalog-built default. See the
    module docstring for how."""
