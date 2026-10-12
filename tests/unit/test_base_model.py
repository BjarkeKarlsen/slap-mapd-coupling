"""Unit tests for slap_mapd_coupling.models.base_model."""

import pytest
import torch

from slap_mapd_coupling.models.base_model import (
    GNNPolicyValueModel,
    PolicyValueHead,
    PolicyValueHeadConfig,
)
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM, GNNEncoder, GNNEncoderConfig


def test_policy_value_head_output_shapes():
    head = PolicyValueHead(
        input_dim=11, config=PolicyValueHeadConfig(hidden_width=8, num_actions=5)
    )
    z = torch.randn(3, 11)
    logits, value = head(z)
    assert logits.shape == (3, 5)
    assert value.shape == (3,)


def test_policy_value_head_gradients_flow():
    head = PolicyValueHead(input_dim=6, config=PolicyValueHeadConfig(hidden_width=4, num_actions=3))
    z = torch.randn(2, 6, requires_grad=True)
    logits, value = head(z)
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters())


@pytest.mark.parametrize("communication", [False, True])
def test_gnn_policy_value_model_end_to_end_shapes(communication):
    config = GNNEncoderConfig(num_rounds=2, hidden_width=8, communication=communication)
    model = GNNPolicyValueModel(
        GNNEncoder(config), head_config=PolicyValueHeadConfig(hidden_width=8, num_actions=4)
    )

    batch_size, n, m = 2, 5, 3
    senders = {}
    if communication:
        senders = dict(
            sender_node_features=torch.randn(batch_size, m, n, NODE_FEATURE_DIM),
            sender_node_mask=torch.ones(batch_size, m, n),
            sender_edge_index=torch.zeros(batch_size, m, 2, 2, dtype=torch.long),
            sender_edge_mask=torch.zeros(batch_size, m, 2),
            sender_own_index=torch.zeros(batch_size, m, dtype=torch.long),
            sender_mask=torch.ones(batch_size, m),
        )

    logits, value = model(
        torch.randn(batch_size, n, NODE_FEATURE_DIM),
        torch.ones(batch_size, n),
        torch.zeros(batch_size, 2, 2, dtype=torch.long),
        torch.zeros(batch_size, 2),
        torch.zeros(batch_size, dtype=torch.long),
        torch.zeros(batch_size),
        **senders,
    )
    assert logits.shape == (batch_size, 4)
    assert value.shape == (batch_size,)


def test_gnn_policy_value_model_logits_are_unmasked_raw_values():
    """#28's heads must NOT apply the action mask themselves -- #26's
    ActionMaskingTorchRLModule owns that -- so nothing here should ever
    force a logit to -inf on its own."""
    config = GNNEncoderConfig(num_rounds=0, hidden_width=8, communication=False)
    model = GNNPolicyValueModel(
        GNNEncoder(config), head_config=PolicyValueHeadConfig(hidden_width=8, num_actions=3)
    )

    logits, _ = model(
        torch.randn(1, 2, NODE_FEATURE_DIM),
        torch.ones(1, 2),
        torch.zeros(1, 2, 2, dtype=torch.long),
        torch.zeros(1, 2),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1),
    )
    assert torch.isfinite(logits).all()
