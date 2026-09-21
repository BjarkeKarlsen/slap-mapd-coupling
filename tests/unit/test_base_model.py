"""Unit tests for slap_mapd_coupling.models.base_model."""

import torch

from slap_mapd_coupling.models.base_model import (
    GNNPolicyValueModel,
    PolicyValueHead,
    PolicyValueHeadConfig,
)
from slap_mapd_coupling.models.gnn_encoder import (
    GNNEncoder,
    GNNEncoderConfig,
    MESSAGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
)


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


def test_gnn_policy_value_model_end_to_end_shapes():
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=2, hidden_width=8))
    model = GNNPolicyValueModel(
        encoder, head_config=PolicyValueHeadConfig(hidden_width=8, num_actions=4)
    )

    batch_size, n, m = 2, 5, 3
    node_features = torch.randn(batch_size, n, NODE_FEATURE_DIM)
    node_mask = torch.ones(batch_size, n)
    adjacency = torch.zeros(batch_size, n, n)
    own_index = torch.zeros(batch_size, dtype=torch.long)
    message_features = torch.randn(batch_size, m, MESSAGE_FEATURE_DIM)
    message_mask = torch.ones(batch_size, m)
    congestion = torch.zeros(batch_size)

    logits, value = model(
        node_features, node_mask, adjacency, own_index, message_features, message_mask, congestion
    )
    assert logits.shape == (batch_size, 4)
    assert value.shape == (batch_size,)


def test_gnn_policy_value_model_logits_are_unmasked_raw_values():
    """#28's heads must NOT apply eq:mask themselves -- #26's
    ActionMaskingTorchRLModule owns that -- so nothing here should ever
    force a logit to -inf on its own."""
    encoder = GNNEncoder(GNNEncoderConfig(num_rounds=0, hidden_width=8))
    model = GNNPolicyValueModel(
        encoder, head_config=PolicyValueHeadConfig(hidden_width=8, num_actions=3)
    )

    node_features = torch.randn(1, 2, NODE_FEATURE_DIM)
    node_mask = torch.ones(1, 2)
    adjacency = torch.zeros(1, 2, 2)
    own_index = torch.zeros(1, dtype=torch.long)
    message_features = torch.zeros(1, 1, MESSAGE_FEATURE_DIM)
    message_mask = torch.zeros(1, 1)
    congestion = torch.zeros(1)

    logits, _ = model(
        node_features, node_mask, adjacency, own_index, message_features, message_mask, congestion
    )
    assert torch.isfinite(logits).all()
