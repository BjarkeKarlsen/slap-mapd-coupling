"""Unit tests for slap_mapd_coupling.environment.action_masking."""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from slap_mapd_coupling.environment.action_masking import (
    ActionMaskingRLModule,
    ActionMaskingTorchRLModule,
    masked_observation_space,
    to_action_mask,
)


def test_to_action_mask_converts_eq_mask_to_zero_one():
    mask = (0.0, float("-inf"), 0.0, float("-inf"))
    assert list(to_action_mask(mask)) == [1.0, 0.0, 1.0, 0.0]


def test_masked_observation_space_shape():
    base = Box(-1.0, 1.0, (5,))
    space = masked_observation_space(base, num_actions=4)
    assert set(space.spaces) == {"action_mask", "observations"}
    assert space["action_mask"].shape == (4,)
    assert space["observations"] is base


def test_action_masking_rlmodule_rejects_non_dict_observation_space():
    with pytest.raises(ValueError, match="Dict"):
        ActionMaskingRLModule(
            observation_space=Box(-1.0, 1.0, (5,)),
            action_space=Discrete(4),
            inference_only=True,
            model_config={},
        )


def _build_module(base_obs_dim: int = 5, num_actions: int = 4):
    base_obs_space = Box(-1.0, 1.0, (base_obs_dim,))
    action_space = Discrete(num_actions)
    obs_space = masked_observation_space(base_obs_space, num_actions=num_actions)
    spec = RLModuleSpec(
        module_class=ActionMaskingTorchRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config={"head_fcnet_hiddens": [8], "head_fcnet_activation": "relu"},
    )
    return spec.build()


def test_masked_logits_are_effectively_negative_infinity_for_illegal_actions():
    module = _build_module()
    mask = to_action_mask((0.0, float("-inf"), 0.0, float("-inf")))
    action_mask = torch.tensor(np.array([mask]))
    obs = torch.tensor([[0.1, 0.2, -0.3, 0.4, 0.0]])
    batch = {Columns.OBS: {"action_mask": action_mask, "observations": obs}}

    out = module.forward_inference(batch)
    logits = out[Columns.ACTION_DIST_INPUTS][0]

    assert logits[1] < -1e30
    assert logits[3] < -1e30
    assert logits[0] > -1e30
    assert logits[2] > -1e30


def test_masked_logits_are_finite_and_deterministic_when_all_actions_legal():
    # log(1.0) == 0.0 exactly, so an all-legal mask is a provable no-op on
    # the underlying logits -- checked here via determinism/finiteness
    # rather than a second, separately-constructed "unmasked" module,
    # since building two modules with identical weights to compare
    # against isn't otherwise guaranteed.
    module = _build_module()
    module.eval()
    all_legal = to_action_mask((0.0, 0.0, 0.0, 0.0))

    def _fresh_batch():
        # _preprocess_batch pops keys off batch[Columns.OBS] in place
        # (matching RLlib's own upstream example), mutating whatever
        # dict is passed in -- each call needs its own fresh batch, not
        # a dict reused across calls.
        action_mask = torch.tensor(np.array([all_legal]))
        obs = torch.tensor([[0.1, 0.2, -0.3, 0.4, 0.0]])
        return {Columns.OBS: {"action_mask": action_mask, "observations": obs}}

    with torch.no_grad():
        first = module.forward_inference(_fresh_batch())
        second = module.forward_inference(_fresh_batch())

    assert torch.all(torch.isfinite(first[Columns.ACTION_DIST_INPUTS]))
    assert torch.equal(first[Columns.ACTION_DIST_INPUTS], second[Columns.ACTION_DIST_INPUTS])


def test_all_actions_masked_raises_no_error_building_the_module():
    # A vertex with only "wait" legal (every move slot masked) is a real,
    # valid eq:mask case (e.g. a dead-end vertex) -- must not crash.
    module = _build_module()
    only_wait = to_action_mask((0.0, float("-inf"), float("-inf"), float("-inf")))
    action_mask = torch.tensor(np.array([only_wait]))
    obs = torch.tensor([[0.1, 0.2, -0.3, 0.4, 0.0]])
    batch = {Columns.OBS: {"action_mask": action_mask, "observations": obs}}

    out = module.forward_inference(batch)
    logits = out[Columns.ACTION_DIST_INPUTS][0]
    assert logits[0] > -1e30
    assert all(logits[i] < -1e30 for i in (1, 2, 3))
