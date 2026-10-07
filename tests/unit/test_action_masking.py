"""Unit tests for slap_mapd_coupling.environment.action_masking."""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from torch import nn

from slap_mapd_coupling.environment.action_masking import (
    ActionMaskingRLModule,
    ActionMaskingTorchRLModule,
    masked_observation_space,
    to_action_mask,
)


class _FlattenDictEncoder(nn.Module):
    """A tiny Catalog-free encoder for a Dict-shaped "observations" space
    -- exercises the same "observations is itself Dict-shaped" scenario
    models/rl_module.py's GNN encoder hits (#30), without this test
    depending on models/ (environment/ doesn't depend on it either)."""

    def __init__(self, keys: tuple, hidden: int) -> None:
        super().__init__()
        self.keys = keys
        self.net = nn.LazyLinear(hidden)

    def forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        flat = torch.cat([obs[k].reshape(obs[k].shape[0], -1) for k in self.keys], dim=-1)
        z = self.net(flat)
        return {ENCODER_OUT: {ACTOR: z, CRITIC: z}}


class _DictObsPPOModule(DefaultPPOTorchRLModule):
    """DefaultPPOTorchRLModule with a hand-built (Catalog-free) network,
    the same "override setup(), don't call super().setup()" pattern
    models/rl_module.py's GNNPPOTorchRLModule uses -- RLlib's own default
    Catalog can't build a network for an arbitrary Dict observation
    space at all (see models/rl_module.py's docstring for why this repo
    needs a custom encoder regardless)."""

    @override(DefaultPPOTorchRLModule)
    def setup(self):
        assert isinstance(self.observation_space, Dict)
        keys = tuple(self.observation_space.spaces)
        self.encoder = _FlattenDictEncoder(keys, hidden=8)
        self.pi = nn.LazyLinear(self.action_space.n)
        self.vf = nn.LazyLinear(1)


class _DictObsActionMaskingModule(ActionMaskingTorchRLModule, _DictObsPPOModule):
    """Same composition models/rl_module.py's GNNActionMaskingRLModule
    uses (ActionMaskingTorchRLModule's forward/masking logic + a
    Catalog-free DefaultPPOTorchRLModule subclass, with MRO placing the
    latter where ActionMaskingTorchRLModule expects plain
    DefaultPPOTorchRLModule), kept local to this test so environment/
    doesn't gain a dependency on models/."""


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


def test_compute_values_skips_redundant_preprocessing_for_dict_shaped_observations():
    """Regression: at train time, RLlib's learner connector pipeline
    hands compute_values a batch where batch[Columns.OBS] is already the
    bare "observations" (itself Dict-shaped for #28's node/message
    tensors, unlike the flat-Box upstream action-masking example this
    module is adapted from) and "action_mask" is a separate TOP-LEVEL
    batch key -- never nested inside batch[Columns.OBS] at all by this
    point. RLlib's own upstream `isinstance(batch[OBS], dict)` check for
    "already preprocessed" can't distinguish this from "not yet
    preprocessed," since Dict-shaped observations are still a dict
    either way, and used to KeyError trying to pop an "action_mask" key
    that was never nested there to begin with."""
    base_obs_space = Dict({"x": Box(-1.0, 1.0, (3,)), "y": Box(-1.0, 1.0, (2,))})
    action_space = Discrete(4)
    obs_space = masked_observation_space(base_obs_space, num_actions=4)
    spec = RLModuleSpec(
        module_class=_DictObsActionMaskingModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config={},
    )
    module = spec.build()
    # A real inference-time call earlier in the same rollout would have
    # set this; compute_values must not depend on batch[OBS] shape to
    # decide whether preprocessing already happened.
    module._checked_observations = True

    batch = {
        Columns.OBS: {"x": torch.zeros(1, 3), "y": torch.zeros(1, 2)},
        "action_mask": torch.ones(1, 4),
    }
    embeddings = torch.zeros(1, 8)

    values = module.compute_values(batch, embeddings=embeddings)
    assert values.shape == (1,)
