"""Action masking as RLlib plumbing (eq:mask, sec:method:rl).

RLlib's newer multi-agent (RLModule) API ships no action masking
built-in; it must be implemented as a custom module -- an explicit
caveat in the thesis text itself (issue #26), separate implementation
time from the training-procedure work (training/, #30). This module is
infrastructure only: it does not decide which actions are legal
(environment/spaces.py's legality_mask, #13, already does that) or build
the Dict-shaped observation an episode actually produces (the
decentralised controller/env adapter, #29, is where that wiring
belongs) -- it only knows how to consume an "action_mask" key once one
exists, and how to build the Dict observation space shape RLlib expects
it in.

Adapted from RLlib's own shipped example
(ray.rllib.examples.rl_modules.classes.action_masking_rlm), following
that pattern rather than inventing a new one, since the framework's own
docs point to it as the reference implementation for exactly this gap.
The environment is expected to expose observations as
gymnasium.spaces.Dict({"action_mask": Box(0,1,(n,)), "observations": ...}
); this RLModule extracts the mask, runs the wrapped PPO module on
"observations" alone, then adds -inf to illegal action logits before
they reach the distribution -- eq:mask's M(v,.) construction, realised
as RLlib infrastructure instead of being re-derived by hand.

Only wired for PPO (training/'s own algorithm, sec:method:training),
matching RLlib's own example's explicit scope note: "implemented for the
PPO algorithm only... not guaranteed to work with other algorithms."
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType

from slap_mapd_coupling.environment.spaces import Mask

torch, nn = try_import_torch()


def to_action_mask(mask: Mask) -> np.ndarray:
    """Converts environment/spaces.py's M(v,.) (0.0 legal / -inf illegal,
    eq:mask) into the {0.0, 1.0} array masked_observation_space's
    "action_mask" Box, and ActionMaskingTorchRLModule below, both
    expect."""
    return np.array([1.0 if value == 0.0 else 0.0 for value in mask], dtype=np.float32)


def masked_observation_space(
    base_observation_space: gym.Space, num_actions: int
) -> gym.spaces.Dict:
    """{"action_mask": Box(0,1,(n,)), "observations": base_observation_space}
    -- the Dict shape ActionMaskingTorchRLModule requires (its own
    __init__ raises if the environment doesn't provide exactly this)."""
    return gym.spaces.Dict(
        {
            "action_mask": gym.spaces.Box(
                low=0.0, high=1.0, shape=(num_actions,), dtype=np.float32
            ),
            "observations": base_observation_space,
        }
    )


class ActionMaskingRLModule(RLModule):
    """Strips the action-mask key off a Dict(action_mask, observations)
    observation space before delegating construction to the wrapped PPO
    module, which only ever sees "observations" -- mirrors RLlib's own
    ActionMaskingRLModule example exactly (see module docstring)."""

    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        if not isinstance(observation_space, gym.spaces.dict.Dict):
            raise ValueError(
                "ActionMaskingRLModule requires an observation space built by "
                "masked_observation_space(): Dict({'action_mask': Box(0,1,(n,)), "
                "'observations': <real observation space>})."
            )

        self.observation_space_with_mask = observation_space
        self.observation_space = observation_space["observations"]
        self._checked_observations = False

        super().__init__(
            observation_space=self.observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )


class ActionMaskingTorchRLModule(ActionMaskingRLModule, DefaultPPOTorchRLModule):
    """eq:mask, realised as RLlib infrastructure: mask illegal action
    logits to -inf before they reach the action distribution, for every
    RLModule forward mode PPO calls (inference/exploration/train), plus
    value-function computation (which also needs the unmasked
    "observations" alone, not the Dict wrapper)."""

    @override(DefaultPPOTorchRLModule)
    def setup(self):
        super().setup()
        # DefaultPPOTorchRLModule's networks were built against the
        # unwrapped "observations" space (ActionMaskingRLModule.__init__
        # already narrowed self.observation_space for that purpose); reset
        # it back to the Dict-with-mask space now that setup is done, so
        # callers checking self.observation_space see the real contract.
        self.observation_space = self.observation_space_with_mask

    @override(DefaultPPOTorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_exploration(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultPPOTorchRLModule)
    def _forward_train(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        outs = super()._forward_train(batch, **kwargs)
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # `_forward_train` (same training step, same batch object -- see
        # _preprocess_batch) may have already stripped "action_mask" off
        # batch[OBS] by the time this runs; RLlib's own upstream example
        # (this class's source) instead checks `isinstance(batch[OBS],
        # dict)` to detect "already preprocessed," which is unsound
        # whenever "observations" is ITSELF Dict-shaped (#28's node/
        # message/etc. tensors, not a flat Box like the upstream
        # example's) -- batch[OBS] is still a dict post-preprocessing
        # too, so that check can't tell the two states apart and
        # `_preprocess_batch` gets called a second time, popping an
        # "action_mask" key that's already gone. Checking for the key
        # itself, not just dict-ness, is the actual "not yet
        # preprocessed" condition regardless of what "observations"
        # looks like.
        if isinstance(batch[Columns.OBS], dict) and "action_mask" in batch[Columns.OBS]:
            action_mask, batch = self._preprocess_batch(batch)
            batch["action_mask"] = action_mask
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        """Splits the Dict(action_mask, observations) batch into its two
        parts, so the wrapped PPO module's forward only ever sees
        "observations"."""
        self._check_batch(batch)
        # RLlib's own TensorType doesn't include "dict of tensors" as a
        # member even though that's the real runtime type of batch[OBS]
        # here (matching the same untyped .pop() calls in RLlib's own
        # upstream example this module is adapted from).
        obs: dict = batch[Columns.OBS]  # type: ignore[assignment]
        action_mask = obs.pop("action_mask")
        batch[Columns.OBS] = obs.pop("observations")
        return action_mask, batch

    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        """eq:mask: M(v,.) = 0 for legal slots, -inf for illegal ones,
        added directly to the action distribution's input logits."""
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        batch[Columns.ACTION_DIST_INPUTS] = batch[Columns.ACTION_DIST_INPUTS] + inf_mask
        return batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> None:
        if self._checked_observations:
            return
        if "action_mask" not in batch[Columns.OBS]:
            raise ValueError(
                "No action_mask found in observation. ActionMaskingTorchRLModule "
                "requires the environment to provide observations built by "
                "masked_observation_space()."
            )
        if "observations" not in batch[Columns.OBS]:
            raise ValueError(
                "No 'observations' key found in observation. ActionMaskingTorchRLModule "
                "requires the environment to provide observations built by "
                "masked_observation_space()."
            )
        self._checked_observations = True
