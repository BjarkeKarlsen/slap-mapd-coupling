"""The policy function of a learned arm (sec:method:rl, sec:method:model).

`ModulePolicy` turns the observations the RLlib adapter returns into one
slot per robot with a trained or untrained RLModule. Evaluation of a learned
arm runs through the same adapter that trained it (evaluation/evaluator.py),
so the policy sees exactly the observations it saw in training. The policy
does not know how the module was loaded.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

# sec:impl:instances: every independent random stream is seeded separately,
# offset by a fixed constant from the episode seed, as the order generator and
# the priority order are. This is the action-sampling stream's offset.
_ACTION_SAMPLING_SEED_OFFSET = 2_000_000_011


class ModulePolicy:
    """pi_theta as a function from the adapter's observations to slots.

    One module call per turn, for the robots that choose in that turn. The
    decentralised arm has one turn, so all robots go through together, which a
    message needs because it is the sender's embedding in the same timestep (the
    message of sec:method:model). The centralised and section arms call once per
    turn, and each robot's row holds its own window and the earlier choices.

    A slot is sampled from the action mask's categorical distribution over the
    legal slots only, the softmax of the raw logits with illegal slots zeroed
    and the result renormalised. An illegal slot can never be drawn, even in
    principle. Whether evaluation samples or takes the most likely slot is
    still open, so this keeps sampling, which is what PPO trains.

    Each robot's draw comes from its own stream, seeded with
    (episode_seed + offset, t, agent_id), so a replay under a fixed seed is
    bit for bit, however often or in what order the policy is called. The
    episode seed comes from `reset`, which must be called before the first
    call of an episode. Otherwise `__call__` raises.

    With `record_runtime` given to `reset`, every call (one per turn) records its time, the
    batch building, the forward pass and the sampling, through that function,
    which is the routing runtime of the episode log.
    """

    def __init__(self, module: RLModule) -> None:
        self._module = module
        self._episode_seed: Optional[int] = None
        self._record_runtime: Optional[Callable[[float], None]] = None

    def reset(
        self, episode_seed: int, record_runtime: Optional[Callable[[float], None]] = None
    ) -> None:
        """Primes the sampling stream for a new episode and sets where the
        call times go. None stops recording."""
        self._episode_seed = episode_seed
        self._record_runtime = record_runtime

    def __call__(self, observations: Mapping[str, Mapping[str, Any]], t: int) -> dict[str, int]:
        """One legal slot for every robot in `observations`, keyed like it. The
        observations are what the adapter returns for one turn, any subset of the robots."""
        if self._episode_seed is None:
            raise RuntimeError("ModulePolicy was called before reset(episode_seed).")
        start = time.perf_counter()
        agent_ids = list(observations)
        if not agent_ids:
            return {}
        # A fresh batch per call, since forward_inference can change it in place.
        batch = _batch([observations[a] for a in agent_ids])
        with torch.no_grad():
            out = self._module.forward_inference(batch)
        logits = out[Columns.ACTION_DIST_INPUTS].detach().numpy()

        slots: dict[str, int] = {}
        for row, agent_id in enumerate(agent_ids):
            rng = np.random.default_rng(
                (self._episode_seed + _ACTION_SAMPLING_SEED_OFFSET, t, int(agent_id))
            )
            slots[agent_id] = _sample_legal_slot(
                logits[row], np.asarray(observations[agent_id]["action_mask"]), rng
            )
        if self._record_runtime is not None:
            self._record_runtime(time.perf_counter() - start)
        return slots


def _batch(rows: Sequence[Mapping[str, Any]]) -> dict:
    """One batch with a row per robot, in the order given. Each row is the
    {"action_mask", "observations"} dict the adapter returns."""

    def stack(key: str, nested: bool) -> torch.Tensor:
        source = [row["observations"][key] if nested else row[key] for row in rows]
        return torch.as_tensor(np.stack(source))

    return {
        Columns.OBS: {
            "action_mask": stack("action_mask", nested=False),
            "observations": {key: stack(key, nested=True) for key in rows[0]["observations"]},
        }
    }


def _sample_legal_slot(
    logits: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
) -> int:
    """A categorical sample over the legal slots. The softmax of the raw
    logits has illegal slots zeroed and is renormalised, so legality holds down
    to floating point and does not rely on the masked logits alone."""
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) * action_mask
    total = probabilities.sum()
    if total <= 0.0:
        raise RuntimeError(
            "No legal action has positive probability. The action mask always leaves slot 0 "
            "(wait) legal, so this is a bug."
        )
    probabilities = probabilities / total
    return int(rng.choice(len(probabilities), p=probabilities))
