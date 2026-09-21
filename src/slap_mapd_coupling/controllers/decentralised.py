"""Decentralised controller: wraps the learned policy pi_theta(.|o_i(t))
(sec:method:rl/sec:method:model, #27/#28/#30) behind the same Controller
Protocol centralised.py/section_based.py implement, so it can be run
through the identical six-stage loop and evaluator.py machinery as the
other two architectures (sec:method:controllers) -- the whole point of
one shared Protocol, per docs/controller_integration.md.

Registry mismatch, flagged rather than worked around silently:
CentralisedController/SectionBasedController register the CLASS itself
(`@register_controller("centralised")` on the class), since
`controllers/registry.py`'s contract is a zero-arg `Callable[[],
Controller]` and both classes are genuinely zero-arg constructible --
they're pure functions of (graph, fleet, tasks, t), nothing else.
DecentralisedController is NOT: it needs a specific trained RLModule (a
set of already-loaded weights) plus the o_i(t)-encoding config it was
trained with, and no zero-arg construction can supply either. Rather
than inventing an untrained/randomly-initialised "default" (which would
silently produce a Controller that behaves nothing like what "the
decentralised regime" means), `set_active_decentralised_controller`
below is the explicit bridge: callers construct a real
DecentralisedController from a loaded model, register it as THIS
process's active one, and get_controller("decentralised") (used
unchanged by environment/multi_agent_env.py and evaluation/evaluator.py)
then returns it. Calling get_controller("decentralised") before that
raises with this explanation rather than silently returning something
meaningless.
"""

from __future__ import annotations

from typing import NoReturn, Sequence

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

from slap_mapd_coupling.controllers.registry import register_controller
from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.environment.spaces import action_for_slot, legality_mask, max_out_degree
from slap_mapd_coupling.models.local_subgraph_encoding import (
    EncodedObservation,
    LocalSubgraphEncodingConfig,
    encode_observation,
)

# sec:impl:instances: every independent random stream in this repo is
# seeded separately, offset by a fixed constant from the episode seed --
# matching environment/multi_agent_env.py's _ORDER_SEED_OFFSET and
# resolution/conflict_resolution.py's priority_permutation. This is the
# decentralised controller's own action-sampling stream's offset.
_ACTION_SAMPLING_SEED_OFFSET = 2_000_000_011


class DecentralisedController:
    """pi_theta(.|o_i(t)): builds o_i(t) (#25), encodes it (#28), runs
    the trained RLModule, and samples an action -- see module docstring
    for why this can't be zero-arg constructed like the other two
    controllers.

    Action selection samples from pi_theta(.|o_i(t)), the masked
    categorical distribution eq:mask defines, restricted to LEGAL slots
    only (renormalised over encode_observation's own {0,1} action_mask,
    not just the raw -inf-stand-in-masked logits) -- so an illegal
    action can never be sampled even in principle, matching eq:mask's
    "legal by construction," not "legal with overwhelming probability."
    Sampling (rather than greedy argmax) is the more literal reading of
    pi_theta as a distribution, and is what PPO actually trains; argmax
    is a defensible alternative for a "deterministic evaluation"
    protocol the thesis doesn't itself specify, flagged here rather than
    silently picked.

    Determinism (AGENTS.md's non-negotiable replay invariant): each
    route() call derives its OWN sampling RNG from (seed, t, agent_id),
    not a single mutable RNG advanced across calls -- reproducible
    regardless of how many times or in what order route() is invoked,
    matching the "seeded separately, offset from the episode seed"
    convention every other random stream in this repo already follows.
    """

    def __init__(
        self,
        module: RLModule,
        encoding_config: LocalSubgraphEncodingConfig,
        experiment_config: ExperimentConfig,
        seed: int,
    ) -> None:
        self._module = module
        self._encoding_config = encoding_config
        self._experiment_config = experiment_config
        self._seed = seed

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        d_max = max_out_degree(graph)
        locations = fleet.locations()
        actions: dict[AgentId, Action] = {}
        for agent_id, location in locations.items():
            observation = build_observation(
                graph, fleet, tasks, agent_id, t, self._experiment_config
            )
            mask = legality_mask(graph, location, d_max=d_max)
            encoded = encode_observation(graph, location, observation, mask, self._encoding_config)

            out = self._module.forward_inference(_single_agent_batch(encoded))
            logits = out[Columns.ACTION_DIST_INPUTS][0].detach().numpy()

            rng = np.random.default_rng((self._seed + _ACTION_SAMPLING_SEED_OFFSET, t, agent_id))
            slot = _sample_legal_slot(logits, encoded.action_mask, rng)
            actions[agent_id] = action_for_slot(graph, location, slot)
        return actions


def _single_agent_batch(encoded: EncodedObservation) -> dict:
    def to_tensor(value):
        return torch.as_tensor(value).unsqueeze(0)

    return {
        Columns.OBS: {
            "action_mask": to_tensor(encoded.action_mask),
            "observations": {
                "node_features": to_tensor(encoded.node_features),
                "node_mask": to_tensor(encoded.node_mask),
                "adjacency": to_tensor(encoded.adjacency),
                "own_index": to_tensor(encoded.own_index),
                "message_features": to_tensor(encoded.message_features),
                "message_mask": to_tensor(encoded.message_mask),
                "congestion": to_tensor(np.float32(encoded.congestion)),
            },
        }
    }


def _sample_legal_slot(
    logits: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
) -> int:
    """Categorical sample over pi_theta(.|o_i(t)): softmax of the RAW
    logits, with illegal slots zeroed out and the result renormalised --
    see class docstring for why this, not just trusting the -inf-masked
    logits' softmax, is what actually guarantees eq:mask's "legal by
    construction" down to floating point."""
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) * action_mask
    total = probabilities.sum()
    if total <= 0.0:
        raise RuntimeError(
            "No legal action has positive probability -- eq:mask guarantees slot 0 "
            "(wait) is always legal, so this indicates a real bug, not a normal outcome."
        )
    probabilities = probabilities / total
    return int(rng.choice(len(probabilities), p=probabilities))


_active_controller: DecentralisedController | None = None


def set_active_decentralised_controller(controller: DecentralisedController) -> None:
    """Registers `controller` as what get_controller("decentralised")
    returns for the rest of this process -- see module docstring for why
    this indirection exists. Call this once, before constructing any
    WarehouseMAPDEnv/running evaluator.run_episode with
    config.controller="decentralised"."""
    global _active_controller
    _active_controller = controller


@register_controller("decentralised")
def _get_active_decentralised_controller() -> NoReturn | DecentralisedController:
    if _active_controller is None:
        raise RuntimeError(
            "controller='decentralised' has no default, checkpoint-free construction "
            "(see controllers/decentralised.py's module docstring): it needs a specific "
            "trained RLModule plus the o_i(t)-encoding config it was trained with. Call "
            "controllers.decentralised.set_active_decentralised_controller(...) first, "
            "with a DecentralisedController built from an already-loaded RLModule -- see "
            "examples/04_decentralised_training.py."
        )
    return _active_controller
