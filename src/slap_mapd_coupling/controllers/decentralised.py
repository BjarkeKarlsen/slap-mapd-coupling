"""Decentralised controller: wraps the learned policy pi_theta (sec:method:rl,
sec:method:model, #27/#28/#30) behind the same Controller
Protocol section_based.py implements, so it can be run
through the identical six-stage loop and evaluator.py machinery as the
other architectures (sec:method:controllers) -- the whole point of
one shared Protocol, per docs/controller_integration.md.

Registry mismatch, flagged rather than worked around silently:
SectionBasedController registers the CLASS itself
(`@register_controller("section")` on the class), since
`controllers/registry.py`'s contract is a zero-arg `Callable[[],
Controller]` and the class is genuinely zero-arg constructible --
it is a pure function of (graph, fleet, tasks, t), nothing else.
DecentralisedController is NOT: it needs a specific trained RLModule (a
set of already-loaded weights) plus the o_i(t)-encoding config it was
trained with, and no zero-arg construction can supply either. Rather
than inventing an untrained/randomly-initialised "default" (which would
silently produce a Controller that behaves nothing like what "the
decentralised regime" means), `set_active_decentralised_controller`
below is the explicit bridge: callers construct a real
DecentralisedController from a loaded model, register it as THIS
process's active one, and get_controller("decentralised") (called by
evaluation/evaluator.py) then returns it. Calling get_controller("decentralised") before that
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
from slap_mapd_coupling.environment.spaces import action_for_slot, max_out_degree
from slap_mapd_coupling.models.local_subgraph_encoding import (
    EncodedObservation,
    LocalSubgraphEncodingConfig,
    check_communication_caps,
    encode_all_observations,
)

# sec:impl:instances: every independent random stream in this repo is
# seeded separately, offset by a fixed constant from the episode seed --
# matching environment/multi_agent_env.py's _ORDER_SEED_OFFSET and
# resolution/conflict_resolution.py's priority_permutation. This is the
# decentralised controller's own action-sampling stream's offset.
_ACTION_SAMPLING_SEED_OFFSET = 2_000_000_011


class DecentralisedController:
    """pi_theta, the learned policy: builds every robot's o_i(t) and its
    senders' views, runs the trained RLModule once for all robots, and
    samples an action for each. See module docstring for why this can't be
    zero-arg constructed like the other two controllers.

    One module call per route(), because a message is the sender's
    embedding and the receiver needs it in the same timestep (the message,
    sec:method:model). Execution stays decentralised, since each robot's row
    holds only its own view and the views of the agents it hears.

    Action selection samples from pi_theta, the masked categorical
    distribution the action mask defines, restricted to LEGAL slots only
    (renormalised over the {0,1} action_mask of the encoded observation, not
    just the raw -inf-stand-in-masked logits), so an illegal action can
    never be sampled even in principle. Sampling (rather than greedy argmax)
    is the more literal reading of pi_theta as a distribution, and is what
    PPO actually trains. Argmax is a defensible alternative for a
    "deterministic evaluation" protocol the thesis doesn't itself specify,
    flagged here rather than silently picked.

    Determinism (AGENTS.md's non-negotiable replay invariant): each agent's
    action is drawn from its OWN sampling RNG, derived from (episode_seed,
    t, agent_id), not a single mutable RNG advanced across calls. That is
    reproducible regardless of how many times or in what order route() is
    invoked, matching the "seeded separately, offset from the episode seed"
    convention every other random stream in this repo follows. episode_seed
    itself comes from reset(), not from construction: the controller is
    registered once per process (module docstring), so a constructor-time
    seed would give every episode the same sampling stream regardless of its
    own seed. Callers must call reset(episode_seed) before the first route()
    of an episode; route() raises clearly if that hasn't happened yet.

    Whether messages are used comes from experiment_config.communication,
    which must equal the communication switch the module was trained with.
    """

    def __init__(
        self,
        module: RLModule,
        encoding_config: LocalSubgraphEncodingConfig,
        experiment_config: ExperimentConfig,
    ) -> None:
        check_communication_caps(encoding_config, experiment_config.communication)
        self._module = module
        self._encoding_config = encoding_config
        self._experiment_config = experiment_config
        self._episode_seed: int | None = None

    def reset(self, episode_seed: int) -> None:
        """Primes the action-sampling stream for a new episode -- see
        class docstring."""
        self._episode_seed = episode_seed

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        if self._episode_seed is None:
            raise RuntimeError(
                "DecentralisedController.route() called before reset(episode_seed); "
                "see class docstring. evaluation/evaluator.py calls reset() at the "
                "start of every episode for a correctly configured run."
            )
        encoded = encode_all_observations(
            graph,
            fleet,
            tasks,
            t,
            self._experiment_config,
            self._encoding_config,
            d_max=max_out_degree(graph),
        )
        if not encoded:
            return {}

        agent_ids = list(encoded)
        out = self._module.forward_inference(_batch([encoded[a] for a in agent_ids]))
        logits = out[Columns.ACTION_DIST_INPUTS].detach().numpy()

        locations = fleet.locations()
        actions: dict[AgentId, Action] = {}
        for row, agent_id in enumerate(agent_ids):
            rng = np.random.default_rng(
                (self._episode_seed + _ACTION_SAMPLING_SEED_OFFSET, t, agent_id)
            )
            slot = _sample_legal_slot(logits[row], encoded[agent_id].action_mask, rng)
            actions[agent_id] = action_for_slot(graph, locations[agent_id], slot)
        return actions


def _batch(encoded: Sequence[EncodedObservation]) -> dict:
    """One batch with a row per robot, in the order given."""
    rows = [item.as_rllib_dict() for item in encoded]

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
    """Categorical sample over pi_theta: softmax of the RAW logits, with
    illegal slots zeroed out and the result renormalised -- see class
    docstring for why this, not just trusting the -inf-masked logits'
    softmax, is what guarantees "legal by construction" down to floating
    point."""
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) * action_mask
    total = probabilities.sum()
    if total <= 0.0:
        raise RuntimeError(
            "No legal action has positive probability -- the action mask always leaves slot 0 "
            "(wait) legal, so this indicates a real bug, not a normal outcome."
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
