"""RLlib callbacks logging the four correctness/validation invariants
(AGENTS.md, tests/unit/test_correctness_invariants.py) during training,
not just at evaluation time (issue #30).

Two of the four are checked here, every step, against the SAME
production helpers the standing invariant test suite uses -- not
reimplemented:
  - no vertex/swap conflict on the realised trace (core/agents.py::
    is_collision_free);
  - every realised storage configuration lies in the feasible set
    (core/storage_state.py::is_feasible).
The other two are enforced at the type level already, not something a
runtime callback needs to actively check: task-lifecycle-set exclusivity
follows from Task.status()/active_tasks_by_agent's own invariants
(core/tasks.py), and replay determinism is a property of a fixed seed
across two runs, not of any single run -- it's the standing test suite's
job (test_correctness_invariants.py), not a per-episode callback's.

A violation is treated as a correctness bug per AGENTS.md ("not a metric
to tune around"): this raises immediately rather than logging a metric
and continuing, since a policy that has already produced an infeasible
realised trace has broken an invariant conflict-resolution/storage-
update are supposed to make structurally impossible -- training on
through it would only hide the bug, not route around it.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from slap_mapd_coupling.core.agents import FleetState, is_collision_free
from slap_mapd_coupling.core.storage_state import is_feasible
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv


def _unwrap(env: gym.Env, env_index: int) -> WarehouseMAPDMultiAgentEnv:
    sub_env = env.envs[env_index].unwrapped  # type: ignore[attr-defined]
    if not isinstance(sub_env, WarehouseMAPDMultiAgentEnv):
        raise TypeError(
            f"CorrectnessInvariantCallbacks requires a WarehouseMAPDMultiAgentEnv "
            f"(training/env.py), got {type(sub_env).__name__}."
        )
    return sub_env


class CorrectnessInvariantCallbacks(DefaultCallbacks):
    """Checks is_collision_free/is_feasible against the realised trace
    every step of every training rollout -- see module docstring.
    is_collision_free needs a (before, after) FleetState pair (eq:
    vertexconflict/eq:swapconflict are both about a TRANSITION, not a
    single snapshot), tracked per env_index across on_episode_start/
    on_episode_step since RLlib's callback API hands us snapshots one
    step at a time, not the pair directly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._previous_fleet: dict[int, FleetState] = {}

    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module=None,
        worker=None,
        base_env: Optional["BaseEnv"] = None,
        policies=None,
        **kwargs: Any,
    ) -> None:
        if env is None:
            return
        self._previous_fleet[env_index] = _unwrap(env, env_index)._env.fleet

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module=None,
        worker=None,
        base_env: Optional["BaseEnv"] = None,
        policies=None,
        **kwargs: Any,
    ) -> None:
        if env is None:
            return
        sub_env = _unwrap(env, env_index)
        before = self._previous_fleet.get(env_index)
        after = sub_env._env.fleet
        if before is not None and not is_collision_free(before, after):
            raise RuntimeError(
                f"eq:vertexconflict/eq:swapconflict violated at t={sub_env._env.t}: "
                f"{before.locations()} -> {after.locations()} -- conflict resolution "
                "must make this structurally impossible (AGENTS.md); this is a "
                "correctness bug, not a metric to tune around."
            )
        if not is_feasible(sub_env._env.storage):
            raise RuntimeError(
                f"eq:feasiblestorage violated at t={sub_env._env.t} -- the storage rule "
                "must keep every realised configuration feasible; this is a correctness "
                "bug, not a metric to tune around."
            )
        self._previous_fleet[env_index] = after
