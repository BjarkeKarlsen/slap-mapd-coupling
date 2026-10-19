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

import math
import numbers
from typing import Any, Mapping, Optional

import gymnasium as gym
from ray.rllib.callbacks.callbacks import RLlibCallback

from slap_mapd_coupling.core.agents import FleetState, is_collision_free
from slap_mapd_coupling.core.storage_state import is_feasible
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv


def _unwrap(env: gym.Env, env_index: int) -> WarehouseMAPDMultiAgentEnv:
    sub_env = env.envs[env_index].unwrapped  # type: ignore[attr-defined]
    if not isinstance(sub_env, WarehouseMAPDMultiAgentEnv):
        raise TypeError(
            f"CorrectnessInvariantCallbacks requires a WarehouseMAPDMultiAgentEnv "
            f"(training/env.py), got {type(sub_env).__name__}."
        )
    return sub_env


class CorrectnessInvariantCallbacks(RLlibCallback):
    """Checks is_collision_free and is_feasible on the realised trace at
    every timestep of every training rollout, not at the turns inside one. The
    vertex rule and the swap rule are about a transition, so the fleet before
    the step is kept per env_index, since RLlib hands over one snapshot at a time.
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
        **kwargs: Any,
    ) -> None:
        if env is None:
            return
        self._previous_fleet[env_index] = _unwrap(env, env_index).core.fleet

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module=None,
        **kwargs: Any,
    ) -> None:
        if env is None:
            return
        sub_env = _unwrap(env, env_index)
        if not sub_env.timestep_completed:
            return  # a turn inside a timestep, the fleet has not moved
        before = self._previous_fleet.get(env_index)
        after = sub_env.core.fleet
        if before is not None and not is_collision_free(before, after):
            raise RuntimeError(
                f"the vertex rule or the swap rule is broken at t={sub_env.core.t}, "
                f"{before.locations()} to {after.locations()}. Conflict resolution must "
                "make this impossible, so this is a correctness bug."
            )
        if not is_feasible(sub_env.core.storage):
            raise RuntimeError(
                f"the capacity constraint is broken at t={sub_env.core.t}. The storage rule "
                "must keep every realised configuration feasible, so this is a correctness bug."
            )
        self._previous_fleet[env_index] = after


def progress_metrics(env: WarehouseMAPDEnv) -> dict[str, float]:
    """Where the episode in progress stands, for the training log. `env` is
    the simulator core, the adapter's `core`.

    `tasks_completed`, `tasks_released`, `backlog` and `steps_since_last_completion` are
    snapshots of the episode in progress, taken at the end of a sampled fragment and averaged
    over the environments. The per-episode totals are the `episode_*` keys logged at the end of
    every episode (`episode_metrics`).

    `steps_since_last_completion` is t minus the latest finish time, or t
    when no task has finished yet. A value that keeps growing is a stall.
    """
    finish_times = [task.finish_time for task in env.tasks if task.finish_time is not None]
    return {
        "tasks_completed": float(len(finish_times)),
        "tasks_released": float(len(env.tasks)),
        "backlog": float(env.log.backlog_by_t[-1]),
        "steps_since_last_completion": float(env.t - max(finish_times, default=0)),
        "episode_step": float(env.t),
    }


def episode_metrics(env: WarehouseMAPDEnv) -> dict[str, float]:
    """What one finished episode came to, for the training log. `env` is the simulator
    core, the adapter's `core`, read when the episode ends.

    The five `return_*` values are the terms of the reward summed over the robots and the
    steps of the episode (environment/reward_function.py::RewardTerms, each signed). They add
    up to the episode return RLlib reports, also when the credit is a group mean, which
    only moves reward between robots. `override_rate` is the share of robot-steps in which a
    robot was overridden by conflict resolution. `episode_max_agent_override_share` is the
    share of all the episode's overrides that fell on the single most overridden robot, and
    is 0 when there were none. A planner run has no reward, so all of these are 0 for it.
    """
    log = env.log
    finish_times = [task.finish_time for task in env.tasks if task.finish_time is not None]
    overrides = sum(log.overrides_by_agent.values())
    agent_steps = env.config.num_agents * env.t
    return {
        "return_shaping": sum(log.return_shaping_by_agent.values()),
        "return_deliver": sum(log.return_delivery_by_agent.values()),
        "return_override": sum(log.return_override_by_agent.values()),
        "return_wait": sum(log.return_wait_by_agent.values()),
        "return_congestion": sum(log.return_congestion_by_agent.values()),
        "override_rate": overrides / agent_steps if agent_steps else 0.0,
        "episode_tasks_completed": float(len(finish_times)),
        "episode_tasks_released": float(len(env.tasks)),
        "episode_last_completion_t": float(max(finish_times, default=0)),
        "episode_max_agent_override_share": (
            max(log.overrides_by_agent.values()) / overrides if overrides else 0.0
        ),
    }


class StandingStillCounter:
    """Share of agent-steps in which an agent did not change vertex."""

    def __init__(self) -> None:
        self._still = 0
        self._total = 0

    def update(self, before: FleetState, after: FleetState) -> None:
        after_at = after.locations()
        for agent_id, at in before.locations().items():
            self._total += 1
            self._still += after_at[agent_id] == at

    def fraction(self) -> Optional[float]:
        return None if self._total == 0 else self._still / self._total

    def reset(self) -> None:
        self._still = 0
        self._total = 0


# The training return under a top-level key of the result. Tune ranks checkpoints by a score only
# when the score is a top-level key, and RLlib nests the return under "env_runners", so
# on_train_result copies it here (see training/trainer.py::CheckpointSettings).
EPISODE_RETURN_KEY = "episode_return_mean"

# Warehouse timesteps of the finished episodes, summed over the runners and the run. Neither
# RLlib step counter gives it, since both count sub-steps or observations, so this is what
# the training budget is checked against (training/trainer.py).
TIMESTEPS_KEY = "timesteps_sampled_lifetime"


class TrainingProgressCallbacks(CorrectnessInvariantCallbacks):
    """The invariant checks plus progress numbers in every iteration.

    Every iteration ends exactly one episode per runner (training/config.py::build_ppo_config),
    so the episode keys exist in every iteration. The values logged from the environment's own state at the end of every sampled
    fragment, under `result["env_runners"]`, are the snapshots `tasks_completed`,
    `tasks_released`, `backlog`, `steps_since_last_completion` and `episode_step`, averaged
    over the environments (see `progress_metrics`).

    The exact per-episode totals are the `episode_*` keys, together with `return_shaping`,
    `return_deliver`, `return_override`, `return_wait`, `return_congestion` and `override_rate`,
    logged when an episode ends (see `episode_metrics`).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._standing_still = StandingStillCounter()

    def on_episode_step(
        self, *, env: Optional[gym.Env] = None, env_index: int, **kwargs: Any
    ) -> None:
        before = self._previous_fleet.get(env_index)
        super().on_episode_step(env=env, env_index=env_index, **kwargs)
        if env is not None and before is not None:
            sub_env = _unwrap(env, env_index)
            if sub_env.timestep_completed:
                self._standing_still.update(before, sub_env.core.fleet)

    def on_train_result(
        self, *, algorithm=None, metrics_logger=None, result: dict, **kwargs: Any
    ) -> None:
        """Copies the mean episode return to the top level of the result, which is where Tune
        looks for the score of a checkpoint. An iteration without a finished episode has none.

        Also stops the run if the iteration has fewer episodes than runners. That happens when
        RLlib's sample timeout drops a slow runner's episode, and the arms would then train on
        different batches (training/config.py::build_ppo_config)."""
        if algorithm is not None:
            expected = max(algorithm.config.num_env_runners, 1)
            episodes = result.get("env_runners", {}).get("num_episodes")
            episodes = int(episodes) if _is_number(episodes) else 0
            if episodes < expected:
                raise RuntimeError(
                    f"This iteration has {episodes} episodes, but there are {expected} runners. "
                    "An iteration must be one complete episode per runner, so set "
                    "sample_timeout_s=None (build_ppo_config does)."
                )
        value = result.get("env_runners", {}).get("episode_return_mean")
        if _is_number(value):
            result[EPISODE_RETURN_KEY] = float(value)
        timesteps = result.get("env_runners", {}).get(TIMESTEPS_KEY)
        if _is_number(timesteps):
            result[TIMESTEPS_KEY] = int(timesteps)

    def on_sample_end(self, *, env_runner=None, metrics_logger=None, **kwargs: Any) -> None:
        if env_runner is None or metrics_logger is None or env_runner.env is None:
            return
        for env_index in range(len(env_runner.env.envs)):
            core = _unwrap(env_runner.env, env_index).core
            for name, value in progress_metrics(core).items():
                metrics_logger.log_value(name, value, reduce="mean")
        fraction = self._standing_still.fraction()
        if fraction is not None:
            metrics_logger.log_value("standing_still_fraction", fraction, reduce="mean")
        self._standing_still.reset()

    def on_episode_end(
        self, *, metrics_logger=None, env: Optional[gym.Env] = None, env_index: int, **kwargs: Any
    ) -> None:
        if env is None or metrics_logger is None:
            return
        core = _unwrap(env, env_index).core
        for name, value in episode_metrics(core).items():
            metrics_logger.log_value(name, value, reduce="mean")
        metrics_logger.log_value(TIMESTEPS_KEY, core.t, reduce="lifetime_sum")


_COUNT_KEYS = (
    "tasks_completed",
    "tasks_released",
    "backlog",
    "steps_since_last_completion",
)


# Logged once per finished episode.
_EPISODE_KEYS = (
    "return_shaping",
    "return_deliver",
    "return_override",
    "return_wait",
    "return_congestion",
    "override_rate",
    "episode_tasks_completed",
    "episode_tasks_released",
    "episode_max_agent_override_share",
)


def _is_number(value: Any) -> bool:
    # numbers.Real, not (int, float), so a numpy float32 counts as a number too.
    return isinstance(value, numbers.Real) and math.isfinite(value)


def iteration_record(i: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """One iteration's progress numbers as a JSON-ready dict, for a file kept beside a run.

    The keys are the ones `format_iteration` prints, then `episode_step`, then the per-episode
    keys of `_EPISODE_KEYS`. The count keys and `episode_step` are snapshots of the episode in
    progress, averaged over the environments (see `progress_metrics`). The exact totals are the
    `_EPISODE_KEYS`. A value that is missing or not a finite number is None.
    """
    runners = result.get("env_runners", {})
    keys = (
        "episode_return_mean",
        *_COUNT_KEYS,
        "standing_still_fraction",
        "episode_step",
        *_EPISODE_KEYS,
    )
    record: dict[str, Any] = {"iteration": i}
    for key in keys:
        value = runners.get(key)
        record[key] = float(value) if _is_number(value) else None
    return record


def format_iteration(i: int, n: int, result: Mapping[str, Any]) -> str:
    """One line per training iteration, in plain numbers (no exponents).

    The episode return mean comes first because it is the number PPO
    reports. An iteration in which no episode ended has none, and then the line
    shows 0. A real return of 0 prints as 0.0, so the two can still be told apart.
    """
    runners = result.get("env_runners", {})
    episode_return = runners.get("episode_return_mean")
    parts = [
        f"episode_return_mean={episode_return:.1f}"
        if _is_number(episode_return)
        else "episode_return_mean=0"
    ]
    for key in _COUNT_KEYS:
        value = runners.get(key)
        parts.append(f"{key}={value:.0f}" if _is_number(value) else f"{key}=n/a")
    fraction = runners.get("standing_still_fraction")
    parts.append(
        f"standing_still_fraction={fraction:.2f}"
        if _is_number(fraction)
        else "standing_still_fraction=n/a"
    )
    return f"  iter {i}/{n}: " + " ".join(parts)
