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
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

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


class CorrectnessInvariantCallbacks(DefaultCallbacks):
    """Checks is_collision_free and is_feasible on the realised trace at
    every step of every training rollout. The vertex rule and the swap rule
    are about a transition, so the fleet before the step is kept per
    env_index, since RLlib hands over one snapshot at a time.
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
        worker=None,
        base_env: Optional["BaseEnv"] = None,
        policies=None,
        **kwargs: Any,
    ) -> None:
        if env is None:
            return
        sub_env = _unwrap(env, env_index)
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


class TrainingProgressCallbacks(CorrectnessInvariantCallbacks):
    """The invariant checks plus progress numbers in every iteration.

    An iteration samples fewer steps than an episode lasts, so the episode
    return is missing most iterations. These values are logged from the
    environment's own state at the end of every fragment instead, under
    `result["env_runners"]`.
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
            self._standing_still.update(before, _unwrap(env, env_index).core.fleet)

    def on_sample_end(self, *, env_runner=None, metrics_logger=None, **kwargs: Any) -> None:
        if env_runner is None or metrics_logger is None or env_runner.env is None:
            return
        for env_index in range(len(env_runner.env.envs)):
            values = progress_metrics(_unwrap(env_runner.env, env_index).core)
            for name, value in values.items():
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
        env_state = _unwrap(env, env_index).core
        finish_times = [t.finish_time for t in env_state.tasks if t.finish_time is not None]
        metrics_logger.log_value("episode_tasks_completed", float(len(finish_times)), reduce="mean")
        metrics_logger.log_value(
            "episode_last_completion_t", float(max(finish_times, default=0)), reduce="mean"
        )


_COUNT_KEYS = (
    "tasks_completed",
    "tasks_released",
    "backlog",
    "steps_since_last_completion",
)


def _is_number(value: Any) -> bool:
    # numbers.Real, not (int, float), so a numpy float32 counts as a number too.
    return isinstance(value, numbers.Real) and math.isfinite(value)


def iteration_record(i: int, result: Mapping[str, Any]) -> dict[str, Any]:
    """One iteration's progress numbers as a JSON-ready dict, for a file kept beside a run.

    The keys are the ones `format_iteration` prints. A value that is missing or not a
    finite number is None, so an iteration with no finished episode stays readable.
    """
    runners = result.get("env_runners", {})
    keys = ("episode_return_mean", *_COUNT_KEYS, "standing_still_fraction")
    record: dict[str, Any] = {"iteration": i}
    for key in keys:
        value = runners.get(key)
        record[key] = float(value) if _is_number(value) else None
    return record


def format_iteration(i: int, n: int, result: Mapping[str, Any]) -> str:
    """One line per training iteration, in plain numbers (no exponents).

    The episode return mean comes first because it is the number PPO
    reports. Most iterations have none, since an episode is longer than an
    iteration, and then the line shows 0. A real return of 0 prints as 0.0,
    so the two can still be told apart.
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
