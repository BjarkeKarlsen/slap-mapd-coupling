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
from dataclasses import dataclass, field
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

    `tasks_completed`, `tasks_released`, `backlog` and `steps_since_last_completion` are
    snapshots of the episode in progress, taken at the end of a sampled fragment and averaged
    over the environments. They are not per-episode totals, since a fragment ends wherever the
    sampling stops, usually part way through an episode. The per-episode totals are the
    `episode_*` keys logged at the end of every episode (`episode_metrics`). Rates over the last
    fragment, which do not depend on where the episode stands, are `window_metrics`.

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


@dataclass
class WindowSnapshot:
    """The cumulative counts of one environment at the end of a fragment, kept so the next
    fragment can be measured against them (`window_metrics`)."""

    t: int = 0
    shaping: float = 0.0
    delivery: float = 0.0
    override: float = 0.0
    congestion: float = 0.0
    completed: int = 0
    overrides_by_agent: dict[Any, int] = field(default_factory=dict)


def window_snapshot(env: WarehouseMAPDEnv) -> WindowSnapshot:
    """The cumulative reward terms, overrides and completed tasks of the episode in progress."""
    log = env.log
    return WindowSnapshot(
        t=env.t,
        shaping=sum(log.return_shaping_by_agent.values()),
        delivery=sum(log.return_delivery_by_agent.values()),
        override=sum(log.return_override_by_agent.values()),
        congestion=sum(log.return_congestion_by_agent.values()),
        completed=sum(task.finish_time is not None for task in env.tasks),
        overrides_by_agent=dict(log.overrides_by_agent),
    )


def window_metrics(
    env: WarehouseMAPDEnv, previous: Optional[WindowSnapshot]
) -> dict[str, float]:
    """What happened in the last fragment, the steps since `previous` was taken, as rates.

    Each `window_*_per_step` value is one term of the reward (environment/reward_function.py::
    RewardTerms, each signed) summed over the robots and the steps of the window, divided by
    the number of steps in the window. The four add up to the return per step over the window.
    `window_override_rate` is the share of robot-steps in the window in which a robot was
    overridden by conflict resolution. `window_max_agent_override_share` is the share of the
    window's overrides that fell on the single most overridden robot, and is 0 when there were
    none. `window_tasks_completed_per_step` is the tasks completed in the window divided by
    its steps.

    The window is the current counts minus `previous`. When the episode clock is not past
    `previous` the environment was reset in between, so the window is the new episode so far.
    Nothing is returned when no step has been taken yet, since a rate over zero steps is
    undefined. A planner run has no reward, so its reward and override values are 0.
    """
    steps = env.t
    if steps <= 0:
        return {}
    now = window_snapshot(env)
    if previous is None or steps <= previous.t:
        previous = WindowSnapshot()
    window_steps = steps - previous.t
    overrides_by_agent = {
        agent: count - previous.overrides_by_agent.get(agent, 0)
        for agent, count in now.overrides_by_agent.items()
    }
    overrides = sum(overrides_by_agent.values())
    agent_steps = window_steps * env.config.num_agents
    return {
        "window_shaping_per_step": (now.shaping - previous.shaping) / window_steps,
        "window_deliver_per_step": (now.delivery - previous.delivery) / window_steps,
        "window_override_per_step": (now.override - previous.override) / window_steps,
        "window_congestion_per_step": (now.congestion - previous.congestion) / window_steps,
        "window_override_rate": overrides / agent_steps,
        "window_max_agent_override_share": (
            max(overrides_by_agent.values()) / overrides if overrides else 0.0
        ),
        "window_tasks_completed_per_step": (now.completed - previous.completed) / window_steps,
    }


def episode_metrics(env: WarehouseMAPDEnv) -> dict[str, float]:
    """What one finished episode came to, for the training log. `env` is the simulator
    core, the adapter's `core`, read when the episode ends.

    The four `return_*` values are the terms of the reward summed over the robots and the
    steps of the episode (environment/reward_function.py::RewardTerms, each signed). They add
    up to the episode return RLlib reports, also under a team-mean credit signal, which
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


class TrainingProgressCallbacks(CorrectnessInvariantCallbacks):
    """The invariant checks plus progress numbers in every iteration.

    An iteration samples fewer steps than an episode lasts, so the episode
    return is missing most iterations. These values are logged from the
    environment's own state at the end of every fragment instead, under
    `result["env_runners"]`.

    `tasks_completed`, `tasks_released`, `backlog`, `steps_since_last_completion` and
    `episode_step` are end-of-fragment snapshots of the episode in progress, averaged over the
    environments. They are not per-episode totals.

    The `window_*` values (`window_shaping_per_step`, `window_deliver_per_step`,
    `window_override_per_step`, `window_congestion_per_step`, `window_override_rate`,
    `window_max_agent_override_share`, `window_tasks_completed_per_step`) are rates over the
    last fragment, the steps since the previous one of the same environment, averaged over the
    environments. They do not depend on where the episode stands, so iterations can be compared
    (see `window_metrics`).

    The exact per-episode totals are the `episode_*` keys, together with `return_shaping`,
    `return_deliver`, `return_override`, `return_congestion` and `override_rate`, which are
    logged when an episode ends and so are missing from an iteration in which none ended (see
    `episode_metrics`).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._standing_still = StandingStillCounter()
        self._previous_window: dict[int, WindowSnapshot] = {}

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
            core = _unwrap(env_runner.env, env_index).core
            values = {
                **progress_metrics(core),
                **window_metrics(core, self._previous_window.get(env_index)),
            }
            self._previous_window[env_index] = window_snapshot(core)
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
        for name, value in episode_metrics(_unwrap(env, env_index).core).items():
            metrics_logger.log_value(name, value, reduce="mean")


_COUNT_KEYS = (
    "tasks_completed",
    "tasks_released",
    "backlog",
    "steps_since_last_completion",
)


# The episode step and the rates over the last fragment, logged every iteration (see
# `window_metrics`). Kept apart from _COUNT_KEYS, which `format_iteration` prints.
_WINDOW_KEYS = (
    "episode_step",
    "window_shaping_per_step",
    "window_deliver_per_step",
    "window_override_per_step",
    "window_congestion_per_step",
    "window_override_rate",
    "window_max_agent_override_share",
    "window_tasks_completed_per_step",
)


# Logged once per finished episode, so None in an iteration where no episode ended.
_EPISODE_KEYS = (
    "return_shaping",
    "return_deliver",
    "return_override",
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

    The keys are the ones `format_iteration` prints, then `episode_step` and the `window_*`
    rates over the last fragment of `_WINDOW_KEYS`, then the per-episode keys of
    `_EPISODE_KEYS`. The count keys are snapshots of the episode in progress and the window
    keys are rates over the last fragment, both averaged over the environments, not per-episode
    totals (see `progress_metrics`, `window_metrics`). The exact totals are the
    `_EPISODE_KEYS`, present only in an iteration in which an episode ended. A value that is
    missing or not a finite number is None, so an iteration with no finished episode stays
    readable.
    """
    runners = result.get("env_runners", {})
    keys = (
        "episode_return_mean",
        *_COUNT_KEYS,
        "standing_still_fraction",
        *_WINDOW_KEYS,
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
