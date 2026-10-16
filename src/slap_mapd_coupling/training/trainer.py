"""Training entry point wrapping RLlib's PPO algorithm under Ray Tune.

`train` takes a finished `PPOConfig` (training/config.py::build_ppo_config), runs it with
Tune until the training budget is reached and returns where the checkpoints are. It holds no
graph and no study setting. Whoever calls it, an example or the `train` command, builds the
config first.

Checkpoints are Tune's own (`tune.CheckpointConfig`). A run writes one every
`CheckpointSettings.frequency` iterations and one at the end, keeps the `num_to_keep` best by
`score_attribute` (and always the latest), and `TrainingRun.best_checkpoint` is the best one by
that score. The score is the training return by default, which is the number PPO reports. It is
not the evaluation result, so a high score is no promise of completed tasks (see the note on
`deliver_reward` in examples/04_decentralised_training.py).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from slap_mapd_coupling.training.callbacks import (
    TrainingProgressCallbacks,
    format_iteration,
    iteration_record,
)
from slap_mapd_coupling.training.config import DEFAULT_POLICY_ID

BUDGET_METRIC = "num_env_steps_sampled_lifetime"
DEFAULT_SCORE_ATTRIBUTE = "env_runners/episode_return_mean"


class CheckpointSettings(BaseModel):
    """What Tune saves during a run and which checkpoint counts as the best.

    Illustrative values, since the thesis leaves the checkpointing of a run open. A frequency of
    0 saves only the checkpoint at the end of the run. `num_to_keep` of None keeps every
    checkpoint. With a number, Tune deletes the lowest scoring ones, but never the latest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency: NonNegativeInt = 10  # iterations between checkpoints
    num_to_keep: PositiveInt | None = 3
    score_attribute: str = DEFAULT_SCORE_ATTRIBUTE
    score_order: Literal["max", "min"] = "max"


class TrainingRun(NamedTuple):
    """Where a run ended up. `checkpoint` is the one at the end of the run, `best_checkpoint`
    the highest scoring one that Tune kept, and the two are the same folder when the last
    iteration scored best."""

    checkpoint: Path
    best_checkpoint: Path
    env_steps_sampled: int
    trial_dir: Path


class ProgressLog(tune.Callback):
    """Prints one line per iteration and appends it to a JSON Lines file."""

    def __init__(self, expected_iterations: int, progress_path: Path | None) -> None:
        self._expected_iterations = expected_iterations
        self._progress_path = progress_path
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("", encoding="utf-8")

    def on_trial_result(self, iteration, trials, trial, result: dict, **info: Any) -> None:
        i = result["training_iteration"]
        print(format_iteration(i, self._expected_iterations, result))
        if self._progress_path is not None:
            with self._progress_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(iteration_record(i, result)) + "\n")


def checkpoint_config(settings: CheckpointSettings) -> tune.CheckpointConfig:
    """Tune's checkpoint configuration for `settings`. The checkpoint at the end of the run is
    always written, so a run that stops between two periodic checkpoints still has its last."""
    return tune.CheckpointConfig(
        num_to_keep=settings.num_to_keep,
        checkpoint_score_attribute=settings.score_attribute,
        checkpoint_score_order=settings.score_order,
        checkpoint_frequency=settings.frequency,
        checkpoint_at_end=True,
    )


def train(
    ppo_config: PPOConfig,
    budget_env_steps: int,
    run_name: str,
    *,
    arm_label: str = "",
    settings: CheckpointSettings | None = None,
    storage_path: Path | None = None,
    progress_path: Path | None = None,
) -> TrainingRun:
    """Trains with Ray Tune, stops at the first iteration at or above the training budget, and
    returns the checkpoint of the end of the run and the best one.

    `ppo_config` is the PPOConfig object, not `to_dict()`. `storage_path` is where Tune writes
    the run (Ray's default, ~/ray_results, when None). An iteration samples one episode per
    runner, so the run overshoots the budget by up to one iteration's samples.
    """
    settings = settings or CheckpointSettings()
    # An iteration samples one episode per runner, so this is the number of iterations.
    expected_iterations = math.ceil(budget_env_steps / ppo_config.total_train_batch_size)
    print(
        f"=== Training the {arm_label + ' ' if arm_label else ''}GNN+PPO controller "
        f"(budget {budget_env_steps} environment steps) ==="
    )
    results = tune.Tuner(
        "PPO",
        # The PPOConfig object itself, as Tune's docs show, though the stub says dict.
        param_space=ppo_config.callbacks(TrainingProgressCallbacks),  # type: ignore[arg-type]
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=str(storage_path.resolve()) if storage_path is not None else None,
            stop={BUDGET_METRIC: budget_env_steps},
            checkpoint_config=checkpoint_config(settings),
            callbacks=[ProgressLog(expected_iterations, progress_path)],
        ),
    ).fit()
    result = results[0]
    if result.error is not None:
        raise result.error
    assert result.checkpoint is not None and result.metrics is not None
    best = result.get_best_checkpoint(settings.score_attribute, settings.score_order)
    return TrainingRun(
        checkpoint=Path(result.checkpoint.path),
        best_checkpoint=Path((best or result.checkpoint).path),
        env_steps_sampled=int(result.metrics[BUDGET_METRIC]),
        trial_dir=Path(result.path),
    )


def load_module(checkpoint: Path, policy_id: str = DEFAULT_POLICY_ID):
    """The trained RLModule of `policy_id` from an Algorithm checkpoint, with no Ray
    cluster and no registered environment. The path is built from Ray's own constants
    for the layout of a checkpoint, which a test guards."""
    from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_LEARNER_GROUP, COMPONENT_RL_MODULE
    from ray.rllib.core.rl_module.rl_module import RLModule

    path = checkpoint / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    return RLModule.from_checkpoint(str(path.resolve() / policy_id))
