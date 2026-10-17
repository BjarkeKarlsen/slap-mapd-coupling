"""training/trainer.py: Tune's checkpointing, the best checkpoint of a run, and loading it."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.training.callbacks import TrainingProgressCallbacks
from slap_mapd_coupling.training.spec import (
    build_study,
    build_study_ppo_config,
    load_spec,
    override_spec,
)
from slap_mapd_coupling.training.trainer import (
    DEFAULT_SCORE_ATTRIBUTE,
    CheckpointSettings,
    checkpoint_config,
    load_module,
    train,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "decentralised.yaml"


def test_the_checkpoint_settings_become_tunes_checkpoint_config():
    config = checkpoint_config(
        CheckpointSettings(frequency=4, num_to_keep=2, score_attribute="x", score_order="min")
    )
    assert config.checkpoint_frequency == 4
    assert config.num_to_keep == 2
    assert config.checkpoint_score_attribute == "x"
    assert config.checkpoint_score_order == "min"
    # The end of the run is always saved, so a run between two checkpoints keeps its last.
    assert config.checkpoint_at_end is True


def test_the_default_score_is_the_episode_return_and_the_best_is_the_highest():
    settings = CheckpointSettings()
    assert settings.score_attribute == DEFAULT_SCORE_ATTRIBUTE == "episode_return_mean"
    assert settings.score_order == "max"


def test_the_default_keeps_every_checkpoint_one_per_three_iterations():
    # A checkpoint is chosen on validation seeds after the run, so none may be deleted.
    settings = CheckpointSettings()
    assert settings.frequency == 3
    assert settings.num_to_keep is None


def test_an_iteration_with_fewer_episodes_than_runners_stops_the_run():
    callbacks = TrainingProgressCallbacks()
    algorithm = SimpleNamespace(config=SimpleNamespace(num_env_runners=7))
    callbacks.on_train_result(algorithm=algorithm, result={"env_runners": {"num_episodes": 7}})
    with pytest.raises(RuntimeError, match="1 episodes, but there are 7 runners"):
        callbacks.on_train_result(algorithm=algorithm, result={"env_runners": {"num_episodes": 1}})
    # No finished episode at all is the same failure.
    with pytest.raises(RuntimeError, match="0 episodes"):
        callbacks.on_train_result(algorithm=algorithm, result={"env_runners": {}})
    # With no remote runner the local runner samples one episode.
    local = SimpleNamespace(config=SimpleNamespace(num_env_runners=0))
    callbacks.on_train_result(algorithm=local, result={"env_runners": {"num_episodes": 1}})


def test_a_nested_score_is_rejected_because_tune_would_keep_the_newest_instead():
    with pytest.raises(ValidationError, match="top-level"):
        CheckpointSettings(score_attribute="env_runners/episode_return_mean")


def test_the_return_is_copied_to_the_top_level_of_the_result():
    callbacks = TrainingProgressCallbacks()
    result = {"env_runners": {"episode_return_mean": 12.5}}
    callbacks.on_train_result(result=result)
    assert result["episode_return_mean"] == 12.5
    # An iteration with no finished episode has no return, and nothing is invented.
    empty = {"env_runners": {"episode_return_mean": float("nan")}}
    callbacks.on_train_result(result=empty)
    assert "episode_return_mean" not in empty


def test_a_misspelt_checkpoint_setting_is_rejected():
    with pytest.raises(ValidationError):
        CheckpointSettings(every=3)  # type: ignore[call-arg]


def test_a_run_saves_periodic_checkpoints_and_returns_its_best(tmp_path):
    # Four iterations of 130 steps, a checkpoint at every one, the best two kept.
    spec = override_spec(
        load_spec(CONFIG), horizon=130, budget_env_steps=520, checkpoint_every=1, keep_checkpoints=2
    )
    progress = tmp_path / "progress.jsonl"
    run = train(
        build_study_ppo_config(spec, build_study(spec)),
        spec.ppo.training_budget_env_steps,
        "tiny",
        settings=spec.checkpoint,
        storage_path=tmp_path / "ray",
        progress_path=progress,
    )

    returns = [
        json.loads(line)["episode_return_mean"] for line in progress.read_text().splitlines()
    ]
    assert len(returns) == 4
    assert run.env_steps_sampled == 520
    # The checkpoint of iteration i is checkpoint_<i - 1>. The best is the highest return, and
    # it is kept along with the latest, so no more than two folders remain.
    best_iteration = max(range(4), key=returns.__getitem__) + 1
    assert run.best_checkpoint.name == f"checkpoint_{best_iteration - 1:06d}"
    kept = sorted(path.name for path in run.trial_dir.glob("checkpoint_*"))
    assert len(kept) <= 3 and run.best_checkpoint.name in kept and run.checkpoint.name in kept
    # Both open as modules without a Ray cluster.
    for checkpoint in (run.checkpoint, run.best_checkpoint):
        assert load_module(checkpoint).model_config["gnn_encoder_config"]["communication"] is False


def test_a_centralised_run_counts_timesteps_and_stops_at_the_budget(tmp_path):
    # Three robots choose in turn, so one episode is three sub-steps per timestep. The budget
    # counts timesteps, so two iterations of one episode each reach exactly 2 * horizon.
    horizon = 130
    spec = override_spec(
        load_spec(CONFIG.with_name("centralised.yaml")),
        horizon=horizon,
        budget_env_steps=2 * horizon,
        checkpoint_every=0,
    )
    progress = tmp_path / "progress.jsonl"
    run = train(
        build_study_ppo_config(spec, build_study(spec)),
        spec.ppo.training_budget_env_steps,
        "tiny_centralised",
        storage_path=tmp_path / "ray",
        progress_path=progress,
    )

    assert len(progress.read_text().splitlines()) == 2
    assert run.env_steps_sampled == 2 * horizon


def test_the_timestep_count_is_copied_to_the_top_level_of_the_result():
    callbacks = TrainingProgressCallbacks()
    result = {"env_runners": {"timesteps_sampled_lifetime": 260.0}}
    callbacks.on_train_result(result=result)
    assert result["timesteps_sampled_lifetime"] == 260
