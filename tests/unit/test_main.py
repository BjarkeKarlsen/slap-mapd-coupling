"""Unit tests for slap_mapd_coupling.main (the CLI).

`simulate` and `sweep` only cover argument validation -- their bodies are
`raise NotImplementedError(...)`, so tests assert that placeholder is what's
reached once validation passes. `train` is real. Its tests replace the
trainer with a fake, so they check what the command relays and not a run
(tests/unit/test_trainer.py runs one).
"""

import json
from pathlib import Path

from click.testing import CliRunner

from slap_mapd_coupling.main import cli
from slap_mapd_coupling.training.trainer import TrainingRun


def test_cli_help_exits_zero():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_simulate_requires_checkpoint_for_decentralised():
    result = CliRunner().invoke(cli, ["simulate", "--controller", "decentralised"])
    assert result.exit_code != 0
    assert "--checkpoint is required" in result.output


def test_simulate_rejects_checkpoint_for_non_decentralised(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("x")
    result = CliRunner().invoke(
        cli, ["simulate", "--planner", "prioritised_astar", "--checkpoint", str(checkpoint)]
    )
    assert result.exit_code != 0
    assert "--checkpoint only applies" in result.output


def test_simulate_rejects_communication_for_non_decentralised():
    result = CliRunner().invoke(
        cli, ["simulate", "--planner", "prioritised_astar", "--communication"]
    )
    assert result.exit_code != 0
    assert "--communication only applies" in result.output


def test_simulate_reaches_not_implemented_once_validation_passes():
    result = CliRunner().invoke(cli, ["simulate", "--planner", "prioritised_astar"])
    assert isinstance(result.exception, NotImplementedError)


def test_simulate_requires_exactly_one_of_controller_and_planner():
    neither = CliRunner().invoke(cli, ["simulate"])
    both = CliRunner().invoke(
        cli, ["simulate", "--controller", "section", "--planner", "prioritised_astar"]
    )
    assert neither.exit_code != 0 and "exactly one" in neither.output
    assert both.exit_code != 0 and "exactly one" in both.output


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "decentralised.yaml"


def test_train_requires_a_config():
    result = CliRunner().invoke(cli, ["train"])
    assert result.exit_code != 0
    assert "--config" in result.output


def test_resume_needs_the_spec_train_wrote(tmp_path):
    result = CliRunner().invoke(cli, ["resume", str(tmp_path)])
    assert result.exit_code != 0
    assert "Expected one *_spec.json" in result.output


def test_train_rejects_communication_for_a_non_decentralised_controller(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--config",
            str(CONFIG),
            "--controller",
            "centralised",
            "--communication",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "communication" in result.output


def test_train_relays_the_config_and_the_options_to_the_trainer(tmp_path, monkeypatch):
    calls = {}

    def fake_train(ppo_config, budget, name, **kwargs):
        calls.update(ppo_config=ppo_config, budget=budget, name=name, **kwargs)
        return TrainingRun(
            checkpoint=tmp_path / "last",
            best_checkpoint=tmp_path / "best",
            env_steps_sampled=budget,
            trial_dir=tmp_path,
        )

    monkeypatch.setattr("slap_mapd_coupling.training.trainer.train", fake_train)
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--config",
            str(CONFIG),
            "--controller",
            "section",
            "--budget-env-steps",
            "900",
            "--checkpoint-every",
            "2",
            "--keep-checkpoints",
            "4",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls["budget"] == 900
    assert calls["name"] == "section_fixed_only"
    assert calls["arm_label"] == "section"
    assert (calls["settings"].frequency, calls["settings"].num_to_keep) == (2, 4)
    assert calls["storage_path"] == tmp_path
    # The effective settings are kept beside the run, and the paths are printed for the campaign.
    spec = json.loads((tmp_path / "section_fixed_only_spec.json").read_text())
    assert spec["controller"] == "section" and spec["ppo"]["training_budget_env_steps"] == 900
    assert f"Checkpoint saved to {tmp_path / 'last'}" in result.output
    assert f"saved to {tmp_path / 'best'}" in result.output


def test_sweep_requires_config():
    result = CliRunner().invoke(cli, ["sweep"])
    assert result.exit_code != 0


def test_sweep_reaches_not_implemented_once_validation_passes(tmp_path):
    config = tmp_path / "grid.yaml"
    config.write_text("{}")
    result = CliRunner().invoke(cli, ["sweep", "--config", str(config)])
    assert isinstance(result.exception, NotImplementedError)
