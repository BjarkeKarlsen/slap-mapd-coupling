"""Example 04 trains on the training seeds and is evaluated on the disjoint evaluation seeds."""

import importlib.util
from pathlib import Path

import pytest

from slap_mapd_coupling.training.config import split_seeds

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "04_decentralised_training.py"


@pytest.fixture(scope="module")
def example_04():
    spec = importlib.util.spec_from_file_location("_example_04_cli", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regime_is_read_with_a_space_or_an_equals_sign(example_04):
    assert example_04.parse_args(["--regime", "matched"]).regime == "matched"
    assert example_04.parse_args(["--regime=matched"]).regime == "matched"
    assert example_04.parse_args([]).regime == "fixed_only"


def test_unknown_regime_is_rejected(example_04):
    with pytest.raises(SystemExit):
        example_04.parse_args(["--regime", "other"])


def test_eval_seed_count_defaults_to_three_and_must_be_positive(example_04):
    assert example_04.parse_args([]).eval_seeds == 3
    assert example_04.parse_args(["--eval-seeds=8"]).eval_seeds == 8
    with pytest.raises(SystemExit):
        example_04.parse_args(["--eval-seeds=0"])


def test_evaluation_seeds_are_the_eval_half_of_the_split(example_04):
    train_seeds, eval_seeds = split_seeds(example_04.build_seed_split(0, 8))
    assert len(eval_seeds) == 8
    assert not set(train_seeds) & set(eval_seeds)


def test_budget_is_read_and_must_be_positive(example_04):
    assert example_04.parse_args(["--budget-env-steps", "600"]).budget_env_steps == 600
    assert example_04.parse_args(["--budget-env-steps=600"]).budget_env_steps == 600
    assert example_04.parse_args([]).budget_env_steps == example_04.ILLUSTRATIVE_BUDGET_ENV_STEPS
    with pytest.raises(SystemExit):
        example_04.parse_args(["--budget-env-steps=0"])


def test_hyperparameters_carry_the_budget_given(example_04):
    assert example_04.build_hyperparameters(900).training_budget_env_steps == 900


def test_hyperparameters_have_no_batch_size_and_keep_the_rollout_length(example_04):
    hyperparameters = example_04.build_hyperparameters(900)
    assert "train_batch_size" not in hyperparameters.model_dump()
    assert hyperparameters.rollout_fragment_length == 30


def test_matched_regime_trains_under_the_evaluation_storage_rule(example_04, monkeypatch):
    # With the evaluation rule changed to one this demo cannot configure, "matched" must
    # follow it and "fixed_only" must not.
    assert example_04.build_config("matched", seed=0).storage_mode == "fixed"
    monkeypatch.setattr(example_04, "EVALUATION_STORAGE_MODE", "demand")
    assert example_04.build_config("fixed_only", seed=0).storage_mode == "fixed"
    with pytest.raises(ValueError, match="demand"):
        example_04.build_config("matched", seed=0)


def test_evaluation_config_uses_the_evaluation_storage_rule_and_seed(example_04):
    config = example_04.evaluation_config(17, "centralised")
    assert (config.seed, config.controller) == (17, "centralised")
    assert config.storage_mode == example_04.EVALUATION_STORAGE_MODE


def test_arrival_rate_defaults_to_one_and_must_be_positive(example_04):
    assert example_04.parse_args([]).arrival_rate == 1.0
    assert example_04.parse_args(["--arrival-rate=0.15"]).arrival_rate == 0.15
    assert example_04.parse_args(["--arrival-rate", "0.15"]).arrival_rate == 0.15
    with pytest.raises(SystemExit):
        example_04.parse_args(["--arrival-rate=0"])


def test_training_and_evaluation_configs_carry_the_arrival_rate(example_04):
    assert example_04.build_config("fixed_only", seed=0).arrival_rate == 1.0
    assert example_04.build_config("matched", seed=0, arrival_rate=0.15).arrival_rate == 0.15
    assert example_04.evaluation_config(3, "section").arrival_rate == 1.0
    assert example_04.evaluation_config(3, "section", 0.15).arrival_rate == 0.15


def test_the_wait_cost_is_one_value_for_the_graph_and_the_config(example_04):
    instance = example_04.build_generated_instance()
    config = example_04.build_config("fixed_only", seed=0)
    assert config.wait_cost == example_04.WAIT_COST
    assert instance.graph.wait_cost == example_04.WAIT_COST


def test_horizon_defaults_to_2000_and_must_be_positive(example_04):
    assert example_04.DEFAULT_HORIZON == 2000
    assert example_04.parse_args([]).horizon == 2000
    assert example_04.parse_args(["--horizon=630"]).horizon == 630
    assert example_04.parse_args(["--horizon", "630"]).horizon == 630
    for bad in ("0", "-5"):
        with pytest.raises(SystemExit):
            example_04.parse_args([f"--horizon={bad}"])


def test_training_and_evaluation_configs_share_the_horizon(example_04):
    assert example_04.build_config("fixed_only", seed=0).horizon == 2000
    assert example_04.build_config("matched", seed=0, horizon=630).horizon == 630
    assert example_04.evaluation_config(3, "section").horizon == 2000
    assert example_04.evaluation_config(3, "section", horizon=630).horizon == 630
    # one episode length for both, so the same seed gives the same config
    assert example_04.evaluation_config(0, "decentralised", horizon=630) == example_04.build_config(
        "matched", seed=0, horizon=630
    )

