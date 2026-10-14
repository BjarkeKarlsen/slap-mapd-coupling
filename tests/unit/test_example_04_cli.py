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


def test_hyperparameters_have_no_batch_size_and_no_rollout_length(example_04):
    fields = example_04.build_hyperparameters(900).model_dump()
    assert "train_batch_size" not in fields
    assert "rollout_fragment_length" not in fields


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


def test_arrival_rate_defaults_to_a_quarter_and_must_be_positive(example_04):
    assert example_04.parse_args([]).arrival_rate == 0.25
    assert example_04.parse_args(["--arrival-rate=0.15"]).arrival_rate == 0.15
    assert example_04.parse_args(["--arrival-rate", "0.15"]).arrival_rate == 0.15
    with pytest.raises(SystemExit):
        example_04.parse_args(["--arrival-rate=0"])


def test_training_and_evaluation_configs_carry_the_arrival_rate(example_04):
    assert example_04.build_config("fixed_only", seed=0).arrival_rate == 0.25
    assert example_04.build_config("matched", seed=0, arrival_rate=0.15).arrival_rate == 0.15
    assert example_04.evaluation_config(3, "section").arrival_rate == 0.25
    assert example_04.evaluation_config(3, "section", 0.15).arrival_rate == 0.15


def test_the_wait_cost_is_one_value_for_the_graph_and_the_config(example_04):
    instance = example_04.build_generated_instance()
    config = example_04.build_config("fixed_only", seed=0)
    assert config.wait_cost == example_04.WAIT_COST
    assert instance.graph.wait_cost == example_04.WAIT_COST


def test_horizon_defaults_to_630_and_must_be_positive(example_04):
    assert example_04.DEFAULT_HORIZON == 630
    assert example_04.parse_args([]).horizon == 630
    assert example_04.parse_args(["--horizon=2000"]).horizon == 2000
    assert example_04.parse_args(["--horizon", "2000"]).horizon == 2000
    for bad in ("0", "-5"):
        with pytest.raises(SystemExit):
            example_04.parse_args([f"--horizon={bad}"])


def test_training_and_evaluation_configs_share_the_horizon(example_04):
    assert example_04.build_config("fixed_only", seed=0).horizon == 630
    assert example_04.build_config("matched", seed=0, horizon=2000).horizon == 2000
    assert example_04.evaluation_config(3, "section").horizon == 630
    assert example_04.evaluation_config(3, "section", horizon=2000).horizon == 2000
    # one episode length for both, so the same seed gives the same config
    assert example_04.evaluation_config(
        0, "decentralised", horizon=2000
    ) == example_04.build_config("matched", seed=0, horizon=2000)


def test_communication_is_off_by_default_and_read_as_a_flag(example_04):
    assert example_04.parse_args([]).communication is False
    assert example_04.parse_args(["--communication"]).communication is True


def test_training_and_evaluation_configs_carry_communication(example_04):
    assert example_04.build_config("fixed_only", seed=0).communication is False
    assert example_04.build_config("matched", seed=0, communication=True).communication is True
    assert example_04.evaluation_config(3, "decentralised").communication is False
    assert example_04.evaluation_config(3, "decentralised", communication=True).communication
    # one config for both, so the same seed gives the same config
    assert example_04.evaluation_config(
        0, "decentralised", communication=True
    ) == example_04.build_config("matched", seed=0, communication=True)


def test_communication_only_changes_the_communication_field(example_04):
    off = example_04.build_config("fixed_only", seed=0)
    on = example_04.build_config("fixed_only", seed=0, communication=True)
    assert on == off.model_copy(update={"communication": True})


def test_the_gnn_encoder_follows_the_communication_of_the_config(example_04):
    instance = example_04.build_generated_instance()
    for communication in (False, True):
        config = example_04.build_config("fixed_only", seed=0, communication=communication)
        arm = example_04.build_arm(instance, "decentralised", config.observation_depth)
        assert example_04.build_gnn_config(config, arm).communication is communication


def test_the_message_cap_is_the_other_agents_of_the_fleet(example_04):
    # an agent hears at most the other FLEET_SIZE - 1 agents, so the cap is exactly enough
    assert example_04.MAX_MESSAGES == example_04.FLEET_SIZE - 1
    instance = example_04.build_generated_instance()
    for controller in ("decentralised", "centralised", "section"):
        arm = example_04.build_arm(instance, controller, 5)
        assert arm.encoding.max_messages == example_04.FLEET_SIZE - 1


@pytest.mark.parametrize("controller", ["centralised", "section"])
def test_communication_is_rejected_for_the_other_arms(example_04, controller, capsys):
    with pytest.raises(SystemExit):
        example_04.parse_args(["--communication"], controller=controller)
    assert "only applies to the decentralised controller" in capsys.readouterr().err
    assert example_04.parse_args([], controller=controller).communication is False


@pytest.mark.parametrize("controller", ["centralised", "section"])
def test_the_entry_points_of_the_other_arms_reject_communication(
    example_04, controller, monkeypatch, capsys
):
    # main parses the arguments first, so nothing is built or trained before the error
    monkeypatch.setattr("sys.argv", ["prog", "--communication"])
    with pytest.raises(SystemExit):
        example_04.main(controller)
    assert "only applies to the decentralised controller" in capsys.readouterr().err


def test_the_training_and_evaluation_configs_carry_the_keep_up_threshold(example_04):
    assert example_04.KEEP_UP_THRESHOLD == 0.8
    assert example_04.build_config("fixed_only", seed=0).keep_up_threshold == 0.8
    assert example_04.evaluation_config(3, "section").keep_up_threshold == 0.8
