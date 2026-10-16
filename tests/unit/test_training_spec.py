"""The YAML training spec builds what example 04 builds, so `slap-mapd train` and the
example train the same run."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slap_mapd_coupling.training.spec import (
    build_experiment_config,
    build_study,
    load_spec,
    override_spec,
    with_budget,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "decentralised.yaml"


@pytest.fixture(scope="module")
def example_04():
    spec = importlib.util.spec_from_file_location(
        "_example_04_spec", ROOT / "examples" / "04_decentralised_training.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec():
    return load_spec(CONFIG)


def test_the_shipped_file_builds_a_study(spec):
    study = build_study(spec)
    assert study.config.controller == "decentralised"
    assert study.arm.pooling == "none"


def test_the_example_reads_its_hyperparameters_from_the_file(example_04, spec):
    assert example_04.build_hyperparameters(900) == with_budget(spec, 900).ppo
    assert example_04.ILLUSTRATIVE_BUDGET_ENV_STEPS == spec.ppo.training_budget_env_steps


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_the_file_and_the_example_build_the_same_study(example_04, spec, controller):
    # The example keeps its own study constants (the campaign script pins some of them), so
    # this is what stops the two from drifting apart.
    study = build_study(override_spec(spec, controller=controller))
    instance = example_04.build_generated_instance()
    skus, capacities, counts = example_04.build_storage(instance.graph)
    config = example_04.build_config("fixed_only", seed=0, controller=controller)

    assert study.instance == instance
    assert (study.skus, study.capacities, study.counts) == (skus, capacities, counts)
    assert study.config == config
    assert study.arm == example_04.build_arm(instance, controller, config.observation_depth)
    assert study.gnn_encoder == example_04.build_gnn_config(config, study.arm)
    assert study.seed_split == example_04.build_seed_split(0, spec.eval_seed_count)


def test_the_regime_decides_the_storage_mode_of_training(spec):
    matched = override_spec(spec, regime="matched")
    assert build_experiment_config(matched).storage_mode == spec.evaluation_storage_mode
    assert build_experiment_config(spec).storage_mode == "fixed"


def test_an_override_changes_only_what_it_names(spec):
    changed = override_spec(
        spec, horizon=200, budget_env_steps=900, seed=7, checkpoint_every=2, keep_checkpoints=5
    )
    assert changed.horizon == 200
    assert changed.ppo.training_budget_env_steps == 900
    assert changed.experiment.seed == 7
    assert (changed.checkpoint.frequency, changed.checkpoint.num_to_keep) == (2, 5)
    assert override_spec(spec) == spec


def test_a_combination_the_arm_rejects_is_rejected_on_override(spec):
    # Communication belongs to the decentralised arm only.
    with pytest.raises(ValueError):
        build_study(override_spec(spec, controller="centralised", communication=True))


def test_a_misspelt_key_in_the_file_is_rejected(tmp_path, spec):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["ppo"]["learning_rate"] = data["ppo"].pop("lr")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_spec(path)


def test_a_file_that_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_spec(path)
