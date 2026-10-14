"""04b_evaluate_checkpoint.py loads a checkpoint trained by
04_decentralised_training.py, so its instance, storage and configuration
must be the same as 04's. A checkpoint built against another instance
loads without an error and produces nonsense actions."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_example_{filename[:3]}", EXAMPLES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_04() -> ModuleType:
    return _load("04_decentralised_training.py")


@pytest.fixture(scope="module")
def example_04b() -> ModuleType:
    return _load("04b_evaluate_checkpoint.py")


def test_04b_builds_the_same_instance_as_04(example_04, example_04b):
    assert example_04b.FLEET_SIZE == example_04.FLEET_SIZE
    assert example_04b.build_instance() == example_04.build_instance()


def test_04b_builds_the_same_storage_as_04(example_04, example_04b):
    graph = example_04.build_instance()
    assert example_04b.build_storage(graph) == example_04.build_storage(graph)


def test_04b_builds_the_same_config_as_04(example_04, example_04b):
    assert example_04b.build_config(seed=0) == example_04.build_config("fixed_only", seed=0)


def test_04b_takes_the_arrival_rate_the_checkpoint_was_trained_at(example_04, example_04b):
    assert example_04b.parse_args(["--checkpoint=x"]).arrival_rate == 1.0
    assert example_04b.parse_args(["--checkpoint=x", "--arrival-rate=0.15"]).arrival_rate == 0.15
    assert example_04b.build_config(seed=0, arrival_rate=0.15) == example_04.build_config(
        "matched", seed=0, arrival_rate=0.15
    )


def test_04b_takes_the_same_horizon_as_04(example_04, example_04b):
    assert example_04b.parse_args(["--checkpoint=x"]).horizon == 2000
    assert example_04b.parse_args(["--checkpoint=x", "--horizon=630"]).horizon == 630
    for bad in ("0", "-5"):
        with pytest.raises(SystemExit):
            example_04b.parse_args(["--checkpoint=x", f"--horizon={bad}"])
    assert example_04b.build_config(seed=0).horizon == example_04.DEFAULT_HORIZON
    assert example_04b.build_config(seed=0, horizon=630) == example_04.build_config(
        "matched", seed=0, horizon=630
    )


def test_04b_takes_communication_and_builds_the_same_config_as_04(example_04, example_04b):
    assert example_04b.parse_args(["--checkpoint=x"]).communication is False
    assert example_04b.parse_args(["--checkpoint=x", "--communication"]).communication is True
    assert example_04b.build_config(seed=0).communication is False
    assert example_04b.build_config(seed=0, communication=True) == example_04.evaluation_config(
        0, "decentralised", communication=True
    )
    assert example_04b.build_config(seed=0, communication=True) == example_04.build_config(
        "matched", seed=0, communication=True
    )


@pytest.mark.parametrize("arm", ["centralised", "section"])
def test_04b_rejects_communication_for_the_other_arms(example_04b, arm, capsys):
    with pytest.raises(SystemExit):
        example_04b.parse_args(["--checkpoint=x", f"--arm={arm}", "--communication"])
    assert "only applies to the decentralised controller" in capsys.readouterr().err
    example_04b.parse_args(["--checkpoint=x", "--arm=decentralised", "--communication"])


def _module_trained_with(communication: bool):
    config = {"gnn_encoder_config": {"communication": communication}}
    return SimpleNamespace(model_config=config)


@pytest.mark.parametrize("trained_with", [False, True])
def test_04b_accepts_the_flag_the_checkpoint_was_trained_with(example_04b, trained_with):
    example_04b.check_communication(_module_trained_with(trained_with), trained_with)


@pytest.mark.parametrize("trained_with", [False, True])
def test_04b_rejects_a_flag_that_differs_from_the_checkpoint(example_04b, trained_with):
    with pytest.raises(RuntimeError, match=f"trained with communication={trained_with}"):
        example_04b.check_communication(_module_trained_with(trained_with), not trained_with)
