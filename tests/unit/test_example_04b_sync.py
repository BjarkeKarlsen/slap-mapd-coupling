"""04b_evaluate_checkpoint.py loads a checkpoint trained by
04_decentralised_training.py, so its instance, storage and configuration
must be the same as 04's. A checkpoint built against another instance
loads without an error and produces nonsense actions."""

import importlib.util
from pathlib import Path
from types import ModuleType

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
