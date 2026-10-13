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
