"""05_full_comparison.py sweeps controllers and load levels into one row per run."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def example_05() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_example_05", EXAMPLES / "05_full_comparison.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_checkpoint_is_named_by_its_arm_and_the_communication_flag(example_05, tmp_path):
    assert example_05.parse_checkpoint(f"section={tmp_path}") == (
        "section",
        "section",
        False,
        tmp_path,
    )
    label, arm, communication, _ = example_05.parse_checkpoint(f"decentralised+comm={tmp_path}")
    assert (label, arm, communication) == ("decentralised+comm", "decentralised", True)
    for bad in ("section", "wide=x", "section+comm=x", "decentralised+msg=x", "section="):
        with pytest.raises(argparse.ArgumentTypeError):
            example_05.parse_checkpoint(bad)


def test_the_planner_config_is_04s_evaluation_config_with_a_planner(example_05):
    config = example_05.planner_config(5, "token_passing", 0.3, 40)

    assert config.planner == "token_passing" and config.controller is None
    assert (config.seed, config.arrival_rate, config.horizon) == (5, 0.3, 40)
    reference = example_05.example_04.evaluation_config(5, "decentralised", 0.3, 40)
    assert config.wait_cost == reference.wait_cost
    assert config.observation_depth == reference.observation_depth


def test_a_planner_sweep_writes_one_row_per_rate_and_seed(example_05, tmp_path):
    out = tmp_path / "load_sweep.jsonl"

    rows = example_05.sweep([], ["token_passing"], [0.2, 0.4], (5, 6), 40, out)

    assert len(rows) == 4
    on_disk = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert on_disk == rows
    for row in rows:
        assert row["condition"] == "token_passing" and row["kind"] == "planner"
        assert row["released"] == row["num_completed_tasks"] + row["backlog"]
    assert {(r["arrival_rate"], r["seed"]) for r in rows} == {
        (0.2, 5),
        (0.2, 6),
        (0.4, 5),
        (0.4, 6),
    }


def test_the_sweep_rejects_a_missing_checkpoint_and_a_repeated_label(
    example_05, tmp_path, monkeypatch
):
    out = ["--out", str(tmp_path)]
    for argv in (
        out + ["--checkpoint", f"section={tmp_path / 'missing'}"],
        out + ["--planner", "token_passing", "--planner", "token_passing"],
        out + ["--arrival-rates", "0.2", "0"],
        out + ["--eval-seeds", "0"],
    ):
        with pytest.raises(SystemExit):
            example_05.parse_args(argv)
    assert example_05.parse_args(out).planner == ["token_passing"]
