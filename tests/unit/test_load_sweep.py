"""scripts/load_sweep.py, the Token Passing load sweep on a training spec."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "load_sweep.py"
CONFIG = ROOT / "configs" / "medium_putaway" / "decentralised.yaml"


def _script():
    spec = importlib.util.spec_from_file_location("load_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_sweep_writes_one_row_per_episode_and_a_second_start_writes_nothing_new(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.1", "0.2", "--seeds", "1",
        "--storage-modes", "demand", "--horizon", "120", "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    rows = (tmp_path / "load_sweep.jsonl").read_text().splitlines()
    assert len(rows) == 2
    assert [json.loads(row)["arrival_rate"] for row in rows] == [0.1, 0.2]
    summary = (tmp_path / "summary.md").read_text()
    assert "Trip distance" in summary and "| demand | 0.1 | 1 |" in summary
    assert "## Capacity" in summary

    assert sweep.main(argv) == 0
    assert (tmp_path / "load_sweep.jsonl").read_text().splitlines() == rows


def _row(rate, seed, completed, backlog, mode="fixed"):
    return dict(
        storage_mode=mode,
        arrival_rate=rate,
        seed=seed,
        num_completed_tasks=completed,
        backlog=backlog,
    )


def test_the_capacity_is_the_highest_rate_that_keeps_up_on_every_seed():
    sweep = _script()
    rows = [
        _row(0.1, 0, 90, 10), _row(0.1, 1, 85, 15),
        _row(0.2, 0, 80, 20), _row(0.2, 1, 81, 19),
        _row(0.3, 0, 90, 10), _row(0.3, 1, 79, 21),  # one seed falls behind
        _row(0.4, 0, 90, 10), _row(0.4, 1, 90, 10),  # passes, but above a failing rate
    ]  # fmt: skip
    found = sweep.capacity(rows, 0.8)
    assert found == {"capacity": 0.2, "first_failing_rate": 0.3, "main_load": 0.12}


def test_no_capacity_when_the_lowest_rate_fails_and_no_failing_rate_when_all_pass():
    sweep = _script()
    assert sweep.capacity([_row(0.1, 0, 10, 90)], 0.8)["capacity"] is None
    assert sweep.capacity([_row(0.1, 0, 90, 10)], 0.8)["first_failing_rate"] is None


def test_rates_per_agent_are_scaled_by_the_fleet_size_and_only_fixed_runs(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates-per-agent", "0.05", "--seeds", "1",
        "--horizon", "60", "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    (row,) = [json.loads(line) for line in (tmp_path / "load_sweep.jsonl").read_text().splitlines()]
    num_agents = sweep.load_spec(CONFIG).experiment.num_agents
    assert row["arrival_rate"] == round(0.05 * num_agents, 6)
    assert row["storage_mode"] == "fixed"
    capacity = json.loads((tmp_path / "capacity.json").read_text())
    assert set(capacity["by_mode"]) == {"fixed"}
