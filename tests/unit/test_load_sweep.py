"""scripts/sweep/load_sweep.py, the Token Passing load sweep on a training spec."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sweep" / "load_sweep.py"
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


def test_vary_runs_once_per_value_and_reports_the_capacity_per_value(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.1", "--seeds", "1", "--horizon", "60",
        "--vary", "demand.skew", "--values", "0.5", "1.5", "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    rows = [json.loads(line) for line in (tmp_path / "load_sweep.jsonl").read_text().splitlines()]
    assert [(row["varied"], row["value"]) for row in rows] == [
        ("demand.skew", 0.5),
        ("demand.skew", 1.5),
    ]
    capacity = json.loads((tmp_path / "capacity.json").read_text())
    assert capacity["varied"] == "demand.skew"
    assert set(capacity["by_value"]) == {"0.5", "1.5"}
    assert "## demand.skew = 0.5" in (tmp_path / "summary.md").read_text()

    assert sweep.main(argv) == 0
    assert len((tmp_path / "load_sweep.jsonl").read_text().splitlines()) == 2


def test_with_setting_changes_one_setting_and_validates_it():
    sweep = _script()
    spec = sweep.load_spec(CONFIG)
    assert sweep.with_setting(spec, "demand.skew", 1.5).demand.skew == 1.5
    try:
        sweep.with_setting(spec, "demand.skew", -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative skew should be rejected")


def test_the_sweep_can_stop_at_a_rate_no_seed_keeps_up_at(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.8", "0.9", "--seeds", "1",
        "--horizon", "60", "--out", str(tmp_path), "--stop-when-overloaded",
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    rows = [json.loads(line) for line in (tmp_path / "load_sweep.jsonl").read_text().splitlines()]
    assert [row["arrival_rate"] for row in rows] == [0.8]
    assert rows[0]["keeps_up"] is False
    assert json.loads((tmp_path / "capacity.json").read_text())["by_mode"]["fixed"]["capacity"] is None


def test_a_rate_whose_stock_does_not_fit_ends_the_sweep_without_a_crash(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "50.0", "--seeds", "1", "--horizon", "60",
        "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    assert not (tmp_path / "load_sweep.jsonl").exists() or not (
        tmp_path / "load_sweep.jsonl"
    ).read_text()
    assert json.loads((tmp_path / "capacity.json").read_text())["by_mode"] == {}


def test_traces_are_written_for_the_first_seeds_only(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.1", "--seeds", "2", "--horizon", "40",
        "--trace-seeds", "1", "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    traces = sorted((tmp_path / "traces").glob("*.jsonl"))
    assert len(traces) == 1 and traces[0].name.startswith("fixed_rate0.1_seed")
    # The header, one line per step, a stock line at the start and every 10 steps, the tasks.
    types = [json.loads(line)["type"] for line in traces[0].read_text().splitlines()]
    assert (types.count("header"), types.count("step"), types.count("stock")) == (
        1,
        40,
        5,
    )
    assert types[-1] == "tasks"
    assert json.loads(traces[0].read_text().splitlines()[0])["positions"]


def test_the_shelf_capacity_can_be_varied(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.1", "--seeds", "1", "--horizon", "40",
        "--vary", "storage.vertex_capacity", "--values", "300", "600", "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    rows = [json.loads(line) for line in (tmp_path / "load_sweep.jsonl").read_text().splitlines()]
    assert [(row["varied"], row["value"]) for row in rows] == [
        ("storage.vertex_capacity", 300.0),
        ("storage.vertex_capacity", 600.0),
    ]


def test_every_episode_writes_its_timing(tmp_path):
    sweep = _script()
    argv = [
        "--config", str(CONFIG), "--rates", "0.1", "--seeds", "2", "--horizon", "40",
        "--out", str(tmp_path),
    ]  # fmt: skip
    assert sweep.main(argv) == 0
    lines = [json.loads(line) for line in (tmp_path / "timing.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    seconds, calls = lines[0]["seconds"], lines[0]["calls"]
    for stage in ("conflict_resolution", "apply_and_lifecycle", "reward", "routing"):
        assert seconds[stage] >= 0.0 and calls[stage] > 0
    assert calls["reward"] == 40  # once per step
