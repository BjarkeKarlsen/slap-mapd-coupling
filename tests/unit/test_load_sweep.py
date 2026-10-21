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
