"""Unit tests for slap_mapd_coupling.viz.study_report."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from slap_mapd_coupling.viz.study_report import (
    assign_looks,
    collect,
    load_sweep,
    main,
    parse_run,
    summary_text,
    sweep_stats,
    trace_stats,
    training_stats,
)
from tests.unit.test_training_run import (
    _full_rows,
    _old_rows,
    _trace,
    _write_progress,
    _write_trace,
)


def _run_folder(root: Path, arms: tuple[str, ...]) -> Path:
    (root / "traces").mkdir(parents=True)
    for arm in arms:
        _write_progress(root / f"{arm}_training_progress.jsonl", _full_rows(30))
        _write_trace(root / "traces" / f"{arm}_episode_1.jsonl")
    return root


def _sweep_rows() -> list[dict]:
    rows = []
    for condition in ("token_passing", "centralised"):
        for rate in (0.15, 0.3):
            for seed, done in ((5, 80), (6, 90)):
                rows.append(
                    {
                        "condition": condition,
                        "kind": "planner" if condition == "token_passing" else "controller",
                        "arrival_rate": rate,
                        "seed": seed,
                        "horizon": 630,
                        "num_completed_tasks": done,
                        "backlog": 10,
                        "released": done + 10,
                        "mean_service_time": 20.0 + seed,
                        "keeps_up": rate < 0.2 or done == 90,
                    }
                )
    return rows


def test_parse_run_reads_an_optional_name():
    assert parse_run("runs/a") == ("", Path("runs/a"))
    assert parse_run("comm=runs/b") == ("comm", Path("runs/b"))


def test_collect_labels_a_named_run_as_the_arm_plus_the_name(tmp_path: Path):
    base = _run_folder(tmp_path / "base", ("centralised", "decentralised"))
    comm = _run_folder(tmp_path / "comm", ("decentralised",))

    conditions = collect([("", base), ("comm", comm)])

    assert [c.label for c in conditions] == ["centralised", "decentralised", "decentralised+comm"]
    assert all(c.log is not None and len(c.traces) == 1 for c in conditions)


def test_training_stats_reads_the_late_window_and_counts_collapses():
    rows = _full_rows(30)
    for i, row in enumerate(rows, start=1):
        row["episode_return_mean"] = 100.0 if i != 25 else 10.0
    path_rows = {k: [r[k] for r in rows] for k in rows[0]}
    log = {k: np.array(v, dtype=float) for k, v in path_rows.items()}

    stats = training_stats(log, window=5)

    assert stats["iterations"] == 30
    assert stats["late_return"] == pytest.approx(100.0)
    assert stats["collapses"] == 1  # the drop at iteration 25, after the warm-up
    assert stats["late_completion_share"] == pytest.approx(
        sum(2.0 * i / 30 for i in range(26, 31)) / 5
    )
    assert stats["late_override_rate"] == pytest.approx(0.1)


def test_training_stats_leaves_out_what_an_old_log_misses():
    log = {
        k: np.array([r[k] if r[k] is not None else math.nan for r in _old_rows(6)])
        for k in _old_rows(6)[0]
    }

    stats = training_stats(log, window=3)

    assert "late_return" not in stats
    assert math.isnan(stats["late_override_rate"])


def test_trace_stats_pools_the_episodes():
    stats = trace_stats([_trace(episode=1), _trace(episode=2)])

    assert stats["episodes"] == 2
    assert stats["completed"] == 4 and stats["released"] == 10
    assert stats["completion_share"] == pytest.approx(0.4)
    assert stats["swap_steps"] == pytest.approx(4.0)


def test_sweep_stats_gives_the_mean_and_the_range_over_seeds(tmp_path: Path):
    path = tmp_path / "load_sweep.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _sweep_rows()) + "\n", encoding="utf-8")

    stats = sweep_stats(load_sweep(tmp_path))  # a folder works as well as the file

    entry = stats["centralised"][0.3]
    assert entry["seeds"] == 2
    assert entry["completed"] == 170 and entry["released"] == 190
    assert entry["share_min"] == pytest.approx(80 / 90)
    assert entry["share_max"] == pytest.approx(90 / 100)
    assert entry["service"] == pytest.approx(25.5)
    assert (entry["keeps_up"], entry["keep_up_known"]) == (1, 2)
    assert (
        stats["centralised"][0.15]["keeps_up"],
        stats["centralised"][0.15]["keep_up_known"],
    ) == (
        2,
        2,
    )


def test_assign_looks_is_distinct_per_label_and_keeps_the_arm_looks():
    labels = ["centralised", "decentralised+comm", "token_passing", "other"]
    looks = assign_looks(labels)

    assert len(set(looks.values())) == len(labels)
    assert looks["centralised"] == assign_looks(["centralised"])["centralised"]


def test_summary_states_the_sample_sizes(tmp_path: Path):
    base = _run_folder(tmp_path / "base", ("centralised",))
    conditions = collect([("", base)])
    training = {c.label: training_stats(c.log, 5) for c in conditions}
    evaluation = {c.label: trace_stats(c.traces) for c in conditions}

    text = summary_text(conditions, training, evaluation, sweep_stats(_sweep_rows()), 5)

    assert "last 5 iterations" in text
    assert "| centralised | 30 |" in text
    assert "| 1 | 2 / 5 | 0.40 |" in text  # one episode, 2 of 5 released completed
    assert "## Load sweep" in text and "token_passing" in text
    assert "| 2 / 2 |" in text and "| 1 / 2 |" in text


def test_main_writes_the_overview_the_sweep_figure_and_the_summary(tmp_path: Path, capsys):
    base = _run_folder(tmp_path / "base", ("centralised", "section"))
    comm = _run_folder(tmp_path / "comm", ("decentralised",))
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "load_sweep.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _sweep_rows()) + "\n", encoding="utf-8"
    )
    out = tmp_path / "report"

    main(["--run", str(base), "--run", f"comm={comm}", "--sweep", str(sweep), "--out", str(out)])

    assert sorted(p.name for p in out.iterdir()) == [
        "load_sweep.png",
        "study_overview.png",
        "study_summary.md",
    ]
    assert "decentralised+comm" in (out / "study_summary.md").read_text(encoding="utf-8")


def test_main_with_only_an_old_log_still_writes_a_summary(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_progress(run / "a_training_progress.jsonl", _old_rows())

    main(["--run", str(run), "--out", str(tmp_path / "report")])

    assert (tmp_path / "report" / "study_summary.md").exists()


def test_main_rejects_no_input_and_a_bad_window(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "report")])
    with pytest.raises(SystemExit):
        main(["--run", str(tmp_path), "--window", "0", "--out", str(tmp_path / "report")])
