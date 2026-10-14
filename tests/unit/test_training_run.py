"""Unit tests for slap_mapd_coupling.viz.training_run."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from slap_mapd_coupling.viz.training_run import (
    Trace,
    find_run_files,
    format_trace_numbers,
    load_progress,
    load_trace,
    main,
    plot_trace,
    plot_trace_comparison,
    plot_training_progress,
    trace_summary,
)

PNG = b"\x89PNG"

# One row per step: (completed, agent 1, agent 2), each agent as (at, intent, moved, overridden).
# Completions at t = 2 and t = 4. The agents try to swap vertices 5 and 6 at t = 5 to 8.
_MOVE_1 = (1, 1, True, False)
_MOVE_2 = (2, 2, True, False)
_WAIT_2 = (2, None, False, False)
_BLOCK_1 = (5, 6, False, True)
_BLOCK_2 = (6, 5, False, True)
_ROWS = [
    (0, _MOVE_1, _MOVE_2),
    (1, _MOVE_1, _WAIT_2),
    (1, _MOVE_1, _MOVE_2),
    (2, _MOVE_1, _MOVE_2),
    (2, _BLOCK_1, _BLOCK_2),
    (2, _BLOCK_1, _BLOCK_2),
    (2, _BLOCK_1, _BLOCK_2),
    (2, _BLOCK_1, _BLOCK_2),
    (2, (5, None, False, False), _BLOCK_2),
    (2, (5, None, False, False), (6, None, False, False)),
]


def _record(at: int, intent: int | None, moved: bool, overridden: bool) -> dict[str, Any]:
    return {
        "at": at,
        "intent": intent,
        "moved": moved,
        "overridden": overridden,
        "task": None,
        "phase": None,
        "goal": None,
    }


def _steps(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "step",
            "t": t,
            "completed": completed,
            "released": completed + 3,
            "backlog": 3,
            "agents": {str(i + 1): _record(*agent) for i, agent in enumerate(agents)},
        }
        for t, (completed, *agents) in enumerate(rows, start=1)
    ]


def _header(seed: int = 5) -> dict[str, Any]:
    return {
        "type": "header",
        "format": 1,
        "config": {"controller": "centralised", "seed": seed, "horizon": 10},
        "vertices": [],
        "edges": [],
        "initial": {"1": 1, "2": 2},
    }


def _trace(arm: str = "centralised", episode: int | None = 1, seed: int = 5) -> Trace:
    return Trace(arm, episode, _header(seed), _steps(_ROWS))


def _write_trace(path: Path, seed: int = 5) -> None:
    lines = [_header(seed)] + _steps(_ROWS)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _write_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _full_rows(n: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "iteration": i,
            "episode_return_mean": 10.0 * i,
            "tasks_completed": 2.0 * i,
            "tasks_released": 30.0,
            "backlog": 30.0 - i,
            "steps_since_last_completion": 5.0,
            "standing_still_fraction": 0.3,
            "episode_step": 100.0,
            "return_shaping": 5.0 * i,
            "return_deliver": 20.0 * i,
            "return_override": -1.0 * i,
            "return_congestion": -0.5 * i,
            "override_rate": 0.1,
            "episode_tasks_completed": 2.0 * i,
            "episode_tasks_released": 30.0,
            "episode_max_agent_override_share": 0.4,
        }
        for i in range(1, n + 1)
    ]


def _old_rows(n: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "iteration": i,
            "episode_return_mean": None,
            "tasks_completed": 1.5 * i,
            "tasks_released": 40.0,
            "backlog": 20.0,
            "steps_since_last_completion": 7.0,
            "standing_still_fraction": 0.4,
        }
        for i in range(1, n + 1)
    ]


# --- readers ---------------------------------------------------------------


def test_load_progress_turns_nulls_into_nan(tmp_path: Path):
    path = tmp_path / "x_training_progress.jsonl"
    _write_progress(path, _old_rows(3))

    log = load_progress(path)

    assert all(math.isnan(v) for v in log["episode_return_mean"])
    assert list(log["tasks_completed"]) == [1.5, 3.0, 4.5]


def test_load_progress_tolerates_missing_keys(tmp_path: Path):
    path = tmp_path / "x_training_progress.jsonl"
    _write_progress(
        path, [{"iteration": 1, "backlog": 2.0}, {"iteration": 2, "override_rate": 0.5}]
    )

    log = load_progress(path)

    assert math.isnan(log["override_rate"][0]) and log["override_rate"][1] == 0.5
    assert log["backlog"][0] == 2.0 and math.isnan(log["backlog"][1])


def test_load_trace_reads_arm_episode_and_seed_from_the_file(tmp_path: Path):
    path = tmp_path / "decentralised_episode_3.jsonl"
    _write_trace(path, seed=7)

    trace = load_trace(path)

    assert (trace.arm, trace.episode, trace.seed) == ("decentralised", 3, 7)
    assert len(trace.steps) == len(_ROWS)


def test_load_trace_falls_back_to_the_controller_for_an_odd_file_name(tmp_path: Path):
    path = tmp_path / "somewhere.jsonl"
    _write_trace(path)

    trace = load_trace(path)

    assert (trace.arm, trace.episode) == ("centralised", None)


# --- numbers ---------------------------------------------------------------


def test_trace_summary_finds_the_longest_gap_without_a_completion():
    numbers = trace_summary(_trace())

    assert (numbers.completed, numbers.released, numbers.backlog) == (2, 5, 3)
    assert numbers.last_completion_t == 4
    # Completions at t = 2 and 4, so the gaps are 2, 2 and 6 steps (4 to the last step, 10).
    assert (numbers.longest_gap_start, numbers.longest_gap_length) == (4, 6)
    assert numbers.steps_after_last_completion == 6


def test_trace_summary_without_a_completion_is_one_gap_from_the_start():
    steps = _steps([(0, _MOVE_1, _MOVE_2)] * 4)

    numbers = trace_summary(Trace("a", 1, _header(), steps))

    assert numbers.last_completion_t is None
    assert (numbers.longest_gap_start, numbers.longest_gap_length) == (0, 4)
    assert numbers.steps_after_last_completion == 4


def test_trace_summary_shares_per_agent():
    numbers = trace_summary(_trace())

    first, second = numbers.agents
    assert (first.moved, first.waited, first.overridden) == pytest.approx((0.4, 0.2, 0.4))
    assert (second.moved, second.waited, second.overridden) == pytest.approx((0.3, 0.2, 0.5))
    assert numbers.overridden_share == pytest.approx(0.45)


def test_trace_summary_counts_the_head_on_swap_and_the_overridden_pairs():
    numbers = trace_summary(_trace())

    assert numbers.swap_steps == 4
    assert numbers.swap_steps_after_last_completion == 4
    assert numbers.swap_pairs == ((1, 2, 4),)
    # Agent 2 is overridden on 6 -> 5 five times after the last completion, agent 1 on 5 -> 6 four.
    assert numbers.top_overrides[:2] == ((6, 5, 5), (5, 6, 4))


def test_trace_summary_does_not_count_a_swap_when_one_agent_does_not_want_it():
    rows = [(0, (5, 6, False, True), (6, None, False, False))]

    numbers = trace_summary(Trace("a", 1, _header(), _steps(rows)))

    assert numbers.swap_steps == 0 and numbers.swap_pairs == ()


def test_trace_summary_rejects_an_empty_trace():
    with pytest.raises(ValueError):
        trace_summary(Trace("a", 1, _header(), []))


def test_format_trace_numbers_names_the_figures_it_found():
    trace = _trace()

    text = format_trace_numbers(trace, trace_summary(trace))

    assert "2 of 5 released tasks completed" in text
    assert "last completion at step 4, then 6 steps without one" in text
    assert "agents 1 and 2 x4" in text


# --- figures ---------------------------------------------------------------


def _is_png(path: Path | None) -> bool:
    return path is not None and path.stat().st_size > 0 and path.read_bytes()[:4] == PNG


def test_plot_training_progress_with_every_key_for_two_arms(tmp_path: Path):
    for arm in ("centralised", "section"):
        _write_progress(tmp_path / f"{arm}{'_training_progress.jsonl'}", _full_rows())
    logs = {
        a: load_progress(tmp_path / f"{a}_training_progress.jsonl")
        for a in ("centralised", "section")
    }

    out = plot_training_progress(logs, tmp_path / "p.png")

    assert _is_png(out)


def test_plot_training_progress_draws_a_long_log(tmp_path: Path):
    path = tmp_path / "a_training_progress.jsonl"
    _write_progress(path, _full_rows(300))

    assert _is_png(plot_training_progress({"a": load_progress(path)}, tmp_path / "p.png"))


def test_plot_training_progress_survives_an_old_style_log(tmp_path: Path):
    path = tmp_path / "a_training_progress.jsonl"
    _write_progress(path, _old_rows())

    out = plot_training_progress({"a": load_progress(path)}, tmp_path / "p.png")

    assert _is_png(out)


def test_plot_training_progress_skips_a_log_with_nothing_to_draw(tmp_path: Path):
    path = tmp_path / "a_training_progress.jsonl"
    _write_progress(path, [{"iteration": 1, "episode_return_mean": None}])

    out = plot_training_progress({"a": load_progress(path)}, tmp_path / "p.png")

    assert out is None and not (tmp_path / "p.png").exists()


def test_plot_trace_writes_a_png(tmp_path: Path):
    assert _is_png(plot_trace(_trace(), tmp_path / "t.png"))


def test_plot_trace_comparison_writes_a_png_for_arms_and_seeds(tmp_path: Path):
    traces = [
        _trace(arm, episode, seed)
        for arm in ("centralised", "decentralised")
        for episode, seed in ((1, 5), (2, 6))
    ]

    assert _is_png(plot_trace_comparison(traces, tmp_path / "c.png"))


def test_plot_trace_comparison_with_one_seed_and_an_unknown_arm(tmp_path: Path):
    traces = [_trace("centralised"), _trace("my_arm")]

    assert _is_png(plot_trace_comparison(traces, tmp_path / "c.png"))


# --- command line ----------------------------------------------------------


def test_find_run_files_looks_in_the_run_dir_and_its_traces_folder(tmp_path: Path):
    _write_progress(tmp_path / "section_training_progress.jsonl", _old_rows())
    (tmp_path / "traces").mkdir()
    _write_trace(tmp_path / "traces" / "section_episode_1.jsonl")

    logs, traces = find_run_files(tmp_path)

    assert list(logs) == ["section"]
    assert [p.name for p in traces] == ["section_episode_1.jsonl"]


def test_main_writes_the_figures_and_prints_the_numbers(tmp_path: Path, capsys):
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    for arm in ("centralised", "section"):
        _write_progress(run / f"{arm}_training_progress.jsonl", _full_rows())
        _write_trace(run / "traces" / f"{arm}_episode_1.jsonl")
    out = tmp_path / "figs"

    main(["--run-dir", str(run), "--out", str(out)])

    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "trace_centralised_episode_1.png",
        "trace_comparison.png",
        "trace_section_episode_1.png",
        "training_progress.png",
    ]
    assert "2 of 5 released tasks completed" in capsys.readouterr().out


def test_main_with_only_an_old_log_writes_the_training_figure(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_progress(run / "a_training_progress.jsonl", _old_rows())

    main(["--run-dir", str(run), "--out", str(tmp_path / "figs")])

    assert [p.name for p in (tmp_path / "figs").iterdir()] == ["training_progress.png"]


def test_main_rejects_a_folder_without_run_files(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["--run-dir", str(tmp_path), "--out", str(tmp_path / "figs")])
