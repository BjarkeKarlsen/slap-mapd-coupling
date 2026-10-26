"""The measured window of step 06 ([A66]) and the stall at fleet level (D18). The measures count
the tasks released after the warm-up, the stall is the longest stretch with no task finished,
and the demand ranking can rotate by a share of the SKUs."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.environment.multi_agent_env import popularity_ranking
from slap_mapd_coupling.evaluation.evaluator import (
    _planner_run,
    _to_run_metrics,
    backlog_slope,
    longest_completion_gap,
)
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "sweep"))
from welch import demand_cycle, welch_average  # noqa: E402

FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"
SEED = 300_000
HORIZON = 600
_STUDY: dict = {}


def _episode(warmup=0, stall_length=None):
    spec = load_spec(FLEET_8)
    if "study" not in _STUDY:
        _STUDY["study"] = build_study(spec)
    study = _STUDY["study"]
    config = build_experiment_config(spec, SEED, "fixed")
    config = ExperimentConfig.model_validate(
        {
            **config.model_dump(),
            "controller": None,
            "planner": "token_passing",
            "horizon": HORIZON,
            "warmup": warmup,
            "stall_length": stall_length,
        }
    )
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(HORIZON):
        step()
    return env, config


def test_without_a_warm_up_every_task_counts_and_the_backlog_is_the_last_of_the_trace():
    env, config = _episode()
    metrics = _to_run_metrics(env, config)
    assert metrics.warmup == 0
    assert metrics.backlog == env.log.backlog_by_t[-1]
    completed = [t for t in env.tasks if t.finish_time is not None]
    assert metrics.num_completed_tasks == len(completed)
    assert metrics.throughput == len(completed) / HORIZON
    assert metrics.backlog_slope == backlog_slope(env.log.backlog_by_t)


def test_with_a_warm_up_only_the_tasks_released_after_it_count():
    warmup = 200
    env, config = _episode(warmup)
    metrics = _to_run_metrics(env, config)
    picks = [
        t for t in env.tasks
        if t.kind == "pick" and t.finish_time is not None and t.release_time >= warmup
    ]
    assert picks
    expected = sum(t.finish_time - t.release_time for t in picks) / len(picks)
    assert metrics.mean_service_time == pytest.approx(expected)
    window = [t for t in env.tasks if t.release_time >= warmup]
    assert metrics.num_completed_tasks == sum(t.finish_time is not None for t in window)
    assert metrics.throughput == metrics.num_completed_tasks / (HORIZON - warmup)
    assert metrics.backlog_slope == backlog_slope(env.log.backlog_by_t, warmup)
    # The same run measured from 0 counts more tasks.
    assert _to_run_metrics(env, config.model_copy(update={"warmup": 0})).num_completed_tasks > (
        metrics.num_completed_tasks
    )


def test_the_longest_gap_runs_from_the_window_start_between_finishes_to_the_end():
    tasks = [SimpleNamespace(finish_time=t) for t in (None, 5, 12, 13, 30, 95)]
    assert longest_completion_gap(tasks, 0, 100) == 65  # 30 to 95
    assert longest_completion_gap(tasks, 0, 200) == 105  # 95 to the end
    assert longest_completion_gap(tasks, 40, 90) == 50  # nothing finished in the window
    assert longest_completion_gap(tasks, 10, 20) == 7  # 13 to the end at 20


def test_the_stall_is_a_long_gap_between_finished_tasks():
    env, config = _episode(stall_length=10_000)
    metrics = _to_run_metrics(env, config)
    assert metrics.longest_completion_gap == longest_completion_gap(env.tasks, 0, HORIZON)
    assert metrics.stalled is False
    short = config.model_copy(update={"stall_length": max(metrics.longest_completion_gap - 1, 1)})
    assert _to_run_metrics(env, short).stalled is True


def test_the_backlog_slope_reads_the_window_after_the_warm_up():
    trace = [0] * 50 + list(range(50))
    assert backlog_slope(trace, 50) == pytest.approx(100.0)
    assert backlog_slope(trace) == pytest.approx(100.0)  # the second half, as before
    assert backlog_slope([5] * 50 + [7] * 50, 50) == pytest.approx(0.0)


def test_a_warm_up_must_leave_a_window():
    config = build_experiment_config(load_spec(FLEET_8), 0, "fixed")
    with pytest.raises(ValueError, match="warmup"):
        ExperimentConfig.model_validate({**config.model_dump(), "warmup": config.horizon})


def test_the_rotation_share_turns_the_ranking_once_every_4000_steps():
    config = build_experiment_config(load_spec(FLEET_8), 0, "fixed")
    assert (config.demand_rotation_shift, config.demand_rotation_share) == (None, 0.05)
    assert config.rotation_shift(100) == 5
    assert config.rotation_shift(1929) == 96
    assert config.rotation_shift(10) == 1  # at least one place
    assert demand_cycle(config, 100) == 4000
    # The environment rotates by the resolved shift.
    env, _ = _episode()
    ids = list(env._sku_ids)
    shift = config.rotation_shift(len(ids))
    assert env._rotation_shift == shift
    assert popularity_ranking(ids, 200, 200, shift)[0] == ids[shift]
    assert demand_cycle(config, len(ids)) == 4000  # the SKU count is a multiple of 20


def test_the_shift_and_the_share_are_not_both_set():
    data = load_spec(FLEET_8).model_dump()
    data["demand"]["rotation_shift"] = 5
    with pytest.raises(ValueError, match="not both"):
        TrainingSpec.model_validate(data)


def test_welch_averages_are_centred():
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    assert welch_average(values, 2) == [0, 1, 2, 3, 4, 5]
    assert welch_average([1, 3, 1, 3, 1, 3, 1], 1) == [1, 5 / 3, 7 / 3, 5 / 3, 7 / 3, 5 / 3]
