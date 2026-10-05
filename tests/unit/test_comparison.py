"""Unit tests for slap_mapd_coupling.evaluation.comparison (eq:rqformal)."""

import pytest
from scipy import stats

from slap_mapd_coupling.evaluation.comparison import compare_paired
from slap_mapd_coupling.evaluation.metrics import RunMetrics


def _run(seed: int, service_time: float, storage_mode: str = "fixed", **overrides) -> RunMetrics:
    values = dict(
        storage_mode=storage_mode,
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=seed,
        horizon=100,
        num_completed_tasks=90,
        throughput=0.9,
        keep_up_ratio=0.9,
        keeps_up=True,
        num_waiting_tasks=1,
        num_active_tasks=2,
        backlog=3,
        # The split only has to add up; put all of it in travel.
        mean_service_time=service_time,
        mean_wait_for_agent=0.0,
        mean_travel_time=service_time,
        mean_blocked_time=0.0,
        num_storage_updates=0 if storage_mode == "fixed" else 4,
        mean_relocated_units=0.0 if storage_mode == "fixed" else 2.0,
        mean_decision_runtime_seconds=0.001,
    )
    values.update(overrides)
    return RunMetrics(**values)


def _arms(fixed_times, adaptive_times, **adaptive_overrides):
    fixed = [_run(seed, t) for seed, t in enumerate(fixed_times)]
    adaptive = [
        _run(seed, t, storage_mode="demand", **adaptive_overrides)
        for seed, t in enumerate(adaptive_times)
    ]
    return fixed, adaptive


def test_consistent_gain_shortens_service_time():
    fixed, adaptive = _arms([10.0, 11.0, 12.0, 10.5], [9.0, 9.8, 11.1, 9.4])
    result = compare_paired(fixed, adaptive)
    assert result.all_keep_up
    assert result.gains == pytest.approx((1.0, 1.2, 0.9, 1.1))
    assert result.mean_gain == pytest.approx(1.05)
    assert result.ci_low > 0.0
    assert result.shortens_service_time


def test_interval_matches_a_paired_t_test():
    fixed, adaptive = _arms([10.0, 11.0, 12.0, 10.5], [9.0, 9.8, 11.1, 9.4])
    result = compare_paired(fixed, adaptive)
    gains = [1.0, 1.2, 0.9, 1.1]
    expected = stats.t.interval(0.95, df=3, loc=1.05, scale=stats.sem(gains))
    assert (result.ci_low, result.ci_high) == pytest.approx(expected)


def test_gain_that_changes_sign_is_not_a_win():
    # Larger mean gain, but it loses in two of five seeds.
    fixed, adaptive = _arms([10.0, 10.0, 10.0, 10.0, 10.0], [7.5, 7.0, 10.5, 7.2, 10.8])
    result = compare_paired(fixed, adaptive)
    assert result.mean_gain > 0.0
    assert result.ci_low <= 0.0
    assert not result.shortens_service_time


def test_a_run_that_does_not_keep_up_leaves_the_comparison_undecided():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0])
    adaptive[1] = _run(1, 10.0, storage_mode="demand", keeps_up=False)
    result = compare_paired(fixed, adaptive)
    assert not result.all_keep_up
    assert result.mean_gain is None
    assert not result.shortens_service_time


def test_unset_keep_up_check_leaves_the_comparison_undecided():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0], keeps_up=None)
    assert not compare_paired(fixed, adaptive).all_keep_up


def test_seeds_must_match():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0])
    adaptive[1] = _run(5, 10.0, storage_mode="demand")
    with pytest.raises(ValueError):
        compare_paired(fixed, adaptive)


def test_runs_must_be_matched_conditions():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0], num_agents=8)
    with pytest.raises(ValueError):
        compare_paired(fixed, adaptive)


def test_fixed_arm_must_be_fixed_storage():
    _, adaptive = _arms([10.0, 11.0], [9.0, 10.0])
    with pytest.raises(ValueError):
        compare_paired(adaptive, adaptive)


def test_one_seed_is_not_enough_for_an_interval():
    fixed, adaptive = _arms([10.0], [9.0])
    with pytest.raises(ValueError):
        compare_paired(fixed, adaptive)
