"""Unit tests for slap_mapd_coupling.evaluation.comparison (eq:rqformal)."""

import pytest
from scipy import stats

from slap_mapd_coupling.evaluation.comparison import compare_paired
from slap_mapd_coupling.evaluation.metrics import RunMetrics


def _run(seed: int, service_time: float, storage_mode: str = "fixed", **overrides) -> RunMetrics:
    """A run whose mean pick service time is `service_time` and whose total service time is
    ten times it, as if over ten tasks."""
    values = dict(
        storage_mode=storage_mode,
        controller="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=seed,
        horizon=100,
        observation_depth=3,
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
        total_service_time=round(10 * service_time),
        total_travel_time=round(10 * service_time),
        num_storage_updates=0 if storage_mode == "fixed" else 4,
        mean_relocated_units=0.0 if storage_mode == "fixed" else 2.0,
        mean_relocation_distance=0.0 if storage_mode == "fixed" else 6.0,
        mean_decision_runtime_seconds=0.001,
        mean_crowding=0.0,
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


def _split_run(seed, storage_mode, wait, travel, blocked):
    """A run that keeps up, with its service time split into the three parts given."""
    return _run(
        seed,
        wait + travel + blocked,
        storage_mode=storage_mode,
        mean_wait_for_agent=wait,
        mean_travel_time=travel,
        mean_blocked_time=blocked,
        total_wait_for_agent=round(10 * wait),
        total_travel_time=round(10 * travel),
        total_blocked_time=round(10 * blocked),
    )


def test_the_relative_gain_is_the_gain_over_the_route_time_under_fixed_storage():
    # Route time under F_fix, total travel plus blocked, is 80 and 120. The gains are 20 and 20.
    fixed = [_split_run(0, "fixed", 2.0, 6.0, 2.0), _split_run(1, "fixed", 0.0, 8.0, 4.0)]
    adaptive = [_split_run(0, "demand", 1.0, 5.0, 2.0), _split_run(1, "demand", 0.0, 7.0, 3.0)]
    result = compare_paired(fixed, adaptive)
    assert result.gains == pytest.approx((20.0, 20.0))
    relative = [2.0 / 8.0, 2.0 / 12.0]
    mean = sum(relative) / 2
    half = stats.t.ppf(0.975, df=1) * stats.tstd(relative) / 2**0.5
    assert result.mean_relative_gain == pytest.approx(mean)
    assert (result.relative_ci_low, result.relative_ci_high) == pytest.approx(
        (mean - half, mean + half)
    )


def test_the_relative_gain_is_unset_when_a_run_does_not_keep_up():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0])
    adaptive[1] = _run(1, 10.0, storage_mode="demand", keeps_up=False)
    result = compare_paired(fixed, adaptive)
    assert result.mean_relative_gain is None
    assert result.relative_ci_low is None and result.relative_ci_high is None


def test_consistent_gain_shortens_service_time():
    fixed, adaptive = _arms([10.0, 11.0, 12.0, 10.5], [9.0, 9.8, 11.1, 9.4])
    result = compare_paired(fixed, adaptive)
    assert result.all_keep_up
    assert result.gains == pytest.approx((10.0, 12.0, 9.0, 11.0))
    assert result.mean_gain == pytest.approx(10.5)
    assert result.ci_low > 0.0
    assert result.shortens_service_time


def test_the_gain_is_in_total_service_time_and_the_mean_pick_gain_is_reported_next_to_it():
    fixed, adaptive = _arms([10.0, 11.0, 12.0, 10.5], [9.0, 9.8, 11.1, 9.4])
    # The adaptive arm makes more tasks, so its total grows while its picks get faster.
    adaptive = [
        run.model_copy(
            update=dict(
                total_service_time=run.total_service_time + 20,
                total_travel_time=run.total_travel_time + 20,
            )
        )
        for run in adaptive
    ]
    result = compare_paired(fixed, adaptive)
    assert result.gains == pytest.approx((-10.0, -8.0, -11.0, -9.0))
    assert not result.shortens_service_time
    assert result.mean_pick_gain == pytest.approx(1.05)
    gains = [1.0, 1.2, 0.9, 1.1]
    expected = stats.t.interval(0.95, df=3, loc=1.05, scale=stats.sem(gains))
    assert (result.pick_ci_low, result.pick_ci_high) == pytest.approx(expected)


def test_interval_matches_a_paired_t_test():
    fixed, adaptive = _arms([10.0, 11.0, 12.0, 10.5], [9.0, 9.8, 11.1, 9.4])
    result = compare_paired(fixed, adaptive)
    gains = [10.0, 12.0, 9.0, 11.0]
    expected = stats.t.interval(0.95, df=3, loc=10.5, scale=stats.sem(gains))
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
    result = compare_paired(fixed, adaptive)
    assert result.all_keep_up is None
    assert result.mean_gain is None
    assert not result.shortens_service_time


def test_a_run_that_falls_behind_is_false_not_none():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0])
    adaptive[1] = _run(1, 10.0, storage_mode="demand", keeps_up=False)
    assert compare_paired(fixed, adaptive).all_keep_up is False


def test_runs_with_different_windows_are_not_a_pair():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0], observation_depth=5)
    with pytest.raises(ValueError, match="observation_depth"):
        compare_paired(fixed, adaptive)


def test_every_scored_quantity_has_an_interval_per_arm():
    crowding = [0.1, 0.2, 0.3, 0.4]
    fixed = [_run(seed, 10.0, mean_crowding=c) for seed, c in enumerate(crowding)]
    adaptive = [_run(seed, 9.0, storage_mode="demand", mean_crowding=0.05) for seed in range(4)]
    result = compare_paired(fixed, adaptive)
    summary = result.fixed_summary["mean_crowding"]
    expected = stats.t.interval(0.95, df=3, loc=0.25, scale=stats.sem(crowding))
    assert summary.mean == pytest.approx(0.25)
    assert (summary.ci_low, summary.ci_high) == pytest.approx(expected)
    assert result.adaptive_summary["mean_crowding"].ci_low == pytest.approx(0.05)
    assert result.adaptive_summary["mean_service_time"].mean == pytest.approx(9.0)
    assert result.adaptive_summary["total_service_time"].mean == pytest.approx(90.0)
    for name in ("keep_up_ratio", "throughput", "mean_relocated_units", "mean_travel_time"):
        assert name in result.fixed_summary


def test_service_time_is_summarised_only_when_every_run_keeps_up():
    fixed, adaptive = _arms([10.0, 11.0], [9.0, 10.0], keeps_up=None)
    result = compare_paired(fixed, adaptive)
    assert "mean_service_time" not in result.fixed_summary
    assert "mean_crowding" in result.fixed_summary


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
