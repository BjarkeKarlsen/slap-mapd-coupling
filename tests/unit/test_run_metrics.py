"""Unit tests for slap_mapd_coupling.evaluation.metrics."""

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.evaluation.metrics import RunMetrics


def _base(**overrides) -> dict:
    defaults = dict(
        storage_mode="fixed",
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=5,
        arrival_rate=1.0,
        seed=0,
        horizon=100,
        num_completed_tasks=0,
        throughput=0.0,
        keep_up_ratio=0.0,
        num_waiting_tasks=0,
        num_active_tasks=0,
        backlog=0,
        num_storage_updates=0,
        mean_relocated_units=0.0,
        mean_decision_runtime_seconds=0.01,
        mean_crowding=0.0,
    )
    defaults.update(overrides)
    return defaults


def _with_service_time(**overrides) -> dict:
    """tau_1 to tau_3 of the thesis's fig:timeline: parts 2 + 5.33 + 1."""
    values = dict(
        num_completed_tasks=3,
        throughput=0.15,
        keep_up_ratio=0.75,
        mean_service_time=25 / 3,
        mean_wait_for_agent=2.0,
        mean_travel_time=16 / 3,
        mean_blocked_time=1.0,
    )
    values.update(overrides)
    return _base(**values)


def test_zero_completions_rejects_nonnull_service_time():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(mean_service_time=3.2))


def test_zero_completions_rejects_any_nonnull_part():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(mean_travel_time=1.0))


def test_completions_require_every_part():
    with pytest.raises(ValidationError):
        RunMetrics(**_with_service_time(mean_travel_time=None))


def test_parts_that_add_up_are_accepted():
    metrics = RunMetrics(**_with_service_time())
    assert metrics.mean_service_time == pytest.approx(8.333, abs=1e-3)


def test_parts_must_add_up_to_service_time():
    with pytest.raises(ValidationError):
        RunMetrics(**_with_service_time(mean_blocked_time=2.0))


def test_crowding_is_a_fraction():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(mean_crowding=1.5))


def test_no_storage_update_relocates_nothing():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(num_storage_updates=0, mean_relocated_units=2.0))


def test_backlog_must_equal_sum_of_waiting_and_active():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(num_waiting_tasks=2, num_active_tasks=3, backlog=6))
