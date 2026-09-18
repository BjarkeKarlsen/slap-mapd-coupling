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
        num_waiting_tasks=0,
        num_active_tasks=0,
        backlog=0,
        num_traversed_edges=0,
        mean_decision_runtime_seconds=0.01,
    )
    defaults.update(overrides)
    return defaults


def test_zero_completions_rejects_nonnull_service_time():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(mean_service_time=3.2))


def test_single_edge_enforces_entropy_convention():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(num_traversed_edges=1, traffic_entropy=0.4, traffic_concentration=0.6))


def test_entropy_concentration_must_sum_to_one():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(num_traversed_edges=2, traffic_entropy=0.3, traffic_concentration=0.5))


def test_backlog_must_equal_sum_of_waiting_and_active():
    with pytest.raises(ValidationError):
        RunMetrics(**_base(num_waiting_tasks=2, num_active_tasks=3, backlog=6))
