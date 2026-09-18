"""Unit tests for slap_mapd_coupling.core.experiment_config."""

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig


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
        wait_cost=0.0,
    )
    defaults.update(overrides)
    return defaults


def test_fixed_storage_rejects_epoch_length():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base(storage_epoch_length=10))


def test_demand_storage_requires_epoch_length_and_cap():
    with pytest.raises(ValidationError, match="storage_epoch_length"):
        ExperimentConfig(**_base(storage_mode="demand"))


def test_communication_requires_decentralised():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base(communication=True))
