"""Unit tests for slap_mapd_coupling.core.storage_state."""

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.storage_state import SkuType, StorageState


def _tea_coffee_state(coffee_count: int = 4) -> StorageState:
    # Thesis's own worked example: cap(A)=20, b_tea=1 * 12 + b_coffee=2 * 4 = 20.
    return StorageState(
        skus={
            "tea": SkuType(sku_id="tea", unit_capacity=1.0),
            "coffee": SkuType(sku_id="coffee", unit_capacity=2.0),
        },
        capacities={1: 20.0},
        counts={"tea": {1: 12}, "coffee": {1: coffee_count}},
    )


def test_storage_state_feasible_configuration_accepted():
    state = _tea_coffee_state(coffee_count=4)
    assert state.used_capacity(1) == pytest.approx(20.0)


def test_storage_state_over_capacity_rejected():
    with pytest.raises(ValidationError, match="over capacity"):
        _tea_coffee_state(coffee_count=5)


def test_storage_state_counts_reference_unknown_sku_rejected():
    with pytest.raises(ValidationError):
        StorageState(skus={}, capacities={1: 10.0}, counts={"ghost": {1: 1}})


def test_with_units_reraises_on_infeasible_update():
    state = _tea_coffee_state(coffee_count=4)
    with pytest.raises(ValidationError):
        state.with_units("coffee", 1, 5)
