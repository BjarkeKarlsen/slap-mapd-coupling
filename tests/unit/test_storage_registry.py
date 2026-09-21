"""Unit tests for slap_mapd_coupling.storage.base and .registry."""

import pytest

from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.storage.base import StorageRule
from slap_mapd_coupling.storage.registry import (
    get_storage_rule,
    register_storage_rule,
    registered_storage_rules,
)


def _state() -> StorageState:
    return StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={1: 20.0},
        counts={"tea": {1: 12}},
    )


def _dummy_rule(x_prev, demand_estimate, traversal_estimate, waiting_estimate):
    return x_prev


def test_dummy_rule_satisfies_the_protocol():
    assert isinstance(_dummy_rule, StorageRule)


def test_register_and_get_roundtrip():
    register_storage_rule("test-dummy")(_dummy_rule)
    assert "test-dummy" in registered_storage_rules()
    rule = get_storage_rule("test-dummy")
    state = _state()
    assert rule(state, {}, {}, {}) is state


def test_get_unknown_storage_rule_raises_key_error():
    with pytest.raises(KeyError):
        get_storage_rule("does-not-exist")


def test_double_registration_raises():
    register_storage_rule("test-dummy-double")(_dummy_rule)
    with pytest.raises(ValueError):
        register_storage_rule("test-dummy-double")(_dummy_rule)
