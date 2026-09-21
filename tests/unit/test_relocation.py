"""Unit tests for slap_mapd_coupling.storage.relocation."""

from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.storage.relocation import apply_relocation_cap


def _skus() -> dict[str, SkuType]:
    return {
        "a": SkuType(sku_id="a", unit_capacity=1.0),
        "b": SkuType(sku_id="b", unit_capacity=1.0),
    }


def _capacities() -> dict[int, float]:
    return {1: 20.0, 2: 20.0, 3: 20.0}


def _state(counts: dict[str, dict[int, int]]) -> StorageState:
    return StorageState(skus=_skus(), capacities=_capacities(), counts=counts)


def test_sufficient_cap_reaches_the_full_target():
    x_prev = _state({"a": {1: 10}, "b": {2: 10}})
    x_target = _state({"a": {3: 10}, "b": {2: 10}})
    priority = [("a", 3)]

    result = apply_relocation_cap(x_prev, x_target, priority, reassignment_cap=10)

    assert result.units("a", 3) == 10
    assert result.units("a", 1) == 0
    assert result.units("b", 2) == 10  # untouched, no relocation needed here


def test_zero_cap_leaves_x_prev_unchanged():
    x_prev = _state({"a": {1: 10}, "b": {2: 10}})
    x_target = _state({"a": {3: 10}, "b": {2: 10}})
    priority = [("a", 3)]

    result = apply_relocation_cap(x_prev, x_target, priority, reassignment_cap=0)

    assert result.units("a", 1) == 10
    assert result.units("a", 3) == 0
    assert result.units("b", 2) == 10


def test_partial_cap_relocates_only_up_to_the_budget():
    x_prev = _state({"a": {1: 10}})
    x_target = _state({"a": {3: 10}})
    priority = [("a", 3)]

    result = apply_relocation_cap(x_prev, x_target, priority, reassignment_cap=5)

    assert result.units("a", 3) == 5
    assert result.units("a", 1) == 5
    assert result.units("a", 1) + result.units("a", 3) == 10  # conserved


def test_priority_order_determines_which_sku_gets_the_shared_budget():
    # Two SKUs each want to relocate 10 units, but only 10 units of
    # budget total -- whichever comes first in `priority` should win.
    x_prev = _state({"a": {1: 10}, "b": {2: 10}})
    x_target = _state({"a": {3: 10}, "b": {3: 10}})

    priority_a_first = [("a", 3), ("b", 3)]
    result = apply_relocation_cap(x_prev, x_target, priority_a_first, reassignment_cap=10)
    assert result.units("a", 3) == 10
    assert result.units("b", 3) == 0
    assert result.units("b", 2) == 10

    priority_b_first = [("b", 3), ("a", 3)]
    result = apply_relocation_cap(x_prev, x_target, priority_b_first, reassignment_cap=10)
    assert result.units("b", 3) == 10
    assert result.units("a", 3) == 0
    assert result.units("a", 1) == 10


def test_lower_priority_cell_can_still_fit_in_remaining_budget():
    # First entry only partially fits, but a smaller later entry can
    # still be fully accepted out of what's left.
    x_prev = _state({"a": {1: 10}, "b": {2: 3}})
    x_target = _state({"a": {3: 10}, "b": {3: 3}})
    priority = [("a", 3), ("b", 3)]

    result = apply_relocation_cap(x_prev, x_target, priority, reassignment_cap=8)

    assert result.units("a", 3) == 8  # capped by budget
    assert result.units("a", 1) == 2
    assert result.units("b", 3) == 0  # no budget left over for b
    assert result.units("b", 2) == 3


def test_per_sku_totals_are_always_conserved():
    x_prev = _state({"a": {1: 4, 2: 6}})
    x_target = _state({"a": {3: 10}})
    priority = [("a", 3)]

    for cap in (0, 1, 4, 7, 10, 999):
        result = apply_relocation_cap(x_prev, x_target, priority, reassignment_cap=cap)
        total = sum(result.counts.get("a", {}).values())
        assert total == 10
