"""Unit tests for slap_mapd_coupling.storage.fixed (F_fix)."""

from slap_mapd_coupling.core.graph import Vertex, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.storage.fixed import f_fix
from slap_mapd_coupling.storage.registry import get_storage_rule


def _state() -> StorageState:
    return StorageState(
        skus={
            "tea": SkuType(sku_id="tea", unit_capacity=1.0),
            "coffee": SkuType(sku_id="coffee", unit_capacity=2.0),
        },
        capacities={1: 20.0, 2: 10.0},
        counts={"tea": {1: 12, 2: 3}, "coffee": {1: 4}},
    )


def _graph() -> WarehouseGraph:
    return WarehouseGraph(vertices={1: Vertex(id=1), 2: Vertex(id=2)}, edges=(), wait_cost=0.0)


def test_f_fix_returns_the_input_state_unchanged():
    x_prev = _state()
    x_t = f_fix(
        x_prev,
        graph=_graph(),
        reassignment_cap=None,
        congestion_weight=None,
        demand_estimate={},
        traversal_estimate={},
        waiting_estimate={},
    )
    assert x_t == x_prev
    assert x_t is x_prev  # true identity, not just an equal copy


def test_f_fix_ignores_the_three_estimates_and_reassignment_cap():
    x_prev = _state()
    x_t = f_fix(
        x_prev,
        graph=_graph(),
        reassignment_cap=5,
        congestion_weight=0.5,
        demand_estimate={"tea": 999.0},
        traversal_estimate={(1, 2): 999.0},
        waiting_estimate={(1, 2): 999.0},
    )
    assert x_t is x_prev


def test_f_fix_registered_under_fixed():
    x_prev = _state()
    rule = get_storage_rule("fixed")
    assert rule(x_prev, _graph(), None, None, {}, {}, {}) is x_prev
