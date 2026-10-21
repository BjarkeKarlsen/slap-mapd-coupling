"""Arrivals and put-away under the arrival stock model (storage/arrival.py).

Every Delta steps new stock arrives at the delivery vertices. Each SKU is filled up to
max(floor, ceil((1 + epsilon) * its orders in the last period)) plus the orders still waiting
for it, the same amount under every rule. The rule decides which shelves it goes to, and
nothing already on a shelf moves. "fixed" fills the home shelf first, "demand" puts the most
demanded SKUs on the nearest shelves, and "congestion" ranks the shelves by the
congestion-aware score.
"""

from pathlib import Path

import pytest

import slap_mapd_coupling.storage.arrival as arrival
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.environment.multi_agent_env import _home_shelves, _split_skus
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.storage.arrival import (
    arrival_amounts,
    get_putaway_rule,
    register_putaway_rule,
    registered_putaway_rules,
)
from slap_mapd_coupling.storage.congestion import ranked_storage_vertices_for
from slap_mapd_coupling.training.spec import (
    ArrivalSettings,
    StorageRuleSettings,
    build_experiment_config,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
DELTA = 100
RULES = ("fixed", "demand", "congestion")


def _line() -> WarehouseGraph:
    """The corridor of tests/unit/test_initial_placement.py: seven plain vertices 10 to 16, with
    a role vertex hanging off every one but the middle.

    corridor  10 11 12 13 14 15 16
    hanging    6  5  4  .  2  1  0
    role       D  S  S  .  D  S  S

    Two role vertices are 2 apart plus their distance along the corridor. The distance from the
    nearest delivery vertex is 3 for vertices 1 and 5 and 4 for vertices 0 and 4, so the storage
    vertices by access distance, ties by number, are [1, 5, 0, 4], and by number [0, 1, 4, 5].
    """
    hanging = {
        10: (6, VertexRole(delivery=True)),
        11: (5, VertexRole(storage=True)),
        12: (4, VertexRole(storage=True)),
        14: (2, VertexRole(delivery=True)),
        15: (1, VertexRole(storage=True)),
        16: (0, VertexRole(storage=True)),
    }
    vertices = {v: Vertex(id=v) for v in range(10, 17)}
    vertices.update({v: Vertex(id=v, role=role) for v, role in hanging.values()})
    pairs = [(a, a + 1) for a in range(10, 16)] + [(a, v) for a, (v, _) in hanging.items()]
    edges = []
    for a, b in pairs:
        edges += [Edge(source=a, target=b, cost=1.0), Edge(source=b, target=a, cost=1.0)]
    return WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=0.0)


ACCESS = {1: 3.0, 5: 3.0, 0: 4.0, 4: 4.0}  # min over delivery vertices of d_G(g, v)


def _state(counts, capacity=4.0, unit_capacity=1.0, skus=("a", "b")):
    return StorageState(
        skus={k: SkuType(sku_id=k, unit_capacity=unit_capacity) for k in skus},
        capacities={v: capacity for v in (0, 1, 4, 5)},
        counts=counts,
    )


def _put_away(
    rule, x_prev, amounts, home=None, demand=None, traversal=None, waiting=None, beta=1.0
):
    return get_putaway_rule(rule)(
        x_prev, _line(), amounts, home or {}, demand or {}, traversal or {}, waiting or {}, beta
    )


# The amounts.


def test_a_sku_with_no_orders_is_filled_up_to_the_floor():
    x_prev = _state({"a": {1: 1}})
    assert arrival_amounts(x_prev, {}, DELTA, 1.0, 3, {}) == {"a": 2, "b": 3}


def test_the_margin_scales_the_orders_of_the_period():
    x_prev = _state({})
    demand = {"a": 3 / DELTA, "b": 1 / DELTA}
    # a: 3 orders, ceil(1.5 * 3) = 5. b: ceil(1.5 * 1) = 2, below the floor of 4.
    assert arrival_amounts(x_prev, demand, DELTA, 0.5, 4, {}) == {"a": 5, "b": 4}


def test_every_waiting_order_adds_a_unit_and_what_is_on_hand_counts():
    x_prev = _state({"a": {1: 2, 5: 1}, "b": {0: 4}})
    demand = {"a": 4 / DELTA}
    # a: up to 2 * 4 = 8, plus 2 waiting, less 3 on hand. b: up to 1, with 4 on hand gets none.
    assert arrival_amounts(x_prev, demand, DELTA, 1.0, 1, {"a": 2}) == {"a": 7, "b": 0}


def test_the_orders_are_the_exact_count_not_the_float_product():
    # 7 / 100 * 100 is 7.000000000000001 in floating point, whose ceiling would be 8.
    assert (7 / DELTA) * DELTA > 7
    x_prev = _state({})
    assert arrival_amounts(x_prev, {"a": 7 / DELTA}, DELTA, 0.0, 1, {})["a"] == 7


# The placement order of each rule, on the corridor.


def test_fixed_fills_the_home_shelf_first_and_overflows_to_the_nearest_shelf():
    x_prev = _state({"a": {4: 1}, "b": {0: 1}})
    result = _put_away("fixed", x_prev, {"a": 5, "b": 2}, home={"a": 4, "b": 0})
    # a: 3 more fit at home (4), the other 2 go to the nearest shelf, 1. b: both at home (0).
    assert result.storage.counts == {"a": {4: 4, 1: 2}, "b": {0: 3}}
    assert result.overflow_units == 2
    assert result.unplaced_units == 0
    assert result.arrived_units == 7


def test_fixed_places_the_skus_in_id_order():
    # Both SKUs live at 4 with room for two units. a comes first by id, whatever the demand.
    x_prev = _state({"a": {4: 1}, "b": {4: 1}})
    result = _put_away("fixed", x_prev, {"a": 2, "b": 2}, home={"a": 4, "b": 4}, demand={"b": 1.0})
    assert result.storage.counts == {"a": {4: 3}, "b": {4: 1, 1: 2}}
    assert result.overflow_units == 2


def test_demand_puts_the_most_demanded_sku_on_the_nearest_shelves():
    x_prev = _state({"a": {4: 1}, "b": {0: 1}})
    result = _put_away("demand", x_prev, {"a": 5, "b": 2}, demand={"a": 0.01, "b": 0.02})
    # b first: both at 1. Then a: 2 more at 1, then 3 at 5. Stock on 4 and 0 stays.
    assert result.storage.counts == {"a": {4: 1, 1: 2, 5: 3}, "b": {0: 1, 1: 2}}
    assert result.overflow_units == 0

    flipped = _put_away("demand", x_prev, {"a": 5, "b": 2}, demand={"a": 0.02, "b": 0.01})
    assert flipped.storage.counts == {"a": {4: 1, 1: 4, 5: 1}, "b": {0: 1, 5: 2}}


def test_demand_breaks_ties_in_demand_by_sku_id():
    result = _put_away("demand", _state({}), {"a": 4, "b": 4}, demand={"a": 0.01, "b": 0.01})
    assert result.storage.counts == {"a": {1: 4}, "b": {5: 4}}


def test_congestion_ranks_the_shelves_by_the_score():
    # Traffic and waits on the last edge into 1, the nearest shelf, push a off it to 5.
    traversal, waiting = {(15, 1): 1.0}, {(15, 1): 10.0}
    demand = {"a": 0.1}
    result = _put_away(
        "congestion", _state({}), {"a": 2}, demand=demand, traversal=traversal, waiting=waiting
    )
    assert result.storage.counts == {"a": {5: 2}}
    graph = _line()
    order = ranked_storage_vertices_for(
        graph, "a", [0, 1, 4, 5], [2, 6], demand, traversal, waiting, 1.0
    )
    assert order[0] == 5 and order[-1] == 1
    # Without the congestion term it is the demand rule's choice.
    plain = _put_away("congestion", _state({}), {"a": 2}, demand=demand, beta=0.0)
    assert plain.storage.counts == {"a": {1: 2}}


def test_congestion_without_beta_raises():
    with pytest.raises(ValueError, match="congestion_weight"):
        _put_away("congestion", _state({}), {"a": 1}, beta=None)


# Capacity, unplaced units and stock that stays.


@pytest.mark.parametrize("rule", RULES)
def test_capacity_holds_and_what_fits_nowhere_is_counted_unplaced(rule):
    x_prev = _state({"a": {1: 1}}, capacity=3.0, unit_capacity=1.5)
    result = _put_away(rule, x_prev, {"a": 10}, home={"a": 1, "b": 1}, demand={"a": 0.1})
    # Two units of 1.5 fit on each shelf, one is already on 1, so 7 fit and 3 do not.
    assert sum(result.storage.counts["a"].values()) == 8
    assert result.unplaced_units == 3
    assert result.arrived_units == 10
    for v, cap in result.storage.capacities.items():
        assert result.storage.used_capacity(v) <= cap


@pytest.mark.parametrize("rule", RULES)
def test_nothing_already_on_a_shelf_moves(rule):
    x_prev = _state({"a": {4: 2, 0: 1}, "b": {1: 3, 5: 1}})
    result = _put_away(
        rule, x_prev, {"a": 3, "b": 2}, home={"a": 4, "b": 1}, demand={"a": 0.01, "b": 0.02}
    )
    for sku_id, per_vertex in x_prev.counts.items():
        for v, count in per_vertex.items():
            assert result.storage.units(sku_id, v) >= count
    added = sum(
        result.storage.units(k, v) - x_prev.units(k, v) for k in ("a", "b") for v in (0, 1, 4, 5)
    )
    assert added == 5


def test_the_shelves_are_filled_by_distance_from_the_nearest_delivery_vertex():
    graph = _line()
    for v, expected in ACCESS.items():
        assert min(graph.distance(g, v) for g in (2, 6)) == expected
    result = _put_away("demand", _state({}, capacity=1.0), {"a": 4}, demand={"a": 0.1})
    assert result.storage.counts == {"a": {1: 1, 5: 1, 0: 1, 4: 1}}


# The home shelf and the split SKUs.


def test_the_home_shelf_holds_most_units_ties_by_vertex_id():
    counts = {"a": {5: 2, 1: 2, 0: 1}, "b": {4: 3, 0: 0}, "c": {1: 0}}
    assert _home_shelves(counts) == {"a": 1, "b": 4}


def test_a_split_sku_is_held_at_more_than_one_vertex():
    state = _state({"a": {1: 2, 5: 1}, "b": {0: 3, 4: 0}}, skus=("a", "b", "c"))
    assert _split_skus(state) == 1


# The registry.


def test_the_registry_holds_the_three_rules_under_the_storage_mode_names():
    assert registered_putaway_rules() == ("congestion", "demand", "fixed")
    assert get_putaway_rule("fixed") is arrival.putaway_fixed
    with pytest.raises(KeyError, match="nowhere"):
        get_putaway_rule("nowhere")
    with pytest.raises(ValueError, match="already registered"):
        register_putaway_rule("demand")(arrival.putaway_demand)


# Whole episodes on the medium instance, Token Passing.


def _spec(initial_units=8):
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    data = spec.model_dump()
    data["storage"].update(initial_placement="demand_blocks", initial_units=initial_units)
    data["storage_rule"] = StorageRuleSettings(
        epoch_length=DELTA, congestion_weight=1.0
    ).model_dump()
    data["arrival"] = ArrivalSettings(margin=1.0, receiving="delivery").model_dump()
    return type(spec).model_validate(data)


def _env(spec, mode, seed=200_000, rate=0.6):
    study = build_study(spec)
    config = build_experiment_config(spec, seed, mode)
    config = ExperimentConfig.model_validate(
        {
            **config.model_dump(),
            "controller": None,
            "planner": "token_passing",
            "arrival_rate": rate,
        }
    )
    return _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )


def _run(spec, mode, steps, **kwargs):
    env, step = _env(spec, mode, **kwargs)
    for _ in range(steps):
        step()
    return env


def _units_by_sku(storage):
    return {k: sum(per_vertex.values()) for k, per_vertex in storage.counts.items()}


def test_the_amounts_and_stock_per_sku_are_the_same_under_every_rule():
    runs = [_run(_spec(), mode, 2 * DELTA + 10) for mode in RULES]
    assert len(runs[0].log.arrived_units_by_update) == 2
    for env in runs[1:]:
        assert env.log.arrived_units_by_update == runs[0].log.arrived_units_by_update
        assert _units_by_sku(env.storage) == _units_by_sku(runs[0].storage)
    assert runs[0].log.relocated_units_by_update == []
    assert runs[0].log.relocation_distance_by_update == []


def test_an_order_with_no_stock_is_served_right_after_the_next_arrival():
    env, step = _env(_spec(initial_units=1), "demand")
    for _ in range(DELTA - 1):
        step()
    waiting = list(env._waiting_orders)
    assert waiting  # one unit per SKU runs out at load 0.6
    step()  # the arrival at t = Delta, then stage 1 of t = Delta
    assert env.t == DELTA
    assert env.log.unplaced_units_by_update == [0]
    served = {(t.release_time, t.sku, t.delivery_vertex): t for t in env.tasks}
    for order in waiting:
        task = served[(order.release_time, order.sku, order.delivery_vertex)]
        assert env.log.stock_wait_by_task[task.task_id] == DELTA - order.release_time
    assert all(order.release_time == DELTA for order in env._waiting_orders)


@pytest.mark.parametrize("mode", RULES)
def test_each_arrival_moves_nothing_and_logs_what_it_placed(mode):
    env, step = _env(_spec(), mode)
    calls = []
    rule = env._putaway_rule

    def recording(x_prev, *args):
        result = rule(x_prev, *args)
        calls.append((x_prev, result))
        return result

    env._putaway_rule = recording
    for _ in range(2 * DELTA):
        step()
    assert len(calls) == 2
    for x_prev, result in calls:
        for sku_id, per_vertex in x_prev.counts.items():
            for v, count in per_vertex.items():
                assert result.storage.units(sku_id, v) >= count
        added = sum(_units_by_sku(result.storage).values()) - sum(_units_by_sku(x_prev).values())
        assert added == result.arrived_units - result.unplaced_units
    log = env.log
    assert log.arrived_units_by_update == [r.arrived_units for _, r in calls]
    assert log.unplaced_units_by_update == [r.unplaced_units for _, r in calls]
    assert log.overflow_units_by_update == [r.overflow_units for _, r in calls]
    assert log.split_skus_by_update == [
        sum(1 for per in r.storage.counts.values() if sum(c > 0 for c in per.values()) > 1)
        for _, r in calls
    ]
    if mode != "fixed":
        assert log.overflow_units_by_update == [0, 0]


def test_a_full_arrival_episode_replays_bit_for_bit():
    first, second = (_run(_spec(), "congestion", 3 * DELTA + 50) for _ in range(2))
    assert len(first.log.arrived_units_by_update) == 3
    assert first.tasks == second.tasks
    assert first.storage == second.storage
    assert first._waiting_orders == second._waiting_orders
    assert first.log.backlog_by_t == second.log.backlog_by_t
    assert first.log.stock_wait_by_task == second.log.stock_wait_by_task
    assert first.log.split_skus_by_update == second.log.split_skus_by_update


def test_fixed_storage_needs_a_home_shelf_for_every_sku():
    spec = _spec()
    study = build_study(spec)
    counts = {k: dict(v) for k, v in study.counts.items()}
    counts["sku_000"] = {}
    config = build_experiment_config(spec, 200_000, "fixed")
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing"}
    )
    with pytest.raises(ValueError, match="no home shelf"):
        _planner_run(
            study.instance.graph, study.skus, counts, study.capacities, config, "token_passing"
        )
