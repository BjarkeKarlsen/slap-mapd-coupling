"""The refill rule of the arrival stock model (steps 04 and 05 of the model revision). Stock is
replaced a refill of Q units at a time, once Q units of a SKU have been ordered since its last
refill. Each refill is one put-away to one shelf, and the starting stock S_k is Q plus the
service-level quantile of a period's orders."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.storage.arrival import (
    pick_distance,
    putaway_demand,
    receiving_distance,
    refill_amounts,
)
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
    period_demand_quantiles,
    refill_starting_stock,
    stock_sizing,
)

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"
SEED = 300_000
Q = 30
# What a 300-step Token Passing episode of FLEET_8 under the top-up rule (margin 1, floor 8,
# shelf capacity 300, lines of 1 to 5 units) gave before the refill rule existed: a digest of its
# tasks, of its final stock, and its number of tasks.
TOPUP = {
    "fixed": ("fb4cf1b59a01faa6", "5286ef6247b1b929", 66),
    "demand": ("da7c73a1dcca49da", "3a04d422e715fec0", 65),
    "random": ("0b5f1a9a624fe2a2", "7e762a69fc7e3298", 66),
}


def _spec(**arrival) -> TrainingSpec:
    spec = load_spec(FLEET_8)
    data = spec.model_dump()
    data["arrival"].update(arrival)
    return TrainingSpec.model_validate(data)


def _topup_spec() -> TrainingSpec:
    data = load_spec(FLEET_8).model_dump()
    data["arrival"].update(rule="topup", margin=1.0, refill_units=None, service_level=None)
    data["storage"].update(vertex_capacity=300.0, initial_units=8, num_skus=100, start_fill=None)
    return TrainingSpec.model_validate(data)


_STUDIES: dict = {}


def _run(mode="fixed", horizon=400, spec=None, seed=SEED, every_step=None):
    """A Token Passing episode of `spec` (FLEET_8 by default). `every_step(env)` is called after
    each step."""
    spec = spec or _spec()
    key = spec.model_dump_json()
    if key not in _STUDIES:
        _STUDIES[key] = build_study(spec)
    study = _STUDIES[key]
    config = build_experiment_config(spec, seed, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing", "horizon": horizon}
    )
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(horizon):
        step()
        if every_step is not None:
            every_step(env)
    return env, study


def _digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def _task_rows(env):
    return [
        [t.task_id, t.release_time, t.pickup_vertex, t.delivery_vertex, t.sku, t.quantity,
         t.kind, t.finish_time]
        for t in env.tasks
    ]


def _deliveries(env):
    """(arrival step, SKU, units) of every put-away, whatever shelf it went to."""
    return sorted((t.release_time, t.sku, t.quantity) for t in env.tasks if t.kind == "putaway")


def test_a_sku_gets_one_refill_per_q_ordered_and_the_rest_carries_over():
    amounts, left = refill_amounts({"a": 29, "b": 30, "c": 95, "d": 0}, Q)
    assert amounts == {"a": 0, "b": 30, "c": 90, "d": 0}
    assert left == {"a": 29, "b": 0, "c": 5, "d": 0}


def _released_now(env) -> Counter:
    """The units of each SKU in the lines released at the current t, which stage 1 of the step
    that ends at t has already counted."""
    units = Counter()
    for t in env.tasks:
        if t.kind == "pick" and t.release_time == env.t:
            units[t.sku] += t.quantity
    for line in env.waiting_orders:
        if line.release_time == env.t:
            units[line.sku] += line.missing
    return units


def test_every_refill_is_one_put_away_of_q_units_after_q_were_ordered():
    def watch(env):
        # Right after an arrival, what a SKU counts towards its next refill is below Q.
        if env.t % 100 == 0:
            now = _released_now(env)
            for sku_id, count in env._ordered_since_refill.items():
                assert 0 <= count - now[sku_id] < Q

    env, _ = _run(every_step=watch)
    putaways = [t for t in env.tasks if t.kind == "putaway"]
    assert putaways, "a 400-step run at 8 agents has refills"
    assert sum(env.log.unplaced_units_by_update) == 0
    assert all(t.quantity == Q for t in putaways)
    assert len(putaways) == sum(env.log.refills_by_update)
    # Over the run, every unit ordered went into a refill or still counts towards the next.
    ordered, refilled = Counter(), Counter()
    for t in env.tasks:
        (ordered if t.kind == "pick" else refilled)[t.sku] += t.quantity
    for line in env.waiting_orders:
        ordered[line.sku] += line.missing
    for sku_id in env._sku_ids:
        assert ordered[sku_id] == refilled[sku_id] + env._ordered_since_refill[sku_id]


def test_a_refill_goes_whole_to_the_first_shelf_with_room_or_is_unplaced():
    vertices = {
        0: Vertex(id=0, role=VertexRole(movable=True, delivery=True, receiving=True)),
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, storage=True)),
        3: Vertex(id=3, role=VertexRole(movable=True)),
        4: Vertex(id=4, role=VertexRole(movable=True)),
    }
    # A corridor 0 - 3 - 4, shelf 1 off 3 and shelf 2 off 4, so shelf 1 is the nearer.
    pairs = ((0, 3), (3, 4), (3, 1), (4, 2))
    edges = tuple(Edge(source=a, target=b, cost=1.0) for a, b in pairs) + tuple(
        Edge(source=b, target=a, cost=1.0) for a, b in pairs
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    skus = {"a": SkuType(sku_id="a", unit_capacity=1.0)}
    # Shelf 1 is nearer but has room for 3 units only, shelf 2 for 5.
    x_prev = StorageState(skus=skus, capacities={1: 3.0, 2: 5.0}, counts={})
    result = putaway_demand(
        x_prev, graph, {"a": 8}, {}, {"a": 1.0}, {}, {}, None,
        trip=True, refill_units=4, unit_estimate={"a": 1.0},
    )
    # The first refill of 4 fits only on shelf 2, the second fits nowhere: never split.
    assert result.storage.counts["a"] == {2: 4}
    assert result.unplaced_units == 4
    assert all(
        result.storage.used_capacity(v) <= cap for v, cap in result.storage.capacities.items()
    )


def test_no_shelf_ever_exceeds_its_capacity():
    def watch(env):
        stock = env._shelf_and_transit()
        for v, cap in stock.capacities.items():
            assert stock.used_capacity(v) <= cap + 1e-9

    for mode in ("fixed", "demand", "random"):
        _run(mode, horizon=300, every_step=watch)


def test_the_deliveries_are_the_same_under_every_storage_rule():
    runs = {mode: _run(mode, horizon=600)[0] for mode in ("fixed", "demand", "congestion", "random")}
    refills = {mode: env.log.refills_by_update for mode, env in runs.items()}
    assert len(set(map(tuple, refills.values()))) == 1
    assert sum(refills["fixed"]) > 0
    # The SKUs refilled at each arrival too, while the shelves they go to differ.
    arrived = {mode: [(r, k) for r, k, _ in _deliveries(env)] for mode, env in runs.items()}
    assert arrived["demand"] == arrived["fixed"] == arrived["random"] == arrived["congestion"]


def test_inventory_position_stays_within_s_minus_q_and_s_after_each_arrival():
    """The inventory position, stock on the shelves and on its way less the units promised and
    waited for, is S_k less what counts towards the next refill. Right after an arrival, before
    the lines of that step, it is in (S_k - Q, S_k]."""
    start = {}

    def watch(env):
        if not start:
            start.update({k: sum(v.values()) for k, v in env._initial_counts.items()})
        stock = env._shelf_and_transit()
        now = _released_now(env) if env.t % 100 == 0 else None
        for sku_id in env._sku_ids:
            on_hand = sum(stock.counts.get(sku_id, {}).values())
            reserved = sum(env._reserved.get(sku_id, {}).values())
            waiting = sum(line.missing for line in env.waiting_orders if line.sku == sku_id)
            position = on_hand - reserved - waiting
            s_k = start[sku_id]
            assert position == s_k - env._ordered_since_refill[sku_id]
            if now is not None:
                assert s_k - Q < position + now[sku_id] <= s_k, (env.t, sku_id)

    _run("demand", horizon=800, every_step=watch)


def test_the_starting_stock_is_q_plus_the_simulated_quantile():
    spec = _spec()
    study = build_study(spec)
    sizing = stock_sizing(spec)
    expected = refill_starting_stock(list(study.skus), spec.demand, sizing)
    assert {k: sum(v.values()) for k, v in study.counts.items()} == expected
    quantiles = period_demand_quantiles(
        len(study.skus), 1.0, spec.arrival_rate, 100, 0.99, None, (1, 5)
    )
    ranked = list(study.skus)  # no rotation at t = 0, so the list order is the ranking
    assert [expected[k] - Q for k in ranked] == quantiles
    assert quantiles[0] > quantiles[-1]


def test_the_simulated_quantile_matches_the_poisson_quantile():
    # One SKU, one unit per line: a period's units are Poisson(rate * Delta).
    mean = 0.2 * 50
    quantile = period_demand_quantiles(1, 0.0, 0.2, 50, 0.99, periods=200_000)[0]
    cdf, k, term = 0.0, 0, math.exp(-mean)
    while cdf + term < 0.99:
        cdf += term
        k += 1
        term *= mean / k
    assert quantile == k


def test_the_number_of_skus_fills_the_start_fill_of_every_fleet():
    for fleet in (8, 16):
        spec = load_spec(ROOT / "configs" / f"fleet_{fleet}" / "decentralised.yaml")
        study = build_study(spec)
        room = sum(
            units * study.skus[k].unit_capacity
            for k, per_vertex in study.counts.items()
            for units in per_vertex.values()
        )
        fill = room / sum(study.capacities.values())
        assert spec.storage.start_fill <= fill < spec.storage.start_fill + 0.02, (fleet, fill)


def test_sku_names_name_the_first_skus_of_a_derived_count():
    storage = _spec().storage
    ids = storage.sku_ids(3)
    assert ids == list(storage.sku_names[:3])
    assert storage.model_copy(update={"sku_names": ("tea",)}).sku_ids(3) == [
        "tea", "sku_001", "sku_002"
    ]


def test_the_derived_capacity_holds_the_starting_stock_of_the_fullest_shelf():
    study = build_study(_spec())
    used = Counter()
    for sku_id, per_vertex in study.counts.items():
        for v, units in per_vertex.items():
            used[v] += units * study.skus[sku_id].unit_capacity
    capacity = next(iter(study.capacities.values()))
    assert set(study.capacities.values()) == {capacity}
    assert capacity == math.ceil(max(used.values()))


def test_fixed_storage_does_not_overflow_at_the_derived_capacity():
    env, _ = _run("fixed", horizon=1000)
    assert sum(env.log.unplaced_units_by_update) == 0
    assert sum(env.log.overflow_units_by_update) == 0


def test_the_put_away_leg_weighs_by_refills_per_step():
    study = build_study(_spec())
    graph = study.instance.graph
    storage = sorted(v for v, x in graph.vertices.items() if x.role.storage)
    delivery = sorted(v for v, x in graph.vertices.items() if x.role.delivery)
    receiving = sorted(v for v, x in graph.vertices.items() if x.role.receiving)
    pick = {v: pick_distance(graph, v, delivery) for v in storage}
    put = {v: receiving_distance(graph, v, receiving) for v in storage}
    skus = {"a": SkuType(sku_id="a", unit_capacity=1.0)}

    # A refill needs room for Q units, so the shelves are made large enough.
    def first_shelf(refill_units, units_per_step):
        big = StorageState(skus=skus, capacities=dict.fromkeys(storage, 1e6), counts={})
        result = putaway_demand(
            big, graph, {"a": refill_units}, {}, {"a": 0.1}, {}, {}, None,
            receiving=receiving, trip=True, refill_units=refill_units,
            unit_estimate={"a": units_per_step},
        )
        return next(iter(result.storage.counts["a"]))

    # A large Q: the put-away leg hardly matters, the nearest shelf to the workstations wins.
    assert pick[first_shelf(10_000, 0.1)] == min(pick.values())
    # Q = 1 with one unit per line: both legs weigh the same, the shortest whole trip wins.
    best = first_shelf(1, 0.1)
    assert pick[best] + put[best] == pytest.approx(min(pick[v] + put[v] for v in storage))


@pytest.mark.parametrize("mode", sorted(TOPUP))
def test_the_topup_rule_gives_the_runs_it_gave_before(mode):
    env, _ = _run(mode, horizon=300, spec=_topup_spec())
    stock = sorted((k, v, c) for k, per in env.storage.counts.items() for v, c in per.items())
    assert (_digest(_task_rows(env)), _digest(stock), len(env.tasks)) == TOPUP[mode]


def test_a_refill_run_replays_bit_for_bit():
    first, _ = _run("random", horizon=300)
    second, _ = _run("random", horizon=300)
    assert _task_rows(first) == _task_rows(second)
    assert first.storage.counts == second.storage.counts


def test_the_rule_settings_are_checked():
    with pytest.raises(ValueError, match="refill_units"):
        _spec(refill_units=None)
    with pytest.raises(ValueError, match="margin"):
        _spec(margin=1.0)
    data = load_spec(FLEET_8).model_dump()
    data["storage"]["initial_units"] = 8
    with pytest.raises(ValueError, match="initial_units"):
        TrainingSpec.model_validate(data)
    config = build_experiment_config(_spec(), 0, "fixed")
    assert (config.stock_rule, config.refill_units, config.service_level) == ("refill", Q, 0.99)
    assert (config.stock_margin, config.stock_floor) == (None, None)
