"""The order distribution with skewed and rotating SKU popularity (sec:pf:tasks).

Without a skew the order stream is the uniform draw it always was. With one, the stream draws
the same random numbers and only maps the SKU number differently, so the orders of a seed stay
the same under every storage rule (sec:impl:instances).
"""

import bisect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import (
    _ORDER_SEED_OFFSET,
    WarehouseMAPDEnv,
    popularity_cumulative,
    popularity_ranking,
    zipf_weights,
)
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.training.spec import (
    DemandSettings,
    build_experiment_config,
    build_storage,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
_NUM_SKUS = 12
_SKUS = {
    f"sku_{i:03d}": SkuType(sku_id=f"sku_{i:03d}", unit_capacity=1.0) for i in range(_NUM_SKUS)
}
_HORIZON = 300

Order = tuple[int, SkuId, VertexId, VertexId]  # (release time, SKU, pickup, delivery)


def _graph() -> WarehouseGraph:
    return generate_warehouse_graph(
        GeneratorParams(
            max_num_aisles=6,
            max_aisle_length=6,
            max_num_cross_aisles=5,
            one_way_fraction=0.0,
            wait_cost=0.0,
            num_storage_vertices=6,
            num_delivery_vertices=3,
            num_endpoints=4,
            seed=0,
        )
    )


def _storage_vertices(graph: WarehouseGraph) -> list[VertexId]:
    return sorted(v for v, vertex in graph.vertices.items() if vertex.role.storage)


def _delivery_vertices(graph: WarehouseGraph) -> list[VertexId]:
    return sorted(v for v, vertex in graph.vertices.items() if vertex.role.delivery)


def _counts(graph: WarehouseGraph, units: int = 4) -> dict[SkuId, dict[VertexId, int]]:
    """SKU number i at storage vertex number i mod |V_str|, as build_storage places them."""
    storage = _storage_vertices(graph)
    return {sku_id: {storage[i % len(storage)]: units} for i, sku_id in enumerate(_SKUS)}


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.5,
        seed=11,
        horizon=_HORIZON,
        wait_cost=0.0,
        observation_depth=2,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _orders(
    graph: WarehouseGraph, counts: dict[SkuId, dict[VertexId, int]], **config
) -> tuple[WarehouseMAPDEnv, list[Order]]:
    """Plays one episode with every robot waiting, and returns the env and its orders."""
    built = _config(**config)
    capacities = {v: 40.0 for v in _storage_vertices(graph)}
    env = WarehouseMAPDEnv(graph, _SKUS, counts, capacities, built)
    for _ in range(built.horizon):
        env.step({agent_id: 0 for agent_id in env.fleet.agents})
    return env, [(t.release_time, t.sku, t.pickup_vertex, t.delivery_vertex) for t in env.tasks]


def _vertices_by_sku(counts: dict[SkuId, dict[VertexId, int]]) -> dict[SkuId, list[VertexId]]:
    return {
        sku_id: sorted(v for v, count in per_vertex.items() if count > 0)
        for sku_id, per_vertex in sorted(counts.items())
    }


def _reference_orders(
    graph: WarehouseGraph,
    counts: dict[SkuId, dict[VertexId, int]],
    seed: int,
    horizon: int,
    arrival_rate: float,
    skew: float | None = None,
    period: int = 0,
    shift: int = 0,
) -> list[Order]:
    """The orders of a seed under fixed storage, drawn without the env.

    With no skew this is the uniform draw of `main` before the demand model. With a skew it is
    the probe's draw (patch_order_draw of the demand probe), period 0 meaning no rotation. The
    stock never moves under fixed storage, so the mapping can be replayed step by step.
    """
    rng = np.random.default_rng(seed + _ORDER_SEED_OFFSET)
    vertices_by_sku = _vertices_by_sku(counts)
    delivery = _delivery_vertices(graph)
    by_number = sorted(vertices_by_sku, key=lambda name: int(name.rsplit("_", 1)[1]))
    orders: list[Order] = []
    for t in range(horizon):
        num_new = int(rng.poisson(arrival_rate))
        if num_new == 0:
            continue
        draws = rng.random((num_new, 3))
        if skew is None:
            in_stock = [sku_id for sku_id, vertices in vertices_by_sku.items() if vertices]
        else:
            offset = ((t // period) * shift) % len(by_number) if period > 0 else 0
            ranked = by_number[offset:] + by_number[:offset]
            raw = [1.0 / (rank + 1) ** skew for rank in range(len(ranked))]
            total_raw = sum(raw)
            weights = [w / total_raw for w in raw]
            stocked = [(k, w) for k, w in zip(ranked, weights) if vertices_by_sku[k]]
            total = sum(w for _, w in stocked)
            cumulative, running = [], 0.0
            for _, w in stocked:
                running += w / total
                cumulative.append(running)
            in_stock = [k for k, _ in stocked]
        for u_sku, u_pickup, u_delivery in draws:
            if skew is None:
                sku_id = in_stock[int(u_sku * len(in_stock))]
            else:
                sku_id = in_stock[min(bisect.bisect_right(cumulative, u_sku), len(in_stock) - 1)]
            pickups = vertices_by_sku[sku_id]
            orders.append(
                (
                    t,
                    sku_id,
                    pickups[int(u_pickup * len(pickups))],
                    delivery[int(u_delivery * len(delivery))],
                )
            )
    return orders


def test_without_a_skew_the_orders_of_a_seed_are_the_uniform_draw_of_main():
    graph = _graph()
    counts = _counts(graph)
    _, orders = _orders(graph, counts)
    assert len(orders) > 300
    assert orders == _reference_orders(graph, counts, seed=11, horizon=_HORIZON, arrival_rate=1.5)


def test_with_a_skew_and_rotation_the_orders_match_the_probes_draw():
    graph = _graph()
    counts = _counts(graph)
    _, orders = _orders(
        graph, counts, demand_skew=1.0, demand_rotation_period=40, demand_rotation_shift=5
    )
    reference = _reference_orders(
        graph, counts, seed=11, horizon=_HORIZON, arrival_rate=1.5, skew=1.0, period=40, shift=5
    )
    assert len(orders) > 300
    assert orders == reference


def test_with_a_skew_and_rotation_the_orders_are_the_same_under_fixed_and_demand_storage():
    graph = _graph()
    counts = _counts(graph)
    demand = dict(demand_skew=1.0, demand_rotation_period=40, demand_rotation_shift=5)
    _, fixed = _orders(graph, counts, **demand)
    demand_env, adaptive = _orders(
        graph,
        counts,
        storage_mode="demand",
        storage_epoch_length=30,
        reassignment_cap=12,
        **demand,
    )

    assert sum(demand_env.log.relocated_units_by_update) > 0  # the stock did move
    assert [(t, sku, d) for t, sku, _, d in fixed] == [(t, sku, d) for t, sku, _, d in adaptive]
    assert [p for _, _, p, _ in fixed] != [p for _, _, p, _ in adaptive]  # pickups follow it


def test_the_most_popular_sku_moves_by_the_shift_exactly_at_multiples_of_the_period():
    skus = list(_SKUS)
    period, shift = 7, 5
    top = [popularity_ranking(skus, t, period, shift)[0] for t in range(10 * period)]
    for t in range(1, len(top)):
        if t % period == 0:
            assert skus.index(top[t]) == (skus.index(top[t - 1]) + shift) % len(skus)
        else:
            assert top[t] == top[t - 1]
    assert top[0] == skus[0]


def test_the_ranking_is_a_left_rotation_of_the_list_order_and_does_not_rotate_without_a_period():
    skus = ["tea", "coffee", "mugs", "biscuits", "sugar"]  # not sorted, the list order counts
    assert popularity_ranking(skus, 3, 2, 3) == skus[3:] + skus[:3]
    assert popularity_ranking(skus, 10**6, None, None) == skus


def test_the_most_popular_sku_of_the_orders_is_the_first_in_the_list_not_the_first_sorted():
    graph = _graph()
    storage = _storage_vertices(graph)
    names = ["tea", "coffee", "mugs", "biscuits", "sugar", "milk"]
    skus = {name: SkuType(sku_id=name, unit_capacity=1.0) for name in names}
    counts = {name: {storage[i % len(storage)]: 4} for i, name in enumerate(names)}
    config = _config(arrival_rate=4.0, demand_skew=4.0, horizon=50)
    env = WarehouseMAPDEnv(graph, skus, counts, {v: 40.0 for v in storage}, config)
    for _ in range(config.horizon):
        env.step({agent_id: 0 for agent_id in env.fleet.agents})
    ordered = Counter(task.sku for task in env.tasks)
    assert ordered.most_common(1)[0][0] == "tea"


def test_the_most_popular_sku_of_the_orders_changes_with_the_ranking():
    graph = _graph()
    counts = _counts(graph)
    period, shift = 50, 5
    _, orders = _orders(
        graph,
        counts,
        arrival_rate=4.0,
        demand_skew=4.0,
        demand_rotation_period=period,
        demand_rotation_shift=shift,
    )
    skus = list(_SKUS)
    for start in range(0, _HORIZON, period):
        window = Counter(sku for t, sku, _, _ in orders if start <= t < start + period)
        expected = skus[((start // period) * shift) % len(skus)]
        assert window.most_common(1)[0][0] == expected


def test_over_a_long_run_the_sku_shares_follow_the_zipf_weights():
    graph = _graph()
    counts = _counts(graph)
    _, orders = _orders(graph, counts, arrival_rate=10.0, demand_skew=1.0)
    shares = Counter(sku for _, sku, _, _ in orders)
    weights = zipf_weights(_NUM_SKUS, 1.0)
    assert len(orders) > 2500
    for rank, sku_id in enumerate(_SKUS):
        assert shares[sku_id] / len(orders) == pytest.approx(weights[rank], abs=0.02)


def test_a_sku_out_of_stock_keeps_its_rank_but_is_never_requested():
    graph = _graph()
    counts = _counts(graph)
    counts["sku_000"] = {}
    _, orders = _orders(graph, counts, arrival_rate=4.0, demand_skew=1.0)
    assert orders
    assert "sku_000" not in {sku for _, sku, _, _ in orders}

    ranked = popularity_ranking(list(_SKUS), 0, None, None)
    in_stock, cumulative = popularity_cumulative(ranked, 1.0, _vertices_by_sku(counts))
    assert in_stock == list(_SKUS)[1:]
    weights = zipf_weights(_NUM_SKUS, 1.0)[1:]
    assert cumulative[0] == pytest.approx(weights[0] / sum(weights))
    assert cumulative[-1] == pytest.approx(1.0)


def test_a_skew_of_zero_gives_every_rank_the_same_weight():
    assert zipf_weights(4, 0.0) == [0.25] * 4


def test_a_skew_alone_is_a_valid_config():
    assert _config(demand_skew=1.0).demand_rotation_period is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (dict(demand_rotation_period=10, demand_rotation_shift=2), "requires demand_skew"),
        (dict(demand_skew=1.0, demand_rotation_period=10), "requires demand_rotation_shift"),
        (dict(demand_skew=1.0, demand_rotation_shift=2), "requires demand_rotation_period"),
        (dict(demand_skew=-0.5), "greater than or equal to 0"),
        (dict(demand_skew=1.0, demand_rotation_period=0, demand_rotation_shift=1), "greater"),
    ],
)
def test_the_config_rejects_an_incomplete_or_invalid_demand_model(fields, message):
    with pytest.raises(ValidationError, match=message):
        _config(**fields)


@pytest.mark.parametrize(
    "fields",
    [dict(skew=1.0, rotation_period=10), dict(skew=1.0, rotation_shift=2), dict()],
)
def test_the_spec_rejects_an_incomplete_demand_section(fields):
    with pytest.raises(ValidationError):
        DemandSettings.model_validate(fields)


@pytest.fixture(scope="module")
def spec():
    return load_spec(ROOT / "configs" / "decentralised.yaml")


@pytest.mark.parametrize("name", ["decentralised", "centralised", "section", "full_view"])
def test_the_shipped_configs_keep_the_uniform_draw(name):
    config = build_experiment_config(load_spec(ROOT / "configs" / f"{name}.yaml"))
    assert config.demand_skew is None
    assert config.demand_rotation_period is None
    assert config.demand_rotation_shift is None


def test_the_demand_section_reaches_the_experiment_config(spec):
    demand = DemandSettings(skew=1.0, rotation_period=200, rotation_shift=5)
    config = build_experiment_config(spec.model_copy(update={"demand": demand}))
    assert (config.demand_skew, config.demand_rotation_period, config.demand_rotation_shift) == (
        1.0,
        200,
        5,
    )


def test_sku_names_keep_the_numeric_order_past_one_hundred(spec):
    many = spec.storage.model_copy(update={"num_skus": 120, "vertex_capacity": 10_000.0})
    skus, _, _ = build_storage(many, build_study(spec).instance.graph)
    assert list(skus) == sorted(skus)
    assert list(skus)[:2] == ["sku_000", "sku_001"]
    assert list(skus)[100] == "sku_100"
