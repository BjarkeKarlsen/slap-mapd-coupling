"""The order stream does not depend on the storage configuration (sec:impl:instances).

The generator of stage 1 is seeded apart from the policy and the environment so that
the same order stream replays under every controller and storage rule. What it
consumes must therefore not depend on where the stock sits.
"""

import numpy as np

import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph

_SKUS = {
    "tea": SkuType(sku_id="tea", unit_capacity=1.0),
    "mug": SkuType(sku_id="mug", unit_capacity=1.0),
}
_HORIZON = 60


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
    return [v for v, vertex in graph.vertices.items() if vertex.role.storage]


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


def _stream(
    counts: dict[SkuId, dict[VertexId, int]], graph: WarehouseGraph | None = None, **config
) -> tuple[WarehouseMAPDEnv, list[tuple[int, VertexId, SkuId]]]:
    """Runs one episode with every robot waiting and returns the env and the stream as
    (release time, delivery vertex, SKU) in release order."""
    graph = graph or _graph()
    capacities = {v: 20.0 for v in _storage_vertices(graph)}
    env = WarehouseMAPDEnv(graph, _SKUS, counts, capacities, _config(**config))
    for _ in range(_HORIZON):
        env.step({agent_id: 0 for agent_id in env.fleet.agents})
    stream = [(t.release_time, t.delivery_vertex, t.sku) for t in env.tasks]
    return env, stream


def test_the_stream_is_identical_under_two_hand_built_placements():
    graph = _graph()
    a, b, c, d, *_ = _storage_vertices(graph)
    one_vertex_each = {"tea": {a: 6}, "mug": {b: 6}}
    spread_out = {"tea": {c: 2, d: 2, a: 2}, "mug": {a: 1, b: 1, d: 4}}

    _, first = _stream(one_vertex_each, graph)
    _, second = _stream(spread_out, graph)

    assert len(first) > 0
    assert first == second  # release times, delivery vertices and SKUs, in order


def test_the_stream_is_identical_when_the_stock_moves_from_one_vertex_to_two():
    graph = _graph()
    a, b, *_ = _storage_vertices(graph)

    _, on_one = _stream({"tea": {a: 8}}, graph)
    _, on_two = _stream({"tea": {a: 4, b: 4}}, graph)

    assert len(on_one) > 0
    assert on_one == on_two


def test_the_stream_is_identical_under_fixed_and_demand_storage():
    graph = _graph()
    a, b, c, *_ = _storage_vertices(graph)
    counts = {"tea": {a: 4, b: 4}, "mug": {c: 3}}

    fixed_env, fixed = _stream(counts, graph)
    demand_env, demand = _stream(
        counts, graph, storage_mode="demand", storage_epoch_length=10, reassignment_cap=3
    )

    assert demand_env.log.relocated_units_by_update  # the rule did run at its epochs
    assert fixed == demand
    assert len(fixed_env.tasks) == len(demand_env.tasks)


def test_the_pickup_vertex_still_holds_the_requested_sku():
    graph = _graph()
    a, b, c, *_ = _storage_vertices(graph)
    counts = {"tea": {a: 4, b: 4}, "mug": {c: 3}}
    env, _ = _stream(counts, graph)  # fixed storage, so the stock never moves
    assert env.tasks
    for task in env.tasks:
        assert env.storage.units(task.sku, task.pickup_vertex) > 0


def test_a_sku_with_no_stock_is_never_requested():
    graph = _graph()
    a, b, *_ = _storage_vertices(graph)
    env, stream = _stream({"tea": {a: 3, b: 3}, "mug": {}}, graph)
    assert stream
    assert {sku for _, _, sku in stream} == {"tea"}


def test_the_demand_for_a_sku_does_not_depend_on_how_many_vertices_hold_it():
    graph = _graph()
    a, b, c, d, *_ = _storage_vertices(graph)
    _, concentrated = _stream({"tea": {a: 6}, "mug": {b: 6}}, graph)
    _, spread = _stream({"tea": {a: 1, b: 1, c: 1, d: 1}, "mug": {b: 6}}, graph)
    assert [sku for *_, sku in concentrated] == [sku for *_, sku in spread]


def test_replay_of_the_stream_is_deterministic_under_a_fixed_seed():
    graph = _graph()
    a, b, *_ = _storage_vertices(graph)
    counts = {"tea": {a: 4}, "mug": {b: 4}}
    _, first = _stream(counts, graph)
    _, second = _stream(counts, graph)
    assert first == second


def test_a_different_seed_gives_a_different_stream():
    graph = _graph()
    a, b, *_ = _storage_vertices(graph)
    counts = {"tea": {a: 4}, "mug": {b: 4}}
    _, first = _stream(counts, graph, seed=1)
    _, second = _stream(counts, graph, seed=2)
    assert first != second


def test_arrivals_are_poisson_with_the_arrival_rate_as_mean():
    graph = _graph()
    a, *_ = _storage_vertices(graph)
    rate = 2.0
    env, stream = _stream({"tea": {a: 4}}, graph, arrival_rate=rate, seed=3, horizon=_HORIZON)
    per_step = np.bincount([t for t, _, _ in stream], minlength=_HORIZON)
    assert abs(per_step.mean() - rate) < 0.5
    assert per_step.max() > 2  # more than one arrival in a step happens
