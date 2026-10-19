"""Robots carry the put-away (sec:pf:storage). Every (SKU, shelf) of an arrival is a put-away
task, its units are on their way until it is delivered, and the storage rules judge a shelf by
the trip distance (sec:method:storage)."""

import hashlib
import json
from pathlib import Path

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.environment.multi_agent_env import OrderLine
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.storage.arrival import (
    nearest_receiving_point,
    putaway_congestion,
    putaway_demand,
    trip_distance,
    trip_exposure,
)
from slap_mapd_coupling.storage.congestion import congestion_exposure, nearest_delivery_point
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[2]
PUTAWAY = ROOT / "configs" / "medium_putaway" / "decentralised.yaml"
ARRIVAL = ROOT / "configs" / "medium_arrival" / "decentralised.yaml"
SEED = 300_000
# What a 300-step Token Passing episode of configs/medium_arrival gave on feat/cold-stock, before
# robot put-away existed: a digest of its tasks and of its final stock, and its measures.
INSTANT = {
    "fixed": ("5561c3634c19d979", "b1be2db23db50a1d", 125, 37.144, 150.33333333333334),
    "demand": ("3d4688f7c1143c6a", "6e11f2df940c8a2d", 126, 33.29365079365079, 77.33333333333333),
    "congestion": (
        "3d4688f7c1143c6a",
        "6e11f2df940c8a2d",
        126,
        33.29365079365079,
        77.33333333333333,
    ),
}


def _env(path, mode="fixed", horizon=300):
    spec = load_spec(path)
    study = build_study(spec)
    config = build_experiment_config(spec, SEED, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing", "horizon": horizon}
    )
    return _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )


def _digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


_TASK_FIELDS = (
    "task_id",
    "release_time",
    "pickup_vertex",
    "delivery_vertex",
    "sku",
    "quantity",
    "assigned_agent",
    "assignment_time",
    "pickup_time",
    "finish_time",
)


def _task_rows(env):
    return [[getattr(task, name) for name in _TASK_FIELDS] for task in env.tasks]


@pytest.mark.parametrize("mode", sorted(INSTANT))
def test_instant_put_away_replays_the_episode_it_gave_before(mode):
    from slap_mapd_coupling.evaluation.evaluator import _to_run_metrics

    env, step = _env(ARRIVAL, mode)
    for _ in range(300):
        step()
    metrics = _to_run_metrics(env, env.config)
    stock = sorted((k, sorted(v.items())) for k, v in env.storage.counts.items())
    tasks, storage, completed, service, putaway = INSTANT[mode]
    assert _digest(_task_rows(env)) == tasks
    assert _digest(stock) == storage
    assert metrics.num_completed_tasks == completed
    assert metrics.mean_service_time == pytest.approx(service)
    assert metrics.mean_putaway_distance == pytest.approx(putaway)
    assert all(task.kind == "pick" for task in env.tasks)


def _run_to_the_first_arrival(mode="fixed"):
    """A robots episode up to and including the arrival at t = 100, with the stock before and
    after the arrival and the tasks it released."""
    env, step = _env(PUTAWAY, mode)
    seen = {}
    arrive = env._arrive

    def recording(delta):
        seen.setdefault("shelf_before", env.storage)
        seen.setdefault("transit_before", {k: dict(v) for k, v in env._in_transit.items()})
        arrive(delta)
        seen.setdefault("shelf_after", env.storage)
        seen.setdefault("transit_after", {k: dict(v) for k, v in env._in_transit.items()})

    env._arrive = recording
    for _ in range(100):
        step()
    return env, seen


@pytest.mark.parametrize("mode", ["fixed", "demand", "congestion"])
def test_an_arrival_leaves_the_shelves_and_makes_one_putaway_per_sku_and_shelf(mode):
    env, seen = _run_to_the_first_arrival(mode)
    assert seen["shelf_after"] == seen["shelf_before"]
    assert seen["transit_before"] == {}
    putaways = [t for t in env.tasks if t.kind == "putaway"]
    assert putaways and all(t.release_time == 100 for t in putaways)
    on_the_way = {
        (k, v): q for k, per_vertex in seen["transit_after"].items() for v, q in per_vertex.items()
    }
    assert {(t.sku, t.delivery_vertex): t.quantity for t in putaways} == on_the_way
    assert len(putaways) == len(on_the_way)  # one per (SKU, shelf)
    # in SKU id, then vertex id order
    assert [(t.sku, t.delivery_vertex) for t in putaways] == sorted(on_the_way)
    receiving = sorted(v for v, x in env.graph.vertices.items() if x.role.receiving)
    for task in putaways:
        assert task.pickup_vertex == nearest_receiving_point(
            env.graph, task.delivery_vertex, receiving
        )
        assert env.graph.vertices[task.pickup_vertex].role.receiving


def _stock(env, sku_id, per_vertex):
    counts = {k: dict(v) for k, v in env.storage.counts.items()}
    counts[sku_id] = dict(per_vertex)
    env.storage = env.storage.model_copy(update={"counts": counts})


class _NoOrders:
    """An order stream that releases nothing, so only the lines a test queues are served."""

    def poisson(self, lam):
        return 0

    def random(self):
        return 0.0

    def integers(self, *args, **kwargs):
        return 0


def test_a_line_for_units_on_their_way_waits_until_the_putaway_is_delivered():
    env, step = _env(PUTAWAY)
    step()
    env._order_rng = _NoOrders()
    sku_id = "sku_000"
    shelf = sorted(v for v, x in env.graph.vertices.items() if x.role.storage)[0]
    delivery = env._delivery_vertices[0]
    _stock(env, sku_id, {})
    env._in_transit = {sku_id: {shelf: 3}}
    env._waiting_orders = [OrderLine(env.t, sku_id, delivery, order_id=9, missing=2)]
    assert env._release_orders() == []
    putaway = env.tasks[0].model_copy(
        update=dict(sku=sku_id, delivery_vertex=shelf, quantity=3, kind="putaway")
    )
    env._shelve(putaway, env.t)
    assert env._in_transit == {sku_id: {}}
    (pick,) = env._release_orders()
    assert (pick.pickup_vertex, pick.quantity, pick.kind) == (shelf, 2, "pick")


def test_stock_on_its_way_counts_in_the_arrivals_and_is_not_ordered_again():
    env, step = _env(PUTAWAY)
    step()
    sku_id = "sku_000"
    capacities = env.storage.capacities
    shelf = max(capacities, key=lambda v: (capacities[v] - env.storage.used_capacity(v), -v))
    _stock(env, sku_id, {})
    # far more than the floor and the orders of one step, and it fits on the shelf
    free = capacities[shelf] - env.storage.used_capacity(shelf)
    units = int(free // env.storage.skus[sku_id].unit_capacity)
    assert units > 2 * env.config.stock_floor
    env._in_transit = {sku_id: {shelf: units}}
    env._arrive(env.config.storage_epoch_length)
    assert not [t for t in env.tasks if t.kind == "putaway" and t.sku == sku_id]
    assert env._in_transit[sku_id] == {shelf: units}


def test_a_pick_that_waited_for_stock_is_made_when_a_putaway_is_delivered():
    env, step = _env(PUTAWAY, "demand")
    step()
    _stock(env, "sku_000", {})  # the most popular SKU, so its lines wait for the arrival
    for _ in range(299):
        step()
    delivered = {}
    for task in env.tasks:
        if task.kind == "putaway" and task.finish_time is not None:
            delivered.setdefault(task.sku, set()).add(task.finish_time)
    waited = [t for t in env.tasks if t.kind == "pick" and env.log.stock_wait_by_task[t.task_id]]
    assert waited
    for task in waited:
        made = task.release_time + env.log.stock_wait_by_task[task.task_id]
        assert made in delivered.get(task.sku, set())


def test_shelf_stock_and_stock_on_its_way_stay_within_capacity_and_the_replay_holds():
    def episode():
        env, step = _env(PUTAWAY, "demand")
        locations = []
        for _ in range(300):
            before = env.fleet
            step()
            assert is_collision_free(before, env.fleet)
            locations.append(env.fleet.locations())
            env._shelf_and_transit()  # StorageState checks the capacity of the sum
            for task in env.tasks:
                assert task.status(env.t) in ("waiting", "active", "completed")
        return env, locations

    first, first_locations = episode()
    second, second_locations = episode()
    assert sum(1 for t in first.tasks if t.kind == "putaway" and t.finish_time is not None)
    assert len(first.log.arrived_units_by_update) == 3
    assert first.tasks == second.tasks
    assert first_locations == second_locations
    assert first.storage == second.storage and first._in_transit == second._in_transit


def test_the_backlog_counts_waiting_and_active_putaways():
    env, step = _env(PUTAWAY)
    for _ in range(150):
        step()
    open_putaways = sum(
        1 for t in env.tasks if t.kind == "putaway" and t.status(env.t) != "completed"
    )
    assert open_putaways
    assert env._backlog(env.t) == sum(
        1 for t in env.tasks if t.status(env.t) != "completed"
    ) + len(env.waiting_orders)


# The trip distance


def _trip_graph():
    """A corridor 1 - 2 - 3 - 4 - 5 - 6 with the receiving vertex R (0) off 1, shelf B (7) off
    3, the delivery vertex D (8) off 5, shelf A (9) off 6 and a parking vertex (10) off 2. A is
    nearer D, 3 against 4, but its trip is 7 + 3 against B's 4 + 4."""
    roles = {v: VertexRole() for v in range(1, 7)}
    roles.update(
        {
            0: VertexRole(receiving=True),
            7: VertexRole(storage=True),
            8: VertexRole(delivery=True),
            9: VertexRole(storage=True),
            10: VertexRole(endpoint=True),
        }
    )
    pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 1), (3, 7), (5, 8), (6, 9), (2, 10)]
    edges = tuple(Edge(source=a, target=b, cost=1.0) for a, b in pairs) + tuple(
        Edge(source=b, target=a, cost=1.0) for a, b in pairs
    )
    vertices = {v: Vertex(id=v, role=role) for v, role in roles.items()}
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _empty_shelves():
    return StorageState(
        skus={"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        capacities={7: 5.0, 9: 5.0},
        counts={},
    )


def test_the_trip_distance_adds_both_legs():
    graph = _trip_graph()
    assert trip_distance(graph, 7, [0], [8]) == 4 + 4
    assert trip_distance(graph, 9, [0], [8]) == 7 + 3


@pytest.mark.parametrize("trip, shelf", [(False, 9), (True, 7)])
def test_the_demand_rule_puts_away_by_the_trip_under_robots(trip, shelf):
    graph = _trip_graph()
    shelves = _empty_shelves()
    result = putaway_demand(
        shelves, graph, {"tea": 3}, {}, {"tea": 1.0}, {}, {}, None, receiving=[0], trip=trip
    )
    assert result.putaway_distance == 3 * (7 if shelf == 9 else 4)
    assert result.storage.counts["tea"] == {shelf: 3}


def test_the_congestion_score_counts_traffic_on_the_receiving_leg():
    graph = _trip_graph()
    # (1, 2) is on the receiving leg of both shelves and on the path from D to neither.
    traffic, waits = {(1, 2): 1.0}, {(1, 2): 2.0}
    assert trip_exposure(graph, 7, [0], [8], traffic, waits) == pytest.approx(2.0)
    assert congestion_exposure(graph, 7, nearest_delivery_point(graph, 7, [8]), traffic, waits) == 0
    # (3, 7) is on B's receiving leg only. With traffic there B loses to A, whose trip is
    # longer by 2 but costs no traffic.
    traffic, waits = {(3, 7): 1.0}, {(3, 7): 2.0}
    shelves = _empty_shelves()
    result = putaway_congestion(
        shelves, graph, {"tea": 3}, {}, {"tea": 1.0}, traffic, waits, 2.0, receiving=[0], trip=True
    )
    assert result.storage.counts["tea"] == {9: 3}
