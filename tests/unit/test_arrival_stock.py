"""Stock that runs out, under the arrival stock model.

An order is queued when it is released and served in release order. It takes one unit of its
SKU from the storage vertex holding it that is nearest its delivery vertex, ties by vertex id,
and only then becomes a task, with its release time kept. An order with no unit anywhere waits
and counts in the backlog. The SKU of an order is drawn without looking at stock. Stock arriving
at a storage epoch is not simulated yet, so these episodes end before the first epoch.
"""

from pathlib import Path

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.environment.multi_agent_env import Order, WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.training.spec import (
    ArrivalSettings,
    StorageRuleSettings,
    build_experiment_config,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
STEPS = 150
# Longer than every episode here, so no episode reaches an arrival.
NO_ARRIVAL = StorageRuleSettings(epoch_length=10_000)


def _spec(placement="demand_blocks", initial_units=8, **storage):
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    data = spec.model_dump()
    data["storage"].update(initial_placement=placement, initial_units=initial_units, **storage)
    data["storage_rule"] = NO_ARRIVAL.model_dump()
    data["arrival"] = ArrivalSettings(margin=1.0, receiving="delivery").model_dump()
    return type(spec).model_validate(data)


def _template(spec):
    data = spec.model_dump()
    data["arrival"] = None
    data["storage_rule"] = None
    data["storage"]["initial_placement"] = "demand_blocks"
    return type(spec).model_validate(data)


def _run(spec, mode="fixed", seed=200_000, steps=STEPS, rate=0.6):
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
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(steps):
        step()
    return env


def _orders(env):
    """Every order released so far, served or waiting, as (release time, SKU, delivery)."""
    served = [(t.release_time, t.sku, t.delivery_vertex) for t in env.tasks]
    waiting = [tuple(order) for order in env._waiting_orders]
    return sorted(served + waiting)


def test_the_orders_do_not_depend_on_the_stock():
    plenty = _run(_spec(initial_units=30, vertex_capacity=1000.0))
    scarce = _run(_spec(initial_units=1))
    assert scarce._waiting_orders  # the scarce stock runs out
    assert _orders(plenty) == _orders(scarce)


def test_the_orders_do_not_depend_on_the_storage_rule():
    fixed = _run(_spec(), mode="fixed")
    demand = _run(_spec(), mode="demand")
    assert _orders(fixed) == _orders(demand)


def test_with_stock_everywhere_the_orders_and_tasks_are_the_template_models():
    # demand_blocks holds each SKU at one vertex, so the nearest holder is the only holder, and
    # with plenty of units no order waits. The template draw then gives the same tasks.
    spec = _spec(initial_units=30, vertex_capacity=1000.0)
    arrival = _run(spec)
    template = _run(_template(spec))
    assert not arrival._waiting_orders
    assert [_key(t) for t in arrival.tasks] == [_key(t) for t in template.tasks]


def _key(task):
    return (task.task_id, task.release_time, task.sku, task.pickup_vertex, task.delivery_vertex)


def test_every_task_takes_one_unit_when_it_is_released():
    spec = _spec(initial_units=30, vertex_capacity=1000.0)
    start = build_study(spec).counts
    env = _run(spec)
    for sku_id, per_vertex in start.items():
        taken = sum(1 for t in env.tasks if t.sku == sku_id)
        assert sum(env.storage.counts[sku_id].values()) == sum(per_vertex.values()) - taken
    assert set(env.log.stock_wait_by_task.values()) == {0}


def test_an_order_with_no_stock_waits_and_counts_in_the_backlog():
    env = _run(_spec(initial_units=1))
    waiting = env._waiting_orders
    assert waiting
    for order in waiting:
        assert sum(env.storage.counts[order.sku].values()) == 0
    unfinished = sum(1 for t in env.tasks if t.status(env.t) in ("waiting", "active"))
    assert env._backlog(env.t) == unfinished + len(waiting)
    # Without arrivals no unit comes back, so every task was served at release.
    assert set(env.log.stock_wait_by_task.values()) == {0}
    assert len(env.log.stock_wait_by_task) == len(env.tasks)


def test_the_queue_is_served_in_release_order():
    env = _run(_spec(initial_units=1), steps=1)
    env._order_rng = _NoOrders()
    sku_id = "sku_000"
    vertex = next(iter(env.storage.counts[sku_id]))
    counts = {k: dict(v) for k, v in env.storage.counts.items()}
    counts[sku_id] = {vertex: 1}
    env.storage = env.storage.model_copy(update={"counts": counts})
    delivery = env._delivery_vertices[0]
    env._waiting_orders = [Order(0, sku_id, delivery), Order(1, sku_id, delivery)]
    env._t = 5
    (task,) = env._release_orders()
    assert task.release_time == 0
    assert env._waiting_orders == [Order(1, sku_id, delivery)]
    assert env.log.stock_wait_by_task[task.task_id] == 5


def test_the_pickup_is_the_holder_nearest_the_delivery_vertex_ties_by_id():
    env = _run(_spec(), steps=1)
    env._order_rng = _NoOrders()
    graph = env.graph
    storage = sorted(v for v, x in graph.vertices.items() if x.role.storage)
    sku_id = "sku_000"
    counts = {k: dict(v) for k, v in env.storage.counts.items()}
    counts[sku_id] = {v: 1 for v in storage}
    env.storage = env.storage.model_copy(update={"counts": counts})
    for delivery in env._delivery_vertices:
        env._waiting_orders = [Order(env.t, sku_id, delivery)]
        (task,) = env._release_orders()
        expected = min(
            (v for v, c in env.storage.counts[sku_id].items() if c > 0 or v == task.pickup_vertex),
            key=lambda v: (graph.distance(v, delivery), v),
        )
        assert task.pickup_vertex == expected
        assert env.storage.counts[sku_id][expected] == 0
        counts = {k: dict(v) for k, v in env.storage.counts.items()}
        counts[sku_id] = {v: 1 for v in storage}
        env.storage = env.storage.model_copy(update={"counts": counts})


def test_the_replay_is_deterministic():
    first, second = _run(_spec(initial_units=1)), _run(_spec(initial_units=1))
    assert first.tasks == second.tasks
    assert first._waiting_orders == second._waiting_orders
    assert first.log.backlog_by_t == second.log.backlog_by_t
    assert first.storage == second.storage


def test_an_arrival_epoch_is_not_simulated_yet():
    spec = _spec()
    data = spec.model_dump()
    data["storage_rule"]["epoch_length"] = 20
    with pytest.raises(NotImplementedError, match="arriving"):
        _run(type(spec).model_validate(data), steps=20)


def test_the_template_model_has_no_order_queue():
    env = _run(_template(_spec()))
    assert env._waiting_orders == []
    assert env.log.stock_wait_by_task == {}


def test_the_arrival_model_runs_under_every_storage_rule_until_the_first_epoch():
    for mode in ("fixed", "demand", "congestion"):
        data = _spec().model_dump()
        data["storage_rule"]["congestion_weight"] = 1.0
        spec = type(_spec()).model_validate(data)
        env = _run(spec, mode=mode, steps=50)
        assert isinstance(env, WarehouseMAPDEnv) and env.t == 50


class _NoOrders:
    """An order stream that releases nothing, so a test fills the queue itself."""

    def poisson(self, _rate):
        return 0
