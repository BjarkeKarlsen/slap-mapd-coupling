"""Stock that runs out, under the arrival stock model.

An order is queued when it is released and served in release order. It takes one unit of its
SKU from the storage vertex holding it that is nearest its delivery vertex, ties by vertex id,
and only then becomes a task, with its release time kept. An order with no unit anywhere waits
and counts in the backlog. The SKU of an order is drawn without looking at stock. These episodes
end before the first arrival, apart from the one that checks an arrival runs. Arrivals and their
put-away are tested in test_arrival_rules.py.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig, LinesPerOrder
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.multi_agent_env import OrderLine, WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.training.spec import (
    ArrivalSettings,
    StorageRuleSettings,
    build_experiment_config,
    build_study,
    load_spec,
    stock_sizing,
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


def _run(spec, mode="fixed", seed=200_000, steps=STEPS, rate=0.6, **config_overrides):
    study = build_study(spec)
    config = build_experiment_config(spec, seed, mode)
    config = ExperimentConfig.model_validate(
        {
            **config.model_dump(),
            "controller": None,
            "planner": "token_passing",
            "arrival_rate": rate,
            **config_overrides,
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
    waiting = [(o.release_time, o.sku, o.delivery_vertex) for o in env._waiting_orders]
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


def test_every_task_reserves_its_units_at_release_and_takes_them_at_pickup():
    spec = _spec(initial_units=30, vertex_capacity=1000.0)
    start = build_study(spec).counts
    env = _run(spec)
    for sku_id, per_vertex in start.items():
        picked = sum(
            t.quantity for t in env.tasks if t.sku == sku_id and t.pickup_time is not None
        )
        promised = sum(t.quantity for t in env.tasks if t.sku == sku_id and t.pickup_time is None)
        assert sum(env.storage.counts[sku_id].values()) == sum(per_vertex.values()) - picked
        assert sum(env._reserved.get(sku_id, {}).values()) == promised
    assert any(t.pickup_time is None for t in env.tasks)  # some units are still only reserved
    assert set(env.log.stock_wait_by_task.values()) == {0}


def test_an_order_with_no_stock_waits_and_counts_in_the_backlog():
    env = _run(_spec(initial_units=1))
    waiting = env._waiting_orders
    assert waiting
    free = env._unreserved(env.storage)
    for order in waiting:
        assert sum(free.counts[order.sku].values()) == 0
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
    env._waiting_orders = [OrderLine(0, sku_id, delivery), OrderLine(1, sku_id, delivery)]
    env._t = 5
    (task,) = env._release_orders()
    assert task.release_time == 0
    assert env._waiting_orders == [OrderLine(1, sku_id, delivery)]
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
        env._waiting_orders = [OrderLine(env.t, sku_id, delivery)]
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


def test_an_arrival_epoch_now_runs():
    spec = _spec()
    data = spec.model_dump()
    data["storage_rule"]["epoch_length"] = 20
    env = _run(type(spec).model_validate(data), steps=20)
    assert env.t == 20
    assert len(env.log.arrived_units_by_update) == 1
    assert env.log.relocated_units_by_update == []


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


# Order lines

LINES = dict(
    lines_per_order=LinesPerOrder(distribution="shifted_poisson", mean=5.0),
    units_per_line=(1, 5),
)


def _lines(env):
    """Every order line released so far, as (release time, order, line, SKU, delivery). A line
    served over several tasks appears once."""
    served = {
        (t.release_time, t.order_id, t.line_index, t.sku, t.delivery_vertex) for t in env.tasks
    }
    waiting = {
        (o.release_time, o.order_id, o.line_index, o.sku, o.delivery_vertex)
        for o in env._waiting_orders
    }
    return sorted(served | waiting)


def test_the_order_lines_do_not_depend_on_the_storage_rule():
    fixed = _run(_spec(), mode="fixed", **LINES)
    demand = _run(_spec(), mode="demand", **LINES)
    assert _lines(fixed)
    assert _lines(fixed) == _lines(demand)


def test_the_lines_of_an_order_share_its_release_and_delivery_and_have_distinct_skus():
    env = _run(_spec(), **LINES)
    by_order: dict[int, list[tuple]] = {}
    for line in _lines(env):
        by_order.setdefault(line[1], []).append(line)
    assert any(len(lines) > 1 for lines in by_order.values())
    for lines in by_order.values():
        assert len({(line[0], line[4]) for line in lines}) == 1
        assert len({line[3] for line in lines}) == len(lines)
        assert sorted(line[2] for line in lines) == list(range(len(lines)))


def test_a_line_asks_for_units_in_the_configured_range():
    env = _run(_spec(), steps=1, **LINES)
    ranked = sorted(env._sku_ids)
    quantities = [line.missing for t in range(500) for line in env._draw_order_lines(t, ranked)]
    assert set(quantities) == {1, 2, 3, 4, 5}


def test_arrival_rate_is_order_lines_per_timestep():
    env = _run(_spec(), steps=1, rate=0.6, **LINES)
    ranked = sorted(env._sku_ids)
    steps = 20_000
    num_lines = sum(len(env._draw_order_lines(t, ranked)) for t in range(steps))
    assert num_lines / steps == pytest.approx(0.6, rel=0.05)


def test_order_lines_only_apply_to_the_arrival_model():
    config = build_experiment_config(_template(_spec()), 0, "fixed")
    for name, value in LINES.items():
        with pytest.raises(ValidationError, match=name):
            ExperimentConfig.model_validate({**config.model_dump(), name: value})
    arrival = build_experiment_config(_spec(), 0, "fixed")
    with pytest.raises(ValidationError, match="units_per_line"):
        ExperimentConfig.model_validate({**arrival.model_dump(), "units_per_line": (5, 1)})


# The pickup rule over several shelves


def _two_shelves(env, sku_id):
    """A delivery vertex and two storage vertices at different distances from it, nearer
    first, with `sku_id` on no other shelf."""
    graph = env.graph
    storage = sorted(v for v, x in graph.vertices.items() if x.role.storage)
    for delivery in env._delivery_vertices:
        by_distance = sorted(storage, key=lambda v: (graph.distance(v, delivery), v))
        near, far = by_distance[0], by_distance[-1]
        if graph.distance(near, delivery) < graph.distance(far, delivery):
            return delivery, near, far
    raise AssertionError("no delivery vertex with two storage vertices at different distances")


def _stock(env, sku_id, per_vertex):
    counts = {k: dict(v) for k, v in env.storage.counts.items()}
    counts[sku_id] = dict(per_vertex)
    env.storage = env.storage.model_copy(update={"counts": counts})


@pytest.mark.parametrize(
    "units, tasks, left",
    [(4, [1, 3], 0), (7, [1, 5], 1), (1, [1], 0)],
)
def test_a_line_takes_from_the_nearest_shelf_first_and_the_rest_waits(units, tasks, left):
    env = _run(_spec(), steps=1)
    env._order_rng = _NoOrders()
    sku_id = "sku_000"
    delivery, near, far = _two_shelves(env, sku_id)
    _stock(env, sku_id, {near: 1, far: 5})
    env._waiting_orders = [OrderLine(env.t, sku_id, delivery, order_id=9, missing=units)]
    released = env._release_orders()
    assert [(t.pickup_vertex, t.quantity) for t in released] == list(
        zip([near, far], tasks)
    )
    assert all(t.order_id == 9 and t.release_time == env.t for t in released)
    assert sum(env.storage.counts[sku_id].values()) == 6 - sum(tasks)
    if left:
        (line,) = env._waiting_orders
        assert line.missing == left and line.order_id == 9
    else:
        assert env._waiting_orders == []


def test_the_rest_of_a_line_is_served_the_same_way_once_stock_arrives():
    env = _run(_spec(), steps=1)
    env._order_rng = _NoOrders()
    sku_id = "sku_000"
    delivery, near, far = _two_shelves(env, sku_id)
    _stock(env, sku_id, {})
    env._waiting_orders = [OrderLine(0, sku_id, delivery, missing=4)]
    assert env._release_orders() == []
    _stock(env, sku_id, {near: 3, far: 5})
    env._t = 7
    released = env._release_orders()
    assert [(t.pickup_vertex, t.quantity) for t in released] == [(near, 3), (far, 1)]
    assert all(env.log.stock_wait_by_task[t.task_id] == 7 for t in released)


def test_the_stock_condition_needs_the_units_the_task_carries():
    env = _run(_spec(), steps=1)
    sku_id = "sku_000"
    delivery, near, _ = _two_shelves(env, sku_id)
    _stock(env, sku_id, {near: 2})
    task = Task(
        task_id=1, release_time=0, pickup_vertex=near, delivery_vertex=delivery, sku=sku_id
    )
    assert task.is_pickup_feasible(env.storage)
    assert task.model_copy(update={"quantity": 2}).is_pickup_feasible(env.storage)
    assert not task.model_copy(update={"quantity": 3}).is_pickup_feasible(env.storage)


def test_with_order_lines_the_stock_stays_within_capacity_and_the_replay_is_deterministic():
    def run():
        data = _spec(initial_units=2).model_dump()
        data["storage_rule"]["epoch_length"] = 40
        return _run(type(_spec()).model_validate(data), steps=STEPS, **LINES)

    first, second = run(), run()
    assert first.tasks == second.tasks
    assert first._waiting_orders == second._waiting_orders
    assert first.storage == second.storage
    assert any(t.quantity > 1 for t in first.tasks)
    for v, capacity in first.storage.capacities.items():
        assert first.storage.used_capacity(v) <= capacity + 1e-9
    assert all(c >= 0 for per_vertex in first.storage.counts.values() for c in per_vertex.values())
    for t in range(first.t + 1):
        for task in first.tasks:
            if task.release_time <= t:
                assert task.status(t) in ("waiting", "active", "completed")


def test_the_sized_starting_stock_counts_the_mean_units_per_line():
    data = _spec(placement="demand_sized_blocks", initial_units=1).model_dump()
    data["storage_rule"]["epoch_length"] = 100
    plain = type(_spec()).model_validate(data)
    data["arrival"]["units_per_line"] = (1, 5)
    lines = type(plain).model_validate(data)
    assert stock_sizing(plain).mean_units == 1.0
    assert stock_sizing(lines).mean_units == 3.0
    assert build_experiment_config(lines, 0, "fixed").units_per_line == (1, 5)
    one, three = build_study(plain).counts, build_study(lines).counts
    top = max(one, key=lambda k: sum(one[k].values()))
    assert sum(three[top].values()) > 2 * sum(one[top].values())


def test_the_log_holds_the_order_and_line_of_every_task_and_the_lines_that_waited():
    env = _run(_spec(initial_units=2), **LINES)
    assert env._waiting_orders  # the small stock runs out
    assert set(env.log.order_line_by_task) == {t.task_id for t in env.tasks}
    for task in env.tasks:
        assert env.log.order_line_by_task[task.task_id] == (task.order_id, task.line_index)
    waiting = {(line.order_id, line.line_index) for line in env._waiting_orders}
    assert waiting <= env.log.lines_waited
