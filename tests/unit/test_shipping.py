"""Shipping and random docks (step 12 of the model revision, D21). An order whose last pick
reaches its workstation becomes one ship task from the workstation to a dock, and the order is
done when it leaves there. Under dock_choice "random" every put-away and ship task takes a dock
drawn at random."""

from collections import Counter
from pathlib import Path

import pytest

import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.evaluation.evaluator import _planner_run, order_metrics
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"
SEED = 300_000
HORIZON = 600


def _spec(**arrival) -> TrainingSpec:
    data = load_spec(FLEET_8).model_dump()
    data["arrival"].update(arrival)
    return TrainingSpec.model_validate(data)


_RUNS: dict = {}


def _run(mode="fixed", **arrival):
    key = (mode, tuple(sorted(arrival.items())))
    if key not in _RUNS:
        spec = _spec(**arrival)
        study = build_study(spec)
        config = build_experiment_config(spec, SEED, mode)
        config = ExperimentConfig.model_validate(
            {**config.model_dump(), "controller": None, "planner": "token_passing",
             "horizon": HORIZON}
        )  # fmt: skip
        env, step = _planner_run(
            study.instance.graph, study.skus, study.counts, study.capacities, config,
            "token_passing",
        )  # fmt: skip
        for _ in range(HORIZON):
            step()
        _RUNS[key] = env
    return _RUNS[key]


def _docks(env) -> set:
    return {v for v, x in env.graph.vertices.items() if x.role.receiving}


def test_every_order_ships_once_when_its_last_pick_is_in():
    env = _run(shipping=True)
    ships = [task for task in env.tasks if task.kind == "ship"]
    assert ships
    assert len({task.order_id for task in ships}) == len(ships)
    picks_by_order: dict = {}
    for task in env.tasks:
        if task.kind == "pick":
            picks_by_order.setdefault(task.order_id, []).append(task)
    waiting = {line.order_id for line in env.waiting_orders}
    for ship in ships:
        picks = picks_by_order[ship.order_id]
        assert ship.order_id not in waiting
        assert all(p.finish_time is not None for p in picks)
        assert ship.release_time == max(p.finish_time for p in picks)
        assert ship.pickup_vertex == picks[0].delivery_vertex
        assert ship.delivery_vertex in _docks(env)
        assert ship.quantity == sum(p.quantity for p in picks)
    # An order with every pick in and no line waiting has shipped.
    for order_id, picks in picks_by_order.items():
        if order_id not in waiting and all(p.finish_time is not None for p in picks):
            assert order_id in {s.order_id for s in ships}
    assert any(s.finish_time is not None for s in ships)


def test_without_shipping_nothing_ships_and_the_run_is_unchanged():
    plain = _run()
    assert not any(task.kind == "ship" for task in plain.tasks)
    # The order stream never sees the dock draws, so the picks are released as before.
    shipping = _run(shipping=True)
    first = lambda env: [  # noqa: E731
        (t.release_time, t.sku, t.order_id) for t in env.tasks if t.kind == "pick"
    ][:20]
    assert first(plain) == first(shipping)


def test_random_docks_spread_the_shipments_over_the_docks():
    env = _run(shipping=True, dock_choice="random")
    docks = Counter(t.delivery_vertex for t in env.tasks if t.kind == "ship")
    assert set(docks) == _docks(env)
    assert {t.pickup_vertex for t in env.tasks if t.kind == "putaway"} <= _docks(env)


def test_an_order_with_shipping_is_done_when_it_leaves_at_the_dock():
    def task(task_id, kind, release, finish, order_id=7):
        return Task(
            task_id=task_id, release_time=release, pickup_vertex=1, delivery_vertex=2,
            sku="a", order_id=order_id, kind=kind, assigned_agent=1,
            assignment_time=release, pickup_time=release, finish_time=finish,
        )  # fmt: skip

    picks = [task(1, "pick", 10, 30), task(2, "pick", 10, 40)]
    by_task = {1: (7, 0), 2: (7, 1)}
    assert order_metrics(picks, by_task, [], set())["mean_order_completion"] == 30
    shipped = order_metrics(picks + [task(3, "ship", 40, 55)], by_task, [], set(), shipping=True)
    assert shipped["mean_order_completion"] == 45
    assert order_metrics(picks, by_task, [], set(), shipping=True)["num_completed_orders"] == 0


def test_shipping_needs_the_docks_and_robot_put_away():
    with pytest.raises(ValueError, match="shipping and dock_choice need"):
        build_experiment_config(_spec(shipping=True, putaway="instant"), SEED, "fixed")


def test_with_random_docks_the_put_away_leg_is_the_mean_over_the_docks():
    from slap_mapd_coupling.storage.arrival import DrawnDocks, receiving_distance, trip_distance

    graph = _run().graph
    docks = sorted(_docks(_run()))
    slot = next(v for v, x in graph.vertices.items() if x.role.storage)
    distances = [graph.distance(g, slot) for g in docks]
    assert receiving_distance(graph, slot, docks) == min(distances)
    assert receiving_distance(graph, slot, DrawnDocks(docks)) == sum(distances) / len(distances)
    stations = [v for v, x in graph.vertices.items() if x.role.delivery]
    assert trip_distance(graph, slot, DrawnDocks(docks), stations) >= trip_distance(
        graph, slot, docks, stations
    )
