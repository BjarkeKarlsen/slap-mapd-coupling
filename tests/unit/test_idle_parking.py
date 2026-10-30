"""Idle robots in the way (step 12 of the model revision, D26). A robot with no task that stands
on the pickup or delivery vertex of an unfinished task, or on a vertex another robot was stopped
from moving onto, gets the nearest free parking leaf as its target until it arrives or gets a
task. A robot in nobody's way keeps its own vertex as its target."""

from pathlib import Path

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.reward_function import current_target, reward_terms
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"


@pytest.fixture(scope="module")
def study():
    return build_study(load_spec(FLEET_8))


def _env(study, idle_parking=True) -> WarehouseMAPDEnv:
    config = build_experiment_config(load_spec(FLEET_8), 0, "fixed")
    config = ExperimentConfig.model_validate({**config.model_dump(), "idle_parking": idle_parking})
    env = WarehouseMAPDEnv(
        study.instance.graph, study.skus, study.counts, study.capacities, config
    )
    # No tasks and every robot on its start, so each case sets up only what it needs.
    env.tasks = []
    return env


def _place(env, agent_id, vertex) -> None:
    agents = dict(env.fleet.agents)
    agents[agent_id] = AgentState(agent_id=agent_id, location=vertex)
    env.fleet = FleetState(agents=agents)


def _roles(env, name):
    return sorted(v for v, x in env.graph.vertices.items() if getattr(x.role, name))


def _parking_leaves(env):
    return [
        v
        for v, x in env.graph.vertices.items()
        if x.role.endpoint and not (x.role.storage or x.role.delivery or x.role.receiving)
    ]


def _waiting_task(env, pickup, delivery) -> Task:
    return Task(
        task_id=999, release_time=env.t, pickup_vertex=pickup, delivery_vertex=delivery,
        sku=next(iter(env.storage.skus)),
    )  # fmt: skip


def test_an_idle_robot_on_a_pickup_is_sent_to_the_nearest_free_parking_leaf(study):
    env = _env(study)
    slot, station = _roles(env, "storage")[0], _roles(env, "delivery")[0]
    _place(env, 1, slot)
    env.tasks = [_waiting_task(env, slot, station)]
    env._update_parking({}, frozenset())
    leaf = env.parking_targets[1]
    taken = set(env.fleet.locations().values())
    free = [v for v in _parking_leaves(env) if v not in taken]
    assert leaf == min(free, key=lambda v: (env.graph.distance(slot, v), v))
    assert set(env.parking_targets) == {1}


def test_an_idle_robot_on_a_delivery_is_sent_away_too(study):
    env = _env(study)
    slot, station = _roles(env, "storage")[0], _roles(env, "delivery")[0]
    _place(env, 1, station)
    env.tasks = [_waiting_task(env, slot, station)]
    env._update_parking({}, frozenset())
    assert 1 in env.parking_targets


def test_an_idle_robot_another_was_stopped_from_reaching_is_sent_away(study):
    env = _env(study)
    walkable = [v for v, x in env.graph.vertices.items() if not x.role.is_task_endpoint]
    _place(env, 1, walkable[0])
    env._update_parking({2: Action(kind="move", target=walkable[0])}, frozenset({2}))
    assert 1 in env.parking_targets
    # Not stopped, nobody else's business: it stays.
    other = _env(study)
    _place(other, 1, walkable[0])
    other._update_parking({2: Action(kind="move", target=walkable[0])}, frozenset())
    assert 1 not in other.parking_targets


def test_two_robots_in_the_way_get_different_leaves(study):
    env = _env(study)
    slots, station = _roles(env, "storage"), _roles(env, "delivery")[0]
    _place(env, 1, slots[0])
    _place(env, 2, slots[1])
    env.tasks = [_waiting_task(env, slots[0], station)]
    env.tasks.append(env.tasks[0].model_copy(update={"task_id": 1000, "pickup_vertex": slots[1]}))
    env._update_parking({}, frozenset())
    assert env.parking_targets[1] != env.parking_targets[2]


def test_the_leaf_is_dropped_on_arrival(study):
    env = _env(study)
    slot, station = _roles(env, "storage")[0], _roles(env, "delivery")[0]
    _place(env, 1, slot)
    env.tasks = [_waiting_task(env, slot, station)]
    env._update_parking({}, frozenset())
    _place(env, 1, env.parking_targets[1])
    env.tasks = []
    env._update_parking({}, frozenset())
    assert 1 not in env.parking_targets


def test_off_nothing_is_sent_anywhere(study):
    env = _env(study, idle_parking=False)
    slot, station = _roles(env, "storage")[0], _roles(env, "delivery")[0]
    _place(env, 1, slot)
    env.tasks = [_waiting_task(env, slot, station)]
    env._update_parking({}, frozenset())
    assert not env.parking_targets


def test_the_parking_leaf_is_the_target_of_the_shaping(study):
    graph = study.instance.graph
    leaf = sorted(v for v, x in graph.vertices.items() if x.role.endpoint)[0]
    here = graph.out_neighbours(leaf)[0]
    assert current_target(here, None, leaf) == leaf
    assert current_target(here, None) == here
    terms = reward_terms(
        graph, location_now=here, task_now=None, location_next=leaf, task_next=None,
        completed_task=False, overridden=False, moved=True, stood_still=False, congestion=0.0,
        discount=1.0, deliver_reward=0.0, override_penalty=0.0, move_cost=0.0, wait_cost=0.0,
        congestion_reward_weight=0.0, parking=leaf,
    )  # fmt: skip
    assert terms.shaping == pytest.approx(graph.distance(here, leaf))
