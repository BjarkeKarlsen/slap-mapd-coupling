"""Unit tests for slap_mapd_coupling.environment.multi_agent_env."""

import random

import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import EpisodeLog, WarehouseMAPDEnv
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.planners import PrioritisedAStarPlanner
from slap_mapd_coupling.planners.driver import step_with_planner

_PLANNER = PrioritisedAStarPlanner()  # keeps no state, so one instance serves every test


def _step(env: WarehouseMAPDEnv):
    return step_with_planner(env, _PLANNER)


def _graph(num_endpoints: int = 4, seed: int = 0) -> WarehouseGraph:
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=num_endpoints,
        seed=seed,
    )
    return generate_warehouse_graph(params)


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=25,
        wait_cost=0.0,
        observation_depth=2,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _env(graph: WarehouseGraph, config: ExperimentConfig) -> WarehouseMAPDEnv:
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config)


def test_reset_places_agents_on_distinct_endpoints():
    graph = _graph()
    endpoints = {v for v, vertex in graph.vertices.items() if vertex.role.endpoint}
    env = _env(graph, _config())
    obs, infos = env.reset()
    assert set(obs) == {1, 2, 3, 4}
    assert len(set(obs.values())) == 4  # distinct starting vertices
    assert set(obs.values()) <= endpoints


def test_reset_raises_when_fewer_endpoints_than_agents():
    graph = _graph(num_endpoints=2)
    with pytest.raises(ValueError, match="endpoints"):
        _env(graph, _config(num_agents=4))


def test_step_returns_dict_keyed_structures_with_all_key():
    env = _env(_graph(), _config())
    env.reset()
    obs, rewards, terminated, truncated, infos = _step(env)
    assert set(obs) == {1, 2, 3, 4}
    assert "__all__" in terminated and "__all__" in truncated and "__all__" in infos
    assert all(isinstance(v, bool) for k, v in terminated.items() if k != "__all__")


def test_episode_truncates_at_horizon():
    env = _env(_graph(), _config(horizon=3))
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = _step(env)
        assert truncated["__all__"] is False
    _, _, _, truncated, _ = _step(env)
    assert truncated["__all__"] is True


def test_multi_step_run_stays_collision_free():
    env = _env(_graph(), _config(horizon=40))
    env.reset()
    before = env.fleet
    for _ in range(40):
        _step(env)
        assert is_collision_free(before, env.fleet)
        before = env.fleet


def test_replay_is_bit_for_bit_deterministic_under_fixed_seed():
    def run() -> list[tuple[int, int]]:
        env = _env(_graph(), _config(seed=13, horizon=30))
        env.reset()
        trace = []
        for _ in range(30):
            obs, *_ = _step(env)
            trace.extend(sorted(obs.items()))
        return trace

    assert run() == run()


def test_different_seeds_produce_different_traces():
    def run(seed: int) -> list[tuple[int, int]]:
        env = _env(_graph(), _config(seed=seed, horizon=15))
        env.reset()
        trace = []
        for _ in range(15):
            obs, *_ = _step(env)
            trace.extend(sorted(obs.items()))
        return trace

    assert run(1) != run(2)


def test_f_fix_never_mutates_storage_counts():
    env = _env(_graph(), _config(horizon=20))
    initial_counts = env.storage.counts
    env.reset()
    for _ in range(20):
        _step(env)
    assert env.storage.counts == initial_counts  # storage_epoch_length=None: F_fix, never updates


def test_every_task_has_a_single_well_defined_status_at_every_t():
    env = _env(_graph(), _config(horizon=30, arrival_rate=2.0))
    env.reset()
    for t in range(30):
        _step(env)
        for task in env.tasks:
            # status() itself would raise if release_time > t; every
            # generated task's release_time is <= the tick it was
            # generated on, so this must never raise for t after release.
            status = task.status(t + 1)
            assert status in ("waiting", "active", "completed")


def test_completed_tasks_stay_completed():
    env = _env(_graph(), _config(horizon=40, arrival_rate=2.0))
    env.reset()
    completed_ids: set[int] = set()
    for t in range(40):
        _step(env)
        now_completed = {task.task_id for task in env.tasks if task.finish_time is not None}
        assert completed_ids <= now_completed  # monotonic: never un-completes
        completed_ids = now_completed
    assert completed_ids  # sanity: at least one task actually completed in 40 steps


def test_planner_runs_get_zero_rewards():
    env = _env(_graph(), _config(horizon=40))
    env.reset()
    for _ in range(40):
        _, rewards, *_ = _step(env)
        assert all(r == 0.0 for r in rewards.values())
    assert any(task.finish_time is not None for task in env.tasks)  # robots did deliver


def test_edge_traversals_are_logged_for_realised_moves():
    env = _env(_graph(), _config(horizon=15))
    env.reset()
    for _ in range(15):
        _step(env)
    assert sum(env.log.edge_traversals.values()) > 0
    for (source, target), count in env.log.edge_traversals.items():
        assert count > 0
        assert any(e.source == source and e.target == target for e in env.graph.edges)


def test_backlog_logged_every_timestep_including_reset():
    env = _env(_graph(), _config(horizon=10))
    env.reset()
    assert len(env.log.backlog_by_t) == 1
    for i in range(10):
        _step(env)
        assert len(env.log.backlog_by_t) == i + 2


def test_decision_runtimes_logged_once_per_step():
    # reset supplies the assignment for t = 0 and the last step supplies none, so
    # the assignment log holds one entry per timestep, the same as the routing log.
    env = _env(_graph(), _config(horizon=6))
    env.reset()
    for _ in range(6):
        _step(env)
    assert len(env.log.assignment_runtime_seconds) == 6
    assert len(env.log.routing_runtime_seconds) == 6


def test_reset_releases_and_assigns_the_orders_of_t0():
    env = _env(_graph(), _config(arrival_rate=10.0))
    env.reset()
    assert env.tasks
    assert all(task.release_time == 0 for task in env.tasks)
    assigned = [task for task in env.tasks if task.assignment_time == 0]
    assert {task.assigned_agent for task in assigned} == {1, 2, 3, 4}  # every free robot has one
    assert len(env.log.assignment_runtime_seconds) == 1


def test_reset_logs_the_backlog_before_the_orders_of_t0_are_released():
    env = _env(_graph(), _config(arrival_rate=10.0))
    env.reset()
    assert env.tasks
    assert env.log.backlog_by_t == [0]


def test_a_step_releases_and_assigns_the_orders_of_the_new_t():
    env = _env(_graph(), _config(arrival_rate=10.0, horizon=5))
    env.reset()
    for done in range(1, 5):
        _step(env)
        assert any(task.release_time == env.t for task in env.tasks)
        assert len(env.log.assignment_runtime_seconds) == done + 1
        assert len(env.log.backlog_by_t) == done + 1


def test_the_step_that_ends_the_episode_releases_and_assigns_nothing():
    env = _env(_graph(), _config(arrival_rate=10.0, horizon=5))
    env.reset()
    for _ in range(5):
        _step(env)
    assert env.t == 5
    assert all(task.release_time < 5 for task in env.tasks)
    assert len(env.log.assignment_runtime_seconds) == 5
    assert len(env.log.backlog_by_t) == 6


def _decentralised_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=25,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        move_cost=0.0,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_step_takes_the_slots_it_is_given():
    env = _env(_graph(), _decentralised_config())
    env.reset()
    wait_actions = {agent_id: 0 for agent_id in env.fleet.agents}  # slot 0 is always wait
    obs, rewards, terminated, truncated, infos = env.step(wait_actions)
    assert obs == dict(env.fleet.locations())


def test_step_with_no_argument_is_a_type_error():
    env = _env(_graph(), _config())
    env.reset()
    with pytest.raises(TypeError):
        env.step()  # type: ignore[call-arg]


def test_record_routing_runtime_appends_to_the_log():
    env = _env(_graph(), _config())
    env.log.record_routing_runtime(0.25)
    env.log.record_routing_runtime(0.5)
    assert env.log.routing_runtime_seconds == [0.25, 0.5]


def test_step_does_not_time_the_actions_it_is_given():
    # Whoever produces the actions times itself, so a bare step logs no routing time.
    env = _env(_graph(), _decentralised_config())
    env.reset()
    env.step({agent_id: 0 for agent_id in env.fleet.agents})
    assert env.log.routing_runtime_seconds == []


def test_step_with_external_wait_actions_keeps_agents_stationary():
    env = _env(_graph(), _decentralised_config(horizon=5))
    env.reset()
    before = env.fleet.locations()
    for _ in range(5):
        env.step({agent_id: 0 for agent_id in env.fleet.agents})
    assert env.fleet.locations() == before


def test_step_with_external_move_action_matches_action_for_slot():
    from slap_mapd_coupling.environment.spaces import action_for_slot

    graph = _graph()
    env = _env(graph, _decentralised_config(horizon=3))
    env.reset()
    agent_id = next(iter(env.fleet.agents))
    location = env.fleet.locations()[agent_id]
    legal = graph.legal_actions(location)
    if len(legal) < 2:
        pytest.skip("agent's start vertex has no legal move slot to exercise")
    move_action = action_for_slot(graph, location, 1)

    actions = {aid: 0 for aid in env.fleet.agents}
    actions[agent_id] = 1
    env.step(actions)

    # The proposal may still be overridden by conflict resolution (stage
    # 4), so this only checks the DECODED proposal matched action_for_slot,
    # via the realised trace when nothing conflicts (num_agents small
    # relative to the instance keeps this deterministic here).
    realised = env.fleet.locations()[agent_id]
    assert realised in {location, move_action.target}


def test_last_proposed_and_overridden_start_empty_and_follow_the_step():
    env = _env(_graph(), _config(num_agents=2, horizon=10))
    env.reset()
    assert env.last_proposed == {}
    assert env.last_overridden == frozenset()
    _step(env)
    assert set(env.last_proposed) == {1, 2}
    assert env.last_overridden <= {1, 2}
    env.reset()
    assert env.last_proposed == {}
    assert env.last_overridden == frozenset()


def _wait_step(env: WarehouseMAPDEnv):
    return env.step({agent_id: 0 for agent_id in env.fleet.agents})


def test_congestion_term_is_in_the_reward():
    # A field of view that covers the whole graph puts every other robot in every window.
    # Waiting keeps the two runs identical, so the only difference is n_cng.
    weighted = _env(_graph(), _decentralised_config(observation_depth=1000, horizon=5))
    unweighted = _env(
        _graph(),
        _decentralised_config(observation_depth=1000, horizon=5, congestion_reward_weight=0.0),
    )
    weighted.reset()
    unweighted.reset()
    for _ in range(3):
        delta = {
            aid: build_observation(
                weighted.graph, weighted.fleet, weighted.tasks, aid, weighted.t, weighted.config
            ).congestion
            for aid in weighted.fleet.agents
        }
        assert all(value > 0.0 for value in delta.values())
        _, with_cng, *_ = _wait_step(weighted)
        _, without_cng, *_ = _wait_step(unweighted)
        for agent_id, value in delta.items():
            assert with_cng[agent_id] == pytest.approx(without_cng[agent_id] - 0.1 * value)


@pytest.mark.parametrize("controller", ["centralised", "section", "decentralised"])
def test_every_arm_gets_a_reward_when_a_robot_delivers(controller):
    env = _env(
        _graph(), _decentralised_config(controller=controller, deliver_reward=100.0, horizon=60)
    )
    env.reset()
    delivered_reward = None
    for _ in range(60):
        _, rewards, *_ = _step(env)
        for task in env.tasks:
            if task.finish_time == env.t:
                delivered_reward = rewards[task.assigned_agent]
        if delivered_reward is not None:
            break
    assert delivered_reward is not None  # the fixture does deliver within the horizon
    assert delivered_reward > 50.0  # the delivery bonus, less at most the other terms


def test_only_a_realised_move_pays_the_move_cost():
    """A robot that moves pays c_move. One that chose to wait, or whose move conflict
    resolution turned into a wait, pays nothing for moving."""
    graph = _graph()
    env = _env(graph, _decentralised_config(horizon=60, move_cost=0.3))
    env.reset()
    rng = random.Random(0)
    seen = set()
    for _ in range(59):
        before = env.fleet.locations()
        paid_before = dict(env.log.return_move_by_agent)
        slots = {aid: rng.randrange(len(graph.legal_actions(v))) for aid, v in before.items()}
        env.step(slots)
        after = env.fleet.locations()
        for aid in env.fleet.agents:
            paid = env.log.return_move_by_agent.get(aid, 0.0) - paid_before.get(aid, 0.0)
            if after[aid] != before[aid]:
                seen.add("moved")
                assert paid == pytest.approx(-0.3)
            else:
                seen.add("overridden" if aid in env.last_overridden else "waited")
                assert paid == 0.0
    assert seen == {"moved", "waited", "overridden"}


def test_congestion_by_agent_equals_the_observation_builders_value():
    env = _env(_graph(), _decentralised_config(horizon=10))
    env.reset()
    for _ in range(6):
        assert set(env.congestion_by_agent) == set(env.fleet.agents)
        for agent_id, value in env.congestion_by_agent.items():
            observation = build_observation(
                env.graph, env.fleet, env.tasks, agent_id, env.t, env.config
            )
            assert value == observation.congestion
        _step(env)


def test_congestion_by_agent_is_read_only():
    env = _env(_graph(), _config())
    with pytest.raises(TypeError):
        env.congestion_by_agent[1] = 0.5  # type: ignore[index]


def test_crowding_log_has_one_entry_per_timestep_and_sums_the_mapping():
    env = _env(_graph(), _config(horizon=6))
    env.reset()
    assert len(env.log.crowding_sum_by_t) == 1
    assert env.log.crowding_sum_by_t[-1] == sum(env.congestion_by_agent.values())
    for done in range(1, 6):
        _step(env)
        assert len(env.log.crowding_sum_by_t) == done + 1
        assert env.log.crowding_sum_by_t[-1] == sum(env.congestion_by_agent.values())
    _step(env)  # the step that ends the episode logs nothing
    assert env.t == 6
    assert len(env.log.crowding_sum_by_t) == 6  # horizon entries


def test_the_log_keeps_the_longest_run_of_blocked_ticks_per_task():
    log = EpisodeLog()
    for _ in range(3):
        log.record_blocked_tick(1)
    log.record_moving_tick(1)
    for _ in range(5):
        log.record_blocked_tick(1)
    log.record_moving_tick(1)
    log.record_blocked_tick(1)
    log.record_blocked_tick(2)
    assert log.blocked_ticks_by_task == {1: 9, 2: 1}
    assert log.longest_blocked_run_by_task == {1: 5, 2: 1}
