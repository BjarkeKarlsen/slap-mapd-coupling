"""Unit tests for slap_mapd_coupling.environment.multi_agent_env."""

import pytest

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph


def _graph(num_endpoints: int = 4, seed: int = 0) -> WarehouseGraph:
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
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
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=25,
        wait_cost=0.0,
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
    obs, rewards, terminated, truncated, infos = env.step()
    assert set(obs) == {1, 2, 3, 4}
    assert "__all__" in terminated and "__all__" in truncated and "__all__" in infos
    assert all(isinstance(v, bool) for k, v in terminated.items() if k != "__all__")


def test_episode_truncates_at_horizon():
    env = _env(_graph(), _config(horizon=3))
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step()
        assert truncated["__all__"] is False
    _, _, _, truncated, _ = env.step()
    assert truncated["__all__"] is True


def test_multi_step_run_stays_collision_free():
    env = _env(_graph(), _config(horizon=40))
    env.reset()
    before = env.fleet
    for _ in range(40):
        env.step()
        assert is_collision_free(before, env.fleet)
        before = env.fleet


def test_replay_is_bit_for_bit_deterministic_under_fixed_seed():
    def run() -> list[tuple[int, int]]:
        env = _env(_graph(), _config(seed=13, horizon=30))
        env.reset()
        trace = []
        for _ in range(30):
            obs, *_ = env.step()
            trace.extend(sorted(obs.items()))
        return trace

    assert run() == run()


def test_different_seeds_produce_different_traces():
    def run(seed: int) -> list[tuple[int, int]]:
        env = _env(_graph(), _config(seed=seed, horizon=15))
        env.reset()
        trace = []
        for _ in range(15):
            obs, *_ = env.step()
            trace.extend(sorted(obs.items()))
        return trace

    assert run(1) != run(2)


def test_f_fix_never_mutates_storage_counts():
    env = _env(_graph(), _config(horizon=20))
    initial_counts = env.storage.counts
    env.reset()
    for _ in range(20):
        env.step()
    assert env.storage.counts == initial_counts  # storage_epoch_length=None: F_fix, never updates


def test_every_task_has_a_single_well_defined_status_at_every_t():
    env = _env(_graph(), _config(horizon=30, arrival_rate=2.0))
    env.reset()
    for t in range(30):
        env.step()
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
        env.step()
        now_completed = {task.task_id for task in env.tasks if task.completion_time is not None}
        assert completed_ids <= now_completed  # monotonic: never un-completes
        completed_ids = now_completed
    assert completed_ids  # sanity: at least one task actually completed in 40 steps


def test_reward_is_zero_for_non_decentralised_controller():
    env = _env(_graph(), _config())
    env.reset()
    for _ in range(5):
        _, rewards, *_ = env.step()
        assert all(r == 0.0 for r in rewards.values())


def test_edge_traversals_are_logged_for_realised_moves():
    env = _env(_graph(), _config(horizon=15))
    env.reset()
    for _ in range(15):
        env.step()
    assert sum(env.log.edge_traversals.values()) > 0
    for (source, target), count in env.log.edge_traversals.items():
        assert count > 0
        assert any(e.source == source and e.target == target for e in env.graph.edges)


def test_total_movement_cost_matches_hand_computed_realised_cost():
    env = _env(_graph(), _config(horizon=10, num_agents=4))
    env.reset()
    before = env.fleet.locations()
    expected = 0.0
    for _ in range(10):
        env.step()
        after = env.fleet.locations()
        for agent_id, before_v in before.items():
            after_v = after[agent_id]
            if before_v == after_v:
                expected += env.graph.wait_cost
            else:
                edge = next(
                    e for e in env.graph.edges if e.source == before_v and e.target == after_v
                )
                expected += edge.cost
        before = after
    assert env.log.total_movement_cost == pytest.approx(expected)


def test_backlog_logged_every_timestep_including_reset():
    env = _env(_graph(), _config(horizon=10))
    env.reset()
    assert len(env.log.backlog_by_t) == 1
    for i in range(10):
        env.step()
        assert len(env.log.backlog_by_t) == i + 2


def test_decision_runtimes_logged_once_per_step():
    env = _env(_graph(), _config(horizon=6))
    env.reset()
    for _ in range(6):
        env.step()
    assert len(env.log.assignment_runtime_seconds) == 6
    assert len(env.log.routing_runtime_seconds) == 6


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
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_step_with_external_actions_never_resolves_a_controller():
    """controller='decentralised' has no registered Controller (#29 isn't
    implemented yet) -- driving every step via `actions` must not need
    one, per the module docstring's lazy-resolution design."""
    env = _env(_graph(), _decentralised_config())
    env.reset()
    wait_actions = {agent_id: 0 for agent_id in env.fleet.agents}  # slot 0 is always wait
    obs, rewards, terminated, truncated, infos = env.step(wait_actions)
    assert obs == dict(env.fleet.locations())
    assert env._controller is None  # never resolved


def test_step_without_actions_raises_for_unregistered_controller():
    env = _env(_graph(), _decentralised_config())
    env.reset()
    with pytest.raises(KeyError, match="decentralised"):
        env.step()


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
