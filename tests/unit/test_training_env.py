"""Unit tests for slap_mapd_coupling.training.env."""

import numpy as np
import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.observation import local_subgraph
from slap_mapd_coupling.environment.spaces import action_for_slot
from slap_mapd_coupling.environment.zones import halo, zone_of, zones_by_aisle
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_instance,
    generate_warehouse_graph,
)
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv


def _params(**overrides) -> GeneratorParams:
    defaults = dict(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=4,
        seed=0,
    )
    defaults.update(overrides)
    return GeneratorParams(**defaults)


def _graph(**overrides):
    return generate_warehouse_graph(_params(**overrides))


# The largest field of view of _graph() at d_obs = 3 has 20 vertices. A window is never
# truncated, so the cap must fit it.
_MAX_LOCAL_NODES = 20
# _graph() has 30 movable vertices, the centralised window and the largest section window.
_WIDE_MAX_LOCAL_NODES = 30


def _instance(graph):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return skus, capacities, counts


def _zones(num_zones: int = 2):
    instance = generate_instance(_params())
    return zones_by_aisle(instance.graph, instance.aisle_index, num_zones)


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=1.0,
        seed=7,
        horizon=10,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _env(
    train_seeds: tuple[int, ...] = (100, 101, 102),
    *,
    credit_signal="per_agent",
    pooling="none",
    zones=None,
    max_local_nodes=_MAX_LOCAL_NODES,
    **config_overrides,
) -> WarehouseMAPDMultiAgentEnv:
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=max_local_nodes, max_messages=2)
    return WarehouseMAPDMultiAgentEnv(
        graph,
        skus,
        counts,
        capacities,
        _config(**config_overrides),
        enc_config,
        train_seeds,
        credit_signal=credit_signal,
        pooling=pooling,
        zones=zones,
    )


def _arm_env(controller: str, pooling: str = "none", **kwargs) -> WarehouseMAPDMultiAgentEnv:
    """An adapter for one arm, with zones for the section arm and a cap that
    fits the arm's largest window."""
    if controller != "decentralised":
        kwargs.setdefault("max_local_nodes", _WIDE_MAX_LOCAL_NODES)
    if controller == "section":
        kwargs.setdefault("zones", _zones())
    return _env(controller=controller, pooling=pooling, **kwargs)


def test_empty_train_seeds_rejected():
    with pytest.raises(ValueError, match="train_seeds"):
        _env(train_seeds=())


def test_reset_observations_match_declared_spaces():
    env = _env()
    obs, infos = env.reset()
    assert set(obs) == set(env.agents)
    for aid in env.agents:
        assert env.observation_spaces[aid].contains(obs[aid])
        assert infos[aid] == {}


@pytest.mark.parametrize("communication", [False, True])
def test_observations_carry_the_sender_keys_exactly_when_communication_is_on(communication):
    env = _env(communication=communication)
    obs, _ = env.reset()
    for _ in range(3):
        for aid in env.agents:
            keys = set(obs[aid]["observations"])
            assert (set(SENDER_KEYS) <= keys) == communication
            assert "move_target_index" in keys
            assert ({"move_sender_index", "move_sender_mask"} <= keys) == communication
            assert env.observation_spaces[aid].contains(obs[aid])
        obs, *_ = env.step({aid: 0 for aid in env.agents})


def test_communication_with_no_room_for_a_sender_is_rejected():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=_MAX_LOCAL_NODES, max_messages=0)
    with pytest.raises(ValueError, match="max_messages"):
        WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            counts,
            capacities,
            _config(communication=True),
            enc_config,
            (1,),
            credit_signal="per_agent",
            pooling="none",
        )


def test_a_cap_too_small_for_a_vertex_and_its_neighbours_is_rejected():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=2, max_messages=1)
    with pytest.raises(ValueError, match="max_local_nodes"):
        WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            counts,
            capacities,
            _config(),
            enc_config,
            (1,),
            credit_signal="per_agent",
            pooling="none",
        )


def test_a_neighbour_outside_the_field_of_view_is_rejected_when_the_env_is_built():
    graph = _graph(random_edge_costs=True, edge_cost_range=(1.5, 2.0))
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=1)
    with pytest.raises(ValueError, match="observation_depth"):
        WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            counts,
            capacities,
            _config(observation_depth=1),
            enc_config,
            (1,),
            credit_signal="per_agent",
            pooling="none",
        )


def test_step_with_wait_actions_keeps_agents_stationary():
    env = _env()
    env.reset()
    before = env.core.fleet.locations()
    for _ in range(5):
        actions = {aid: 0 for aid in env.agents}  # slot 0 is always wait
        obs, rewards, terminated, truncated, infos = env.step(actions)
    assert env.core.fleet.locations() == before
    assert set(rewards) == set(env.agents)
    assert terminated["__all__"] is False


def test_successive_resets_cycle_through_training_seeds_only():
    """sec:method:training: seeds are split disjointly between training
    and evaluation. RLlib always calls reset(seed=None) at episode start
    (training/env.py's module docstring), so successive resets must cycle
    through train_seeds in order and never stray into an eval seed."""
    train_seeds = (10, 11, 12)
    eval_seeds = (13, 14)
    env = _env(train_seeds=train_seeds)

    seen = []
    for _ in range(7):  # more than one full cycle
        env.reset()
        seen.append(env.last_seed)

    assert seen == [10, 11, 12, 10, 11, 12, 10]
    assert set(seen).isdisjoint(eval_seeds)


def test_two_envs_with_the_same_train_seeds_produce_the_same_sequence():
    train_seeds = (20, 21, 22)
    env_a = _env(train_seeds=train_seeds)
    env_b = _env(train_seeds=train_seeds)

    sequence_a = [_reset_and_get_seed(env_a) for _ in range(5)]
    sequence_b = [_reset_and_get_seed(env_b) for _ in range(5)]

    assert sequence_a == sequence_b


def _reset_and_get_seed(env: WarehouseMAPDMultiAgentEnv) -> int:
    env.reset()
    assert env.last_seed is not None
    return env.last_seed


def test_explicit_seed_overrides_the_training_cycle():
    env = _env(train_seeds=(30, 31))
    env.reset(seed=999)
    assert env.last_seed == 999
    env.reset()  # back to the cycle, starting at index 0
    assert env.last_seed == 30


def test_step_respects_action_mask_and_truncates_at_horizon():
    env = _env(horizon=6)
    obs, _ = env.reset()
    truncated = {"__all__": False}
    for _ in range(6):
        actions = {}
        for aid in env.agents:
            legal = np.flatnonzero(obs[aid]["action_mask"])
            actions[aid] = int(np.random.choice(legal))
        obs, rewards, terminated, truncated, infos = env.step(actions)
    assert truncated["__all__"] is True


ETA_COLUMN = 4  # the distance label in a node feature row (models/gnn_encoder.py)


def _own_eta(obs: dict, aid: str) -> float:
    view = obs[aid]["observations"]
    return float(view["node_features"][int(view["own_index"])][ETA_COLUMN])


def test_the_reset_observation_shows_the_target_of_a_robot_assigned_at_t0():
    env = _env(arrival_rate=10.0)
    obs, _ = env.reset()
    inner = env.core
    assigned = {task.assigned_agent: task for task in inner.tasks if task.assignment_time == 0}
    assert assigned
    locations = inner.fleet.locations()
    for agent_id in inner.fleet.agents:
        task = assigned.get(agent_id)
        target = locations[agent_id] if task is None else task.pickup_vertex
        expected = env.graph.distance(locations[agent_id], target)
        assert (expected > 0) == (task is not None)
        assert _own_eta(obs, str(agent_id)) == pytest.approx(expected)


def test_every_observation_shows_the_task_assigned_at_its_own_timestep():
    # A task assigned at t must already be in the observation of t, the one the robot
    # acts on. The assignment is read back from the task list after the next step.
    env = _env(arrival_rate=0.3, horizon=40)
    obs, _ = env.reset()
    inner = env.core
    checked = 0
    for _ in range(40):
        t, locations = inner.t, inner.fleet.locations()
        seen = obs
        obs, *_ = env.step({aid: 0 for aid in env.agents})
        for task in inner.tasks:
            if task.assignment_time == t:
                assert task.assigned_agent is not None
                expected = env.graph.distance(locations[task.assigned_agent], task.pickup_vertex)
                assert expected > 0
                assert _own_eta(seen, str(task.assigned_agent)) == pytest.approx(expected)
                checked += 1
    assert checked > 1  # the assignment of reset and at least one later one


# --- the three arms --------------------------------------------------------------------


def _movable(graph) -> frozenset:
    return frozenset(v for v, vertex in graph.vertices.items() if vertex.role.movable)


def _expected_window(env: WarehouseMAPDMultiAgentEnv, location, zones) -> frozenset:
    if env.controller == "decentralised":
        return frozenset(local_subgraph(env.graph, location, 3))
    if env.controller == "centralised":
        return _movable(env.graph)
    zone = zones[zone_of(zones)[location]]
    return zone | halo(env.graph, zone, 3)


def _expected_pool(env: WarehouseMAPDMultiAgentEnv, location, zones) -> frozenset:
    if env.controller == "centralised":
        return _movable(env.graph)
    return zones[zone_of(zones)[location]]


@pytest.mark.parametrize(
    "controller, pooling",
    [
        ("decentralised", "none"),
        ("centralised", "none"),
        ("centralised", "mean"),
        ("section", "none"),
        ("section", "mean"),
    ],
)
def test_each_arm_builds_and_its_observations_cover_its_window(controller, pooling):
    zones = _zones()
    env = _arm_env(controller, pooling, zones=zones if controller == "section" else None)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(5):
        locations = env.core.fleet.locations()
        for aid in env.agents:
            assert env.observation_spaces[aid].contains(obs[aid])
            view = obs[aid]["observations"]
            window = env.window_of(int(aid))
            assert frozenset(window) == _expected_window(env, locations[int(aid)], zones)
            assert view["node_mask"].sum() == len(window)
            assert ("pool_mask" in view) == (pooling != "none")
            if pooling != "none":
                pool = _expected_pool(env, locations[int(aid)], zones)
                assert view["pool_mask"].sum() == len(pool)
                assert np.all(view["pool_mask"] <= view["node_mask"])
        actions = {
            aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in env.agents
        }
        obs, *_ = env.step(actions)


def _robot_at_a_zone_boundary(env: WarehouseMAPDMultiAgentEnv, owner: dict):
    """A robot with a free outgoing neighbour in another zone, and the slot of that move."""
    locations = env.core.fleet.locations()
    occupied = set(locations.values())
    for agent_id, v in sorted(locations.items()):
        for slot in range(1, len(env.graph.legal_actions(v))):
            target = action_for_slot(env.graph, v, slot).target
            if owner[target] != owner[v] and target not in occupied:
                return agent_id, slot, target
    raise AssertionError("no robot stands next to a free vertex of another zone")


def test_the_section_window_follows_the_robot_into_the_next_zone():
    # Four robots fill the four endpoints, and endpoint 11 of zone 0 leads into zone 1.
    zones = _zones()
    owner = zone_of(zones)
    env = _arm_env("section", "mean", zones=zones, num_agents=4)
    env.reset()
    agent_id, slot, target = _robot_at_a_zone_boundary(env, owner)
    start = env.core.fleet.locations()[agent_id]
    before = env.window_of(agent_id)
    assert frozenset(before) == _expected_window(env, start, zones)

    actions = {aid: 0 for aid in env.agents}
    actions[str(agent_id)] = slot
    obs, *_ = env.step(actions)

    assert env.core.fleet.locations()[agent_id] == target
    after = env.window_of(agent_id)
    assert after != before
    assert frozenset(after) == _expected_window(env, target, zones)
    view = obs[str(agent_id)]["observations"]
    assert view["node_mask"].sum() == len(after)
    assert view["pool_mask"].sum() == len(zones[owner[target]])


def _random_rollout(env: WarehouseMAPDMultiAgentEnv, steps: int, seed: int):
    """(start locations, actions, rewards) of every step under a fixed action stream."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    record = []
    for _ in range(steps):
        start = env.core.fleet.locations()
        actions = {
            aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in env.agents
        }
        obs, rewards, *_ = env.step(actions)
        record.append((start, actions, rewards))
    return record


@pytest.mark.parametrize("credit_signal", ["per_agent", "team_mean"])
def test_the_logged_reward_terms_add_up_to_the_rewards_the_env_returned(credit_signal):
    controller = "decentralised" if credit_signal == "per_agent" else "centralised"
    env = _arm_env(
        controller, credit_signal=credit_signal, arrival_rate=3.0, horizon=60, num_agents=4
    )
    rollout = _random_rollout(env, 60, seed=3)
    log = env.core.log
    returned = {aid: sum(rewards[aid] for *_, rewards in rollout) for aid in env.agents}
    if credit_signal == "per_agent":
        for aid in env.agents:
            terms = (
                log.return_shaping_by_agent[int(aid)]
                + log.return_delivery_by_agent[int(aid)]
                + log.return_override_by_agent[int(aid)]
                + log.return_congestion_by_agent[int(aid)]
            )
            assert terms == pytest.approx(returned[aid], abs=1e-9)
    summed_terms = sum(
        sum(sums.values())
        for sums in (
            log.return_shaping_by_agent,
            log.return_delivery_by_agent,
            log.return_override_by_agent,
            log.return_congestion_by_agent,
        )
    )
    assert summed_terms == pytest.approx(sum(returned.values()), abs=1e-9)
    # the run is not vacuous, every term but the congestion one has something in it
    overrides = sum(log.overrides_by_agent.values())
    assert overrides > 0
    assert sum(log.return_override_by_agent.values()) == pytest.approx(-0.5 * overrides)
    assert sum(log.return_delivery_by_agent.values()) > 0
    assert sum(log.return_shaping_by_agent.values()) != 0.0
    assert sum(log.return_congestion_by_agent.values()) < 0.0


def test_team_mean_gives_every_robot_the_fleet_mean_under_the_centralised_arm():
    own = _random_rollout(_arm_env("centralised", arrival_rate=3.0), 15, seed=1)
    team = _random_rollout(
        _arm_env("centralised", credit_signal="team_mean", arrival_rate=3.0), 15, seed=1
    )
    assert any(len(set(rewards.values())) > 1 for _, _, rewards in own)
    for (_, actions, rewards), (_, team_actions, team_rewards) in zip(own, team):
        assert actions == team_actions
        mean = sum(rewards.values()) / len(rewards)
        assert team_rewards == {aid: pytest.approx(mean) for aid in rewards}


def test_team_mean_under_the_section_arm_averages_over_the_zone_at_the_start_of_the_step():
    zones = _zones()
    owner = zone_of(zones)
    kwargs = dict(zones=zones, num_agents=4, arrival_rate=3.0)
    own_env = _arm_env("section", **kwargs)
    team_env = _arm_env("section", credit_signal="team_mean", **kwargs)
    rewards_by_env = []
    for env in (own_env, team_env):
        env.reset()
        agent_id, slot, _ = _robot_at_a_zone_boundary(env, owner)
        start = env.core.fleet.locations()
        actions = {aid: 0 for aid in env.agents}
        actions[str(agent_id)] = slot
        _, rewards, *_ = env.step(actions)
        rewards_by_env.append(rewards)
    own, team = rewards_by_env

    end = own_env.core.fleet.locations()
    assert owner[end[agent_id]] != owner[start[agent_id]]  # the robot crossed a boundary

    def zone_means(locations):
        groups: dict[int, list[float]] = {}
        for aid, v in locations.items():
            groups.setdefault(owner[v], []).append(own[str(aid)])
        return {
            str(aid): sum(groups[owner[v]]) / len(groups[owner[v]])
            for aid, v in locations.items()
        }

    expected = zone_means(start)
    assert team == {aid: pytest.approx(value) for aid, value in expected.items()}
    # The zone at the end of the step would give the crossing robot another mean.
    assert zone_means(end)[str(agent_id)] != pytest.approx(expected[str(agent_id)])


def test_team_mean_with_the_decentralised_arm_is_rejected():
    with pytest.raises(ValueError, match="team_mean"):
        _arm_env("decentralised", credit_signal="team_mean")


def test_pooling_with_the_decentralised_arm_is_rejected():
    with pytest.raises(ValueError, match="pooling"):
        _arm_env("decentralised", "mean")


def test_the_section_arm_without_zones_is_rejected():
    with pytest.raises(ValueError, match="zones"):
        _env(controller="section", max_local_nodes=_WIDE_MAX_LOCAL_NODES)


@pytest.mark.parametrize("controller", ["decentralised", "centralised"])
def test_zones_for_another_arm_are_rejected(controller):
    with pytest.raises(ValueError, match="zones"):
        _env(controller=controller, zones=_zones(), max_local_nodes=_WIDE_MAX_LOCAL_NODES)


def test_zones_that_do_not_cover_the_movable_vertices_are_rejected():
    first, _ = _zones()
    with pytest.raises(ValueError, match="cover"):
        _arm_env("section", zones=(first,))


@pytest.mark.parametrize("controller", ["centralised", "section"])
def test_a_window_above_the_cap_is_rejected_when_the_env_is_built(controller):
    # Both arms have a window of more than 20 vertices on _graph(), and none is truncated.
    with pytest.raises(ValueError, match="max_local_nodes"):
        _arm_env(controller, max_local_nodes=_MAX_LOCAL_NODES)


def test_a_planner_config_is_rejected():
    with pytest.raises(ValueError, match="planner"):
        _env(controller=None, planner="token_passing")
