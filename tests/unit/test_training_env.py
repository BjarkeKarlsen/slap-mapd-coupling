"""Unit tests for slap_mapd_coupling.training.env."""

import numpy as np
import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv


def _graph():
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=4,
        seed=0,
    )
    return generate_warehouse_graph(params)


def _instance(graph):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return skus, capacities, counts


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
    train_seeds: tuple[int, ...] = (100, 101, 102), **config_overrides
) -> WarehouseMAPDMultiAgentEnv:
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    return WarehouseMAPDMultiAgentEnv(
        graph, skus, counts, capacities, _config(**config_overrides), enc_config, train_seeds
    )


def test_rejects_non_decentralised_controller():
    # observation_depth is required for every controller now (sec:pf:env),
    # unlike the reward fields below, which stay None for a non-learned one.
    with pytest.raises(ValueError, match="decentralised"):
        _env(
            controller="centralised",
            discount=None,
            deliver_reward=None,
            override_penalty=None,
            congestion_reward_weight=None,
        )


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


def test_step_with_wait_actions_keeps_agents_stationary():
    env = _env()
    env.reset()
    before = env._env.fleet.locations()
    for _ in range(5):
        actions = {aid: 0 for aid in env.agents}  # slot 0 is always wait
        obs, rewards, terminated, truncated, infos = env.step(actions)
    assert env._env.fleet.locations() == before
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
