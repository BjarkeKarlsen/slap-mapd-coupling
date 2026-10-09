"""Unit tests for slap_mapd_coupling.training.config."""

import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.config import (
    PPOHyperparameters,
    SeedSplitConfig,
    build_ppo_config,
    split_seeds,
)


def _graph():
    params = GeneratorParams(
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
    return generate_warehouse_graph(params)


def _instance(graph):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return skus, capacities, counts


def _config(**overrides):
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=1.0,
        seed=7,
        horizon=20,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _ppo_hp(**overrides) -> PPOHyperparameters:
    defaults = dict(
        lr=3e-4,
        clip_param=0.2,
        gae_lambda=0.95,
        minibatch_size=32,
        num_epochs=2,
        rollout_fragment_length=20,
        entropy_coeff=0.0,
        vf_loss_coeff=1.0,
        train_batch_size=60,
        credit_signal="per_agent",
    )
    defaults.update(overrides)
    return PPOHyperparameters(**defaults)


def _seed_split(**overrides) -> SeedSplitConfig:
    defaults = dict(base_seed=100, train_seed_count=5, eval_seed_count=3)
    defaults.update(overrides)
    return SeedSplitConfig(**defaults)


def test_split_seeds_produces_disjoint_contiguous_ranges():
    config = SeedSplitConfig(base_seed=100, train_seed_count=5, eval_seed_count=3)
    train_seeds, eval_seeds = split_seeds(config)
    assert train_seeds == (100, 101, 102, 103, 104)
    assert eval_seeds == (105, 106, 107)
    assert set(train_seeds).isdisjoint(eval_seeds)


def test_ppo_hyperparameters_requires_every_field_no_defaults():
    with pytest.raises(Exception):
        PPOHyperparameters()  # type: ignore[call-arg]


def test_build_ppo_config_rejects_mismatched_num_actions():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    config = _config()
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8)
    wrong_head_config = PolicyValueHeadConfig(hidden_width=8, num_actions=999)

    with pytest.raises(ValueError, match="num_actions"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            config,
            enc_config,
            gnn_config,
            wrong_head_config,
            _ppo_hp(),
            _seed_split(),
        )


def test_build_ppo_config_rejects_missing_discount():
    """ExperimentConfig itself already requires discount for
    controller='decentralised', so model_construct (which skips that
    validator) stands in for a caller bug that slips a config through
    some other path."""
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    config = _config().model_construct(**{**_config().model_dump(), "discount": None})
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8)
    head_config = PolicyValueHeadConfig(hidden_width=8, num_actions=max_out_degree(graph) + 1)

    with pytest.raises(ValueError, match="discount"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            config,
            enc_config,
            gnn_config,
            head_config,
            _ppo_hp(),
            _seed_split(),
        )


def test_build_ppo_config_rejects_team_mean_credit_for_decentralised():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8)
    head_config = PolicyValueHeadConfig(hidden_width=8, num_actions=max_out_degree(graph) + 1)

    with pytest.raises(ValueError, match="credit_signal"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            _config(),
            enc_config,
            gnn_config,
            head_config,
            _ppo_hp(credit_signal="team_mean"),
            _seed_split(),
        )


def test_build_ppo_config_builds_a_valid_ppo_config():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    config = _config()
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8)
    d_max = max_out_degree(graph)
    head_config = PolicyValueHeadConfig(hidden_width=8, num_actions=d_max + 1)

    ppo_config = build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        config,
        enc_config,
        gnn_config,
        head_config,
        _ppo_hp(),
        _seed_split(base_seed=42, train_seed_count=3, eval_seed_count=2),
        num_env_runners=0,
    )

    assert ppo_config.gamma == pytest.approx(0.99)
    assert ppo_config.lr == pytest.approx(3e-4)
    assert ppo_config.clip_param == pytest.approx(0.2)
    assert ppo_config.rollout_fragment_length == 20
    assert ppo_config.seed == 42
