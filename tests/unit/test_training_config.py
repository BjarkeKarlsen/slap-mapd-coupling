"""Unit tests for slap_mapd_coupling.training.config."""

import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.environment.zones import zones_by_aisle
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_instance,
    generate_warehouse_graph,
)
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.config import (
    PPOHyperparameters,
    SeedSplitConfig,
    build_ppo_config,
    split_seeds,
)


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


def _zones():
    instance = generate_instance(_params())
    return zones_by_aisle(instance.graph, instance.aisle_index, 2)


# _graph() has 30 movable vertices, the centralised window and the largest section window.
_WIDE_MAX_LOCAL_NODES = 30


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
        entropy_coeff=0.0,
        vf_loss_coeff=1.0,
        training_budget_env_steps=120,
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


def test_training_budget_has_no_default_and_must_be_positive():
    # The thesis leaves the budget TBD (tab:trainparams), so no number is picked here.
    fields = {**_ppo_hp().model_dump()}
    del fields["training_budget_env_steps"]
    with pytest.raises(ValueError, match="training_budget_env_steps"):
        PPOHyperparameters(**fields)
    with pytest.raises(ValueError, match="training_budget_env_steps"):
        _ppo_hp(training_budget_env_steps=0)


def test_build_ppo_config_rejects_a_cap_too_small_for_a_vertex_and_its_neighbours():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    assert max_out_degree(graph) + 1 > 2
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=2, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none")

    with pytest.raises(ValueError, match="max_local_nodes"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            _config(),
            enc_config,
            gnn_config,
            PolicyValueHeadConfig(hidden_width=8),
            _ppo_hp(),
            _seed_split(),
        )


def test_build_ppo_config_rejects_a_neighbour_outside_the_field_of_view():
    graph = _graph(random_edge_costs=True, edge_cost_range=(1.5, 2.0))
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none")

    with pytest.raises(ValueError, match="observation_depth"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            _config(observation_depth=1),
            enc_config,
            gnn_config,
            PolicyValueHeadConfig(hidden_width=8),
            _ppo_hp(),
            _seed_split(),
        )


def test_the_head_config_has_no_action_count():
    """The head's size follows the observation's slot arrays."""
    with pytest.raises(ValueError):
        PolicyValueHeadConfig(hidden_width=8, num_actions=5)  # type: ignore[call-arg]


@pytest.mark.parametrize("experiment_communication", [False, True])
def test_build_ppo_config_rejects_a_communication_mismatch(experiment_communication):
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    config = _config(communication=experiment_communication)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=2)
    gnn_config = GNNEncoderConfig(
        num_rounds=1, hidden_width=8, communication=not experiment_communication, pooling="none"
    )
    head_config = PolicyValueHeadConfig(hidden_width=8)

    with pytest.raises(ValueError, match="communication"):
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


def test_build_ppo_config_accepts_communication_when_both_switches_agree():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    # the largest field of view of _graph() at d_obs = 3 has 20 vertices
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=20, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=True, pooling="none")
    head_config = PolicyValueHeadConfig(hidden_width=8)

    ppo_config = build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        _config(communication=True),
        enc_config,
        gnn_config,
        head_config,
        _ppo_hp(),
        _seed_split(),
    )
    assert ppo_config.gamma == pytest.approx(0.99)


def test_build_ppo_config_rejects_communication_with_no_room_for_a_sender():
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=0)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=True, pooling="none")
    head_config = PolicyValueHeadConfig(hidden_width=8)

    with pytest.raises(ValueError, match="max_messages"):
        build_ppo_config(
            graph,
            skus,
            counts,
            capacities,
            _config(communication=True),
            enc_config,
            gnn_config,
            head_config,
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
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none")
    head_config = PolicyValueHeadConfig(hidden_width=8)

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
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none")
    head_config = PolicyValueHeadConfig(hidden_width=8)

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
    # the largest field of view of _graph() at d_obs = 3 has 20 vertices
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=20, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none")
    head_config = PolicyValueHeadConfig(hidden_width=8)

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
    assert ppo_config.rollout_fragment_length == _config().horizon
    assert ppo_config.seed == 42


def test_ppo_hyperparameters_have_no_rollout_length_and_no_batch_size():
    # Both follow from the horizon, and extra="forbid" rejects one passed by an old caller.
    for name in ("rollout_fragment_length", "train_batch_size"):
        assert name not in PPOHyperparameters.model_fields
        with pytest.raises(ValueError, match=name):
            _ppo_hp(**{name: 20})


@pytest.mark.parametrize("num_env_runners", [0, 2, 7])
def test_build_ppo_config_samples_one_episode_per_runner_per_iteration(num_env_runners):
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    config = _config()
    ppo_config = build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        config,
        LocalSubgraphEncodingConfig(max_local_nodes=20, max_messages=2),
        GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none"),
        PolicyValueHeadConfig(hidden_width=8),
        _ppo_hp(),
        _seed_split(base_seed=42, train_seed_count=3, eval_seed_count=2),
        num_env_runners=num_env_runners,
    )
    assert ppo_config.rollout_fragment_length == config.horizon
    assert ppo_config.total_train_batch_size == max(num_env_runners, 1) * config.horizon
    # RLlib's own check of the batch against the fragment length, which rejects most
    # other pairings.
    ppo_config.validate_train_batch_size_vs_rollout_fragment_length()


def _build_for_arm(controller: str, pooling: str, credit_signal: str = "per_agent", **kwargs):
    """build_ppo_config for one arm, with zones for the section arm."""
    graph = _graph()
    skus, capacities, counts = _instance(graph)
    max_local_nodes = kwargs.pop("max_local_nodes", _WIDE_MAX_LOCAL_NODES)
    zones = kwargs.pop("zones", _zones() if controller == "section" else None)
    train_seed_count = kwargs.pop("train_seed_count", 2)
    return build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        _config(controller=controller),
        LocalSubgraphEncodingConfig(max_local_nodes=max_local_nodes, max_messages=2),
        GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling=pooling),
        PolicyValueHeadConfig(hidden_width=8),
        _ppo_hp(credit_signal=credit_signal),
        _seed_split(train_seed_count=train_seed_count),
        zones=zones,
        **kwargs,
    )


@pytest.mark.parametrize("controller", ["centralised", "section"])
@pytest.mark.parametrize("credit_signal", ["per_agent", "team_mean"])
def test_build_ppo_config_accepts_the_wider_arms_with_either_credit_signal(
    controller, credit_signal
):
    ppo_config = _build_for_arm(controller, "mean", credit_signal)
    assert ppo_config.gamma == pytest.approx(0.99)


def test_build_ppo_config_rejects_pooling_with_the_decentralised_arm():
    # The module would expect a pool_mask that the decentralised observations never carry.
    with pytest.raises(ValueError, match="gnn_encoder_config.pooling='mean'"):
        _build_for_arm("decentralised", "mean", max_local_nodes=20)


def test_build_ppo_config_rejects_the_section_arm_without_zones():
    with pytest.raises(ValueError, match="zones"):
        _build_for_arm("section", "mean", zones=None)


@pytest.mark.parametrize("controller", ["centralised", "section"])
def test_build_ppo_config_rejects_a_window_above_the_cap(controller):
    with pytest.raises(ValueError, match="max_local_nodes"):
        _build_for_arm(controller, "mean", max_local_nodes=20)


@pytest.mark.parametrize(
    "controller, pooling",
    [("decentralised", "none"), ("centralised", "mean"), ("section", "mean")],
)
def test_one_ppo_iteration_runs_for_each_controller(controller, pooling):
    # The largest field of view of _graph() at d_obs = 3 has 20 vertices.
    cap = 20 if controller == "decentralised" else _WIDE_MAX_LOCAL_NODES
    # The batch is one episode per runner, here 20 steps. A test makes it three episodes.
    ppo_config = _build_for_arm(controller, pooling, max_local_nodes=cap).training(
        train_batch_size=60
    )
    algo = ppo_config.build_algo()
    try:
        result = algo.train()
    finally:
        algo.stop()
    assert result["num_env_steps_sampled_lifetime"] >= 60
    assert "shared_policy" in result["learners"]


def _registered_env_creator(train_seed_count: int = 5):
    """The env_creator that build_ppo_config registers, as RLlib calls it."""
    from ray.tune.registry import ENV_CREATOR, _global_registry

    from slap_mapd_coupling.training.config import _ENV_NAME

    _build_for_arm("decentralised", "none", max_local_nodes=20, train_seed_count=train_seed_count)
    return _global_registry.get(ENV_CREATOR, _ENV_NAME)


def _first_seed_per_worker(creator, worker_indices) -> list[int]:
    """The seed of the first episode of an env built for each worker index,
    with the EnvContext RLlib passes (no Ray is started)."""
    from ray.rllib.env.env_context import EnvContext

    seeds = []
    for worker_index in worker_indices:
        env = creator(EnvContext({}, worker_index=worker_index, vector_index=0))
        env.reset()
        seeds.append(env.last_seed)
    return seeds


def test_runners_draw_different_first_training_seeds():
    """With several env runners one PPO batch must see several scenarios, so
    runners must not all start the training cycle at the same seed."""
    seeds = _first_seed_per_worker(_registered_env_creator(), range(5))
    assert seeds == [100, 101, 102, 103, 104]


def test_the_single_process_worker_starts_at_the_first_training_seed():
    """num_env_runners=0 builds the env for worker 0, which draws the cycle as before."""
    assert _first_seed_per_worker(_registered_env_creator(), [0]) == [100]


def test_more_runners_than_training_seeds_wrap_around_inside_the_training_split():
    creator = _registered_env_creator(train_seed_count=5)
    seeds = _first_seed_per_worker(creator, range(7))  # 7 runners, 5 seeds
    assert seeds == [100, 101, 102, 103, 104, 100, 101]
    assert set(seeds) <= set(range(100, 105))
    assert set(seeds).isdisjoint({105, 106, 107})  # the evaluation seeds


def test_first_seed_index_numbers_the_envs_of_all_runners_consecutively():
    from slap_mapd_coupling.training.config import first_seed_index

    assert first_seed_index(0, 0) == 0
    assert [first_seed_index(w, v, envs_per_runner=2) for w in range(3) for v in range(2)] == [
        0, 1, 2, 3, 4, 5,
    ]
