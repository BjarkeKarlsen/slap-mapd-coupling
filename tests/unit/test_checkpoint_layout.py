"""A saved checkpoint loads through the path the examples use.

04b_evaluate_checkpoint.py loads the module from the learner folder of an RLlib
checkpoint, with the path built from Ray's public constants. A Ray upgrade that moves
the layout has to fail here and not in the middle of an evaluation.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.config import (
    DEFAULT_POLICY_ID,
    PPOHyperparameters,
    SeedSplitConfig,
    build_ppo_config,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "04_decentralised_training.py"

# The largest field of view of this graph at d_obs = 3 has 20 vertices.
_MAX_LOCAL_NODES = 20


@pytest.fixture(scope="module")
def example_04() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_example_04_checkpoint", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_algo():
    graph = generate_warehouse_graph(
        GeneratorParams(
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
    )
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    config = ExperimentConfig(
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
    hyperparameters = PPOHyperparameters(
        lr=3e-4,
        clip_param=0.2,
        gae_lambda=0.95,
        minibatch_size=32,
        num_epochs=2,
        rollout_fragment_length=20,
        entropy_coeff=0.0,
        vf_loss_coeff=1.0,
        train_batch_size=60,
        training_budget_env_steps=60,
        credit_signal="per_agent",
    )
    ppo_config = build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        config,
        LocalSubgraphEncodingConfig(max_local_nodes=_MAX_LOCAL_NODES, max_messages=2),
        GNNEncoderConfig(num_rounds=1, hidden_width=8, communication=False, pooling="none"),
        PolicyValueHeadConfig(hidden_width=8),
        hyperparameters,
        SeedSplitConfig(base_seed=100, train_seed_count=2, eval_seed_count=1),
    )
    return ppo_config.build_algo()


def test_a_saved_checkpoint_loads_through_the_path_the_examples_use(example_04, tmp_path):
    algo = _tiny_algo()
    try:
        saved = Path(algo.save_to_path(tmp_path / "checkpoint"))
        trained = algo.get_module(DEFAULT_POLICY_ID)
        loaded = example_04.load_module(saved)
        trained_parameters = dict(trained.named_parameters())
        loaded_parameters = dict(loaded.named_parameters())
        assert loaded_parameters.keys() == trained_parameters.keys()
        for name, value in trained_parameters.items():
            assert torch.equal(loaded_parameters[name], value), name
    finally:
        algo.stop()

