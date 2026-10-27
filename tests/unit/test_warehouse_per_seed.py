"""A new warehouse every episode (step 12 of the model revision, D27). The episode seed draws
the warehouse and the start stock, the window tables of every warehouse are stored on disk, and
the module reads the tables of the warehouses a batch holds. Samples from two warehouses in one
batch must give the logits and values the padded samples give."""

from pathlib import Path

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import slap_mapd_coupling.training.env as env_module
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.models.local_subgraph_encoding import encoded_observation_space
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.models.window_tables import VERTEX_KEY_STRIDE, compact_rllib_dict
from slap_mapd_coupling.training import policy as policy_module
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
    seed_spec,
)

ROOT = Path(__file__).resolve().parents[2]
STEP12 = ROOT / "configs" / "step12" / "fleet_8.yaml"


def _spec(store: Path, controller: str = "decentralised") -> TrainingSpec:
    data = load_spec(STEP12).model_dump()
    data.update(
        warehouse_per_seed=True,
        warehouse_store=str(store),
        train_seed_count=3,
        validation_seed_count=1,
        test_seed_count=1,
        horizon=30,
        controller=controller,
    )
    return TrainingSpec.model_validate(data)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    spec = _spec(tmp_path_factory.mktemp("warehouses"))
    return spec, build_study(spec)


def _adapter(spec, study, seeds):
    config = build_experiment_config(spec)
    return WarehouseMAPDMultiAgentEnv(
        study.instance.graph, study.skus, study.counts, study.capacities, config,
        study.arm.encoding, seeds, pooling=study.arm.pooling, zones=study.arm.zones,
        pool=study.pool,
    )  # fmt: skip


def test_each_seed_draws_its_own_warehouse(study):
    spec, built = study
    pool = built.pool
    seeds = split_seeds(built.seed_split)
    first, second = (pool.warehouse(seed) for seed in seeds.train[:2])
    assert first.instance.graph != second.instance.graph
    assert first.counts != second.counts
    assert pool.warehouse(seeds.train[0]) is first
    again = seed_spec(spec, seeds.train[0])
    assert again.instance["seed"] == again.storage.seed == seeds.train[0]


def test_the_pool_stores_every_seed_of_the_split_and_pads_them_alike(study):
    _, built = study
    pool = built.pool
    seeds = split_seeds(built.seed_split)
    manifest = pool.manifest()
    every = [*seeds.train, *seeds.validation, *seeds.test]
    assert manifest["seeds"] == sorted(every)
    shapes = {pool.tables(seed).node_mask.shape[1] for seed in every}
    assert shapes == {manifest["max_local_nodes"]} == {built.arm.encoding.max_local_nodes}


def test_a_reset_plays_the_warehouse_of_its_seed(study):
    spec, built = study
    seeds = split_seeds(built.seed_split).train
    env = _adapter(spec, built, seeds)
    for seed in seeds[:2]:
        obs, _ = env.reset(seed=seed)
        assert env.graph is built.pool.warehouse(seed).instance.graph
        assert env.core.graph is env.graph
        keys = [int(sample["observations"]["vertex"]) for sample in obs.values()]
        assert {key // VERTEX_KEY_STRIDE for key in keys} == {seed}


def _record(monkeypatch, env, seeds, sub_steps):
    """(padded, compact) sample pairs from episodes on each seed in turn."""
    pairs = []
    padded_items: dict = {}
    real_encode = env_module.encode_all_observations

    def encoding(*args, **kwargs):
        padded_items.clear()
        padded_items.update(real_encode(*args, **{**kwargs, "padded": True}))
        return real_encode(*args, **kwargs)

    def recording(item, position, *keys):
        compact = compact_rllib_dict(item, position, *keys)
        (padded,) = [p for p in padded_items.values() if p.location == item.location]
        pairs.append((padded.as_rllib_dict(), compact))
        return compact

    monkeypatch.setattr(env_module, "encode_all_observations", encoding)
    monkeypatch.setattr(env_module, "compact_rllib_dict", recording)
    rng = np.random.default_rng(0)
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        for _ in range(sub_steps):
            actions = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in obs}
            obs, *_ = env.step(actions)
    return pairs


@pytest.fixture(scope="module", params=["decentralised", "centralised", "section"])
def arm_study(request, tmp_path_factory):
    spec = _spec(tmp_path_factory.mktemp("warehouses"), request.param)
    return spec, build_study(spec)


def test_the_module_reads_two_warehouses_in_one_batch_as_padded_samples(monkeypatch, arm_study):
    torch.manual_seed(0)
    spec, built = arm_study
    pooling = built.arm.pooling
    seeds = split_seeds(built.seed_split).train[:2]
    env = _adapter(spec, built, seeds)
    pairs = _record(monkeypatch, env, seeds, sub_steps=6)
    assert {int(c["observations"]["vertex"]) // VERTEX_KEY_STRIDE for _, c in pairs} == set(seeds)

    encoder = {"num_rounds": 2, "hidden_width": 8, "communication": False, "pooling": pooling}
    head = {"hidden_width": 8}
    stored = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=env.observation_spaces["1"],
        action_space=env.action_spaces["1"],
        model_config={
            "gnn_encoder_config": encoder,
            "policy_value_head_config": head,
            "window_store": built.pool.as_config(),
        },
    ).build()
    d_max = built.pool.manifest()["d_max"]
    padded = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=masked_observation_space(
            encoded_observation_space(built.arm.encoding, d_max, False, pooling), d_max + 1
        ),
        action_space=env.action_spaces["1"],
        model_config={"gnn_encoder_config": encoder, "policy_value_head_config": head},
    ).build()
    padded.load_state_dict(stored.state_dict())
    with torch.no_grad():
        expected = padded.forward_inference(policy_module._batch([p for p, _ in pairs]))
        actual = stored.forward_inference(policy_module._batch([c for _, c in pairs]))
        values = stored.compute_values(policy_module._batch([c for _, c in pairs]))
        expected_values = padded.compute_values(policy_module._batch([p for p, _ in pairs]))
    assert torch.equal(
        actual[Columns.ACTION_DIST_INPUTS], expected[Columns.ACTION_DIST_INPUTS]
    )
    assert torch.equal(values, expected_values)


def test_without_a_warehouse_per_seed_nothing_changes(tmp_path):
    data = {**load_spec(STEP12).model_dump(), "warehouse_per_seed": False}
    plain = build_study(TrainingSpec.model_validate(data))
    assert plain.pool is None
