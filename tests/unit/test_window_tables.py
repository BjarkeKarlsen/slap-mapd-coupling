"""Unit tests for slap_mapd_coupling.models.window_tables.

The compact sample and the tables must give back the padded sample of
models/local_subgraph_encoding.py exactly, for every arm, so the encoder and every
result are unchanged.
"""

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import slap_mapd_coupling.training.env as env_module
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.models.gnn_encoder import SENDER_KEYS
from slap_mapd_coupling.models.local_subgraph_encoding import encoded_observation_space
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.models.window_tables import compact_rllib_dict, expand_compact
from slap_mapd_coupling.training import policy as policy_module
from tests.unit.test_training_env import _arm_env, _env, _zones

# (controller, pooling, communication)
ARMS = [
    ("decentralised", "none", False),
    ("decentralised", "none", True),
    ("centralised", "mean", False),
    ("section", "mean", False),
    ("full_view", "mean", False),
]


def _adapter(controller, pooling, communication, **kwargs):
    if controller == "decentralised":
        return _env(communication=communication, **kwargs)
    return _arm_env(controller, pooling, **kwargs)


def _record_samples(monkeypatch, env, sub_steps: int, seed: int = 0):
    """Plays random legal sub-steps and returns every (padded, compact) pair. The adapter
    builds its samples without the window arrays (padded=False), so each step is also
    built padded, from the same state, for the comparison."""
    pairs = []
    padded_items: dict = {}
    real_encode = env_module.encode_all_observations

    def encoding(*args, **kwargs):
        padded_items.clear()
        padded_items.update(real_encode(*args, **{**kwargs, "padded": True}))
        return real_encode(*args, **kwargs)

    def recording(item, position, *keys):
        compact = compact_rllib_dict(item, position, *keys)
        location = item.location
        (padded,) = [p for p in padded_items.values() if p.location == location]
        pairs.append((padded.as_rllib_dict(), compact))
        return compact

    monkeypatch.setattr(env_module, "encode_all_observations", encoding)
    monkeypatch.setattr(env_module, "compact_rllib_dict", recording)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(sub_steps):
        actions = {aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in obs}
        obs, *_ = env.step(actions)
    return pairs


@pytest.mark.parametrize("controller, pooling, communication", ARMS)
def test_the_lookup_rebuilds_the_padded_sample_bit_for_bit(
    monkeypatch, controller, pooling, communication
):
    env = _adapter(controller, pooling, communication, num_agents=3, horizon=20, arrival_rate=2.0)
    pairs = _record_samples(monkeypatch, env, sub_steps=30)
    assert len(pairs) > 30
    for padded, compact in pairs:
        np.testing.assert_array_equal(compact["action_mask"], padded["action_mask"])
        rebuilt = expand_compact(env.window_tables, compact["observations"], pooling, communication)
        # With communication on the senders stay keys, which the module comparison below checks.
        expected_keys = set(padded["observations"]) - set(SENDER_KEYS)
        if communication:
            expected_keys |= {"row_key", "sender_row_key"}
            heard = int(padded["observations"]["sender_mask"].sum())
            assert (compact["observations"]["sender_row_key"] >= 0).sum() == heard
        assert set(rebuilt) == expected_keys
        for key, value in padded["observations"].items():
            if key in SENDER_KEYS:
                continue
            expected = np.asarray(value)
            assert rebuilt[key].dtype == expected.dtype or key == "own_index", key
            np.testing.assert_array_equal(rebuilt[key], expected, err_msg=key)


def test_the_compact_sample_stores_the_flags_as_uint8_and_nothing_per_window():
    env = _arm_env("centralised", "mean", num_agents=3)
    obs, _ = env.reset(seed=0)
    sample = obs[next(iter(obs))]["observations"]
    assert sample["flags"].dtype == np.uint8
    assert sample["eta"].dtype == np.float32
    assert not {"edge_index", "edge_mask", "node_mask", "pool_mask", "node_features"} & set(sample)


def test_the_tables_hold_one_window_per_distinct_window():
    zones = _zones()
    centralised = _arm_env("centralised", "mean").window_tables
    section = _arm_env("section", "mean", zones=zones).window_tables
    decentralised = _env().window_tables
    assert len(centralised.node_mask) == 1
    assert len(section.node_mask) == len(zones)
    assert len(decentralised.node_mask) == len(decentralised.vertices)
    assert centralised.vertices == section.vertices == decentralised.vertices


def _modules(env, pooling, communication):
    """An untrained module on compact samples and one on padded samples, with the same weights."""
    encoder_config = {
        "num_rounds": 2,
        "hidden_width": 8,
        "communication": communication,
        "pooling": pooling,
    }
    head_config = {"hidden_width": 8}
    compact = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=env.observation_spaces["1"],
        action_space=env.action_spaces["1"],
        model_config={
            "gnn_encoder_config": encoder_config,
            "policy_value_head_config": head_config,
            "window_tables": env.window_tables.as_arrays(),
        },
    ).build()
    d_max = max_out_degree(env.graph)
    padded_space = masked_observation_space(
        encoded_observation_space(env._encoding_config, d_max, communication, pooling), d_max + 1
    )
    padded = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=padded_space,
        action_space=env.action_spaces["1"],
        model_config={
            "gnn_encoder_config": encoder_config,
            "policy_value_head_config": head_config,
        },
    ).build()
    padded.load_state_dict(compact.state_dict())
    return compact, padded


@pytest.mark.parametrize("controller, pooling, communication", ARMS)
def test_the_module_gives_the_same_logits_and_values_on_both_layouts(
    monkeypatch, controller, pooling, communication
):
    torch.manual_seed(0)
    env = _adapter(controller, pooling, communication, num_agents=3, horizon=10, arrival_rate=2.0)
    pairs = _record_samples(monkeypatch, env, sub_steps=12)
    compact_module, padded_module = _modules(env, pooling, communication)

    padded_batch = policy_module._batch([padded for padded, _ in pairs])
    compact_batch = policy_module._batch([compact for _, compact in pairs])
    with torch.no_grad():
        expected = padded_module.forward_inference(padded_batch)[Columns.ACTION_DIST_INPUTS]
        actual = compact_module.forward_inference(compact_batch)[Columns.ACTION_DIST_INPUTS]
        expected_values = padded_module.compute_values(policy_module._batch([p for p, _ in pairs]))
        actual_values = compact_module.compute_values(policy_module._batch([c for _, c in pairs]))
    assert torch.equal(actual, expected)
    assert torch.equal(actual_values, expected_values)


def test_the_tables_are_not_in_the_state_dict():
    env = _arm_env("centralised", "mean", num_agents=3)
    module, _ = _modules(env, "mean", False)
    assert not any("window_of" in key or "edge_index" in key for key in module.state_dict())
