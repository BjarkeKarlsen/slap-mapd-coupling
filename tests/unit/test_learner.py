"""Unit tests for slap_mapd_coupling.training.learner and the keyed message path.

A row with communication names its senders by row key. These tests check that a
training minibatch that cuts a timestep, with its context rows, the chunked value pass
and the gradients all give what the same rows give with the senders' views stored in
the row (the padded layout of models/local_subgraph_encoding.py).
"""

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from slap_mapd_coupling.environment import action_masking
from slap_mapd_coupling.training import policy as policy_module
from slap_mapd_coupling.training.config import key_prefix
from slap_mapd_coupling.training.env import row_key
from slap_mapd_coupling.models.window_tables import MAX_ROW_KEY
from slap_mapd_coupling.training.learner import add_context_rows
from tests.unit.test_window_tables import _modules, _record_samples
from tests.unit.test_training_env import _env


@pytest.fixture
def recorded(monkeypatch):
    """Every (padded, compact) sample of a short episode with communication, and the two
    modules with the same weights."""
    torch.manual_seed(0)
    env = _env(communication=True, num_agents=3, horizon=12, arrival_rate=2.0)
    pairs = _record_samples(monkeypatch, env, sub_steps=12)
    compact_module, padded_module = _modules(env, "none", True)
    return pairs, compact_module, padded_module


def _obs(samples):
    """The batched observation of compact or padded samples, as the learner holds it."""
    return policy_module._batch(samples)[Columns.OBS]["observations"]


def _rows(obs, index):
    return {key: value[index] for key, value in obs.items()}


def _z(module, obs):
    return module.encoder({Columns.OBS: obs})["encoder_out"]["critic"]


def test_a_minibatch_that_cuts_a_timestep_gets_its_senders_as_context_rows(recorded):
    pairs, compact_module, padded_module = recorded
    full = _obs([compact for _, compact in pairs])
    padded = _obs([p for p, _ in pairs])
    sorted_keys, order = torch.sort(full["row_key"])
    # Every other row, so most senders sit outside the minibatch.
    index = torch.arange(0, len(full["row_key"]), 2)
    minibatch = _rows(full, index)
    with pytest.raises(ValueError, match="row key"):
        _z(compact_module, dict(minibatch))

    add_context_rows(minibatch, full, sorted_keys, order)
    assert set(minibatch["context_row_key"].tolist()).isdisjoint(minibatch["row_key"].tolist())
    with torch.no_grad():
        expected = _z(padded_module, _rows(padded, index))
        actual = _z(compact_module, minibatch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_the_gradient_through_context_rows_equals_the_padded_gradient(recorded):
    pairs, compact_module, padded_module = recorded
    full = _obs([compact for _, compact in pairs])
    padded = _obs([p for p, _ in pairs])
    sorted_keys, order = torch.sort(full["row_key"])
    index = torch.arange(1, len(full["row_key"]), 3)
    minibatch = _rows(full, index)
    add_context_rows(minibatch, full, sorted_keys, order)

    _z(padded_module, _rows(padded, index)).square().sum().backward()
    _z(compact_module, minibatch).square().sum().backward()
    padded_grads = dict(padded_module.named_parameters())
    for name, parameter in compact_module.named_parameters():
        expected = padded_grads[name].grad
        if expected is None:
            assert parameter.grad is None or not parameter.grad.any(), name
            continue
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("chunk_rows", [1, 4, 5, 1000])
def test_the_chunked_value_pass_equals_the_unchunked_one(recorded, monkeypatch, chunk_rows):
    pairs, compact_module, padded_module = recorded
    compacts = [compact for _, compact in pairs]
    with torch.no_grad():
        expected = padded_module.compute_values(policy_module._batch([p for p, _ in pairs]))
    monkeypatch.setattr(action_masking, "VALUE_CHUNK_ROWS", chunk_rows)
    with torch.no_grad():
        actual = compact_module.compute_values(policy_module._batch(compacts))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_a_context_row_missing_from_the_train_batch_raises(recorded):
    pairs, _, _ = recorded
    full = _obs([compact for _, compact in pairs])
    keep = full["row_key"] != full["sender_row_key"][full["sender_row_key"] >= 0][0]
    partial = _rows(full, keep)
    sorted_keys, order = torch.sort(partial["row_key"])
    minibatch = _rows(partial, torch.arange(len(partial["row_key"])))
    with pytest.raises(ValueError, match="not in the train batch"):
        add_context_rows(minibatch, partial, sorted_keys, order)


def test_row_keys_are_unique_across_runners_episodes_steps_and_robots():
    keys = {
        row_key(key_prefix(worker), episode, t, agent)
        for worker in range(3)
        for episode in range(3)
        for t in (0, 1, 2047)
        for agent in (1, 2, 64)
    }
    assert len(keys) == 3 * 3 * 3 * 3
    assert max(keys) <= MAX_ROW_KEY
    assert row_key(key_prefix(16383, 15), 65535, 2**20 - 1, 255) == MAX_ROW_KEY
    with pytest.raises(ValueError):
        row_key(0, 0, 0, 256)


def test_every_sample_of_a_step_is_keyed_by_that_step(monkeypatch):
    env = _env(communication=True, num_agents=3, horizon=6, arrival_rate=2.0)
    obs, _ = env.reset(seed=0)
    for _ in range(4):
        t = env.core.t
        keys = {aid: int(view["observations"]["row_key"]) for aid, view in obs.items()}
        assert keys == {aid: row_key(0, 0, t, int(aid)) for aid in obs}
        for view in obs.values():
            senders = view["observations"]["sender_row_key"]
            assert set(senders[senders >= 0].tolist()) <= set(keys.values())
        obs, *_ = env.step({aid: 0 for aid in obs})
    np.testing.assert_array_equal(env.core.t, 4)
