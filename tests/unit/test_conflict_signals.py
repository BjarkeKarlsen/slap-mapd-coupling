"""Unit tests for the conflict signals, an extension the thesis does not contain.

The two signals are the robot's own override flag of the step that just ended and its own rank
in the priority order sigma (environment/observation.py). They are behind a switch that is off
by default, so most tests here check that the thesis' observation is untouched with it off, and
that it adds exactly the two scalars, in the same place for every arm, with it on.
"""

import dataclasses
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.observation import (
    ConflictSignals,
    ConflictSignalsConfig,
    Observation,
    ObservationWithConflictSignals,
    build_observation,
    communication_neighbours,
    encode_priority_rank,
    local_subgraph,
)
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.environment.zones import zones_by_aisle
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_instance,
)
from slap_mapd_coupling.models.gnn_encoder import GNNEncoder, GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
    encoded_observation_space,
)
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.resolution.conflict_resolution import priority_permutation
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv
from slap_mapd_coupling.training.policy import ModulePolicy

ON = ConflictSignalsConfig(enabled=True)
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


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


# ---------------------------------------------------------------- the observation


def test_the_switch_is_off_by_default_and_the_encoder_config_carries_it():
    assert ConflictSignalsConfig().enabled is False
    assert ConflictSignalsConfig().rank_encoding == "normalised"
    config = GNNEncoderConfig(num_rounds=1, hidden_width=4, communication=False, pooling="none")
    assert config.conflict_signals.enabled is False
    # a config saved before the switch existed has no entry, and means off
    old = {"num_rounds": 1, "hidden_width": 4, "communication": False, "pooling": "none"}
    assert GNNEncoderConfig(**old) == config
    # and a saved config round-trips with the switch on
    on = GNNEncoderConfig(**old, conflict_signals=ON)
    assert GNNEncoderConfig(**on.model_dump()) == on


def test_without_the_switch_the_observation_is_the_plain_thesis_observation():
    graph = _line_graph(5)
    fleet = _fleet({1: 0, 2: 1, 3: 4})
    observation = build_observation(graph, fleet, [], 1, 0, _config())
    assert type(observation) is Observation
    assert {f.name for f in dataclasses.fields(observation)} == {
        "visible_vertices",
        "features",
        "congestion",
    }


def test_with_the_switch_the_observation_adds_exactly_the_two_scalars():
    graph = _line_graph(5)
    fleet = _fleet({1: 0, 2: 1, 3: 4})
    config = _config()
    plain = build_observation(graph, fleet, [], 1, 0, config)
    signals = ConflictSignals(priority=(3, 1, 2), overridden=frozenset({1}))
    extended = build_observation(graph, fleet, [], 1, 0, config, conflict_signals=signals)
    assert isinstance(extended, ObservationWithConflictSignals)
    assert {f.name for f in dataclasses.fields(extended)} - {
        f.name for f in dataclasses.fields(plain)
    } == {"overridden_last_step", "priority_rank"}
    # everything else is the plain observation
    assert extended.visible_vertices == plain.visible_vertices
    assert extended.features == plain.features
    assert extended.congestion == plain.congestion
    assert extended.overridden_last_step is True
    assert extended.priority_rank == pytest.approx(0.5)  # agent 1 is second of three


def test_the_rank_is_the_position_in_sigma_and_normalised_by_default():
    assert encode_priority_rank(0, 4, "normalised") == 0.0
    assert encode_priority_rank(3, 4, "normalised") == 1.0
    assert encode_priority_rank(1, 4, "normalised") == pytest.approx(1 / 3)
    assert encode_priority_rank(0, 1, "normalised") == 0.0  # one robot, no division by zero
    assert encode_priority_rank(2, 4, "raw") == 2.0
    with pytest.raises(ValueError, match="outside"):
        encode_priority_rank(4, 4, "normalised")


def test_the_rank_follows_the_permutation_conflict_resolution_draws():
    graph = _line_graph(8)
    fleet = _fleet({a: a for a in range(1, 6)})
    sigma = priority_permutation(11, tuple(range(1, 6)))
    signals = ConflictSignals(priority=sigma, overridden=frozenset())
    config = _config(num_agents=5)
    ranks = {
        a: build_observation(graph, fleet, [], a, 0, config, conflict_signals=signals).priority_rank
        for a in range(1, 6)
    }
    assert ranks == {a: sigma.index(a) / 4 for a in range(1, 6)}
    assert sorted(ranks.values()) == [0.0, 0.25, 0.5, 0.75, 1.0]  # rank 0 is the highest priority


def test_an_agent_outside_sigma_raises():
    graph = _line_graph(4)
    fleet = _fleet({1: 0, 2: 1})
    signals = ConflictSignals(priority=(2,), overridden=frozenset())
    with pytest.raises(ValueError, match="priority order"):
        build_observation(graph, fleet, [], 1, 0, _config(), conflict_signals=signals)


# ---------------------------------------------------------------- the padded arrays


def test_the_observation_space_is_unchanged_with_the_switch_off_and_gains_two_keys_with_it_on():
    enc = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    for communication in (False, True):
        for pooling in ("none", "mean"):
            base = encoded_observation_space(enc, 2, communication, pooling)
            explicit_off = encoded_observation_space(
                enc, 2, communication, pooling, ConflictSignalsConfig(enabled=False)
            )
            on = encoded_observation_space(enc, 2, communication, pooling, ON)
            assert explicit_off == base
            assert set(on.spaces) - set(base.spaces) == {"overridden_last_step", "priority_rank"}
            assert set(base.spaces) <= set(on.spaces)
            for key in base.spaces:
                assert on.spaces[key] == base.spaces[key]
            assert on.spaces["overridden_last_step"].shape == ()
            assert on.spaces["priority_rank"].shape == ()


def test_a_raw_rank_has_no_upper_bound_in_the_space():
    enc = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    raw = ConflictSignalsConfig(enabled=True, rank_encoding="raw")
    space = encoded_observation_space(enc, 2, False, "none", raw)
    assert space.spaces["priority_rank"].high == np.inf


def test_the_rllib_dict_has_the_two_keys_only_with_the_switch_on_and_lies_in_the_space():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(num_agents=2)
    enc = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    d_max = max_out_degree(graph)
    off = encode_all_observations(graph, fleet, [], 0, config, enc, d_max)
    signals = ConflictSignals(priority=(2, 1), overridden=frozenset({2}))
    on = encode_all_observations(graph, fleet, [], 0, config, enc, d_max, conflict_signals=signals)
    assert "overridden_last_step" not in off[1].as_rllib_dict()["observations"]
    assert "priority_rank" not in off[1].as_rllib_dict()["observations"]
    space = encoded_observation_space(enc, d_max, False, "none", ON)
    space_extra = {"overridden_last_step", "priority_rank"}
    for agent_id, flag, rank in ((1, 0.0, 1.0), (2, 1.0, 0.0)):
        observations = on[agent_id].as_rllib_dict()["observations"]
        assert observations["overridden_last_step"] == flag
        assert observations["priority_rank"] == rank
        assert space.contains(observations)
        # with the switch on only the two keys are added, the rest is what the switch off gives
        rest = {k: v for k, v in observations.items() if k not in space_extra}
        assert set(rest) == set(off[agent_id].as_rllib_dict()["observations"])


# ---------------------------------------------------------------- the encoder


def _encoder(communication=False, pooling="none", signals=ON, rounds=2) -> GNNEncoder:
    torch.manual_seed(0)
    return GNNEncoder(
        GNNEncoderConfig(
            num_rounds=rounds,
            hidden_width=8,
            communication=communication,
            pooling=pooling,
            conflict_signals=signals,
        )
    )


@pytest.mark.parametrize("communication", [False, True])
@pytest.mark.parametrize("pooling", ["none", "mean"])
def test_output_dim_grows_by_two_only_with_the_switch_on(communication, pooling):
    off = _encoder(communication, pooling, ConflictSignalsConfig())
    on = _encoder(communication, pooling, ON)
    assert on.output_dim == off.output_dim + 2


def test_the_signals_are_the_last_two_entries_of_the_readout_after_congestion():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(num_agents=2)
    signals = ConflictSignals(priority=(2, 1), overridden=frozenset({1}))
    observation = build_observation(graph, fleet, [], 1, 0, config, conflict_signals=signals)
    z = _encoder().forward(graph, 2, observation)
    assert z.shape == (_encoder().output_dim,)
    assert z[-3].item() == pytest.approx(observation.congestion)
    assert z[-2].item() == 1.0  # overridden last step
    assert z[-1].item() == 1.0  # agent 1 has the lower priority, the last rank


def test_the_encoder_requires_the_signals_exactly_when_the_switch_is_on():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(num_agents=2)
    plain = build_observation(graph, fleet, [], 1, 0, config)
    with pytest.raises(ValueError, match="conflict"):
        _encoder().forward(graph, 2, plain)
    signals = ConflictSignals(priority=(2, 1), overridden=frozenset())
    extended = build_observation(graph, fleet, [], 1, 0, config, conflict_signals=signals)
    with pytest.raises(ValueError, match="conflict"):
        _encoder(signals=ConflictSignalsConfig()).forward(graph, 2, extended)


@pytest.mark.parametrize("communication", [False, True])
@pytest.mark.parametrize("pooling", ["none", "mean"])
@pytest.mark.parametrize("rounds", [0, 2])
def test_the_batched_path_matches_the_oracle_with_the_signals_on(communication, pooling, rounds):
    graph = _line_graph(9)
    fleet = _fleet({1: 1, 2: 3, 3: 4, 4: 7})
    config = _config(num_agents=4, communication=communication, observation_depth=3)
    sigma = priority_permutation(5, (1, 2, 3, 4))
    signals = ConflictSignals(priority=sigma, overridden=frozenset({2, 4}))
    enc = LocalSubgraphEncodingConfig(max_local_nodes=9, max_messages=3)
    d_max = max_out_degree(graph)
    encoded = encode_all_observations(
        graph, fleet, [], 0, config, enc, d_max, pooling=pooling, conflict_signals=signals
    )
    encoder = _encoder(communication, pooling, rounds=rounds)

    tensors = {}
    for agent_id, item in encoded.items():
        for key, value in item.as_rllib_dict()["observations"].items():
            tensors.setdefault(key, []).append(np.asarray(value))
    batch = {k: torch.as_tensor(np.stack(v)) for k, v in tensors.items()}
    kwargs = {
        key: batch[key]
        for key in (
            "node_features",
            "node_mask",
            "edge_index",
            "edge_mask",
            "own_index",
            "congestion",
            "overridden_last_step",
            "priority_rank",
        )
    }
    if communication:
        kwargs.update({k: batch[k] for k in batch if k.startswith("sender_")})
    if pooling != "none":
        kwargs["pool_mask"] = batch["pool_mask"]
    with torch.no_grad():
        z_batch = encoder.forward_padded(**kwargs)

    locations = fleet.locations()
    for row, agent_id in enumerate(sorted(locations)):
        observation = build_observation(
            graph, fleet, [], agent_id, 0, config, conflict_signals=signals
        )
        senders = []
        if communication:
            for other in communication_neighbours(graph, fleet, agent_id, 3)[:3]:
                other_obs = build_observation(
                    graph, fleet, [], other, 0, config, conflict_signals=signals
                )
                senders.append((locations[other], other_obs))
        with torch.no_grad():
            z_oracle = encoder.forward(graph, locations[agent_id], observation, senders)
        assert z_batch[row].shape == z_oracle.shape
        assert torch.allclose(z_batch[row], z_oracle, atol=1e-5)
        assert z_batch[row, -2].item() == float(agent_id in signals.overridden)
        assert z_batch[row, -1].item() == pytest.approx(sigma.index(agent_id) / 3)


def test_forward_padded_requires_the_signal_tensors_exactly_when_the_switch_is_on():
    n, e = 4, 6
    base = dict(
        node_features=torch.zeros(1, n, 6),
        node_mask=torch.ones(1, n),
        edge_index=torch.zeros(1, e, 2, dtype=torch.int64),
        edge_mask=torch.zeros(1, e),
        own_index=torch.zeros(1, dtype=torch.int64),
        congestion=torch.zeros(1),
    )
    with pytest.raises(ValueError, match="conflict"):
        _encoder().forward_padded(**base)
    with pytest.raises(ValueError, match="conflict"):
        _encoder(signals=ConflictSignalsConfig()).forward_padded(
            **base, overridden_last_step=torch.zeros(1), priority_rank=torch.zeros(1)
        )
    z = _encoder().forward_padded(
        **base, overridden_last_step=torch.ones(1), priority_rank=torch.full((1,), 0.25)
    )
    assert z[0, -2:].tolist() == [1.0, 0.25]


# ---------------------------------------------------------------- the adapter and the core


def _params() -> GeneratorParams:
    return GeneratorParams(
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


_WIDE = 30  # the movable vertices of the generated graph, the widest window of any arm
_NARROW = 20  # the largest field of view at d_obs = 3


def _adapter(controller="decentralised", signals=ON, communication=False, num_agents=3, **kw):
    instance = generate_instance(_params())
    graph = instance.graph
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    zones = zones_by_aisle(graph, instance.aisle_index, 2) if controller == "section" else None
    return WarehouseMAPDMultiAgentEnv(
        graph,
        skus,
        counts,
        capacities,
        _config(controller=controller, communication=communication, num_agents=num_agents),
        LocalSubgraphEncodingConfig(
            max_local_nodes=_NARROW if controller == "decentralised" else _WIDE, max_messages=2
        ),
        (100, 101),
        credit_signal="per_agent",
        pooling="none" if controller == "decentralised" else "mean",
        zones=zones,
        conflict_signals=signals,
        **kw,
    )


def _signals_of(observations):
    return {
        a: (
            float(o["observations"]["overridden_last_step"]),
            float(o["observations"]["priority_rank"]),
        )
        for a, o in observations.items()
    }


def _swap_pair(graph):
    """Two movable vertices with an edge each way."""
    for edge in graph.edges:
        a, b = edge.source, edge.target
        if graph.vertices[a].role.movable and graph.vertices[b].role.movable:
            if any(e.source == b and e.target == a for e in graph.edges):
                return a, b
    raise AssertionError("no two-way edge")


def _slot(graph, source, target):
    return 1 + list(graph.out_neighbours(source)).index(target)


def test_the_adapter_without_the_switch_has_the_same_keys_as_before():
    env = _adapter(signals=ConflictSignalsConfig())
    observations, _ = env.reset(seed=3)
    keys = set(observations["1"]["observations"])
    assert "overridden_last_step" not in keys and "priority_rank" not in keys
    assert (
        env.observation_spaces["1"]
        == _adapter(signals=ConflictSignalsConfig()).observation_spaces["1"]
    )
    assert env.observation_spaces["1"]["observations"].contains(observations["1"]["observations"])


def test_the_adapter_with_the_switch_adds_exactly_the_two_keys_for_every_arm():
    for controller in ("decentralised", "centralised", "section"):
        off = _adapter(controller, ConflictSignalsConfig())
        on = _adapter(controller, ON)
        obs_off, _ = off.reset(seed=3)
        obs_on, _ = on.reset(seed=3)
        added = set(obs_on["1"]["observations"]) - set(obs_off["1"]["observations"])
        assert added == {"overridden_last_step", "priority_rank"}
        assert set(obs_off["1"]["observations"]) <= set(obs_on["1"]["observations"])
        assert on.observation_spaces["1"].contains(obs_on["1"])


def test_the_core_exposes_the_priority_order_conflict_resolution_uses():
    env = _adapter()
    env.reset(seed=11)
    assert env.core.priority == priority_permutation(11, (1, 2, 3))
    env.reset(seed=12)
    assert env.core.priority == priority_permutation(12, (1, 2, 3))


def test_the_rank_is_fixed_for_the_episode_and_follows_sigma():
    env = _adapter(num_agents=3)
    observations, _ = env.reset(seed=11)
    sigma = priority_permutation(11, (1, 2, 3))
    expected = {str(a): sigma.index(a) / 2 for a in (1, 2, 3)}
    for _ in range(5):
        assert {a: r for a, (_, r) in _signals_of(observations).items()} == expected
        observations, *_ = env.step({a: 0 for a in env.agents})
    assert sorted(expected.values()) == [0.0, 0.5, 1.0]


def test_the_override_flag_is_false_at_reset_true_after_an_override_false_after_a_clean_step():
    env = _adapter(num_agents=2)
    observations, _ = env.reset(seed=11)
    assert {a: flag for a, (flag, _) in _signals_of(observations).items()} == {"1": 0.0, "2": 0.0}

    # a head-on swap, both robots want the other's vertex, so both are overridden to wait
    a, b = _swap_pair(env.graph)
    env.core.fleet = _fleet({1: a, 2: b})
    slots = {"1": _slot(env.graph, a, b), "2": _slot(env.graph, b, a)}
    observations, *_ = env.step(slots)
    assert env.core.last_overridden == frozenset({1, 2})
    assert {k: flag for k, (flag, _) in _signals_of(observations).items()} == {"1": 1.0, "2": 1.0}
    assert env.core.fleet.locations() == {1: a, 2: b}  # nobody moved

    # the same proposals again are overridden again, the signal says so again
    observations, *_ = env.step(slots)
    assert {k: flag for k, (flag, _) in _signals_of(observations).items()} == {"1": 1.0, "2": 1.0}

    # a step in which nobody is overridden clears the flag
    observations, *_ = env.step({"1": 0, "2": 0})
    assert env.core.last_overridden == frozenset()
    assert {k: flag for k, (flag, _) in _signals_of(observations).items()} == {"1": 0.0, "2": 0.0}

    # a new episode starts clean even right after an override
    env.core.fleet = _fleet({1: a, 2: b})
    env.step(slots)
    observations, _ = env.reset(seed=11)
    assert {k: flag for k, (flag, _) in _signals_of(observations).items()} == {"1": 0.0, "2": 0.0}


def test_only_the_overridden_robot_is_flagged_when_the_other_gets_through():
    env = _adapter(num_agents=2)
    env.reset(seed=11)
    a, b = _swap_pair(env.graph)
    env.core.fleet = _fleet({1: a, 2: b})
    # robot 1 asks for robot 2's vertex and robot 2 waits there, so only robot 1 is overridden
    observations, *_ = env.step({"1": _slot(env.graph, a, b), "2": 0})
    assert {k: flag for k, (flag, _) in _signals_of(observations).items()} == {"1": 1.0, "2": 0.0}


def test_replay_with_the_switch_on_is_bit_for_bit():
    def run():
        env = _adapter(num_agents=3)
        observations, _ = env.reset(seed=21)
        trace = []
        for t in range(8):
            slots = {a: (t + int(a)) % 3 for a in env.agents}
            for a, slot in slots.items():
                mask = observations[a]["action_mask"]
                slots[a] = slot if mask[slot] > 0 else 0
            observations, *_ = env.step(slots)
            trace.append((_signals_of(observations), env.core.fleet.locations()))
        return trace

    assert run() == run()


# ---------------------------------------------------------------- the module


def _module_for(env, controller, communication, signals):
    pooling = "none" if controller == "decentralised" else "mean"
    spec = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=env.observation_spaces["1"],
        action_space=env.action_spaces["1"],
        model_config={
            "gnn_encoder_config": GNNEncoderConfig(
                num_rounds=2,
                hidden_width=8,
                communication=communication,
                pooling=pooling,
                conflict_signals=signals,
            ).model_dump(),
            "policy_value_head_config": {"hidden_width": 8},
        },
    )
    return spec.build()


@pytest.mark.parametrize(
    "controller, communication",
    [
        ("decentralised", False),
        ("decentralised", True),
        ("centralised", False),
        ("section", False),
    ],
)
def test_the_module_builds_and_runs_with_the_switch_on_for_each_arm(controller, communication):
    torch.manual_seed(0)
    env = _adapter(controller, ON, communication)
    observations, _ = env.reset(seed=4)
    module = _module_for(env, controller, communication, ON)
    assert module.encoder.gnn.config.conflict_signals.enabled is True
    assert (
        module.encoder.gnn.output_dim
        == _module_for(
            env, controller, communication, ConflictSignalsConfig()
        ).encoder.gnn.output_dim
        + 2
    )

    batch = {
        Columns.OBS: {
            "action_mask": torch.as_tensor(
                np.stack([o["action_mask"] for o in observations.values()])
            ),
            "observations": {
                key: torch.as_tensor(
                    np.stack([o["observations"][key] for o in observations.values()])
                )
                for key in observations["1"]["observations"]
            },
        }
    }
    out = module.forward_inference(batch)
    assert out[Columns.ACTION_DIST_INPUTS].shape == (3, env.action_spaces["1"].n)
    assert torch.isfinite(out[Columns.ACTION_DIST_INPUTS].clamp(min=-1e30)).all()

    # the policy function drives the adapter for a few steps with the module
    policy = ModulePolicy(module)
    policy.reset(4)
    for t in range(3):
        slots = policy(observations, t)
        observations, *_ = env.step(slots)
    assert set(observations) == {"1", "2", "3"}


def test_a_module_trained_without_the_switch_is_read_back_as_off():
    spec_config = {"num_rounds": 1, "hidden_width": 4, "communication": False, "pooling": "none"}
    example_04 = _load_example("04_decentralised_training.py")
    module = SimpleModule({"gnn_encoder_config": spec_config})
    assert example_04.conflict_signals_of(module) == ConflictSignalsConfig()
    with_switch = {**spec_config, "conflict_signals": {"enabled": True, "rank_encoding": "raw"}}
    assert example_04.conflict_signals_of(SimpleModule({"gnn_encoder_config": with_switch})) == (
        ConflictSignalsConfig(enabled=True, rank_encoding="raw")
    )


class SimpleModule:
    def __init__(self, model_config):
        self.model_config = model_config


def _load_example(filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_example_{filename[:3]}", EXAMPLES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_04_takes_the_switch_as_flags_off_by_default():
    example_04 = _load_example("04_decentralised_training.py")
    args = example_04.parse_args([])
    assert args.conflict_signals is False
    assert args.priority_rank_encoding == "normalised"
    args = example_04.parse_args(["--conflict-signals", "--priority-rank-encoding=raw"])
    assert args.conflict_signals is True and args.priority_rank_encoding == "raw"
    # the other arms take it too, since the signals are the robot's own information
    assert example_04.parse_args(["--conflict-signals"], controller="section").conflict_signals
    config = example_04.build_config("fixed_only", seed=0)
    arm = example_04.build_arm(example_04.build_generated_instance(), "decentralised", 5)
    assert example_04.build_gnn_config(config, arm).conflict_signals.enabled is False
    assert example_04.build_gnn_config(config, arm, ON).conflict_signals == ON


def test_the_local_subgraph_helper_is_untouched_by_the_signals():
    # a guard that the window, and so the arm difference, does not depend on the switch
    graph = _line_graph(8)
    fleet = _fleet({1: 3, 2: 4})
    config = _config(num_agents=2)
    signals = ConflictSignals(priority=(1, 2), overridden=frozenset())
    plain = build_observation(graph, fleet, [], 1, 0, config)
    extended = build_observation(graph, fleet, [], 1, 0, config, conflict_signals=signals)
    assert plain.visible_vertices == extended.visible_vertices == local_subgraph(graph, 3, 3)
