"""Unit tests for slap_mapd_coupling.controllers.decentralised."""

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

import slap_mapd_coupling.controllers.decentralised as decentralised
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.controllers.registry import get_controller
from slap_mapd_coupling.core.agents import AgentState, FleetState, is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, is_feasible
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import action_for_slot, action_space, max_out_degree
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
    encoded_observation_space,
)
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule


@pytest.fixture(autouse=True)
def _reset_active_controller():
    """set_active_decentralised_controller sets process-global state
    (see the module docstring for why); reset it around every test in
    this file so one test's configuration can't leak into another's."""
    decentralised._active_controller = None
    yield
    decentralised._active_controller = None


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _decentralised_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=2,
        arrival_rate=1.0,
        seed=0,
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


def _build_module(
    graph: WarehouseGraph, enc_config: LocalSubgraphEncodingConfig, communication: bool = False
):
    d_max = max_out_degree(graph)
    base_space = encoded_observation_space(enc_config, d_max, communication)
    obs_space = masked_observation_space(base_space, d_max + 1)
    spec = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=obs_space,
        action_space=action_space(graph),
        model_config={
            "gnn_encoder_config": {
                "num_rounds": 1,
                "hidden_width": 8,
                "communication": communication,
            },
            "policy_value_head_config": {"hidden_width": 8},
        },
    )
    return spec.build()


def test_get_controller_raises_without_an_active_controller_configured():
    with pytest.raises(RuntimeError, match="set_active_decentralised_controller"):
        get_controller("decentralised")


def test_set_active_decentralised_controller_wires_get_controller():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config()
    controller = decentralised.DecentralisedController(module, enc_config, config)
    controller.reset(episode_seed=0)

    decentralised.set_active_decentralised_controller(controller)

    assert get_controller("decentralised") is controller


def test_route_before_reset_raises():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    fleet = FleetState(agents={1: AgentState(agent_id=1, location=0)})

    with pytest.raises(RuntimeError, match="reset"):
        controller.route(graph, fleet, [], t=0)


def test_route_only_ever_produces_legal_actions():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    controller.reset(episode_seed=0)

    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )
    for t in range(20):
        actions = controller.route(graph, fleet, [], t=t)
        for agent_id, action in actions.items():
            location = fleet.locations()[agent_id]
            legal = graph.legal_actions(location)
            assert action in legal, f"illegal action {action} at vertex {location}, t={t}"


def test_route_is_deterministic_for_a_fixed_episode_seed():
    """The same episode seed replays bit for bit, however many episodes
    the same controller instance has already run through reset()."""
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )

    controller_a = decentralised.DecentralisedController(module, enc_config, config)
    controller_b = decentralised.DecentralisedController(module, enc_config, config)
    controller_a.reset(episode_seed=42)
    controller_b.reset(episode_seed=42)

    for t in (0, 1, 7):
        assert controller_a.route(graph, fleet, [], t=t) == controller_b.route(
            graph, fleet, [], t=t
        )

    # Re-resetting the SAME instance to a later, unrelated episode and
    # then back to 42 must replay identically too -- the stream comes
    # from reset()'s argument, not from how many episodes came before.
    controller_a.reset(episode_seed=999)
    controller_a.route(graph, fleet, [], t=0)
    controller_a.reset(episode_seed=42)
    assert controller_a.route(graph, fleet, [], t=0) == controller_b.route(graph, fleet, [], t=0)


def test_route_can_differ_across_episode_seeds_for_the_same_logits():
    """Different episode seeds sample different actions from the same
    trained module (same weights, same state -> same logits every time),
    so the difference can only come from the sampling stream reset()
    seeds."""
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )
    controller = decentralised.DecentralisedController(module, enc_config, config)

    seen = set()
    for seed in range(10):
        controller.reset(episode_seed=seed)
        actions = controller.route(graph, fleet, [], t=0)
        seen.add(tuple(sorted((aid, a.kind, a.target) for aid, a in actions.items())))
    assert len(seen) > 1  # different seeds actually produce different draws sometimes


def test_sample_legal_slot_never_picks_a_masked_out_slot():
    rng = np.random.default_rng(0)
    logits = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)  # would favor slots freely
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(200):
        slot = decentralised._sample_legal_slot(logits, mask, rng)
        assert mask[slot] == 1.0


# The largest field of view of _small_instance() at d_obs = 3 has 18 vertices. A window is
# never truncated, so the cap must fit it.
_SMALL_INSTANCE_CAP = 18


def _small_instance():
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.0,
        wait_cost=0.5,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=3,
        seed=1,
    )
    graph = generate_warehouse_graph(params)
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 20.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 8, storage_vertices[1]: 8}}
    return graph, skus, capacities, counts


@pytest.mark.parametrize("communication", [False, True])
def test_full_episode_via_the_registry_respects_the_correctness_invariants(communication):
    graph, skus, capacities, counts = _small_instance()
    config = _decentralised_config(
        num_agents=3, horizon=15, arrival_rate=0.3, wait_cost=0.5, communication=communication
    )
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=_SMALL_INSTANCE_CAP, max_messages=2)
    module = _build_module(graph, enc_config, communication)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    controller.reset(episode_seed=config.seed)
    decentralised.set_active_decentralised_controller(controller)

    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
    env.reset()
    before = env.fleet
    for t in range(config.horizon):
        env.step()  # actions=None -> resolves "decentralised" via the registry
        assert is_collision_free(before, env.fleet), f"collision at t={t}"
        assert is_feasible(env.storage), f"infeasible storage at t={t}"
        before = env.fleet


@pytest.mark.parametrize("communication", [False, True])
def test_run_episode_resets_the_controller_so_route_never_raises(communication):
    """evaluation.evaluator.run_episode must reset the registered
    controller to config.seed itself: this test never calls
    controller.reset(), so it would hit DecentralisedController.route()'s
    own "called before reset()" RuntimeError if run_episode didn't."""
    graph, skus, capacities, counts = _small_instance()
    config = _decentralised_config(
        num_agents=3, horizon=15, arrival_rate=0.3, wait_cost=0.5, communication=communication
    )
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=_SMALL_INSTANCE_CAP, max_messages=2)
    module = _build_module(graph, enc_config, communication)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    decentralised.set_active_decentralised_controller(controller)

    run_episode(graph, skus, counts, capacities, config)  # would raise if not reset internally


@pytest.mark.parametrize("communication", [False, True])
def test_run_episode_is_deterministic_for_the_same_config_seed(communication):
    graph, skus, capacities, counts = _small_instance()
    config = _decentralised_config(
        num_agents=3, horizon=15, arrival_rate=0.3, wait_cost=0.5, communication=communication
    )
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=_SMALL_INSTANCE_CAP, max_messages=2)
    module = _build_module(graph, enc_config, communication)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    decentralised.set_active_decentralised_controller(controller)

    first = run_episode(graph, skus, counts, capacities, config)
    second = run_episode(graph, skus, counts, capacities, config)
    # mean_decision_runtime_seconds is wall-clock timing, not part of replay.
    exclude = {"mean_decision_runtime_seconds"}
    assert first.model_dump(exclude=exclude) == second.model_dump(exclude=exclude)


class _CountingModule:
    """Wraps an RLModule and records the batch size of every forward_inference call."""

    def __init__(self, module):
        self._module = module
        self.batch_sizes: list[int] = []

    def forward_inference(self, batch):
        self.batch_sizes.append(batch[Columns.OBS]["action_mask"].shape[0])
        return self._module.forward_inference(batch)


def _per_agent_actions(module, graph, fleet, config, enc_config, t, episode_seed):
    """The actions of one timestep from one module call per agent, drawn from the
    same (episode_seed + offset, t, agent_id) streams as DecentralisedController."""
    encoded = encode_all_observations(
        graph, fleet, [], t, config, enc_config, max_out_degree(graph)
    )
    offset = decentralised._ACTION_SAMPLING_SEED_OFFSET
    actions = {}
    for agent_id, item in encoded.items():
        out = module.forward_inference(decentralised._batch([item]))
        logits = out[Columns.ACTION_DIST_INPUTS][0].detach().numpy()
        rng = np.random.default_rng((episode_seed + offset, t, agent_id))
        slot = decentralised._sample_legal_slot(logits, item.action_mask, rng)
        actions[agent_id] = action_for_slot(graph, fleet.locations()[agent_id], slot)
    return actions


def _three_robots():
    # robots 1 and 2 hear each other, robot 3 hears robot 2, within d_obs = 3
    return FleetState(
        agents={
            1: AgentState(agent_id=1, location=0),
            2: AgentState(agent_id=2, location=1),
            3: AgentState(agent_id=3, location=4),
        }
    )


@pytest.mark.parametrize("communication", [False, True])
def test_route_makes_one_module_call_for_all_robots(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    counting = _CountingModule(_build_module(graph, enc_config, communication))
    config = _decentralised_config(num_agents=3, communication=communication)
    controller = decentralised.DecentralisedController(counting, enc_config, config)
    controller.reset(episode_seed=3)

    for t in range(4):
        controller.route(graph, _three_robots(), [], t=t)

    assert counting.batch_sizes == [3, 3, 3, 3]


@pytest.mark.parametrize("communication", [False, True])
def test_route_actions_equal_those_from_one_module_call_per_agent(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config, communication)
    config = _decentralised_config(num_agents=3, communication=communication)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    controller.reset(episode_seed=3)
    fleet = _three_robots()

    for t in range(10):
        expected = _per_agent_actions(module, graph, fleet, config, enc_config, t, 3)
        assert controller.route(graph, fleet, [], t=t) == expected


@pytest.mark.parametrize("communication", [False, True])
def test_batched_logits_equal_the_per_agent_logits(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config, communication)
    config = _decentralised_config(num_agents=3, communication=communication)
    encoded = encode_all_observations(
        graph, _three_robots(), [], 0, config, enc_config, max_out_degree(graph)
    )

    with torch.no_grad():
        together = module.forward_inference(decentralised._batch(list(encoded.values())))
        for row, item in enumerate(encoded.values()):
            alone = module.forward_inference(decentralised._batch([item]))
            assert torch.allclose(
                together[Columns.ACTION_DIST_INPUTS][row],
                alone[Columns.ACTION_DIST_INPUTS][0],
                atol=1e-5,
            )


@pytest.mark.parametrize("communication", [False, True])
def test_batch_stacks_every_observation_key_including_the_move_slot_arrays(communication):
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    config = _decentralised_config(num_agents=3, communication=communication)
    d_max = max_out_degree(graph)
    encoded = encode_all_observations(graph, _three_robots(), [], 0, config, enc_config, d_max)

    batch = decentralised._batch(list(encoded.values()))

    observations = batch[Columns.OBS]["observations"]
    assert set(observations) == set(encoded[1].as_rllib_dict()["observations"])
    assert observations["move_target_index"].shape == (3, d_max)
    assert observations["move_target_index"].dtype == torch.int64
    assert ("move_sender_index" in observations) == communication
    assert ("move_sender_mask" in observations) == communication


def test_batch_stacks_the_pool_mask_like_every_other_key():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    config = _decentralised_config(num_agents=3)
    d_max = max_out_degree(graph)
    encoded = encode_all_observations(
        graph, _three_robots(), [], 0, config, enc_config, d_max, pooling="mean"
    )

    observations = decentralised._batch(list(encoded.values()))[Columns.OBS]["observations"]

    expected = np.stack([item.pool_mask for item in encoded.values()])
    assert observations["pool_mask"].dtype == torch.float32
    np.testing.assert_array_equal(observations["pool_mask"].numpy(), expected)


def test_constructing_a_controller_with_communication_and_no_room_for_a_sender_is_rejected():
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=0)
    config = _decentralised_config(num_agents=2, communication=True)

    with pytest.raises(ValueError, match="max_messages"):
        decentralised.DecentralisedController(object(), enc_config, config)


def test_the_realised_trace_replays_bit_for_bit_with_communication_on():
    """Correctness check 4 for the learned controller. Every robot's action comes
    from its own (episode_seed, t, agent_id) stream, so the trace of one seed is
    the same on every run."""
    graph, skus, capacities, counts = _small_instance()
    config = _decentralised_config(
        num_agents=3, horizon=20, arrival_rate=0.3, wait_cost=0.5, communication=True
    )
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=_SMALL_INSTANCE_CAP, max_messages=2)
    module = _build_module(graph, enc_config, communication=True)
    controller = decentralised.DecentralisedController(module, enc_config, config)
    decentralised.set_active_decentralised_controller(controller)

    def run() -> list:
        controller.reset(config.seed)
        env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
        env.reset()
        trace = []
        for _ in range(config.horizon):
            env.step()
            trace.append(tuple(sorted(env.fleet.locations().items())))
        return trace

    first = run()
    assert first == run()
    assert len(set(first)) > 1  # the robots actually move
