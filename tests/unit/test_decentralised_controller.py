"""Unit tests for slap_mapd_coupling.controllers.decentralised."""

import numpy as np
import pytest
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
from slap_mapd_coupling.environment.spaces import action_space, max_out_degree
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
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


def _build_module(graph: WarehouseGraph, enc_config: LocalSubgraphEncodingConfig):
    d_max = max_out_degree(graph)
    base_space = encoded_observation_space(enc_config)
    obs_space = masked_observation_space(base_space, d_max + 1)
    spec = RLModuleSpec(
        module_class=GNNActionMaskingRLModule,
        observation_space=obs_space,
        action_space=action_space(graph),
        model_config={
            "gnn_encoder_config": {"num_rounds": 1, "hidden_width": 8},
            "policy_value_head_config": {"hidden_width": 8, "num_actions": d_max + 1},
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
    controller = decentralised.DecentralisedController(module, enc_config, config, seed=0)

    decentralised.set_active_decentralised_controller(controller)

    assert get_controller("decentralised") is controller


def test_route_only_ever_produces_legal_actions():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    controller = decentralised.DecentralisedController(module, enc_config, config, seed=0)

    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )
    for t in range(20):
        actions = controller.route(graph, fleet, [], t=t)
        for agent_id, action in actions.items():
            location = fleet.locations()[agent_id]
            legal = graph.legal_actions(location)
            assert action in legal, f"illegal action {action} at vertex {location}, t={t}"


def test_route_is_deterministic_for_a_fixed_seed():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )

    controller_a = decentralised.DecentralisedController(module, enc_config, config, seed=42)
    controller_b = decentralised.DecentralisedController(module, enc_config, config, seed=42)

    for t in (0, 1, 7):
        assert controller_a.route(graph, fleet, [], t=t) == controller_b.route(
            graph, fleet, [], t=t
        )


def test_route_can_differ_across_seeds():
    graph = _line_graph(6)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    module = _build_module(graph, enc_config)
    config = _decentralised_config(num_agents=2)
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=0), 2: AgentState(agent_id=2, location=5)}
    )

    seen = set()
    for seed in range(10):
        controller = decentralised.DecentralisedController(module, enc_config, config, seed=seed)
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


def test_full_episode_via_registry_resolved_controller_respects_correctness_invariants():
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
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

    config = _decentralised_config(num_agents=3, horizon=15, arrival_rate=0.3, wait_cost=0.5)
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=12, max_messages=2)
    module = _build_module(graph, enc_config)
    controller = decentralised.DecentralisedController(module, enc_config, config, seed=config.seed)
    decentralised.set_active_decentralised_controller(controller)

    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
    env.reset()
    before = env.fleet
    for t in range(config.horizon):
        env.step()  # actions=None -> resolves "decentralised" via the registry
        assert is_collision_free(before, env.fleet), f"collision at t={t}"
        assert is_feasible(env.storage), f"infeasible storage at t={t}"
        before = env.fleet
