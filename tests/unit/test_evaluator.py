"""Unit tests for slap_mapd_coupling.evaluation.evaluator."""

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.evaluation.evaluator import _traffic_measures, run_episode
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph


def _graph(num_endpoints: int = 4, seed: int = 0) -> WarehouseGraph:
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=num_endpoints,
        seed=seed,
    )
    return generate_warehouse_graph(params)


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=25,
        wait_cost=0.0,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _run(graph: WarehouseGraph, config: ExperimentConfig):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return run_episode(graph, skus, counts, capacities, config)


def test_run_episode_identifying_fields_match_config():
    config = _config()
    metrics = _run(_graph(), config)
    assert metrics.storage_mode == config.storage_mode
    assert metrics.controller == config.controller
    assert metrics.num_agents == config.num_agents
    assert metrics.arrival_rate == config.arrival_rate
    assert metrics.seed == config.seed
    assert metrics.horizon == config.horizon


def test_throughput_matches_completed_over_horizon():
    config = _config(horizon=40, arrival_rate=2.0)
    metrics = _run(_graph(), config)
    assert metrics.throughput == metrics.num_completed_tasks / config.horizon


def test_backlog_equals_waiting_plus_active():
    metrics = _run(_graph(), _config(horizon=30, arrival_rate=1.5))
    assert metrics.backlog == metrics.num_waiting_tasks + metrics.num_active_tasks


def test_negligible_arrival_rate_produces_no_completions_and_none_fields():
    # arrival_rate must be > 0 (PositiveFloat); vanishingly small over a
    # short horizon is effectively zero without violating that.
    metrics = _run(_graph(), _config(horizon=5, arrival_rate=1e-9))
    assert metrics.num_completed_tasks == 0
    assert metrics.mean_service_time is None
    assert metrics.movement_cost_per_task is None
    assert metrics.mean_blocked_time is None


def test_a_run_with_traffic_reports_entropy_and_concentration_as_complements():
    metrics = _run(_graph(), _config(horizon=30, arrival_rate=2.0))
    assert metrics.num_traversed_edges > 0
    assert metrics.traffic_entropy is not None
    assert metrics.traffic_concentration is not None
    assert abs((metrics.traffic_entropy + metrics.traffic_concentration) - 1.0) < 1e-9


def test_decision_runtimes_are_nonnegative_and_present():
    metrics = _run(_graph(), _config(horizon=10))
    assert metrics.mean_decision_runtime_seconds >= 0.0


def test_traffic_measures_no_traversals():
    assert _traffic_measures({}) == (0, None, None)


def test_traffic_measures_single_edge_is_maximally_concentrated():
    assert _traffic_measures({(1, 2): 5}) == (1, 0.0, 1.0)


def test_traffic_measures_uniform_usage_gives_entropy_one():
    # Four edges, equal traversal counts -> perfectly uniform -> H_T = 1.
    num_traversed, entropy, concentration = _traffic_measures(
        {(1, 2): 10, (2, 3): 10, (3, 4): 10, (4, 5): 10}
    )
    assert num_traversed == 4
    assert entropy == 1.0
    assert concentration == 0.0


def test_traffic_measures_skewed_usage_gives_entropy_below_one():
    num_traversed, entropy, concentration = _traffic_measures({(1, 2): 100, (2, 3): 1})
    assert num_traversed == 2
    assert 0.0 < entropy < 1.0
    assert concentration == 1.0 - entropy
