"""Unit tests for slap_mapd_coupling.evaluation.evaluator."""

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.environment.multi_agent_env import _relocated_units
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph


def _graph(num_endpoints: int = 4, seed: int = 0) -> WarehouseGraph:
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
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
        observation_depth=2,
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
    assert metrics.mean_wait_for_agent is None
    assert metrics.mean_travel_time is None
    assert metrics.mean_blocked_time is None


def test_service_time_splits_into_three_parts_that_add_up():
    metrics = _run(_graph(), _config(horizon=40, arrival_rate=1.0))
    assert metrics.num_completed_tasks > 0
    assert metrics.mean_service_time is not None
    assert metrics.mean_wait_for_agent is not None
    assert metrics.mean_travel_time is not None
    assert metrics.mean_blocked_time is not None
    assert metrics.mean_travel_time > 0.0
    parts = metrics.mean_wait_for_agent + metrics.mean_travel_time + metrics.mean_blocked_time
    assert abs(parts - metrics.mean_service_time) < 1e-9


def test_keeps_up_is_unset_without_a_threshold():
    metrics = _run(_graph(), _config(horizon=30))
    assert metrics.keeps_up is None
    assert metrics.keep_up_ratio == metrics.throughput / metrics.arrival_rate


def test_keeps_up_compares_throughput_with_the_threshold():
    config = _config(horizon=40, arrival_rate=1.0, keep_up_threshold=0.5)
    metrics = _run(_graph(), config)
    assert metrics.keeps_up == (metrics.throughput >= 0.5 * config.arrival_rate)


def test_crowding_is_a_fraction_for_the_centralised_controller():
    """d_obs is required for every controller (sec:pf:env,
    sec:pf:controllers), so a centralised (planner) run reports a
    numeric mean_crowding too, not just the decentralised arm."""
    metrics = _run(_graph(), _config(horizon=20, controller="centralised"))
    assert metrics.mean_crowding is not None
    assert 0.0 <= metrics.mean_crowding <= 1.0


def test_fixed_storage_has_no_update_and_relocates_nothing():
    metrics = _run(_graph(), _config(horizon=20))
    assert metrics.num_storage_updates == 0
    assert metrics.mean_relocated_units == 0.0


def test_adaptive_storage_updates_every_epoch_within_the_cap():
    config = _config(storage_mode="demand", storage_epoch_length=5, reassignment_cap=3, horizon=23)
    metrics = _run(_graph(), config)
    assert metrics.num_storage_updates == 23 // 5
    assert 0.0 <= metrics.mean_relocated_units <= 3


def test_relocated_units_counts_arrivals_at_new_cells():
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {1: 10.0, 2: 10.0, 3: 10.0}
    before = StorageState(skus=skus, capacities=capacities, counts={"tea": {1: 5, 2: 0}})
    after = StorageState(skus=skus, capacities=capacities, counts={"tea": {1: 2, 2: 2, 3: 1}})
    # Three units left vertex 1: two arrive at 2, one at 3.
    assert _relocated_units(before, after) == 3
    assert _relocated_units(before, before) == 0


def test_decision_runtimes_are_nonnegative_and_present():
    metrics = _run(_graph(), _config(horizon=10))
    assert metrics.mean_decision_runtime_seconds >= 0.0
