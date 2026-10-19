"""Unit tests for slap_mapd_coupling.evaluation.evaluator."""

import json
import subprocess
import sys

import pytest

import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.environment.multi_agent_env import _relocated_units, _relocation_distance
from slap_mapd_coupling.evaluation.evaluator import (
    backlog_slope,
    completed_share,
    keeps_up,
    run_episode,
)
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
        planner="prioritised_astar",
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
    assert metrics.controller == config.planner
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


def test_keeps_up_is_unset_with_only_one_threshold():
    assert _run(_graph(), _config(horizon=30, keep_up_threshold=0.5)).keeps_up is None
    assert _run(_graph(), _config(horizon=30, backlog_slope_max=0.1)).keeps_up is None


def test_keeps_up_checks_the_completed_share_and_the_backlog_slope():
    config = _config(horizon=40, arrival_rate=1.0, keep_up_threshold=0.5, backlog_slope_max=0.1)
    metrics = _run(_graph(), config)
    assert metrics.completed_share == metrics.num_completed_tasks / (
        metrics.num_completed_tasks + metrics.backlog
    )
    assert metrics.backlog_slope is not None
    assert metrics.keeps_up == (
        metrics.completed_share >= 0.5 and metrics.backlog_slope <= 0.1
    )


def test_stalled_compares_the_longest_stand_still_with_the_stall_length():
    unset = _run(_graph(), _config(horizon=40, arrival_rate=1.5))
    assert unset.stalled is None
    longest = unset.longest_stand_still
    assert longest > 0
    at_longest = _config(horizon=40, arrival_rate=1.5, stall_length=longest)
    assert _run(_graph(), at_longest).stalled is False
    shorter = _config(horizon=40, arrival_rate=1.5, stall_length=max(longest - 1, 1))
    assert _run(_graph(), shorter).stalled is (longest > 1)


def test_completed_share_counts_the_released_tasks():
    assert completed_share(336, 378 - 336) == 336 / 378
    assert completed_share(0, 0) is None


_T = 600


def _keeps_up(trace: list[int]) -> bool | None:
    return keeps_up(completed_share(500, trace[-1]), backlog_slope(trace), 0.8, 0.01)


def test_a_backlog_that_levels_off_keeps_up():
    trace = [min(t, 20) + (t % 3) for t in range(_T + 1)]
    assert backlog_slope(trace) == pytest.approx(0.0, abs=1e-3)
    assert _keeps_up(trace) is True


def test_a_backlog_that_grows_linearly_does_not_keep_up():
    trace = [t // 10 for t in range(_T + 1)]
    assert backlog_slope(trace) == pytest.approx(0.1, abs=1e-3)
    assert _keeps_up(trace) is False


def test_a_backlog_that_jumps_once_and_then_stays_flat_keeps_up():
    trace = [5 if t < 200 else 40 for t in range(_T + 1)]
    assert backlog_slope(trace) == pytest.approx(0.0, abs=1e-9)
    assert _keeps_up(trace) is True


def test_the_backlog_slope_reads_only_the_second_half_of_the_run():
    trace = [t if t < _T // 2 else 10 for t in range(_T + 1)]
    assert backlog_slope(trace) == pytest.approx(0.0, abs=1e-9)
    assert backlog_slope([3]) is None


def test_crowding_is_a_fraction_for_the_reference_planner():
    """d_obs is required for every controller (sec:pf:env,
    sec:pf:controllers), so a reference planner run reports a
    numeric mean_crowding too, not just the decentralised arm."""
    metrics = _run(_graph(), _config(horizon=20))
    assert metrics.mean_crowding is not None
    assert 0.0 <= metrics.mean_crowding <= 1.0


def test_fixed_storage_has_no_update_and_relocates_nothing():
    metrics = _run(_graph(), _config(horizon=20))
    assert metrics.num_storage_updates == 0
    assert metrics.mean_relocated_units == 0.0
    assert metrics.mean_relocation_distance == 0.0


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


def test_run_episode_with_a_trace_path_returns_the_same_metrics(tmp_path):
    config = _config()
    graph = _graph()
    path = tmp_path / "traces" / "episode.jsonl"
    with_trace = _run_traced(graph, config, path)
    without = _run(graph, config)
    # Decision runtime is wall-clock time, so it differs between any two runs.
    exclude = {"mean_decision_runtime_seconds"}
    assert with_trace.model_dump(exclude=exclude) == without.model_dump(exclude=exclude)
    lines = path.read_text().splitlines()
    assert len(lines) == config.horizon + 1
    assert json.loads(lines[0])["type"] == "header"


def _run_traced(graph: WarehouseGraph, config: ExperimentConfig, path):
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5}}
    return run_episode(graph, skus, counts, capacities, config, trace_path=path)


def test_a_planner_episode_never_imports_ray():
    """A planner run stays free of Ray. The adapter that a learned arm runs through is
    imported inside that branch of run_episode only. A fresh process is needed, since
    this one has imported Ray for other tests."""
    code = """
import sys
import slap_mapd_coupling.storage.fixed
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.evaluation.evaluator import (
    backlog_slope,
    completed_share,
    keeps_up,
    run_episode,
)
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph

graph = generate_warehouse_graph(GeneratorParams(
    max_num_aisles=6, max_aisle_length=6, max_num_cross_aisles=5, one_way_fraction=0.0,
    wait_cost=0.0, num_storage_vertices=6, num_delivery_vertices=2, num_endpoints=4, seed=0))
storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
config = ExperimentConfig(
    storage_mode="fixed", planner="token_passing", congestion_sensitive=False,
    communication=False, num_agents=3, arrival_rate=1.0, seed=7, horizon=10, wait_cost=0.0,
    observation_depth=2)
metrics = run_episode(
    graph, {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}, {"tea": {storage[0]: 5}},
    {v: 10.0 for v in storage}, config)
assert metrics.horizon == 10
assert "ray" not in sys.modules, "a planner episode imported ray"
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class _LineGraph:
    """Vertices on a line, d_G(v, w) = |v - w|, enough for the relocation measure."""

    @staticmethod
    def distance(v: int, w: int) -> float:
        return float(abs(v - w))


def _state(counts: dict[str, dict[int, int]]) -> StorageState:
    skus = {name: SkuType(sku_id=name, unit_capacity=1.0) for name in counts}
    capacities = {v: 50.0 for v in range(0, 11)}
    return StorageState(skus=skus, capacities=capacities, counts=counts)


def test_relocation_distance_of_a_hand_made_update():
    # Tea: 5 units leave vertex 0, 3 arrive at 1 and 2 at 6. Cost 3*1 + 2*6 = 15.
    # Mug: vertices 2 and 8 each lose 1 unit, vertices 4 and 6 each gain 1. Cheapest pair
    # first is (2, 4) at 2, then (8, 6) at 2, total 4 (the greedy match, not 2->6 + 8->4).
    before = _state({"tea": {0: 5}, "mug": {2: 1, 8: 1}})
    after = _state({"tea": {1: 3, 6: 2}, "mug": {4: 1, 6: 1}})
    assert _relocation_distance(before, after, _LineGraph()) == 15.0 + 4.0


def test_relocation_distance_breaks_ties_by_vertex_ids():
    # Vertex 5 lost 1 unit and vertex 3 lost 1 unit, vertex 4 gained 1. Both pairs are at
    # distance 1, and the lower from id (3) is matched first. The unit at 5 is then
    # unmatched, so exactly one unit is counted, at distance 1.
    before = _state({"tea": {3: 1, 5: 1}})
    after = _state({"tea": {4: 1}})
    assert _relocation_distance(before, after, _LineGraph()) == 1.0
    # Ties on the from id are broken by the lower to id: 4 loses 2, 3 and 5 gain 1 each.
    before = _state({"tea": {4: 2}})
    after = _state({"tea": {3: 1, 5: 1}})
    assert _relocation_distance(before, after, _LineGraph()) == 2.0


def test_relocation_distance_is_zero_when_nothing_moved():
    state = _state({"tea": {1: 4, 2: 1}})
    assert _relocation_distance(state, state, _LineGraph()) == 0.0


def test_relocation_distance_is_logged_per_update_under_demand_storage():
    config = _config(storage_mode="demand", storage_epoch_length=5, reassignment_cap=3, horizon=60)
    metrics = _run(_graph(), config)
    assert metrics.num_storage_updates == 60 // 5
    assert metrics.mean_relocation_distance >= 0.0
    if metrics.mean_relocated_units > 0:
        assert metrics.mean_relocation_distance > 0.0
