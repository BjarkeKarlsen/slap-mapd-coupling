"""The measures of the arrival stock model: the RunMetrics fields, how an episode is reduced to
them, the paired comparison, the configs of configs/medium_arrival, and the test stock of
`slap-mapd evaluate --arrival-rate`.

Under the arrival model an order waits for stock, a task keeps its order's release time, and the
storage update is the put-away of arriving stock. A template row keeps every arrival field at
its default.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.evaluation.comparison import compare_paired
from slap_mapd_coupling.environment.multi_agent_env import _prime_shelves
from slap_mapd_coupling.evaluation.evaluator import _planner_run, _to_run_metrics
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.training.evaluate_run import build_test_study
from slap_mapd_coupling.training.spec import (
    ArrivalSettings,
    StockSizing,
    StorageRuleSettings,
    build_experiment_config,
    build_instance,
    build_storage,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
DELTA = 100
RULES = ("fixed", "demand", "congestion")
ARRIVAL_CONFIGS = sorted((ROOT / "configs" / "medium_arrival").glob("*.yaml"))


# RunMetrics


def _row(**overrides) -> dict:
    values = dict(
        storage_mode="demand",
        controller="token_passing",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=0.3,
        seed=0,
        horizon=200,
        observation_depth=3,
        num_completed_tasks=50,
        throughput=0.25,
        keep_up_ratio=0.83,
        num_waiting_tasks=1,
        num_active_tasks=2,
        backlog=3,
        mean_service_time=20.0,
        mean_wait_for_agent=6.0,
        mean_travel_time=12.0,
        mean_blocked_time=2.0,
        mean_crowding=0.1,
        num_storage_updates=2,
        mean_relocated_units=0.0,
        mean_relocation_distance=0.0,
        mean_decision_runtime_seconds=0.001,
    )
    values.update(overrides)
    return values


def _arrival_row(**overrides) -> dict:
    values = dict(
        stock_model="arrival",
        mean_stock_wait=1.5,
        num_waiting_orders=2,
        backlog=5,
        mean_putaway_distance=54.0,
        mean_arrived_units=30.0,
        mean_split_skus=4.5,
        unplaced_units=0,
        overflow_units=0,
    )
    values.update(overrides)
    return _row(**values)


def test_a_template_row_keeps_every_arrival_field_at_its_default():
    metrics = RunMetrics(**_row(mean_relocated_units=4.0, mean_relocation_distance=12.0))
    assert metrics.stock_model == "template"
    assert metrics.mean_stock_wait is None
    assert metrics.num_waiting_orders == 0
    assert (metrics.mean_putaway_distance, metrics.mean_arrived_units) == (0.0, 0.0)
    assert (metrics.mean_split_skus, metrics.unplaced_units, metrics.overflow_units) == (0, 0, 0)


@pytest.mark.parametrize(
    "field, value",
    [
        ("mean_stock_wait", 1.0),
        ("num_waiting_orders", 1),
        ("mean_putaway_distance", 3.0),
        ("mean_arrived_units", 3.0),
        ("mean_split_skus", 1.0),
        ("unplaced_units", 1),
        ("overflow_units", 1),
    ],
)
def test_a_template_row_rejects_an_arrival_field(field, value):
    overrides = {field: value}
    if field == "num_waiting_orders":
        overrides["backlog"] = 3 + value
    with pytest.raises(ValidationError, match="stock_model='arrival'"):
        RunMetrics(**_row(**overrides))


def test_the_backlog_counts_the_orders_waiting_for_stock():
    metrics = RunMetrics(**_arrival_row())
    assert metrics.backlog == metrics.num_waiting_tasks + metrics.num_active_tasks + 2
    with pytest.raises(ValidationError, match="num_waiting_orders"):
        RunMetrics(**_arrival_row(backlog=3))


def test_an_arrival_row_with_completed_tasks_needs_the_stock_wait():
    with pytest.raises(ValidationError, match="mean_stock_wait must be set"):
        RunMetrics(**_arrival_row(mean_stock_wait=None))


def test_an_arrival_row_without_completed_tasks_has_no_stock_wait():
    empty = dict(
        num_completed_tasks=0,
        throughput=0.0,
        keep_up_ratio=0.0,
        mean_service_time=None,
        mean_wait_for_agent=None,
        mean_travel_time=None,
        mean_blocked_time=None,
    )
    assert RunMetrics(**_arrival_row(mean_stock_wait=None, **empty)).mean_stock_wait is None
    with pytest.raises(ValidationError, match="mean_stock_wait must be None"):
        RunMetrics(**_arrival_row(**empty))


def test_the_stock_wait_is_part_of_the_wait_for_an_agent():
    RunMetrics(**_arrival_row(mean_stock_wait=6.0))
    with pytest.raises(ValidationError, match="cannot exceed"):
        RunMetrics(**_arrival_row(mean_stock_wait=6.5))


def test_an_arrival_row_relocates_nothing():
    with pytest.raises(ValidationError, match="relocates nothing"):
        RunMetrics(**_arrival_row(mean_relocated_units=1.0))
    with pytest.raises(ValidationError, match="relocates nothing"):
        RunMetrics(**_arrival_row(mean_relocation_distance=1.0))


def test_an_arrival_row_without_an_arrival_has_no_putaway():
    with pytest.raises(ValidationError, match="no arrival"):
        RunMetrics(**_arrival_row(num_storage_updates=0))
    no_putaway = dict(
        mean_putaway_distance=0.0,
        mean_arrived_units=0.0,
        mean_split_skus=0.0,
        num_storage_updates=0,
    )
    assert RunMetrics(**_arrival_row(**no_putaway)).num_storage_updates == 0


# An episode reduced to a row, Token Passing on the medium instance.


def _spec(initial_units=8):
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    data = spec.model_dump()
    data["storage"].update(initial_placement="demand_blocks", initial_units=initial_units)
    data["storage_rule"] = StorageRuleSettings(
        epoch_length=DELTA, congestion_weight=1.0
    ).model_dump()
    data["arrival"] = ArrivalSettings(margin=1.0, receiving="delivery").model_dump()
    return type(spec).model_validate(data)


def _episode(spec, mode, horizon, seed=200_000):
    study = build_study(spec)
    config = build_experiment_config(spec, seed, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing", "horizon": horizon}
    )
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(horizon):
        step()
    return env, config


@pytest.mark.parametrize("mode", ["fixed", "demand"])
def test_an_arrival_episode_reports_its_log(mode):
    # One unit per SKU runs out at load 0.6, and the episode ends with the arrival at t = Delta,
    # before stage 1 serves the orders still waiting.
    env, config = _episode(_spec(initial_units=1), mode, DELTA)
    metrics = _to_run_metrics(env, config)
    log = env.log
    completed = [task for task in env.tasks if task.finish_time is not None]
    assert completed and env.waiting_orders
    assert metrics.stock_model == "arrival"
    assert metrics.num_storage_updates == len(log.arrived_units_by_update) == 1
    assert metrics.mean_stock_wait == pytest.approx(
        sum(log.stock_wait_by_task[task.task_id] for task in completed) / len(completed)
    )
    assert metrics.mean_stock_wait <= metrics.mean_wait_for_agent
    assert metrics.num_waiting_orders == len(env.waiting_orders)
    assert metrics.backlog == log.backlog_by_t[-1]
    assert metrics.backlog == (
        metrics.num_waiting_tasks + metrics.num_active_tasks + metrics.num_waiting_orders
    )
    assert metrics.mean_putaway_distance == log.putaway_distance_by_update[0]
    assert metrics.mean_arrived_units == log.arrived_units_by_update[0]
    assert metrics.mean_split_skus == log.split_skus_by_update[0]
    assert metrics.unplaced_units == sum(log.unplaced_units_by_update)
    assert metrics.overflow_units == sum(log.overflow_units_by_update)
    assert (metrics.mean_relocated_units, metrics.mean_relocation_distance) == (0.0, 0.0)


def test_the_means_are_over_arrivals():
    env, config = _episode(_spec(), "congestion", 2 * DELTA + 10)
    metrics = _to_run_metrics(env, config)
    log = env.log
    assert metrics.num_storage_updates == 2
    assert metrics.mean_putaway_distance == pytest.approx(sum(log.putaway_distance_by_update) / 2)
    assert metrics.mean_arrived_units == pytest.approx(sum(log.arrived_units_by_update) / 2)
    assert metrics.mean_split_skus == pytest.approx(sum(log.split_skus_by_update) / 2)


def test_a_template_episode_keeps_the_arrival_defaults():
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    env, config = _episode(spec, "fixed", 50)
    metrics = _to_run_metrics(env, config)
    assert metrics.stock_model == "template"
    assert metrics.mean_stock_wait is None and metrics.num_waiting_orders == 0
    assert metrics.mean_putaway_distance == 0.0 and metrics.num_storage_updates == 0


# The paired comparison


def _run(seed, service_time, storage_mode="fixed", **overrides):
    values = _arrival_row(
        storage_mode=storage_mode,
        seed=seed,
        keeps_up=True,
        mean_service_time=service_time,
        mean_wait_for_agent=service_time / 2,
        mean_travel_time=service_time / 2,
        mean_blocked_time=0.0,
        mean_putaway_distance=110.0 if storage_mode == "fixed" else 54.0 + seed,
        mean_stock_wait=1.0 + seed,
    )
    values.update(overrides)
    return RunMetrics(**values)


def test_a_template_run_and_an_arrival_run_are_not_a_pair():
    fixed = [_run(seed, 10.0) for seed in range(3)]
    template = [
        RunMetrics(**_row(seed=seed, keeps_up=True, mean_relocated_units=4.0)) for seed in range(3)
    ]
    with pytest.raises(ValueError, match="stock_model"):
        compare_paired(fixed, template)


def test_two_arrival_arms_are_paired_with_their_putaway_and_stock_wait():
    fixed = [_run(seed, 10.0 + seed) for seed in range(3)]
    adaptive = [_run(seed, 8.0 + seed, "demand") for seed in range(3)]
    result = compare_paired(fixed, adaptive)
    assert result.gains == pytest.approx((2.0, 2.0, 2.0))
    assert result.fixed_summary["mean_putaway_distance"].mean == pytest.approx(110.0)
    assert result.adaptive_summary["mean_putaway_distance"].mean == pytest.approx(55.0)
    assert result.adaptive_summary["mean_split_skus"].mean == pytest.approx(4.5)
    assert result.fixed_summary["mean_stock_wait"].mean == pytest.approx(2.0)


def test_the_stock_wait_is_left_out_when_a_run_falls_behind():
    fixed = [_run(seed, 10.0, keeps_up=False) for seed in range(3)]
    adaptive = [_run(seed, 8.0, "demand") for seed in range(3)]
    result = compare_paired(fixed, adaptive)
    assert "mean_stock_wait" not in result.fixed_summary
    assert "mean_putaway_distance" in result.fixed_summary


# configs/medium_arrival


def test_there_is_an_arrival_config_for_every_medium_config():
    assert [path.name for path in ARRIVAL_CONFIGS] == sorted(
        path.name for path in (ROOT / "configs" / "medium").glob("*.yaml")
    )


@pytest.mark.parametrize("path", ARRIVAL_CONFIGS, ids=lambda path: path.stem)
def test_an_arrival_config_differs_from_its_medium_namesake_in_the_stock_only(path):
    arrival = load_spec(path).model_dump()
    template = load_spec(ROOT / "configs" / "medium" / path.name).model_dump()
    assert arrival.pop("arrival") == {
        "margin": 1.0,
        "receiving": "delivery",
        "lines_per_order": None,
        "units_per_line": None,
        "cold_stock_top_k": 10,
    }
    assert template.pop("arrival") is None
    assert arrival["storage"].pop("initial_placement") == "demand_sized_blocks"
    template["storage"].pop("initial_placement")
    assert arrival.pop("storage_rule") == {
        "epoch_length": 100,
        "reassignment_cap": None,
        "congestion_weight": 1.0,
    }
    template.pop("storage_rule")
    assert arrival == template


@pytest.mark.parametrize("path", ARRIVAL_CONFIGS, ids=lambda path: path.stem)
def test_an_arrival_config_builds_a_study_and_plays_every_rule(path):
    spec = load_spec(path)
    study = build_study(spec)
    _, _, expected = build_storage(
        spec.storage, study.instance.graph, spec.demand, StockSizing(0.6, 100, 1.0)
    )
    assert study.counts == expected
    for mode in RULES:
        config = build_experiment_config(spec, 0, mode)
        assert (config.storage_mode, config.stock_model) == (mode, "arrival")
        assert (config.stock_margin, config.stock_floor, config.stock_receiving) == (
            1.0,
            8,
            "delivery",
        )
        assert config.storage_epoch_length == 100 and config.reassignment_cap is None
        assert config.congestion_weight == (1.0 if mode == "congestion" else None)


# The test stock of another load


@pytest.mark.parametrize("path", ARRIVAL_CONFIGS, ids=lambda path: path.stem)
def test_an_arrival_instance_has_unit_edge_costs_and_the_wait_cost_of_van_knippenberg(path):
    """Every edge costs 1 and c_wait is 0.2, after van Knippenberg et al. (2021)."""
    spec = load_spec(path)
    graph = build_instance(spec).graph
    assert {edge.cost for edge in graph.edges} == {1.0}
    assert spec.experiment.wait_cost == 0.2
    assert graph.wait_cost == 0.2


def test_the_test_study_sizes_its_stock_at_the_load_it_plays():
    spec = load_spec(ARRIVAL_CONFIGS[0])
    study = build_study(spec)
    test_spec, test_study = build_test_study(spec, study, 0.3)
    assert test_spec.arrival_rate == 0.3
    _, _, expected = build_storage(
        spec.storage, study.instance.graph, spec.demand, StockSizing(0.3, 100, 1.0)
    )
    assert test_study.counts == expected
    assert test_study.counts != study.counts
    assert test_study.instance.graph == study.instance.graph


def test_the_training_load_keeps_the_training_study():
    spec = load_spec(ARRIVAL_CONFIGS[0])
    study = build_study(spec)
    test_spec, test_study = build_test_study(spec, study, None)
    assert test_spec is spec and test_study is study


def test_a_template_test_study_has_the_training_stock_at_any_load():
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    study = build_study(spec)
    _, test_study = build_test_study(spec, study, 0.3)
    assert test_study.counts == study.counts
    assert test_study.capacities == study.capacities
    assert test_study.skus == study.skus
    assert test_study.instance.graph == study.instance.graph
    assert test_study.arm.zones == study.arm.zones


# The cold-stock measure


def test_the_prime_shelves_are_the_nearest_the_top_k_fill():
    study = build_study(_spec())
    graph = study.instance.graph
    storage = list(study.capacities)
    delivery = [v for v, x in graph.vertices.items() if x.role.delivery]
    nearest = min(storage, key=lambda v: (sum(graph.distance(v, g) for g in delivery), v))
    # 100 SKUs on 10 shelves is 10 SKUs a shelf, so the top 10 fill one shelf and the top 11 two.
    assert _prime_shelves(graph, storage, 100, 10) == [nearest]
    assert len(_prime_shelves(graph, storage, 100, 11)) == 2
    assert len(_prime_shelves(graph, storage, 100, 1000)) == len(storage)


def test_the_cold_share_counts_skus_outside_the_top_k_on_the_prime_shelves():
    env, config = _episode(_spec(), "fixed", 1)
    env.config = config.model_copy(update={"cold_stock_top_k": 10})
    env._prime_shelves = _prime_shelves(env.graph, list(env._storage_capacities), 100, 10)
    (prime,) = env._prime_shelves
    ranked = env._sku_ids  # no rotation before the first period
    hot, cold = ranked[0], ranked[50]
    counts = {k: {} for k in env.storage.counts}
    counts[hot] = {prime: 4}
    counts[cold] = {prime: 6}
    env.storage = env.storage.model_copy(update={"counts": counts})
    volume = 6 * env._skus[cold].unit_capacity
    assert env._cold_prime_share(1) == pytest.approx(volume / env._storage_capacities[prime])


def test_a_shelf_gives_up_its_oldest_units_first_and_logs_their_age():
    env, _ = _episode(_spec(), "fixed", 1)
    env.log.picked_unit_age_sum = env.log.picked_units = 0
    env._lots[("sku_000", 7)] = [[0, 2], [100, 3]]
    env._take_units("sku_000", 7, 4, 150)
    # Two units of age 150, then two of age 50.
    assert (env.log.picked_unit_age_sum, env.log.picked_units) == (2 * 150 + 2 * 50, 4)
    assert env._lots[("sku_000", 7)] == [[100, 1]]


def test_an_arrival_episode_reports_the_cold_stock_measures():
    spec = _spec()
    data = spec.model_dump()
    data["arrival"]["cold_stock_top_k"] = 10
    env, config = _episode(type(spec).model_validate(data), "demand", 2 * DELTA + 10)
    metrics = _to_run_metrics(env, config)
    shares = env.log.cold_prime_share_by_update
    assert len(shares) == metrics.num_storage_updates == 2
    assert metrics.mean_cold_prime_share == pytest.approx(sum(shares) / 2)
    assert env.log.picked_units > 0
    assert metrics.mean_unit_age_at_pick == pytest.approx(
        env.log.picked_unit_age_sum / env.log.picked_units
    )


def test_without_top_k_the_cold_share_is_off_and_the_template_keeps_the_defaults():
    env, config = _episode(_spec(), "fixed", DELTA + 10)
    assert _to_run_metrics(env, config).mean_cold_prime_share is None
    spec = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    env, config = _episode(spec, "fixed", 50)
    metrics = _to_run_metrics(env, config)
    assert metrics.mean_cold_prime_share is None and metrics.mean_unit_age_at_pick is None
    with pytest.raises(ValidationError, match="only applies"):
        ExperimentConfig.model_validate({**config.model_dump(), "cold_stock_top_k": 10})
