"""Unit tests for the progress numbers in slap_mapd_coupling.training.callbacks."""

import json

import numpy as np
import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.planners import PrioritisedAStarPlanner
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.training.callbacks import (
    StandingStillCounter,
    format_iteration,
    iteration_record,
    progress_metrics,
)


def _env(horizon: int = 30) -> WarehouseMAPDEnv:
    graph = generate_warehouse_graph(
        GeneratorParams(
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
    )
    storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    config = ExperimentConfig(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=horizon,
        wait_cost=0.0,
        observation_depth=2,
    )
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    counts = {"tea": {storage[0]: 5, storage[1]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, {v: 10.0 for v in storage}, config)


def test_progress_metrics_at_reset_has_nothing_completed():
    env = _env()
    env.reset()
    # The orders of t = 0 are released in reset, but the backlog is logged before them.
    assert progress_metrics(env) == {
        "tasks_completed": 0.0,
        "tasks_released": float(len(env.tasks)),
        "backlog": 0.0,
        "steps_since_last_completion": 0.0,
        "episode_step": 0.0,
    }


def test_progress_metrics_follow_the_environment_after_a_few_steps():
    env = _env()
    env.reset()
    planner = PrioritisedAStarPlanner()
    for _ in range(12):
        step_with_planner(env, planner)
    values = progress_metrics(env)
    finish_times = [task.finish_time for task in env.tasks if task.finish_time is not None]
    assert values["episode_step"] == 12.0
    assert values["tasks_released"] == float(len(env.tasks))
    assert values["tasks_completed"] == float(len(finish_times))
    assert values["backlog"] == float(env.log.backlog_by_t[-1])
    assert values["steps_since_last_completion"] == 12.0 - max(finish_times, default=0)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=v) for aid, v in locations.items()}
    )


def test_standing_still_counter_counts_agent_steps_without_a_move():
    counter = StandingStillCounter()
    assert counter.fraction() is None
    counter.update(_fleet({1: 10, 2: 20}), _fleet({1: 10, 2: 21}))
    counter.update(_fleet({1: 10, 2: 21}), _fleet({1: 10, 2: 21}))
    assert counter.fraction() == pytest.approx(3 / 4)
    counter.reset()
    assert counter.fraction() is None


def test_format_iteration_prints_the_return_only_when_it_is_a_finite_number():
    full = {
        "tasks_completed": 3.0,
        "tasks_released": 10.0,
        "backlog": 7.0,
        "steps_since_last_completion": 120.0,
        "standing_still_fraction": 0.25,
    }
    line = format_iteration(2, 5, {"env_runners": {**full, "episode_return_mean": 1.5}})
    assert line.startswith("  iter 2/5:")
    assert line == (
        "  iter 2/5: episode_return_mean=1.5 tasks_completed=3 tasks_released=10 backlog=7 "
        "steps_since_last_completion=120 standing_still_fraction=0.25"
    )
    big = {**full, "tasks_released": 1234.5, "episode_return_mean": 1390.04}
    line = format_iteration(1, 1, {"env_runners": big})
    assert "tasks_released=1234" in line or "tasks_released=1235" in line
    assert "episode_return_mean=1390.0" in line and "e+" not in line
    for missing in (None, float("nan")):
        line = format_iteration(2, 5, {"env_runners": {**full, "episode_return_mean": missing}})
        assert "episode_return_mean=0 " in line and "no episode ended" not in line
    line = format_iteration(2, 5, {"env_runners": full})
    assert "episode_return_mean=0 " in line
    # A real return of 0 prints as 0.0, so it differs from "no episode ended yet".
    line = format_iteration(2, 5, {"env_runners": {**full, "episode_return_mean": 0.0}})
    assert "episode_return_mean=0.0 " in line
    # A numpy float32 is a number, not a missing value.
    value = np.float32(502.6)
    line = format_iteration(2, 5, {"env_runners": {**full, "episode_return_mean": value}})
    assert "episode_return_mean=502.6 " in line


def test_format_iteration_marks_a_missing_key_instead_of_failing():
    line = format_iteration(1, 1, {"env_runners": {"tasks_completed": 1.0}})
    assert "tasks_completed=1" in line
    assert "backlog=n/a" in line
    assert "standing_still_fraction=n/a" in line
    assert "episode_return_mean=0 " in line
    assert "backlog=n/a" in format_iteration(1, 1, {})


def test_iteration_record_keeps_numbers_and_turns_everything_else_into_none():
    result = {
        "env_runners": {
            "episode_return_mean": float("nan"),
            "tasks_completed": np.float32(3.0),
            "tasks_released": 10.0,
            "standing_still_fraction": 0.25,
        }
    }
    record = iteration_record(4, result)
    assert record == {
        "iteration": 4,
        "episode_return_mean": None,
        "tasks_completed": 3.0,
        "tasks_released": 10.0,
        "backlog": None,
        "steps_since_last_completion": None,
        "standing_still_fraction": 0.25,
    }
    assert json.loads(json.dumps(record)) == record
    assert iteration_record(1, {})["tasks_completed"] is None
