"""Unit tests for slap_mapd_coupling.training.callbacks, the progress numbers and the
invariant checks on the training adapter."""

import json

import numpy as np
import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.zones import zones_by_aisle
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_instance,
    generate_warehouse_graph,
)
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.planners import PrioritisedAStarPlanner
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.training.callbacks import (
    StandingStillCounter,
    TrainingProgressCallbacks,
    episode_metrics,
    format_iteration,
    iteration_record,
    progress_metrics,
)
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv


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
        "episode_step": None,
        **_NO_EPISODE_KEYS,
    }
    assert json.loads(json.dumps(record)) == record
    assert iteration_record(1, {})["tasks_completed"] is None


_EPISODE_KEY_NAMES = (
    "return_shaping",
    "return_deliver",
    "return_override",
    "return_wait",
    "return_congestion",
    "override_rate",
    "episode_tasks_completed",
    "episode_tasks_released",
    "episode_max_agent_override_share",
)
_NO_EPISODE_KEYS = {key: None for key in _EPISODE_KEY_NAMES}


def test_iteration_record_carries_the_per_episode_keys_and_null_when_no_episode_ended():
    ended = {
        "return_shaping": 12.5,
        "return_deliver": np.float32(4.0),
        "return_override": -30.0,
        "return_wait": -1.5,
        "return_congestion": 0.0,
        "override_rate": 0.125,
        "episode_tasks_completed": 9.0,
        "episode_tasks_released": 40.0,
        "episode_max_agent_override_share": 0.5,
    }
    record = iteration_record(2, {"env_runners": {"episode_return_mean": -13.5, **ended}})
    for key, value in ended.items():
        assert record[key] == float(value)
    assert record["episode_return_mean"] == -13.5
    assert json.loads(json.dumps(record)) == record
    # an iteration in which no episode ended has none of them, which is null, not 0
    quiet = iteration_record(3, {"env_runners": {"tasks_completed": 2.0}})
    for key in _EPISODE_KEY_NAMES:
        assert key in quiet and quiet[key] is None
    assert quiet["tasks_completed"] == 2.0
    nan = iteration_record(4, {"env_runners": {"override_rate": float("nan")}})
    assert nan["override_rate"] is None


def test_iteration_record_carries_the_episode_step_and_has_no_window_keys():
    record = iteration_record(2, {"env_runners": {"episode_step": 630.0}})
    assert record["episode_step"] == 630.0
    assert not any(key.startswith(("window_", "running_")) for key in record)
    assert iteration_record(3, {"env_runners": {}})["episode_step"] is None


# --- the callbacks on the training adapter ----------------------------------------------


def _adapter(controller: str, horizon: int = 12) -> WarehouseMAPDMultiAgentEnv:
    instance = generate_instance(
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
    graph = instance.graph
    storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    config = ExperimentConfig(
        storage_mode="fixed",
        controller=controller,
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=2.0,
        seed=7,
        horizon=horizon,
        wait_cost=0.0,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    counts = {"tea": {storage[0]: 5, storage[1]: 5}}
    # 30 movable vertices, the largest window of every arm on this graph
    return WarehouseMAPDMultiAgentEnv(
        graph,
        skus,
        counts,
        {v: 10.0 for v in storage},
        config,
        LocalSubgraphEncodingConfig(max_local_nodes=30, max_messages=1),
        (1, 2),
        pooling="none" if controller == "decentralised" else "mean",
        zones=zones_by_aisle(graph, instance.aisle_index, 2) if controller == "section" else None,
    )


class _Vector:
    """The vector of sub-environments an RLlib env runner holds."""

    def __init__(self, env: WarehouseMAPDMultiAgentEnv) -> None:
        self.envs = [env]


class _Runner:
    def __init__(self, env: _Vector) -> None:
        self.env = env


class _Logger:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def log_value(self, name: str, value: float, reduce: str) -> None:
        self.values.setdefault(name, []).append(value)


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_the_callbacks_check_and_log_every_arm_through_the_core(controller):
    env = _adapter(controller)
    vector, logger = _Vector(env), _Logger()
    callbacks = TrainingProgressCallbacks()
    obs, _ = env.reset()
    callbacks.on_episode_start(episode=None, env=vector, env_index=0)
    rng = np.random.default_rng(0)
    truncated = {"__all__": False}
    while not truncated["__all__"]:
        actions = {aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in obs}
        obs, _, _, truncated, _ = env.step(actions)
        callbacks.on_episode_step(episode=None, env=vector, env_index=0)
    callbacks.on_sample_end(env_runner=_Runner(vector), metrics_logger=logger)
    callbacks.on_episode_end(metrics_logger=logger, env=vector, env_index=0)

    expected = progress_metrics(env.core)
    assert expected["episode_step"] == 12.0
    for name, value in expected.items():
        assert logger.values[name] == [value]
    (fraction,) = logger.values["standing_still_fraction"]
    assert 0.0 <= fraction <= 1.0
    assert logger.values["episode_tasks_completed"] == [expected["tasks_completed"]]


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_the_callbacks_raise_on_a_vertex_conflict_in_the_core(controller):
    env = _adapter(controller)
    vector = _Vector(env)
    callbacks = TrainingProgressCallbacks()
    env.reset()
    callbacks.on_episode_start(episode=None, env=vector, env_index=0)
    locations = env.core.fleet.locations()
    shared = locations[1]
    env.core.fleet = _fleet({aid: shared for aid in locations})  # every robot on one vertex
    env.timestep_completed = True  # as after the last turn of a timestep
    with pytest.raises(RuntimeError, match="vertex rule"):
        callbacks.on_episode_step(episode=None, env=vector, env_index=0)


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_on_episode_end_logs_the_terms_that_add_up_to_the_episode_return(controller):
    env = _adapter(controller)
    vector, logger = _Vector(env), _Logger()
    callbacks = TrainingProgressCallbacks()
    obs, _ = env.reset()
    callbacks.on_episode_start(episode=None, env=vector, env_index=0)
    rng = np.random.default_rng(0)
    truncated = {"__all__": False}
    episode_return = 0.0
    while not truncated["__all__"]:
        actions = {aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in obs}
        obs, rewards, _, truncated, _ = env.step(actions)
        episode_return += sum(rewards.values())
        callbacks.on_episode_step(episode=None, env=vector, env_index=0)
    callbacks.on_episode_end(metrics_logger=logger, env=vector, env_index=0)

    (shaping,) = logger.values["return_shaping"]
    (deliver,) = logger.values["return_deliver"]
    (override,) = logger.values["return_override"]
    (congestion,) = logger.values["return_congestion"]
    assert shaping + deliver + override + congestion == pytest.approx(episode_return, abs=1e-9)
    assert deliver >= 0.0 and override <= 0.0 and congestion <= 0.0

    log = env.core.log
    overrides = sum(log.overrides_by_agent.values())
    agent_steps = 3 * 12
    assert logger.values["override_rate"] == [pytest.approx(overrides / agent_steps)]
    assert logger.values["episode_tasks_released"] == [float(len(env.core.tasks))]
    assert logger.values["episode_tasks_completed"] == [
        float(sum(task.finish_time is not None for task in env.core.tasks))
    ]
    (share,) = logger.values["episode_max_agent_override_share"]
    assert share == (max(log.overrides_by_agent.values()) / overrides if overrides else 0.0)


def test_episode_metrics_max_override_share_is_the_largest_agents_part_of_all_overrides():
    env = _env()
    env.reset()
    log = env.log
    assert episode_metrics(env)["episode_max_agent_override_share"] == 0.0
    assert episode_metrics(env)["override_rate"] == 0.0
    log.overrides_by_agent.update({1: 3, 2: 1, 3: 0})
    env._t = 10  # four agents, ten steps
    metrics = episode_metrics(env)
    assert metrics["episode_max_agent_override_share"] == pytest.approx(3 / 4)
    assert metrics["override_rate"] == pytest.approx(4 / 40)


def test_a_planner_run_leaves_the_reward_term_log_at_zero():
    env = _env()  # a planner config, which has no reward weights
    env.reset()
    planner = PrioritisedAStarPlanner()
    for _ in range(8):
        step_with_planner(env, planner)
    log = env.log
    assert not (log.return_shaping_by_agent or log.return_delivery_by_agent)
    assert not (log.return_override_by_agent or log.return_congestion_by_agent)
    assert not log.return_wait_by_agent
    assert not log.overrides_by_agent
    metrics = episode_metrics(env)
    for key in (
        "return_shaping",
        "return_deliver",
        "return_override",
        "return_wait",
        "return_congestion",
    ):
        assert metrics[key] == 0.0
    assert metrics["override_rate"] == 0.0


# --- what a sample logs ----------------------------------------------------------------


def test_a_sample_logs_the_snapshot_keys_and_no_rates_over_the_fragment():
    env = _adapter("decentralised", horizon=30)
    obs, _ = env.reset()
    vector, logger = _Vector(env), _Logger()
    callbacks = TrainingProgressCallbacks()
    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = {aid: int(rng.choice(np.flatnonzero(obs[aid]["action_mask"]))) for aid in obs}
        obs, *_ = env.step(actions)
        callbacks.on_episode_step(episode=None, env=vector, env_index=0)
    callbacks.on_sample_end(env_runner=_Runner(vector), metrics_logger=logger)
    assert set(logger.values) == set(progress_metrics(env.core)) | {"standing_still_fraction"}
    assert logger.values["episode_step"] == [5.0]
    assert not any(key.startswith("window_") for key in logger.values)
