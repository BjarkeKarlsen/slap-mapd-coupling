"""Runs one episode end-to-end through WarehouseMAPDEnv and reduces its
EpisodeLog + final task list into one RunMetrics row (sec:pf:measures).

The order generator (P_ord) is already seeded separately from the policy
and the environment (environment/multi_agent_env.py's
_ORDER_SEED_OFFSET), so the same order stream replays identically across
controllers/storage rules sharing a seed -- what makes eq:rqformal's
paired comparison meaningful (sec:impl:instances, evaluation/comparison.py).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping

from slap_mapd_coupling.controllers.assignment import Assigner
from slap_mapd_coupling.controllers.base import Controller
from slap_mapd_coupling.controllers.registry import get_controller
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.evaluation.trace import EpisodeTraceWriter
from slap_mapd_coupling.planners import get_planner
from slap_mapd_coupling.planners.base import Planner
from slap_mapd_coupling.planners.driver import step_with_planner


def run_episode(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    config: ExperimentConfig,
    trace_path: Path | None = None,
) -> RunMetrics:
    """Run one full episode (config.horizon timesteps) and reduce it to
    one RunMetrics row.

    A planner config drives the env with planners/driver.py. A planner that
    picks its own tasks is passed to the env as its assigner, and the env resets it
    with the episode seed. A controller config goes through the registry, and the
    controller is reset with the episode seed before stepping, so its
    action-sampling stream replays bit-for-bit under a fixed seed.

    With `trace_path` the episode also writes a per-step trace there
    (evaluation/trace.py). The writer only reads the environment, so the
    returned RunMetrics are the same with or without it.
    """
    controller: Controller | None = None
    planner: Planner | None = None
    assigner: Assigner | None = None
    if config.planner is not None:
        planner = get_planner(config.planner)
        if isinstance(planner, Assigner):
            assigner = planner
    else:
        assert config.controller is not None
        controller = get_controller(config.controller)
        controller.reset(config.seed)
    env = WarehouseMAPDEnv(
        graph, skus, initial_storage_counts, storage_capacities, config, assigner=assigner
    )

    def step() -> None:
        if planner is not None:
            step_with_planner(env, planner)
        else:
            assert controller is not None
            _step_with_controller(env, controller)

    if trace_path is None:
        for _ in range(config.horizon):
            step()
    else:
        with EpisodeTraceWriter(trace_path, env) as writer:
            for _ in range(config.horizon):
                step()
                writer.record(env)
    return _to_run_metrics(env, config)


def _step_with_controller(env: WarehouseMAPDEnv, controller: Controller) -> None:
    """Temporary, removed together with the controller registry. Converts the
    controller's `route` output to slots, times it and steps the env."""
    start = time.perf_counter()
    actions = controller.route(env.graph, env.fleet, env.tasks, env.t)
    locations = env.fleet.locations()
    slots = {
        agent_id: slot_for_action(env.graph, locations[agent_id], action)
        for agent_id, action in actions.items()
    }
    env.log.record_routing_runtime(time.perf_counter() - start)
    env.step(slots)


def _to_run_metrics(env: WarehouseMAPDEnv, config: ExperimentConfig) -> RunMetrics:
    horizon = config.horizon
    completed = [task for task in env.tasks if task.finish_time is not None]
    num_completed = len(completed)

    # Keeping up, eq:throughput.
    throughput = num_completed / horizon  # Lambda_T
    keep_up_ratio = throughput / config.arrival_rate
    keeps_up = (
        None
        if config.keep_up_threshold is None
        else throughput >= config.keep_up_threshold * config.arrival_rate
    )

    # The result and its split, eq:meanservice / eq:split. After assignment
    # an agent either moves or stands still at every timestep, and every
    # standing-still tick is logged as blocked, so travel is what is left
    # of the active phase -- no separate travel counter is needed.
    mean_wait: float | None = None
    mean_travel: float | None = None
    mean_blocked: float | None = None
    mean_service_time: float | None = None
    if num_completed:
        waits, travels, blocks = [], [], []
        for task in completed:
            # by construction: a completed task was assigned first (core/tasks.py)
            assert task.finish_time is not None and task.assignment_time is not None
            blocked = env.log.blocked_ticks_by_task.get(task.task_id, 0)
            waits.append(task.assignment_time - task.release_time)
            travels.append(task.finish_time - task.assignment_time - blocked)
            blocks.append(blocked)
        mean_wait = sum(waits) / num_completed
        mean_travel = sum(travels) / num_completed
        mean_blocked = sum(blocks) / num_completed  # W_T, eq:waiting
        mean_service_time = (sum(waits) + sum(travels) + sum(blocks)) / num_completed

    num_waiting = sum(1 for task in env.tasks if task.status(horizon) == "waiting")
    num_active = sum(1 for task in env.tasks if task.status(horizon) == "active")
    backlog = env.log.backlog_by_t[-1]  # B_T -- logged online, not recomputed here

    # The mean crowding: the mean of delta_i(t) over agents and
    # timesteps, recorded every timestep for every controller.
    crowding_sums = env.log.crowding_sum_by_t
    mean_crowding = sum(crowding_sums) / (config.num_agents * len(crowding_sums))

    # Relocations per storage update, eq:relocation.
    relocations = env.log.relocated_units_by_update
    mean_relocated = sum(relocations) / len(relocations) if relocations else 0.0

    total_runtime = sum(env.log.assignment_runtime_seconds) + sum(env.log.routing_runtime_seconds)
    mean_decision_runtime = total_runtime / horizon  # kappa_T, eq:runtime

    return RunMetrics(
        storage_mode=config.storage_mode,
        controller=config.router_name,
        congestion_sensitive=config.congestion_sensitive,
        communication=config.communication,
        num_agents=config.num_agents,
        arrival_rate=config.arrival_rate,
        seed=config.seed,
        horizon=horizon,
        observation_depth=config.observation_depth,
        num_completed_tasks=num_completed,
        throughput=throughput,
        keep_up_ratio=keep_up_ratio,
        keeps_up=keeps_up,
        num_waiting_tasks=num_waiting,
        num_active_tasks=num_active,
        backlog=backlog,
        mean_service_time=mean_service_time,
        mean_wait_for_agent=mean_wait,
        mean_travel_time=mean_travel,
        mean_blocked_time=mean_blocked,
        mean_crowding=mean_crowding,
        num_storage_updates=len(relocations),
        mean_relocated_units=mean_relocated,
        mean_decision_runtime_seconds=mean_decision_runtime,
    )
