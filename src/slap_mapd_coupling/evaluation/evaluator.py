"""Runs one episode end-to-end through WarehouseMAPDEnv and reduces its
EpisodeLog + final task list into one RunMetrics row (sec:pf:measures).

The order generator (P_ord) is already seeded separately from the policy
and the environment (environment/multi_agent_env.py's
_ORDER_SEED_OFFSET), so the same order stream replays identically across
controllers/storage rules sharing a seed -- what makes eq:rqformal's
paired comparison meaningful (sec:impl:instances, evaluation/comparison.py).
This module doesn't re-seed anything itself; it only drives
WarehouseMAPDEnv and reads its already-online-accumulated log at the end.
"""

from __future__ import annotations

from typing import Mapping

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.metrics import RunMetrics


def run_episode(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    config: ExperimentConfig,
) -> RunMetrics:
    """Run one full episode (config.horizon timesteps) and reduce it to
    one RunMetrics row."""
    env = WarehouseMAPDEnv(graph, skus, initial_storage_counts, storage_capacities, config)
    env.reset()
    for _ in range(config.horizon):
        env.step()
    return _to_run_metrics(env, config)


def _to_run_metrics(env: WarehouseMAPDEnv, config: ExperimentConfig) -> RunMetrics:
    horizon = config.horizon
    completed = [task for task in env.tasks if task.completion_time is not None]
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
            assert task.completion_time is not None and task.assignment_time is not None
            blocked = env.log.blocked_ticks_by_task.get(task.task_id, 0)
            waits.append(task.assignment_time - task.release_time)
            travels.append(task.completion_time - task.assignment_time - blocked)
            blocks.append(blocked)
        mean_wait = sum(waits) / num_completed
        mean_travel = sum(travels) / num_completed
        mean_blocked = sum(blocks) / num_completed  # W_T, eq:waiting
        mean_service_time = (sum(waits) + sum(travels) + sum(blocks)) / num_completed

    num_waiting = sum(1 for task in env.tasks if task.status(horizon) == "waiting")
    num_active = sum(1 for task in env.tasks if task.status(horizon) == "active")
    backlog = env.log.backlog_by_t[-1]  # B_T -- logged online, not recomputed here

    # Crowding, eq:crowding: the mean of delta_i(t) over agents and timesteps.
    crowding_sums = env.log.crowding_sum_by_t
    mean_crowding = (
        sum(crowding_sums) / (config.num_agents * len(crowding_sums)) if crowding_sums else None
    )

    # Relocations per storage update, eq:relocation.
    relocations = env.log.relocated_units_by_update
    mean_relocated = sum(relocations) / len(relocations) if relocations else 0.0

    total_runtime = sum(env.log.assignment_runtime_seconds) + sum(env.log.routing_runtime_seconds)
    mean_decision_runtime = total_runtime / horizon  # kappa_T, eq:runtime

    return RunMetrics(
        storage_mode=config.storage_mode,
        controller=config.controller,
        congestion_sensitive=config.congestion_sensitive,
        communication=config.communication,
        num_agents=config.num_agents,
        arrival_rate=config.arrival_rate,
        seed=config.seed,
        horizon=horizon,
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
