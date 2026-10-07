"""Runs one episode end-to-end through WarehouseMAPDEnv and reduces its
EpisodeLog + final task list into one RunMetrics row (sec:pf:measures).

The order generator (P_ord) is already seeded separately from the policy
and the environment (environment/multi_agent_env.py's
_ORDER_SEED_OFFSET), so the same order stream replays identically across
controllers/storage rules sharing a seed -- what makes eq:rqformal's
paired comparison meaningful (sec:impl:instances). This module doesn't
re-seed anything itself; it only drives WarehouseMAPDEnv and reads its
already-online-accumulated log at the end.
"""

from __future__ import annotations

import math
from typing import Mapping

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.storage.base import EdgeKey


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
    throughput = num_completed / horizon  # Lambda_T, eq:throughput

    if num_completed:
        service_times = []
        for task in completed:
            assert task.completion_time is not None  # by construction: `completed`'s own filter
            service_times.append(task.completion_time - task.release_time)
        mean_service_time = sum(service_times) / num_completed  # zeta_bar_T, eq:throughput
        movement_cost_per_task = env.log.total_movement_cost / num_completed  # c_bar_T
        mean_blocked_time = (
            sum(env.log.blocked_ticks_by_task.get(task.task_id, 0) for task in completed)
            / num_completed
        )  # W_T, eq:waiting
    else:
        mean_service_time = None
        movement_cost_per_task = None
        mean_blocked_time = None

    num_waiting = sum(1 for task in env.tasks if task.status(horizon) == "waiting")
    num_active = sum(1 for task in env.tasks if task.status(horizon) == "active")
    backlog = env.log.backlog_by_t[-1]  # B_T, eq:functional -- logged online, not recomputed here

    num_traversed, entropy, concentration = _traffic_measures(env.log.edge_traversals)

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
        mean_service_time=mean_service_time,
        movement_cost_per_task=movement_cost_per_task,
        num_waiting_tasks=num_waiting,
        num_active_tasks=num_active,
        backlog=backlog,
        mean_blocked_time=mean_blocked_time,
        num_traversed_edges=num_traversed,
        traffic_entropy=entropy,
        traffic_concentration=concentration,
        mean_decision_runtime_seconds=mean_decision_runtime,
    )


def _traffic_measures(
    edge_traversals: dict[EdgeKey, int],
) -> tuple[int, float | None, float | None]:
    """|E+_T|, H_T, C_T (eq:entropy/eq:concentration), including the
    thesis's own by-convention cases: |E+_T|=0 reported separately (both
    None), |E+_T|=1 maximally concentrated by definition (H_T=0, C_T=1).
    """
    num_traversed = len(edge_traversals)
    if num_traversed == 0:
        return 0, None, None
    if num_traversed == 1:
        return 1, 0.0, 1.0

    total = sum(edge_traversals.values())
    probabilities = [count / total for count in edge_traversals.values()]
    # log's base cancels in the ratio (both num/denom use the same base),
    # so any consistent base works; natural log needs no extra import.
    raw_entropy = -sum(p * math.log(p) for p in probabilities)
    entropy = raw_entropy / math.log(num_traversed)
    return num_traversed, entropy, 1.0 - entropy
