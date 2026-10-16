"""Runs one episode end-to-end through WarehouseMAPDEnv and reduces its
EpisodeLog + final task list into one RunMetrics row (sec:pf:measures).

The order generator (P_ord) is seeded separately from the policy and the
environment (environment/multi_agent_env.py's _ORDER_SEED_OFFSET), so the same
order stream replays identically across controllers and storage rules that
share a seed. That is what makes the paired comparison of the gain test
meaningful (sec:impl:instances, evaluation/comparison.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from slap_mapd_coupling.controllers.assignment import Assigner
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.evaluation.trace import EpisodeTraceWriter
from slap_mapd_coupling.planners import get_planner
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.planners.registry import PlannerName

if TYPE_CHECKING:  # a planner run must not import Ray
    from slap_mapd_coupling.models.gnn_encoder import Pooling
    from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
    from slap_mapd_coupling.training.policy import ModulePolicy


def run_episode(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    config: ExperimentConfig,
    trace_path: Path | None = None,
    *,
    policy: ModulePolicy | None = None,
    encoding_config: LocalSubgraphEncodingConfig | None = None,
    zones: Sequence[frozenset[VertexId]] | None = None,
    pooling: Pooling | None = None,
) -> RunMetrics:
    """Run one full episode (config.horizon timesteps) and reduce it to
    one RunMetrics row.

    A planner config drives the env with planners/driver.py. A planner that
    picks its own tasks is passed to the env as its assigner, and the env resets it
    with the episode seed. Nothing else is read for a planner config, and Ray is
    not imported.

    A controller config needs `policy`, `encoding_config` and `pooling`, or it
    raises ValueError. `zones` is needed for the section arm only. The episode
    runs through the adapter that trains the arm (training/env.py), built with
    the one evaluation seed `config.seed` and reset with it, so the policy sees
    the observations it saw in training. `encoding_config` and `pooling` must be
    those the module was trained with. The policy is reset with the episode seed, so its
    sampling replays bit for bit. The policy samples a legal slot with a seeded stream per robot,
    and whether evaluation should take the most likely slot instead is still open.

    With `trace_path` the episode also writes a per-step trace there
    (evaluation/trace.py). The writer only reads the environment, so the
    returned RunMetrics are the same with or without it.
    """
    if config.planner is not None:
        env, step = _planner_run(
            graph, skus, initial_storage_counts, storage_capacities, config, config.planner
        )
    else:
        env, step = _controller_run(
            graph,
            skus,
            initial_storage_counts,
            storage_capacities,
            config,
            policy,
            encoding_config,
            zones,
            pooling,
        )

    if trace_path is None:
        for _ in range(config.horizon):
            step()
    else:
        with EpisodeTraceWriter(trace_path, env) as writer:
            for _ in range(config.horizon):
                step()
                writer.record(env)
    return _to_run_metrics(env, config)


def _planner_run(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    config: ExperimentConfig,
    planner_name: PlannerName,
) -> tuple[WarehouseMAPDEnv, Callable[[], None]]:
    """The simulator core of a planner run, and a function that advances it one
    step with the planner driver. A planner that picks its own tasks is the
    core's assigner."""
    planner = get_planner(planner_name)
    assigner = planner if isinstance(planner, Assigner) else None
    env = WarehouseMAPDEnv(
        graph, skus, initial_storage_counts, storage_capacities, config, assigner=assigner
    )
    return env, lambda: step_with_planner(env, planner)


def _controller_run(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    config: ExperimentConfig,
    policy: ModulePolicy | None,
    encoding_config: LocalSubgraphEncodingConfig | None,
    zones: Sequence[frozenset[VertexId]] | None,
    pooling: Pooling | None,
) -> tuple[WarehouseMAPDEnv, Callable[[], None]]:
    """The simulator core of a learned arm, reset for the episode, and a function
    that advances it one step through the adapter and the policy."""
    if policy is None or encoding_config is None or pooling is None:
        raise ValueError(
            "a controller config needs policy, encoding_config and pooling, the ones the "
            "module was trained with."
        )
    from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv

    adapter = WarehouseMAPDMultiAgentEnv(
        graph,
        skus,
        initial_storage_counts,
        storage_capacities,
        config,
        encoding_config,
        (config.seed,),
        pooling=pooling,
        zones=zones,
    )
    observations: dict[str, Any]
    observations, _ = adapter.reset(seed=config.seed)
    # The reset made a new log, and the policy records its time on it.
    policy.reset(config.seed, record_runtime=adapter.core.log.record_routing_runtime)

    def step() -> None:
        nonlocal observations
        observations = adapter.step(policy(observations, adapter.core.t))[0]

    return adapter.core, step


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

    # Routing time is whatever produced the actions: planning for a planner, and the
    # observation build, forward pass and sampling for a learned arm.
    total_runtime = sum(env.log.assignment_runtime_seconds) + sum(env.log.routing_runtime_seconds)
    mean_decision_runtime = total_runtime / horizon  # kappa_T, the mean runtime

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
