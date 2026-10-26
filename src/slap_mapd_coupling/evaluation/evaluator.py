"""Runs one episode end-to-end through WarehouseMAPDEnv and reduces its
EpisodeLog + final task list into one RunMetrics row (sec:pf:measures).

The order generator (P_ord) is seeded separately from the policy and the
environment (environment/multi_agent_env.py's _ORDER_SEED_OFFSET), so the same
order stream replays identically across controllers and storage rules that
share a seed. That is what makes the paired comparison of the gain test
meaningful (sec:impl:instances, evaluation/comparison.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Sequence

from slap_mapd_coupling.controllers.assignment import Assigner
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.core.tasks import Task, TaskId
from slap_mapd_coupling.environment.multi_agent_env import OrderLine, WarehouseMAPDEnv
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
    timing_path: Path | None = None,
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

    With `timing_path` one JSON line is appended there with the wall-clock seconds and calls of
    every part of the step (`episode_timing`). Engineering only, it changes no result.
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
    metrics = _to_run_metrics(env, config)
    if timing_path is not None:
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(episode_timing(env, config)) + "\n")
    return metrics


def episode_timing(env: WarehouseMAPDEnv, config: ExperimentConfig) -> dict[str, Any]:
    """The timing log of one episode: who ran, then seconds and calls per part of the step.
    Routing is whatever produced the actions (planning, or observations and forward pass for a
    learned arm), and `observations` is the observation build alone, inside routing."""
    log = env.log
    return {
        "controller": config.router_name,
        "storage_mode": config.storage_mode,
        "num_agents": config.num_agents,
        "arrival_rate": config.arrival_rate,
        "seed": config.seed,
        "horizon": config.horizon,
        "seconds": {
            **log.stage_seconds,
            "routing": sum(log.routing_runtime_seconds),
            "assignment": sum(log.assignment_runtime_seconds),
        },
        "calls": {
            **log.stage_calls,
            "routing": len(log.routing_runtime_seconds),
            "assignment": len(log.assignment_runtime_seconds),
        },
    }


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
    that advances it one timestep through the adapter and the policy. A timestep is
    one or more turns, so the function steps the adapter until it has run the core."""
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
        while True:
            observations = adapter.step(policy(observations, adapter.core.t))[0]
            if adapter.timestep_completed:
                return

    return adapter.core, step


def completed_share(num_completed: int, backlog: int) -> float | None:
    """|C_T| / (|C_T| + B_T), the share of the released tasks that finished. None when
    nothing was released."""
    released = num_completed + backlog
    return num_completed / released if released else None


def keeps_up(num_completed: int, backlog: int, keep_up_threshold: float | None) -> bool | None:
    """One part of the keep-up check, eq:throughput, for one kind of task:
    |C_T| / T >= f_up (|C_T| + B_T) / T. Both sides are multiplied by T, so no rounding of the
    division decides a run on the threshold. None unless f_up is set."""
    if keep_up_threshold is None:
        return None
    return num_completed >= keep_up_threshold * (num_completed + backlog)


def keep_up_parts(
    completed_picks: int,
    pick_backlog: int,
    completed_putaways: int,
    putaway_backlog: int,
    keep_up_threshold: float | None,
) -> tuple[bool | None, bool | None, bool | None]:
    """The keep-up check of eq:throughput, applied to picks and to put-aways separately
    ([A59]): (keeps up, picks pass, put-aways pass). The run keeps up iff both parts pass. With
    no put-away released the put-away part passes. All three None unless f_up is set."""
    picks = keeps_up(completed_picks, pick_backlog, keep_up_threshold)
    putaways = keeps_up(completed_putaways, putaway_backlog, keep_up_threshold)
    if picks is None or putaways is None:
        return None, None, None
    return picks and putaways, picks, putaways


def row_keeps_up(row: Mapping[str, Any], keep_up_threshold: float | None) -> bool | None:
    """The keep-up check of [A59] read off a saved row, so a sweep can apply a threshold to
    rows it wrote earlier. None for a row with put-aways but no `putaway_backlog`, one written
    before [A59]. Its put-aways released and not completed cannot stand in, because they count
    the put-aways of an arrival at T, which B_T leaves out, and on the 2026-10-22 sweep that
    turned 20 passing seeds into 1. A row without put-aways is a template row, checked over
    every task."""
    completed_putaways = row.get("num_completed_putaways", 0)
    if "putaway_backlog" in row:
        putaway_backlog = row["putaway_backlog"]
    elif row.get("num_released_putaways", 0) == 0:
        putaway_backlog = 0
    else:
        return None
    pick_backlog = max(row["backlog"] - putaway_backlog, 0)
    return keep_up_parts(
        row["num_completed_tasks"] - completed_putaways,
        pick_backlog,
        completed_putaways,
        putaway_backlog,
        keep_up_threshold,
    )[0]


def backlog_slope(backlog_by_t: Sequence[int]) -> float | None:
    """The least-squares slope of B_t over the second half of the trace, per 100 timesteps
    ([A55]). None with fewer than two points."""
    half = list(backlog_by_t[len(backlog_by_t) // 2 :])
    n = len(half)
    if n < 2:
        return None
    mean_x = (n - 1) / 2
    mean_y = sum(half) / n
    spread = sum((x - mean_x) ** 2 for x in range(n))
    return 100 * sum((x - mean_x) * (y - mean_y) for x, y in enumerate(half)) / spread


def robot_busy_share(tasks: Sequence[Task], num_agents: int, horizon: int) -> float:
    """The share of agent-timesteps in [0, T) in which an agent holds an active task. An agent
    holds at most one active task at a time, so the tasks' active spans add up."""
    if num_agents == 0 or horizon == 0:
        return 0.0
    busy = 0
    for task in tasks:
        if task.assignment_time is None:
            continue
        end = horizon if task.finish_time is None else min(task.finish_time, horizon)
        busy += max(end - task.assignment_time, 0)
    return busy / (num_agents * horizon)


def endpoint_wait_metrics(
    endpoint_wait: Mapping[TaskId, tuple[int, int]] | None,
    picks: Sequence[Task],
    putaways: Sequence[Task],
) -> dict[str, float | None]:
    """Token Passing's endpoint wait, averaged over the completed picks and put-aways, split by
    the end that was taken. Every value None without a Token Passing assigner, or without a
    completed task of that kind."""

    def means(tasks: Sequence[Task]) -> tuple[float | None, float | None]:
        if endpoint_wait is None or not tasks:
            return None, None
        counts = [endpoint_wait.get(task.task_id, (0, 0)) for task in tasks]
        return (
            sum(c[0] for c in counts) / len(tasks),
            sum(c[1] for c in counts) / len(tasks),
        )

    pick_pickup, pick_delivery = means(picks)
    putaway_pickup, putaway_delivery = means(putaways)
    return dict(
        mean_pick_wait_pickup_taken=pick_pickup,
        mean_pick_wait_delivery_taken=pick_delivery,
        mean_putaway_wait_pickup_taken=putaway_pickup,
        mean_putaway_wait_delivery_taken=putaway_delivery,
    )


def order_metrics(
    tasks: Sequence[Task],
    order_line_by_task: Mapping[TaskId, tuple[int, int]],
    waiting_lines: Sequence[OrderLine],
    lines_waited: Collection[tuple[int, int]],
) -> dict[str, Any]:
    """The order measures of the arrival model. An order is complete when none of its lines
    still waits and all its tasks are completed. Its completion time is the longest service time
    among its tasks, the order completion time. Tasks per line are counted over the lines with
    no units still missing."""
    waiting_orders = {line.order_id for line in waiting_lines}
    waiting = {(line.order_id, line.line_index) for line in waiting_lines}
    tasks_by_order: dict[int, list[Task]] = {}
    tasks_per_line: dict[tuple[int, int], int] = {}
    for task in tasks:
        order_line = order_line_by_task.get(task.task_id)
        if order_line is None:
            continue
        tasks_by_order.setdefault(order_line[0], []).append(task)
        if order_line not in waiting:
            tasks_per_line[order_line] = tasks_per_line.get(order_line, 0) + 1
    completions = []
    for order_id, order_tasks in tasks_by_order.items():
        if order_id in waiting_orders:
            continue
        times = [task.service_time for task in order_tasks]
        if all(time is not None for time in times):
            completions.append(max(time for time in times if time is not None))
    return dict(
        mean_order_completion=sum(completions) / len(completions) if completions else None,
        num_completed_orders=len(completions),
        mean_tasks_per_line=(
            sum(tasks_per_line.values()) / len(tasks_per_line) if tasks_per_line else None
        ),
        num_lines_waited=len(lines_waited),
    )


def total_service(
    tasks: Sequence[Task],
    blocked_ticks_by_task: Mapping[TaskId, int],
    waiting_lines: Sequence[OrderLine],
    horizon: int,
) -> dict[str, int]:
    """The total service time zeta^tot_T and its three parts. Every task released by T adds
    min(t^finish, T) - t^released, split into the wait up to its assignment or T, then its
    blocked ticks and its moving ticks up to T. Every line still waiting for stock at T adds
    T - its release, all of it wait. `total_service_time_finished` sums the completed tasks
    only."""
    wait = travel = blocked = finished = 0
    for task in tasks:
        end = horizon if task.finish_time is None else min(task.finish_time, horizon)
        assigned = horizon if task.assignment_time is None else min(task.assignment_time, horizon)
        ticks = blocked_ticks_by_task.get(task.task_id, 0)
        wait += assigned - task.release_time
        blocked += ticks
        travel += end - assigned - ticks
        if task.finish_time is not None and task.finish_time <= horizon:
            finished += task.finish_time - task.release_time
    wait += sum(horizon - line.release_time for line in waiting_lines)
    return dict(
        total_service_time=wait + travel + blocked,
        total_wait_for_agent=wait,
        total_travel_time=travel,
        total_blocked_time=blocked,
        total_service_time_finished=finished,
    )


def _to_run_metrics(env: WarehouseMAPDEnv, config: ExperimentConfig) -> RunMetrics:
    horizon = config.horizon
    completed = [task for task in env.tasks if task.finish_time is not None]
    num_completed = len(completed)

    # The tasks released in the run, for the keep-up check below, and throughput, reported.
    backlog = env.log.backlog_by_t[-1]  # B_T -- logged online, not recomputed here
    share = completed_share(num_completed, backlog)
    # Stalls: the longest time an agent holding a task stood still in a row.
    longest_stand_still = max(env.log.longest_blocked_run_by_task.values(), default=0)
    throughput = num_completed / horizon  # Lambda_T

    # Mean service time and its split, eq:meanservice / eq:split, over the completed picks.
    # After assignment an agent either moves or stands still at every timestep, and every
    # standing-still tick is logged as blocked, so travel is what is left of the active
    # phase -- no separate travel counter is needed.
    picks = [task for task in completed if task.kind == "pick"]
    putaways = [task for task in completed if task.kind == "putaway"]
    num_picks = len(picks)
    mean_wait: float | None = None
    mean_travel: float | None = None
    mean_blocked: float | None = None
    mean_service_time: float | None = None
    if num_picks:
        waits, travels, blocks = [], [], []
        for task in picks:
            # by construction: a completed task was assigned first (core/tasks.py)
            assert task.finish_time is not None and task.assignment_time is not None
            blocked = env.log.blocked_ticks_by_task.get(task.task_id, 0)
            waits.append(task.assignment_time - task.release_time)
            travels.append(task.finish_time - task.assignment_time - blocked)
            blocks.append(blocked)
        mean_wait = sum(waits) / num_picks
        mean_travel = sum(travels) / num_picks
        mean_blocked = sum(blocks) / num_picks  # W_T, eq:waiting
        mean_service_time = (sum(waits) + sum(travels) + sum(blocks)) / num_picks

    # B_t is logged before stage 1 and stage 6 release tasks at t, so the put-aways of an
    # arrival at T are not in B_T, and they are left out here too. No pick is released at T.
    num_waiting = sum(
        1
        for task in env.tasks
        if task.release_time < horizon and task.status(horizon) == "waiting"
    )
    num_active = sum(1 for task in env.tasks if task.status(horizon) == "active")
    # B^put_T, the put-aways among them, counted the same way ([A59]). The rest of B_T, with the
    # orders waiting for stock, is B^pick_T.
    putaway_backlog = sum(
        1
        for task in env.tasks
        if task.kind == "putaway"
        and task.release_time < horizon
        and task.status(horizon) in ("waiting", "active")
    )
    keep_up, keep_up_picks, keep_up_putaways = keep_up_parts(
        num_picks,
        backlog - putaway_backlog,
        len(putaways),
        putaway_backlog,
        config.keep_up_threshold,
    )
    # Under the arrival model, orders released and still waiting for stock. They count in B_T.
    # Always 0 under the template model.
    num_waiting_orders = len(env.waiting_orders)

    # The mean crowding: the mean of delta_i(t) over agents and
    # timesteps, recorded every timestep for every controller.
    crowding_sums = env.log.crowding_sum_by_t
    mean_crowding = sum(crowding_sums) / (config.num_agents * len(crowding_sums))

    # Relocations per storage update, eq:relocation.
    relocations = env.log.relocated_units_by_update
    mean_relocated = sum(relocations) / len(relocations) if relocations else 0.0
    distances = env.log.relocation_distance_by_update
    mean_relocation_distance = sum(distances) / len(distances) if distances else 0.0

    # The arrival stock model: the wait for stock, part of the wait before assignment, and
    # the put-away of each arrival. The template model logs none of them and keeps the
    # defaults, so its rows are what they were before these fields existed.
    arrival_fields: dict[str, Any] = {}
    num_storage_updates = len(relocations)
    if config.stock_model == "arrival":
        arrived = env.log.arrived_units_by_update
        num_storage_updates = len(arrived)  # the arrivals, under every storage rule

        def per_arrival(values: Sequence[float]) -> float:
            return sum(values) / num_storage_updates if num_storage_updates else 0.0

        arrival_fields = dict(
            mean_stock_wait=(
                sum(env.log.stock_wait_by_task[task.task_id] for task in picks) / num_picks
                if num_picks
                else None
            ),
            num_waiting_orders=num_waiting_orders,
            num_released_putaways=sum(1 for task in env.tasks if task.kind == "putaway"),
            num_completed_putaways=len(putaways),
            mean_putaway_service_time=(
                sum(task.service_time or 0 for task in putaways) / len(putaways)
                if putaways
                else None
            ),
            mean_arrived_units=per_arrival(arrived),
            mean_split_skus=per_arrival(env.log.split_skus_by_update),
            unplaced_units=sum(env.log.unplaced_units_by_update),
            overflow_units=sum(env.log.overflow_units_by_update),
            num_refills=sum(env.log.refills_by_update),
            num_refilled_skus=len(env.log.refilled_skus),
            mean_cold_prime_share=(
                sum(env.log.cold_prime_share_by_update) / len(env.log.cold_prime_share_by_update)
                if env.log.cold_prime_share_by_update
                else None
            ),
            peak_shelf_fill_share=max(env.log.shelf_fill_peak_by_update, default=None),
            mean_shelf_fill_share=(
                sum(env.log.shelf_fill_mean_by_update) / len(env.log.shelf_fill_mean_by_update)
                if env.log.shelf_fill_mean_by_update
                else None
            ),
            mean_unit_age_at_pick=(
                env.log.picked_unit_age_sum / env.log.picked_units
                if env.log.picked_units
                else None
            ),
            **order_metrics(
                env.tasks, env.log.order_line_by_task, env.waiting_orders, env.log.lines_waited
            ),
        )
        arrival_fields["orders_per_timestep"] = arrival_fields["num_completed_orders"] / horizon

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
        throughput_per_agent=throughput / config.num_agents if config.num_agents else 0.0,
        completed_share=share,
        keeps_up=keep_up,
        keeps_up_picks=keep_up_picks,
        keeps_up_putaways=keep_up_putaways,
        longest_stand_still=longest_stand_still,
        stalled=None if config.stall_length is None else longest_stand_still > config.stall_length,
        num_waiting_tasks=num_waiting,
        num_active_tasks=num_active,
        backlog=backlog,
        putaway_backlog=putaway_backlog,
        backlog_slope=backlog_slope(env.log.backlog_by_t),
        mean_service_time=mean_service_time,
        mean_wait_for_agent=mean_wait,
        mean_travel_time=mean_travel,
        mean_blocked_time=mean_blocked,
        mean_crowding=mean_crowding,
        robot_busy_share=robot_busy_share(env.tasks, config.num_agents, horizon),
        **endpoint_wait_metrics(getattr(env.assigner, "endpoint_wait", None), picks, putaways),
        num_storage_updates=num_storage_updates,
        mean_relocated_units=mean_relocated,
        mean_relocation_distance=mean_relocation_distance,
        mean_decision_runtime_seconds=mean_decision_runtime,
        stock_model=config.stock_model,
        **total_service(env.tasks, env.log.blocked_ticks_by_task, env.waiting_orders, horizon),
        **arrival_fields,
    )
