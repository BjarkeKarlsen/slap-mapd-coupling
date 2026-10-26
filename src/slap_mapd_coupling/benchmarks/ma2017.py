"""Reproducing Token Passing on the small warehouse of Ma et al. (2017), a check of the simulator
and of planners/token_passing.py against a published result (app:checks).

Their setup (Section 6, Figure 4 and Table 1): a 21 x 35 four-neighbour grid, a sequence of 500
tasks whose pickup and delivery are drawn at random among the task endpoints, added to the task
set in order at a fixed frequency, and the agents' starting cells as the only non-task endpoints.
They report the makespan, the timestep at which the last task is finished, and the service time,
the mean number of timesteps from a task being added to it being finished.

Choices the paper leaves open:

- The map is rebuilt from Figure 4 (`MA2017_MAP`). It has 200 task endpoints, 152 cells agents
  may start on and 635 open cells.
- The paper does not say which starting cells a fleet of fewer than 50 uses, or the sequence
  itself. Both are drawn from the seed, the starting cells uniformly among the 152.
- Task k (counted from 0) is added at the first timestep t with k < (t + 1) f, so at frequency
  0.2 one task every 5 steps and at frequency 10 ten tasks a step.
- A pickup and its delivery are distinct endpoints.

The run is the simulator core with Token Passing as its assigner and planner, exactly as in the
study, only with the task stream replaced (`TaskListEnv`). No storage and no orders.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.grid import START, TASK_ENDPOINT, cells, grid_graph, parse_map
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.planners.token_passing import TokenPassingPlanner
# F_fix registers itself on import, and the core looks it up by name.
from slap_mapd_coupling.storage import fixed as _registers_fixed  # noqa: F401

MA2017_MAP = """
...................................
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.@@@@@@@@@@.@@@@@@@@@@.rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.......................rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.@@@@@@@@@@.@@@@@@@@@@.rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.......................rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.@@@@@@@@@@.@@@@@@@@@@.rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.......................rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.@@@@@@@@@@.@@@@@@@@@@.rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.......................rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
.rr.rr.@@@@@@@@@@.@@@@@@@@@@.rr.rr.
.rr.rr.eeeeeeeeee.eeeeeeeeee.rr.rr.
...................................
"""

NUM_TASKS = 500
FREQUENCIES = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
FLEETS = (10, 20, 30, 40, 50)

# Table 1 of Ma et al. (2017), Token Passing: (frequency, agents) -> (makespan, service time,
# runtime per timestep in ms).
MA2017_TP: dict[tuple[float, int], tuple[int, float, float]] = {
    (0.2, 10): (2532, 38.54, 0.13),
    (0.2, 20): (2540, 39.77, 0.26),
    (0.2, 30): (2546, 38.71, 0.25),
    (0.2, 40): (2540, 38.88, 0.24),
    (0.2, 50): (2540, 40.03, 0.32),
    (0.5, 10): (1309, 132.79, 0.24),
    (0.5, 20): (1094, 42.69, 1.16),
    (0.5, 30): (1069, 43.97, 1.51),
    (0.5, 40): (1090, 43.01, 0.70),
    (0.5, 50): (1083, 43.66, 1.36),
    (1.0, 10): (1198, 311.78, 0.20),
    (1.0, 20): (757, 95.98, 1.03),
    (1.0, 30): (607, 53.80, 2.81),
    (1.0, 40): (624, 48.80, 2.14),
    (1.0, 50): (597, 49.14, 3.76),
    (2.0, 10): (1167, 407.62, 0.19),
    (2.0, 20): (683, 190.76, 0.96),
    (2.0, 30): (529, 114.39, 2.31),
    (2.0, 40): (464, 95.32, 3.43),
    (2.0, 50): (432, 75.63, 6.28),
    (5.0, 10): (1162, 473.78, 0.20),
    (5.0, 20): (655, 247.08, 1.02),
    (5.0, 30): (478, 170.78, 2.22),
    (5.0, 40): (418, 155.33, 4.15),
    (5.0, 50): (395, 124.59, 5.92),
    (10.0, 10): (1163, 495.93, 0.22),
    (10.0, 20): (643, 275.24, 1.09),
    (10.0, 30): (526, 192.01, 1.98),
    (10.0, 40): (407, 154.63, 1.65),
    (10.0, 50): (333, 131.42, 5.62),
}

_SKU: SkuId = "task"  # the one SKU every task names, since Ma et al. have no stock


class PlannedTask(BaseModel):
    """One task of the sequence, before it is added."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: int  # the timestep it joins the task set
    pickup: VertexId
    delivery: VertexId


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frequency: float
    num_agents: int
    seed: int
    finished: int  # tasks finished within the horizon
    makespan: int | None  # None if a task was left unfinished
    service_time: float  # over the finished tasks
    runtime_ms: float  # Token Passing's assignment and planning, per timestep
    num_diverged: int  # see planners/token_passing.py, should stay 0


def instance(num_agents: int, seed: int) -> WarehouseGraph:
    """The graph of Figure 4 with `num_agents` starting cells drawn from the seed."""
    rows = parse_map(MA2017_MAP)
    candidates = cells(rows, START)
    if num_agents > len(candidates):
        raise ValueError(f"{num_agents} agents, but the map has {len(candidates)} starting cells.")
    starts = random.Random(seed).sample(candidates, num_agents)
    return grid_graph(rows, starts)


def task_sequence(frequency: float, seed: int, num_tasks: int = NUM_TASKS) -> list[PlannedTask]:
    """The sequence of tasks, each added at the first t with k < (t + 1) f."""
    endpoints = cells(parse_map(MA2017_MAP), TASK_ENDPOINT)
    rng = random.Random(seed + 1)  # not the stream that drew the starting cells
    rate = Fraction(frequency).limit_denominator(1000)
    sequence = []
    for k in range(num_tasks):
        pickup, delivery = rng.sample(endpoints, 2)
        sequence.append(PlannedTask(added=int(k / rate), pickup=pickup, delivery=delivery))
    return sequence


class TaskListEnv(WarehouseMAPDEnv):
    """The simulator core with its order stream replaced by a fixed task sequence."""

    def __init__(
        self,
        graph: WarehouseGraph,
        sequence: Sequence[PlannedTask],
        config: ExperimentConfig,
        assigner: TokenPassingPlanner,
    ) -> None:
        self._sequence = list(sequence)
        # One SKU with one unit, which the core needs to start. No task takes it.
        shelf = min(v for v, vertex in graph.vertices.items() if vertex.role.storage)
        skus: Mapping[SkuId, SkuType] = {_SKU: SkuType(sku_id=_SKU, unit_capacity=1.0)}
        super().__init__(graph, skus, {_SKU: {shelf: 1}}, {shelf: 1.0}, config, assigner=assigner)

    def _generate_tasks(self) -> list[Task]:
        """Stage 1: the tasks of the sequence added at this timestep."""
        new_tasks = []
        while self._next_task_id <= len(self._sequence):
            planned = self._sequence[self._next_task_id - 1]
            if planned.added > self._t:
                break
            new_tasks.append(
                Task(
                    task_id=self._next_task_id,
                    release_time=self._t,
                    pickup_vertex=planned.pickup,
                    delivery_vertex=planned.delivery,
                    sku=_SKU,
                )
            )
            self._next_task_id += 1
        return new_tasks


def config_for(num_agents: int, frequency: float, seed: int, horizon: int) -> ExperimentConfig:
    """A run config for the core. The arrival rate is not read, since the sequence replaces
    the order stream, and the observation depth only sets the crowding measure."""
    return ExperimentConfig(
        storage_mode="fixed",
        planner="token_passing",
        congestion_sensitive=False,
        communication=False,
        num_agents=num_agents,
        arrival_rate=frequency,
        seed=seed,
        horizon=horizon,
        wait_cost=1.0,
        observation_depth=1,
    )


def run(num_agents: int, frequency: float, seed: int, horizon: int = 10_000) -> RunResult:
    """One run, until every task is finished or the horizon is reached."""
    graph = instance(num_agents, seed)
    sequence = task_sequence(frequency, seed)
    planner = TokenPassingPlanner()
    env = TaskListEnv(graph, sequence, config_for(num_agents, frequency, seed, horizon), planner)
    while env.t < horizon:
        step_with_planner(env, planner)
        if len(env.tasks) == len(sequence) and all(t.finish_time is not None for t in env.tasks):
            break
    finished = [task for task in env.tasks if task.finish_time is not None]
    # Token Passing's own time: the assignment (the token round) and the planning of the step.
    planner_seconds = sum(env.log.assignment_runtime_seconds) + sum(env.log.routing_runtime_seconds)
    return RunResult(
        frequency=frequency,
        num_agents=num_agents,
        seed=seed,
        finished=len(finished),
        makespan=(
            max(task.finish_time for task in finished)  # type: ignore[type-var]
            if len(finished) == len(sequence)
            else None
        ),
        service_time=(
            sum(task.finish_time - task.release_time for task in finished)  # type: ignore[operator]
            / len(finished)
            if finished
            else float("nan")
        ),
        runtime_ms=1000.0 * planner_seconds / max(env.t, 1),
        num_diverged=planner.num_diverged,
    )
