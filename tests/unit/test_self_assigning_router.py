"""The hook that lets a planner pick its own tasks and keep state across steps."""

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.controllers.base import SelfAssigningRouter
from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.core.tasks import Task, TaskId
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.planners import PrioritisedAStarPlanner, router_for


class _TakesTheNewestTask:
    """Gives every waiting task to agent 1 (the shared rule would spread them out)."""

    def __init__(self) -> None:
        self.resets: list[int] = []

    def reset(self, episode_seed: int) -> None:
        self.resets.append(episode_seed)

    def assign(
        self, graph: WarehouseGraph, fleet: FleetState, tasks: list[Task], t: int
    ) -> dict[TaskId, AgentId]:
        waiting = [task.task_id for task in tasks if task.status(t) == "waiting"]
        busy = any(task.status(t) == "active" and task.assigned_agent == 1 for task in tasks)
        return {} if busy or not waiting else {waiting[-1]: 1}

    def plan(
        self, graph: WarehouseGraph, fleet: FleetState, tasks: list[Task], t: int
    ) -> dict[AgentId, Action]:
        return {agent_id: Action(kind="wait") for agent_id in fleet.agents}


def _env() -> WarehouseMAPDEnv:
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
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    config = ExperimentConfig(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=2.0,
        seed=3,
        horizon=10,
        wait_cost=0.0,
        observation_depth=2,
    )
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config)


def test_a_router_that_picks_its_own_tasks_replaces_the_shared_assignment():
    env = _env()
    planner = _TakesTheNewestTask()
    router = router_for(planner)
    assert isinstance(router, SelfAssigningRouter)
    env._controller = router
    for _ in range(5):
        env.step()
    assigned = [task for task in env.tasks if task.assignment_time is not None]
    assert assigned
    assert {task.assigned_agent for task in assigned} == {1}


def test_a_planner_with_no_assign_keeps_the_shared_assignment():
    router = router_for(PrioritisedAStarPlanner())
    assert not isinstance(router, SelfAssigningRouter)
    env = _env()
    for _ in range(5):
        env.step()
    assert {task.assigned_agent for task in env.tasks if task.assigned_agent} != {1}


def test_env_reset_resets_the_router_with_the_episode_seed():
    env = _env()
    planner = _TakesTheNewestTask()
    env._controller = router_for(planner)
    env.step()
    env.reset(seed=11)
    assert planner.resets == [11]
    env.reset()
    assert planner.resets == [11, env.config.seed]
