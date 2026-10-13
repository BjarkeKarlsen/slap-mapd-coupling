"""The Assigner argument: something that picks the tasks in place of the shared rule."""

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.controllers.assignment import Assigner, assign_tasks
from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.core.tasks import Task, TaskId
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.planners import PrioritisedAStarPlanner, TokenPassingPlanner
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.resolution.conflict_resolution import priority_permutation


class _TakesTheNewestTask:
    """Gives every waiting task to agent 1 (the shared rule would spread them out)."""

    def __init__(self) -> None:
        self.resets: list[int] = []
        self.calls: list[str] = []

    def reset(self, episode_seed: int) -> None:
        self.resets.append(episode_seed)
        self.calls.append("reset")

    def assign(
        self, graph: WarehouseGraph, fleet: FleetState, tasks: list[Task], t: int
    ) -> dict[TaskId, AgentId]:
        self.calls.append("assign")
        waiting = [task.task_id for task in tasks if task.status(t) == "waiting"]
        busy = any(task.status(t) == "active" and task.assigned_agent == 1 for task in tasks)
        return {} if busy or not waiting else {waiting[-1]: 1}

    def plan(
        self, graph: WarehouseGraph, fleet: FleetState, tasks: list[Task], t: int
    ) -> dict[AgentId, Action]:
        return {agent_id: Action(kind="wait") for agent_id in fleet.agents}


class _SharedRuleAssigner:
    """Applies the shared rule with the episode's own priority, as no assigner does."""

    def __init__(self, num_agents: int) -> None:
        self._num_agents = num_agents
        self._priority: tuple[AgentId, ...] = ()

    def reset(self, episode_seed: int) -> None:
        self._priority = priority_permutation(episode_seed, tuple(range(1, self._num_agents + 1)))

    def assign(
        self, graph: WarehouseGraph, fleet: FleetState, tasks: list[Task], t: int
    ) -> dict[TaskId, AgentId]:
        return assign_tasks(graph, fleet, tasks, t, self._priority)


def _env(assigner: Assigner | None = None) -> WarehouseMAPDEnv:
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
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config, assigner=assigner)


def _run(env: WarehouseMAPDEnv, steps: int) -> list[tuple[int, int | None]]:
    planner = PrioritisedAStarPlanner()
    for _ in range(steps):
        step_with_planner(env, planner)
    return sorted((task.task_id, task.assigned_agent) for task in env.tasks)


def test_the_stub_is_an_assigner_and_so_is_token_passing_but_prioritised_astar_is_not():
    assert isinstance(_TakesTheNewestTask(), Assigner)
    assert isinstance(TokenPassingPlanner(), Assigner)
    assert not isinstance(PrioritisedAStarPlanner(), Assigner)


def test_an_assigner_replaces_the_shared_assignment():
    planner = _TakesTheNewestTask()
    env = _env(planner)  # the first assignment happens at construction
    _run(env, 5)
    assigned = [task for task in env.tasks if task.assignment_time is not None]
    assert assigned
    assert {task.assigned_agent for task in assigned} == {1}


def test_with_no_assigner_the_shared_rule_with_the_episode_priority_is_used():
    shared = _run(_env(), 6)
    assert shared
    assert {agent for _, agent in shared if agent is not None} != {1}
    assert _run(_env(_SharedRuleAssigner(3)), 6) == shared


def test_the_assigner_is_reset_with_the_episode_seed_on_every_reset():
    planner = _TakesTheNewestTask()
    env = _env(planner)
    assert planner.resets == [env.config.seed]  # the reset inside the constructor
    env.reset(seed=11)
    assert planner.resets == [env.config.seed, 11]
    env.reset(seed=12)
    env.reset()
    assert planner.resets == [env.config.seed, 11, 12, env.config.seed]


def test_the_assigner_is_reset_before_it_assigns_at_t0():
    planner = _TakesTheNewestTask()
    env = _env(planner)
    planner.calls.clear()
    env.reset(seed=11)
    assert planner.calls[:2] == ["reset", "assign"]


def test_run_episode_passes_a_planner_that_picks_its_own_tasks_as_the_assigner():
    # Token Passing needs its own reset before the assignment of t = 0, so the episode
    # equals one driven by hand with the planner passed as the assigner.
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
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5}}
    config = ExperimentConfig(
        storage_mode="fixed",
        planner="token_passing",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=0.5,
        seed=3,
        horizon=40,
        wait_cost=0.0,
        observation_depth=2,
    )
    metrics = run_episode(graph, skus, counts, capacities, config)

    planner = TokenPassingPlanner()
    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config, assigner=planner)
    for _ in range(config.horizon):
        step_with_planner(env, planner)
    assert metrics.num_completed_tasks == sum(1 for t in env.tasks if t.finish_time is not None)
    assert metrics.backlog == env.log.backlog_by_t[-1]
    assert metrics.num_completed_tasks > 0
