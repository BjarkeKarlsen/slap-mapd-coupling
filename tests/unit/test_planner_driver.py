"""Unit tests for slap_mapd_coupling.planners.driver."""

import pytest

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import action_for_slot
from slap_mapd_coupling.planners import PrioritisedAStarPlanner
from slap_mapd_coupling.planners.driver import plan_slots, step_with_planner


class _ScriptedPlanner:
    """Plays back the action each agent is told to take."""

    def __init__(self, actions: dict[int, Action]) -> None:
        self._actions = actions

    def plan(self, graph, fleet, tasks, t) -> dict[int, Action]:
        return dict(self._actions)


def _hub_env(*at: int, arrival_rate: float = 1e-9) -> WarehouseMAPDEnv:
    """One agent per vertex in `at`, placed there. Vertex 0 is a hub with four neighbours,
    so it has a wait slot and four move slots."""
    roles = {
        0: VertexRole(),
        1: VertexRole(endpoint=True),
        2: VertexRole(endpoint=True),
        3: VertexRole(storage=True),
        4: VertexRole(delivery=True),
    }
    vertices = {v: Vertex(id=v, role=role) for v, role in roles.items()}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0) for v in (1, 2, 3, 4) for a, b in ((0, v), (v, 0))
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    config = ExperimentConfig(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=len(at),
        arrival_rate=arrival_rate,
        seed=0,
        horizon=10,
        wait_cost=0.0,
        observation_depth=2,
    )
    env = WarehouseMAPDEnv(graph, skus, {"tea": {3: 5}}, {3: 10.0}, config)
    env.fleet = FleetState(
        agents={i: AgentState(agent_id=i, location=v) for i, v in enumerate(at, start=1)}
    )
    return env


def test_plan_slots_converts_wait_and_every_move_to_its_slot():
    env = _hub_env(0)
    legal = env.graph.legal_actions(0)
    assert len(legal) == 5  # wait and the four moves
    for slot, action in enumerate(legal):
        slots = plan_slots(_ScriptedPlanner({1: action}), env)
        assert slots == {1: slot}
        assert action_for_slot(env.graph, 0, slots[1]) == action


def test_plan_slots_converts_every_agent_at_its_own_vertex():
    env = _hub_env(0, 1)
    planner = _ScriptedPlanner(
        {1: Action(kind="move", target=3), 2: Action(kind="move", target=0)}
    )
    slots = plan_slots(planner, env)
    assert action_for_slot(env.graph, 0, slots[1]) == Action(kind="move", target=3)
    assert action_for_slot(env.graph, 1, slots[2]) == Action(kind="move", target=0)


def test_plan_slots_rejects_an_action_that_is_not_legal_at_the_vertex():
    env = _hub_env(0)
    with pytest.raises(ValueError):
        plan_slots(_ScriptedPlanner({1: Action(kind="move", target=99)}), env)


def test_plan_slots_asks_the_planner_for_the_state_the_env_holds():
    env = _hub_env(0)
    seen = []

    class _Spy(_ScriptedPlanner):
        def plan(self, graph, fleet, tasks, t):
            seen.append((graph, fleet, tasks, t))
            return super().plan(graph, fleet, tasks, t)

    plan_slots(_Spy({1: Action(kind="wait")}), env)
    assert seen == [(env.graph, env.fleet, env.tasks, env.t)]


def test_step_with_planner_steps_the_env_and_records_one_routing_time():
    env = _hub_env(0)
    env.log.routing_runtime_seconds.clear()
    target = Action(kind="move", target=3)
    observations, *_ = step_with_planner(env, _ScriptedPlanner({1: target}))
    assert observations[1] == 3
    assert env.t == 1
    assert len(env.log.routing_runtime_seconds) == 1
    assert env.log.routing_runtime_seconds[0] >= 0.0
    step_with_planner(env, _ScriptedPlanner({1: Action(kind="wait")}))
    assert len(env.log.routing_runtime_seconds) == 2


def test_the_planners_own_actions_reach_conflict_resolution_unchanged():
    env = _hub_env(0, 1, arrival_rate=2.0)
    planner = PrioritisedAStarPlanner()
    moved = False
    for _ in range(8):
        expected = planner.plan(env.graph, env.fleet, env.tasks, env.t)
        step_with_planner(env, planner)
        assert env.last_proposed == expected
        moved = moved or any(action.kind == "move" for action in expected.values())
    assert moved  # the orders gave the planner something to do
