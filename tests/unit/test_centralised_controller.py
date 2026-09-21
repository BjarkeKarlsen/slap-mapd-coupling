"""Unit tests for slap_mapd_coupling.controllers.centralised."""

from slap_mapd_coupling.controllers.centralised import CentralisedController
from slap_mapd_coupling.core.agents import AgentState, FleetState, is_collision_free
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.resolution.conflict_resolution import (
    priority_permutation,
    resolve_conflicts,
)


def _line_graph(n: int) -> WarehouseGraph:
    """A path 0-1-2-...-(n-1), bidirectional, unit edge costs."""
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


def _task(task_id: int, pickup: int, delivery: int, assignment_time: int = 0) -> Task:
    return Task(
        task_id=task_id,
        release_time=0,
        pickup_vertex=pickup,
        delivery_vertex=delivery,
        sku="sku-1",
    ).assign(agent_id=task_id, t=assignment_time)


def test_free_agent_with_no_task_waits():
    graph = _line_graph(5)
    fleet = _fleet({1: 2})
    actions = CentralisedController().route(graph, fleet, tasks=[], t=0)
    assert actions[1].kind == "wait"


def test_agent_heads_towards_pickup_before_pickup():
    graph = _line_graph(5)
    fleet = _fleet({1: 0})
    task = Task(task_id=1, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    actions = CentralisedController().route(graph, fleet, tasks=[task], t=0)
    assert actions[1] == graph.legal_actions(0)[1]  # move towards vertex 1, on the way to 4


def test_agent_heads_towards_delivery_after_pickup():
    graph = _line_graph(5)
    fleet = _fleet({1: 4})  # agent already at the pickup vertex
    task = (
        Task(task_id=1, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1")
        .assign(agent_id=1, t=0)
        .pick_up(t=0)
    )
    actions = CentralisedController().route(graph, fleet, tasks=[task], t=0)
    assert actions[1].kind == "move" and actions[1].target == 3  # towards delivery at 0


def test_agent_already_at_goal_waits():
    graph = _line_graph(5)
    fleet = _fleet({1: 4})
    task = Task(task_id=1, release_time=0, pickup_vertex=4, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    actions = CentralisedController().route(graph, fleet, tasks=[task], t=0)
    assert actions[1].kind == "wait"


def test_earlier_assignment_time_has_priority():
    # Two agents converge on a single vertex from opposite ends of a line;
    # whichever task was assigned first should get the direct route and
    # the other should be planned around it (wait or detour), not both
    # naively colliding.
    graph = _line_graph(3)  # 0 - 1 - 2
    fleet = _fleet({1: 0, 2: 2})
    early = Task(task_id=1, release_time=0, pickup_vertex=1, delivery_vertex=1, sku="a").assign(
        agent_id=1, t=0
    )
    late = Task(task_id=2, release_time=0, pickup_vertex=1, delivery_vertex=1, sku="a").assign(
        agent_id=2, t=5
    )
    actions = CentralisedController().route(graph, fleet, tasks=[early, late], t=5)

    assert actions[1].kind == "move" and actions[1].target == 1  # early agent moves in directly
    assert actions[2].kind == "wait"  # late agent yields rather than also moving to vertex 1


def test_free_agents_planned_after_agents_with_tasks_break_ties_by_id():
    graph = _line_graph(3)
    fleet = _fleet({5: 0, 1: 2})  # agent 5 is free, agent 1 has a task
    task = Task(task_id=1, release_time=0, pickup_vertex=1, delivery_vertex=1, sku="a").assign(
        agent_id=1, t=0
    )
    actions = CentralisedController().route(graph, fleet, tasks=[task], t=0)
    assert actions[1].kind == "move" and actions[1].target == 1
    assert actions[5].kind == "wait"  # free agent has no goal, stays put


def test_routed_actions_are_already_collision_free_without_resolution_overrides():
    # A corridor where two agents' tasks pull them toward the same vertex;
    # the planner's own reservations should avoid any conflict, so
    # resolve_conflicts should not need to override anyone.
    graph = _line_graph(5)
    fleet = _fleet({1: 0, 2: 4})
    tasks = [
        _task(task_id=1, pickup=2, delivery=2, assignment_time=0),
        _task(task_id=2, pickup=2, delivery=2, assignment_time=1),
    ]
    before = fleet
    actions = CentralisedController().route(graph, before, tasks, t=1)

    priority = priority_permutation(seed=0, agent_ids=list(before.agents))
    result = resolve_conflicts(graph, before, actions, priority)

    assert is_collision_free(before, result.fleet)
    assert result.overridden == frozenset()


def test_full_multi_step_run_stays_collision_free():
    # Run several timesteps of route() -> resolve_conflicts() in a small
    # grid-like corridor with three agents on overlapping errands, and
    # check every single step of the realised trace for collisions -- the
    # correctness invariant AGENTS.md treats as non-negotiable.
    graph = _line_graph(6)
    fleet = _fleet({1: 0, 2: 5, 3: 2})
    tasks = [
        _task(task_id=1, pickup=5, delivery=0, assignment_time=0),
        _task(task_id=2, pickup=0, delivery=5, assignment_time=1),
        _task(task_id=3, pickup=3, delivery=3, assignment_time=2),
    ]
    controller = CentralisedController()
    priority = priority_permutation(seed=7, agent_ids=list(fleet.agents))

    state = fleet
    for t in range(10):
        proposed = controller.route(graph, state, tasks, t)
        result = resolve_conflicts(graph, state, proposed, priority)
        assert is_collision_free(state, result.fleet)
        state = result.fleet
