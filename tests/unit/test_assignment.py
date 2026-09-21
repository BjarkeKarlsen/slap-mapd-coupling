"""Unit tests for slap_mapd_coupling.controllers.assignment."""

from slap_mapd_coupling.controllers.assignment import assign_tasks
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


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


def _task(task_id: int, release_time: int, pickup: int, delivery: int = 99) -> Task:
    return Task(
        task_id=task_id,
        release_time=release_time,
        pickup_vertex=pickup,
        delivery_vertex=delivery,
        sku="sku-1",
    )


def test_nearest_free_agent_is_assigned():
    graph = _line_graph(10)
    fleet = _fleet({1: 0, 2: 5})
    task = _task(task_id=1, release_time=0, pickup=4)

    result = assign_tasks(graph, fleet, [task], t=0)
    assert result == {1: 2}  # agent 2 (at 5) is closer to pickup 4 than agent 1 (at 0)


def test_matched_agent_removed_from_free_pool_for_next_task():
    graph = _line_graph(10)
    fleet = _fleet({1: 0, 2: 9})
    tasks = [
        _task(task_id=1, release_time=0, pickup=0),
        _task(task_id=2, release_time=1, pickup=1),
    ]

    result = assign_tasks(graph, fleet, tasks, t=1)
    assert result == {1: 1, 2: 2}


def test_tasks_processed_oldest_release_time_first():
    graph = _line_graph(10)
    fleet = _fleet({1: 5})  # only one free agent
    tasks = [
        _task(task_id=10, release_time=2, pickup=0),
        _task(task_id=20, release_time=0, pickup=9),
    ]

    result = assign_tasks(graph, fleet, tasks, t=2)
    assert result == {20: 1}  # older task (release_time=0) wins the only free agent


def test_no_free_agents_leaves_tasks_unassigned():
    graph = _line_graph(5)
    fleet = _fleet({1: 0})
    active = _task(task_id=1, release_time=0, pickup=1).assign(agent_id=1, t=0)
    waiting = _task(task_id=2, release_time=0, pickup=2)

    result = assign_tasks(graph, fleet, [active, waiting], t=0)
    assert result == {}  # agent 1 is busy with task 1, no one free for task 2


def test_already_assigned_and_completed_tasks_are_not_reassigned():
    graph = _line_graph(5)
    fleet = _fleet({1: 0})
    completed = (
        _task(task_id=1, release_time=0, pickup=1)
        .assign(agent_id=1, t=0)
        .pick_up(t=0)
        .complete(t=1)
    )
    waiting = _task(task_id=2, release_time=1, pickup=2)

    result = assign_tasks(graph, fleet, [completed, waiting], t=1)
    assert result == {2: 1}


def test_tie_broken_by_agent_id_ascending():
    graph = _line_graph(5)
    fleet = _fleet({2: 0, 1: 0})  # both agents equidistant from the pickup
    task = _task(task_id=1, release_time=0, pickup=1)

    result = assign_tasks(graph, fleet, [task], t=0)
    assert result == {1: 1}
