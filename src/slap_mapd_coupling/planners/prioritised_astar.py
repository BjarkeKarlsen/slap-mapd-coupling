"""Prioritised space-time A* planning, the reference planner outside the three controllers."""

from __future__ import annotations

import heapq
from typing import Sequence

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.planners.hops import hop_distances_to

Path = tuple[VertexId, ...]
State = tuple[VertexId, int]  # (vertex, timesteps elapsed since this call's t)


class PrioritisedAStarPlanner:
    """Prioritised planning (space-time A*) over the full graph, the reference planner.

    Agents are planned in a fixed order, earliest t_j^assigned first. Free
    agents come last, ordered by agent_id. Each agent plans around the
    cells already reserved by higher-priority agents. Every call replans
    all routes from scratch and returns only the first action of each, so
    it is a pure function of (graph, fleet, tasks, t). Conflict resolution
    still runs afterwards, so the reservations make its job easy and are
    not the only guard against collisions.

    Design choices the thesis leaves open:

    - The planning horizon is not quantified, so each agent plans to its
      own goal (Task.current_goal, or its own vertex if free) with the
      search bounded by its hop distance plus the number of reserved
      cells. A fixed lookahead would need its own ExperimentConfig
      field (#103).
    - Space-time A* minimises timesteps to goal (the action rule spends
      one timestep per action), not realised movement cost.
    """

    def plan(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        locations = fleet.locations()
        active_by_agent = active_tasks_by_agent(tasks, t)
        order = _priority_order(fleet, active_by_agent)

        reserved_vertices: dict[int, set[VertexId]] = {}
        reserved_edges: dict[int, set[tuple[VertexId, VertexId]]] = {}
        actions: dict[AgentId, Action] = {}
        # Bound on how many offsets a not-moving agent's vertex stays reserved (see _reserve).
        stationary_window = len(graph.vertices)

        for agent_id in order:
            start = locations[agent_id]
            task = active_by_agent.get(agent_id)
            goal = task.current_goal if task is not None else start
            path = _plan(graph, start, goal, reserved_vertices, reserved_edges)
            _reserve(path, reserved_vertices, reserved_edges, stationary_window)
            actions[agent_id] = _first_action(start, path)

        return actions


def _assignment_time(task: Task) -> int:
    assert task.assignment_time is not None  # every active task has t_j^assigned set
    return task.assignment_time


def _priority_order(fleet: FleetState, active_by_agent: dict[AgentId, Task]) -> tuple[AgentId, ...]:
    with_task = sorted(
        (aid for aid in fleet.agents if aid in active_by_agent),
        key=lambda aid: (_assignment_time(active_by_agent[aid]), aid),
    )
    free = sorted(aid for aid in fleet.agents if aid not in active_by_agent)
    return tuple(with_task) + tuple(free)


def _plan(
    graph: WarehouseGraph,
    start: VertexId,
    goal: VertexId,
    reserved_vertices: dict[int, set[VertexId]],
    reserved_edges: dict[int, set[tuple[VertexId, VertexId]]],
) -> Path:
    """Space-time A* from `start` to `goal`, avoiding every already-reserved
    (vertex, offset) and the swap conflicts implied by already-reserved
    (from, to, offset) moves. Falls back to a single-state "stay put" path
    if no route to `goal` is found within the bound (see the class
    docstring).
    """
    if start == goal:
        return (start,)

    hop_dist = hop_distances_to(graph, goal)
    if start not in hop_dist:
        raise KeyError(
            f"{goal} is unreachable from {start}; instance validation "
            "(instances/validation.py) should have caught this at generation time."
        )

    slack = sum(len(v) for v in reserved_vertices.values()) + sum(
        len(e) for e in reserved_edges.values()
    )
    max_offset = hop_dist[start] + slack + 1

    g_score: dict[State, int] = {(start, 0): 0}
    came_from: dict[State, State] = {}
    closed: set[State] = set()
    open_heap: list[tuple[int, int, VertexId]] = [(hop_dist[start], 0, start)]

    while open_heap:
        _, offset, vertex = heapq.heappop(open_heap)
        state: State = (vertex, offset)
        if state in closed:
            continue
        closed.add(state)
        if vertex == goal:
            return _reconstruct(came_from, state)
        if offset >= max_offset:
            continue

        next_offset = offset + 1
        blocked_vertices = reserved_vertices.get(next_offset, set())
        blocked_edges = reserved_edges.get(next_offset, set())

        for successor in (vertex, *graph.out_neighbours(vertex)):
            if successor not in hop_dist:
                continue  # genuinely can't reach goal from here, not worth exploring
            if successor in blocked_vertices:
                continue  # vertex conflict against an already-planned higher-priority agent
            if successor != vertex and (successor, vertex) in blocked_edges:
                continue  # swap conflict, symmetric case
            next_state: State = (successor, next_offset)
            tentative_g = next_offset
            if tentative_g < g_score.get(next_state, max_offset + 1):
                g_score[next_state] = tentative_g
                came_from[next_state] = state
                f_score = tentative_g + hop_dist[successor]
                heapq.heappush(open_heap, (f_score, next_offset, successor))

    return (start,)  # no route found within the bound: wait this timestep


def _reconstruct(came_from: dict[State, State], state: State) -> Path:
    path = [state[0]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
    path.reverse()
    return tuple(path)


def _reserve(
    path: Path,
    reserved_vertices: dict[int, set[VertexId]],
    reserved_edges: dict[int, set[tuple[VertexId, VertexId]]],
    stationary_window: int,
) -> None:
    """Register `path` in the shared reservation tables.

    A length-1 path means this agent isn't moving during this planning
    call at all (genuinely free with nowhere to go, or a failed search
    falling back to waiting) -- reserving only offset 0 for it would let
    a later agent's search treat its vertex as free again one step
    later, which it demonstrably is not: nothing in this call gives that
    agent any reason to move between now and the next replanning call.
    Reserved across `stationary_window` offsets instead, so a
    not-moving agent reliably blocks other plans from routing through
    it, not just for the very first tick.
    """
    if len(path) == 1:
        vertex = path[0]
        for offset in range(stationary_window):
            reserved_vertices.setdefault(offset, set()).add(vertex)
        return

    for offset, vertex in enumerate(path):
        reserved_vertices.setdefault(offset, set()).add(vertex)
    for offset in range(1, len(path)):
        previous, current = path[offset - 1], path[offset]
        if previous != current:
            reserved_edges.setdefault(offset, set()).add((previous, current))


def _first_action(start: VertexId, path: Path) -> Action:
    if len(path) < 2 or path[1] == start:
        return Action(kind="wait")
    return Action(kind="move", target=path[1])
