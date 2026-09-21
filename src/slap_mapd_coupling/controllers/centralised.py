"""Centralised controller: prioritised planning (space-time A*) over the full graph
(sec:method:controllers).
"""

from __future__ import annotations

import heapq
from typing import Sequence

from slap_mapd_coupling.controllers.registry import register_controller
from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task

Path = tuple[VertexId, ...]
State = tuple[VertexId, int]  # (vertex, timesteps elapsed since this call's t)


@register_controller("centralised")
class CentralisedController:
    """pi^route: prioritised planning (space-time A*) over the full graph
    (sec:method:controllers). Agents are planned in a fixed order --
    earliest y_j first, i.e. whichever active task each agent currently
    serves was assigned soonest; free agents (no active task) have no
    task-driven urgency the thesis defines an order for, so they're
    planned last, ordered by agent_id for determinism. Each agent after
    the first plans around the space-time cells already reserved by
    higher-priority agents.

    A pure function of (graph, fleet, tasks, t), same as assignment.py
    and resolution/ -- this controller replans every agent's full route
    from scratch on every call rather than caching a multi-step plan
    across timesteps, and only the first action of each plan is actually
    returned. Downstream, resolution/conflict_resolution.py's operator
    still runs on the joint action this produces (sec:method:overview's
    stage 4 always follows stage 3, regardless of which controller is
    active), so this planner's job is to make that operator's job easy
    (ideally a no-op), not to be the sole source of collision-freedom.

    Flagged design choices, neither pinned down by the thesis text:

    - "The current planning horizon" (sec:method:controllers) is never
      quantified anywhere in the problem formulation or method sections,
      and there is no ExperimentConfig field for it. Each agent is
      planned to its own goal (Task.current_goal, or its own vertex if
      free -- it simply waits) rather than to a fixed lookahead window;
      the space-time search per agent is bounded by that agent's
      unobstructed hop-distance to its goal plus the number of cells
      currently reserved by higher-priority agents (a standard, generous
      termination bound for windowed/cooperative A* -- enough slack to
      wait out any blockage the reservations so far could cause). If a
      bounded lookahead window turns out to matter for the results, it
      belongs on ExperimentConfig as its own named parameter, not hidden
      inside this controller.
    - "Space-time A*" is read as the standard MAPF sense: minimising
      timesteps to goal (eq:actions spends exactly one timestep per
      action, wait or move, regardless of edge cost c(v,w)), not
      minimising eq:onestepcost's realised movement cost -- that is a
      separate accounting dimension (sec:impl:cost), not this search's
      metric, and the thesis's own eq:assignmentrule already uses d_G
      (cost) for a different purpose (ranking candidate agents), not for
      timing a route.
    """

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        locations = fleet.locations()
        active_by_agent = _active_tasks_by_agent(tasks, t)
        order = _priority_order(fleet, active_by_agent)

        reserved_vertices: dict[int, set[VertexId]] = {}
        reserved_edges: dict[int, set[tuple[VertexId, VertexId]]] = {}
        actions: dict[AgentId, Action] = {}
        # A generous, deterministic bound on how many offsets a
        # not-moving agent's vertex needs to stay reserved for -- see
        # _reserve's docstring for why this matters at all.
        stationary_window = len(graph.vertices)

        for agent_id in order:
            start = locations[agent_id]
            task = active_by_agent.get(agent_id)
            goal = task.current_goal if task is not None else start
            path = _plan(graph, start, goal, reserved_vertices, reserved_edges)
            _reserve(path, reserved_vertices, reserved_edges, stationary_window)
            actions[agent_id] = _first_action(start, path)

        return actions


def _active_tasks_by_agent(tasks: Sequence[Task], t: int) -> dict[AgentId, Task]:
    active: dict[AgentId, Task] = {}
    for task in tasks:
        if task.status(t) == "active":
            assert task.assigned_agent is not None  # guaranteed by Task's own validator
            active[task.assigned_agent] = task
    return active


def _assignment_time(task: Task) -> int:
    assert task.assignment_time is not None  # every active task has y_j set
    return task.assignment_time


def _priority_order(fleet: FleetState, active_by_agent: dict[AgentId, Task]) -> tuple[AgentId, ...]:
    with_task = sorted(
        (aid for aid in fleet.agents if aid in active_by_agent),
        key=lambda aid: (_assignment_time(active_by_agent[aid]), aid),
    )
    free = sorted(aid for aid in fleet.agents if aid not in active_by_agent)
    return tuple(with_task) + tuple(free)


def _hop_distances_to(graph: WarehouseGraph, goal: VertexId) -> dict[VertexId, int]:
    """Unweighted (timestep) shortest-hop distance from every vertex to
    goal, over the graph reversed -- the admissible/consistent heuristic
    for the space-time search below. See the class docstring for why
    this uses hop count, not d_G's edge cost."""
    reverse: dict[VertexId, list[VertexId]] = {v: [] for v in graph.vertices}
    for edge in graph.edges:
        reverse[edge.target].append(edge.source)

    dist = {goal: 0}
    frontier = [goal]
    while frontier:
        next_frontier = []
        for vertex in frontier:
            for predecessor in reverse[vertex]:
                if predecessor not in dist:
                    dist[predecessor] = dist[vertex] + 1
                    next_frontier.append(predecessor)
        frontier = next_frontier
    return dist


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
    if no route to `goal` is found within the bound (see class docstring);
    the caller's own reservation of that fallback still keeps later,
    lower-priority agents' searches honest about where this agent is.
    """
    if start == goal:
        return (start,)

    hop_dist = _hop_distances_to(graph, goal)
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
                continue  # eq:vertexconflict against an already-planned higher-priority agent
            if successor != vertex and (successor, vertex) in blocked_edges:
                continue  # eq:swapconflict, symmetric case
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
