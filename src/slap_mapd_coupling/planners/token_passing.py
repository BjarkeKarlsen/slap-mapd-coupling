"""Token Passing, the reference baseline (Ma et al. 2017).

A token holds every agent's committed path. An agent whose path has ended asks
for the token, one agent at a time in agent id order (Ma et al. leave the order
open). The agent then

1. takes the available task nearest to it and plans a path vertex, pickup,
   delivery. A task is available if no other agent's path ends at its pickup
   or at its delivery vertex. Ties go to the lower task id.
2. if there is no available task and no waiting task has its delivery vertex
   where the agent stands, stays where it is.
3. otherwise goes to the nearest free non-task endpoint.

Each path is a space-time A* path against the paths already in the token, where
every other agent is taken to stand at the last vertex of its path for ever. So a
path never ends where another agent may need to pass, which is the reason for the
rules above and what makes the algorithm complete on well-formed instances
(sec:pf:env). Each step the planner returns the next step of every committed path.

It picks its own tasks (SelfAssigningPlanner), because the guarantee depends on
that task rule, so it replaces the shared pi_assign for the run. The thesis
assigns tasks the same way for all controllers, and the baseline is outside them.

Choices the paper leaves open, and the ones that differ from it:

- Agents ask in agent id order.
- A vertex is taken at the step an agent is on it, the step before it arrives and
  the step after it leaves. Two plans therefore never rely on one agent entering a
  vertex as another leaves it, which conflict resolution would block when the
  leaving agent has the lower priority (sec:method:resolution). Conflict
  resolution never has to override a plan, and a test asserts it.
- "Nearest" is the hop distance, since an action takes one timestep whatever the
  edge cost, and the search minimises timesteps like prioritised A*.
- A non-task endpoint is a vertex in V_ep that is not also a storage or delivery
  vertex. It must not be the end of another path or the pickup or delivery vertex
  of a waiting task.
- The environment picks a task up at the first timestep an agent is on its pickup
  vertex after a move, and completes it on first arriving at the delivery vertex.
  A path therefore reaches the pickup at a step after the first and ends at its
  first arrival at the delivery vertex.
- If an agent is not where its path says, which conflict resolution should not
  cause, it drops the path and replans from where it stands. `num_diverged`
  counts these, and it stays 0 in the tests.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Mapping, Sequence

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId, active_tasks_by_agent
from slap_mapd_coupling.planners.hops import hop_distances_to

Path = tuple[VertexId, ...]
# A path and the timestep it starts at. path[k] is the vertex at timestep t0 + k, and the
# agent stays on path[-1] after that.
Committed = tuple[int, Path]


class TokenPassingPlanner:
    """Token Passing, a SelfAssigningPlanner. See the module docstring."""

    def __init__(self) -> None:
        self._paths: dict[AgentId, Committed] = {}
        self._graph: WarehouseGraph | None = None
        self._hops_to: dict[VertexId, dict[VertexId, int]] = {}
        self.num_diverged = 0
        # Per task, the timesteps it waited while an agent was idle and another path ended at
        # its (pickup, delivery) vertex. A diagnostic of the task rule, read by the evaluator.
        self.endpoint_wait: dict[TaskId, tuple[int, int]] = {}

    def reset(self, episode_seed: int) -> None:
        """Empty the token. Nothing here is random, so the seed is not used."""
        self._paths = {}
        self._graph = None
        self._hops_to = {}
        self.num_diverged = 0
        self.endpoint_wait = {}

    def assign(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[TaskId, AgentId]:
        """The token round at time t. Commits a path for every agent that asked."""
        self._use(graph)
        locations = fleet.locations()
        self._sync(locations, t)
        held = active_tasks_by_agent(tasks, t)
        unassigned = sorted(
            (task for task in tasks if task.status(t) == "waiting"), key=lambda task: task.task_id
        )

        assignments: dict[TaskId, AgentId] = {}
        idle = False  # an agent asked, held no task and found none it could take
        for agent_id in sorted(locations):
            if not self._asks(agent_id, t):
                continue
            here = locations[agent_id]
            task = held.get(agent_id)
            if task is not None:
                # Only after a dropped path, since a path ends where its task is completed.
                goals = (
                    (task.delivery_vertex,)
                    if task.pickup_time is not None
                    else (
                        task.pickup_vertex,
                        task.delivery_vertex,
                    )
                )
                self._commit(agent_id, t, self._plan(agent_id, here, goals, t, True) or (here,))
                continue
            chosen = self._take_a_task(agent_id, here, unassigned, t)
            if chosen is not None:
                task, path = chosen
                assignments[task.task_id] = agent_id
                unassigned.remove(task)
                self._commit(agent_id, t, path)
            else:
                idle = True
                self._commit(agent_id, t, self._stay_or_park(agent_id, here, unassigned, t))
        if idle:
            self._count_endpoint_wait(unassigned)
        return assignments

    def _count_endpoint_wait(self, unassigned: Sequence[Task]) -> None:
        """Count a timestep for every task still waiting whose pickup or delivery vertex is
        where a path ends, the reason rule 1 kept it from the idle agent."""
        ends = {path[-1] for _, path in self._paths.values()}
        for task in unassigned:
            pickup = task.pickup_vertex in ends
            delivery = task.delivery_vertex in ends
            if pickup or delivery:
                at_pickup, at_delivery = self.endpoint_wait.get(task.task_id, (0, 0))
                self.endpoint_wait[task.task_id] = (at_pickup + pickup, at_delivery + delivery)

    def plan(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        """The next step of every committed path. Run `assign` first at each timestep,
        since an agent with no committed path waits."""
        actions: dict[AgentId, Action] = {}
        for agent_id, here in fleet.locations().items():
            actions[agent_id] = self._next_action(agent_id, here, t)
        return actions

    # -- the token ----------------------------------------------------------------

    def _use(self, graph: WarehouseGraph) -> None:
        if graph is not self._graph:
            self._graph = graph
            self._hops_to = {}

    def _sync(self, locations: Mapping[AgentId, VertexId], t: int) -> None:
        """Give an agent with no path a path that stays where it is, and replace the
        path of an agent that is not where its path says."""
        for agent_id, here in locations.items():
            entry = self._paths.get(agent_id)
            if entry is None:
                self._paths[agent_id] = (t, (here,))
            elif self._expected(entry, t) != here:
                self.num_diverged += 1
                self._paths[agent_id] = (t, (here,))

    @staticmethod
    def _expected(entry: Committed, t: int) -> VertexId:
        t0, path = entry
        return path[min(max(t - t0, 0), len(path) - 1)]

    def _asks(self, agent_id: AgentId, t: int) -> bool:
        t0, path = self._paths[agent_id]
        return t - t0 >= len(path) - 1

    def _commit(self, agent_id: AgentId, t: int, path: Path) -> None:
        self._paths[agent_id] = (t, path)

    def _next_action(self, agent_id: AgentId, here: VertexId, t: int) -> Action:
        entry = self._paths.get(agent_id)
        if entry is None or self._expected(entry, t) != here:
            return Action(kind="wait")
        t0, path = entry
        k = t - t0
        if k + 1 >= len(path) or path[k + 1] == here:
            return Action(kind="wait")
        return Action(kind="move", target=path[k + 1])

    def _path_ends(self, agent_id: AgentId) -> set[VertexId]:
        """The last vertex of every other agent's path."""
        return {path[-1] for other, (_, path) in self._paths.items() if other != agent_id}

    # -- choosing what an asking agent does ----------------------------------------

    def _take_a_task(
        self, agent_id: AgentId, here: VertexId, unassigned: Sequence[Task], t: int
    ) -> tuple[Task, Path] | None:
        ends = self._path_ends(agent_id)
        available = [
            task
            for task in unassigned
            if task.pickup_vertex not in ends and task.delivery_vertex not in ends
        ]
        available.sort(key=lambda task: (self._hop(here, task.pickup_vertex), task.task_id))
        for task in available:
            path = self._plan(agent_id, here, (task.pickup_vertex, task.delivery_vertex), t, True)
            if path is not None:
                return task, path
        return None

    def _stay_or_park(
        self, agent_id: AgentId, here: VertexId, unassigned: Sequence[Task], t: int
    ) -> Path:
        if not any(task.delivery_vertex == here for task in unassigned):
            return (here,)
        assert self._graph is not None
        ends = self._path_ends(agent_id)
        task_vertices = {
            v for task in unassigned for v in (task.pickup_vertex, task.delivery_vertex)
        }
        parking = sorted(
            (
                v
                for v, vertex in self._graph.vertices.items()
                if vertex.role.endpoint
                and not vertex.role.storage
                and not vertex.role.delivery
                and not vertex.role.receiving
                and v not in ends
                and v not in task_vertices
            ),
            key=lambda v: (self._hop(here, v), v),
        )
        for vertex in parking:
            path = self._plan(agent_id, here, (vertex,), t, False)
            if path is not None:
                return path
        return (here,)

    def _hop(self, source: VertexId, goal: VertexId) -> int:
        """Hops from source to goal. A vertex that cannot reach goal is infinitely far."""
        return self._distances_to(goal).get(source, 1 << 30)

    def _distances_to(self, goal: VertexId) -> dict[VertexId, int]:
        assert self._graph is not None
        if goal not in self._hops_to:
            self._hops_to[goal] = hop_distances_to(self._graph, goal)
        return self._hops_to[goal]

    # -- space-time A* against the token --------------------------------------------

    def _plan(
        self,
        agent_id: AgentId,
        start: VertexId,
        goals: tuple[VertexId, ...],
        t: int,
        completes_task: bool,
    ) -> Path | None:
        """Shortest path from `start` through `goals` in order, to be followed from time t,
        that no other agent's committed path conflicts with. The path ends at the last
        goal, where the agent then stays for ever, so that vertex must be free of
        every other path from then on. None if there is no such path.

        A goal counts as reached only at a step after the first, because the
        environment checks pickup and delivery on the location after a move
        (see the module docstring). With `completes_task` the path must also end on
        first reaching the last goal, since the task completes there.
        """
        assert self._graph is not None
        graph = self._graph
        to_goal = [self._distances_to(goal) for goal in goals]
        if start not in to_goal[0]:
            return None
        # Hops left once goal k is reached, for the heuristic.
        legs_after = [0] * len(goals)
        for k in range(len(goals) - 2, -1, -1):
            legs_after[k] = legs_after[k + 1] + to_goal[k + 1].get(goals[k], 1 << 30)

        blocked, horizon, resting, last_blocked = self._reservations(agent_id, t)
        max_offset = horizon + len(graph.vertices) + legs_after[0] + to_goal[0][start] + 1

        def h(vertex: VertexId, stage: int) -> int:
            if stage == len(goals):
                return 0
            return to_goal[stage].get(vertex, 1 << 30) + legs_after[stage]

        State = tuple[VertexId, int, int]  # (vertex, offset from t, goals reached)
        first: State = (start, 0, 0)
        came_from: dict[State, State] = {}
        closed: set[State] = set()
        heap: list[tuple[int, int, int, VertexId]] = [(h(start, 0), 0, 0, start)]
        best: dict[State, int] = {first: 0}

        while heap:
            _, offset, stage, vertex = heapq.heappop(heap)
            state: State = (vertex, offset, stage)
            if state in closed:
                continue
            closed.add(state)
            if stage == len(goals):
                return self._rebuild(came_from, state)
            if offset >= max_offset:
                continue
            next_offset = offset + 1
            taken = blocked(next_offset)
            for successor in (vertex, *graph.out_neighbours(vertex)):
                if successor in taken or h(successor, stage) >= 1 << 30:
                    continue
                next_stage = stage
                if successor == goals[stage]:
                    next_stage = stage + 1
                    while next_stage < len(goals) and goals[next_stage] == successor:
                        next_stage += 1  # pickup and delivery on one vertex
                    if next_stage == len(goals) and not (
                        successor not in resting and next_offset > last_blocked.get(successor, 0)
                    ):
                        # Reached the last goal but could not stay there. A task would
                        # complete on arrival, and a parking move should not touch it.
                        if completes_task:
                            continue
                        next_stage = stage
                nxt: State = (successor, next_offset, next_stage)
                if next_offset < best.get(nxt, max_offset + 1):
                    best[nxt] = next_offset
                    came_from[nxt] = state
                    heapq.heappush(
                        heap,
                        (
                            next_offset + h(successor, next_stage),
                            next_offset,
                            next_stage,
                            successor,
                        ),
                    )
        return None

    def _reservations(
        self, agent_id: AgentId, t: int
    ) -> tuple[Callable[[int], set[VertexId]], int, set[VertexId], dict[VertexId, int]]:
        """What the other agents' committed paths take, in steps from t.

        Returns a function giving the vertices that are taken at an offset, the
        offset from which only the vertices every other path ends on are taken,
        those end vertices, and for each vertex the last offset it is taken at
        before that. A vertex is taken at an offset if another agent is on it then,
        was on it the offset before or is on it the offset after, so that no plan has an
        agent entering a vertex in the step another leaves it, whichever of the two was
        planned first.
        """
        others = [(t0, path) for other, (t0, path) in self._paths.items() if other != agent_id]
        horizon = max((t0 + len(path) - 1 - t for t0, path in others), default=0) + 1

        def at(entry: Committed, offset: int) -> VertexId:
            t0, path = entry
            return path[min(max(t + offset - t0, 0), len(path) - 1)]

        taken: dict[int, set[VertexId]] = {}
        last_blocked: dict[VertexId, int] = {}
        for offset in range(1, horizon + 1):
            vertices: set[VertexId] = set()
            for entry in others:
                vertices.add(at(entry, offset - 1))
                vertices.add(at(entry, offset))
                vertices.add(at(entry, offset + 1))
            taken[offset] = vertices
            for vertex in vertices:
                last_blocked[vertex] = offset
        resting = {path[-1] for _, path in others}

        def blocked(offset: int) -> set[VertexId]:
            return taken[offset] if offset <= horizon else resting

        return blocked, horizon, resting, last_blocked

    @staticmethod
    def _rebuild(came_from: dict[Any, Any], state: Any) -> Path:
        path = [state[0]]
        while state in came_from:
            state = came_from[state]
            path.append(state[0])
        path.reverse()
        return tuple(path)
