"""Shared greedy task assignment pi_assign (the assignment rule, sec:method:controllers).

Identical across all three controller architectures. At every timestep the
free agents (the free-agent condition) are matched to the tasks in Q_t
(released and not yet assigned, see the task stages) ordered by
t_j^released ascending. Each task is matched greedily to the free agent
minimising d_G(l_i(t), s_j), and that agent leaves the free pool before the
next task is considered. This is a greedy heuristic for bipartite matching,
not the Hungarian algorithm's optimum, so swapping in an optimal matcher
changes this function and nothing in Pi.

Ties keep the rule a function of the state. Tasks released in the same
timestep are taken by task index, and free agents at equal distance are
ranked by the episode's priority order of the conflict resolution operator
(sec:method:resolution), earlier agent first.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId, free_agents


@runtime_checkable
class Assigner(Protocol):
    """Something that takes the place of the shared rule in stage 2.

    Token Passing picks its own tasks, because its guarantee depends on its own
    task rule. The environment resets the assigner with the episode seed before
    the assignment of t = 0, and asks it for the assignment of every timestep.
    """

    def reset(self, episode_seed: int) -> None:
        """Forget what was kept from the last episode."""
        ...

    def assign(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[TaskId, AgentId]:
        """task_id -> agent_id for every task taken at time t. Does not mutate a Task."""
        ...


def assign_tasks(
    graph: WarehouseGraph,
    fleet: FleetState,
    tasks: Iterable[Task],
    t: int,
    priority: Sequence[AgentId],
) -> dict[TaskId, AgentId]:
    """pi_assign: task_id -> agent_id for every task matched this timestep.

    Only tasks in Q_t (task.status(t) == "waiting") are candidates. Tasks
    already assigned or completed are untouched (Task enforces "no
    re-tasking", core/tasks.py). The caller applies the returned mapping via
    Task.assign(agent_id, t), and this function does not mutate any Task.

    Tasks released in the same timestep are taken by task index. Free agents
    at equal distance to a pickup are ranked by priority, the episode's
    priority order of the conflict resolution operator, so the earlier agent
    wins. Every free agent must appear in priority, as in resolve_conflicts.
    """
    rank = {agent_id: k for k, agent_id in enumerate(priority)}
    waiting = sorted(
        (task for task in tasks if task.status(t) == "waiting"),
        key=lambda task: (task.release_time, task.task_id),
    )
    free = free_agents(fleet, tasks, t)
    locations = fleet.locations()

    assignments: dict[TaskId, AgentId] = {}
    for task in waiting:
        if not free:
            break
        best_agent = min(
            free,
            key=lambda agent_id: (
                graph.distance(locations[agent_id], task.pickup_vertex),
                rank[agent_id],
            ),
        )
        assignments[task.task_id] = best_agent
        free.discard(best_agent)
    return assignments
