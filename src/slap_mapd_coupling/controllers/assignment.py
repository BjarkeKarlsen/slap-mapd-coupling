"""Shared greedy task assignment pi_assign (eq:assignmentrule, sec:method:controllers).

Identical across all three controller architectures: at every timestep,
free agents (eq:free) are matched to tasks in Q_t (released, not yet
assigned, eq:lifecycle) ordered by r_j ascending, each task greedily
matched to the free agent minimising d_G(l_i(t), s_j), removing that agent
from the free pool before the next task is considered. A greedy heuristic
for bipartite matching, not the Hungarian algorithm's optimum
(sec:method:controllers) -- swapping in an optimal matcher is a one-line
ablation, not a change to Pi, which is exactly why this is its own
function rather than inlined into any one controller.
"""

from __future__ import annotations

from typing import Iterable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId, free_agents


def assign_tasks(
    graph: WarehouseGraph,
    fleet: FleetState,
    tasks: Iterable[Task],
    t: int,
) -> dict[TaskId, AgentId]:
    """pi_assign: task_id -> agent_id for every task matched this timestep.

    Only tasks in Q_t (task.status(t) == "waiting") are candidates; tasks
    already assigned or completed are untouched (Task enforces "no
    re-tasking," core/tasks.py). The caller applies the returned mapping
    via Task.assign(agent_id, t) -- this function only decides who goes
    where, it does not mutate any Task.

    eq:assignmentrule's argmin is set-valued on ties, which would make
    replay non-deterministic (AGENTS.md's replay invariant); this isn't
    pinned down upstream, so ties are broken here, explicitly: by
    task_id ascending among tasks sharing r_j, and by agent_id ascending
    among free agents sharing d_G(l_i(t), s_j).
    """
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
                agent_id,
            ),
        )
        assignments[task.task_id] = best_agent
        free.discard(best_agent)
    return assignments
