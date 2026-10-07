"""The reward R_i(t) (sec:method:rl).

R_i(t) = gamma*Phi_i(t+1) - Phi_i(t)
         + n_deliver * 1[a_i completes a task at t]
         - n_blocked * 1[a_i is overridden at t]
         - n_cng     * delta_i(t)

Phi_i(t) = -eta_i(l_i(t),t), built from the distance label
eta_i(v,t) = d_G(v, q_i(t)). The first term is
potential-based shaping (Ng, Harada & Russell 1999), which leaves the
game's equilibria unchanged, also for a potential that changes whenever
the current target switches (Devlin & Kudenko 2012).

The override indicator is ResolutionResult.overridden from the conflict
resolution operator (sec:method:resolution).

Free agent: the current target defines q_i(t) only for
an agent serving a task. Here a free agent's target is its own vertex, so
it gets no shaping signal while idle. This is an open thesis decision,
see issue #102.
"""

from __future__ import annotations

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


def current_target(location: VertexId, task: Task | None) -> VertexId:
    """q_i(t), the current target: task.current_goal if
    the agent is serving one, else its own vertex (see module docstring)."""
    return location if task is None else task.current_goal


def potential(graph: WarehouseGraph, location: VertexId, task: Task | None) -> float:
    """Phi_i(t) = -eta_i(l_i(t),t) = -d_G(l_i(t), q_i(t)), from the distance
    label."""
    return -graph.distance(location, current_target(location, task))


def reward(
    graph: WarehouseGraph,
    *,
    location_now: VertexId,
    task_now: Task | None,
    location_next: VertexId,
    task_next: Task | None,
    completed_task: bool,
    overridden: bool,
    congestion: float,
    discount: float,
    deliver_reward: float,
    override_penalty: float,
    congestion_reward_weight: float,
) -> float:
    """R_i(t), the reward.

    `task_now`/`task_next` are whichever task the agent is serving just
    before / just after this timestep's transition -- None for a free
    agent. `congestion` is delta_i(t), the congestion feature, computed
    upstream and only weighted here.
    """
    phi_now = potential(graph, location_now, task_now)
    phi_next = potential(graph, location_next, task_next)
    shaping = discount * phi_next - phi_now
    return (
        shaping
        + deliver_reward * completed_task
        - override_penalty * overridden
        - congestion_reward_weight * congestion
    )
