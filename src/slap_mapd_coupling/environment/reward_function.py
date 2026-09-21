"""Potential-based shaping reward R_i(t) (sec:method:rl, eq:reward).

R_i(t) = gamma*Phi_i(t+1) - Phi_i(t)
         + r_deliver * 1[a_i completes a task at t]
         - r_blk     * 1[a_i is overridden at t]
         - r_cng     * delta_i(t)

Phi_i(t) = -eta_i(l_i(t),t), eta_i(v,t) = d_G(v, q_i(t)) (eq:potential).
The first term is potential-based shaping (Ng, Harada & Russell 1999): it
does not alter the underlying stochastic game's equilibria, and the
guarantee extends to dynamic potentials (Devlin & Kudenko 2012) --
required here specifically, since q_i(t) and hence Phi_i change at every
reassignment (eq:lifecycle), making Phi_i non-stationary by construction.

The override indicator is exactly resolution/conflict_resolution.py's
ResolutionResult.overridden -- do not finalise r_blk (override_penalty)
before that operator is independently verified: the penalty is
meaningless until overrides are detected correctly (sec:method:rl's own
explicit caveat).

Free-agent q_i(t): the thesis leaves this genuinely open (sec:pf:tasks'
"How q_i(t) is set for a free agent... is a Method-level choice" is never
actually resolved -- thesis-progress/GAPS.tex's [G18]). Decided here, not
guessed at silently: a free agent's target is its own current vertex, so
eta_i(t) = 0 and it contributes no shaping signal while idle -- Phi_i only
becomes non-trivial once the agent is assigned a task. Ng et al.'s
guarantee holds for any potential function, so this choice doesn't affect
correctness, only what the idle-agent shaping signal looks like.
"""

from __future__ import annotations

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


def current_target(location: VertexId, task: Task | None) -> VertexId:
    """q_i(t): task.current_goal if the agent is serving one, else its own
    current vertex (the free-agent decision above)."""
    return location if task is None else task.current_goal


def potential(graph: WarehouseGraph, location: VertexId, task: Task | None) -> float:
    """Phi_i(t) = -eta_i(l_i(t),t) = -d_G(l_i(t), q_i(t)) (eq:potential)."""
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
    """R_i(t) (eq:reward).

    `task_now`/`task_next` are whichever task the agent is serving just
    before / just after this timestep's transition -- None for a free
    agent. `congestion` is delta_i(t) (eq:congestion), computed upstream;
    this function only weighs it, it doesn't recompute it.
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
