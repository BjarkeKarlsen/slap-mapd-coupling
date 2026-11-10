"""The reward R_i(t) (sec:method:rl).

R_i(t) = gamma*Phi_i(t+1) - Phi_i(t)
         + n_deliver * 1[a_i completes a task at t]
         - n_blocked * 1[a_i is overridden at t]
         - c_move    * 1[a_i moves at t]
         - c_wait    * omega_i(t)
         - n_cng     * delta_i(t)

Phi_i(t) = -eta_i(l_i(t),t), built from the distance label
eta_i(v,t) = d_G(v, q_i(t)). The first term is
potential-based shaping (Ng, Harada & Russell 1999), which leaves the
game's equilibria unchanged, also for a potential that changes whenever
the current target switches (Devlin & Kudenko 2012).

The override indicator is ResolutionResult.overridden from the conflict
resolution operator (sec:method:resolution).

omega_i(t) is the blocked indicator of the service-time split: 1 when the agent
is serving a task and stands still at t, by a chosen wait or by an override, and
0 for an agent that serves no task, so a parked free agent pays nothing. An
override therefore costs c_wait + n_blocked and a chosen wait c_wait alone, so
yielding is never dearer than being overridden.

Every move pays c_move, whether the agent serves a task or not. A move is a change of
vertex between t and t+1, so a move that conflict resolution turned into a wait is not one.

Free agent: the current target defines q_i(t) only for
an agent serving a task. Here a free agent's target is its own vertex, or its
parking leaf when it was sent to one (D26), so it gets no shaping signal while
idle and in nobody's way.

[A79]: Phi_i(t+1) is the potential the robot sees at t+1, read after stage 1 and 2
of t+1 (new tasks) and the parking update. So the task and the parking leaf of the
two potentials are passed apart, and a new target is charged once, on the step it
is handed out. The environment computes the reward after both
(environment/multi_agent_env.py::step).
"""

from __future__ import annotations

from dataclasses import dataclass

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


def current_target(
    location: VertexId, task: Task | None, parking: VertexId | None = None
) -> VertexId:
    """q_i(t), the current target: task.current_goal if
    the agent is serving one, else its parking leaf when it has one (D26), else its own
    vertex (see module docstring)."""
    if task is not None:
        return task.current_goal
    return location if parking is None else parking


def potential(
    graph: WarehouseGraph, location: VertexId, task: Task | None, parking: VertexId | None = None
) -> float:
    """Phi_i(t) = -eta_i(l_i(t),t) = -d_G(l_i(t), q_i(t)), from the distance
    label."""
    return -graph.distance(location, current_target(location, task, parking))


@dataclass(frozen=True)
class RewardTerms:
    """The six terms of the reward R_i(t), each already signed, so that
    their sum is the reward. Kept apart so a training log can say what an
    episode return is made of."""

    shaping: float  # gamma*Phi_i(t+1) - Phi_i(t)
    delivery: float  # + n_deliver * 1[a_i completes a task at t], never negative
    override: float  # - n_blocked * 1[a_i is overridden at t], never positive
    move: float  # - c_move * 1[a_i moves at t], never positive
    wait: float  # - c_wait * omega_i(t), never positive
    congestion: float  # - n_cng * delta_i(t)

    @property
    def total(self) -> float:
        """R_i(t). Added left to right in the order of the reward, so the value
        is the same, to the last bit, as summing the six terms inline."""
        return (
            self.shaping + self.delivery + self.override + self.move + self.wait + self.congestion
        )


def reward_terms(
    graph: WarehouseGraph,
    *,
    location_now: VertexId,
    task_now: Task | None,
    location_next: VertexId,
    task_next: Task | None,
    completed_task: bool,
    overridden: bool,
    moved: bool,
    stood_still: bool,
    congestion: float,
    discount: float,
    deliver_reward: float,
    override_penalty: float,
    move_cost: float,
    wait_cost: float,
    congestion_reward_weight: float,
    parking_now: VertexId | None = None,
    parking_next: VertexId | None = None,
) -> RewardTerms:
    """The six terms of R_i(t), the reward, separately. See `reward` for the
    arguments. `parking_now` and `parking_next` are the idle robot's parking leaf (D26) at t
    and at t+1, read after the parking update of t+1 ([A79]). Each is the target of its
    potential when the robot has no task at that time."""
    phi_now = potential(graph, location_now, task_now, parking_now)
    phi_next = potential(graph, location_next, task_next, parking_next)
    return RewardTerms(
        shaping=discount * phi_next - phi_now,
        delivery=deliver_reward * completed_task,
        override=-override_penalty * overridden,
        move=-move_cost * moved,
        wait=-wait_cost * stood_still,
        congestion=-congestion_reward_weight * congestion,
    )


def reward(
    graph: WarehouseGraph,
    *,
    location_now: VertexId,
    task_now: Task | None,
    location_next: VertexId,
    task_next: Task | None,
    completed_task: bool,
    overridden: bool,
    moved: bool,
    stood_still: bool,
    congestion: float,
    discount: float,
    deliver_reward: float,
    override_penalty: float,
    move_cost: float,
    wait_cost: float,
    congestion_reward_weight: float,
) -> float:
    """R_i(t), the reward: the sum of the terms of `reward_terms`.

    `task_now`/`task_next` are whichever task the agent is serving at t and
    at t+1 -- None for a free agent. `task_next` is read after the assignment
    of t+1, so a task handed out then is already the agent's ([A79]).
    `moved` is 1[a_i moves at t] and `move_cost` is c_move. `stood_still` is
    omega_i(t) and `wait_cost` is c_wait. `congestion` is
    delta_i(t), the congestion feature, computed upstream and only weighted here.
    """
    return reward_terms(
        graph,
        location_now=location_now,
        task_now=task_now,
        location_next=location_next,
        task_next=task_next,
        completed_task=completed_task,
        overridden=overridden,
        moved=moved,
        stood_still=stood_still,
        congestion=congestion,
        discount=discount,
        deliver_reward=deliver_reward,
        override_penalty=override_penalty,
        move_cost=move_cost,
        wait_cost=wait_cost,
        congestion_reward_weight=congestion_reward_weight,
    ).total
