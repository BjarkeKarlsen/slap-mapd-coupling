"""Tasks tau_j = (t_j^released, s_j, g_j, k_j) and the lifecycle sets Q_t, B_t, C_t."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, StorageState

TaskId = int


class Task(BaseModel):
    """tau_j = (t_j^released, s_j, g_j, k_j), the task definition, plus
    t_j^assigned/t_j^pickup/t_j^finish.

    Status is a method computed from the timestamps (the task stages),
    never a stored field, so it cannot drift out of sync with them.
    assignment_time/pickup_time/finish_time start unset and are each
    settable exactly once via .assign()/.pick_up()/.complete(): this is
    how "no re-tasking once assigned" (sec:pf:scope) is enforced at the
    type level, not just by controller discipline.

    t_j^pickup is in the thesis notation table (tab:notation): the first
    timestep the agent assigned to tau_j reaches its pickup vertex s_j.
    It marks when the agent's target changes from s_j to g_j (see
    `current_goal` below for q_i(t) restricted to one task), but doesn't
    change the task's own status, which stays active from assignment
    until delivery.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: TaskId
    release_time: NonNegativeInt  # t_j^released
    pickup_vertex: VertexId  # s_j, caller-checked to be in V_str
    delivery_vertex: VertexId  # g_j, caller-checked to be in V_del
    sku: SkuId  # k_j
    assigned_agent: AgentId | None = None  # sigma(j)
    assignment_time: NonNegativeInt | None = None  # t_j^assigned
    pickup_time: NonNegativeInt | None = None  # t_j^pickup
    finish_time: NonNegativeInt | None = None  # t_j^finish

    @model_validator(mode="after")
    def _timestamp_ordering(self) -> "Task":
        if self.assignment_time is not None and self.assignment_time < self.release_time:
            raise ValueError(
                f"t_j^assigned={self.assignment_time} < t_j^released={self.release_time}; "
                "requires t_j^released <= t_j^assigned."
            )
        if self.pickup_time is not None:
            if self.assignment_time is None:
                raise ValueError(
                    "t_j^pickup set without t_j^assigned: t_j^released <= t_j^assigned <= "
                    "t_j^pickup requires t_j^assigned first."
                )
            if self.pickup_time < self.assignment_time:
                raise ValueError(
                    f"t_j^pickup={self.pickup_time} < t_j^assigned={self.assignment_time}; "
                    "requires t_j^assigned <= t_j^pickup."
                )
        if self.finish_time is not None:
            if self.pickup_time is None:
                raise ValueError(
                    "t_j^finish set without t_j^pickup: an agent cannot deliver before pickup."
                )
            if self.finish_time < self.pickup_time:
                raise ValueError(
                    f"t_j^finish={self.finish_time} < t_j^pickup={self.pickup_time}; "
                    "requires t_j^pickup <= t_j^finish."
                )
        if (self.assigned_agent is None) != (self.assignment_time is None):
            raise ValueError(
                "assigned_agent and assignment_time must be set together "
                "(sigma(j) and t_j^assigned)."
            )
        return self

    def status(self, t: int) -> Literal["waiting", "active", "completed"]:
        """Membership in Q_t/B_t/C_t (the task stages).

        Raises for t < release_time: a task not yet released has no
        lifecycle status.
        """
        if t < self.release_time:
            raise ValueError(f"t={t} is before release_time={self.release_time}.")
        if self.assignment_time is None or t < self.assignment_time:
            return "waiting"  # t_j^released <= t < t_j^assigned
        if self.finish_time is None or t < self.finish_time:
            return "active"  # t_j^assigned <= t < t_j^finish
        return "completed"  # t_j^finish <= t

    def assign(self, agent_id: AgentId, t: int) -> "Task":
        """Type-level enforcement of "no re-tasking" (sec:pf:scope).

        Constructs a fresh Task rather than `self.model_copy(update=...)`:
        Pydantic v2's model_copy skips validators entirely, so it would
        let an out-of-order t_j^assigned through unchecked; re-constructing
        re-runs `_timestamp_ordering`.
        """
        if self.assignment_time is not None:
            raise ValueError(
                f"Task {self.task_id} already assigned at t_j^assigned={self.assignment_time}."
            )
        return Task.model_validate(
            {**self.model_dump(), "assigned_agent": agent_id, "assignment_time": t}
        )

    def pick_up(self, t: int) -> "Task":
        """Marks t_j^pickup: the agent has reached s_j. See the class
        docstring."""
        if self.assignment_time is None:
            raise ValueError(f"Task {self.task_id} cannot be picked up before assignment.")
        if self.pickup_time is not None:
            raise ValueError(
                f"Task {self.task_id} already picked up at t_j^pickup={self.pickup_time}."
            )
        return Task.model_validate({**self.model_dump(), "pickup_time": t})

    def complete(self, t: int) -> "Task":
        if self.pickup_time is None:
            raise ValueError(f"Task {self.task_id} cannot complete before pickup.")
        if self.finish_time is not None:
            raise ValueError(
                f"Task {self.task_id} already completed at t_j^finish={self.finish_time}."
            )
        return Task.model_validate({**self.model_dump(), "finish_time": t})

    def is_pickup_feasible(self, storage: StorageState) -> bool:
        """The stock condition: x_{t_j^released}(k_j, s_j) > 0.

        Caller passes the StorageState snapshot as-of t_j^released, Task
        holds no storage-history reference itself.
        """
        return storage.units(self.sku, self.pickup_vertex) > 0

    @property
    def service_time(self) -> int | None:
        """zeta_j = t_j^finish - t_j^released (the service time); None
        until completed."""
        return None if self.finish_time is None else self.finish_time - self.release_time

    @property
    def current_goal(self) -> VertexId:
        """q_i(t) restricted to this task: s_j before pickup, g_j after
        (sec:pf:observations). Only meaningful while the task is active
        (assigned, not completed) -- callers check status(t) themselves,
        this property doesn't re-derive it."""
        return self.pickup_vertex if self.pickup_time is None else self.delivery_vertex


def free_agents(fleet: FleetState, tasks: Iterable[Task], t: int) -> set[AgentId]:
    """The free-agent condition: a_i is free at t iff no task active at t
    has sigma(j) == i."""
    occupied = {task.assigned_agent for task in tasks if task.status(t) == "active"}
    return set(fleet.agents.keys()) - occupied


def active_tasks_by_agent(tasks: Iterable[Task], t: int) -> dict[AgentId, Task]:
    """The active task (if any) each agent is currently serving at t --
    at most one per agent, since the task stages' B_t membership plus
    "no re-tasking" (sec:pf:scope) together guarantee an agent serves at
    most one active task at once. Shared by controllers/centralised.py,
    environment/multi_agent_env.py and environment/observation.py rather
    than each recomputing it.
    """
    active: dict[AgentId, Task] = {}
    for task in tasks:
        if task.status(t) == "active":
            assert task.assigned_agent is not None  # guaranteed by Task's own validator
            active[task.assigned_agent] = task
    return active
