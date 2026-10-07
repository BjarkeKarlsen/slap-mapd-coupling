"""Tasks tau_j = (r_j, s_j, g_j, k_j) and the lifecycle sets Q_t, B_t, C_t."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, StorageState

TaskId = int


class Task(BaseModel):
    """tau_j = (r_j, s_j, g_j, k_j), eq:task, plus y_j/p_j/d_j.

    Status is a method computed from the timestamps (eq:lifecycle), never
    a stored field, so it cannot drift out of sync with them.
    assignment_time/pickup_time/completion_time start unset and are each
    settable exactly once via .assign()/.pick_up()/.complete(): this is
    how "no re-tasking once assigned" (sec:pf:scope) is enforced at the
    type level, not just by controller discipline.

    pickup_time (p_j) is not part of the thesis's own eq:task notation --
    the model names q_i(t), "the vertex agent a_i is currently trying to
    reach, ... the pickup vertex s_j of its assigned task before pickup
    and the delivery vertex g_j after" (sec:pf:observations), but never
    specifies what marks "after." This field is that marker, added here
    because a routing controller genuinely needs it (a location-only
    check is unsound: once an agent leaves s_j, location != s_j again,
    which would send it back to pickup); see `current_goal` below for
    q_i(t) restricted to one task. Flagged for the thesis text too, not
    just the code -- see thesis-progress's note alongside this change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: TaskId
    release_time: NonNegativeInt  # r_j
    pickup_vertex: VertexId  # s_j, caller-checked to be in V_str
    delivery_vertex: VertexId  # g_j, caller-checked to be in V_del
    sku: SkuId  # k_j
    assigned_agent: AgentId | None = None  # sigma(j)
    assignment_time: NonNegativeInt | None = None  # y_j
    pickup_time: NonNegativeInt | None = None  # p_j (not thesis notation, see docstring)
    completion_time: NonNegativeInt | None = None  # d_j

    @model_validator(mode="after")
    def _timestamp_ordering(self) -> "Task":
        if self.assignment_time is not None and self.assignment_time < self.release_time:
            raise ValueError(
                f"y_j={self.assignment_time} < r_j={self.release_time}; requires r_j <= y_j."
            )
        if self.pickup_time is not None:
            if self.assignment_time is None:
                raise ValueError("p_j set without y_j: r_j <= y_j <= p_j requires y_j first.")
            if self.pickup_time < self.assignment_time:
                raise ValueError(
                    f"p_j={self.pickup_time} < y_j={self.assignment_time}; requires y_j <= p_j."
                )
        if self.completion_time is not None:
            if self.pickup_time is None:
                raise ValueError("d_j set without p_j: an agent cannot deliver before pickup.")
            if self.completion_time < self.pickup_time:
                raise ValueError(
                    f"d_j={self.completion_time} < p_j={self.pickup_time}; requires p_j <= d_j."
                )
        if (self.assigned_agent is None) != (self.assignment_time is None):
            raise ValueError(
                "assigned_agent and assignment_time must be set together (sigma(j) and y_j)."
            )
        return self

    def status(self, t: int) -> Literal["waiting", "active", "completed"]:
        """Membership in Q_t/B_t/C_t (eq:lifecycle).

        Raises for t < release_time: a task not yet released has no
        lifecycle status.
        """
        if t < self.release_time:
            raise ValueError(f"t={t} is before release_time={self.release_time}.")
        if self.assignment_time is None or t < self.assignment_time:
            return "waiting"  # r_j <= t < y_j
        if self.completion_time is None or t < self.completion_time:
            return "active"  # y_j <= t < d_j
        return "completed"  # d_j <= t

    def assign(self, agent_id: AgentId, t: int) -> "Task":
        """Type-level enforcement of "no re-tasking" (sec:pf:scope).

        Constructs a fresh Task rather than `self.model_copy(update=...)`:
        Pydantic v2's model_copy skips validators entirely, so it would
        let an out-of-order y_j through unchecked; re-constructing re-runs
        `_timestamp_ordering`.
        """
        if self.assignment_time is not None:
            raise ValueError(f"Task {self.task_id} already assigned at y_j={self.assignment_time}.")
        return Task.model_validate(
            {**self.model_dump(), "assigned_agent": agent_id, "assignment_time": t}
        )

    def pick_up(self, t: int) -> "Task":
        """Marks p_j: the agent has reached s_j. See the class docstring
        for why this timestamp exists even though eq:task doesn't name it."""
        if self.assignment_time is None:
            raise ValueError(f"Task {self.task_id} cannot be picked up before assignment.")
        if self.pickup_time is not None:
            raise ValueError(f"Task {self.task_id} already picked up at p_j={self.pickup_time}.")
        return Task.model_validate({**self.model_dump(), "pickup_time": t})

    def complete(self, t: int) -> "Task":
        if self.pickup_time is None:
            raise ValueError(f"Task {self.task_id} cannot complete before pickup.")
        if self.completion_time is not None:
            raise ValueError(
                f"Task {self.task_id} already completed at d_j={self.completion_time}."
            )
        return Task.model_validate({**self.model_dump(), "completion_time": t})

    def is_pickup_feasible(self, storage: StorageState) -> bool:
        """eq:coupling: x_{r_j}(k_j, s_j) > 0.

        Caller passes the StorageState snapshot as-of r_j; Task holds no
        storage-history reference itself.
        """
        return storage.units(self.sku, self.pickup_vertex) > 0

    @property
    def service_time(self) -> int | None:
        """zeta_j = d_j - r_j (eq:servicetime); None until completed."""
        return None if self.completion_time is None else self.completion_time - self.release_time

    @property
    def current_goal(self) -> VertexId:
        """q_i(t) restricted to this task: s_j before pickup, g_j after
        (sec:pf:observations). Only meaningful while the task is active
        (assigned, not completed) -- callers check status(t) themselves,
        this property doesn't re-derive it."""
        return self.pickup_vertex if self.pickup_time is None else self.delivery_vertex


def free_agents(fleet: FleetState, tasks: Iterable[Task], t: int) -> set[AgentId]:
    """eq:free: a_i is free at t iff no task active at t has sigma(j) == i."""
    occupied = {task.assigned_agent for task in tasks if task.status(t) == "active"}
    return set(fleet.agents.keys()) - occupied


def active_tasks_by_agent(tasks: Iterable[Task], t: int) -> dict[AgentId, Task]:
    """The active task (if any) each agent is currently serving at t --
    at most one per agent, since eq:lifecycle's B_t membership plus
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
