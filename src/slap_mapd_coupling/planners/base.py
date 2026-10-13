"""Planner protocol: a non-learned multi-agent path planner outside the three controllers."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId


@runtime_checkable
class Planner(Protocol):
    """A planner proposes one action per agent for one timestep.

    planners/driver.py runs any Planner from outside the environment. Conflict
    resolution still runs on the proposed actions afterwards.
    """

    def plan(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        """One intended action per agent at time t, before conflict resolution."""
        ...


@runtime_checkable
class SelfAssigningPlanner(Planner, Protocol):
    """A stateful planner that also picks its own tasks.

    It is passed to the environment as its Assigner (controllers/assignment.py),
    which resets it and asks it for the assignment in place of the shared rule.
    """

    def reset(self, episode_seed: int) -> None:
        """Forget every committed path, for a new episode."""
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
