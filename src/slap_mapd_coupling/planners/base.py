"""Planner protocol: a non-learned multi-agent path planner outside the three controllers."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId


@runtime_checkable
class Planner(Protocol):
    """A planner proposes one action per agent for one timestep.

    planners/router.py lets the environment loop run any Planner, so a new
    reference planner (ICBS, ICTS, ODrM*) is one new file here. Conflict
    resolution still runs on the proposed actions afterwards.

    A planner may be a pure function of its arguments (prioritised A*) or keep
    state across steps (Token Passing, a SelfAssigningPlanner below), which is
    why PlannerRouter forwards `reset` to it.
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

    Token Passing chooses a task for an agent in the same step that it plans
    the agent's path, since its guarantee depends on that task rule. The
    environment loop therefore asks such a planner for the assignment in place
    of the shared pi_assign (controllers/assignment.py), and calls `reset` so
    that the committed paths of one episode never reach the next.
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
