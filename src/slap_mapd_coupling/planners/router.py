"""PlannerRouter: lets the environment loop run a Planner where it runs a controller."""

from __future__ import annotations

from typing import Sequence

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, TaskId
from slap_mapd_coupling.planners.base import Planner, SelfAssigningPlanner


class PlannerRouter:
    """Exposes a Planner through the reset/route calls the environment loop makes.

    A reference planner is not one of the three controllers, so it has no
    entry in the controller registry. This adapter is how the loop drives it.
    """

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    def reset(self, episode_seed: int) -> None:
        """Forwards to the planner if it keeps state. A planner that is a pure function
        of (graph, fleet, tasks, t), like prioritised A*, has nothing to reset."""
        reset = getattr(self._planner, "reset", None)
        if reset is not None:
            reset(episode_seed)

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        return self._planner.plan(graph, fleet, tasks, t)


class SelfAssigningPlannerRouter(PlannerRouter):
    """A PlannerRouter whose planner also picks the tasks (SelfAssigningRouter)."""

    def __init__(self, planner: SelfAssigningPlanner) -> None:
        super().__init__(planner)
        self._self_assigning = planner

    def assign(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[TaskId, AgentId]:
        return self._self_assigning.assign(graph, fleet, tasks, t)


def router_for(planner: Planner) -> PlannerRouter:
    """The router that lets the environment loop run `planner`."""
    if isinstance(planner, SelfAssigningPlanner):
        return SelfAssigningPlannerRouter(planner)
    return PlannerRouter(planner)
