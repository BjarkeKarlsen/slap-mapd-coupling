"""PlannerRouter: lets the environment loop run a Planner where it runs a controller."""

from __future__ import annotations

from typing import Sequence

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.planners.base import Planner


class PlannerRouter:
    """Exposes a Planner through the reset/route calls the environment loop makes.

    A reference planner is not one of the three controllers, so it has no
    entry in the controller registry. This adapter is how the loop drives it.
    """

    def __init__(self, planner: Planner) -> None:
        self._planner = planner

    def reset(self, episode_seed: int) -> None:
        """No-op, since planners are pure functions of (graph, fleet, tasks, t)."""

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        return self._planner.plan(graph, fleet, tasks, t)
