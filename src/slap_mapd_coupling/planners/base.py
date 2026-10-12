"""Planner protocol: a non-learned multi-agent path planner outside the three controllers."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


@runtime_checkable
class Planner(Protocol):
    """A planner proposes one action per agent for one timestep.

    planners/router.py lets the environment loop run any Planner, so a new
    reference planner (ICBS, ICTS, ODrM*) is one new file here. Conflict
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
