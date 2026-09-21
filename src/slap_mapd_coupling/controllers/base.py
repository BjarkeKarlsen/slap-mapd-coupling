"""Controller protocol: shared task assignment plus a swappable routing policy pi_route."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task


@runtime_checkable
class Controller(Protocol):
    """pi = (pi_assign, pi_route) (sec:pf:controllers), realised by
    Table tab:information's three architectures.

    pi_assign (controllers/assignment.py's assign_tasks) is shared,
    identical across all three architectures by construction
    (sec:method:controllers), and deliberately not part of this
    Protocol: every concrete controller calls it directly rather than
    reimplementing it, so a controller swap can only ever change
    pi_route -- the one thing this Protocol actually requires.
    """

    def route(
        self,
        graph: WarehouseGraph,
        fleet: FleetState,
        tasks: Sequence[Task],
        t: int,
    ) -> dict[AgentId, Action]:
        """One intended action per agent (pre-conflict-resolution),
        using whichever state Table tab:information permits this
        architecture to see."""
        ...
