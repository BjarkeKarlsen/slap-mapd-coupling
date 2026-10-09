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

    def reset(self, episode_seed: int) -> None:
        """Primes this controller for a new episode drawn with
        episode_seed, the same seed the priority order and the order
        generator derive their own streams from (4.Implementation.tex:
        450-452, sec:impl:instances). A no-op for a controller with no
        randomness of its own (centralised.py, the future section_based.py);
        DecentralisedController derives its action-sampling stream from
        this seed rather than a constructor-time one, so bit-for-bit
        replay under a fixed seed holds regardless of how many episodes
        the same controller instance has already run."""
        ...
