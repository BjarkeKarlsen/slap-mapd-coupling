"""Name -> Planner lookup for the non-learned reference planners."""

from __future__ import annotations

from typing import Callable, Literal

from slap_mapd_coupling.planners.base import Planner
from slap_mapd_coupling.planners.prioritised_astar import PrioritisedAStarPlanner
from slap_mapd_coupling.planners.token_passing import TokenPassingPlanner

PlannerName = Literal["prioritised_astar", "token_passing"]

_PLANNERS: dict[str, Callable[[], Planner]] = {
    "prioritised_astar": PrioritisedAStarPlanner,
    "token_passing": TokenPassingPlanner,
}


def get_planner(name: str) -> Planner:
    if name not in _PLANNERS:
        raise KeyError(f"Unknown planner {name!r}; available: {sorted(_PLANNERS)}.")
    return _PLANNERS[name]()
