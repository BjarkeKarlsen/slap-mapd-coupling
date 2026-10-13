"""get_router: the one place that turns an ExperimentConfig into what routes the fleet."""

from __future__ import annotations

from slap_mapd_coupling.controllers.base import Controller
from slap_mapd_coupling.controllers.registry import get_controller
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.planners import get_planner, router_for


def get_router(config: ExperimentConfig) -> Controller:
    """The registered controller if config.controller is set, otherwise a
    PlannerRouter around config.planner (a reference planner, not a controller)."""
    if config.controller is not None:
        return get_controller(config.controller)
    assert config.planner is not None
    return router_for(get_planner(config.planner))
