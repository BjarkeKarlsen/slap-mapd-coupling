"""Name -> Controller registry ("centralised" | "section" | "decentralised").

The environment loop (environment/multi_agent_env.py) only ever calls the
active controller through get_controller, so adding a new controller
architecture never touches the loop itself (docs/controller_integration.md).
"""

from __future__ import annotations

from typing import Callable, Literal

from slap_mapd_coupling.controllers.base import Controller

ControllerName = Literal["centralised", "section", "decentralised"]

_REGISTRY: dict[str, Callable[[], Controller]] = {}


def register_controller(
    name: str,
) -> Callable[[Callable[[], Controller]], Callable[[], Controller]]:
    """Class/factory decorator: @register_controller("centralised")."""

    def decorator(factory: Callable[[], Controller]) -> Callable[[], Controller]:
        if name in _REGISTRY:
            raise ValueError(f"Controller {name!r} is already registered.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def get_controller(name: str) -> Controller:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown controller {name!r}; registered: {registered_controllers()}. "
            "Controllers register themselves on import (e.g. controllers/centralised.py) "
            "-- make sure the module defining it has actually been imported."
        )
    return _REGISTRY[name]()


def registered_controllers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
