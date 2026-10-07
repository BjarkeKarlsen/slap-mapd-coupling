"""Name -> StorageRule registry ("fixed" | "demand" | "congestion").

The environment loop (environment/multi_agent_env.py) only ever calls the
active rule through get_storage_rule, so adding a new rule never touches
the loop itself (docs/storage_rule_integration.md).

Unlike controllers/registry.py, this registry stores the StorageRule
itself, not a zero-argument factory: a StorageRule is already a plain
callable of its four eq:storageupdate arguments (storage/base.py), so
there is no separate "construct, then call" step the way a Controller's
route() needs an instance first.
"""

from __future__ import annotations

from typing import Callable

from slap_mapd_coupling.storage.base import StorageRule

_REGISTRY: dict[str, StorageRule] = {}


def register_storage_rule(name: str) -> Callable[[StorageRule], StorageRule]:
    """Decorator: @register_storage_rule("fixed")."""

    def decorator(rule: StorageRule) -> StorageRule:
        if name in _REGISTRY:
            raise ValueError(f"Storage rule {name!r} is already registered.")
        _REGISTRY[name] = rule
        return rule

    return decorator


def get_storage_rule(name: str) -> StorageRule:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown storage rule {name!r}; registered: {registered_storage_rules()}. "
            "Storage rules register themselves on import (e.g. storage/fixed.py) "
            "-- make sure the module defining it has actually been imported."
        )
    return _REGISTRY[name]


def registered_storage_rules() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
