"""Unit tests for slap_mapd_coupling.controllers.base and .registry."""

import pytest

from slap_mapd_coupling.controllers.base import Controller
from slap_mapd_coupling.controllers.registry import (
    get_controller,
    register_controller,
    registered_controllers,
)


class _DummyController:
    def route(self, graph, fleet, tasks, t):
        return {}


class _NotAController:
    pass


def test_dummy_controller_satisfies_the_protocol():
    assert isinstance(_DummyController(), Controller)


def test_non_controller_does_not_satisfy_the_protocol():
    assert not isinstance(_NotAController(), Controller)


def test_register_and_get_roundtrip():
    register_controller("test-dummy")(_DummyController)
    assert "test-dummy" in registered_controllers()
    controller = get_controller("test-dummy")
    assert isinstance(controller, _DummyController)
    assert controller.route(None, None, [], 0) == {}


def test_get_unknown_controller_raises_key_error():
    with pytest.raises(KeyError):
        get_controller("does-not-exist")


def test_double_registration_raises():
    register_controller("test-dummy-double")(_DummyController)
    with pytest.raises(ValueError):
        register_controller("test-dummy-double")(_DummyController)
