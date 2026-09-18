"""Unit tests for slap_mapd_coupling.viz.theme."""

from slap_mapd_coupling.core.graph import VertexRole
from slap_mapd_coupling.viz.theme import style_for_role


def test_style_for_role_priority_delivery_over_storage():
    role = VertexRole(movable=True, delivery=True, storage=True)
    style, secondary = style_for_role(role)
    assert style.shape == "square"  # delivery's shape
    assert secondary == ["storage"]


def test_style_for_role_plain_vertex_has_no_secondary_roles():
    role = VertexRole(movable=True)
    style, secondary = style_for_role(role)
    assert style.shape == "circle"  # plain's shape
    assert secondary == []
