"""Shared colour/shape palette for both viz backends (matplotlib and pygame).

Plain data only (hex strings, not matplotlib/pygame objects), so a given
instance renders identically whichever backend draws it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from slap_mapd_coupling.core.graph import VertexRole


class RoleStyle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fill: str  # hex colour
    edge: str  # hex colour
    shape: str  # "circle" | "square" | "diamond" -- each backend maps this to its own API


PALETTE: dict[str, str] = {
    "edge": "#9aa5b1",
    "one_way_edge": "#2f6fae",
    "transit": "#5a6673",
    "agent": "#c0392b",
    "text": "#43505c",
}

ROLE_STYLES: dict[str, RoleStyle] = {
    "delivery": RoleStyle(fill="#ffd9b3", edge="#c9761a", shape="square"),
    "storage": RoleStyle(fill="#cfe3f7", edge="#2f6fae", shape="circle"),
    "endpoint": RoleStyle(fill="#d7f0d8", edge="#3f8f45", shape="diamond"),
    "plain": RoleStyle(fill="#ffffff", edge=PALETTE["transit"], shape="circle"),
}

# First match wins as the primary style. Roles are NOT mutually exclusive
# (sec:pf:env) -- this order is a rendering judgement call, not derived
# from the math: delivery and storage are where operational work
# happens, endpoint is a fallback/idle role.
ROLE_PRIORITY: tuple[str, ...] = ("delivery", "storage", "endpoint")


def style_for_role(role: VertexRole) -> tuple[RoleStyle, list[str]]:
    """Primary style (by ROLE_PRIORITY) plus any other active role names.

    A vertex with more than one active role (e.g. both storage and
    endpoint) gets the higher-priority role's marker as primary; the
    caller draws the remaining names as a secondary indicator (e.g. an
    outline ring) rather than losing the information.
    """
    active = [name for name in ROLE_PRIORITY if getattr(role, name)]
    primary = active[0] if active else "plain"
    return ROLE_STYLES[primary], active[1:]
