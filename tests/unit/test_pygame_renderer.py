"""Unit tests for slap_mapd_coupling.viz.pygame_renderer.

pygame is an optional extra ([viz] in pyproject.toml) -- skip this whole
module if it isn't installed, rather than failing the suite.
"""

import os

import pytest

pygame = pytest.importorskip("pygame")

# Standard headless-testing trick for SDL/pygame: no real display needed,
# works in CI. Must be set before pygame.init() is ever called.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from slap_mapd_coupling.core.agents import AgentState, FleetState  # noqa: E402
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph  # noqa: E402
from slap_mapd_coupling.viz.pygame_renderer import WarehouseRenderer  # noqa: E402


def test_pygame_renderer_smoke():
    vertices = {
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, delivery=True)),
    }
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0)}
    fleet = FleetState(agents={1: AgentState(agent_id=1, location=1)})

    renderer = WarehouseRenderer(graph, positions, window_size=(200, 200))
    try:
        renderer.render(fleet)
    finally:
        renderer.close()
