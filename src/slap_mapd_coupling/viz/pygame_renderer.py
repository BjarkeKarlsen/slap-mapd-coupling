"""Interactive warehouse renderer (pygame). Optional -- see pyproject.toml's [viz] extra."""

from __future__ import annotations

from slap_mapd_coupling.core.agents import FleetState
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import StorageState
from slap_mapd_coupling.viz import theme

_MARGIN = 60


class WarehouseRenderer:
    """Build once, then call `.render(fleet)` every step.

    Precomputes pixel positions from the same `positions` dict the
    matplotlib backend uses, so both backends agree visually on a given
    instance. The static graph (edges + vertex markers) is pre-rendered
    onto one cached background Surface once, in __init__, since it never
    changes between steps -- redrawing the whole graph every frame (as a
    naive per-step render loop would) is wasted work once this runs at
    real environment-step rates.
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        positions: dict[VertexId, tuple[float, float]],
        window_size: tuple[int, int] = (900, 700),
    ) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode(window_size)
        self._pixel_positions = self._to_pixels(positions, window_size)
        self._background = self._render_background(graph, window_size)

    def _to_pixels(
        self, positions: dict[VertexId, tuple[float, float]], window_size: tuple[int, int]
    ) -> dict[VertexId, tuple[int, int]]:
        xs = [x for x, _ in positions.values()]
        ys = [y for _, y in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width, height = window_size
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        sx = (width - 2 * _MARGIN) / x_span
        sy = (height - 2 * _MARGIN) / y_span
        scale = min(sx, sy)

        pixels: dict[VertexId, tuple[int, int]] = {}
        for vertex_id, (x, y) in positions.items():
            px = int(_MARGIN + (x - x_min) * scale)
            # Flip Y: positions use "up is positive" (matplotlib
            # convention); pygame's origin is top-left with y growing down.
            py = int(_MARGIN + (y_max - y) * scale)
            pixels[vertex_id] = (px, py)
        return pixels

    def _hex_to_rgb(self, hex_colour: str) -> tuple[int, int, int]:
        h = hex_colour.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    def _render_background(self, graph: WarehouseGraph, window_size: tuple[int, int]):
        pygame = self._pygame
        surface = pygame.Surface(window_size)
        surface.fill((255, 255, 255))

        drawn: set[tuple[VertexId, VertexId]] = set()
        for edge in graph.edges:
            pair = (edge.source, edge.target)
            if pair in drawn or (pair[1], pair[0]) in drawn:
                continue
            p1 = self._pixel_positions[edge.source]
            p2 = self._pixel_positions[edge.target]
            reverse_exists = any(
                e.source == edge.target and e.target == edge.source for e in graph.edges
            )
            colour = self._hex_to_rgb(theme.PALETTE["edge" if reverse_exists else "one_way_edge"])
            pygame.draw.line(surface, colour, p1, p2, width=2 if reverse_exists else 3)
            drawn.add(pair)

        for vertex_id, vertex in graph.vertices.items():
            style, _secondary = theme.style_for_role(vertex.role)
            pygame.draw.circle(
                surface, self._hex_to_rgb(style.fill), self._pixel_positions[vertex_id], 10
            )
            pygame.draw.circle(
                surface, self._hex_to_rgb(style.edge), self._pixel_positions[vertex_id], 10, width=2
            )

        return surface

    def render(self, fleet: FleetState, *, storage: StorageState | None = None) -> bool:
        """Draw one frame: cached background + current agent positions.

        Returns False if the window's close button was clicked (a QUIT
        event was seen), True otherwise -- so a driving loop can exit
        cleanly instead of continuing to render (or looking frozen) after
        the user has closed the window.
        """
        pygame = self._pygame
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        self._screen.blit(self._background, (0, 0))

        agent_colour = self._hex_to_rgb(theme.PALETTE["agent"])
        for agent_id, vertex_id in fleet.locations().items():
            pos = self._pixel_positions[vertex_id]
            pygame.draw.circle(self._screen, agent_colour, pos, 8)

        pygame.display.flip()
        return running

    def close(self) -> None:
        self._pygame.quit()
