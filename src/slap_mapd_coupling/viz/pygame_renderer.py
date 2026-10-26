"""Interactive warehouse renderer (pygame). Optional -- see pyproject.toml's [viz] extra.

Two ways to use it. Live, build a `WarehouseRenderer` once and call
`.render_env(env)` after every step. Afterwards, replay a trace file
(evaluation/trace.py):

    python -m slap_mapd_coupling.viz.pygame_renderer trace.jsonl

A frame is drawn from the same lines the trace writes (a step line and a
stock line), so a live run and the replay of its trace look the same.

The window shows one thing at a time. Three tabs above the map choose what
the map shows:

- Agents. Every agent with its number, a dashed line to where it is
  walking, its last steps and the SKU it carries.
- Stock. The units on every slot, split by SKU popularity rank and filled
  against the slot capacity, the tasks waiting on each slot, and the stock
  waiting at the docks by SKU name.
- Congestion. A red patch on every vertex where agents holding a task stood
  still over the last 40 steps, how long each still agent has waited, and a
  red line from a blocked agent to the agent standing where it wanted to go.

The tabs of the sidebar choose one panel: the episode counts, the stock,
the docks, the congestion, the agents, the last tasks done, or the details
of what is selected. Click an agent or a vertex on the map, or a row of a
panel, to select it. Escape clears it.

The bar at the bottom pauses, steps back and forward and continues, a
slider jumps to any step, and a second slider sets the speed in steps a
second. A live run keeps every step it has drawn, so it can be paused,
stepped back through and continued from where it stopped, and the renderer
paces it at the chosen speed. Keys: space pauses or continues, left and
right step (with shift, 10 steps), home and end jump, up and down halve or
double the speed, 1 to 3 choose the map view.
"""

from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from slap_mapd_coupling.core.agents import FleetState
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import StorageState
from slap_mapd_coupling.viz import theme

if TYPE_CHECKING:
    from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv

_MARGIN = 30
_PANEL = 400  # width of the sidebar
_TABS = 40  # height of the tab strip above the map
_BAR = 52  # height of the control bar at the bottom
_TRAIL = 12  # steps of past positions drawn behind an agent
_WINDOW = 40  # steps of history the congestion view counts over
# Steps between two stock snapshots a live run keeps for stepping back. The current step
# always shows the stock of that step.
_LIVE_STOCK_EVERY = 10
# The speed slider runs from the slowest to the fastest speed, in steps a second, on a log
# scale.
_SLOWEST, _FASTEST = 0.5, 240.0

_WHITE = (255, 255, 255)
_PAPER = (246, 247, 249)
_RULE = (218, 223, 229)
_RED = (200, 30, 30)
_SELECT = (20, 120, 220)
_MUTED = (165, 172, 180)

# Agent colour by what it is doing. A put-away gets its own colour, so the share of trips
# that carry stock in is visible at a glance.
_PHASE_COLOURS = {
    None: "#8a949e",
    "to_pickup": "#e08a1e",
    "to_delivery": "#c0392b",
    "putaway": "#7a4fa3",
}
# Popularity rank, most popular first: dark red through orange to pale blue.
_RANK_STOPS = ("#b2182b", "#f4a582", "#92c5de", "#d1e5f0")
_ROLE_LABELS = {
    "storage": "slot",
    "delivery": "workstation",
    "receiving": "dock",
    "endpoint": "parking",
}
# What the map can show, one at a time, in tab order. The keys 1 to 3 follow it.
VIEWS = {"agents": "Agents", "stock": "Stock", "congestion": "Congestion"}
# The panels of the sidebar, one at a time, in tab order.
PANELS = {
    "episode": "Episode",
    "stock": "Stock",
    "docks": "Docks",
    "congestion": "Congestion",
    "agents": "Agents",
    "done": "Done",
    "selected": "Selected",
}


# The panel title of each sidebar tab.
_PANEL_TITLES = {
    "episode": "Episode",
    "stock": "Stock",
    "docks": "Docks",
    "congestion": "Congestion",
    "agents": "Agents",
    "done": "Last tasks done",
    "selected": "Selected",
}
# The x of each column of the agent table, from the card's left edge.
_COLUMNS = (0, 52, 104, 250, 310)


def _stat(label: str, value: Any, colour=None, action=None) -> tuple:
    """A panel row with a label on the left and its value on the right."""
    return ("stat", label, value, colour, action)


def _hex(colour: str) -> tuple[int, int, int]:
    h = colour.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rank_colour(rank: int, count: int) -> tuple[int, int, int]:
    """The colour of popularity rank `rank` of `count` SKUs, rank 0 the most popular."""
    share = 0.0 if count <= 1 else rank / (count - 1)
    stops = [_hex(c) for c in _RANK_STOPS]
    at = share * (len(stops) - 1)
    i = min(int(at), len(stops) - 2)
    w = at - i
    return tuple(round(a + (b - a) * w) for a, b in zip(stops[i], stops[i + 1]))  # type: ignore[return-value]


def _agent_colour(agent: Mapping[str, Any]) -> tuple[int, int, int]:
    key = "putaway" if agent.get("kind") == "putaway" else agent.get("phase")
    return _hex(_PHASE_COLOURS[key])


def _doing(agent: Mapping[str, Any]) -> str:
    if agent.get("kind") == "putaway":
        return "to slot" if agent.get("phase") == "to_delivery" else "to dock"
    return {"to_pickup": "to pickup", "to_delivery": "to workstation"}.get(
        agent.get("phase"), "idle"
    )


def header_from_graph(
    graph: WarehouseGraph, positions: Mapping[VertexId, tuple[float, float]]
) -> dict[str, Any]:
    """The part of a trace header the renderer reads, built from a graph."""
    return {
        "vertices": [
            {
                "id": v,
                "storage": vertex.role.storage,
                "delivery": vertex.role.delivery,
                "endpoint": vertex.role.endpoint,
                "receiving": vertex.role.receiving,
            }
            for v, vertex in sorted(graph.vertices.items())
        ],
        "edges": [[e.source, e.target, e.cost] for e in graph.edges],
        "positions": {str(v): [float(x), float(y)] for v, (x, y) in positions.items()},
        "capacities": {},
        "skus": {},
        "config": {},
    }


class Timeline:
    """The steps of an episode, and the stock lines between them, for stepping through."""

    def __init__(self) -> None:
        self.steps: list[Mapping[str, Any]] = []
        self.stock: list[Mapping[str, Any]] = []
        self._stock_t: list[int] = []

    def add_stock(self, line: Mapping[str, Any]) -> None:
        self.stock.append(line)
        self._stock_t.append(line["t"])

    def stock_at(self, t: int) -> Mapping[str, Any] | None:
        """The last stock line at or before step t, or None when there is none."""
        i = bisect.bisect_right(self._stock_t, t) - 1
        return self.stock[i] if i >= 0 else None

    def history(self, index: int) -> list[Mapping[str, Any]]:
        """The step lines of the last `_WINDOW` steps up to step `index`."""
        return list(self.steps[max(0, index - _WINDOW + 1) : index + 1])


class TraceReplay(Timeline):
    """A trace file, read once."""

    def __init__(self, path: Path) -> None:
        from slap_mapd_coupling.evaluation.trace import load_stock, load_trace

        super().__init__()
        self.header, steps = load_trace(path)
        self.steps = steps
        for line in load_stock(path):
            self.add_stock(line)


class WarehouseRenderer:
    """Build once, then call `.render_env(env)` (or `.render(fleet)`) every step.

    Pixel positions come from the same `positions` dict the matplotlib
    backend uses, so both backends agree on a given instance. The static
    graph is drawn once onto a cached background surface. The tabs and the
    buttons are drawn every frame, and a click is matched against what the
    last frame drew.
    """

    def __init__(
        self,
        graph: WarehouseGraph | None,
        positions: Mapping[VertexId, tuple[float, float]] | None = None,
        window_size: tuple[int, int] = (1440, 900),
        *,
        header: Mapping[str, Any] | None = None,
    ) -> None:
        import pygame

        if header is None:
            if graph is None or positions is None:
                raise ValueError("give a graph and its positions, or a trace header.")
            header = header_from_graph(graph, positions)
        if not header.get("positions"):
            raise ValueError("the trace header has no positions, so it cannot be drawn.")
        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("slap-mapd-coupling")
        self._screen = pygame.display.set_mode(window_size)
        self._size = window_size
        self._header = header
        self._font = pygame.font.Font(None, 20)
        self._small = pygame.font.Font(None, 16)
        self._title = pygame.font.Font(None, 22)
        self._title.set_bold(True)
        self._title_large = pygame.font.Font(None, 28)
        self._title_large.set_bold(True)
        self._heading = pygame.font.Font(None, 15)
        self._heading.set_bold(True)
        self._value = pygame.font.Font(None, 20)
        self._value.set_bold(True)
        self._label = pygame.font.Font(None, 18)
        self._label.set_bold(True)
        self._roles = {v["id"]: v for v in header["vertices"]}
        self._edges = {(int(s), int(t)) for s, t, _ in header["edges"]}
        positions = {int(v): (x, y) for v, (x, y) in header["positions"].items()}
        self._pixels, self._unit = self._to_pixels(positions)
        self._radius = max(3, int(self._unit * 0.22))
        self._background = self._render_background()
        self.view = "agents"
        self.panel = "episode"
        self.selected: tuple[str, Any] | None = None  # ("agent", "3") or ("vertex", 41)
        self.paused = False
        self.speed = 8.0  # steps a second, live and in a replay
        self._clock = pygame.time.Clock()
        self._drag: Callable[[], Any] | None = None
        self._live = Timeline()
        self._hits: list[tuple[Any, Callable[[], Any]]] = []
        self._frame: Mapping[str, Any] | None = None

    # ------------------------------------------------------------ geometry
    def _map_rect(self):
        return self._pygame.Rect(0, _TABS, self._size[0] - _PANEL, self._size[1] - _TABS - _BAR)

    def _to_pixels(
        self, positions: Mapping[VertexId, tuple[float, float]]
    ) -> tuple[dict[VertexId, tuple[int, int]], float]:
        xs = [x for x, _ in positions.values()]
        ys = [y for _, y in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = self._size[0] - _PANEL
        height = self._size[1] - _TABS - _BAR
        scale = min(
            (width - 2 * _MARGIN) / max(x_max - x_min, 1e-6),
            (height - 2 * _MARGIN) / max(y_max - y_min, 1e-6),
        )
        pixels: dict[VertexId, tuple[int, int]] = {}
        for vertex_id, (x, y) in positions.items():
            # Flip y: positions grow upwards, pygame's y grows downwards.
            pixels[vertex_id] = (
                int(_MARGIN + (x - x_min) * scale),
                int(_TABS + _MARGIN + (y_max - y) * scale),
            )
        # One pixel unit is the shortest edge drawn, so markers scale with the warehouse.
        lengths = [
            math.dist(pixels[s], pixels[t]) for s, t in self._edges if pixels[s] != pixels[t]
        ]
        return pixels, max(min(lengths, default=20.0), 6.0)

    def _role(self, v: VertexId) -> str | None:
        entry = self._roles.get(v, {})
        for name in theme.ROLE_PRIORITY:
            if entry.get(name):
                return name
        return None

    # ---------------------------------------------------------- background
    def _render_background(self):
        pygame = self._pygame
        surface = pygame.Surface(self._size)
        surface.fill(_WHITE)
        edge, one_way = _hex(theme.PALETTE["edge"]), _hex(theme.PALETTE["one_way_edge"])
        for s, t in self._edges:
            p1, p2 = self._pixels[s], self._pixels[t]
            if (t, s) in self._edges:
                if s < t:
                    pygame.draw.line(surface, edge, p1, p2, 2)
                continue
            pygame.draw.line(surface, one_way, p1, p2, 2)
            self._arrowhead(surface, p1, p2, one_way)
        r = self._radius
        for v, p in self._pixels.items():
            role = self._role(v)
            if role is None:
                pygame.draw.circle(surface, _WHITE, p, max(2, r // 2))
                pygame.draw.circle(surface, _hex(theme.PALETTE["transit"]), p, max(2, r // 2), 1)
            elif role != "storage":  # a slot is drawn every frame, with its stock
                self._marker(surface, role, p, r)
        return surface

    def _arrowhead(self, surface, p1, p2, colour) -> None:
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        tip = (
            p2[0] - math.cos(angle) * self._radius,
            p2[1] - math.sin(angle) * self._radius,
        )
        size = max(5, self._radius)
        left = (
            tip[0] - size * math.cos(angle - 0.45),
            tip[1] - size * math.sin(angle - 0.45),
        )
        right = (
            tip[0] - size * math.cos(angle + 0.45),
            tip[1] - size * math.sin(angle + 0.45),
        )
        self._pygame.draw.polygon(surface, colour, (tip, left, right))

    def _marker(self, surface, role: str, p: tuple[int, int], r: int) -> None:
        pygame = self._pygame
        style = theme.ROLE_STYLES[role]
        fill, rim = _hex(style.fill), _hex(style.edge)
        x, y = p
        if style.shape == "square":
            points = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        elif style.shape == "diamond":
            points = [(x, y - r - 1), (x + r + 1, y), (x, y + r + 1), (x - r - 1, y)]
        elif style.shape == "triangle":
            points = [(x, y - r - 1), (x + r + 1, y + r), (x - r - 1, y + r)]
        else:
            pygame.draw.circle(surface, fill, p, r)
            pygame.draw.circle(surface, rim, p, r, 2)
            return
        pygame.draw.polygon(surface, fill, points)
        pygame.draw.polygon(surface, rim, points, 2)

    # --------------------------------------------------------------- events
    def handle(self, event) -> str | None:
        """React to one pygame event: a click on a tab, a panel row, a button or the map,
        or a key. Returns what a stepping loop has to do ("pause", "back", "back10",
        "forward", "forward10", "start", "end", "slower", "faster", "seek:<share>"), or
        None."""
        pygame = self._pygame
        if event.type == pygame.KEYDOWN:
            views = list(VIEWS)
            key = getattr(event, "unicode", "")
            if key in ("1", "2", "3"):
                self.view = views[int(key) - 1]
                return None
            if event.key == pygame.K_ESCAPE:
                self.selected = None
                return None
            jump = "10" if event.mod & pygame.KMOD_SHIFT else ""
            return {
                pygame.K_SPACE: "pause",
                pygame.K_LEFT: "back" + jump,
                pygame.K_RIGHT: "forward" + jump,
                pygame.K_HOME: "start",
                pygame.K_END: "end",
                pygame.K_UP: "faster",
                pygame.K_DOWN: "slower",
            }.get(event.key)
        if event.type == pygame.MOUSEBUTTONUP:
            self._drag = None
        if event.type == pygame.MOUSEMOTION and self._drag is not None:
            return self._drag()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, act in reversed(self._hits):
                if rect.collidepoint(event.pos):
                    if getattr(act, "drags", False):
                        self._drag = act
                    return act()
            if self._map_rect().collidepoint(event.pos):
                self._select_at(event.pos)
        return None

    def _select_at(self, pos) -> None:
        if self._frame is not None:
            reach = max(self._radius * 1.4, 8)
            for a, agent in self._frame["step"]["agents"].items():
                if math.dist(self._pixels[agent["at"]], pos) <= reach:
                    self._pick(("agent", a))
                    return
        near = min(self._pixels, key=lambda v: math.dist(self._pixels[v], pos))
        if math.dist(self._pixels[near], pos) <= self._radius + 5:
            self._pick(("vertex", near))
        else:
            self.selected = None

    def _pick(self, selection) -> None:
        self.selected = selection
        self.panel = "selected"

    def _hit(self, rect, action: Callable[[], Any]) -> None:
        self._hits.append((rect, action))

    # --------------------------------------------------------------- frames
    def render(self, fleet: FleetState, *, storage: StorageState | None = None) -> bool:
        """Draw the agents where they stand, and the stock on the slots when `storage` is
        given. For a frame with tasks and counts, use `render_env`.

        Returns False once the window's close button was clicked, True otherwise, so a
        driving loop can stop cleanly."""
        step = {
            "t": None,
            "agents": {
                str(a): {"at": v, "goal": None, "phase": None, "kind": None}
                for a, v in fleet.locations().items()
            },
        }
        stock = None
        header = self._header
        if storage is not None:
            stock = {
                "ranking": list(storage.skus),
                "shelves": {
                    str(v): {
                        "units": {
                            k: storage.units(k, v) for k in storage.skus if storage.units(k, v)
                        },
                        "queued": 0,
                        "coming": 0,
                        "putaways": 0,
                    }
                    for v in storage.capacities
                },
            }
            header = {
                **header,
                "capacities": {str(v): c for v, c in storage.capacities.items()},
                "skus": {k: s.unit_capacity for k, s in storage.skus.items()},
            }
        return self.draw(step, stock, header=header)

    def render_env(self, env: WarehouseMAPDEnv) -> bool:
        """Draw the env at its current step. Returns False once the window was closed."""
        from slap_mapd_coupling.evaluation.trace import step_line, stock_line

        self._header = {
            **self._header,
            "capacities": {str(v): c for v, c in env.storage.capacities.items()},
            "skus": {k: s.unit_capacity for k, s in env.storage.skus.items()},
            "config": env.config.model_dump(mode="json"),
        }
        return self.draw(step_line(env, {}), stock_line(env))

    def draw(
        self,
        step: Mapping[str, Any],
        stock: Mapping[str, Any] | None,
        *,
        header: Mapping[str, Any] | None = None,
    ) -> bool:
        """Draw one step of a live run from a step line and a stock line
        (evaluation/trace.py), and keep it for stepping back. While paused it keeps the
        window open and returns once the run is continued. Returns False once the window
        was closed."""
        pygame = self._pygame
        header = header or self._header
        live = self._live
        t = step.get("t")
        if live.steps and t is not None and live.steps[-1].get("t") is not None:
            if t <= live.steps[-1]["t"]:
                self._live = live = Timeline()  # a new episode
        live.steps.append(step)
        if stock is not None and (not live.stock or (t or 0) % _LIVE_STOCK_EVERY == 0):
            live.add_stock(stock)
        head = len(live.steps) - 1
        index = head
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.paused = False
                    return False
                action = self.handle(event)
                if action is not None:
                    index = self._step_action(action, index, head)
            shown = live.steps[index]
            # The current step shows its own stock, a step back the last one kept.
            at = stock if index == head else live.stock_at(shown.get("t") or 0)
            status = "live, paused at step {} of {}".format(shown.get("t"), step.get("t"))
            self.draw_frame(
                shown,
                at,
                header,
                live.history(index),
                (index, head) if self.paused else None,
                status if self.paused else "live",
                live=True,
            )
            pygame.display.flip()
            if not self.paused:
                self._clock.tick(self.speed)
                return True
            self._clock.tick(30)

    def _step_action(self, action: str, index: int, last: int) -> int:
        """The step shown after a control action, and pause, continue or a new speed."""
        if action == "faster":
            self.speed = min(self.speed * 2, _FASTEST)
        elif action == "slower":
            self.speed = max(self.speed / 2, _SLOWEST)
        elif action.startswith("speed:"):
            share = min(max(float(action[6:]), 0.0), 1.0)
            self.speed = round(_SLOWEST * (_FASTEST / _SLOWEST) ** share, 1)
        elif action == "pause":
            self.paused = not self.paused
        elif action.startswith("seek:"):
            self.paused = True
            index = round(min(max(float(action[5:]), 0.0), 1.0) * last)
        elif action.startswith("back"):
            self.paused = True
            index = max(0, index - (10 if action.endswith("10") else 1))
        elif action.startswith("forward"):
            self.paused = True
            index = min(last, index + (10 if action.endswith("10") else 1))
        elif action == "start":
            self.paused, index = True, 0
        elif action == "end":
            index = last
        return index

    def draw_frame(
        self,
        step,
        stock,
        header,
        history=None,
        slider: tuple[int, int] | None = None,
        status: str = "",
        *,
        live: bool = False,
    ) -> None:
        """Draw one frame onto the window without handling events or flipping it. `slider`
        is (shown index, last index) of the steps that can be stepped through."""
        screen = self._screen
        self._hits = []
        history = history or [step]
        self._frame = {"step": step, "stock": stock, "header": header}
        screen.blit(self._background, (0, 0))
        debug = congestion(history)
        ranks = {k: i for i, k in enumerate(stock["ranking"])} if stock else {}
        shelves = stock["shelves"] if stock else {}
        docks = stock.get("docks", {}) if stock else {}
        if self.view == "congestion":
            self._draw_heat(debug["heat"])
        self._draw_slots(header, shelves, ranks)
        if self.view == "stock":
            self._draw_docks(docks)
        self._draw_agents(step, _trails(history), debug)
        self._draw_selection(step)
        self._draw_view_tabs()
        self._draw_sidebar(step, stock, header, ranks, debug)
        self._draw_controls(slider, status, live)
        self._draw_hover(step, shelves, header, ranks, docks)

    def save(self, path: Path) -> None:
        """Save the frame on the window as an image."""
        self._pygame.image.save(self._screen, str(path))

    # ------------------------------------------------------------------ map
    def _draw_heat(self, heat: Mapping[VertexId, int]) -> None:
        pygame = self._pygame
        if not heat:
            return
        layer = pygame.Surface(self._size, pygame.SRCALPHA)
        top = max(heat.values())
        for v, count in heat.items():
            if v not in self._pixels:
                continue
            radius = int(self._radius * (1.2 + 1.8 * count / top))
            alpha = 40 + int(110 * count / top)
            pygame.draw.circle(layer, (*_RED, alpha), self._pixels[v], radius)
        self._screen.blit(layer, (0, 0))

    def _draw_slots(self, header, shelves, ranks) -> None:
        pygame = self._pygame
        screen = self._screen
        r = self._radius + 1
        capacities = header.get("capacities", {})
        sizes = header.get("skus", {})
        rim = _hex(theme.ROLE_STYLES["storage"].edge)
        for v, p in self._pixels.items():
            if self._role(v) != "storage" and str(v) not in capacities:
                continue
            box = pygame.Rect(p[0] - r, p[1] - r, 2 * r, 2 * r)
            pygame.draw.rect(screen, _WHITE, box)
            entry = shelves.get(str(v))
            cap = capacities.get(str(v))
            if entry and cap and self.view == "stock":
                # The fill rises from the bottom, one band per SKU, most popular lowest.
                bottom = float(box.bottom)
                held = sorted(entry["units"].items(), key=lambda kv: ranks.get(kv[0], 0))
                for sku, units in held:
                    top = bottom - box.height * units * sizes.get(sku, 1.0) / cap
                    band = pygame.Rect(
                        box.left,
                        round(top),
                        box.width,
                        max(1, round(bottom) - round(top)),
                    )
                    colour = rank_colour(ranks.get(sku, 0), len(ranks))
                    pygame.draw.rect(screen, colour, band.clip(box))
                    bottom = top
            pygame.draw.rect(screen, rim if self.view == "stock" else _MUTED, box, 1)
            if entry and self.view == "stock":
                self._badges(p, r, entry)

    def _draw_docks(self, docks: Mapping[str, Mapping[str, int]]) -> None:
        """The stock waiting at each dock, SKU names and units, under the dock."""
        colour = _hex(theme.ROLE_STYLES["receiving"].edge)
        for v, units in docks.items():
            if int(v) not in self._pixels or not units:
                continue
            x, y = self._pixels[int(v)]
            held = sorted(units.items(), key=lambda kv: -kv[1])
            text = ", ".join(f"{k} {u}" for k, u in held[:2]) + (" ..." if len(held) > 2 else "")
            label = self._small.render(text, True, colour)
            self._screen.blit(label, (max(2, x - self._radius), y + self._radius + 3))

    def _badges(self, p, r, entry) -> None:
        """Waiting picks above the slot (no agent yet red, an agent on the way orange) and
        open put-aways below it (purple)."""
        pygame = self._pygame
        x, y = p
        marks = [
            (entry.get("queued", 0), "#c0392b", (x + r, y - r)),
            (entry.get("coming", 0), "#e08a1e", (x - r, y - r)),
            (entry.get("putaways", 0), "#7a4fa3", (x + r, y + r)),
        ]
        for count, colour, centre in marks:
            if not count:
                continue
            text = self._small.render(str(count), True, _WHITE)
            rect = pygame.Rect(0, 0, max(text.get_width() + 4, 12), 12)
            rect.center = centre
            pygame.draw.rect(self._screen, _hex(colour), rect, border_radius=6)
            self._screen.blit(text, text.get_rect(center=rect.center))

    def _draw_agents(self, step, trails, debug) -> None:
        """Every agent with its number, always. What else is drawn follows the view."""
        pygame = self._pygame
        screen = self._screen
        r = max(7, int(self._radius * 1.2))
        if self.view == "congestion":
            for a, blocker in debug["blocked_by"].items():
                at = step["agents"][a]["at"]
                there = step["agents"][blocker]["at"]
                pygame.draw.line(screen, _RED, self._pixels[at], self._pixels[there], 3)
        for a, agent in step["agents"].items():
            at = self._pixels[agent["at"]]
            colour = _agent_colour(agent)
            if self.view == "agents":
                trail = [self._pixels[v] for v in trails.get(a, ()) if v in self._pixels]
                if len(trail) > 1:
                    faded = tuple(round(c + (255 - c) * 0.55) for c in colour)
                    pygame.draw.lines(screen, faded, False, trail, 2)
                goal = agent.get("goal")
                if goal is not None and goal in self._pixels:
                    self._dashed(at, self._pixels[goal], colour)
                    pygame.draw.circle(screen, colour, self._pixels[goal], r + 3, 1)
                intent = agent.get("intent")
                if intent is not None and intent in self._pixels and intent != agent["at"]:
                    self._arrowhead(screen, at, self._pixels[intent], colour)
            elif self.view == "stock":
                colour = _MUTED
            else:
                still = debug["still_for"].get(a, 0)
                colour = _RED if still else _MUTED
            pygame.draw.circle(screen, colour, at, r)
            if self.view == "congestion" and (agent.get("overridden") or a in debug["blocked"]):
                pygame.draw.circle(screen, (0, 0, 0), at, r + 2, 2)
            label = self._label.render(a, True, _WHITE)
            screen.blit(label, label.get_rect(center=at))
            if self.view == "agents" and agent.get("phase") == "to_delivery" and agent.get("sku"):
                quantity = agent.get("quantity")
                carried = agent["sku"] + ("" if quantity in (None, 1) else f" x{quantity}")
                screen.blit(
                    self._small.render(carried, True, colour),
                    (at[0] + r + 2, at[1] - r - 8),
                )
            still = debug["still_for"].get(a, 0)
            if self.view == "congestion" and still >= 2:
                tag = self._small.render(f"still {still}", True, _RED)
                screen.blit(tag, (at[0] + r + 2, at[1] + 2))

    def _draw_selection(self, step) -> None:
        if self.selected is None:
            return
        pygame = self._pygame
        kind, key = self.selected
        if kind == "agent":
            agent = step["agents"].get(key)
            if agent is None:
                return
            at = self._pixels[agent["at"]]
            pygame.draw.circle(self._screen, _SELECT, at, max(7, int(self._radius * 1.2)) + 6, 3)
            if agent.get("goal") in self._pixels:
                pygame.draw.line(self._screen, _SELECT, at, self._pixels[agent["goal"]], 2)
        elif key in self._pixels:
            pygame.draw.circle(self._screen, _SELECT, self._pixels[key], self._radius + 7, 3)

    def _dashed(self, p1, p2, colour) -> None:
        length = math.dist(p1, p2)
        if length < 1:
            return
        dash = 6
        for i in range(0, int(length), 2 * dash):
            a = i / length
            b = min(i + dash, length) / length
            self._pygame.draw.line(
                self._screen,
                colour,
                (p1[0] + (p2[0] - p1[0]) * a, p1[1] + (p2[1] - p1[1]) * a),
                (p1[0] + (p2[0] - p1[0]) * b, p1[1] + (p2[1] - p1[1]) * b),
                1,
            )

    # ----------------------------------------------------------------- tabs
    def _tabs(
        self, items: Mapping[str, str], current: str, x: int, y: int, width: int, choose
    ) -> int:
        """A row of tabs, the current one filled. Returns the y below them."""
        pygame = self._pygame
        accent = _hex(theme.PALETTE["one_way_edge"])
        gap = 4
        w = (width - gap * (len(items) - 1)) / len(items)
        for i, (name, label) in enumerate(items.items()):
            tab = pygame.Rect(round(x + i * (w + gap)), y, round(w), 28)
            on = name == current
            pygame.draw.rect(self._screen, accent if on else _WHITE, tab, border_radius=6)
            pygame.draw.rect(self._screen, accent, tab, 1, border_radius=6)
            text = self._font.render(label, True, _WHITE if on else accent)
            self._screen.blit(text, text.get_rect(center=tab.center))
            self._hit(tab, lambda name=name: choose(name))
        return y + 28

    def _draw_view_tabs(self) -> None:
        width = self._size[0] - _PANEL
        self._pygame.draw.rect(self._screen, _PAPER, (0, 0, width, _TABS))
        labels = {name: f"{i + 1}  {label}" for i, (name, label) in enumerate(VIEWS.items())}
        self._tabs(labels, self.view, 10, 6, min(480, width - 20), self._set_view)

    def _set_view(self, name: str) -> None:
        self.view = name

    def _set_panel(self, name: str) -> None:
        self.panel = name

    # -------------------------------------------------------------- sidebar
    def _draw_sidebar(self, step, stock, header, ranks, debug) -> None:
        pygame = self._pygame
        screen = self._screen
        x0 = self._size[0] - _PANEL
        pygame.draw.rect(screen, _PAPER, (x0, 0, _PANEL, self._size[1]))
        pygame.draw.line(screen, _RULE, (x0, 0), (x0, self._size[1]))
        first = dict(list(PANELS.items())[:4])
        second = dict(list(PANELS.items())[4:])
        y = self._tabs(first, self.panel, x0 + 10, 6, _PANEL - 20, self._set_panel)
        y = self._tabs(second, self.panel, x0 + 10, y + 4, _PANEL - 20, self._set_panel)
        card = pygame.Rect(x0 + 10, y + 10, _PANEL - 20, self._size[1] - y - 30 - 150)
        pygame.draw.rect(screen, _WHITE, card, border_radius=8)
        pygame.draw.rect(screen, _RULE, card, 1, border_radius=8)
        self._draw_rows(self._panel_rows(step, stock, header, ranks, debug), card)
        self._draw_legend(x0 + 10, card.bottom + 10, _PANEL - 20)

    def _draw_rows(self, rows: list, card) -> None:
        """Draw a panel's rows inside its card, every row from the same left edge.

        A row is one of
        ("title", text): the panel's name, large, with a rule under it,
        ("heading", text): the name of a group of rows below it,
        ("stat", label, value, colour, action): a label on the left and its value on the
        right, the value in `colour` when one is given,
        ("cells", texts, colour, action): one line of a table, the texts in fixed columns,
        ("note", text, colour): a line of plain text.
        `action`, when given, is what a click on the row does."""
        pygame = self._pygame
        screen = self._screen
        text = _hex(theme.PALETTE["text"])
        left, right = card.left + 14, card.right - 14
        y = card.top + 12
        for row in rows:
            kind = row[0]
            height = {"title": 34, "heading": 28, "stat": 21, "cells": 21, "note": 21}[kind]
            if y + height > card.bottom - 6:
                screen.blit(self._small.render("more below, not shown", True, _MUTED), (left, y))
                break
            action = None
            if kind == "title":
                screen.blit(self._title_large.render(row[1], True, text), (left, y))
                pygame.draw.line(screen, _RULE, (left, y + 27), (right, y + 27))
            elif kind == "heading":
                screen.blit(self._heading.render(row[1].upper(), True, _MUTED), (left, y + 10))
            elif kind == "stat":
                _, label, value, colour, action = row
                screen.blit(self._font.render(label, True, text), (left, y))
                shown = self._value.render(str(value), True, colour or text)
                screen.blit(shown, (right - shown.get_width(), y))
            elif kind == "cells":
                _, cells, colour, action = row
                for x, cell in zip(_COLUMNS, cells):
                    screen.blit(self._font.render(str(cell), True, colour), (left + x, y))
            else:
                screen.blit(self._font.render(row[1], True, row[2]), (left, y))
            if action is not None:
                self._hit(pygame.Rect(card.left, y - 2, card.width, height), action)
            y += height

    def _panel_rows(self, step, stock, header, ranks, debug) -> list:
        name = self.panel
        rows: list = [("title", _PANEL_TITLES[name])]
        if name == "selected":
            return rows + (
                self._selected_rows(step, stock, header, ranks, debug)
                or [("note", "Click an agent or a vertex on the map.", _MUTED)]
            )
        if name == "episode":
            config = header.get("config", {})
            t = step.get("t")
            who = config.get("planner") or config.get("controller")
            if who:
                rows.append(("heading", "Run"))
                rows.append(_stat("routing", who))
                rows.append(_stat("storage rule", config.get("storage_mode")))
                rows.append(_stat("agents", config.get("num_agents")))
                rows.append(_stat("seed", config.get("seed")))
            if t is not None:
                rows.append(("heading", "Progress"))
                horizon = config.get("horizon")
                rows.append(_stat("step", f"{t} of {horizon}" if horizon else t))
                if t < (config.get("warmup") or 0):
                    rows.append(("note", "in the warm-up, not scored", _MUTED))
            if "orders_done" in step:
                rows.append(
                    _stat("orders done", f"{step['orders_done']} of {step['orders_released']}")
                )
            if "completed" in step:
                rows.append(_stat("tasks done", f"{step['completed']} of {step['released']}"))
                rows.append(
                    _stat("tasks done per step", f"{step['completed'] / max(t or 1, 1):.3f}")
                )
                rows.append(_stat("backlog", step["backlog"]))
            return rows
        if stock is None and name in ("stock", "docks"):
            return rows + [("note", "This trace has no stock.", _MUTED)]
        shelves = list(stock["shelves"].values()) if stock else []
        if name == "stock":
            capacities = header.get("capacities", {})
            sizes = header.get("skus", {})
            totals: dict[str, int] = {}
            used = 0.0
            for e in shelves:
                for k, units in e["units"].items():
                    totals[k] = totals.get(k, 0) + units
                    used += units * sizes.get(k, 1.0)
            rows.append(("heading", "Slots"))
            rows.append(_stat("full", f"{used / max(sum(capacities.values()), 1e-9):.0%}"))
            if "orders_short" in stock:
                queued = sum(e["queued"] for e in shelves)
                coming = sum(e["coming"] for e in shelves)
                rows.append(_stat("picks waiting for an agent", queued, _hex("#c0392b")))
                rows.append(_stat("picks with an agent coming", coming, _hex("#e08a1e")))
                rows.append(_stat("order lines short of stock", stock["orders_short"]))
            rows.append(("heading", "Most popular SKUs, units on the slots"))
            for k in stock["ranking"][:8]:
                rank = ranks[k]
                rows.append(
                    _stat(f"{rank + 1}. {k}", totals.get(k, 0), rank_colour(rank, len(ranks)))
                )
            return rows
        if name == "docks":
            dock_colour = _hex(theme.ROLE_STYLES["receiving"].edge)
            docks = stock.get("docks", {})
            putaways = sum(e.get("putaways", 0) for e in shelves)
            rows.append(("heading", "Arrived stock"))
            rows.append(
                _stat("units waiting at the docks", sum(sum(u.values()) for u in docks.values()))
            )
            rows.append(_stat("put-aways not yet on a slot", putaways))
            for v, units in docks.items():
                rows.append(("heading", f"Dock {v}"))
                for k, u in sorted(units.items(), key=lambda kv: -kv[1]):
                    rows.append(
                        _stat(k, u, dock_colour, lambda v=int(v): self._pick(("vertex", v)))
                    )
            return rows
        if name == "congestion":
            rows.append(("heading", "Now"))
            rows.append(_stat("agents still with a task", len(debug["still_with_task"]), _RED))
            rows.append(_stat("agents blocked", len(debug["blocked"]), _RED))
            rows.append(_stat("agents overridden", debug["overridden"], _RED))
            longest = max(debug["still_for"].values(), default=0)
            rows.append(_stat("longest still", f"{longest} steps", _RED))
            rows.append(_stat("whole fleet still", f"{debug['fleet_still']} steps", _RED))
            if debug["blocked_by"]:
                rows.append(("heading", "Who waits on whom"))
                for a, blocker in sorted(debug["blocked_by"].items()):
                    rows.append(
                        _stat(
                            f"agent {a}",
                            f"waits on {blocker}",
                            _RED,
                            lambda a=a: self._pick(("agent", a)),
                        )
                    )
            rows.append(("heading", f"Still steps, last {_WINDOW} steps"))
            for v, c in sorted(debug["heat"].items(), key=lambda kv: -kv[1])[:6]:
                rows.append(_stat(f"vertex {v}", c, _RED, lambda v=v: self._pick(("vertex", v))))
            return rows
        if name == "agents":
            rows.append(("cells", ("agent", "at", "doing", "goal", "still"), _MUTED, None))
            for a, agent in step["agents"].items():
                goal = agent.get("goal")
                cells = (
                    a,
                    agent["at"],
                    _doing(agent),
                    "-" if goal is None else goal,
                    debug["still_for"].get(a, 0),
                )
                rows.append(
                    ("cells", cells, _agent_colour(agent), lambda a=a: self._pick(("agent", a)))
                )
            return rows
        recent = list(reversed(step.get("recent", [])))
        if not recent:
            return rows + [("note", "No task done yet.", _MUTED)]
        for t_done, kind, sku, quantity, source, target, agent_id in recent:
            what = "put away" if kind == "putaway" else "picked"
            rows.append(("heading", f"Step {t_done}, agent {agent_id}"))
            rows.append(
                _stat(
                    f"{sku} x{quantity} {what}",
                    f"{source} to {target}",
                    None,
                    lambda a=agent_id: self._pick(("agent", str(a))),
                )
            )
        return rows

    def _selected_rows(self, step, stock, header, ranks, debug) -> list:
        if self.selected is None:
            return []
        kind, key = self.selected
        if kind == "agent":
            agent = step["agents"].get(key)
            if agent is None:
                return []
            level = agent.get("congestion")
            rows = [
                ("heading", f"Agent {key}"),
                _stat("doing", _doing(agent), _agent_colour(agent)),
                _stat("at vertex", agent["at"]),
                _stat(
                    "next vertex", agent.get("intent") if agent.get("intent") is not None else "-"
                ),
                _stat("goal", agent.get("goal") if agent.get("goal") is not None else "-"),
                ("heading", "Task"),
                _stat("task", agent.get("task") if agent.get("task") is not None else "none"),
                _stat("kind", agent.get("kind") or "-"),
            ]
            if agent.get("sku") is not None:
                rows.append(_stat("SKU", f"{agent['sku']} x{agent.get('quantity') or 1}"))
            rows.append(("heading", "Congestion"))
            rows.append(_stat("still for", f"{debug['still_for'].get(key, 0)} steps"))
            rows.append(_stat("congestion", "-" if level is None else f"{level:.2f}"))
            if key in debug["blocked_by"]:
                blocker = debug["blocked_by"][key]
                rows.append(
                    _stat(
                        "blocked by",
                        f"agent {blocker}",
                        _RED,
                        lambda: self._pick(("agent", blocker)),
                    )
                )
            if agent.get("overridden"):
                rows.append(("note", "overridden by conflict resolution", _RED))
            return rows
        v = key
        rows = [("heading", f"Vertex {v}, {_ROLE_LABELS.get(self._role(v) or '', 'aisle')}")]
        shelves = stock["shelves"] if stock else {}
        entry = shelves.get(str(v))
        cap = header.get("capacities", {}).get(str(v))
        if cap is not None:
            sizes = header.get("skus", {})
            held = entry["units"] if entry else {}
            used = sum(u * sizes.get(k, 1.0) for k, u in held.items())
            rows.append(_stat("space used", f"{used:g} of {cap:g}"))
        if entry:
            rows.append(_stat("picks waiting", entry["queued"], _hex("#c0392b")))
            rows.append(_stat("agent coming", entry["coming"], _hex("#e08a1e")))
            rows.append(_stat("put-aways coming", entry["putaways"], _hex("#7a4fa3")))
            if entry["units"]:
                rows.append(("heading", "Units by SKU"))
                for k, u in sorted(entry["units"].items(), key=lambda kv: ranks.get(kv[0], 0)):
                    rank = ranks.get(k, 0)
                    rows.append(_stat(f"{k}, rank {rank + 1}", u, rank_colour(rank, len(ranks))))
            if entry.get("incoming"):
                rows.append(("heading", "On its way in"))
                for k, u in entry["incoming"].items():
                    rows.append(_stat(k, u, _hex("#7a4fa3")))
        waiting = (stock or {}).get("docks", {}).get(str(v), {})
        if waiting:
            rows.append(("heading", "Waiting here"))
            for k, u in waiting.items():
                rows.append(_stat(k, u, _hex("#7a4fa3")))
        agents = [
            (a, agent, "here" if agent["at"] == v else "walks here")
            for a, agent in step["agents"].items()
            if agent["at"] == v or agent.get("goal") == v
        ]
        if agents:
            rows.append(("heading", "Agents"))
            for a, agent, where in agents:
                rows.append(
                    _stat(
                        f"agent {a}",
                        where,
                        _agent_colour(agent),
                        lambda a=a: self._pick(("agent", a)),
                    )
                )
        rows.append(_stat(f"still steps here, last {_WINDOW}", debug["heat"].get(v, 0), _RED))
        return rows

    def _draw_legend(self, x: int, y: int, width: int) -> None:
        """What the colours of the current map view mean."""
        pygame = self._pygame
        screen = self._screen
        text = _hex(theme.PALETTE["text"])
        screen.blit(self._title.render(f"{VIEWS[self.view]} view", True, text), (x, y))
        y += 24

        def dot(cx, cy, colour, label):
            pygame.draw.circle(screen, colour, (cx + 6, cy + 7), 6)
            screen.blit(self._small.render(label, True, text), (cx + 16, cy + 2))

        if self.view == "agents":
            dot(x, y, _hex(_PHASE_COLOURS[None]), "idle")
            dot(x + 120, y, _hex(_PHASE_COLOURS["to_pickup"]), "to pickup")
            dot(x, y + 20, _hex(_PHASE_COLOURS["to_delivery"]), "to workstation")
            dot(x + 120, y + 20, _hex(_PHASE_COLOURS["putaway"]), "put-away")
            screen.blit(
                self._small.render("dashed line: where the agent walks", True, text),
                (x, y + 44),
            )
        elif self.view == "stock":
            screen.blit(
                self._small.render("slot fill by SKU rank, popular to rare", True, text),
                (x, y),
            )
            for i in range(160):
                colour = rank_colour(i, 160)
                pygame.draw.line(screen, colour, (x + i * 2, y + 16), (x + i * 2, y + 26), 2)
            dot(x, y + 34, _hex("#c0392b"), "picks waiting")
            dot(x + 120, y + 34, _hex("#e08a1e"), "agent coming")
            dot(x + 240, y + 34, _hex("#7a4fa3"), "put-aways")
        else:
            dot(x, y, _RED, "agent still with a task")
            dot(x + 180, y, _MUTED, "moving or idle")
            screen.blit(
                self._small.render("red patch: still steps here, last 40 steps", True, text),
                (x, y + 22),
            )
            screen.blit(
                self._small.render("red line: blocked by that agent", True, text),
                (x, y + 40),
            )

    # ------------------------------------------------------------- controls
    def _draw_controls(self, slider, status: str, live: bool) -> None:
        pygame = self._pygame
        screen = self._screen
        width = self._size[0] - _PANEL
        y = self._size[1] - _BAR
        pygame.draw.rect(screen, _PAPER, (0, y, width, _BAR))
        pygame.draw.line(screen, _RULE, (0, y), (width, y))
        accent = _hex(theme.PALETTE["one_way_edge"])
        x = 12
        if slider is None and not live:
            return
        pause = "continue" if self.paused else "pause"
        buttons = [(pause, "pause")]
        if slider is not None:
            buttons = [
                ("|<", "start"),
                ("<", "back"),
                (pause, "pause"),
                (">", "forward"),
                (">|", "end"),
            ]
        for label, action in buttons:
            text = self._font.render(label, True, _WHITE)
            button = pygame.Rect(x, y + 12, max(text.get_width() + 18, 34), 28)
            pygame.draw.rect(screen, accent, button, border_radius=6)
            screen.blit(text, text.get_rect(center=button.center))
            self._hit(button, lambda action=action: action)
            x = button.right + 6
        # The speed slider sits at the right end of the bar, the step slider between.
        speed_track = pygame.Rect(width - 170, y + 28, 150, 8)
        share = math.log(self.speed / _SLOWEST) / math.log(_FASTEST / _SLOWEST)
        self._slider(speed_track, share, "speed:")
        label = self._small.render(
            f"speed {self.speed:g} steps a second", True, _hex(theme.PALETTE["text"])
        )
        screen.blit(label, (speed_track.left, y + 4))
        if slider is not None:
            index, last = slider
            track = pygame.Rect(x + 10, y + 28, speed_track.left - x - 40, 8)
            self._slider(track, index / max(last, 1), "seek:")
            x = track.left
        screen.blit(self._small.render(status, True, _hex(theme.PALETTE["text"])), (x, y + 4))

    def _slider(self, track, share: float, action: str) -> None:
        """A slider bar filled to `share`, which answers a click or a drag with `action`
        followed by the share under the mouse."""
        pygame = self._pygame
        accent = _hex(theme.PALETTE["one_way_edge"])
        pygame.draw.rect(self._screen, _hex(theme.PALETTE["edge"]), track, border_radius=3)
        done = pygame.Rect(track.left, track.top, int(track.width * share), track.height)
        pygame.draw.rect(self._screen, accent, done, border_radius=3)
        pygame.draw.circle(self._screen, accent, (done.right, track.centery), 8)

        def act() -> str:
            return action + str((pygame.mouse.get_pos()[0] - track.left) / track.width)

        act.drags = True  # type: ignore[attr-defined]
        self._hit(track.inflate(0, 24), act)

    # ---------------------------------------------------------------- hover
    def _draw_hover(self, step, shelves, header, ranks, docks) -> None:
        pygame = self._pygame
        mx, my = pygame.mouse.get_pos()
        if not self._map_rect().collidepoint((mx, my)):
            return
        near = min(self._pixels, key=lambda v: math.dist(self._pixels[v], (mx, my)))
        if math.dist(self._pixels[near], (mx, my)) > self._radius + 4:
            return
        lines = [f"vertex {near}, {_ROLE_LABELS.get(self._role(near) or '', 'aisle')}"]
        for a, agent in step["agents"].items():
            if agent["at"] == near:
                lines.append(f"agent {a}, {_doing(agent)}")
        lines.append("click for details")
        surfaces = [self._small.render(s, True, (20, 20, 20)) for s in lines]
        w = max(s.get_width() for s in surfaces) + 12
        h = 15 * len(surfaces) + 8
        x = min(mx + 14, self._size[0] - _PANEL - w - 4)
        y = min(my + 14, self._size[1] - _BAR - h - 4)
        pygame.draw.rect(self._screen, (255, 255, 240), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self._screen, (120, 120, 120), (x, y, w, h), 1, border_radius=4)
        for i, s in enumerate(surfaces):
            self._screen.blit(s, (x + 6, y + 4 + 15 * i))

    def close(self) -> None:
        self._pygame.quit()


def _trails(history: Sequence[Mapping[str, Any]]) -> dict[str, list[VertexId]]:
    """Where every agent stood over the last steps of `history`."""
    trails: dict[str, list[VertexId]] = {}
    for step in history:
        for a, agent in step["agents"].items():
            trail = trails.setdefault(a, [])
            if not trail or trail[-1] != agent["at"]:
                trail.append(agent["at"])
    return {a: trail[-_TRAIL:] for a, trail in trails.items()}


def congestion(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the congestion view shows, from the step lines up to the current one.

    `still_for` is how many steps in a row an agent holding a task has not moved, up to now.
    `heat` counts, per vertex, the steps an agent holding a task stood still there.
    `blocked` are the agents that wanted to move and did not, and `blocked_by` names the
    agent standing on the vertex one of them wanted. `fleet_still` is how many steps in a
    row no agent has moved while one held a task, the stall of the fleet."""
    still_for: dict[str, int] = {}
    heat: dict[VertexId, int] = {}
    fleet_still = 0
    for step in history:
        moved_any = False
        busy = False
        for a, agent in step["agents"].items():
            moved_any |= bool(agent.get("moved"))
            holding = agent.get("task") is not None
            busy |= holding
            if holding and not agent.get("moved"):
                still_for[a] = still_for.get(a, 0) + 1
                heat[agent["at"]] = heat.get(agent["at"], 0) + 1
            else:
                still_for[a] = 0
        fleet_still = fleet_still + 1 if busy and not moved_any else 0
    current = history[-1]["agents"] if history else {}
    where = {agent["at"]: a for a, agent in current.items()}
    blocked = {
        a
        for a, agent in current.items()
        if agent.get("intent") is not None
        and agent["intent"] != agent["at"]
        and not agent.get("moved")
    }
    blocked_by = {
        a: where[current[a]["intent"]]
        for a in blocked
        if where.get(current[a]["intent"]) not in (None, a)
    }
    return {
        "still_for": still_for,
        "still_with_task": [a for a, n in still_for.items() if n > 0],
        "heat": heat,
        "blocked": blocked,
        "blocked_by": blocked_by,
        "overridden": sum(1 for agent in current.values() if agent.get("overridden")),
        "fleet_still": fleet_still,
    }


def replay(path: Path, *, window_size: tuple[int, int] = (1440, 900), fps: float = 8.0) -> None:
    """Open a trace in a window and step through it (controls in the module docstring)."""
    data = TraceReplay(path)
    if not data.steps:
        raise ValueError(f"{path} has no steps.")
    renderer = WarehouseRenderer(None, header=data.header, window_size=window_size)
    renderer.paused = True
    renderer.speed = fps
    pygame = renderer._pygame
    clock = pygame.time.Clock()
    index = 0
    last = len(data.steps) - 1
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    return
                action = renderer.handle(event)
                if action is not None:
                    index = renderer._step_action(action, index, last)
            step = data.steps[index]
            state = "paused" if renderer.paused else "playing"
            status = f"{path.name}   step {step['t']} of {data.steps[-1]['t']}   {state}"
            renderer.draw_frame(
                step,
                data.stock_at(step["t"]),
                data.header,
                data.history(index),
                (index, last),
                status,
            )
            pygame.display.flip()
            if not renderer.paused:
                if index == last:
                    renderer.paused = True
                else:
                    index += 1
            clock.tick(30 if renderer.paused else renderer.speed)
    finally:
        renderer.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay an episode trace in a window.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--fps", type=float, default=8.0, help="steps per second when playing")
    parser.add_argument("--size", type=int, nargs=2, default=(1440, 900), metavar=("W", "H"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    replay(args.trace, window_size=tuple(args.size), fps=args.fps)


if __name__ == "__main__":
    main()
