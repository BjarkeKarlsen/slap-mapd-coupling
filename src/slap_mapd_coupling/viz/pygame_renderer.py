"""Interactive warehouse renderer (pygame). Optional -- see pyproject.toml's [viz] extra.

Two ways to use it. Live, build a `WarehouseRenderer` once and call
`.render_env(env)` after every step. Afterwards, replay a trace file
(evaluation/trace.py), or two side by side on the same step:

    python -m slap_mapd_coupling.viz.pygame_renderer trace.jsonl
    python -m slap_mapd_coupling.viz.pygame_renderer fixed.jsonl --compare demand.jsonl

A frame is drawn from the same lines the trace writes (a step line and a
stock line), so a live run and the replay of its trace look the same.

The window shows one thing at a time. The tabs above the map choose what
the map shows, and the tiles next to them always show the key numbers:

- Agents. Every agent with its number, a dashed line to where it is
  walking, its last steps and the SKU it carries.
- Stock. The units on every slot, split by SKU popularity rank and filled
  against the slot capacity, the tasks waiting on each slot, the units
  waiting at each dock and the open picks of each workstation.
- Congestion. A red patch on every vertex where agents holding a task stood
  still over the last 40 steps, how long each still agent has waited, and a
  red line from a blocked agent to the agent standing where it wanted to go.

The tabs of the sidebar choose one panel: the episode counts, the stock,
the docks, the congestion, the agents, the last tasks done, or the details
of what is selected. Click an agent or a vertex on the map, or a row of a
panel, to select it. Click a column title of the agent table to sort by it.

The map zooms with the mouse wheel and pans when dragged. The bar at the
bottom has a timeline of the backlog over the episode, with a red mark
where the whole fleet stood still and an orange mark where agents were
blocked, and a click on it jumps there. Below it are the buttons to pause,
step and continue, the step slider and the speed slider. A live run keeps
every step it has drawn, so it can be paused, stepped back through and
continued from where it stopped.

Keys: space pauses or continues, left and right step (with shift, 10
steps), home and end jump, up and down halve or double the speed, 1 to 3
choose the map view, F follows the selected agent, R resets the zoom, S
saves a screenshot, H shows the help and the legend, Escape clears the
selection.
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
_TOP = 52  # height of the strip above the map, the view tabs and the key numbers
_BAR = 104  # height of the bar at the bottom, the timeline and the controls
_TRAIL = 12  # steps of past positions drawn behind an agent
_WINDOW = 40  # steps of history the congestion view counts over
# Steps between two stock snapshots a live run keeps for stepping back. The current step
# always shows the stock of that step.
_LIVE_STOCK_EVERY = 10
# The speed slider runs from the slowest to the fastest speed, in steps a second, on a log
# scale.
_SLOWEST, _FASTEST = 0.5, 240.0
_ZOOM_MIN, _ZOOM_MAX = 1.0, 8.0
_DRAG = 5  # pixels the mouse must move before a press on the map is a pan, not a click

_WHITE = (255, 255, 255)
_PAPER = (246, 247, 249)
_RULE = (218, 223, 229)
_RED = (200, 30, 30)
_ORANGE = (224, 138, 30)
_PURPLE = (122, 79, 163)
_SELECT = (20, 120, 220)
_MUTED = (140, 148, 158)

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
# The panels of the sidebar, one at a time, in tab order, and the title of each.
PANELS = {
    "episode": "Episode",
    "stock": "Stock",
    "docks": "Docks",
    "congestion": "Congestion",
    "agents": "Agents",
    "done": "Done",
    "selected": "Selected",
}
_PANEL_TITLES = {**PANELS, "done": "Last tasks done"}
# What the agent list can be sorted by: a name and the sort key.
_SORTS = (
    ("agent", lambda a, agent, still: int(a)),
    ("doing", lambda a, agent, still: _doing(agent)),
    ("goal", lambda a, agent, still: -1 if agent.get("goal") is None else agent["goal"]),
    ("still", lambda a, agent, still: -still),
)
_HELP = (
    ("space", "pause or continue"),
    ("left, right", "step back or forward, with shift 10 steps"),
    ("home, end", "first or last step"),
    ("up, down", "faster or slower"),
    ("1, 2, 3", "agents, stock or congestion view"),
    ("wheel, drag", "zoom and pan the map"),
    ("F", "follow the selected agent"),
    ("R", "reset the zoom"),
    ("S", "save a screenshot"),
    ("H", "this help"),
    ("Escape", "clear the selection"),
)


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


def _pretty(name: Any) -> str:
    """A config name as a reader would write it: token_passing becomes Token passing."""
    text = str(name).replace("_", " ")
    return text[:1].upper() + text[1:]


def _steps(n: int) -> str:
    return f"{n} step" + ("" if n == 1 else "s")


def _stat(label: str, value: Any, colour=None, action=None) -> tuple:
    """A panel row with a label on the left and its value on the right."""
    return ("stat", label, value, colour, action)


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
        # Per step, the backlog and whether the fleet stood still or an agent was blocked.
        self.backlog: list[int] = []
        self.stalled: list[bool] = []
        self.blocked: list[bool] = []

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

    def marks(self) -> None:
        """Bring the per-step series of the timeline up to date with the steps."""
        for step in self.steps[len(self.backlog) :]:
            agents = step["agents"].values()
            self.backlog.append(step.get("backlog", 0))
            busy = any(agent.get("task") is not None for agent in agents)
            self.stalled.append(busy and not any(agent.get("moved") for agent in agents))
            self.blocked.append(
                any(
                    agent.get("intent") is not None
                    and agent["intent"] != agent["at"]
                    and not agent.get("moved")
                    for agent in agents
                )
            )


class TraceReplay(Timeline):
    """A trace file, read once.

    The units on the slots come from the stock lines, written every few steps. What
    waits on each slot, dock and workstation is worked out at every step from the task
    list at the end of the trace, since a put-away often waits at its dock for fewer
    steps than lie between two stock lines. A trace without a task list shows the stock
    lines as they are."""

    def __init__(self, path: Path) -> None:
        from slap_mapd_coupling.evaluation.trace import load_stock, load_tasks, load_trace

        super().__init__()
        self.path = path
        self.header, steps = load_trace(path)
        self.steps = steps
        for line in load_stock(path):
            self.add_stock(line)
        self.tasks = load_tasks(path)
        self.marks()

    def stock_at(self, t: int) -> Mapping[str, Any] | None:
        line = super().stock_at(t)
        if line is None or not self.tasks:
            return line
        shelves = {
            v: {"units": e["units"], "queued": 0, "coming": 0, "putaways": 0, "incoming": {}}
            for v, e in line["shelves"].items()
        }
        docks: dict[str, dict[str, int]] = {}
        workstations: dict[str, int] = {}

        def shelf(v: int) -> dict[str, Any]:
            return shelves.setdefault(
                str(v), {"units": {}, "queued": 0, "coming": 0, "putaways": 0, "incoming": {}}
            )

        def after(time: int | None) -> bool:
            return time is None or time > t

        # The same rules as stock_line in evaluation/trace.py, read from the task list.
        for task in self.tasks:
            if task["released"] > t:
                continue
            if task["kind"] == "pick":
                if after(task["finished"]):
                    workstations[str(task["to"])] = workstations.get(str(task["to"]), 0) + 1
                if after(task["pickup_at"]):
                    held = task["assigned"] is not None and task["assigned"] <= t
                    shelf(task["from"])["coming" if held else "queued"] += 1
            elif after(task["finished"]):
                entry = shelf(task["to"])
                entry["putaways"] += 1
                entry["incoming"][task["sku"]] = (
                    entry["incoming"].get(task["sku"], 0) + task["quantity"]
                )
                if after(task["pickup_at"]):
                    at_dock = docks.setdefault(str(task["from"]), {})
                    at_dock[task["sku"]] = at_dock.get(task["sku"], 0) + task["quantity"]
        return {**line, "shelves": shelves, "docks": docks, "workstations": workstations}


class WarehouseRenderer:
    """Build once, then call `.render_env(env)` (or `.render(fleet)`) every step.

    Pixel positions come from the same `positions` dict the matplotlib
    backend uses, so both backends agree on a given instance. The static
    graph is drawn onto a cached background surface, redrawn only when the
    map is zoomed or panned. The tabs and the buttons are drawn every frame,
    and a click is matched against what the last frame drew.

    With `surface` the renderer draws onto that surface, a part of a larger
    window, instead of opening its own window. Without `sidebar` it draws
    the map, the key numbers and the bar only, as each half of a comparison
    does.
    """

    def __init__(
        self,
        graph: WarehouseGraph | None,
        positions: Mapping[VertexId, tuple[float, float]] | None = None,
        window_size: tuple[int, int] = (1440, 900),
        *,
        header: Mapping[str, Any] | None = None,
        surface: Any = None,
        sidebar: bool = True,
        title: str = "",
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
        if surface is None:
            pygame.display.set_caption("slap-mapd-coupling")
            surface = pygame.display.set_mode(window_size)
        self._screen = surface
        self._offset = surface.get_abs_offset()
        self._size = surface.get_size()
        self._panel_w = _PANEL if sidebar else 0
        self._title = title
        self._header = header
        self._font = pygame.font.Font(None, 20)
        self._small = pygame.font.Font(None, 16)
        self._heading = pygame.font.Font(None, 15)
        self._heading.set_bold(True)
        self._title_font = pygame.font.Font(None, 22)
        self._title_font.set_bold(True)
        self._title_large = pygame.font.Font(None, 28)
        self._title_large.set_bold(True)
        self._value = pygame.font.Font(None, 20)
        self._value.set_bold(True)
        self._label = pygame.font.Font(None, 18)
        self._label.set_bold(True)
        self._roles = {v["id"]: v for v in header["vertices"]}
        self._edges = {(int(s), int(t)) for s, t, _ in header["edges"]}
        positions = {int(v): (x, y) for v, (x, y) in header["positions"].items()}
        self._base, unit = self._fit(positions)
        self._base_radius = max(3, int(unit * 0.22))
        self.zoom = 1.0
        self._pan = (0.0, 0.0)
        self._place()
        self.view = "agents"
        self.panel = "episode"
        self.selected: tuple[str, Any] | None = None  # ("agent", "3") or ("vertex", 41)
        self.follow = False
        self.help = False
        self.hide_idle = False
        self.sort = "agent"
        self.paused = False
        self.speed = 8.0  # steps a second, live and in a replay
        self._clock = pygame.time.Clock()
        self._drag: Callable[[], Any] | None = None
        self._press: tuple[int, int] | None = None
        self._panning = False
        self._note = ""
        self._scroll: dict[str, int] = {}  # how far each sidebar panel is scrolled down
        self._live = Timeline()
        self._hits: list[tuple[Any, Callable[[], Any]]] = []
        self._frame: Mapping[str, Any] | None = None

    # ------------------------------------------------------------ geometry
    def _map_rect(self):
        return self._pygame.Rect(
            0, _TOP, self._size[0] - self._panel_w, self._size[1] - _TOP - _BAR
        )

    def _fit(self, positions) -> tuple[dict[VertexId, tuple[float, float]], float]:
        """Pixel positions that fit the whole warehouse in the map, and the shortest edge."""
        xs = [x for x, _ in positions.values()]
        ys = [y for _, y in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        area = self._map_rect()
        scale = min(
            (area.width - 2 * _MARGIN) / max(x_max - x_min, 1e-6),
            (area.height - 2 * _MARGIN) / max(y_max - y_min, 1e-6),
        )
        # Flip y: positions grow upwards, pygame's y grows downwards.
        base = {
            v: (area.left + _MARGIN + (x - x_min) * scale, area.top + _MARGIN + (y_max - y) * scale)
            for v, (x, y) in positions.items()
        }
        lengths = [math.dist(base[s], base[t]) for s, t in self._edges if base[s] != base[t]]
        return base, max(min(lengths, default=20.0), 6.0)

    def _place(self) -> None:
        """Pixel positions and marker size at the current zoom and pan, and the background."""
        centre = self._map_rect().center
        self._pixels = {
            v: (
                int(centre[0] + (x - centre[0]) * self.zoom + self._pan[0]),
                int(centre[1] + (y - centre[1]) * self.zoom + self._pan[1]),
            )
            for v, (x, y) in self._base.items()
        }
        self._radius = max(3, int(self._base_radius * self.zoom**0.75))
        self._background = self._render_background()

    def _zoom_at(self, factor: float, at: tuple[int, int]) -> None:
        """Zoom by `factor`, keeping the point under the mouse where it is."""
        zoom = min(max(self.zoom * factor, _ZOOM_MIN), _ZOOM_MAX)
        factor = zoom / self.zoom
        centre = self._map_rect().center
        dx, dy = self._pan
        self._pan = (
            (at[0] - centre[0]) * (1 - factor) + dx * factor,
            (at[1] - centre[1]) * (1 - factor) + dy * factor,
        )
        self.zoom = zoom
        if zoom == _ZOOM_MIN:
            self._pan = (0.0, 0.0)
        self._place()

    def reset_view(self) -> None:
        self.zoom, self._pan, self.follow = 1.0, (0.0, 0.0), False
        self._place()

    def _follow(self, step) -> None:
        """Keep the selected agent in the middle of the map."""
        if not self.follow or self.selected is None or self.selected[0] != "agent":
            return
        agent = step["agents"].get(self.selected[1])
        if agent is None:
            return
        centre = self._map_rect().center
        x, y = self._base[agent["at"]]
        pan = ((centre[0] - x) * self.zoom, (centre[1] - y) * self.zoom)
        if pan != self._pan:
            self._pan = pan
            self._place()

    def _role(self, v: VertexId) -> str | None:
        entry = self._roles.get(v, {})
        for name in theme.ROLE_PRIORITY:
            if entry.get(name):
                return name
        return None

    def _mouse(self) -> tuple[int, int]:
        x, y = self._pygame.mouse.get_pos()
        return x - self._offset[0], y - self._offset[1]

    # ---------------------------------------------------------- background
    def _render_background(self):
        pygame = self._pygame
        surface = pygame.Surface(self._size)
        surface.fill(_WHITE)
        surface.set_clip(self._map_rect())
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
        surface.set_clip(None)
        return surface

    def _arrowhead(self, surface, p1, p2, colour) -> None:
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        tip = (p2[0] - math.cos(angle) * self._radius, p2[1] - math.sin(angle) * self._radius)
        size = max(5, self._radius)
        left = (tip[0] - size * math.cos(angle - 0.45), tip[1] - size * math.sin(angle - 0.45))
        right = (tip[0] - size * math.cos(angle + 0.45), tip[1] - size * math.sin(angle + 0.45))
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
        """React to one pygame event: a click on a tab, a panel row, a button, the timeline
        or the map, a drag or a wheel on the map, or a key. Returns what a stepping loop
        has to do ("pause", "back", "back10", "forward", "forward10", "start", "end",
        "slower", "faster", "seek:<share>", "speed:<share>"), or None."""
        pygame = self._pygame
        if event.type == pygame.KEYDOWN:
            return self._key(event)
        pos = getattr(event, "pos", None)
        if pos is not None:
            pos = (pos[0] - self._offset[0], pos[1] - self._offset[1])
        if event.type == pygame.MOUSEWHEEL:
            at = self._mouse()
            if self._map_rect().collidepoint(at):
                self._zoom_at(1.25 if event.y > 0 else 0.8, at)
            elif self._panel_w and at[0] >= self._size[0] - self._panel_w:
                self._scroll[self.panel] = max(0, self._scroll.get(self.panel, 0) - 40 * event.y)
            return None
        if event.type == pygame.MOUSEBUTTONUP:
            self._drag = None
            pressed, self._press = self._press, None
            if pressed is not None and not self._panning:
                self._select_at(pressed)
            self._panning = False
            return None
        if event.type == pygame.MOUSEMOTION:
            if self._drag is not None:
                return self._drag()
            if self._press is not None and (self._panning or math.dist(self._press, pos) > _DRAG):
                self._panning = True
                dx, dy = event.rel
                self._pan = (self._pan[0] + dx, self._pan[1] + dy)
                self.follow = False
                self._place()
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not (0 <= pos[0] < self._size[0] and 0 <= pos[1] < self._size[1]):
                return None
            for rect, act in reversed(self._hits):
                if rect.collidepoint(pos):
                    if getattr(act, "drags", False):
                        self._drag = act
                    return act()
            if self._map_rect().collidepoint(pos):
                self._press = pos
        return None

    def _key(self, event) -> str | None:
        pygame = self._pygame
        key = getattr(event, "unicode", "").lower()
        views = list(VIEWS)
        if key in ("1", "2", "3"):
            self.view = views[int(key) - 1]
        elif key == "f":
            self.follow = not self.follow
        elif key == "r":
            self.reset_view()
        elif key == "h":
            self.help = not self.help
        elif key == "s":
            return "shot"
        elif event.key == pygame.K_ESCAPE:
            self.selected = None
            self.follow = False
        else:
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
            self.follow = False

    def _pick(self, selection) -> None:
        self.selected = selection
        if self._panel_w:
            self.panel = "selected"

    def _hit(self, rect, action: Callable[[], Any]) -> None:
        self._hits.append((rect, action))

    def screenshot(self, name: str = "viewer") -> Path:
        """Save the whole window as a PNG in the working directory and return its path."""
        t = self._frame["step"].get("t") if self._frame else None
        path = Path.cwd() / f"{name}_t{t}.png"
        self._pygame.image.save(self._pygame.display.get_surface(), str(path))
        self._note = f"saved {path.name}"
        return path

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
        live.marks()
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
                if action == "shot":
                    self.screenshot()
                elif action is not None:
                    index = self._step_action(action, index, head)
            shown = live.steps[index]
            # The current step shows its own stock, a step back the last one kept.
            at = stock if index == head else live.stock_at(shown.get("t") or 0)
            status = "live, paused" if self.paused else "live"
            self.draw_frame(
                shown, at, header, live.history(index), (index, head), status,
                live=True, timeline=live,
            )  # fmt: skip
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
        timeline: Timeline | None = None,
    ) -> None:
        """Draw one frame onto the surface without handling events or flipping it. `slider`
        is (shown index, last index) of the steps that can be stepped through, and
        `timeline` holds those steps for the chart above the controls."""
        pygame = self._pygame
        screen = self._screen
        self._hits = []
        history = history or [step]
        self._frame = {"step": step, "stock": stock, "header": header}
        self._follow(step)
        screen.blit(self._background, (0, 0))
        debug = congestion(history)
        ranks = {k: i for i, k in enumerate(stock["ranking"])} if stock else {}
        self._sku_count = len(ranks)
        shelves = stock["shelves"] if stock else {}
        docks = stock.get("docks", {}) if stock else {}
        screen.set_clip(self._map_rect())
        if self.view == "congestion":
            self._draw_heat(debug["heat"])
        self._draw_slots(header, shelves, ranks)
        if self.view == "stock" and stock is not None:
            self._draw_queues(docks, stock.get("workstations", {}))
        self._draw_agents(step, _trails(history), debug)
        self._draw_selection(step)
        screen.set_clip(None)
        self._draw_top(step, debug)
        if self._panel_w:
            self._draw_sidebar(step, stock, header, ranks, debug)
        self._draw_controls(slider, status, live, timeline)
        if self.help:
            self._draw_help()
        else:
            self._draw_hover(step)

    def save(self, path: Path) -> None:
        """Save the frame on the surface as an image."""
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
                        box.left, round(top), box.width, max(1, round(bottom) - round(top))
                    )
                    colour = rank_colour(ranks.get(sku, 0), len(ranks))
                    pygame.draw.rect(screen, colour, band.clip(box))
                    bottom = top
            pygame.draw.rect(screen, rim if self.view == "stock" else _MUTED, box, 1)
            if entry and self.view == "stock":
                self._badges(p, r, entry)

    def _draw_queues(self, docks, workstations) -> None:
        """Under each dock the units waiting there, under each workstation its open picks."""
        for v, units in docks.items():
            if int(v) in self._pixels and units:
                total = sum(units.values())
                self._pill(
                    self._pixels[int(v)], f"{total} unit" + ("" if total == 1 else "s"), _PURPLE
                )
        for v, count in workstations.items():
            if int(v) in self._pixels and count:
                self._pill(
                    self._pixels[int(v)],
                    f"{count} pick" + ("" if count == 1 else "s"),
                    _hex("#c9761a"),
                )

    def _pill(self, p, text: str, colour) -> None:
        pygame = self._pygame
        label = self._small.render(text, True, _WHITE)
        rect = pygame.Rect(0, 0, label.get_width() + 10, 15)
        rect.midtop = (p[0], p[1] + self._radius + 4)
        pygame.draw.rect(self._screen, colour, rect, border_radius=7)
        self._screen.blit(label, label.get_rect(center=rect.center))

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
                colour = _RED if debug["still_for"].get(a, 0) else _MUTED
            pygame.draw.circle(screen, colour, at, r)
            if self.view == "congestion" and (agent.get("overridden") or a in debug["blocked"]):
                pygame.draw.circle(screen, (0, 0, 0), at, r + 2, 2)
            label = self._label.render(a, True, _WHITE)
            screen.blit(label, label.get_rect(center=at))
            if self.view == "agents" and agent.get("phase") == "to_delivery" and agent.get("sku"):
                quantity = agent.get("quantity")
                carried = agent["sku"] + ("" if quantity in (None, 1) else f" x{quantity}")
                screen.blit(
                    self._small.render(carried, True, colour), (at[0] + r + 2, at[1] - r - 8)
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

    # ------------------------------------------------------------- top strip
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

    def _draw_top(self, step, debug) -> None:
        """The view tabs, the help button and the key numbers, above the map."""
        pygame = self._pygame
        screen = self._screen
        width = self._size[0] - self._panel_w
        pygame.draw.rect(screen, _PAPER, (0, 0, width, _TOP))
        pygame.draw.line(screen, _RULE, (0, _TOP - 1), (width, _TOP - 1))
        labels = {name: f"{i + 1}  {label}" for i, (name, label) in enumerate(VIEWS.items())}
        x = 10
        if self._title:
            title = self._title_font.render(self._title, True, _hex(theme.PALETTE["text"]))
            screen.blit(title, (x, 18))
            x += title.get_width() + 14
        x = self._tabs(labels, self.view, x, 12, 330, self._set_view) and x + 330
        accent = _hex(theme.PALETTE["one_way_edge"])
        help_button = pygame.Rect(x + 6, 12, 28, 28)
        pygame.draw.rect(screen, accent if self.help else _WHITE, help_button, border_radius=14)
        pygame.draw.rect(screen, accent, help_button, 1, border_radius=14)
        mark = self._font.render("?", True, _WHITE if self.help else accent)
        screen.blit(mark, mark.get_rect(center=help_button.center))
        self._hit(help_button, self._toggle_help)
        t = step.get("t")
        tiles = [("step", "-" if t is None else t)]
        if "completed" in step:
            tiles += [
                ("done per step", f"{step['completed'] / max(t or 1, 1):.3f}"),
                ("backlog", step["backlog"]),
            ]
        tiles.append(("still", len(debug["still_with_task"])))
        if "orders_done" in step:
            tiles.append(("orders done", step["orders_done"]))
        right = width - 10
        for label, value in reversed(tiles):
            shown = self._value.render(str(value), True, _hex(theme.PALETTE["text"]))
            name = self._heading.render(label.upper(), True, _MUTED)
            w = max(shown.get_width(), name.get_width()) + 16
            if right - w < help_button.right + 10:
                break
            tile = pygame.Rect(right - w, 6, w, 40)
            pygame.draw.rect(screen, _WHITE, tile, border_radius=6)
            pygame.draw.rect(screen, _RULE, tile, 1, border_radius=6)
            screen.blit(name, (tile.left + 8, tile.top + 6))
            screen.blit(shown, (tile.left + 8, tile.top + 20))
            right = tile.left - 6

    def _set_view(self, name: str) -> None:
        self.view = name

    def _set_panel(self, name: str) -> None:
        self.panel = name

    def _toggle_help(self) -> None:
        self.help = not self.help

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
        card = pygame.Rect(x0 + 10, y + 10, _PANEL - 20, self._size[1] - y - 20)
        pygame.draw.rect(screen, _WHITE, card, border_radius=8)
        pygame.draw.rect(screen, _RULE, card, 1, border_radius=8)
        self._draw_rows(self._panel_rows(step, stock, header, ranks, debug), card)

    def _draw_rows(self, rows: list, card) -> None:
        """Draw a panel's rows inside its card, every row from the same left edge.

        A row is one of
        ("title", text): the panel's name, large, with a rule under it,
        ("heading", text): the name of a group of rows below it,
        ("stat", label, value, colour, action): a label on the left and its value on the
        right, the value in `colour` when one is given,
        ("sku", rank, name, amount, action): a SKU in a box, its popularity rank first in a
        square of the rank's colour, then its name, then the amount in a box of its own,
        ("bar", label, share, value, colour): a label and its value, and under them a bar
        filled to `share`,
        ("box", badge, badge_colour, round, label, detail, value, action): an item in a box,
        a badge on the left (a circle when `round`, else a square) with its short text, the
        label, a detail in grey after it, and the value in a box of its own on the right,
        ("chips", label, options, current, choose): a label and a row of small buttons, the
        current one filled, `choose(option)` called on a click,
        ("note", text, colour): a line of plain text.
        `action`, when given, is what a click on the row does. The rows scroll with the
        mouse wheel when they do not fit the card."""
        pygame = self._pygame
        screen = self._screen
        text = _hex(theme.PALETTE["text"])
        left, right = card.left + 14, card.right - 14
        y = card.top + 12
        heights = {
            "title": 34, "heading": 28, "stat": 21, "note": 21,
            "bar": 32,
            "sku": 29,
            "box": 29,
            "chips": 30,
        }  # fmt: skip
        total = sum(heights[row[0]] for row in rows)
        room = card.height - 24
        scroll = min(self._scroll.get(self.panel, 0), max(0, total - room))
        self._scroll[self.panel] = scroll
        y -= scroll
        screen.set_clip(card.inflate(-2, -2))
        for row in rows:
            kind = row[0]
            height = heights[kind]
            if y + height <= card.top + 4:
                y += height  # scrolled out of view at the top
                continue
            if y > card.bottom - 6:
                break
            action = None
            if kind == "title":
                screen.blit(self._title_large.render(row[1], True, text), (left, y))
                pygame.draw.line(screen, _RULE, (left, y + 27), (right, y + 27))
            elif kind == "heading":
                screen.blit(self._heading.render(row[1].upper(), True, _MUTED), (left, y + 10))
            elif kind == "stat":
                _, label, value, colour, action = row
                screen.blit(self._font.render(_pretty(label), True, text), (left, y))
                # A zero is grey, so colour only shows where there is something to see.
                zero = str(value).split(" ")[0] in ("0", "0.000", "0%")
                shown = self._value.render(str(value), True, _MUTED if zero else colour or text)
                screen.blit(shown, (right - shown.get_width(), y))
            elif kind == "sku":
                _, rank, sku, amount, action = row
                self._draw_sku(pygame.Rect(left, y, right - left, 25), rank, sku, amount)
            elif kind == "box":
                _, badge, badge_colour, round_, label, detail, value, action = row
                self._draw_box(
                    pygame.Rect(left, y, right - left, 25), str(badge), badge_colour, round_,
                    label, detail, value,
                )  # fmt: skip
            elif kind == "chips":
                _, label, options, current, choose = row
                screen.blit(self._font.render(_pretty(label), True, text), (left, y + 5))
                x = right
                for option in reversed(options):
                    shown = self._small.render(
                        option, True, _WHITE if option == current else _SELECT
                    )
                    chip = pygame.Rect(0, 0, shown.get_width() + 14, 22)
                    chip.topright = (x, y + 1)
                    on = option == current
                    pygame.draw.rect(screen, _SELECT if on else _WHITE, chip, border_radius=11)
                    pygame.draw.rect(screen, _SELECT, chip, 1, border_radius=11)
                    screen.blit(shown, shown.get_rect(center=chip.center))
                    self._hit(chip, lambda option=option: choose(option))
                    x = chip.left - 4
            elif kind == "bar":
                _, label, share, value, colour = row
                screen.blit(self._font.render(_pretty(label), True, text), (left, y))
                shown = self._value.render(str(value), True, text)
                screen.blit(shown, (right - shown.get_width(), y))
                track = pygame.Rect(left, y + 19, right - left, 6)
                pygame.draw.rect(screen, _RULE, track, border_radius=3)
                filled = track.copy()
                filled.width = int(track.width * min(max(share, 0.0), 1.0))
                if filled.width:
                    pygame.draw.rect(screen, colour, filled, border_radius=3)
            else:
                screen.blit(self._font.render(_pretty(row[1]), True, row[2]), (left, y))
            if action is not None:
                visible = pygame.Rect(card.left, y - 2, card.width, height).clip(card)
                if visible.height:
                    self._hit(visible, action)
            y += height
        screen.set_clip(None)
        if total > room:
            more = "scroll for more" if scroll < total - room else "top: scroll up"
            hint = self._small.render(more, True, _MUTED)
            screen.blit(hint, (right - hint.get_width(), card.bottom - 16))

    def _draw_box(
        self, box, badge: str, badge_colour, round_: bool, label: str, detail: str, value: Any
    ) -> None:
        """One boxed row: a badge on the left, the label and a grey detail, the value."""
        pygame = self._pygame
        screen = self._screen
        text = _hex(theme.PALETTE["text"])
        pygame.draw.rect(screen, _PAPER, box, border_radius=6)
        pygame.draw.rect(screen, _RULE, box, 1, border_radius=6)
        if round_:
            centre = (box.left + box.height // 2, box.centery)
            pygame.draw.circle(screen, badge_colour, centre, box.height // 2 - 1)
            badge_right = box.left + box.height
        else:
            square = pygame.Rect(box.left, box.top, 34, box.height)
            pygame.draw.rect(screen, badge_colour, square, border_radius=6)
            centre, badge_right = square.center, square.right
        # Dark text on a pale badge, white on a dark one.
        ink = text if sum(badge_colour) > 560 else _WHITE
        shown_badge = self._value.render(badge, True, ink)
        screen.blit(shown_badge, shown_badge.get_rect(center=centre))
        name = self._font.render(label, True, text)
        x = badge_right + 10
        screen.blit(name, (x, box.centery - name.get_height() // 2))
        if detail:
            grey = self._small.render(detail, True, _MUTED)
            screen.blit(grey, (x + name.get_width() + 8, box.centery - grey.get_height() // 2))
        if value is None or value == "":
            return
        shown = self._value.render(str(value), True, text)
        pill = pygame.Rect(0, 0, max(shown.get_width() + 16, 40), box.height - 6)
        pill.midright = (box.right - 3, box.centery)
        pygame.draw.rect(screen, _WHITE, pill, border_radius=5)
        pygame.draw.rect(screen, _RULE, pill, 1, border_radius=5)
        screen.blit(shown, shown.get_rect(center=pill.center))

    def _draw_sku(self, box, rank: int, sku: str, amount: Any) -> None:
        """One boxed SKU row: the rank in its colour, the name, the amount."""
        self._draw_box(box, str(rank + 1), rank_colour(rank, self._sku_count), False, sku, "",
                       amount)  # fmt: skip

    def _sku_rows(self, units: Mapping[str, int], ranks: Mapping[str, int], action=None) -> list:
        """The boxed rows of a SKU list, most popular first."""
        held = sorted(units.items(), key=lambda kv: ranks.get(kv[0], len(ranks)))
        return [("sku", ranks.get(k, len(ranks)), k, u, action) for k, u in held]

    def _set_sort(self, title: str) -> None:
        self.sort = title

    def _set_idle(self, option: str) -> None:
        self.hide_idle = option == "hide"

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
            accent = _hex(theme.PALETTE["one_way_edge"])
            who = config.get("planner") or config.get("controller")
            if who:
                rows.append(("heading", "Run"))
                rows.append(_stat("routing", _pretty(who)))
                rows.append(_stat("storage rule", _pretty(config.get("storage_mode"))))
                rows.append(_stat("agents", config.get("num_agents")))
                rows.append(_stat("seed", config.get("seed")))
            if t is not None:
                rows.append(("heading", "Progress"))
                horizon = config.get("horizon")
                if horizon:
                    rows.append(("bar", "step", t / horizon, f"{t} of {horizon}", accent))
                else:
                    rows.append(_stat("step", t))
                if t < (config.get("warmup") or 0):
                    rows.append(("note", "in the warm-up, not scored", _MUTED))
            if "orders_done" in step:
                rows.append(("heading", "Orders"))
                done, released = step["orders_done"], step["orders_released"]
                rows.append(
                    ("bar", "orders complete", done / max(released, 1), f"{done} of {released}",
                     accent)
                )  # fmt: skip
            if "picks_done" in step:
                rows.append(("heading", "Picks, slot to workstation"))
                done, released = step["picks_done"], step["picks_released"]
                rows.append(
                    ("bar", "picks done", done / max(released, 1), f"{done} of {released}",
                     _hex(_PHASE_COLOURS["to_delivery"]))
                )  # fmt: skip
                rows.append(("heading", "Put-aways, dock to slot"))
                done, released = step["putaways_done"], step["putaways_released"]
                rows.append(
                    ("bar", "put-aways done", done / max(released, 1), f"{done} of {released}",
                     _PURPLE)
                )  # fmt: skip
            if "completed" in step:
                rows.append(("heading", "All tasks"))
                rows.append(_stat("tasks done", f"{step['completed']} of {step['released']}"))
                rows.append(
                    _stat("tasks done per step", f"{step['completed'] / max(t or 1, 1):.3f}")
                )
                rows.append(_stat("backlog, tasks not yet done", step["backlog"]))
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
            top = {k: totals.get(k, 0) for k in stock["ranking"][:10]}
            return rows + self._sku_rows(top, ranks)
        if name == "docks":
            docks = stock.get("docks", {})
            putaways = sum(e.get("putaways", 0) for e in shelves)
            rows.append(("heading", "Arrived stock"))
            waiting = sum(sum(u.values()) for u in docks.values())
            rows.append(_stat("units waiting at the docks", waiting))
            rows.append(_stat("put-aways not yet on a slot", putaways))
            workstations = stock.get("workstations", {})
            if workstations:
                rows.append(("heading", "Open picks per workstation"))
                colour = _hex(theme.ROLE_STYLES["delivery"].edge)
                for v, count in workstations.items():
                    rows.append(
                        ("box", "W", colour, False, f"workstation {v}", "", count,
                         lambda v=int(v): self._pick(("vertex", v)))
                    )  # fmt: skip
            for v, units in docks.items():
                rows.append(("heading", f"Dock {v}, units by SKU"))
                rows += self._sku_rows(units, ranks, lambda v=int(v): self._pick(("vertex", v)))
            return rows
        if name == "congestion":
            rows.append(("heading", "Now"))
            rows.append(_stat("agents still with a task", len(debug["still_with_task"]), _RED))
            rows.append(_stat("agents blocked", len(debug["blocked"]), _RED))
            rows.append(_stat("agents overridden", debug["overridden"], _RED))
            longest = max(debug["still_for"].values(), default=0)
            rows.append(_stat("longest still", _steps(longest), _RED))
            rows.append(_stat("whole fleet still", _steps(debug["fleet_still"]), _RED))
            if debug["blocked_by"]:
                rows.append(("heading", "Who waits on whom"))
                for a, blocker in sorted(debug["blocked_by"].items()):
                    rows.append(
                        ("box", a, _RED, True, f"waits on agent {blocker}",
                         f"at {step['agents'][a]['at']}", "", lambda a=a: self._pick(("agent", a)))
                    )  # fmt: skip
            rows.append(("heading", f"Still steps, last {_WINDOW} steps"))
            for v, c in sorted(debug["heat"].items(), key=lambda kv: -kv[1])[:6]:
                role = _ROLE_LABELS.get(self._role(v) or "", "aisle")
                rows.append(
                    ("box", "V", _RED, False, f"vertex {v}", role, _steps(c),
                     lambda v=v: self._pick(("vertex", v)))
                )  # fmt: skip
            return rows
        if name == "agents":
            idle = sum(1 for agent in step["agents"].values() if agent.get("phase") is None)
            rows.append(
                ("chips", f"idle agents ({idle})", ("show", "hide"),
                 "hide" if self.hide_idle else "show", self._set_idle)
            )  # fmt: skip
            rows.append(
                ("chips", "sort by", [title for title, _ in _SORTS], self.sort,
                 self._set_sort)
            )  # fmt: skip
            (key,) = [k for title, k in _SORTS if title == self.sort]
            agents = [
                (a, agent, debug["still_for"].get(a, 0))
                for a, agent in step["agents"].items()
                if not (self.hide_idle and agent.get("phase") is None)
            ]
            for a, agent, still in sorted(agents, key=lambda item: key(*item)):
                goal = agent.get("goal")
                where = f"at {agent['at']}" + ("" if goal is None else f", to {goal}")
                rows.append(
                    ("box", a, _agent_colour(agent), True, _doing(agent), where,
                     f"still {still}" if still else "", lambda a=a: self._pick(("agent", a)))
                )  # fmt: skip
            return rows
        recent = list(reversed(step.get("recent", [])))
        if not recent:
            return rows + [("note", "No task done yet.", _MUTED)]
        for t_done, kind, sku, quantity, source, target, agent_id in recent:
            putaway = kind == "putaway"
            colour = _PURPLE if putaway else _hex(_PHASE_COLOURS["to_delivery"])
            what = "put away" if putaway else "picked"
            rows.append(
                ("box", agent_id, colour, True, f"{sku} x{quantity} {what}",
                 f"{source} to {target}", f"step {t_done}",
                 lambda a=agent_id: self._pick(("agent", str(a))))
            )  # fmt: skip
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

            def shown(value):
                return "-" if value is None else value

            rows = [
                ("heading", f"Agent {key}"),
                _stat("doing", _doing(agent), _agent_colour(agent)),
                _stat("at vertex", agent["at"]),
                _stat("next vertex", shown(agent.get("intent"))),
                _stat("goal", shown(agent.get("goal"))),
                _stat("follow on the map", "on" if self.follow else "off", _SELECT,
                      self._toggle_follow),
                ("heading", "Task"),
                _stat("task", shown(agent.get("task"))),
                _stat("kind", shown(agent.get("kind"))),
            ]  # fmt: skip
            if agent.get("sku") is not None:
                rows.append(_stat("SKU", f"{agent['sku']} x{agent.get('quantity') or 1}"))
            rows.append(("heading", "Congestion"))
            rows.append(_stat("still for", _steps(debug["still_for"].get(key, 0))))
            rows.append(_stat("congestion", "-" if level is None else f"{level:.2f}"))
            if key in debug["blocked_by"]:
                blocker = debug["blocked_by"][key]
                rows.append(
                    _stat("blocked by", f"agent {blocker}", _RED,
                          lambda: self._pick(("agent", blocker)))
                )  # fmt: skip
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
            rows.append(_stat("put-aways coming", entry["putaways"], _PURPLE))
            if entry["units"]:
                rows.append(("heading", "Units by SKU"))
                rows += self._sku_rows(entry["units"], ranks)
            if entry.get("incoming"):
                rows.append(("heading", "On its way in"))
                rows += self._sku_rows(entry["incoming"], ranks)
        open_picks = (stock or {}).get("workstations", {}).get(str(v))
        if open_picks:
            rows.append(_stat("open picks", open_picks))
        waiting = (stock or {}).get("docks", {}).get(str(v), {})
        if waiting:
            rows.append(("heading", "Waiting here"))
            rows += self._sku_rows(waiting, ranks)
        agents = [
            (a, agent, "here" if agent["at"] == v else "walks here")
            for a, agent in step["agents"].items()
            if agent["at"] == v or agent.get("goal") == v
        ]
        if agents:
            rows.append(("heading", "Agents"))
            for a, agent, where in agents:
                rows.append(
                    _stat(f"agent {a}", where, _agent_colour(agent),
                          lambda a=a: self._pick(("agent", a)))
                )  # fmt: skip
        rows.append(_stat(f"still steps here, last {_WINDOW}", debug["heat"].get(v, 0), _RED))
        return rows

    def _toggle_follow(self) -> None:
        self.follow = not self.follow

    # ------------------------------------------------------------- help
    def _draw_help(self) -> None:
        """The keys and what the colours of the current view mean, over the map."""
        pygame = self._pygame
        screen = self._screen
        area = self._map_rect()
        box = pygame.Rect(0, 0, min(460, area.width - 40), min(420, area.height - 40))
        box.topright = (area.right - 20, area.top + 20)
        layer = pygame.Surface(box.size, pygame.SRCALPHA)
        layer.fill((255, 255, 255, 240))
        screen.blit(layer, box.topleft)
        pygame.draw.rect(screen, _RULE, box, 1, border_radius=8)
        text = _hex(theme.PALETTE["text"])
        x, y = box.left + 16, box.top + 14
        screen.blit(self._title_large.render("Help", True, text), (x, y))
        y += 34
        for key, what in _HELP:
            screen.blit(self._value.render(key, True, text), (x, y))
            screen.blit(self._font.render(what, True, text), (x + 110, y))
            y += 19
        y += 8
        screen.blit(self._heading.render(f"{VIEWS[self.view]} view".upper(), True, _MUTED), (x, y))
        y += 18

        def dot(cx, cy, colour, label):
            pygame.draw.circle(screen, colour, (cx + 6, cy + 7), 6)
            screen.blit(self._small.render(label, True, text), (cx + 16, cy + 2))

        if self.view == "agents":
            dot(x, y, _hex(_PHASE_COLOURS[None]), "idle")
            dot(x + 120, y, _hex(_PHASE_COLOURS["to_pickup"]), "to pickup")
            dot(x + 240, y, _hex(_PHASE_COLOURS["to_delivery"]), "to workstation")
            dot(x, y + 20, _hex(_PHASE_COLOURS["putaway"]), "put-away")
            screen.blit(
                self._small.render("dashed line: where the agent walks", True, text),
                (x + 120, y + 22),
            )
        elif self.view == "stock":
            screen.blit(
                self._small.render("slot fill by SKU rank, popular to rare", True, text), (x, y)
            )
            for i in range(160):
                colour = rank_colour(i, 160)
                pygame.draw.line(screen, colour, (x + i * 2, y + 16), (x + i * 2, y + 26), 2)
            dot(x, y + 34, _hex("#c0392b"), "picks waiting")
            dot(x + 120, y + 34, _hex("#e08a1e"), "agent coming")
            dot(x + 240, y + 34, _PURPLE, "put-aways")
        else:
            dot(x, y, _RED, "agent still with a task")
            dot(x + 200, y, _MUTED, "moving or idle")
            screen.blit(
                self._small.render(
                    f"red patch: still steps here, last {_WINDOW} steps", True, text
                ),
                (x, y + 22),
            )
            screen.blit(
                self._small.render("red line: blocked by that agent", True, text), (x, y + 40)
            )

    # ------------------------------------------------------------- controls
    def _draw_controls(self, slider, status: str, live: bool, timeline: Timeline | None) -> None:
        pygame = self._pygame
        screen = self._screen
        width = self._size[0] - self._panel_w
        y = self._size[1] - _BAR
        pygame.draw.rect(screen, _PAPER, (0, y, width, _BAR))
        pygame.draw.line(screen, _RULE, (0, y), (width, y))
        accent = _hex(theme.PALETTE["one_way_edge"])
        if slider is not None and timeline is not None:
            self._draw_timeline(pygame.Rect(12, y + 8, width - 24, 40), timeline, slider)
        x, row = 12, y + 60
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
            button = pygame.Rect(x, row, max(text.get_width() + 18, 34), 28)
            pygame.draw.rect(screen, accent, button, border_radius=6)
            screen.blit(text, text.get_rect(center=button.center))
            self._hit(button, lambda action=action: action)
            x = button.right + 6
        # The speed slider sits at the right end of the bar, the step slider between.
        speed_track = pygame.Rect(width - 170, row + 16, 150, 8)
        share = math.log(self.speed / _SLOWEST) / math.log(_FASTEST / _SLOWEST)
        self._slider(speed_track, share, "speed:")
        label = self._small.render(
            f"speed {self.speed:g} steps a second", True, _hex(theme.PALETTE["text"])
        )
        screen.blit(label, (speed_track.left, row))
        if slider is not None:
            index, last = slider
            track = pygame.Rect(x + 10, row + 16, speed_track.left - x - 40, 8)
            self._slider(track, index / max(last, 1), "seek:")
            x = track.left
        if self._note:
            status = f"{status}   {self._note}"
        screen.blit(self._small.render(status, True, _hex(theme.PALETTE["text"])), (x, row))

    def _draw_timeline(self, area, timeline: Timeline, slider: tuple[int, int]) -> None:
        """The backlog over the steps that can be stepped through, a red mark where the whole
        fleet stood still, an orange mark where an agent was blocked, and the shown step."""
        pygame = self._pygame
        screen = self._screen
        index, last = slider
        pygame.draw.rect(screen, _WHITE, area, border_radius=4)
        pygame.draw.rect(screen, _RULE, area, 1, border_radius=4)
        n = len(timeline.backlog)
        if n == 0:
            return
        top = max(max(timeline.backlog), 1)
        columns = max(1, area.width - 4)
        points = []
        for c in range(columns):
            lo = c * n // columns
            hi = max(lo + 1, (c + 1) * n // columns)
            x = area.left + 2 + c
            if any(timeline.stalled[lo:hi]):
                pygame.draw.line(screen, _RED, (x, area.top + 2), (x, area.top + 9))
            elif any(timeline.blocked[lo:hi]):
                pygame.draw.line(screen, _ORANGE, (x, area.top + 2), (x, area.top + 6))
            value = max(timeline.backlog[lo:hi])
            points.append((x, area.bottom - 3 - (area.height - 14) * value / top))
        if len(points) > 1:
            pygame.draw.lines(screen, _hex(theme.PALETTE["one_way_edge"]), False, points, 2)
        cursor = area.left + 2 + columns * index / max(last, 1)
        pygame.draw.line(screen, _SELECT, (cursor, area.top), (cursor, area.bottom), 2)
        label = self._small.render(f"backlog, peak {top}", True, _MUTED)
        screen.blit(label, (area.right - label.get_width() - 6, area.top + 3))

        def act() -> str:
            return "seek:" + str((self._mouse()[0] - area.left) / area.width)

        act.drags = True  # type: ignore[attr-defined]
        self._hit(area, act)

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
            return action + str((self._mouse()[0] - track.left) / track.width)

        act.drags = True  # type: ignore[attr-defined]
        self._hit(track.inflate(0, 24), act)

    # ---------------------------------------------------------------- hover
    def _draw_hover(self, step) -> None:
        pygame = self._pygame
        mx, my = self._mouse()
        if not self._map_rect().collidepoint((mx, my)) or self._press is not None:
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
        x = min(mx + 14, self._size[0] - self._panel_w - w - 4)
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


def replay(
    path: Path,
    *,
    window_size: tuple[int, int] = (1440, 900),
    fps: float = 8.0,
    compare: Path | None = None,
) -> None:
    """Open a trace in a window and step through it (controls in the module docstring).
    With `compare`, open a second trace beside it, both kept on the same step, without the
    sidebar."""
    import pygame

    traces = [TraceReplay(path)] + ([TraceReplay(compare)] if compare is not None else [])
    for data in traces:
        if not data.steps:
            raise ValueError(f"{data.path} has no steps.")
    pygame.init()
    pygame.display.set_caption("slap-mapd-coupling")
    window = pygame.display.set_mode(window_size)
    width = window_size[0] // len(traces)
    renderers = []
    for i, data in enumerate(traces):
        config = data.header.get("config", {})
        title = _pretty(config.get("storage_mode", "")) if compare is not None else ""
        part = window.subsurface((i * width, 0, width, window_size[1]))
        renderers.append(
            WarehouseRenderer(
                None, header=data.header, surface=part, sidebar=compare is None, title=title
            )
        )
    for renderer in renderers:
        renderer.paused, renderer.speed = True, fps
    first = renderers[0]
    clock = pygame.time.Clock()
    # The steps are matched by index, so two traces of one configuration stay on one t.
    last = min(len(data.steps) for data in traces) - 1
    index = 0
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_q
                ):
                    return
                if event.type == pygame.KEYDOWN:
                    actions = [renderer.handle(event) for renderer in renderers]
                    action = actions[0]
                else:
                    x = pygame.mouse.get_pos()[0]
                    owner = renderers[min(x // width, len(renderers) - 1)]
                    action = owner.handle(event)
                    for renderer in renderers:
                        if renderer is not owner and event.type == pygame.MOUSEBUTTONUP:
                            renderer.handle(event)
                if action == "shot":
                    first.screenshot(Path(path).stem)
                elif action is not None:
                    index = first._step_action(action, index, last)
                    for renderer in renderers[1:]:
                        renderer.paused, renderer.speed = first.paused, first.speed
            for renderer, data in zip(renderers, traces):
                step = data.steps[index]
                state = "paused" if renderer.paused else "playing"
                status = f"{data.path.name}   step {step['t']} of {data.steps[last]['t']}   {state}"
                renderer.draw_frame(
                    step, data.stock_at(step["t"]), data.header, data.history(index),
                    (index, last), status, timeline=data,
                )  # fmt: skip
            pygame.display.flip()
            if not first.paused:
                if index == last:
                    for renderer in renderers:
                        renderer.paused = True
                else:
                    index += 1
            clock.tick(30 if first.paused else first.speed)
    finally:
        pygame.quit()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay an episode trace in a window.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="a second trace beside it")
    parser.add_argument("--fps", type=float, default=8.0, help="steps per second when playing")
    parser.add_argument("--size", type=int, nargs=2, default=(1440, 900), metavar=("W", "H"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    replay(args.trace, window_size=tuple(args.size), fps=args.fps, compare=args.compare)


if __name__ == "__main__":
    main()
