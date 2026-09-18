"""Static warehouse/occupancy plots (matplotlib).

Layered and composable, matching thesis-progress/scripts/images's
warehouse_visuals.py convention: each function creates its own Axes if
none is given, always returns it, and later layers draw on top of
earlier ones via the same `ax`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe, matches thesis-progress/scripts/images/figure.py

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import math

from matplotlib.patches import Circle, RegularPolygon, Rectangle

from slap_mapd_coupling.core.agents import FleetState
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, StorageState
from slap_mapd_coupling.viz import theme

_MARKERS = {"circle": "o", "square": "s", "diamond": "D"}


def _patch_for_shape(shape: str, xy: tuple[float, float], size: float, **kwargs):
    if shape == "square":
        s = size
        return Rectangle((xy[0] - s / 2, xy[1] - s / 2), s, s, **kwargs)
    if shape == "diamond":
        return RegularPolygon(xy, numVertices=4, radius=size / 1.5, orientation=math.pi / 4, **kwargs)
    return Circle(xy, size / 2, **kwargs)


def plot_graph(
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Base layer: edges (two-way plain, one-way arrowed and accent-coloured)
    and vertices styled by role (primary shape/colour, plus an outline ring
    for any secondary active role -- see viz.theme.style_for_role).
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 7))

    drawn: set[tuple[VertexId, VertexId]] = set()
    for edge in graph.edges:
        pair = (edge.source, edge.target)
        if pair in drawn or (pair[1], pair[0]) in drawn:
            continue
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        reverse_exists = any(e.source == edge.target and e.target == edge.source for e in graph.edges)
        if reverse_exists:
            ax.plot([x1, x2], [y1, y2], color=theme.PALETTE["edge"], linewidth=1.4, zorder=1)
            drawn.add(pair)
        else:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=theme.PALETTE["one_way_edge"], linewidth=1.8),
                zorder=1,
            )
            drawn.add(pair)

    for vertex_id, vertex in graph.vertices.items():
        style, secondary_roles = theme.style_for_role(vertex.role)
        xy = positions[vertex_id]
        marker_size = 0.34
        patch = _patch_for_shape(
            style.shape, xy, marker_size, facecolor=style.fill, edgecolor=style.edge, linewidth=1.4, zorder=2
        )
        ax.add_patch(patch)
        if secondary_roles:
            ring = Circle(xy, marker_size * 0.85, facecolor="none", edgecolor=theme.ROLE_STYLES[secondary_roles[0]].edge, linewidth=1.6, linestyle="--", zorder=3)
            ax.add_patch(ring)

    handles = [
        Line2D([], [], marker=_MARKERS[s.shape], ls="", mfc=s.fill, mec=s.edge, ms=11, label=name)
        for name, s in theme.ROLE_STYLES.items()
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles), frameon=False)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    return ax


def plot_agents(ax: Axes, fleet: FleetState, positions: dict[VertexId, tuple[float, float]]) -> Axes:
    """Overlay: one marker + label per agent."""
    for agent_id, vertex_id in fleet.locations().items():
        x, y = positions[vertex_id]
        ax.scatter([x], [y], s=180, c=theme.PALETTE["agent"], zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(str(agent_id), (x, y), textcoords="offset points", xytext=(0, 12), ha="center", color=theme.PALETTE["agent"])
    return ax


def plot_storage_heatmap(
    ax: Axes,
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    storage: StorageState,
    sku: SkuId,
) -> Axes:
    """Overlay: per-vertex unit count for one SKU, sized by count."""
    max_units = max((storage.units(sku, v) for v in graph.vertices if graph.vertices[v].role.storage), default=0)
    if max_units == 0:
        return ax
    for vertex_id, vertex in graph.vertices.items():
        if not vertex.role.storage:
            continue
        units = storage.units(sku, vertex_id)
        if units == 0:
            continue
        x, y = positions[vertex_id]
        ax.scatter([x], [y], s=200 * (units / max_units), c="none", edgecolors="#c0392b", linewidths=2.0, zorder=4)
        ax.annotate(str(units), (x, y), textcoords="offset points", xytext=(10, -10), fontsize=8, color="#c0392b")
    return ax


def plot_traffic(
    ax: Axes,
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    edge_traversals: dict[tuple[VertexId, VertexId], int],
) -> Axes:
    """Overlay: one arrow per traversed edge, width scaled by traversal count.

    Visualises the mu_T(e) input to eq:entropy / eq:concentration.
    """
    max_flow = max(edge_traversals.values(), default=0)
    if max_flow == 0:
        return ax
    for (src, dst), count in edge_traversals.items():
        if count == 0:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#f44336",
                alpha=0.7,
                linewidth=0.5 + 3.0 * (count / max_flow),
            ),
            zorder=4,
        )
    return ax
