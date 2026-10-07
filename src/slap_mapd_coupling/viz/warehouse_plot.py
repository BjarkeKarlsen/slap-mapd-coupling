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
import string

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
        # orientation=0 gives a diamond (pointy top/bottom/left/right),
        # matching plot_graph's own legend marker="D" -- orientation=pi/4
        # would instead rotate it into an axis-aligned square, which
        # looked like a mix of the two shapes rather than either one.
        return RegularPolygon(xy, numVertices=4, radius=size / 1.5, orientation=0, **kwargs)
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
        reverse_exists = any(
            e.source == edge.target and e.target == edge.source for e in graph.edges
        )
        if reverse_exists:
            ax.plot([x1, x2], [y1, y2], color=theme.PALETTE["edge"], linewidth=1.4, zorder=1)
            drawn.add(pair)
        else:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=theme.PALETTE["one_way_edge"], linewidth=1.8
                ),
                zorder=1,
            )
            drawn.add(pair)

    for vertex_id, vertex in graph.vertices.items():
        style, secondary_roles = theme.style_for_role(vertex.role)
        xy = positions[vertex_id]
        marker_size = 0.34
        patch = _patch_for_shape(
            style.shape,
            xy,
            marker_size,
            facecolor=style.fill,
            edgecolor=style.edge,
            linewidth=1.4,
            zorder=2,
        )
        ax.add_patch(patch)
        if secondary_roles:
            ring = Circle(
                xy,
                marker_size * 0.85,
                facecolor="none",
                edgecolor=theme.ROLE_STYLES[secondary_roles[0]].edge,
                linewidth=1.6,
                linestyle="--",
                zorder=3,
            )
            ax.add_patch(ring)

    handles = [
        Line2D([], [], marker=_MARKERS[s.shape], ls="", mfc=s.fill, mec=s.edge, ms=11, label=name)
        for name, s in theme.ROLE_STYLES.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        frameon=False,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.autoscale_view()
    return ax


def plot_agents(
    ax: Axes, fleet: FleetState, positions: dict[VertexId, tuple[float, float]]
) -> Axes:
    """Overlay: one marker + label per agent."""
    for agent_id, vertex_id in fleet.locations().items():
        x, y = positions[vertex_id]
        ax.scatter(
            [x], [y], s=180, c=theme.PALETTE["agent"], zorder=5, edgecolors="white", linewidths=1.2
        )
        ax.annotate(
            str(agent_id),
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            color=theme.PALETTE["agent"],
        )
    return ax


def plot_storage_heatmap(
    ax: Axes,
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    storage: StorageState,
    sku: SkuId,
) -> Axes:
    """Overlay: per-vertex unit count for one SKU, sized by count."""
    max_units = max(
        (storage.units(sku, v) for v in graph.vertices if graph.vertices[v].role.storage), default=0
    )
    if max_units == 0:
        return ax
    for vertex_id, vertex in graph.vertices.items():
        if not vertex.role.storage:
            continue
        units = storage.units(sku, vertex_id)
        if units == 0:
            continue
        x, y = positions[vertex_id]
        ax.scatter(
            [x],
            [y],
            s=200 * (units / max_units),
            c="none",
            edgecolors="#c0392b",
            linewidths=2.0,
            zorder=4,
        )
        ax.annotate(
            str(units),
            (x, y),
            textcoords="offset points",
            xytext=(10, -10),
            fontsize=8,
            color="#c0392b",
        )
    return ax


def assign_vertex_names(graph: WarehouseGraph) -> dict[VertexId, str]:
    """Short display name for every vertex playing a storage, delivery, or
    endpoint role (plain vertices get none), in vertex-id order: storage
    vertices get A, B, C, ... (matches 01b_storage_and_config.py's own
    A/B/C worked example); delivery and endpoint vertices get D1, D2, ...
    / E1, E2, ... -- short handles for future overlays that need to refer
    to a specific node (a path trace, per-node congestion).

    Storage is checked unconditionally, not via theme.ROLE_PRIORITY (which
    picks delivery over storage for the marker *shape* -- a rendering
    judgement call, per its own comment, not a statement about which
    vertices hold StorageState capacity): a vertex that's both storage and
    delivery still has real capacity/inventory to report via
    plot_storage_contents/plot_storage_capacity_list, so it must keep its
    letter regardless of which shape plot_graph draws for it. Such a
    vertex is not additionally given a D#/E# name; callers wanting only
    the non-storage names can filter this dict by
    `not graph.vertices[v].role.storage`.
    """
    names: dict[VertexId, str] = {}
    storage_letters = iter(string.ascii_uppercase)
    delivery_count = 0
    endpoint_count = 0
    for vertex_id in sorted(graph.vertices):
        role = graph.vertices[vertex_id].role
        if role.storage:
            names[vertex_id] = next(storage_letters)
        elif role.delivery:
            delivery_count += 1
            names[vertex_id] = f"D{delivery_count}"
        elif role.endpoint:
            endpoint_count += 1
            names[vertex_id] = f"E{endpoint_count}"
    return names


def plot_node_names(
    ax: Axes,
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    names: dict[VertexId, str],
) -> Axes:
    """Overlay: every named vertex's short handle (see assign_vertex_names)
    written on the node, coloured by its role like plot_graph's own
    markers -- the one place that draws a vertex's name, for storage,
    delivery, and endpoint vertices alike. plot_storage_contents only
    draws a storage vertex's "sku:count" contents underneath, not its
    name, so callers pass assign_vertex_names' full output here."""
    for vertex_id, name in names.items():
        role = graph.vertices[vertex_id].role
        style, _secondary = theme.style_for_role(role)
        x, y = positions[vertex_id]
        ax.annotate(
            name,
            (x, y),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=style.edge,
            zorder=6,
        )
    return ax


def plot_storage_contents(
    ax: Axes,
    positions: dict[VertexId, tuple[float, float]],
    storage: StorageState,
) -> Axes:
    """Overlay: each storage vertex's "sku:count" contents (units > 0 only)
    labelled underneath it. The vertex's own name (letter) is
    plot_node_names' job, not this function's -- storage.capacities is
    keyed exactly by V_str, so this needs no separate letters/names
    argument to know which vertices are storage."""
    for vertex_id in storage.capacities:
        held = [(sku_id, storage.units(sku_id, vertex_id)) for sku_id in storage.skus]
        held = [(sku_id, units) for sku_id, units in held if units > 0]
        if not held:
            continue
        x, y = positions[vertex_id]
        label = "\n".join(f"{sku_id}:{units}" for sku_id, units in held)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(12, -10 * len(held)),
            fontsize=8,
            linespacing=1.4,
            zorder=7,
        )
    return ax


def plot_storage_capacity_list(
    ax: Axes,
    storage: StorageState,
    letters: dict[VertexId, str],
) -> Axes:
    """Side list (outside the axes, to its right, boxed): one line per
    storage vertex, "letter -- used/max cap", followed by each SKU's
    unit_capacity (b_k, eq:feasiblestorage) as its own "sku=b_k" line under
    "Space per SKU" -- the b_k values are what let a reader verify the used
    totals themselves (e.g. coffee:4 at b_k=2.0 contributes 8)."""
    lines = ["Storage capacity", ""]
    for vertex_id, letter in letters.items():
        used = storage.used_capacity(vertex_id)
        cap = storage.capacities[vertex_id]
        lines.append(f"{letter}   {used:g}/{cap:g}")
    lines += ["", "Space per SKU", ""]
    for sku_id, sku in storage.skus.items():
        lines.append(f"{sku_id}={sku.unit_capacity:g}")
    ax.text(
        1.04,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#9aa5b1"),
    )
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
