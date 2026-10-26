"""A four-neighbour grid map as a WarehouseGraph (app:checks, the Ma et al. 2017 reproduction).

A grid is a special case of the graph: every open cell is a vertex with a two-way edge of cost 1
to each open neighbour, the conversion van Knippenberg et al. (2021, Figure 5) use for benchmark
grids. One character per cell, one line per row:

- `@` a blocked cell, no vertex
- `.` an open cell
- `e` a task endpoint, where a task is picked up or delivered. Its vertex is both a storage and
  a delivery vertex, since Ma et al. draw pickup and delivery from the same cells
- `r` a cell an agent may start on. Only the cells in `starts` become non-task endpoints (V_ep),
  the rest are open cells, since Ma et al. make the starting cells the only non-task endpoints

A vertex's id is its row times the width plus its column, so `cell_of` turns it back.
"""

from __future__ import annotations

from collections.abc import Collection

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexId, VertexRole, WarehouseGraph

BLOCKED = "@"
OPEN = "."
TASK_ENDPOINT = "e"
START = "r"
_CELLS = {BLOCKED, OPEN, TASK_ENDPOINT, START}


def parse_map(text: str) -> tuple[str, ...]:
    """The rows of a map, blank lines and surrounding spaces left out. Raises on rows of
    unequal length or an unknown character."""
    rows = tuple(line.strip() for line in text.strip().splitlines() if line.strip())
    if not rows:
        raise ValueError("the map has no rows.")
    width = len(rows[0])
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {r} has {len(row)} cells, row 0 has {width}.")
        unknown = set(row) - _CELLS
        if unknown:
            raise ValueError(f"row {r} has unknown cells {sorted(unknown)}.")
    return rows


def vertex_of(row: int, column: int, width: int) -> VertexId:
    return row * width + column


def cell_of(vertex: VertexId, width: int) -> tuple[int, int]:
    return divmod(vertex, width)


def cells(rows: tuple[str, ...], kind: str) -> list[VertexId]:
    """The vertices of every cell of one kind, in vertex order."""
    width = len(rows[0])
    return [
        vertex_of(r, c, width)
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch == kind
    ]


def grid_graph(
    rows: tuple[str, ...], starts: Collection[VertexId], wait_cost: float = 1.0
) -> WarehouseGraph:
    """The graph of a parsed map, with `starts`, a subset of the `r` cells, as the non-task
    endpoints."""
    width = len(rows[0])
    start_cells = set(cells(rows, START))
    if not set(starts) <= start_cells:
        raise ValueError(f"starts {sorted(set(starts) - start_cells)} are not `r` cells.")
    vertices: dict[VertexId, Vertex] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == BLOCKED:
                continue
            v = vertex_of(r, c, width)
            if ch == TASK_ENDPOINT:
                role = VertexRole(storage=True, delivery=True)
            elif v in starts:
                role = VertexRole(endpoint=True)
            else:
                role = VertexRole()
            vertices[v] = Vertex(id=v, role=role)
    edges = []
    for v in vertices:
        r, c = cell_of(v, width)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(rows) and 0 <= nc < width:
                w = vertex_of(nr, nc, width)
                if w in vertices:
                    edges.append(Edge(source=v, target=w, cost=1.0))
    return WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=wait_cost)
