"""Warehouse graph G=(V,E): vertex roles (V_mov/V_str/V_del/V_ep) and shortest-path distances."""

from __future__ import annotations
from collections import deque
import heapq
from typing import Any, Callable, Literal, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PrivateAttr,
    model_validator,
)

VertexId = int


class VertexRole(BaseModel):
    """Non-exclusive role membership for one vertex.

    Roles overlap by definition: V_str, V_del, V_rcv and V_ep are all
    subsets of V_mov and need not be pairwise disjoint (sec:pf:env). This
    is deliberately not an enum.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    movable: bool = True
    storage: bool = False
    delivery: bool = False
    endpoint: bool = False
    receiving: bool = False

    @model_validator(mode="after")
    def _roles_require_movable(self) -> "VertexRole":
        if not self.movable and (
            self.storage or self.delivery or self.endpoint or self.receiving
        ):
            raise ValueError(
                "storage/delivery/receiving/endpoint roles require movable=True "
                "(V_str, V_del, V_rcv, V_ep are subsets of V_mov, sec:pf:env)."
            )
        return self

    @property
    def is_task_endpoint(self) -> bool:
        return self.storage or self.delivery or self.endpoint or self.receiving


class Vertex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: VertexId
    role: VertexRole = Field(default_factory=VertexRole)
    # [A77]: the section of SKUs a workstation serves, named by the smallest id among the
    # section's workstations. None for every other vertex, and for a section of one workstation.
    section: VertexId | None = None


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: VertexId
    target: VertexId
    cost: PositiveFloat  # c(e) > 0, eq:onestepcost
    one_way: bool = False  # generator-set: True iff no reverse edge was also emitted

    @model_validator(mode="after")
    def _no_self_loop(self) -> "Edge":
        if self.source == self.target:
            raise ValueError(
                f"E must not contain self-loops (sec:pf:env): ({self.source}, {self.target})."
            )
        return self


class Action(BaseModel):
    """One element of U(v)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["wait", "move"]
    target: VertexId | None = None  # None iff kind == "wait"

    @model_validator(mode="after")
    def _target_matches_kind(self) -> "Action":
        if self.kind == "wait" and self.target is not None:
            raise ValueError("wait actions must not carry a target vertex.")
        if self.kind == "move" and self.target is None:
            raise ValueError("move actions must carry a target vertex.")
        return self


class WarehouseGraph(BaseModel):
    """G = (V, E).

    Immutable once built. All-pairs shortest-path distances over G[V_mov]
    are precomputed once at construction time, since d_G is queried every
    timestep by observation-building and storage-rule code and must not
    pay shortest-path cost then.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vertices: dict[VertexId, Vertex]
    edges: tuple[Edge, ...]
    wait_cost: NonNegativeFloat  # c_wait >= 0, a scalar study parameter (not per-vertex)

    _out: dict[VertexId, tuple[Edge, ...]] = PrivateAttr(default_factory=dict)
    # out_neighbours(v), the targets of _out[v], built once.
    _targets: dict[VertexId, tuple[VertexId, ...]] = PrivateAttr(default_factory=dict)
    _legal_actions: dict[VertexId, tuple[Action, ...]] = PrivateAttr(default_factory=dict)
    # within(v, depth), filled on first use. The graph never changes, so neither does the set.
    _within: dict[tuple[VertexId, float], tuple[VertexId, ...]] = PrivateAttr(default_factory=dict)
    # shortest_path_edges(v, w), filled on first use, for the same reason.
    _paths: dict[
        tuple[VertexId, VertexId], tuple[tuple[VertexId, VertexId], ...]
    ] = PrivateAttr(default_factory=dict)
    # The row of each vertex in the distance table, and d_G itself as that table (see
    # distance_table). Both are filled when the graph is built.
    _row: dict[VertexId, int] = PrivateAttr(default_factory=dict)
    _table: Any = PrivateAttr(default=None)
    # window_rows(window), filled on first use.
    _window_rows: dict[
        tuple[VertexId, ...], tuple[np.ndarray, dict[VertexId, int]]
    ] = PrivateAttr(default_factory=dict)
    # cached(name, build): tables other modules derive from the graph alone.
    _derived: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _edges_reference_known_movable_vertices(self) -> "WarehouseGraph":
        for e in self.edges:
            if e.source not in self.vertices or e.target not in self.vertices:
                raise ValueError(f"Edge {(e.source, e.target)} references an unknown vertex.")
            if not self.vertices[e.source].role.movable or not self.vertices[e.target].role.movable:
                raise ValueError(
                    f"Edge {(e.source, e.target)} touches a non-movable vertex; "
                    "N+(v) is only ever defined over V_mov (eq:actions)."
                )
        return self

    @model_validator(mode="after")
    def _validate_topology(self) -> "WarehouseGraph":
        for key, vertex in self.vertices.items():
            if key != vertex.id:
                raise ValueError(f"vertices key {key} does not match Vertex.id {vertex.id}")
        pairs: set[tuple[VertexId, VertexId]] = set()
        for edge in self.edges:
            pair = (edge.source, edge.target)
            if pair in pairs:
                raise ValueError(f"Duplicate directed edge {pair}")
            pairs.add(pair)
            if edge.source not in self.vertices or edge.target not in self.vertices:
                raise ValueError(f"Edge {pair} references an unknown vertex")
            if (
                not self.vertices[edge.source].role.movable
                or not self.vertices[edge.target].role.movable
            ):
                raise ValueError(f"Edge {pair} touches a non-movable vertex")
        self._assert_endpoint_safe(pairs)
        return self

    def _assert_endpoint_safe(self, pairs: set[tuple[VertexId, VertexId]]) -> None:
        endpoints = {v for v, item in self.vertices.items() if item.role.is_task_endpoint}
        adjacency: dict[VertexId, list[VertexId]] = {v: [] for v in self.vertices}
        for source, target in pairs:
            adjacency[source].append(target)
        for source in endpoints:
            visited = {source}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for target in adjacency[current]:
                    if target in visited:
                        continue
                    visited.add(target)
                    if target not in endpoints:
                        queue.append(target)
            missing = sorted(endpoints - visited)
            if missing:
                raise ValueError(
                    f"Warehouse graph is not endpoint-safe: {source} cannot reach "
                    f"{missing} without another task endpoint"
                )

    def model_post_init(self, __context: object) -> None:
        out: dict[VertexId, list[Edge]] = {v: [] for v in self.vertices}
        for edge in self.edges:
            out[edge.source].append(edge)
        # Outgoing edges in increasing target number, so slot k is the k-th smallest neighbour.
        self._out = {v: tuple(sorted(items, key=lambda e: e.target)) for v, items in out.items()}
        self._targets = {v: tuple(e.target for e in items) for v, items in self._out.items()}
        self._legal_actions = {
            v: (Action(kind="wait"),) + tuple(Action(kind="move", target=e.target) for e in items)
            for v, items in self._out.items()
        }
        self._row = {v: i for i, v in enumerate(self.vertices)}
        self._table = self._all_pairs_shortest_paths()

    def _all_pairs_shortest_paths(self) -> np.ndarray:
        # Dijkstra from each source in V_mov, run by scipy in C. One table of |V| x |V| numbers
        # (about 40 MB at 128 agents) instead of a Python entry per pair, which took about 1 GB.
        # Edges only join movable vertices (checked above), so D_G moves only over V_mov. A
        # non-movable source keeps a row of NaN, as does a pair with no path.
        row = self._row
        n = len(row)
        cost: dict[tuple[int, int], float] = {}
        for edge in self.edges:
            if edge.source not in row or edge.target not in row:
                continue  # the validator rejects an edge to an unknown vertex, with its own error
            pair = (row[edge.source], row[edge.target])
            cost[pair] = min(edge.cost, cost.get(pair, float("inf")))
        sources = np.array(
            [row[v] for v, item in self.vertices.items() if item.role.movable], dtype=np.int64
        )
        table = np.full((n, n), np.nan)
        if len(sources):
            pairs = np.array(list(cost), dtype=np.int64).reshape(-1, 2)
            weights = csr_matrix(
                (np.array(list(cost.values()), dtype=np.float64), (pairs[:, 0], pairs[:, 1])),
                shape=(n, n),
            )
            found = dijkstra(weights, directed=True, indices=sources)
            found[np.isinf(found)] = np.nan
            table[sources] = found
        table.flags.writeable = False
        return table

    def __eq__(self, other: object) -> bool:
        """Equal when the vertices, edges and wait cost are. pydantic would also compare the
        private attributes, which are derived from these or filled on use, and the distance
        table is an array, which has no single truth value."""
        if not isinstance(other, WarehouseGraph):
            return NotImplemented
        return (self.vertices, self.edges, self.wait_cost) == (
            other.vertices,
            other.edges,
            other.wait_cost,
        )

    __hash__ = None  # type: ignore[assignment]  # a dict field, so never hashable (as before)

    def _kept(self) -> dict[str, Any]:
        """The private attributes, read past pydantic's attribute lookup, which cost about
        1.5 us a read. The methods below run every step, so they read through this."""
        private = self.__pydantic_private__
        assert private is not None
        return private

    def out_neighbours(self, v: VertexId) -> tuple[VertexId, ...]:
        """N+(v) in the action set, ordered by increasing vertex number."""
        targets: dict[VertexId, tuple[VertexId, ...]] = self._kept()["_targets"]
        return targets.get(v, ())

    def legal_actions(self, v: VertexId) -> tuple[Action, ...]:
        """U(v) in the action set: wait, then one move per N+(v) in increasing vertex number."""
        actions: dict[VertexId, tuple[Action, ...]] = self._kept()["_legal_actions"]
        return actions[v]

    def within(self, v: VertexId, depth: float) -> tuple[VertexId, ...]:
        """The movable vertices w with d_G(v, w) <= depth, in vertex order. Worked out once per
        (v, depth) and kept, since the environment asks for every robot's field of view on every
        timestep (environment/observation.py::local_subgraph)."""
        key = (v, depth)
        kept: dict[tuple[VertexId, float], tuple[VertexId, ...]] = self._kept()["_within"]
        found = kept.get(key)
        if found is None:
            found = tuple(
                w
                for w, vertex in self.vertices.items()
                if vertex.role.movable and self.distance(v, w) <= depth
            )
            kept[key] = found
        return found

    def shortest_path_edges(
        self, source: VertexId, target: VertexId
    ) -> tuple[tuple[VertexId, VertexId], ...]:
        """The edges of one shortest path from source to target, by Dijkstra with predecessor
        tracking, ties broken by neighbour order. Worked out once per (source, target) and kept,
        since the congestion-aware storage and put-away rules ask for the same paths on every
        update (storage/congestion.py, storage/arrival.py).

        Raises KeyError if target is unreachable from source, as distance does.
        """
        key = (source, target)
        paths: dict[tuple[VertexId, VertexId], tuple[tuple[VertexId, VertexId], ...]]
        paths = self._kept()["_paths"]
        found = paths.get(key)
        if found is not None:
            return found
        if source == target:
            self._paths[key] = ()
            return ()
        dist: dict[VertexId, float] = {source: 0.0}
        predecessor: dict[VertexId, VertexId] = {}
        frontier: list[tuple[float, VertexId]] = [(0.0, source)]
        while frontier:
            d, v = heapq.heappop(frontier)
            if d > dist.get(v, float("inf")):
                continue
            if v == target:
                break
            for edge in self._out.get(v, ()):
                nd = d + edge.cost
                if nd < dist.get(edge.target, float("inf")):
                    dist[edge.target] = nd
                    predecessor[edge.target] = v
                    heapq.heappush(frontier, (nd, edge.target))

        if target not in dist:
            raise KeyError(f"{target} is unreachable from {source}.")

        vertices = [target]
        while vertices[-1] != source:
            vertices.append(predecessor[vertices[-1]])
        vertices.reverse()
        found = tuple(zip(vertices, vertices[1:]))
        self._paths[key] = found
        return found

    def distance(self, v: VertexId, w: VertexId) -> float:
        """d_G(v, w) over G[V_mov].

        Raises KeyError (not float('inf')) if (v, w) is genuinely
        unreachable: finiteness is an instance precondition checked by
        instances/validation.py, not something this method should mask.
        """
        # The simulator, the storage rules and the assignment call this millions of times.
        private = self._kept()
        row = private["_row"]
        if v not in row or w not in row:
            raise KeyError((v, w))
        found: float = private["_table"].item(row[v], row[w])
        if found != found:  # NaN, no path
            raise KeyError((v, w))
        return found

    def vertex_rows(self) -> Mapping[VertexId, int]:
        """The row of each vertex in distance_table(), in the order of `vertices`."""
        row: dict[VertexId, int] = self._kept()["_row"]
        return row

    def distance_table(self) -> np.ndarray:
        """d_G as an array: entry [vertex_rows()[v], vertex_rows()[w]] is distance(v, w), the
        same number. NaN where distance raises KeyError. The observation reads a whole window's
        distance labels from it at once (models/local_subgraph_encoding.py)."""
        found: np.ndarray = self._kept()["_table"]
        return found

    def cached(self, name: str, build: Callable[[], Any]) -> Any:
        """build(), worked out on the first call with this name and kept, for a table another
        module derives from the graph alone. The graph never changes, so neither does the
        table."""
        derived = self._kept()["_derived"]
        if name not in derived:
            derived[name] = build()
        return derived[name]

    def window_rows(
        self, window: tuple[VertexId, ...]
    ) -> tuple[np.ndarray, Mapping[VertexId, int]]:
        """For a window (a tuple of vertices), the row of each of its vertices in
        distance_table() and the position of each vertex in the window. Worked out once per
        window, since the robots' windows are fixed per vertex and asked for on every step."""
        kept: dict[tuple[VertexId, ...], tuple[np.ndarray, dict[VertexId, int]]]
        kept = self._kept()["_window_rows"]
        found = kept.get(window)
        if found is None:
            row = self.vertex_rows()
            rows = np.fromiter((row[v] for v in window), dtype=np.int64, count=len(window))
            rows.flags.writeable = False
            found = (rows, {v: i for i, v in enumerate(window)})
            kept[window] = found
        return found
