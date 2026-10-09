"""Warehouse graph G=(V,E): vertex roles (V_mov/V_str/V_del/V_ep) and shortest-path distances."""

from __future__ import annotations
from collections import deque
import heapq
from typing import Literal

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

    Roles overlap by definition: V_str, V_del and V_ep are all subsets of
    V_mov and need not be pairwise disjoint (sec:pf:env). This is
    deliberately not an enum.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    movable: bool = True
    storage: bool = False
    delivery: bool = False
    endpoint: bool = False

    @model_validator(mode="after")
    def _roles_require_movable(self) -> "VertexRole":
        if not self.movable and (self.storage or self.delivery or self.endpoint):
            raise ValueError(
                "storage/delivery/endpoint roles require movable=True "
                "(V_str, V_del, V_ep are subsets of V_mov, sec:pf:env)."
            )
        return self

    @property
    def is_task_endpoint(self) -> bool:
        return self.storage or self.delivery or self.endpoint

class Vertex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: VertexId
    role: VertexRole = Field(default_factory=VertexRole)


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
    _legal_actions: dict[VertexId, tuple[Action, ...]] = PrivateAttr(default_factory=dict)
    _dist: dict[tuple[VertexId, VertexId], float] = PrivateAttr(default_factory=dict)

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
            if not self.vertices[edge.source].role.movable or not self.vertices[edge.target].role.movable:
                raise ValueError(f"Edge {pair} touches a non-movable vertex")
        self._assert_endpoint_safe(pairs)
        return self

    def _assert_endpoint_safe(self, pairs: set[tuple[VertexId, VertexId]]) -> None:
        endpoints = {v for v, item in self.vertices.items() if item.role.is_task_endpoint}
        adjacency = {v: [] for v in self.vertices}
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
        self._out = {v: tuple(sorted(items, key=lambda e: e.target)) for v, items in out.items()}
        self._legal_actions = {
            v: (Action(kind="wait"),) + tuple(Action(kind="move", target=e.target) for e in items)
            for v, items in self._out.items()
        }
        self._dist = self._all_pairs_shortest_paths()

    def _all_pairs_shortest_paths(self) -> dict[tuple[VertexId, VertexId], float]:
        # Dijkstra's algorithm from each source in V_mov, O(|V_mov| * (|E| + |V_mov| log |V_mov|)) time.
        # D_G moves only over V_mov, so we can ignore any edges touching non-movable vertices.
        distances: dict[tuple[VertexId, VertexId], float] = {}
        movable = [v for v, item in self.vertices.items() if item.role.movable]
        for source in movable:
            best = {source: 0.0}
            frontier = [(0.0, source)]
            while frontier:
                distance, current = heapq.heappop(frontier)
                if distance > best.get(current, float("inf")):
                    continue
                for edge in self._out.get(current, ()):
                    candidate = distance + edge.cost
                    if candidate < best.get(edge.target, float("inf")):
                        best[edge.target] = candidate
                        heapq.heappush(frontier, (candidate, edge.target))
            distances.update({(source, target): value for target, value in best.items()})
        return distances

    def out_neighbours(self, v: VertexId) -> tuple[VertexId, ...]:
        """N+(v): movable successors of v (eq:actions)."""
        return tuple(e.target for e in self._out.get(v, ()))

    def legal_actions(self, v: VertexId) -> tuple[Action, ...]:
        """U(v) = {wait} u {move(w): w in N+(v)} (eq:actions). O(1) precomputed lookup."""
        return self._legal_actions[v]

    def distance(self, v: VertexId, w: VertexId) -> float:
        """d_G(v, w) over G[V_mov].

        Raises KeyError (not float('inf')) if (v, w) is genuinely
        unreachable: finiteness is an instance precondition checked by
        instances/validation.py, not something this method should mask.
        """
        return self._dist[(v, w)]
