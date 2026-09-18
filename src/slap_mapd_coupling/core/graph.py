"""Warehouse graph G=(V,E): vertex roles (V_mov/V_str/V_del/V_ep) and shortest-path distances."""

from __future__ import annotations

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

    def model_post_init(self, __context: object) -> None:
        out: dict[VertexId, list[Edge]] = {v: [] for v in self.vertices}
        for e in self.edges:
            out[e.source].append(e)
        self._out = {v: tuple(es) for v, es in out.items()}

        legal_actions: dict[VertexId, tuple[Action, ...]] = {}
        for v, es in self._out.items():
            moves = tuple(Action(kind="move", target=e.target) for e in es)
            legal_actions[v] = (Action(kind="wait"),) + moves
        self._legal_actions = legal_actions

        self._dist = self._all_pairs_shortest_paths()

    def _all_pairs_shortest_paths(self) -> dict[tuple[VertexId, VertexId], float]:
        """Dijkstra per source over G[V_mov] (c(e) > 0 rules out negative weights)."""
        movable = [v for v, vertex in self.vertices.items() if vertex.role.movable]
        dist: dict[tuple[VertexId, VertexId], float] = {}
        for source in movable:
            best: dict[VertexId, float] = {source: 0.0}
            frontier: list[tuple[float, VertexId]] = [(0.0, source)]
            while frontier:
                d, v = heapq.heappop(frontier)
                if d > best.get(v, float("inf")):
                    continue
                for e in self._out.get(v, ()):
                    nd = d + e.cost
                    if nd < best.get(e.target, float("inf")):
                        best[e.target] = nd
                        heapq.heappush(frontier, (nd, e.target))
            for target, d in best.items():
                dist[(source, target)] = d
        return dist

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
