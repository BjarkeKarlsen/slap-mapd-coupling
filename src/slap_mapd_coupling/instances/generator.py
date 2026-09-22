"""Parametric warehouse-graph generator (aisle count, aisle length,
cross-aisles, one-way fraction)."""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt

from slap_mapd_coupling.core.graph import Edge, Vertex, VertexId, VertexRole, WarehouseGraph
from slap_mapd_coupling.instances.validation import (
    InstanceGenerationReport,
    check_connectivity,
    check_well_formedness,
)


class GeneratorParams(BaseModel):
    """A Pydantic model, not loose kwargs, so a future sweep YAML can nest
    this directly under ExperimentConfig without a second parsing layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_aisles: PositiveInt
    aisle_length: PositiveInt  # storage/transit cells per aisle
    num_cross_aisles: PositiveInt
    one_way_fraction: float = Field(ge=0.0, le=1.0)  # fraction of segments realised as one-way
    default_edge_cost: PositiveFloat = 1.0
    wait_cost: NonNegativeFloat  # c_wait -- no default, a named study parameter
    num_storage_vertices: PositiveInt
    num_delivery_vertices: PositiveInt
    num_endpoints: PositiveInt
    seed: int


class GeneratedInstance(BaseModel):
    """A WarehouseGraph plus the layout position generate_instance built it
    with -- for callers that want to render the graph (viz.warehouse_plot,
    viz.pygame_renderer) without falling back to a generic graph layout.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph: WarehouseGraph
    positions: dict[VertexId, tuple[float, float]]


def generate_warehouse_graph(params: GeneratorParams) -> WarehouseGraph:
    """Build one WarehouseGraph from GeneratorParams.

    Deterministic given params.seed (no hidden global RNG state) -- this
    is required for bit-for-bit replay. Does not run
    instances/validation.py's checks itself.

    Layout: a main transit corridor (one vertex per aisle column), each
    column dropping a vertical stack of `aisle_length` cells below it
    (`num_storage_vertices` of which, chosen by a seeded random sample
    over every aisle cell, are marked as storage faces -- so which
    physical cells hold inventory varies per seed, unlike
    `num_endpoints`/`num_delivery_vertices` vertices hanging off the
    corridor at either end, whose row -- top/bottom -- is a fixed zoning
    convention, not seed-dependent), extra horizontal cross-aisle edges at
    a few evenly-spaced heights. Every non-lattice claim beyond storage
    placement rests on `one_way_fraction`: that fraction of otherwise
    bidirectional segments is instead realised as a single directed edge.

    See `generate_instance` for the same graph plus the layout position
    used to build it (e.g. for plotting) -- this function and that one
    share a private `_build` so the two can never drift out of sync.
    """
    graph, _positions = _build(params)
    return graph


def generate_instance(params: GeneratorParams) -> GeneratedInstance:
    """Like `generate_warehouse_graph`, but also returns the (row, col)
    -derived layout position computed for each vertex while building it --
    for callers that want to render the result (see `viz/`).
    """
    graph, positions = _build(params)
    return GeneratedInstance(graph=graph, positions=positions)


# Spacing for the position derived from each vertex's (row, col) key below;
# purely a rendering concern (core.graph.Vertex deliberately has no
# coordinates -- Section IV never defines one), kept next to the layout
# that produces it rather than in viz/, since only this function knows the
# (kind, row, col) scheme.
_DX = 1.5
_DY = 1.2


def _build(params: GeneratorParams) -> tuple[WarehouseGraph, dict[VertexId, tuple[float, float]]]:
    rng = random.Random(params.seed)

    next_id = 0
    vertices: dict[VertexId, Vertex] = {}
    pos_to_id: dict[tuple[str, int, int], VertexId] = {}
    positions: dict[VertexId, tuple[float, float]] = {}

    def add_vertex(kind: str, row: int, col: int, role: VertexRole) -> VertexId:
        nonlocal next_id
        vid = next_id
        next_id += 1
        vertices[vid] = Vertex(id=vid, role=role)
        pos_to_id[(kind, row, col)] = vid
        positions[vid] = (col * _DX, -row * _DY)
        return vid

    for col in range(params.num_aisles):
        add_vertex("corridor", 0, col, VertexRole(movable=True))

    for col in range(params.num_aisles):
        for row in range(1, params.aisle_length + 1):
            add_vertex("aisle", row, col, VertexRole(movable=True))

    cross_rows: set[int] = set()
    if params.num_cross_aisles > 0 and params.aisle_length > 1:
        step = max(1, params.aisle_length // (params.num_cross_aisles + 1))
        for k in range(1, params.num_cross_aisles + 1):
            cross_rows.add(min(k * step, params.aisle_length - 1))

    # Position each endpoint/delivery vertex near the aisle column it will
    # actually attach to (`i % num_aisles`, matching the edge-building loop
    # below), stacking one row further out per repeat when there are more
    # of them than aisles -- NOT at a position indexed purely by i, which
    # would draw a long, visually confusing edge across the whole layout
    # whenever num_endpoints/num_delivery_vertices > num_aisles wraps a
    # vertex's attachment back to a column far from its own index.
    endpoint_ids = [
        add_vertex(
            "endpoint",
            -1 - i // params.num_aisles,
            i % params.num_aisles,
            VertexRole(movable=True, endpoint=True),
        )
        for i in range(params.num_endpoints)
    ]
    delivery_ids = [
        add_vertex(
            "delivery",
            params.aisle_length + 1 + i // params.num_aisles,
            i % params.num_aisles,
            VertexRole(movable=True, delivery=True),
        )
        for i in range(params.num_delivery_vertices)
    ]

    # A seeded random subset, not "the first num_storage_vertices in
    # generation order" (which always clustered every instance's storage
    # vertices in the leftmost columns, identically regardless of seed --
    # issue #75): which physical cells hold inventory is exactly the kind
    # of per-instance variation a seed sweep needs, unlike endpoint/
    # delivery's row (top/bottom), which is a deliberate, fixed zoning
    # convention, not something to randomise.
    aisle_ids = [vid for (kind, _row, _col), vid in pos_to_id.items() if kind == "aisle"]
    storage_ids = rng.sample(aisle_ids, min(params.num_storage_vertices, len(aisle_ids)))
    for vid in storage_ids:
        vertices[vid] = Vertex(id=vid, role=VertexRole(movable=True, storage=True))

    # Bridge edges: each is the ONLY connection into a leaf vertex (a
    # delivery point, an endpoint, or an entire aisle stack). Randomising
    # these one-way would strand that leaf half the time (its single edge
    # ends up oriented the wrong way, making the vertex unreachable) --
    # that is a generator defect, not a legitimate one-way aisle, so
    # bridge edges always stay bidirectional. Only interior edges (within
    # an aisle, or along a corridor/cross-aisle row) have an alternate
    # route and are safe to randomise.
    bridge_edges: list[tuple[VertexId, VertexId]] = []
    interior_edges: list[tuple[VertexId, VertexId]] = []

    for col in range(params.num_aisles):
        bridge_edges.append((pos_to_id[("corridor", 0, col)], pos_to_id[("aisle", 1, col)]))

    for col in range(params.num_aisles):
        for row in range(1, params.aisle_length):
            interior_edges.append(
                (pos_to_id[("aisle", row, col)], pos_to_id[("aisle", row + 1, col)])
            )

    for col in range(params.num_aisles - 1):
        interior_edges.append(
            (pos_to_id[("corridor", 0, col)], pos_to_id[("corridor", 0, col + 1)])
        )

    for row in cross_rows:
        for col in range(params.num_aisles - 1):
            interior_edges.append(
                (pos_to_id[("aisle", row, col)], pos_to_id[("aisle", row, col + 1)])
            )

    # Iterate over the delivery/endpoint vertices themselves (not over
    # aisle columns) and cycle the column each one attaches to: if there
    # are more delivery/endpoint vertices than aisles, cycling over
    # columns instead would leave the excess ones with no edge at all --
    # a silently disconnected vertex, not a legitimate one-way discard.
    for i, target in enumerate(delivery_ids):
        col = i % params.num_aisles
        bridge_edges.append((pos_to_id[("aisle", params.aisle_length, col)], target))

    for i, target in enumerate(endpoint_ids):
        col = i % params.num_aisles
        bridge_edges.append((pos_to_id[("corridor", 0, col)], target))

    edges: list[Edge] = []
    for a, b in bridge_edges:
        edges.append(Edge(source=a, target=b, cost=params.default_edge_cost))
        edges.append(Edge(source=b, target=a, cost=params.default_edge_cost))
    for a, b in interior_edges:
        if rng.random() < params.one_way_fraction:
            src, dst = (a, b) if rng.random() < 0.5 else (b, a)
            edges.append(Edge(source=src, target=dst, cost=params.default_edge_cost, one_way=True))
        else:
            edges.append(Edge(source=a, target=b, cost=params.default_edge_cost))
            edges.append(Edge(source=b, target=a, cost=params.default_edge_cost))

    graph = WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=params.wait_cost)
    return graph, positions


def generate_and_validate(
    params: GeneratorParams,
    *,
    fleet_size: int,
    require_well_formed: bool = False,
) -> tuple[WarehouseGraph, InstanceGenerationReport]:
    """Generate one instance and immediately validate it.

    Runs connectivity (always) and well-formedness (iff
    require_well_formed) against the result, returning the graph AND the
    report even when accepted=False, so a batch caller can inspect why a
    rejected seed failed and compute a failure rate.
    """
    graph = generate_warehouse_graph(params)
    connectivity = check_connectivity(graph)
    well_formedness = check_well_formedness(graph, fleet_size) if require_well_formed else None

    accepted = connectivity.ok and (well_formedness is None or well_formedness.ok)
    report = InstanceGenerationReport(
        seed=params.seed,
        accepted=accepted,
        connectivity=connectivity,
        well_formedness=well_formedness,
    )
    return graph, report
