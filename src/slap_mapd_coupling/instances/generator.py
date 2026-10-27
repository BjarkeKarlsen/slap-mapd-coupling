"""Topology-first generator for irregular, well-formed warehouse graphs.

Two layouts, chosen by ``GeneratorParams.layout``. "v2", the warehouse of every fleet
(``fleet_defaults``), is described at ``_build_v2``: every endpoint is a leaf off the walkable
backbone, so it is well-formed by construction. "v1", the layout before it, is described
below and kept so older configs and the graph v1 results still build the same instance.

Graph v1 generation has four separate stages:

1. Build a random warehouse topology containing only plain movable vertices.
2. Orient some transit segments as one-way while retaining strong connectivity.
3. Randomly assign storage, delivery, and parking roles to EXISTING vertices,
   and receiving roles to vertices of row 0, the inbound side.
4. Assign edge costs and construct the immutable ``WarehouseGraph``.

Storage, delivery, and parking vertices are therefore allowed in the middle of
an aisle or cross-aisle. They are not automatically added as leaf vertices.
A role assignment is accepted only when every pair of role vertices can reach
each other without using a third role vertex as an intermediate vertex.

The integer lattice gives every displayed node a unique position with equal
spacing; it does not determine whether a vertex is plain, storage, delivery,
or parking. Its column is also recorded as the vertex's aisle number
(``aisle_index``). The display positions are visualization metadata only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from slap_mapd_coupling.core.graph import (
    Edge,
    Vertex,
    VertexId,
    VertexRole,
    WarehouseGraph,
)
from slap_mapd_coupling.instances.validation import (
    InstanceGenerationReport,
    check_connectivity,
    check_well_formedness,
)

Position = tuple[float, float]
GridCell = tuple[int, int]
UndirectedSegment = tuple[VertexId, VertexId]
DirectedEdge = tuple[VertexId, VertexId]
Adjacency = dict[VertexId, list[VertexId]]


class GeneratorParams(BaseModel):
    """Bounds and randomization settings for one warehouse instance.

    Graph v1: ``max_num_aisles`` and ``max_aisle_length`` are upper bounds. The actual
    aisle count and the length of every aisle are sampled from ``seed``.
    ``num_transit_vertices`` is the total number of graph vertices because
    roles are assigned to existing topology vertices. Set it to ``None`` to
    sample the graph size as well.

    Graph v2 (``_build_v2``): ``num_aisles`` aisles with ``rack_rows`` rows in each rack
    block, and ``num_transit_vertices`` is the number of walkable vertices, the leaves coming
    on top. ``station_placement`` puts the workstations on the front wall or on rack faces drawn
    from the seed (D24). Without ``num_storage_vertices`` every rack face left is a slot. With
    it, that many slots are spread evenly over the rack faces left, and the rest are parking
    (D25).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout: Literal["v1", "v2"] = "v1"

    max_num_aisles: PositiveInt | None = None
    max_aisle_length: PositiveInt | None = None
    max_num_cross_aisles: NonNegativeInt | None = None
    num_transit_vertices: PositiveInt | None = None

    # Graph v2 only: the aisle count, the rows of each rack block, and the extra links between
    # each pair of neighbouring aisles in each rack block.
    num_aisles: PositiveInt | None = None
    rack_rows: PositiveInt | None = None
    links_per_aisle_pair: NonNegativeInt = 1
    require_irregular_shape: bool = True
    # Graph v2 only: the workstations spread along the front wall, or on rack faces drawn from
    # the seed anywhere in the warehouse (D24).
    station_placement: Literal["front_wall", "rack_faces"] = "front_wall"
    # Graph v2 only: the inner aisles of the lower rack block all reach the front lane, or hang
    # down from the main pair with lengths drawn from the seed like the upper block's, so some
    # stop short of the front lane. The walkable count stays 16 m either way.
    lower_block: Literal["full", "irregular"] = "full"
    # Graph v2 only: the docks at the front left corner, or spread evenly along the whole left
    # wall from the front to the back.
    dock_placement: Literal["corner", "left_wall"] = "corner"

    one_way_fraction: float = Field(ge=0.0, le=1.0)

    default_edge_cost: PositiveFloat = 1.0
    random_edge_costs: bool = False
    edge_cost_range: tuple[PositiveFloat, PositiveFloat] = (0.5, 2.0)
    wait_cost: NonNegativeFloat

    num_storage_vertices: PositiveInt | None = None
    num_delivery_vertices: PositiveInt
    num_endpoints: PositiveInt
    # V_rcv, drawn from row 0, the inbound side (sec:impl:instances). Zero
    # leaves every instance as before the receiving role existed.
    num_receiving_vertices: NonNegativeInt = 0

    # Set this above zero when an experiment explicitly requires some role
    # vertices to have at least two distinct neighbours. A value of zero means
    # all well-formed random placements are allowed, including topology leaves.
    min_internal_role_vertices: NonNegativeInt = 0

    # True also requires the plain vertices left after placing the roles to be strongly
    # connected, a sufficient condition for well-formedness that rejects many well-formed
    # placements. False places roles by the exact well-formedness check alone, as the fleet
    # warehouses of ``fleet_defaults`` do. True keeps the instance every older config gives.
    role_backbone: bool = True

    seed: int

    @property
    def num_role_vertices(self) -> int:
        """Number of vertices treated as endpoints by well-formedness (graph v1)."""
        return (
            (self.num_storage_vertices or 0)
            + self.num_delivery_vertices
            + self.num_endpoints
            + self.num_receiving_vertices
        )

    @property
    def max_num_vertices(self) -> int:
        assert self.max_num_aisles is not None and self.max_aisle_length is not None
        return self.max_num_aisles * (self.max_aisle_length + 1)

    @model_validator(mode="after")
    def _request_must_be_feasible(self) -> "GeneratorParams":
        """Reject impossible settings instead of silently clamping them."""
        minimum_cost, maximum_cost = self.edge_cost_range
        if minimum_cost > maximum_cost:
            raise ValueError("edge_cost_range must be ordered as (minimum, maximum).")

        if self.layout == "v2":
            return self._v2_request_must_be_feasible()

        for name in (
            "max_num_aisles",
            "max_aisle_length",
            "max_num_cross_aisles",
            "num_storage_vertices",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"layout='v1' requires {name}.")
        for name in ("num_aisles", "rack_rows"):
            if getattr(self, name) is not None:
                raise ValueError(f"{name} only applies to layout='v2'.")
        if (
            self.station_placement != "front_wall"
            or self.dock_placement != "corner"
            or self.lower_block != "full"
        ):
            raise ValueError(
                "station_placement, dock_placement and lower_block only apply to layout='v2'."
            )
        assert self.max_num_cross_aisles is not None and self.max_aisle_length is not None

        if self.max_num_cross_aisles > self.max_aisle_length:
            raise ValueError("max_num_cross_aisles cannot exceed max_aisle_length.")

        # This implementation deliberately leaves at least one plain vertex
        # from which endpoint-avoiding paths can be formed.
        minimum_vertices = self.num_role_vertices + 1
        if minimum_vertices > self.max_num_vertices:
            raise ValueError(
                f"The {self.num_role_vertices} role vertices require at "
                f"least {minimum_vertices} total vertices, but the configured "
                f"shape permits at most {self.max_num_vertices}."
            )

        if self.num_transit_vertices is not None:
            if self.num_transit_vertices < minimum_vertices:
                raise ValueError(
                    f"num_transit_vertices must be at least "
                    f"{minimum_vertices}: one existing vertex per requested "
                    "role plus at least one plain vertex."
                )
            if self.num_transit_vertices > self.max_num_vertices:
                raise ValueError(
                    f"num_transit_vertices={self.num_transit_vertices} "
                    f"exceeds the configured maximum of "
                    f"{self.max_num_vertices}."
                )

        if self.min_internal_role_vertices > self.num_role_vertices:
            raise ValueError(
                "min_internal_role_vertices cannot exceed the total number "
                "of storage, delivery, receiving and parking role vertices."
            )

        return self

    def _v2_request_must_be_feasible(self) -> "GeneratorParams":
        """Graph v2: the shape settings are there and the walkable count fits the shape."""
        for name in ("num_aisles", "rack_rows", "num_transit_vertices"):
            if getattr(self, name) is None:
                raise ValueError(f"layout='v2' requires {name}.")
        for name in ("max_num_aisles", "max_aisle_length", "max_num_cross_aisles"):
            if getattr(self, name) is not None:
                raise ValueError(f"{name} only applies to layout='v1'.")
        if self.one_way_fraction != 0.0:
            raise ValueError(
                "layout='v2' makes the main pair one-way and every other edge two-way, "
                "so one_way_fraction must be 0."
            )
        if self.min_internal_role_vertices or not self.role_backbone:
            raise ValueError(
                "min_internal_role_vertices and role_backbone only apply to layout='v1'."
            )
        assert self.num_aisles is not None and self.rack_rows is not None
        assert self.num_transit_vertices is not None
        if self.num_aisles < 3:
            raise ValueError("layout='v2' needs at least 3 aisles, two outer and one inner.")
        bounds = _v2_upper_inner_bounds(self.num_aisles, self.rack_rows, self.links_per_aisle_pair)
        upper_sum = _v2_upper_inner_sum(
            self.num_transit_vertices,
            self.num_aisles,
            self.rack_rows,
            self.links_per_aisle_pair,
        )
        if not bounds[0] <= upper_sum <= bounds[1]:
            raise ValueError(
                f"{self.num_transit_vertices} walkable vertices do not fit {self.num_aisles} "
                f"aisles of {self.rack_rows} rack rows per block: the inner aisles of the "
                f"upper block would hold {upper_sum} vertices, outside {bounds}."
            )
        if self.links_per_aisle_pair > self.rack_rows - 2:
            raise ValueError("links_per_aisle_pair cannot exceed rack_rows - 2.")
        return self


class GeneratedInstance(BaseModel):
    """A graph, its visualization positions, and its realized random shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout: Literal["v1", "v2"] = "v1"
    graph: WarehouseGraph
    positions: dict[VertexId, Position]
    # Aisle number of every vertex, left to right from 0. Cross-aisle vertices
    # carry the number of the aisle they sit on. Taken from the lattice column
    # when the vertex is created, so it does not depend on the display positions.
    # Graph v2: a lane or main pair vertex takes the aisle of its column's band, the aisle
    # column and the rack column either side of it, and a leaf the aisle of its vertex.
    aisle_index: dict[VertexId, int]
    # Graph v2: the rack rows of each aisle over both blocks, and the rows of the extra links.
    aisle_lengths: tuple[PositiveInt, ...]
    cross_aisle_rows: tuple[PositiveInt, ...]
    # Graph v2: the walkable vertices, the leaves not counted.
    num_transit_vertices: PositiveInt
    # Topologies sampled before this one was accepted. The discard rate over
    # a batch of seeds follows from it (sec:impl:instances).
    topology_attempts: PositiveInt = 1

    @property
    def num_aisles(self) -> int:
        return len(self.aisle_lengths)

    @model_validator(mode="after")
    def _metadata_must_match_graph(self) -> "GeneratedInstance":
        if set(self.positions) != set(self.graph.vertices):
            raise ValueError("Position keys must exactly match graph vertex IDs.")

        if set(self.aisle_index) != set(self.graph.vertices):
            raise ValueError("aisle_index keys must exactly match graph vertex IDs.")

        if len(set(self.positions.values())) != len(self.positions):
            raise ValueError("Every graph vertex must have a unique display position.")

        if self.layout == "v2":
            walkable = sum(
                not vertex.role.is_task_endpoint for vertex in self.graph.vertices.values()
            )
            if walkable != self.num_transit_vertices:
                raise ValueError(
                    f"{walkable} walkable vertices, but num_transit_vertices is "
                    f"{self.num_transit_vertices}."
                )
            return self

        expected_vertices = sum(aisle_length + 1 for aisle_length in self.aisle_lengths)
        if expected_vertices != self.num_transit_vertices:
            raise ValueError("aisle_lengths do not match num_transit_vertices.")

        if len(self.graph.vertices) != self.num_transit_vertices:
            raise ValueError(
                "Roles must be assigned to existing topology vertices; "
                "generation unexpectedly changed the vertex count."
            )

        return self


class InstanceGenerationError(ValueError):
    """Raised when no valid deterministic generation attempt succeeds."""


@dataclass(frozen=True)
class _RealizedShape:
    """Random dimensions selected within ``GeneratorParams`` bounds."""

    aisle_lengths: tuple[int, ...]
    cross_aisle_rows: tuple[int, ...]

    @property
    def num_aisles(self) -> int:
        return len(self.aisle_lengths)

    @property
    def num_vertices(self) -> int:
        return sum(length + 1 for length in self.aisle_lengths)


@dataclass
class _MutableTopology:
    """Plain topology plus lattice coordinates used only for visualization."""

    vertices: dict[VertexId, Vertex] = field(default_factory=dict)
    positions: dict[VertexId, Position] = field(default_factory=dict)
    aisle_index: dict[VertexId, int] = field(default_factory=dict)
    grid: dict[GridCell, VertexId] = field(default_factory=dict)
    next_id: int = 0

    def add_plain_vertex(self, *, cell: GridCell) -> VertexId:
        """Add one initially plain node at one unique display cell."""
        if cell in self.grid:
            raise ValueError(f"Display cell {cell} is already occupied.")

        vertex_id = self.next_id
        self.next_id += 1

        self.vertices[vertex_id] = Vertex(
            id=vertex_id,
            role=VertexRole(movable=True),
        )
        self.positions[vertex_id] = _to_plot_position(cell)
        self.aisle_index[vertex_id] = cell[1]
        self.grid[cell] = vertex_id
        return vertex_id


_GRID_SPACING = 1.5
_MAX_TOPOLOGY_ATTEMPTS = 100
_MAX_ROLE_ATTEMPTS = 10_000


def _to_plot_position(cell: GridCell) -> Position:
    """Convert one logical display cell into plotting coordinates."""
    row, column = cell
    return column * _GRID_SPACING, -row * _GRID_SPACING


# Vertices per agent (sec:impl:instances), the sizes van Knippenberg et al. (2021) use for 32
# and 64 agents.
VERTICES_PER_AGENT = 16
# One warehouse per fleet size, the same in every run (app:instances).
FLEET_INSTANCE_SEED = 42


def fleet_defaults(num_agents: int) -> dict[str, Any]:
    """The warehouse of a fleet of ``num_agents`` (sec:impl:instances, app:instances), graph v2.

    16 m walkable vertices, the leaves on top, so the floor space per robot is the same at
    every fleet size. m/2 workstations (delivery) and m/2 docks (receiving), two agents per
    station, and m parking leaves. The slots follow from the rack faces. The aisle count and
    the rack rows per block are those of ``_v2_shape``, about square, every edge cost 1 and
    the seed ``FLEET_INSTANCE_SEED``. A config with an ``instance`` section builds that
    warehouse instead.
    """
    num_aisles, rack_rows = _v2_shape(num_agents)
    num_stations = max(1, num_agents // 2)
    return {
        "layout": "v2",
        "num_aisles": num_aisles,
        "rack_rows": rack_rows,
        "links_per_aisle_pair": _V2_LINKS_PER_AISLE_PAIR,
        "num_transit_vertices": VERTICES_PER_AGENT * num_agents,
        "one_way_fraction": 0.0,
        "default_edge_cost": 1.0,
        "random_edge_costs": False,
        "num_delivery_vertices": num_stations,
        "num_receiving_vertices": num_stations,
        "num_endpoints": num_agents,
        "seed": FLEET_INSTANCE_SEED,
    }


def fleet_defaults_v1(num_agents: int) -> dict[str, Any]:
    """The graph v1 warehouse of a fleet of ``num_agents``, kept to rebuild the graph v1 results.

    |V| = 16 m, m parking vertices, m/2 storage vertices and m/2 delivery and as many
    receiving vertices, so two agents per dock at every fleet size, every edge two-way
    with cost 1. The shape bounds keep the
    warehouse about square: aisles of at most sqrt(|V|) steps, at most 1.1 sqrt(|V|)
    aisles, and at most 0.7 times the aisle length bound in cross-aisles, enough for every
    seed of app:instances to be well-formed. Roles are placed by the exact well-formedness
    check alone (``role_backbone=False``). The seed is ``FLEET_INSTANCE_SEED``. A config
    with an ``instance`` section builds that warehouse instead.
    """
    num_vertices = VERTICES_PER_AGENT * num_agents
    side = math.sqrt(num_vertices)
    aisle_length = math.ceil(side)
    num_docks = max(1, num_agents // 2)
    return {
        "max_num_aisles": math.ceil(1.1 * side),
        "max_aisle_length": aisle_length,
        "max_num_cross_aisles": math.ceil(0.7 * aisle_length),
        "num_transit_vertices": num_vertices,
        "one_way_fraction": 0.0,
        "default_edge_cost": 1.0,
        "random_edge_costs": False,
        "num_storage_vertices": max(1, num_agents // 2),
        "num_delivery_vertices": num_docks,
        "num_receiving_vertices": num_docks,
        "num_endpoints": num_agents,
        "role_backbone": False,
        "seed": FLEET_INSTANCE_SEED,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_warehouse_graph(params: GeneratorParams) -> WarehouseGraph:
    """Build only the immutable graph."""
    return generate_instance(params).graph


def generate_instance(params: GeneratorParams) -> GeneratedInstance:
    """Build one deterministic, random, well-formed warehouse instance."""
    return _build(params)


def generate_valid_instance(
    params: GeneratorParams,
    *,
    fleet_size: int,
) -> GeneratedInstance:
    """Training-facing API: return a usable instance or raise an error."""
    instance = generate_instance(params)
    connectivity = check_connectivity(instance.graph)
    well_formedness = check_well_formedness(
        instance.graph,
        fleet_size,
    )

    if not connectivity.ok or not well_formedness.ok:
        raise InstanceGenerationError(
            "Generated graph is incompatible with the requested scenario: "
            f"connectivity_ok={connectivity.ok}, "
            f"endpoint_count_ok={well_formedness.endpoint_count_ok}, "
            f"blocked_pairs={well_formedness.blocked_pairs[:10]}."
        )

    return instance


def generate_and_validate(
    params: GeneratorParams,
    *,
    fleet_size: int,
    require_well_formed: bool = True,
) -> tuple[WarehouseGraph, InstanceGenerationReport]:
    """Compatibility API for diagnostics and batch reports."""
    instance = generate_instance(params)
    graph = instance.graph
    connectivity = check_connectivity(graph)
    well_formedness = check_well_formedness(graph, fleet_size) if require_well_formed else None

    accepted = connectivity.ok and (well_formedness is None or well_formedness.ok)
    report = InstanceGenerationReport(
        seed=params.seed,
        accepted=accepted,
        connectivity=connectivity,
        well_formedness=well_formedness,
        topology_attempts=instance.topology_attempts,
    )
    return graph, report


# ---------------------------------------------------------------------------
# Main construction pipeline
# ---------------------------------------------------------------------------


def _build(params: GeneratorParams) -> GeneratedInstance:
    """Generate a well-formed instance. Graph v2 is well-formed by construction
    (``_build_v2``). Graph v1 discards topologies that admit none.

    Every instance must be well-formed, so a sampled topology that no role
    assignment can make well-formed (for example a near-tree with too few
    cycles) is discarded and the next one is sampled. Attempt ``k`` draws
    from streams derived from ``(seed, k)``, so the same seed always gives
    the same instance and distinct seeds never share a stream. Attempt 0
    uses the streams a single-attempt build would, so a seed that succeeds
    first time is unchanged. ``GeneratedInstance.topology_attempts`` records
    how many topologies were sampled, which gives the discard rate.
    """
    if params.layout == "v2":
        return _build_v2(params)
    last_reason = ""
    for attempt in range(_MAX_TOPOLOGY_ATTEMPTS):
        instance, last_reason = _try_build(params, attempt)
        if instance is not None:
            return instance

    raise InstanceGenerationError(
        f"Seed {params.seed}: none of {_MAX_TOPOLOGY_ATTEMPTS} sampled "
        f"topologies admitted a well-formed role assignment (last: "
        f"{last_reason}). The parameters are probably infeasible. Increase "
        "max_num_cross_aisles, max_num_aisles or max_aisle_length, or reduce "
        "the role count or min_internal_role_vertices."
    )


def _attempt_rng(params: GeneratorParams, attempt: int, stream: str) -> random.Random:
    """Random stream for one build attempt, independent per attempt and per use."""
    if attempt == 0:
        return random.Random(f"{params.seed}:{stream}")
    return random.Random(f"{params.seed}:{attempt}:{stream}")


def _try_build(
    params: GeneratorParams,
    attempt: int,
) -> tuple[GeneratedInstance | None, str]:
    """Sample one topology and place roles on it, or say why it was discarded."""
    try:
        shape = _sample_shape(
            params=params,
            rng=_attempt_rng(params, attempt, "shape"),
        )
    except InstanceGenerationError as error:
        # A size drawn at random can have no irregular shape. A fixed size
        # that has none is a parameter error and no resampling fixes it.
        if params.num_transit_vertices is not None:
            raise
        return None, str(error)
    description = f"aisle_lengths={shape.aisle_lengths}, cross_aisle_rows={shape.cross_aisle_rows}"

    topology = _create_plain_topology(shape)

    segments = _create_transit_segments(
        grid=topology.grid,
        shape=shape,
    )

    try:
        directed_edges = _orient_transit_segments(
            segments=segments,
            vertices=set(topology.vertices),
            one_way_fraction=params.one_way_fraction,
            rng=_attempt_rng(params, attempt, "directions"),
        )
    except InstanceGenerationError:
        return None, f"{description}, one_way_fraction not realisable"

    role_rng = _attempt_rng(params, attempt, "roles")
    row_zero = tuple(sorted(topology.grid[(0, column)] for column in range(shape.num_aisles)))
    sampled = _sample_well_formed_role_vertices(
        vertex_ids=set(topology.vertices),
        directed_edges=directed_edges,
        num_role_vertices=params.num_role_vertices,
        min_internal_vertices=params.min_internal_role_vertices,
        require_backbone=params.role_backbone,
        rng=role_rng,
        receiving_candidates=row_zero,
        num_receiving=params.num_receiving_vertices,
    )
    if sampled is None:
        return None, f"{description}, no well-formed role assignment"
    role_vertices, receiving_ids = sampled

    topology.vertices = _assign_roles(
        plain_vertices=topology.vertices,
        role_vertex_ids=role_vertices,
        receiving_ids=receiving_ids,
        params=params,
        rng=role_rng,
    )

    graph = _create_graph(
        vertices=topology.vertices,
        directed_edges=directed_edges,
        params=params,
        cost_rng=_attempt_rng(params, attempt, "costs"),
    )

    instance = GeneratedInstance(
        graph=graph,
        positions=topology.positions,
        aisle_index=topology.aisle_index,
        aisle_lengths=shape.aisle_lengths,
        cross_aisle_rows=shape.cross_aisle_rows,
        num_transit_vertices=shape.num_vertices,
        topology_attempts=attempt + 1,
    )
    return instance, ""


# ---------------------------------------------------------------------------
# Random topology shape
# ---------------------------------------------------------------------------


def _sample_shape(
    params: GeneratorParams,
    rng: random.Random,
) -> _RealizedShape:
    """Sample total size, individual aisle lengths, and cross-aisle rows."""
    # Graph v1 only, whose validator requires these.
    assert params.max_num_aisles is not None and params.max_aisle_length is not None
    assert params.max_num_cross_aisles is not None
    if params.num_transit_vertices is None:
        target_vertices = rng.randint(
            params.num_role_vertices + 1,
            params.max_num_vertices,
        )
    else:
        target_vertices = params.num_transit_vertices

    aisle_lengths = _sample_aisle_lengths(
        target_vertices=target_vertices,
        max_num_aisles=params.max_num_aisles,
        max_aisle_length=params.max_aisle_length,
        require_irregular_shape=params.require_irregular_shape,
        rng=rng,
    )
    cross_aisle_rows = _sample_cross_aisle_rows(
        aisle_lengths=aisle_lengths,
        max_num_cross_aisles=params.max_num_cross_aisles,
        rng=rng,
    )

    return _RealizedShape(
        aisle_lengths=aisle_lengths,
        cross_aisle_rows=cross_aisle_rows,
    )


def _sample_aisle_lengths(
    *,
    target_vertices: int,
    max_num_aisles: int,
    max_aisle_length: int,
    require_irregular_shape: bool,
    rng: random.Random,
) -> tuple[int, ...]:
    """Construct bounded aisle lengths with an exact vertex count.

    An aisle of length ``length`` contains ``length + 1`` vertices because
    row zero is included. Therefore, for ``num_aisles`` aisles, the lengths
    must satisfy::

        sum(aisle_lengths) = target_vertices - num_aisles

    Every length is in ``[1, max_aisle_length]``. When irregularity is
    required, this function selects only aisle counts for which unequal
    lengths are mathematically possible. It does not use a retry loop.
    """
    minimum_aisles = max(
        1,
        _ceil_div(target_vertices, max_aisle_length + 1),
    )
    maximum_aisles = min(
        max_num_aisles,
        target_vertices // 2,
    )

    if minimum_aisles > maximum_aisles:
        raise InstanceGenerationError(
            f"Cannot distribute {target_vertices} vertices over at most "
            f"{max_num_aisles} aisles with maximum length "
            f"{max_aisle_length}. Feasible aisle-count range is empty: "
            f"[{minimum_aisles}, {maximum_aisles}]."
        )

    feasible_counts = list(range(minimum_aisles, maximum_aisles + 1))

    if require_irregular_shape:
        irregular_counts = [
            num_aisles
            for num_aisles in feasible_counts
            if _unequal_lengths_are_possible(
                target_vertices=target_vertices,
                num_aisles=num_aisles,
                max_aisle_length=max_aisle_length,
            )
        ]

        if not irregular_counts:
            raise InstanceGenerationError(
                "Unequal aisle lengths are mathematically impossible for "
                f"target_vertices={target_vertices}, "
                f"max_num_aisles={max_num_aisles}, and "
                f"max_aisle_length={max_aisle_length}. Either set "
                "require_irregular_shape=False or change one of those "
                "bounds."
            )

        feasible_counts = irregular_counts

    num_aisles = rng.choice(feasible_counts)
    aisle_lengths = _random_bounded_lengths(
        target_sum=target_vertices - num_aisles,
        count=num_aisles,
        minimum=1,
        maximum=max_aisle_length,
        rng=rng,
    )

    if require_irregular_shape and len(set(aisle_lengths)) == 1:
        aisle_lengths = _make_lengths_unequal(
            aisle_lengths,
            maximum=max_aisle_length,
            rng=rng,
        )

    result = tuple(aisle_lengths)

    # These assertions describe generator invariants rather than user input.
    if sum(length + 1 for length in result) != target_vertices:
        raise AssertionError("Aisle sampler changed the requested total vertex count.")
    if any(length < 1 or length > max_aisle_length for length in result):
        raise AssertionError("Aisle sampler produced a length outside its configured bounds.")
    if require_irregular_shape and len(result) > 1 and len(set(result)) == 1:
        raise AssertionError("Aisle sampler failed to construct an irregular shape.")

    return result


def _unequal_lengths_are_possible(
    *,
    target_vertices: int,
    num_aisles: int,
    max_aisle_length: int,
) -> bool:
    """Return whether an unequal bounded composition exists.

    For two or more aisles, unequal lengths exist exactly when the required
    length sum is strictly between the all-minimum and all-maximum sums.
    """
    if num_aisles < 2:
        return False

    required_length_sum = target_vertices - num_aisles
    all_minimum_sum = num_aisles
    all_maximum_sum = num_aisles * max_aisle_length

    return all_minimum_sum < required_length_sum < all_maximum_sum


def _random_bounded_lengths(
    *,
    target_sum: int,
    count: int,
    minimum: int,
    maximum: int,
    rng: random.Random,
) -> list[int]:
    """Randomly construct an exact bounded integer composition.

    At each position, the selected value leaves a feasible remainder for all
    unfilled positions. Consequently construction cannot get stuck and needs
    no retry loop.
    """
    if not count * minimum <= target_sum <= count * maximum:
        raise InstanceGenerationError(
            f"Cannot split target sum {target_sum} into {count} values "
            f"bounded by [{minimum}, {maximum}]."
        )

    values: list[int] = []
    remaining_sum = target_sum

    for index in range(count):
        remaining_count = count - index - 1

        smallest_value = max(
            minimum,
            remaining_sum - remaining_count * maximum,
        )
        largest_value = min(
            maximum,
            remaining_sum - remaining_count * minimum,
        )

        value = rng.randint(smallest_value, largest_value)
        values.append(value)
        remaining_sum -= value

    # Avoid a positional bias caused by constructing from left to right.
    rng.shuffle(values)
    return values


def _make_lengths_unequal(
    lengths: list[int],
    *,
    maximum: int,
    rng: random.Random,
) -> list[int]:
    """Preserve the sum while changing an all-equal vector to unequal."""
    if len(lengths) < 2:
        raise InstanceGenerationError("At least two aisles are required for unequal aisle lengths.")

    donor_candidates = [index for index, length in enumerate(lengths) if length > 1]
    receiver_candidates = [index for index, length in enumerate(lengths) if length < maximum]

    valid_pairs = [
        (donor, receiver)
        for donor in donor_candidates
        for receiver in receiver_candidates
        if donor != receiver
    ]

    if not valid_pairs:
        raise InstanceGenerationError(
            "Could not break equal aisle lengths without violating bounds."
        )

    donor, receiver = rng.choice(valid_pairs)
    result = list(lengths)
    result[donor] -= 1
    result[receiver] += 1
    return result


def _sample_cross_aisle_rows(
    *,
    aisle_lengths: tuple[int, ...],
    max_num_cross_aisles: int,
    rng: random.Random,
) -> tuple[int, ...]:
    """Sample horizontal rows that connect at least one adjacent pair."""
    feasible_rows = [
        row
        for row in range(1, max(aisle_lengths) + 1)
        if any(
            left_length >= row and right_length >= row
            for left_length, right_length in zip(
                aisle_lengths,
                aisle_lengths[1:],
            )
        )
    ]

    maximum = min(max_num_cross_aisles, len(feasible_rows))
    if maximum == 0:
        return ()

    # Prefer at least one cycle-producing cross-aisle. Some sampled rows may
    # still contain only one horizontal segment; invalid sparse topologies are
    # naturally rejected later by constrained role placement.
    num_cross_aisles = rng.randint(1, maximum)
    return tuple(sorted(rng.sample(feasible_rows, num_cross_aisles)))


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


# ---------------------------------------------------------------------------
# Plain topology construction
# ---------------------------------------------------------------------------


def _create_plain_topology(shape: _RealizedShape) -> _MutableTopology:
    """Create all graph nodes as plain before any role is assigned."""
    topology = _MutableTopology()

    for column, aisle_length in enumerate(shape.aisle_lengths):
        for row in range(aisle_length + 1):
            topology.add_plain_vertex(cell=(row, column))

    return topology


def _create_transit_segments(
    grid: dict[GridCell, VertexId],
    shape: _RealizedShape,
) -> list[UndirectedSegment]:
    """Create vertical aisles, row-zero corridor, and random cross-aisles."""
    segments: list[UndirectedSegment] = []

    for column, aisle_length in enumerate(shape.aisle_lengths):
        for row in range(aisle_length):
            segments.append((grid[(row, column)], grid[(row + 1, column)]))

    for row in (0,) + shape.cross_aisle_rows:
        for column in range(shape.num_aisles - 1):
            left_cell = (row, column)
            right_cell = (row, column + 1)

            if left_cell in grid and right_cell in grid:
                segments.append((grid[left_cell], grid[right_cell]))

    return segments


# ---------------------------------------------------------------------------
# Edge orientation
# ---------------------------------------------------------------------------


def _orient_transit_segments(
    *,
    segments: list[UndirectedSegment],
    vertices: set[VertexId],
    one_way_fraction: float,
    rng: random.Random,
) -> set[DirectedEdge]:
    """Make selected segments one-way while preserving strong connectivity."""
    directed_edges = {
        edge for first, second in segments for edge in ((first, second), (second, first))
    }

    requested_one_way = round(len(segments) * one_way_fraction)
    if requested_one_way == 0:
        return directed_edges

    candidates = list(segments)
    rng.shuffle(candidates)
    accepted_one_way = 0

    for first, second in candidates:
        possible_removals = [(first, second), (second, first)]
        rng.shuffle(possible_removals)

        for edge_to_remove in possible_removals:
            directed_edges.remove(edge_to_remove)

            if _is_strongly_connected(vertices, directed_edges):
                accepted_one_way += 1
                break

            directed_edges.add(edge_to_remove)

        if accepted_one_way == requested_one_way:
            return directed_edges

    raise InstanceGenerationError(
        f"Cannot realize one_way_fraction={one_way_fraction}: only "
        f"{accepted_one_way}/{requested_one_way} one-way segments preserve "
        "strong connectivity for this topology."
    )


# ---------------------------------------------------------------------------
# Constrained role placement
# ---------------------------------------------------------------------------


def _sample_well_formed_role_vertices(
    *,
    vertex_ids: set[VertexId],
    directed_edges: set[DirectedEdge],
    num_role_vertices: int,
    min_internal_vertices: int,
    require_backbone: bool = True,
    rng: random.Random,
    receiving_candidates: tuple[VertexId, ...] = (),
    num_receiving: int = 0,
) -> tuple[tuple[VertexId, ...], frozenset[VertexId]] | None:
    """Randomly select existing nodes whose role assignment is well-formed.

    The receiving vertices are drawn from ``receiving_candidates`` and the
    other roles from the rest. They are returned apart, the other role
    vertices first. With ``num_receiving`` zero the draws are the same as
    before the receiving role existed.

    A candidate assignment is accepted when:

    1. enough selected vertices are non-leaves when requested;
    2. all remaining plain vertices form one strongly connected backbone;
    3. every selected role vertex can enter and leave that plain backbone; and
    4. the exact endpoint-avoiding reachability predicate succeeds.

    Conditions 2 and 3 are a constructive sufficient condition: every role
    vertex can travel to any other through plain nodes only. They apply only
    with ``require_backbone``, and without it condition 4, the well-formedness
    condition itself (app:wellformed), decides alone.
    """
    ordered_vertices = sorted(vertex_ids)
    outgoing, incoming = _adjacency_tables(vertex_ids, directed_edges)

    undirected_neighbours = {
        vertex: set(outgoing[vertex]) | set(incoming[vertex]) for vertex in vertex_ids
    }

    if num_receiving > len(receiving_candidates):
        return None
    for _attempt in range(_MAX_ROLE_ATTEMPTS):
        if num_receiving:
            receiving = set(rng.sample(receiving_candidates, num_receiving))
            rest = [v for v in ordered_vertices if v not in receiving]
            selected = receiving | set(rng.sample(rest, num_role_vertices - num_receiving))
        else:
            receiving = set()
            selected = set(rng.sample(ordered_vertices, num_role_vertices))

        internal_count = sum(len(undirected_neighbours[vertex]) >= 2 for vertex in selected)
        if internal_count < min_internal_vertices:
            continue

        if require_backbone and not _roles_share_plain_backbone(
            vertex_ids=vertex_ids,
            role_vertices=selected,
            outgoing=outgoing,
            incoming=incoming,
        ):
            continue

        if not _endpoints_are_well_formed(
            endpoint_ids=selected,
            outgoing=outgoing,
        ):
            continue

        result = sorted(selected - receiving) if receiving else list(selected)
        rng.shuffle(result)
        return tuple(result), frozenset(receiving)

    return None


def _roles_share_plain_backbone(
    *,
    vertex_ids: set[VertexId],
    role_vertices: set[VertexId],
    outgoing: Adjacency,
    incoming: Adjacency,
) -> bool:
    """Check a constructive sufficient condition for well-formedness.

    After all role vertices are removed, the remaining plain-node subgraph
    must be strongly connected. Each role vertex must also have at least one
    outgoing edge into and one incoming edge from that plain subgraph.
    """
    plain_vertices = vertex_ids - role_vertices
    if not plain_vertices:
        return False

    if not _is_induced_subgraph_strongly_connected(
        plain_vertices,
        outgoing,
        incoming,
    ):
        return False

    for endpoint in role_vertices:
        can_leave_for_plain = any(neighbour in plain_vertices for neighbour in outgoing[endpoint])
        can_arrive_from_plain = any(neighbour in plain_vertices for neighbour in incoming[endpoint])

        if not can_leave_for_plain or not can_arrive_from_plain:
            return False

    return True


def _endpoints_are_well_formed(
    *,
    endpoint_ids: set[VertexId],
    outgoing: Adjacency,
) -> bool:
    """Apply the exact directed endpoint-avoiding reachability predicate."""
    for source in endpoint_ids:
        reached_endpoints = _reachable_endpoints_without_crossing_others(
            source=source,
            endpoint_ids=endpoint_ids,
            outgoing=outgoing,
        )
        required_targets = endpoint_ids - {source}

        if not required_targets.issubset(reached_endpoints):
            return False

    return True


def _reachable_endpoints_without_crossing_others(
    *,
    source: VertexId,
    endpoint_ids: set[VertexId],
    outgoing: Adjacency,
) -> set[VertexId]:
    """Reach endpoints but never expand them as intermediate vertices."""
    visited = {source}
    reached_endpoints: set[VertexId] = set()
    queue: deque[VertexId] = deque([source])

    while queue:
        current = queue.popleft()

        for neighbour in outgoing[current]:
            if neighbour in visited:
                continue

            visited.add(neighbour)

            if neighbour in endpoint_ids:
                reached_endpoints.add(neighbour)
                continue

            queue.append(neighbour)

    return reached_endpoints


def _assign_roles(
    *,
    plain_vertices: dict[VertexId, Vertex],
    role_vertex_ids: tuple[VertexId, ...],
    params: GeneratorParams,
    rng: random.Random,
    receiving_ids: frozenset[VertexId] = frozenset(),
) -> dict[VertexId, Vertex]:
    """Assign role labels to selected existing topology vertices."""
    selected = list(role_vertex_ids)
    rng.shuffle(selected)

    assert params.num_storage_vertices is not None  # graph v1 only, which requires it
    storage_end = params.num_storage_vertices
    delivery_end = storage_end + params.num_delivery_vertices

    storage_ids = set(selected[:storage_end])
    delivery_ids = set(selected[storage_end:delivery_end])
    parking_ids = set(selected[delivery_end:])

    return {
        vertex_id: Vertex(
            id=vertex_id,
            role=VertexRole(
                movable=True,
                storage=vertex_id in storage_ids,
                delivery=vertex_id in delivery_ids,
                endpoint=vertex_id in parking_ids,
                receiving=vertex_id in receiving_ids,
            ),
        )
        for vertex_id in plain_vertices
    }


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _adjacency_tables(
    vertices: set[VertexId],
    directed_edges: set[DirectedEdge],
) -> tuple[Adjacency, Adjacency]:
    outgoing: Adjacency = {vertex: [] for vertex in vertices}
    incoming: Adjacency = {vertex: [] for vertex in vertices}

    for source, target in directed_edges:
        outgoing[source].append(target)
        incoming[target].append(source)

    return outgoing, incoming


def _is_strongly_connected(
    vertices: set[VertexId],
    directed_edges: set[DirectedEdge],
) -> bool:
    if not vertices:
        return True

    outgoing, incoming = _adjacency_tables(vertices, directed_edges)
    start = next(iter(vertices))

    return (
        _reachable_within(start, vertices, outgoing) == vertices
        and _reachable_within(start, vertices, incoming) == vertices
    )


def _is_induced_subgraph_strongly_connected(
    vertices: set[VertexId],
    outgoing: Adjacency,
    incoming: Adjacency,
) -> bool:
    if not vertices:
        return False

    start = next(iter(vertices))
    return (
        _reachable_within(start, vertices, outgoing) == vertices
        and _reachable_within(start, vertices, incoming) == vertices
    )


def _reachable_within(
    start: VertexId,
    allowed: set[VertexId],
    adjacency: Adjacency,
) -> set[VertexId]:
    visited = {start}
    queue: deque[VertexId] = deque([start])

    while queue:
        current = queue.popleft()

        for neighbour in adjacency[current]:
            if neighbour not in allowed or neighbour in visited:
                continue

            visited.add(neighbour)
            queue.append(neighbour)

    return visited


# ---------------------------------------------------------------------------
# Edge costs and final graph
# ---------------------------------------------------------------------------


def _create_graph(
    *,
    vertices: dict[VertexId, Vertex],
    directed_edges: set[DirectedEdge],
    params: GeneratorParams,
    cost_rng: random.Random,
) -> WarehouseGraph:
    """Assign physical-segment costs and build the immutable graph."""
    segment_costs: dict[tuple[VertexId, VertexId], float] = {}

    def cost_for(source: VertexId, target: VertexId) -> float:
        physical_segment = (
            min(source, target),
            max(source, target),
        )
        if physical_segment not in segment_costs:
            segment_costs[physical_segment] = _sample_edge_cost(
                params,
                cost_rng,
            )
        return segment_costs[physical_segment]

    edges = tuple(
        Edge(
            source=source,
            target=target,
            cost=cost_for(source, target),
            one_way=(target, source) not in directed_edges,
        )
        for source, target in sorted(directed_edges)
    )

    return WarehouseGraph(
        vertices=vertices,
        edges=edges,
        wait_cost=params.wait_cost,
    )


def _sample_edge_cost(
    params: GeneratorParams,
    rng: random.Random,
) -> float:
    if not params.random_edge_costs:
        return float(params.default_edge_cost)

    minimum_cost, maximum_cost = params.edge_cost_range
    return rng.uniform(
        float(minimum_cost),
        float(maximum_cost),
    )


# ---------------------------------------------------------------------------
# Graph v2
# ---------------------------------------------------------------------------

# Columns from one aisle to the next: the aisle and the two rack columns between it and its
# neighbour, back to back, so the faces of neighbouring aisles never share a display cell.
_V2_AISLE_PITCH = 3
# The extra links of ``fleet_defaults``, one per pair of neighbouring aisles in each block.
_V2_LINKS_PER_AISLE_PAIR = 1
# ``_v2_shape`` prefers shapes whose inner aisles fill this share of the upper block, so it
# has some dead ends and no block mostly empty.
_V2_UPPER_FILL = (0.5, 0.9)


def _v2_width(num_aisles: int) -> int:
    """Columns of the lanes and the main pair, from the first aisle to the last."""
    return _V2_AISLE_PITCH * (num_aisles - 1) + 1


def _v2_upper_inner_sum(
    num_walkable: int, num_aisles: int, rack_rows: int, links_per_aisle_pair: int
) -> int:
    """The vertices left for the inner aisles of the upper block once the four rows across
    (front lane, the main pair, back lane), the lower block, the two outer aisles of the upper
    block and the extra links are counted."""
    width = _v2_width(num_aisles)
    links = 2 * (num_aisles - 1) * links_per_aisle_pair
    return num_walkable - 4 * width - (num_aisles + 2) * rack_rows - 2 * links


def _v2_upper_inner_bounds(
    num_aisles: int, rack_rows: int, links_per_aisle_pair: int
) -> tuple[int, int]:
    """The fewest and most vertices the inner aisles of the upper block can hold. Each is long
    enough for its extra links, which skip the row next to the main pair, and at least one
    stops short of the back lane."""
    shortest = links_per_aisle_pair + 1
    num_inner = num_aisles - 2
    return num_inner * shortest, num_inner * rack_rows - 1


def _v2_shape(num_agents: int) -> tuple[int, int]:
    """The aisle count and rack rows per block of the fleet's warehouse. Of the shapes that
    hold 16 m walkable vertices, the most nearly square, the lanes as long as the warehouse
    is deep, with the upper block filled by a share in ``_V2_UPPER_FILL`` where one exists.
    Ties go to fewer aisles."""
    num_walkable = VERTICES_PER_AGENT * num_agents
    preferred: list[tuple[int, int, int]] = []
    feasible: list[tuple[int, int, int]] = []
    for num_aisles in range(3, num_walkable):
        width = _v2_width(num_aisles)
        if 4 * width > num_walkable:
            break
        for rack_rows in range(2, num_walkable):
            if _V2_LINKS_PER_AISLE_PAIR > rack_rows - 2:
                continue
            upper_sum = _v2_upper_inner_sum(
                num_walkable, num_aisles, rack_rows, _V2_LINKS_PER_AISLE_PAIR
            )
            low, high = _v2_upper_inner_bounds(num_aisles, rack_rows, _V2_LINKS_PER_AISLE_PAIR)
            if upper_sum < low:
                break
            if upper_sum > high:
                continue
            depth = 2 * rack_rows + 4
            candidate = (abs(width - depth), num_aisles, rack_rows)
            feasible.append(candidate)
            fill = upper_sum / ((num_aisles - 2) * rack_rows)
            if _V2_UPPER_FILL[0] <= fill <= _V2_UPPER_FILL[1]:
                preferred.append(candidate)
    shapes = preferred or feasible
    if not shapes:
        raise InstanceGenerationError(
            f"No graph v2 shape holds {num_walkable} walkable vertices for {num_agents} agents."
        )
    _, num_aisles, rack_rows = min(shapes)
    return num_aisles, rack_rows


def _v2_stations_and_slots(
    leaves: list[tuple[GridCell, GridCell, VertexRole]], params: GeneratorParams
) -> list[tuple[GridCell, GridCell, VertexRole]]:
    """The rack faces of graph v2 turned into workstations, slots and parking. Under
    ``station_placement`` "rack_faces" the workstations are rack faces drawn from the seed
    (D24). With ``num_storage_vertices`` that many of the faces left are slots, spread evenly
    over them in row order, and the rest become parking (D25)."""
    faces = sorted((cell, at) for cell, at, role in leaves if role.storage)
    others = [leaf for leaf in leaves if not leaf[2].storage]
    stations: list[tuple[GridCell, GridCell]] = []
    if params.station_placement == "rack_faces":
        num_stations = params.num_delivery_vertices
        if num_stations > len(faces):
            raise InstanceGenerationError(
                f"{num_stations} workstations do not fit {len(faces)} rack faces."
            )
        stations = random.Random(f"{params.seed}:v2:stations").sample(faces, num_stations)
        taken = set(stations)
        faces = [face for face in faces if face not in taken]
    num_slots = params.num_storage_vertices
    if num_slots is None:
        slots, parking = faces, []
    else:
        if num_slots > len(faces):
            raise InstanceGenerationError(
                f"{num_slots} slots do not fit the {len(faces)} rack faces left."
            )
        chosen = {int((index + 0.5) * len(faces) / num_slots) for index in range(num_slots)}
        slots = [face for index, face in enumerate(faces) if index in chosen]
        parking = [face for index, face in enumerate(faces) if index not in chosen]
    return (
        others
        + [(cell, at, VertexRole(delivery=True)) for cell, at in stations]
        + [(cell, at, VertexRole(storage=True)) for cell, at in slots]
        + [(cell, at, VertexRole(endpoint=True)) for cell, at in parking]
    )


def _build_v2(params: GeneratorParams) -> GeneratedInstance:
    """Graph v2: van Knippenberg et al.'s (2021) principles on a rack warehouse (D1 to D6).

    Columns are aisles, every third column, with the rack columns between them. Rows from the
    front: the front lane, the lower rack block, the main pair, the upper rack block and the
    back lane. The lanes and the main pair cross the whole width. Every aisle runs from the
    front lane up through the main pair. The two outer aisles reach the back lane, and the
    inner ones stop at lengths drawn from the seed, so some are dead ends. With ``lower_block``
    "irregular" the inner aisles of the lower block also hang from the main pair at lengths
    drawn from the seed, both blocks sharing the same walkable count, so some stop short of
    the front lane too. Extra links join
    neighbouring aisles at rows drawn from the seed, the role van Knippenberg's diagonal edges
    play, never on a row next to a lane or the main pair. Every edge is two-way except the
    main pair, two one-way rows in opposite directions, joined at both ends by the outer
    aisles so they form a loop.

    Every endpoint is a leaf, a vertex with one edge each way to a walkable vertex. Slots are
    the rack faces either side of every aisle vertex in the rack rows and, on the bottom row,
    the front lane's rack side between aisles. Workstations (delivery) sit on the front wall,
    spread evenly, or on rack faces drawn from the seed (D24). Docks (receiving) start at the front left corner, up the side wall through
    the lower block, then along the front wall, or spread evenly along the whole left wall.
    Parking starts at the back right corner, down
    the right wall, then along the back wall, then the left wall from the back. A wall cell of
    an outer aisle that no dock or parking takes is a slot. With ``num_storage_vertices`` only
    that many of the rack faces left after the workstations are slots, spread evenly in row
    order, and the others are parking too (D25). The main pair carries no leaf.

    A path cannot pass through a vertex with one edge, so no path between two endpoints
    crosses a third, and the graph is well-formed once the walkable part is strongly connected
    and there are m parking leaves (Ma et al. 2017).
    """
    num_aisles = params.num_aisles
    rack_rows = params.rack_rows
    num_walkable = params.num_transit_vertices
    assert num_aisles is not None and rack_rows is not None and num_walkable is not None
    links_per_pair = params.links_per_aisle_pair
    width = _v2_width(num_aisles)
    main_low, main_high = rack_rows + 1, rack_rows + 2
    back = 2 * rack_rows + 3

    def aisle_column(aisle: int) -> int:
        return _V2_AISLE_PITCH * aisle

    upper_sum = _v2_upper_inner_sum(num_walkable, num_aisles, rack_rows, links_per_pair)
    if params.lower_block == "full":
        upper_inner = _random_bounded_lengths(
            target_sum=upper_sum,
            count=num_aisles - 2,
            minimum=links_per_pair + 1,
            maximum=rack_rows,
            rng=random.Random(f"{params.seed}:v2:upper"),
        )
        lower_inner = [rack_rows] * (num_aisles - 2)
    else:
        # The inner aisles of both blocks share what the full lower block held, so the walkable
        # count is the same, and the lengths are drawn for both blocks at once.
        both = _random_bounded_lengths(
            target_sum=upper_sum + (num_aisles - 2) * rack_rows,
            count=2 * (num_aisles - 2),
            minimum=links_per_pair + 1,
            maximum=rack_rows,
            rng=random.Random(f"{params.seed}:v2:aisles"),
        )
        lower_inner, upper_inner = both[: num_aisles - 2], both[num_aisles - 2 :]
    upper_lengths = (rack_rows, *upper_inner, rack_rows)
    lower_lengths = (rack_rows, *lower_inner, rack_rows)

    # The walkable cells, and the undirected segments and one-way edges between them.
    walkable: list[GridCell] = []
    segments: list[tuple[GridCell, GridCell]] = []
    one_way: list[tuple[GridCell, GridCell]] = []
    for row in (0, main_low, main_high, back):
        walkable.extend((row, column) for column in range(width))
    for column in range(width - 1):
        segments.append(((0, column), (0, column + 1)))
        segments.append(((back, column), (back, column + 1)))
        one_way.append(((main_low, column), (main_low, column + 1)))
        one_way.append(((main_high, column + 1), (main_high, column)))
    aisle_top: list[int] = []
    aisle_bottom: list[int] = []
    for aisle, (lower, upper) in enumerate(zip(lower_lengths, upper_lengths)):
        column = aisle_column(aisle)
        top = main_high + upper
        bottom = main_low - lower  # 1 when the aisle reaches the front lane
        aisle_top.append(top)
        aisle_bottom.append(bottom)
        rows = [row for row in range(bottom, top + 1) if row not in (main_low, main_high)]
        walkable.extend((row, column) for row in rows)
        chain = (
            ([0] if lower == rack_rows else [])
            + list(range(bottom, top + 1))
            + ([back] if upper == rack_rows else [])
        )
        segments.extend(((a, column), (b, column)) for a, b in zip(chain, chain[1:]))

    link_rng = random.Random(f"{params.seed}:v2:links")
    link_rows: set[int] = set()
    link_cells: set[GridCell] = set()
    for aisle in range(num_aisles - 1):
        left, right = aisle_column(aisle), aisle_column(aisle + 1)
        upper_top = min(aisle_top[aisle], aisle_top[aisle + 1])
        lower_bottom = max(aisle_bottom[aisle], aisle_bottom[aisle + 1], 2)
        # Links skip the rows that touch the front lane or the main pair, which already join
        # the aisles there, and only join rows both aisles reach.
        lower_rows = link_rng.sample(range(lower_bottom, rack_rows), links_per_pair)
        upper_rows = link_rng.sample(range(main_high + 2, upper_top + 1), links_per_pair)
        for row in sorted(lower_rows + upper_rows):
            cells = [(row, column) for column in range(left, right + 1)]
            walkable.extend(cells[1:-1])
            link_cells.update(cells[1:-1])
            link_rows.add(row)
            segments.extend(zip(cells, cells[1:]))

    # The leaves: (cell, the walkable cell it hangs off, role).
    leaves: list[tuple[GridCell, GridCell, VertexRole]] = []
    slot = VertexRole(storage=True)
    for aisle in range(num_aisles - 1):
        for column in range(aisle_column(aisle) + 1, aisle_column(aisle + 1)):
            leaves.append(((1, column), (0, column), slot))
    for aisle in range(num_aisles):
        column = aisle_column(aisle)
        for row in range(max(2, aisle_bottom[aisle]), aisle_top[aisle] + 1):
            if row in (main_low, main_high):
                continue
            for side in (column - 1, column + 1):
                if 0 < side < width - 1 and (row, side) not in link_cells:
                    leaves.append(((row, side), (row, column), slot))

    rack_rows_of_wall = [r for r in range(1, back) if r not in (main_low, main_high)]
    left_wall = [((row, -1), (row, 0)) for row in [0] + rack_rows_of_wall + [back]]
    right_wall = [((row, width), (row, width - 1)) for row in [0] + rack_rows_of_wall + [back]]
    front_wall = [((-1, column), (0, column)) for column in range(width)]
    back_wall = [((back + 1, column), (back, column)) for column in range(width)]

    num_docks = params.num_receiving_vertices
    lower_left = [cell for cell in left_wall if cell[0][0] <= rack_rows]
    if params.dock_placement == "left_wall":
        if num_docks > len(left_wall):
            raise InstanceGenerationError(
                f"{num_docks} docks do not fit a left wall of {len(left_wall)} cells."
            )
        dock_cells = [
            left_wall[int((index + 0.5) * len(left_wall) / num_docks)]
            for index in range(num_docks)
        ]
    else:
        dock_cells = (lower_left + front_wall)[:num_docks]
    taken = {cell for cell, _ in dock_cells}
    free_front = [cell for cell in front_wall if cell[0] not in taken]
    num_stations = params.num_delivery_vertices
    on_wall = params.station_placement == "front_wall"
    if num_docks > len(lower_left) + len(front_wall) or (
        on_wall and num_stations > len(free_front)
    ):
        raise InstanceGenerationError(
            f"{num_docks} docks and {num_stations} workstations do not fit a front wall of "
            f"{width} cells and a lower side wall of {len(lower_left)}."
        )
    station_cells = (
        [
            free_front[int((index + 0.5) * len(free_front) / num_stations)]
            for index in range(num_stations)
        ]
        if on_wall
        else []
    )
    parking_order = right_wall[::-1][:-1] + back_wall[::-1] + left_wall[::-1]
    parking_cells = [cell for cell in parking_order if cell[0] not in taken][: params.num_endpoints]
    if len(parking_cells) < params.num_endpoints:
        raise InstanceGenerationError(
            f"{params.num_endpoints} parking leaves do not fit the {len(parking_order)} free "
            "wall cells."
        )
    leaves += [(cell, at, VertexRole(receiving=True)) for cell, at in dock_cells]
    leaves += [(cell, at, VertexRole(delivery=True)) for cell, at in station_cells]
    leaves += [(cell, at, VertexRole(endpoint=True)) for cell, at in parking_cells]
    used = {cell for cell, _, _ in leaves}
    for cell, at in left_wall + right_wall:
        if cell not in used and cell[0] in rack_rows_of_wall:
            leaves.append((cell, at, slot))
    leaves = _v2_stations_and_slots(leaves, params)

    # Vertex ids: the walkable cells row by row from the front, then the leaves likewise.
    walkable_ids = {cell: index for index, cell in enumerate(sorted(walkable))}
    if len(walkable_ids) != len(walkable) or len(walkable) != num_walkable:
        raise AssertionError(
            f"graph v2 built {len(walkable_ids)} walkable vertices, not {num_walkable}."
        )
    ids = dict(walkable_ids)
    roles: dict[VertexId, VertexRole] = {v: VertexRole(movable=True) for v in walkable_ids.values()}
    edges: set[DirectedEdge] = set()
    for a, b in segments:
        edges.update({(walkable_ids[a], walkable_ids[b]), (walkable_ids[b], walkable_ids[a])})
    edges.update((walkable_ids[a], walkable_ids[b]) for a, b in one_way)
    for cell, at, role in sorted(leaves, key=lambda leaf: leaf[0]):
        if cell in ids:
            raise AssertionError(f"graph v2 placed two vertices on {cell}.")
        leaf = len(ids)
        ids[cell] = leaf
        roles[leaf] = role
        edges.update({(leaf, walkable_ids[at]), (walkable_ids[at], leaf)})

    def band(column: int) -> int:
        return min(num_aisles - 1, max(0, (column + 1) // _V2_AISLE_PITCH))

    graph = _create_graph(
        vertices={v: Vertex(id=v, role=role) for v, role in roles.items()},
        directed_edges=edges,
        params=params,
        cost_rng=random.Random(f"{params.seed}:v2:costs"),
    )
    return GeneratedInstance(
        layout="v2",
        graph=graph,
        positions={
            v: (column * _GRID_SPACING, row * _GRID_SPACING) for (row, column), v in ids.items()
        },
        aisle_index={v: band(column) for (_, column), v in ids.items()},
        aisle_lengths=tuple(lower + upper for lower, upper in zip(lower_lengths, upper_lengths)),
        cross_aisle_rows=tuple(sorted(link_rows)),
        num_transit_vertices=num_walkable,
    )
