"""Topology-first generator for irregular, well-formed warehouse graphs.

Generation has four separate stages:

1. Build a random warehouse topology containing only plain movable vertices.
2. Orient some transit segments as one-way while retaining strong connectivity.
3. Randomly assign storage, delivery, and parking roles to EXISTING vertices.
4. Assign edge costs and construct the immutable ``WarehouseGraph``.

Storage, delivery, and parking vertices are therefore allowed in the middle of
an aisle or cross-aisle. They are not automatically added as leaf vertices.
A role assignment is accepted only when every pair of role vertices can reach
each other without using a third role vertex as an intermediate vertex.

The integer lattice is visualization metadata only. It gives every displayed
node a unique position with equal spacing; it does not determine whether a
vertex is plain, storage, delivery, or parking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random

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

    ``max_num_aisles`` and ``max_aisle_length`` are upper bounds. The actual
    aisle count and the length of every aisle are sampled from ``seed``.

    ``num_transit_vertices`` is the total number of graph vertices because
    roles are assigned to existing topology vertices. Set it to ``None`` to
    sample the graph size as well.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_num_aisles: PositiveInt
    max_aisle_length: PositiveInt
    max_num_cross_aisles: NonNegativeInt
    num_transit_vertices: PositiveInt | None = None
    require_irregular_shape: bool = True

    one_way_fraction: float = Field(ge=0.0, le=1.0)

    default_edge_cost: PositiveFloat = 1.0
    random_edge_costs: bool = False
    edge_cost_range: tuple[PositiveFloat, PositiveFloat] = (0.5, 2.0)
    wait_cost: NonNegativeFloat

    num_storage_vertices: PositiveInt
    num_delivery_vertices: PositiveInt
    num_endpoints: PositiveInt

    # Set this above zero when an experiment explicitly requires some role
    # vertices to have at least two distinct neighbours. A value of zero means
    # all well-formed random placements are allowed, including topology leaves.
    min_internal_role_vertices: NonNegativeInt = 0

    seed: int

    @property
    def num_role_vertices(self) -> int:
        """Number of vertices treated as endpoints by well-formedness."""
        return self.num_storage_vertices + self.num_delivery_vertices + self.num_endpoints

    @property
    def max_num_vertices(self) -> int:
        return self.max_num_aisles * (self.max_aisle_length + 1)

    @model_validator(mode="after")
    def _request_must_be_feasible(self) -> "GeneratorParams":
        """Reject impossible settings instead of silently clamping them."""
        minimum_cost, maximum_cost = self.edge_cost_range
        if minimum_cost > maximum_cost:
            raise ValueError("edge_cost_range must be ordered as (minimum, maximum).")

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
                "of storage, delivery, and parking role vertices."
            )

        return self


class GeneratedInstance(BaseModel):
    """A graph, its visualization positions, and its realized random shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph: WarehouseGraph
    positions: dict[VertexId, Position]
    aisle_lengths: tuple[PositiveInt, ...]
    cross_aisle_rows: tuple[PositiveInt, ...]
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

        if len(set(self.positions.values())) != len(self.positions):
            raise ValueError("Every graph vertex must have a unique display position.")

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
        self.grid[cell] = vertex_id
        return vertex_id


_GRID_SPACING = 1.5
_MAX_TOPOLOGY_ATTEMPTS = 100
_MAX_ROLE_ATTEMPTS = 10_000


def _to_plot_position(cell: GridCell) -> Position:
    """Convert one logical display cell into plotting coordinates."""
    row, column = cell
    return column * _GRID_SPACING, -row * _GRID_SPACING


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
    """Generate a well-formed instance, discarding topologies that admit none.

    Every instance must be well-formed, so a sampled topology that no role
    assignment can make well-formed (for example a near-tree with too few
    cycles) is discarded and the next one is sampled. Attempt ``k`` draws
    from streams derived from ``(seed, k)``, so the same seed always gives
    the same instance and distinct seeds never share a stream. Attempt 0
    uses the streams a single-attempt build would, so a seed that succeeds
    first time is unchanged. ``GeneratedInstance.topology_attempts`` records
    how many topologies were sampled, which gives the discard rate.
    """
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
    role_vertices = _sample_well_formed_role_vertices(
        vertex_ids=set(topology.vertices),
        directed_edges=directed_edges,
        num_role_vertices=params.num_role_vertices,
        min_internal_vertices=params.min_internal_role_vertices,
        rng=role_rng,
    )
    if role_vertices is None:
        return None, f"{description}, no well-formed role assignment"

    topology.vertices = _assign_roles(
        plain_vertices=topology.vertices,
        role_vertex_ids=role_vertices,
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
    rng: random.Random,
) -> tuple[VertexId, ...] | None:
    """Randomly select existing nodes whose role assignment is well-formed.

    A candidate assignment is accepted when:

    1. enough selected vertices are non-leaves when requested;
    2. all remaining plain vertices form one strongly connected backbone;
    3. every selected role vertex can enter and leave that plain backbone; and
    4. the exact endpoint-avoiding reachability predicate succeeds.

    Conditions 2 and 3 are a constructive sufficient condition: every role
    vertex can travel to any other through plain nodes only. Condition 4 is
    retained as an explicit guard against implementation mistakes.
    """
    ordered_vertices = sorted(vertex_ids)
    outgoing, incoming = _adjacency_tables(vertex_ids, directed_edges)

    undirected_neighbours = {
        vertex: set(outgoing[vertex]) | set(incoming[vertex]) for vertex in vertex_ids
    }

    for _attempt in range(_MAX_ROLE_ATTEMPTS):
        selected = set(rng.sample(ordered_vertices, num_role_vertices))

        internal_count = sum(len(undirected_neighbours[vertex]) >= 2 for vertex in selected)
        if internal_count < min_internal_vertices:
            continue

        if not _roles_share_plain_backbone(
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

        result = list(selected)
        rng.shuffle(result)
        return tuple(result)

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
) -> dict[VertexId, Vertex]:
    """Assign role labels to selected existing topology vertices."""
    selected = list(role_vertex_ids)
    rng.shuffle(selected)

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
