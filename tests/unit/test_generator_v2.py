"""Graph v2 (instances/generator.py::_build_v2): every endpoint a leaf off the walkable
backbone, a one-way main pair through the rack area, workstations on the front wall, docks at
the front left corner and parking from the back right corner. Well-formed by construction.
"""

import json
from collections import deque
from functools import cache
from pathlib import Path

import pytest

from slap_mapd_coupling.environment.zones import zones_by_aisle
from slap_mapd_coupling.instances.generator import (
    VERTICES_PER_AGENT,
    GeneratedInstance,
    GeneratorParams,
    InstanceGenerationError,
    fleet_defaults,
    fleet_defaults_v1,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness

DATA = Path(__file__).parent / "data"
FLEETS = (8, 16, 32, 64)


@cache
def _instance(num_agents: int, seed: int | None = None) -> GeneratedInstance:
    settings = fleet_defaults(num_agents)
    if seed is not None:
        settings["seed"] = seed
    return generate_instance(GeneratorParams(**settings, wait_cost=0.0))


def _neighbours(instance: GeneratedInstance) -> dict[int, set[int]]:
    """The undirected neighbours of every vertex."""
    neighbours: dict[int, set[int]] = {v: set() for v in instance.graph.vertices}
    for edge in instance.graph.edges:
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
    return neighbours


def _walkable(instance: GeneratedInstance) -> set[int]:
    return {v for v, x in instance.graph.vertices.items() if not x.role.is_task_endpoint}


def _row(instance: GeneratedInstance, v: int) -> int:
    return round(instance.positions[v][1] / 1.5)


def _column(instance: GeneratedInstance, v: int) -> int:
    return round(instance.positions[v][0] / 1.5)


def _main_pair(instance: GeneratedInstance) -> set[int]:
    return {v for edge in instance.graph.edges if edge.one_way for v in (edge.source, edge.target)}


def _strongly_connected(instance: GeneratedInstance, vertices: set[int]) -> bool:
    forward: dict[int, list[int]] = {v: [] for v in instance.graph.vertices}
    backward: dict[int, list[int]] = {v: [] for v in instance.graph.vertices}
    for edge in instance.graph.edges:
        forward[edge.source].append(edge.target)
        backward[edge.target].append(edge.source)

    def reach(start: int, out: dict[int, list[int]]) -> set[int]:
        seen, queue = {start}, deque([start])
        while queue:
            v = queue.popleft()
            for w in out[v]:
                if w in vertices and w not in seen:
                    seen.add(w)
                    queue.append(w)
        return seen

    start = min(vertices)
    return reach(start, forward) == vertices and reach(start, backward) == vertices


# The shape of the fleet warehouses.


@pytest.mark.parametrize("num_agents", FLEETS)
def test_every_endpoint_is_a_leaf_and_no_walkable_vertex_is_an_endpoint(num_agents):
    instance = _instance(num_agents)
    neighbours = _neighbours(instance)
    walkable = _walkable(instance)
    for v, vertex in instance.graph.vertices.items():
        if vertex.role.is_task_endpoint:
            assert len(neighbours[v]) == 1
            assert neighbours[v] <= walkable
        else:
            assert len(neighbours[v]) >= 2


@pytest.mark.parametrize("num_agents", FLEETS)
def test_the_counts_follow_the_fleet_size(num_agents):
    instance = _instance(num_agents)
    roles = [vertex.role for vertex in instance.graph.vertices.values()]
    assert len(_walkable(instance)) == VERTICES_PER_AGENT * num_agents
    assert sum(role.delivery for role in roles) == num_agents // 2
    assert sum(role.receiving for role in roles) == num_agents // 2
    assert sum(role.endpoint for role in roles) == num_agents
    # A leaf has exactly one role.
    for role in roles:
        assert sum((role.storage, role.delivery, role.receiving, role.endpoint)) <= 1
    # The slots off the walls: the front lane's rack side, two cells between each pair of
    # aisles, and a rack face on each inner side of every aisle vertex in the rack rows above
    # the bottom one, less the two faces each extra link passes through.
    params = GeneratorParams(**fleet_defaults(num_agents), wait_cost=0.0)
    rack_rows, num_aisles = params.rack_rows, params.num_aisles
    width = max(_column(instance, v) for v in instance.graph.vertices)
    faces = 0
    for aisle, length in enumerate(instance.aisle_lengths):
        sides = 1 if aisle in (0, num_aisles - 1) else 2
        faces += sides * (length - 1)  # the bottom rack row faces the front lane instead
    num_links = 2 * (num_aisles - 1) * params.links_per_aisle_pair
    inner_slots = [
        v
        for v, x in instance.graph.vertices.items()
        if x.role.storage and _column(instance, v) not in (-1, width)
    ]
    assert len(inner_slots) == 2 * (num_aisles - 1) + faces - 2 * num_links
    assert rack_rows * 2 == instance.aisle_lengths[0]


@pytest.mark.parametrize("num_agents", FLEETS)
def test_the_main_pair_is_one_way_forms_a_loop_and_carries_no_leaf(num_agents):
    instance = _instance(num_agents)
    graph = instance.graph
    main = _main_pair(instance)
    rows = {_row(instance, v) for v in main}
    assert len(rows) == 2
    low, high = sorted(rows)
    assert high == low + 1
    for edge in graph.edges:
        if edge.one_way:
            step = _column(instance, edge.target) - _column(instance, edge.source)
            assert step == (1 if _row(instance, edge.source) == low else -1)
    # The main pair rows hold no other vertex, and every walkable vertex on them is in it.
    on_rows = {v for v in graph.vertices if _row(instance, v) in rows}
    assert on_rows == main
    # Joined at both ends: the main pair alone, with the two-way edges between its rows, is
    # strongly connected.
    assert _strongly_connected(instance, main)
    neighbours = _neighbours(instance)
    for v in main:
        assert not any(graph.vertices[w].role.is_task_endpoint for w in neighbours[v])
    # Every edge off the main pair is two-way.
    for edge in graph.edges:
        along_a_main_row = (
            edge.source in main
            and edge.target in main
            and _row(instance, edge.source) == _row(instance, edge.target)
        )
        assert edge.one_way == along_a_main_row


@pytest.mark.parametrize("num_agents", FLEETS)
def test_workstations_hang_off_the_front_lane_and_docks_sit_at_the_front_left_corner(
    num_agents,
):
    instance = _instance(num_agents)
    graph = instance.graph
    neighbours = _neighbours(instance)
    walkable = _walkable(instance)
    front_lane = {v for v in walkable if _row(instance, v) == 0}
    main_low = min(_row(instance, v) for v in _main_pair(instance))
    for v, vertex in graph.vertices.items():
        if vertex.role.delivery:
            assert neighbours[v] <= front_lane
            assert _row(instance, v) == -1
        if vertex.role.receiving:
            # On the left wall below the main pair, or round the corner on the front wall.
            assert (_column(instance, v) == -1 and _row(instance, v) < main_low) or (
                _row(instance, v) == -1
            )
    # The nearest slot is 2 steps from a workstation.
    stations = [v for v, x in graph.vertices.items() if x.role.delivery]
    slots = [v for v, x in graph.vertices.items() if x.role.storage]
    assert min(graph.distance(s, g) for s in slots for g in stations) == 2


@pytest.mark.parametrize("num_agents", FLEETS)
def test_parking_starts_at_the_back_right_corner(num_agents):
    instance = _instance(num_agents)
    parking = [v for v, x in instance.graph.vertices.items() if x.role.endpoint]
    width = max(_column(instance, v) for v in instance.graph.vertices)
    on_right_wall = [v for v in parking if _column(instance, v) == width]
    assert on_right_wall
    back = max(_row(instance, v) for v in _walkable(instance))
    assert any(_row(instance, v) == back for v in on_right_wall)


@pytest.mark.parametrize("num_agents", FLEETS)
def test_some_aisles_are_dead_ends_and_the_outer_aisles_reach_the_back_lane(num_agents):
    instance = _instance(num_agents)
    lengths = instance.aisle_lengths
    full = lengths[0]
    assert lengths[-1] == full
    assert min(lengths) < full
    assert instance.cross_aisle_rows


@pytest.mark.parametrize("num_agents", FLEETS)
def test_every_vertex_has_an_aisle_and_the_zones_cover_the_graph(num_agents):
    instance = _instance(num_agents)
    assert set(instance.aisle_index) == set(instance.graph.vertices)
    assert set(instance.aisle_index.values()) == set(range(instance.num_aisles))
    neighbours = _neighbours(instance)
    for v, vertex in instance.graph.vertices.items():
        if vertex.role.is_task_endpoint:
            (at,) = neighbours[v]
            assert instance.aisle_index[v] == instance.aisle_index[at]
    zones = zones_by_aisle(instance.graph, instance.aisle_index, 3)
    members = [set(zone) for zone in zones]
    movable = {v for v, x in instance.graph.vertices.items() if x.role.movable}
    assert set().union(*members) == movable
    assert sum(len(m) for m in members) == len(movable)


# Well-formedness by construction.


@pytest.mark.parametrize("num_agents", FLEETS)
def test_every_seed_is_strongly_connected_and_passes_both_checks(num_agents):
    # 50 seeds where a build is quick. A 64-agent build takes about 20 s, mostly the graph's
    # all-pairs distances, so the larger fleets take a few seeds here and 50 in the instance
    # report (scripts/instance/instance_report.py).
    seeds = range(50) if num_agents <= 16 else range(3)
    for seed in seeds:
        instance = _instance(num_agents, seed)
        assert instance.topology_attempts == 1
        assert _strongly_connected(instance, _walkable(instance))
        assert check_connectivity(instance.graph).ok
        assert check_well_formedness(instance.graph, num_agents).ok


# Step 12: workstations on rack faces (D24) and a set number of slots (D25).


@cache
def _step12_instance(num_agents: int, num_slots: int | None, seed: int = 42) -> GeneratedInstance:
    settings = {
        **fleet_defaults(num_agents),
        "station_placement": "rack_faces",
        "num_storage_vertices": num_slots,
        "seed": seed,
    }
    return generate_instance(GeneratorParams(**settings, wait_cost=0.0))


def _roles_by_cell(instance: GeneratedInstance) -> dict[tuple[float, float], str]:
    names = ("storage", "delivery", "receiving", "endpoint")
    return {
        instance.positions[v]: next((n for n in names if getattr(x.role, n)), "walkable")
        for v, x in instance.graph.vertices.items()
    }


@pytest.mark.parametrize("num_agents", (8, 16))
def test_workstations_on_rack_faces_take_faces_that_were_slots(num_agents):
    before = _roles_by_cell(_instance(num_agents))
    after = _roles_by_cell(_step12_instance(num_agents, None))
    stations = [cell for cell, role in after.items() if role == "delivery"]
    assert len(stations) == num_agents // 2
    assert all(before[cell] == "storage" for cell in stations)
    # The front wall cells the workstations held are gone, everything else is as before.
    for cell, role in before.items():
        if role == "delivery":
            assert cell not in after
        elif cell not in stations:
            assert after[cell] == role


def test_the_workstations_move_with_the_seed():
    stations = [
        {c for c, r in _roles_by_cell(_step12_instance(8, None, seed)).items() if r == "delivery"}
        for seed in (0, 1)
    ]
    assert stations[0] != stations[1]
    assert _step12_instance(8, None, 0) == _step12_instance(8, None, 0)


@pytest.mark.parametrize("num_agents", (8, 16))
def test_a_set_number_of_slots_leaves_the_other_faces_as_parking(num_agents):
    all_faces = _step12_instance(num_agents, None)
    faces = sum(x.role.storage for x in all_faces.graph.vertices.values())
    instance = _step12_instance(num_agents, 20)
    roles = [x.role for x in instance.graph.vertices.values()]
    assert sum(role.storage for role in roles) == 20
    assert sum(role.endpoint for role in roles) == num_agents + faces - 20
    assert len(_walkable(instance)) == VERTICES_PER_AGENT * num_agents
    neighbours = _neighbours(instance)
    for v, vertex in instance.graph.vertices.items():
        if vertex.role.is_task_endpoint:
            assert len(neighbours[v]) == 1
    assert check_connectivity(instance.graph).ok
    assert check_well_formedness(instance.graph, num_agents).ok


def test_the_slots_are_spread_over_the_rows():
    instance = _step12_instance(8, 20)
    rows = {_row(instance, v) for v, x in instance.graph.vertices.items() if x.role.storage}
    assert len(rows) >= 5


@pytest.mark.parametrize("num_agents", (8, 16))
def test_docks_along_the_left_wall_reach_from_the_front_to_the_back(num_agents):
    settings = {**fleet_defaults(num_agents), "dock_placement": "left_wall"}
    instance = generate_instance(GeneratorParams(**settings, wait_cost=0.0))
    docks = [v for v, x in instance.graph.vertices.items() if x.role.receiving]
    assert len(docks) == num_agents // 2
    assert all(_column(instance, v) == -1 for v in docks)
    rows = sorted(_row(instance, v) for v in docks)
    main_low = min(_row(instance, v) for v in _main_pair(instance))
    assert rows[0] < main_low < rows[-1]
    assert check_well_formedness(instance.graph, num_agents).ok


def test_more_slots_than_faces_is_rejected():
    with pytest.raises(InstanceGenerationError, match="slots do not fit"):
        _step12_instance(8, 10_000)


def test_the_same_seed_gives_the_same_instance_and_another_seed_another():
    first = generate_instance(GeneratorParams(**fleet_defaults(8), wait_cost=0.0))
    assert first == _instance(8)
    assert _instance(8, 1).graph != first.graph


# The settings.


def test_v2_rejects_the_graph_v1_settings_and_needs_its_own():
    settings = fleet_defaults(8)
    with pytest.raises(ValueError, match="only applies to layout='v1'"):
        GeneratorParams(**settings, max_num_aisles=4, wait_cost=0.0)
    with pytest.raises(ValueError, match="requires rack_rows"):
        GeneratorParams(**{**settings, "rack_rows": None}, wait_cost=0.0)
    with pytest.raises(ValueError, match="one_way_fraction"):
        GeneratorParams(**{**settings, "one_way_fraction": 0.2}, wait_cost=0.0)
    with pytest.raises(ValueError, match="do not fit"):
        GeneratorParams(**{**settings, "num_transit_vertices": 1000}, wait_cost=0.0)


def test_v1_needs_its_own_settings():
    settings = fleet_defaults_v1(8)
    with pytest.raises(ValueError, match="only applies to layout='v2'"):
        GeneratorParams(**settings, rack_rows=4, wait_cost=0.0)
    del settings["num_storage_vertices"]
    with pytest.raises(ValueError, match="requires num_storage_vertices"):
        GeneratorParams(**settings, wait_cost=0.0)


# Graph v1 still builds the instance of the graph v1 results.


@pytest.mark.parametrize("num_agents", (8, 16))
def test_layout_v1_rebuilds_the_stored_graph_v1_instance(num_agents):
    params = GeneratorParams(**fleet_defaults_v1(num_agents), wait_cost=0.0)
    assert params.layout == "v1"
    instance = generate_instance(params)
    graph = instance.graph
    built = {
        "positions": {str(v): list(instance.positions[v]) for v in graph.vertices},
        "roles": {
            str(v): [
                name
                for name in ("storage", "delivery", "receiving", "endpoint")
                if getattr(vertex.role, name)
            ]
            for v, vertex in graph.vertices.items()
        },
        "edges": [[e.source, e.target] for e in graph.edges],
    }
    stored = json.loads((DATA / f"graph_v1_fleet_{num_agents}_seed_42.json").read_text())
    assert built == stored
