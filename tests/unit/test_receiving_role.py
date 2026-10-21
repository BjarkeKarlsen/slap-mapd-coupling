"""The receiving role, V_rcv, drawn from row 0, the inbound side (sec:impl:instances)."""

import hashlib
import json
from pathlib import Path

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import role_flags
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance
from slap_mapd_coupling.instances.validation import check_well_formedness
from slap_mapd_coupling.planners import TokenPassingPlanner
from slap_mapd_coupling.training.spec import build_instance, load_spec

ROOT = Path(__file__).resolve().parents[2]
# The roles and edges of every shipped config's instance before the receiving role existed,
# from feat/cold-stock. A config without receiving vertices must still give exactly these.
DIGESTS = {
    "configs/centralised.yaml": "889d7e03de4c5d03",
    "configs/decentralised.yaml": "889d7e03de4c5d03",
    "configs/full_view.yaml": "889d7e03de4c5d03",
    "configs/section.yaml": "889d7e03de4c5d03",
    **{
        f"configs/{folder}/{name}.yaml": "c81bed88bc76e9d2"
        for folder in ("medium", "medium_arrival")
        for name in ("centralised", "decentralised", "full_view", "section")
    },
}


def _digest(graph: WarehouseGraph) -> str:
    roles = sorted(
        (v, x.role.storage, x.role.delivery, x.role.endpoint) for v, x in graph.vertices.items()
    )
    edges = sorted((e.source, e.target, e.cost) for e in graph.edges)
    return hashlib.sha256(json.dumps([roles, edges]).encode()).hexdigest()[:16]


@pytest.mark.parametrize("path", sorted(DIGESTS))
def test_without_receiving_vertices_every_config_gives_the_instance_it_gave_before(path):
    graph = build_instance(load_spec(ROOT / path)).graph
    assert not any(vertex.role.receiving for vertex in graph.vertices.values())
    assert _digest(graph) == DIGESTS[path]


def _medium_params(**overrides) -> GeneratorParams:
    instance = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml").instance
    return GeneratorParams(**instance, wait_cost=0.2, num_endpoints=8, **overrides)


def test_the_receiving_vertices_are_on_row_0_and_the_instance_is_well_formed():
    instance = generate_instance(_medium_params(num_receiving_vertices=4))
    graph = instance.graph
    receiving = [v for v, vertex in graph.vertices.items() if vertex.role.receiving]
    assert len(receiving) == 4
    # row 0 is drawn at height 0
    assert all(instance.positions[v][1] == 0.0 for v in receiving)
    # a receiving vertex has no other role
    for v in receiving:
        role = graph.vertices[v].role
        assert not (role.storage or role.delivery or role.endpoint)
    assert check_well_formedness(graph, 8).ok


def test_the_receiving_vertices_count_as_role_vertices():
    params = _medium_params(num_receiving_vertices=4)
    assert params.num_role_vertices == _medium_params().num_role_vertices + 4


def test_a_receiving_vertex_has_only_the_receiving_flag():
    assert role_flags(VertexRole(receiving=True)) == (False, False, False, False, True)
    assert VertexRole(receiving=True).is_task_endpoint


def test_a_receiving_vertex_must_be_movable():
    with pytest.raises(ValueError, match="movable"):
        VertexRole(movable=False, receiving=True)


def test_token_passing_never_parks_on_a_receiving_vertex():
    # The hub (0) with endpoints 1, 2 and 3, a delivery vertex 4 and a corridor 0 - 5 - 6 to
    # the only shelf. Endpoint 1 is also a receiving vertex.
    roles = {
        0: VertexRole(),
        1: VertexRole(endpoint=True, receiving=True),
        2: VertexRole(endpoint=True),
        3: VertexRole(endpoint=True),
        4: VertexRole(delivery=True),
        5: VertexRole(),
        6: VertexRole(storage=True),
    }
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (5, 6)]
    edges = tuple(Edge(source=a, target=b, cost=1.0) for a, b in pairs) + tuple(
        Edge(source=b, target=a, cost=1.0) for a, b in pairs
    )
    graph = WarehouseGraph(
        vertices={v: Vertex(id=v, role=role) for v, role in roles.items()},
        edges=edges,
        wait_cost=0.0,
    )
    fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=4), 2: AgentState(agent_id=2, location=6)}
    )
    task = Task(task_id=1, release_time=0, pickup_vertex=6, delivery_vertex=4, sku="tea")
    planner = TokenPassingPlanner()
    # Agent 1 stands on the waiting task's delivery vertex, so it moves off to park. The
    # nearest endpoint by vertex id is 1, a receiving vertex, so it parks on 2.
    assert planner.assign(graph, fleet, [task], 0) == {1: 2}
    _, path = planner._paths[1]
    assert path[-1] == 2
