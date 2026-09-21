"""Unit tests for slap_mapd_coupling.environment.spaces."""

import pytest

from slap_mapd_coupling.core.graph import Action, Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.environment.spaces import (
    action_for_slot,
    action_space,
    legality_mask,
    max_out_degree,
    slot_for_action,
)


def _star_graph() -> WarehouseGraph:
    """Vertex 0 has three outgoing moves (to 1, 2, 3); 1/2/3 have none."""
    vertices = {i: Vertex(id=i) for i in range(4)}
    edges = tuple(Edge(source=0, target=i, cost=1.0) for i in (1, 2, 3))
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_max_out_degree_matches_the_busiest_vertex():
    graph = _star_graph()
    assert max_out_degree(graph) == 3


def test_action_space_has_d_max_plus_one_slots():
    graph = _star_graph()
    assert action_space(graph).n == 4  # d_max=3, plus one wait slot


def test_legality_mask_shape_is_d_max_plus_one():
    graph = _star_graph()
    mask = legality_mask(graph, vertex=0)
    assert len(mask) == 4


def test_wait_slot_always_legal():
    graph = _star_graph()
    assert legality_mask(graph, vertex=0)[0] == 0.0
    assert legality_mask(graph, vertex=1)[0] == 0.0  # vertex 1 has no moves at all


def test_busiest_vertex_has_no_masked_move_slots():
    graph = _star_graph()
    mask = legality_mask(graph, vertex=0)
    assert mask == (0.0, 0.0, 0.0, 0.0)


def test_leaf_vertex_masks_every_move_slot():
    graph = _star_graph()
    mask = legality_mask(graph, vertex=1)
    assert mask == (0.0, float("-inf"), float("-inf"), float("-inf"))


def test_action_for_slot_zero_is_always_wait():
    graph = _star_graph()
    assert action_for_slot(graph, vertex=0, slot=0) == Action(kind="wait")
    assert action_for_slot(graph, vertex=1, slot=0) == Action(kind="wait")


def test_action_for_slot_decodes_a_legal_move():
    graph = _star_graph()
    legal_moves = {graph.legal_actions(0)[k].target for k in (1, 2, 3)}
    assert legal_moves == {1, 2, 3}


def test_action_for_slot_raises_on_masked_slot():
    graph = _star_graph()
    with pytest.raises(ValueError):
        action_for_slot(graph, vertex=1, slot=1)  # vertex 1 has no move slot 1


def test_slot_for_action_round_trips_with_action_for_slot():
    graph = _star_graph()
    for slot in range(len(graph.legal_actions(0))):
        action = action_for_slot(graph, vertex=0, slot=slot)
        assert slot_for_action(graph, vertex=0, action=action) == slot


def test_slot_for_action_raises_for_illegal_action():
    graph = _star_graph()
    with pytest.raises(ValueError):
        slot_for_action(graph, vertex=1, action=Action(kind="move", target=2))
