"""Unit tests for slap_mapd_coupling.models.local_subgraph_encoding."""

import numpy as np
import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import build_observation, local_subgraph
from slap_mapd_coupling.environment.spaces import legality_mask, max_out_degree
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM, SENDER_KEYS
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    check_slot_targets,
    check_windows,
    encode_all_observations,
    encode_observation,
    encode_view,
    encoded_observation_space,
)

# The column of the claimed flag, the vertex features in the order of gnn_encoder.py.
CLAIMED = 6


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _fleet(locations: dict[int, int]) -> FleetState:
    return FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in locations.items()}
    )


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=1.0,
        seed=0,
        horizon=50,
        wait_cost=0.0,
        observation_depth=2,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        move_cost=0.0,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def test_encode_observation_shapes_match_config():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=3)

    encoded = encode_observation(graph, 2, observation, mask, d_max, encoding_config)

    assert encoded.node_features.shape == (8, NODE_FEATURE_DIM)
    assert encoded.node_mask.shape == (8,)
    assert encoded.edge_index.shape == (8 * d_max, 2)
    assert encoded.edge_mask.shape == (8 * d_max,)
    assert encoded.action_mask.shape == (d_max + 1,)


def test_encode_observation_own_index_points_at_agent_vertex():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    config = _config(observation_depth=3)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(graph, 2, observation, mask, d_max, encoding_config)

    own_features = observation.features[2]
    np.testing.assert_allclose(
        encoded.node_features[encoded.own_index],
        [
            float(own_features.storage),
            float(own_features.delivery),
            float(own_features.endpoint),
            float(own_features.transit),
            own_features.eta,
            float(own_features.occupied),
            float(own_features.claimed),
            float(own_features.receiving),
        ],
    )
    assert encoded.node_mask[encoded.own_index] == 1.0


def test_encode_observation_writes_the_claim_column():
    graph = _line_graph(5)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=3)
    observation = build_observation(
        graph, fleet, [], agent_id=1, t=0, config=config, claimed={3}
    )
    d_max = max_out_degree(graph)
    encoded = encode_observation(
        graph,
        2,
        observation,
        legality_mask(graph, vertex=2, d_max=d_max),
        d_max,
        LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1),
    )
    index = {v: i for i, v in enumerate(observation.visible_vertices)}
    assert encoded.node_features[index[3], CLAIMED] == 1.0
    assert encoded.node_features[index[2], CLAIMED] == 0.0


def test_encode_observation_node_mask_marks_padding():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    assert len(observation.visible_vertices) == 3  # {1, 2, 3}
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(graph, 2, observation, mask, d_max, encoding_config)

    assert encoded.node_mask.sum() == 3
    assert encoded.node_mask[3:].sum() == 0


def _valid_pairs(encoded, visible) -> set[frozenset[int]]:
    """The vertex pairs of the valid edge rows, as vertex ids."""
    return {
        frozenset((visible[a], visible[b]))
        for (a, b), valid in zip(encoded.edge_index.tolist(), encoded.edge_mask)
        if valid == 1.0
    }


def test_encode_observation_edge_list_matches_line_graph_neighbours():
    graph = _line_graph(6)
    fleet = _fleet({1: 2})
    config = _config(observation_depth=1)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(graph, 2, observation, mask, d_max, encoding_config)

    # visible = {1, 2, 3}; edges 1-2 and 2-3, no edge 1-3. Each pair is one row.
    assert _valid_pairs(encoded, observation.visible_vertices) == {
        frozenset((1, 2)),
        frozenset((2, 3)),
    }
    assert encoded.edge_mask.sum() == 2
    assert encoded.edge_index.dtype == np.int64
    assert not encoded.edge_index[2:].any()  # padded rows point at index 0


def test_encode_observation_lists_a_one_way_edge_and_a_two_way_edge_once():
    vertices = {i: Vertex(id=i) for i in range(3)}
    edges = (
        Edge(source=0, target=1, cost=1.0),  # one way only
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=1, cost=1.0),  # both ways
        Edge(source=2, target=0, cost=1.0),
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    fleet = _fleet({1: 0})
    observation = build_observation(
        graph, fleet, [], agent_id=1, t=0, config=_config(observation_depth=5)
    )
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=0, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=4, max_messages=1)

    encoded = encode_observation(graph, 0, observation, mask, d_max, encoding_config)

    assert _valid_pairs(encoded, observation.visible_vertices) == {
        frozenset((0, 1)),
        frozenset((1, 2)),
        frozenset((0, 2)),
    }
    assert encoded.edge_mask.sum() == 3  # four directed edges, three pairs


def test_a_field_of_view_above_the_cap_raises_instead_of_being_truncated():
    graph = _line_graph(11)
    fleet = _fleet({1: 5})
    task = Task(task_id=1, release_time=0, pickup_vertex=10, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    config = _config(observation_depth=5)
    observation = build_observation(graph, fleet, [task], agent_id=1, t=0, config=config)
    assert len(observation.visible_vertices) == 11  # {0..10}
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=5, d_max=d_max)

    too_small = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=1)
    with pytest.raises(ValueError, match="never truncated"):
        encode_observation(graph, 5, observation, mask, d_max, too_small)
    # a cap the field of view fits keeps every vertex
    fits = LocalSubgraphEncodingConfig(max_local_nodes=11, max_messages=1)
    encoded = encode_observation(graph, 5, observation, mask, d_max, fits)
    assert encoded.node_mask.sum() == 11
    assert encoded.edge_mask.sum() == 10


def test_encode_observation_congestion_and_action_mask():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 3})
    config = _config(observation_depth=2, congestion_sensitive=True)
    observation = build_observation(graph, fleet, [], agent_id=1, t=0, config=config)
    d_max = max_out_degree(graph)
    mask = legality_mask(graph, vertex=2, d_max=d_max)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    encoded = encode_observation(graph, 2, observation, mask, d_max, encoding_config)

    assert encoded.congestion == pytest.approx(observation.congestion)
    assert set(np.unique(encoded.action_mask)).issubset({0.0, 1.0})
    assert encoded.action_mask[0] == 1.0  # wait is always legal


def _encode_all(fleet, communication, max_messages=2, depth=3, num_vertices=10, pooling="none"):
    graph = _line_graph(num_vertices)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=max_messages)
    config = _config(observation_depth=depth, communication=communication)
    encoded = encode_all_observations(
        graph, fleet, [], 0, config, encoding_config, max_out_degree(graph), pooling=pooling
    )
    return graph, encoding_config, encoded


# agent 1 at vertex 4 hears agent 5 (distance 1), agents 2 and 3 (distance 2) and agent 4
# (distance 3), in that order
_FLEET_LOCATIONS = {1: 4, 4: 7, 3: 2, 2: 6, 5: 5}


@pytest.mark.parametrize("pooling", ["none", "mean"])
@pytest.mark.parametrize("communication", [False, True])
def test_encoded_observation_space_matches_the_encoded_arrays(communication, pooling):
    graph, encoding_config, encoded = _encode_all(
        _fleet(_FLEET_LOCATIONS), communication, pooling=pooling
    )
    space = encoded_observation_space(
        encoding_config, max_out_degree(graph), communication, pooling=pooling
    )

    assert ("pool_mask" in space.spaces) == (pooling == "mean")
    for item in encoded.values():
        assert space.contains(item.as_rllib_dict()["observations"])
    assert space["edge_index"].shape == (8 * 2, 2)
    assert space["edge_index"].dtype == np.int64


def test_sender_keys_are_absent_with_communication_off_and_present_with_it_on():
    _, encoding_config, off = _encode_all(_fleet(_FLEET_LOCATIONS), communication=False)
    _, _, on = _encode_all(_fleet(_FLEET_LOCATIONS), communication=True)

    off_keys = set(off[1].as_rllib_dict()["observations"])
    on_keys = set(on[1].as_rllib_dict()["observations"])
    assert off_keys.isdisjoint(SENDER_KEYS)
    assert on_keys == off_keys | set(SENDER_KEYS) | {"move_sender_index", "move_sender_mask"}
    assert off[1].sender_mask is None
    assert off[1].move_sender_index is None


def test_sender_arrays_have_the_documented_shapes_and_dtypes():
    graph, encoding_config, encoded = _encode_all(_fleet(_FLEET_LOCATIONS), communication=True)
    item = encoded[1]
    d_max = max_out_degree(graph)

    assert item.sender_node_features.shape == (2, 8, NODE_FEATURE_DIM)
    assert item.sender_node_mask.shape == (2, 8)
    assert item.sender_edge_index.shape == (2, 8 * d_max, 2)
    assert item.sender_edge_index.dtype == np.int64
    assert item.sender_edge_mask.shape == (2, 8 * d_max)
    assert item.sender_own_index.shape == (2,)
    assert item.sender_own_index.dtype == np.int64
    assert item.sender_mask.shape == (2,)


def test_a_senders_block_equals_that_robots_own_encoding():
    _, _, encoded = _encode_all(_fleet(_FLEET_LOCATIONS), communication=True, max_messages=4)
    receiver = encoded[1]
    heard = (5, 2, 3, 4)

    for slot, sender_id in enumerate(heard):
        own = encoded[sender_id]
        np.testing.assert_array_equal(receiver.sender_node_features[slot], own.node_features)
        np.testing.assert_array_equal(receiver.sender_node_mask[slot], own.node_mask)
        np.testing.assert_array_equal(receiver.sender_edge_index[slot], own.edge_index)
        np.testing.assert_array_equal(receiver.sender_edge_mask[slot], own.edge_mask)
        assert receiver.sender_own_index[slot] == own.own_index
        assert receiver.sender_mask[slot] == 1.0


def test_encode_view_is_the_robots_own_block_of_encode_observation():
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    observation = build_observation(graph, fleet, [], 1, 0, _config(observation_depth=3))
    d_max = max_out_degree(graph)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)

    view = encode_view(graph, 2, observation, d_max, encoding_config)
    encoded = encode_observation(
        graph, 2, observation, legality_mask(graph, 2, d_max), d_max, encoding_config
    )

    np.testing.assert_array_equal(view.node_features, encoded.node_features)
    np.testing.assert_array_equal(view.edge_index, encoded.edge_index)
    assert view.own_index == encoded.own_index


def test_senders_are_closest_first_and_the_cap_keeps_the_closest():
    _, _, encoded = _encode_all(_fleet(_FLEET_LOCATIONS), communication=True, max_messages=2)
    receiver = encoded[1]

    assert receiver.sender_mask.tolist() == [1.0, 1.0]
    # agent 5 is the closest, then agent 2 beats agent 3 on the tie at distance 2
    np.testing.assert_array_equal(receiver.sender_node_features[0], encoded[5].node_features)
    np.testing.assert_array_equal(receiver.sender_node_features[1], encoded[2].node_features)


def test_sender_blocks_do_not_depend_on_the_order_agents_are_listed_in_the_fleet():
    shuffled = {5: 5, 2: 6, 1: 4, 3: 2, 4: 7}
    _, _, first = _encode_all(_fleet(_FLEET_LOCATIONS), communication=True)
    _, _, second = _encode_all(_fleet(shuffled), communication=True)

    assert list(first) == list(second) == [1, 2, 3, 4, 5]
    for agent_id in first:
        expected = first[agent_id].as_rllib_dict()["observations"]
        actual = second[agent_id].as_rllib_dict()["observations"]
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)


def test_fewer_senders_than_the_cap_leave_zero_padding():
    _, _, encoded = _encode_all(_fleet({1: 0, 2: 1, 3: 9}), communication=True, max_messages=3)
    receiver = encoded[1]

    assert receiver.sender_mask.tolist() == [1.0, 0.0, 0.0]  # agent 3 is out of range
    assert not receiver.sender_node_features[1:].any()
    assert not receiver.sender_node_mask[1:].any()
    assert not receiver.sender_edge_mask[1:].any()
    assert encoded[3].sender_mask.sum() == 0  # a robot alone in its field of view hears nobody


def test_the_observations_are_keyed_by_agent_id_in_ascending_order():
    _, _, encoded = _encode_all(_fleet(_FLEET_LOCATIONS), communication=False)
    assert list(encoded) == [1, 2, 3, 4, 5]


def test_communication_with_no_room_for_a_sender_is_rejected():
    graph = _line_graph(6)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=0)
    config = _config(observation_depth=3, communication=True)

    with pytest.raises(ValueError, match="max_messages"):
        encoded_observation_space(encoding_config, max_out_degree(graph), communication=True)
    with pytest.raises(ValueError, match="max_messages"):
        encode_all_observations(
            graph, _fleet({1: 0, 2: 1}), [], 0, config, encoding_config, max_out_degree(graph)
        )
    # with communication off the cap is never read
    encoded_observation_space(encoding_config, max_out_degree(graph), communication=False)


def _star_graph() -> WarehouseGraph:
    """Vertex 0 in the middle, leaves 1, 2 and 3 at costs 1, 2 and 3. The
    edges are listed out of order, so slot order has to come from the
    vertex numbers."""
    vertices = {i: Vertex(id=i) for i in range(4)}
    edges = tuple(
        Edge(source=a, target=b, cost=float(leaf))
        for leaf in (3, 1, 2)
        for a, b in ((0, leaf), (leaf, 0))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _encode_star(locations, communication, max_messages=2, max_local_nodes=6):
    graph = _star_graph()
    encoding_config = LocalSubgraphEncodingConfig(
        max_local_nodes=max_local_nodes, max_messages=max_messages
    )
    config = _config(observation_depth=3, communication=communication)
    encoded = encode_all_observations(
        graph, _fleet(locations), [], 0, config, encoding_config, max_out_degree(graph)
    )
    return graph, encoding_config, encoded


def test_move_target_index_follows_increasing_vertex_number_and_pads_with_zero():
    _, _, encoded = _encode_star({1: 0, 2: 1}, communication=False)

    centre, leaf = encoded[1], encoded[2]
    assert centre.move_target_index.dtype == np.int64
    assert centre.action_mask[1:].tolist() == [1.0, 1.0, 1.0]
    # eta is the distance back to the robot's own vertex, so it names each leaf's cost
    assert centre.node_features[centre.move_target_index, 4].tolist() == [1.0, 2.0, 3.0]
    assert leaf.action_mask[1:].tolist() == [1.0, 0.0, 0.0]  # one neighbour, two padding slots
    assert leaf.move_target_index[0] != leaf.own_index
    assert leaf.move_target_index[1:].tolist() == [0, 0]


@pytest.mark.parametrize("communication", [False, True])
def test_encoded_observation_space_declares_the_slot_arrays(communication):
    graph, encoding_config, encoded = _encode_star({1: 0, 2: 1, 3: 3}, communication)
    space = encoded_observation_space(encoding_config, max_out_degree(graph), communication)

    assert space["move_target_index"].shape == (3,)
    assert ("move_sender_index" in space.spaces) == communication
    assert ("move_sender_mask" in space.spaces) == communication
    for item in encoded.values():
        assert space.contains(item.as_rllib_dict()["observations"])


def test_a_move_slots_sender_is_the_heard_robot_standing_on_its_vertex():
    # robot 1 stands on the centre and hears robot 2 (distance 1), then robot 3 (distance 3).
    # Slot 1 leads to vertex 1 (robot 2), slot 2 to vertex 2 (empty), slot 3 to vertex 3 (robot 3).
    _, _, encoded = _encode_star({1: 0, 2: 1, 3: 3}, communication=True, max_messages=2)
    item = encoded[1]

    assert item.move_sender_index.dtype == np.int64
    assert item.move_sender_mask.tolist() == [1.0, 0.0, 1.0]
    assert item.move_sender_index.tolist() == [0, 0, 1]
    # the index points at the block that holds that robot's own view
    np.testing.assert_array_equal(
        item.sender_node_features[item.move_sender_index[2]], encoded[3].node_features
    )


def test_a_robot_that_is_not_heard_gives_no_sender_for_its_slot():
    # the cap keeps only the closest sender, robot 2, so robot 3 on vertex 3 is not heard
    _, _, encoded = _encode_star({1: 0, 2: 1, 3: 3}, communication=True, max_messages=1)

    assert encoded[1].move_sender_mask.tolist() == [1.0, 0.0, 0.0]
    assert encoded[1].move_sender_index.tolist() == [0, 0, 0]


def test_a_robot_with_no_neighbours_heard_has_an_all_zero_sender_slot_array():
    _, _, encoded = _encode_star({1: 0}, communication=True)

    assert not encoded[1].move_sender_mask.any()
    assert not encoded[1].move_sender_index.any()


def test_a_cap_below_d_max_plus_one_is_rejected():
    graph = _star_graph()  # d_max = 3, so a view needs room for 4 vertices
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=3, max_messages=1)

    with pytest.raises(ValueError, match="max_local_nodes"):
        encoded_observation_space(encoding_config, max_out_degree(graph), communication=False)
    with pytest.raises(ValueError, match="max_local_nodes"):
        check_slot_targets(graph, 3, encoding_config)
    observation = build_observation(graph, _fleet({1: 0}), [], 1, 0, _config(observation_depth=3))
    with pytest.raises(ValueError, match="do not fit"):
        encode_view(graph, 0, observation, max_out_degree(graph), encoding_config)


def _far_neighbour_graph() -> WarehouseGraph:
    """Vertices 0 and 1 are joined by an edge of cost 3, and 1 and 2 by one of cost 1."""
    vertices = {i: Vertex(id=i) for i in range(3)}
    edges = (
        Edge(source=0, target=1, cost=3.0),
        Edge(source=1, target=0, cost=3.0),
        Edge(source=1, target=2, cost=1.0),
        Edge(source=2, target=1, cost=1.0),
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def test_a_neighbour_outside_the_field_of_view_raises_in_encode_view():
    graph = _far_neighbour_graph()
    config = _config(observation_depth=2)  # vertex 1 is at cost 3 from vertex 0
    observation = build_observation(graph, _fleet({1: 0}), [], 1, 0, config)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=4, max_messages=1)

    assert 1 not in observation.visible_vertices
    with pytest.raises(ValueError, match="outside its field of view"):
        encode_view(graph, 0, observation, max_out_degree(graph), encoding_config)


def test_check_slot_targets_rejects_a_neighbour_outside_the_field_of_view():
    graph = _far_neighbour_graph()
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=4, max_messages=1)

    with pytest.raises(ValueError, match="observation_depth"):
        check_slot_targets(graph, 2, encoding_config)
    check_slot_targets(graph, 3, encoding_config)  # cost distance 3 <= d_obs = 3 is inside


def test_check_slot_targets_rejects_a_field_of_view_above_the_cap():
    graph = _line_graph(11)  # with d_obs = 5 the field of view of vertex 5 is all 11 vertices
    too_small = LocalSubgraphEncodingConfig(max_local_nodes=10, max_messages=1)
    with pytest.raises(ValueError, match="max_local_nodes"):
        check_slot_targets(graph, 5, too_small)
    check_slot_targets(graph, 5, LocalSubgraphEncodingConfig(max_local_nodes=11, max_messages=1))


# Windows on the line graph with vertices 0 to 7 for robot 1 at vertex 3 and robot 2 at
# vertex 5, given as explicit vertex sets. The zones are {0, 1, 2, 3} and {4, 5, 6, 7} with
# d_obs = 1, so each halo is the one vertex across the boundary. Robot 2's halo is listed
# last, so the padded order follows the window and not the vertex numbers.
_WINDOW_LOCATIONS = {1: 3, 2: 5}
_WINDOWS = {
    "ball": {1: (2, 3, 4), 2: (4, 5, 6)},
    "all movable vertices": {1: tuple(range(8)), 2: tuple(range(8))},
    "zone plus halo": {1: (0, 1, 2, 3, 4), 2: (4, 5, 6, 7, 3)},
}


def _encode_windows(windows, observation_depth=1, max_local_nodes=8, **pooling):
    graph = _line_graph(8)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=max_local_nodes, max_messages=1)
    config = _config(observation_depth=observation_depth)
    return encode_all_observations(
        graph,
        _fleet(_WINDOW_LOCATIONS),
        [],
        0,
        config,
        encoding_config,
        max_out_degree(graph),
        windows=windows,
        **pooling,
    )


@pytest.mark.parametrize("name", sorted(_WINDOWS))
def test_the_padded_arrays_follow_the_window(name):
    windows = _WINDOWS[name]
    encoded = _encode_windows(windows)

    for agent_id, location in _WINDOW_LOCATIONS.items():
        window = windows[agent_id]
        other = _WINDOW_LOCATIONS[3 - agent_id]
        item = encoded[agent_id]
        n = len(window)
        assert item.node_mask.tolist() == [1.0] * n + [0.0] * (8 - n)
        # a transit vertex, eta is the distance to the robot's own vertex (no task), and
        # occupied marks the other robot, and nothing is claimed
        expected = [
            [0.0, 0.0, 0.0, 1.0, abs(v - location), float(v == other), 0.0, 0.0] for v in window
        ]
        np.testing.assert_array_equal(item.node_features[:n], expected)
        assert not item.node_features[n:].any()
        assert item.own_index == window.index(location)
        # slot 1 leads to location - 1 and slot 2 to location + 1
        assert item.move_target_index.tolist() == [
            window.index(location - 1),
            window.index(location + 1),
        ]
        assert _valid_pairs(item, window) == {
            frozenset((v, v + 1)) for v in window if v + 1 in window
        }


@pytest.mark.parametrize("name", sorted(_WINDOWS))
def test_the_congestion_feature_does_not_change_with_the_window(name):
    # with d_obs = 2 each robot's field of view holds the other robot and three free
    # vertices, so delta is 1/4 whatever the window
    default = _encode_windows(None, observation_depth=2)
    encoded = _encode_windows(_WINDOWS[name], observation_depth=2)

    for agent_id in _WINDOW_LOCATIONS:
        assert default[agent_id].congestion == pytest.approx(0.25)
        assert encoded[agent_id].congestion == default[agent_id].congestion


def test_a_window_above_the_cap_raises():
    with pytest.raises(ValueError, match="never truncated"):
        _encode_windows(_WINDOWS["all movable vertices"], max_local_nodes=7)


def test_windows_must_name_every_robot():
    with pytest.raises(ValueError, match="no window"):
        _encode_windows({1: (2, 3, 4)})


def test_check_windows_accepts_a_ball_all_movable_vertices_and_a_zone_plus_halo():
    graph = _line_graph(8)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)
    zones = ((0, 1, 2, 3), (4, 5, 6, 7))
    halos = ((4,), (3,))

    check_windows(graph, {v: local_subgraph(graph, v, 1) for v in range(8)}, encoding_config)
    check_windows(graph, {v: tuple(range(8)) for v in range(8)}, encoding_config)
    check_windows(graph, {v: zones[v // 4] + halos[v // 4] for v in range(8)}, encoding_config)


@pytest.mark.parametrize(
    "window, max_local_nodes, match",
    [
        ((2, 4), 8, "leaves out 3 itself"),
        ((0, 1, 2, 3), 8, r"outgoing neighbours \[4\]"),  # the zone without its halo
        (tuple(range(8)), 7, "max_local_nodes"),
        ((2, 3, 3, 4), 8, "more than once"),
    ],
)
def test_check_windows_rejects_a_bad_window(window, max_local_nodes, match):
    graph = _line_graph(8)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=max_local_nodes, max_messages=1)
    with pytest.raises(ValueError, match=match):
        check_windows(graph, {3: window}, encoding_config)


def test_no_windows_gives_the_same_arrays_as_before_windows_existed():
    """The arrays the encoder gave before windows existed, on a fixture whose
    field of view fits the cap."""
    graph = _line_graph(6)
    fleet = _fleet({1: 2, 2: 4})
    task = Task(task_id=1, release_time=0, pickup_vertex=5, delivery_vertex=0, sku="sku-1").assign(
        agent_id=1, t=0
    )
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=6, max_messages=1)
    encoded = encode_all_observations(
        graph, fleet, [task], 0, _config(observation_depth=2), encoding_config, 2
    )

    # robot 1 at 2 sees {0, ..., 4} with eta to its pickup at 5, and robot 2 on vertex 4.
    # Robot 2 at 4 has no task, so eta is the distance to its own vertex.
    first, second = encoded[1], encoded[2]
    expected_first = [
        [0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    expected_second = [
        [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    np.testing.assert_array_equal(first.node_features, expected_first)
    assert first.node_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert first.edge_index.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]] + [[0, 0]] * 8
    assert first.edge_mask.tolist() == [1.0] * 4 + [0.0] * 8
    assert first.own_index == 2
    assert first.move_target_index.tolist() == [1, 3]
    assert first.congestion == pytest.approx(0.25)
    np.testing.assert_array_equal(second.node_features, expected_second)
    assert second.node_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert second.edge_index.tolist() == [[0, 1], [1, 2], [2, 3]] + [[0, 0]] * 9
    assert second.own_index == 2
    assert second.move_target_index.tolist() == [1, 3]
    assert second.congestion == pytest.approx(1 / 3)


@pytest.mark.parametrize("communication", [False, True])
def test_no_windows_equals_the_field_of_view_as_every_robots_window(communication):
    graph = _line_graph(10)
    fleet = _fleet(_FLEET_LOCATIONS)
    config = _config(observation_depth=3, communication=communication)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=2)
    windows = {aid: local_subgraph(graph, loc, 3) for aid, loc in _FLEET_LOCATIONS.items()}

    default = encode_all_observations(graph, fleet, [], 0, config, encoding_config, 2)
    explicit = encode_all_observations(
        graph, fleet, [], 0, config, encoding_config, 2, windows=windows
    )

    for agent_id in default:
        expected = default[agent_id].as_rllib_dict()
        actual = explicit[agent_id].as_rllib_dict()
        np.testing.assert_array_equal(actual["action_mask"], expected["action_mask"])
        assert actual["observations"].keys() == expected["observations"].keys()
        for key, value in expected["observations"].items():
            np.testing.assert_array_equal(actual["observations"][key], value)


# The zones of _WINDOWS["zone plus halo"], one per robot.
_ZONES = {1: (0, 1, 2, 3), 2: (4, 5, 6, 7)}


def test_pooling_off_gives_no_pool_mask():
    encoded = _encode_windows(_WINDOWS["zone plus halo"])
    for item in encoded.values():
        assert item.pool_mask is None
        assert "pool_mask" not in item.as_rllib_dict()["observations"]
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1)
    assert "pool_mask" not in encoded_observation_space(encoding_config, 2, False).spaces


def test_the_pool_mask_marks_the_zone_and_leaves_out_the_halo_and_the_padding():
    encoded = _encode_windows(_WINDOWS["zone plus halo"], pooling="mean", pool_vertices=_ZONES)
    # each window lists its zone first and its one halo vertex fifth, then three padded slots
    for item in encoded.values():
        assert item.pool_mask.dtype == np.float32
        assert item.pool_mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        observations = item.as_rllib_dict()["observations"]
        np.testing.assert_array_equal(observations["pool_mask"], item.pool_mask)


@pytest.mark.parametrize("name", sorted(_WINDOWS))
def test_without_pool_vertices_a_robot_pools_over_its_whole_window(name):
    encoded = _encode_windows(_WINDOWS[name], pooling="mean")
    for item in encoded.values():
        np.testing.assert_array_equal(item.pool_mask, item.node_mask)


def test_pooling_leaves_every_other_array_unchanged():
    off = _encode_windows(_WINDOWS["zone plus halo"])
    on = _encode_windows(_WINDOWS["zone plus halo"], pooling="mean", pool_vertices=_ZONES)
    for agent_id in off:
        expected = off[agent_id].as_rllib_dict()["observations"]
        actual = dict(on[agent_id].as_rllib_dict()["observations"])
        del actual["pool_mask"]
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)


@pytest.mark.parametrize(
    "pooling, pool_vertices, match",
    [
        ("none", _ZONES, "pooling is off"),
        ("mean", {1: _ZONES[1]}, "no pool"),
        ("mean", {1: _ZONES[1], 2: (4, 5, 6, 7, 2)}, r"vertices \[2\], outside its window"),
        ("mean", {1: _ZONES[1], 2: ()}, "no vertex"),
    ],
)
def test_bad_pool_vertices_are_rejected(pooling, pool_vertices, match):
    with pytest.raises(ValueError, match=match):
        _encode_windows(_WINDOWS["zone plus halo"], pooling=pooling, pool_vertices=pool_vertices)


def _encode_turn(fleet, acting=None, chosen=None, communication=False, **config_overrides):
    graph = _line_graph(6)
    config = _config(observation_depth=3, communication=communication, **config_overrides)
    return encode_all_observations(
        graph,
        fleet,
        [],
        0,
        config,
        LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=1),
        max_out_degree(graph),
        acting=acting,
        chosen=chosen,
    )


def test_only_the_acting_robots_are_encoded():
    fleet = _fleet({1: 0, 2: 2, 3: 5})
    assert set(_encode_turn(fleet, acting=[3, 1])) == {1, 3}
    assert set(_encode_turn(fleet)) == {1, 2, 3}


def test_chosen_vertices_become_claims_and_close_those_moves():
    fleet = _fleet({1: 0, 2: 2})
    # robot 2 chose vertex 1, the only vertex robot 1 can move to
    encoded = _encode_turn(fleet, acting=[1], chosen={1: {2: 1}})[1]
    observation = build_observation(
        _line_graph(6), fleet, [], agent_id=1, t=0, config=_config(observation_depth=3)
    )
    index = {v: i for i, v in enumerate(observation.visible_vertices)}
    assert encoded.node_features[index[1], CLAIMED] == 1.0
    assert encoded.node_features[index[0], CLAIMED] == 0.0
    assert encoded.action_mask.tolist()[:2] == [1.0, 0.0]  # wait stays, the move to 1 is closed


def test_a_robot_without_an_entry_in_chosen_has_no_claims():
    fleet = _fleet({1: 0, 2: 1})
    encoded = _encode_turn(fleet, chosen={})
    assert encoded[1].action_mask.tolist()[1] == 1.0  # robot 2 stands on vertex 1, still open
    assert not encoded[1].node_features[:, CLAIMED].any()


def test_communication_needs_every_robot_to_act():
    fleet = _fleet({1: 0, 2: 1})
    with pytest.raises(ValueError, match="communication"):
        _encode_turn(fleet, acting=[1], communication=True)
