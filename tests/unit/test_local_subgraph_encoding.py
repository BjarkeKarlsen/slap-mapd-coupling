"""Unit tests for slap_mapd_coupling.models.local_subgraph_encoding."""

import numpy as np
import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.observation import build_observation
from slap_mapd_coupling.environment.spaces import legality_mask, max_out_degree
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM, SENDER_KEYS
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    encode_all_observations,
    encode_observation,
    encode_view,
    encoded_observation_space,
)


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
        ],
    )
    assert encoded.node_mask[encoded.own_index] == 1.0


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


def test_encode_observation_truncates_nodes_keeping_own_vertex_and_closest_eta():
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

    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=4, max_messages=1)
    encoded = encode_observation(graph, 5, observation, mask, d_max, encoding_config)

    assert encoded.node_mask.sum() == 4
    assert encoded.node_mask[encoded.own_index] == 1.0
    # target is vertex 10 (eta 0 there); closest-by-eta among the rest are
    # 10, 9, 8 (eta 0, 1, 2) -- the three smallest etas other than the
    # agent's own vertex (5, eta 5, which is force-kept regardless).
    kept_etas = sorted(
        float(encoded.node_features[i][4]) for i in range(4) if i != encoded.own_index
    )
    assert kept_etas == [0.0, 1.0, 2.0]
    # edges to dropped vertices are left out, and every row stays inside the kept vertices
    valid_rows = encoded.edge_index[encoded.edge_mask == 1.0]
    assert (valid_rows < 4).all()
    assert len(valid_rows) == 2  # kept vertices 5, 8, 9, 10 leave only 8-9 and 9-10


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


def _encode_all(fleet, communication, max_messages=2, depth=3, num_vertices=10):
    graph = _line_graph(num_vertices)
    encoding_config = LocalSubgraphEncodingConfig(max_local_nodes=8, max_messages=max_messages)
    config = _config(observation_depth=depth, communication=communication)
    encoded = encode_all_observations(
        graph, fleet, [], 0, config, encoding_config, max_out_degree(graph)
    )
    return graph, encoding_config, encoded


# agent 1 at vertex 4 hears agent 5 (distance 1), agents 2 and 3 (distance 2) and agent 4
# (distance 3), in that order
_FLEET_LOCATIONS = {1: 4, 4: 7, 3: 2, 2: 6, 5: 5}


@pytest.mark.parametrize("communication", [False, True])
def test_encoded_observation_space_matches_the_encoded_arrays(communication):
    graph, encoding_config, encoded = _encode_all(_fleet(_FLEET_LOCATIONS), communication)
    space = encoded_observation_space(encoding_config, max_out_degree(graph), communication)

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
    assert on_keys == off_keys | set(SENDER_KEYS)
    assert off[1].sender_mask is None


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
