"""Unit tests for slap_mapd_coupling.environment.reward_function."""

import pytest

from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.reward_function import current_target, potential, reward


def _line_graph(n: int) -> WarehouseGraph:
    vertices = {i: Vertex(id=i) for i in range(n)}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0)
        for i in range(n - 1)
        for a, b in ((i, i + 1), (i + 1, i))
    )
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _task(pickup: int, delivery: int) -> Task:
    return Task(
        task_id=1, release_time=0, pickup_vertex=pickup, delivery_vertex=delivery, sku="tea"
    ).assign(agent_id=1, t=0)


def test_current_target_free_agent_is_own_location():
    assert current_target(location=3, task=None) == 3


def test_current_target_with_task_before_pickup_is_pickup_vertex():
    task = _task(pickup=4, delivery=0)
    assert current_target(location=1, task=task) == 4


def test_current_target_with_task_after_pickup_is_delivery_vertex():
    task = _task(pickup=4, delivery=0).pick_up(t=2)
    assert current_target(location=4, task=task) == 0


def test_potential_zero_for_free_agent_at_own_location():
    graph = _line_graph(5)
    assert potential(graph, location=2, task=None) == 0.0


def test_potential_is_negative_distance_to_goal():
    graph = _line_graph(5)
    task = _task(pickup=4, delivery=0)
    assert potential(graph, location=0, task=task) == pytest.approx(-4.0)


def test_reward_is_pure_shaping_with_no_other_terms():
    graph = _line_graph(5)
    task = _task(pickup=4, delivery=0)
    # Agent moves from 0 to 1, one step closer to pickup vertex 4.
    r = reward(
        graph,
        location_now=0,
        task_now=task,
        location_next=1,
        task_next=task,
        completed_task=False,
        overridden=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=1.0,
        congestion_reward_weight=1.0,
    )
    phi_now = -4.0  # d_G(0, 4)
    phi_next = -3.0  # d_G(1, 4)
    assert r == pytest.approx(0.9 * phi_next - phi_now)


def test_deliver_reward_added_on_completion():
    graph = _line_graph(5)
    base = dict(
        graph=graph,
        location_now=4,
        task_now=None,
        location_next=4,
        task_next=None,
        overridden=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=5.0,
        override_penalty=1.0,
        congestion_reward_weight=1.0,
    )
    with_delivery = reward(**base, completed_task=True)
    without_delivery = reward(**base, completed_task=False)
    assert with_delivery - without_delivery == pytest.approx(5.0)


def test_override_penalty_subtracted():
    graph = _line_graph(5)
    base = dict(
        graph=graph,
        location_now=4,
        task_now=None,
        location_next=4,
        task_next=None,
        completed_task=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=2.0,
        congestion_reward_weight=1.0,
    )
    overridden = reward(**base, overridden=True)
    not_overridden = reward(**base, overridden=False)
    assert not_overridden - overridden == pytest.approx(2.0)


def test_congestion_penalty_scales_with_weight_and_value():
    graph = _line_graph(5)
    base = dict(
        graph=graph,
        location_now=4,
        task_now=None,
        location_next=4,
        task_next=None,
        completed_task=False,
        overridden=False,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=1.0,
        congestion_reward_weight=3.0,
    )
    congested = reward(**base, congestion=0.5)
    uncongested = reward(**base, congestion=0.0)
    assert uncongested - congested == pytest.approx(1.5)
