"""Unit tests for slap_mapd_coupling.environment.reward_function."""

import struct

import pytest

from slap_mapd_coupling.core.graph import Edge, Vertex, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.reward_function import (
    RewardTerms,
    current_target,
    potential,
    reward,
    reward_terms,
)


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
        moved=True,
        stood_still=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=1.0,
        move_cost=0.0,
        wait_cost=1.0,
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
        moved=False,
        stood_still=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=5.0,
        override_penalty=1.0,
        move_cost=1.0,
        wait_cost=1.0,
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
        moved=False,
        stood_still=False,
        congestion=0.0,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=2.0,
        move_cost=1.0,
        wait_cost=1.0,
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
        moved=False,
        stood_still=False,
        discount=0.9,
        deliver_reward=1.0,
        override_penalty=1.0,
        move_cost=1.0,
        wait_cost=1.0,
        congestion_reward_weight=3.0,
    )
    congested = reward(**base, congestion=0.5)
    uncongested = reward(**base, congestion=0.0)
    assert uncongested - congested == pytest.approx(1.5)


def _bits(x: float) -> bytes:
    return struct.pack(">d", x)


def _reference_reward(
    graph, *, location_now, task_now, location_next, task_next, completed_task, overridden,
    moved, stood_still, congestion, discount, deliver_reward, override_penalty, move_cost,
    wait_cost, congestion_reward_weight,
) -> float:  # fmt: skip
    """The reward as it was written before reward_terms existed, one inline sum."""
    phi_now = potential(graph, location_now, task_now)
    phi_next = potential(graph, location_next, task_next)
    shaping = discount * phi_next - phi_now
    return (
        shaping
        + deliver_reward * completed_task
        - override_penalty * overridden
        - move_cost * moved
        - wait_cost * stood_still
        - congestion_reward_weight * congestion
    )


def _cases(graph):
    towards = _task(pickup=4, delivery=0)
    carrying = _task(pickup=4, delivery=0).pick_up(t=2)
    base = dict(
        location_now=0,
        task_now=towards,
        location_next=1,
        task_next=towards,
        completed_task=False,
        overridden=False,
        moved=True,
        stood_still=False,
        congestion=0.0,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        move_cost=0.1,
        wait_cost=0.1,
        congestion_reward_weight=0.1,
    )
    return [
        base,
        # an override and a completion at the same step, with congestion
        {
            **base,
            "location_now": 3,
            "task_now": carrying,
            "location_next": 3,
            "task_next": None,
            "completed_task": True,
            "overridden": True,
            "moved": False,
            "congestion": 0.37,
        },
        {**base, "completed_task": True, "deliver_reward": 5.0, "congestion": 1 / 3},
        {**base, "overridden": True, "override_penalty": 2.0, "discount": 0.9},
        {
            **base,
            "location_now": 4,
            "task_now": None,
            "location_next": 4,
            "task_next": None,
            "moved": False,
        },
        {**base, "move_cost": 0.3, "discount": 0.97},
        {**base, "congestion": 0.1 + 0.2, "congestion_reward_weight": 3.0, "discount": 0.97},
        {**base, "overridden": True, "completed_task": True, "congestion": 0.8, "discount": 1.0},
        # standing still while serving a task, chosen or overridden
        {**base, "location_next": 0, "moved": False, "stood_still": True, "wait_cost": 0.3},
        {**base, "location_next": 0, "moved": False, "stood_still": True, "overridden": True},
    ]


def test_reward_terms_sum_to_the_reward_to_the_last_bit():
    graph = _line_graph(5)
    for case in _cases(graph):
        terms = reward_terms(graph, **case)
        expected = _reference_reward(graph, **case)
        assert _bits(terms.total) == _bits(expected)
        assert _bits(reward(graph, **case)) == _bits(expected)
        assert _bits(
            terms.shaping
            + terms.delivery
            + terms.override
            + terms.move
            + terms.wait
            + terms.congestion
        ) == _bits(expected)


def test_reward_terms_hold_each_term_with_its_sign():
    graph = _line_graph(5)
    case = _cases(graph)[1]  # override and completion at the same step
    terms = reward_terms(graph, **case)
    assert isinstance(terms, RewardTerms)
    assert terms.delivery == pytest.approx(1.0)
    assert terms.override == pytest.approx(-0.5)
    assert terms.move == 0.0  # the carrier stays on vertex 3
    assert terms.wait == 0.0  # it was overridden but the case does not mark it as standing still
    assert terms.congestion == pytest.approx(-0.1 * 0.37)
    # the carrier stands on the delivery vertex's neighbour and the task is done, so
    # phi_now = -d(3, 0) = -3 and phi_next = 0 (a free agent at its own vertex)
    assert terms.shaping == pytest.approx(0.99 * 0.0 + 3.0)


def test_reward_terms_are_zero_where_their_indicator_is_off():
    graph = _line_graph(5)
    case = {**_cases(graph)[0], "moved": False}
    terms = reward_terms(graph, **case)
    assert terms.move == 0.0
    assert terms.delivery == 0.0
    assert terms.override == 0.0
    assert terms.wait == 0.0
    assert terms.congestion == 0.0
    assert terms.shaping != 0.0


def test_standing_still_while_serving_costs_the_wait_cost():
    graph = _line_graph(5)
    case = {
        **_cases(graph)[0],
        "location_next": 0,
        "moved": False,
        "stood_still": True,
        "wait_cost": 0.3,
    }
    still = reward_terms(graph, **case)
    moving = reward_terms(graph, **{**case, "stood_still": False})
    assert still.wait == pytest.approx(-0.3)
    assert moving.wait == 0.0
    assert still.total - moving.total == pytest.approx(-0.3)


def test_an_override_costs_the_wait_cost_and_the_penalty_and_a_chosen_wait_only_the_first():
    graph = _line_graph(5)
    base = {**_cases(graph)[0], "location_next": 0, "moved": False, "stood_still": True}
    chosen = reward_terms(graph, **{**base, "overridden": False})
    forced = reward_terms(graph, **{**base, "overridden": True})
    assert chosen.total - forced.total == pytest.approx(0.5)  # the override penalty
    assert forced.wait == pytest.approx(chosen.wait) == pytest.approx(-0.1)
    assert forced.total < chosen.total  # yielding is never dearer than being overridden


def test_a_move_costs_the_move_cost_whether_or_not_the_agent_serves_a_task():
    graph = _line_graph(5)
    serving = reward_terms(graph, **_cases(graph)[0])
    free = {**_cases(graph)[0], "location_now": 2, "task_now": None, "location_next": 3}
    idle = reward_terms(graph, **{**free, "task_next": None})
    assert serving.move == pytest.approx(-0.1)
    assert idle.move == pytest.approx(-0.1)
    assert idle.wait == 0.0
