"""Unit tests for slap_mapd_coupling.environment.storage_estimates and its wiring into
the step loop (stage 6): the per-step estimates of the last storage period only."""

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.environment.storage_estimates import WindowEstimator
from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_and_validate,
    generate_warehouse_graph,
)
from slap_mapd_coupling.planners import PrioritisedAStarPlanner, TokenPassingPlanner
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.storage.base import DemandEstimate, TrafficEstimate
from slap_mapd_coupling.storage.demand import ranked_storage_vertices


def _graph() -> WarehouseGraph:
    return generate_warehouse_graph(
        GeneratorParams(
            max_num_aisles=6,
            max_aisle_length=6,
            max_num_cross_aisles=5,
            one_way_fraction=0.0,
            wait_cost=0.0,
            num_storage_vertices=6,
            num_delivery_vertices=2,
            num_endpoints=4,
            seed=0,
        )
    )


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="demand",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=40,
        wait_cost=0.0,
        observation_depth=2,
        storage_epoch_length=10,
        reassignment_cap=3,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _env(config: ExperimentConfig, graph: WarehouseGraph | None = None) -> WarehouseMAPDEnv:
    graph = graph or _graph()
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {
        "tea": SkuType(sku_id="tea", unit_capacity=1.0),
        "mug": SkuType(sku_id="mug", unit_capacity=1.0),
    }
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5}, "mug": {storage_vertices[1]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config)


class _Calls:
    """Stands in for the storage rule and records what stage 6 hands it. It leaves the
    placement as it is."""

    def __init__(self) -> None:
        self.calls: list[tuple[DemandEstimate, TrafficEstimate, TrafficEstimate]] = []

    def __call__(self, x_prev, graph, cap, beta, demand, traversal, waiting):
        self.calls.append((demand, traversal, waiting))
        return x_prev


# --- the estimator on its own --------------------------------------------------------


def test_estimates_are_counts_per_step_of_the_window():
    estimator = WindowEstimator()
    for _ in range(30):
        estimator.record_order("tea")
    for _ in range(3):
        estimator.record_traversal((1, 2))
    estimator.record_wait((2, 3))
    estimator.record_wait((2, 3))

    estimates = estimator.close_window(100)

    assert estimates.demand == {"tea": 0.3}  # 30 orders in 100 steps
    assert estimates.traversal == {(1, 2): 0.03}
    assert estimates.waiting == {(2, 3): 0.02}
    assert estimates.num_orders == 30


def test_the_same_rate_over_a_longer_window_gives_the_same_estimate():
    short, long = WindowEstimator(), WindowEstimator()
    for _ in range(30):
        short.record_order("tea")
    for _ in range(60):
        long.record_order("tea")
    assert short.close_window(100).demand == long.close_window(200).demand


def test_the_next_window_starts_from_zero():
    estimator = WindowEstimator()
    estimator.record_order("tea")
    estimator.record_traversal((1, 2))
    estimator.record_wait((1, 2))
    estimator.close_window(10)

    empty = estimator.close_window(10)

    assert empty.demand == {} and empty.traversal == {} and empty.waiting == {}
    assert empty.num_orders == 0


def test_an_empty_window_reads_as_no_orders():
    assert WindowEstimator().close_window(5).num_orders == 0


def test_a_window_needs_a_positive_length():
    with pytest.raises(ValueError, match="window_length"):
        WindowEstimator().close_window(0)


# --- the wiring into stage 6 ---------------------------------------------------------


def test_the_rule_receives_the_per_step_counts_of_each_window_and_only_that_window():
    planner = PrioritisedAStarPlanner()
    config = _config(horizon=35)
    env = _env(config)
    rule = _Calls()
    env._storage_rule = rule
    delta = config.storage_epoch_length
    assert delta is not None

    # What happened at each step, recomputed from the public state of the env.
    traversals: list[tuple[int, tuple[int, int]]] = []  # (step, edge)
    waits: list[tuple[int, tuple[int, int]]] = []
    for _ in range(35):
        step = env.t
        before = env.fleet.locations()
        serving = active_tasks_by_agent(env.tasks, step)
        edges_before = dict(env.log.edge_traversals)
        step_with_planner(env, planner)
        after = env.fleet.locations()
        for agent_id, action in env.last_proposed.items():
            if action.kind == "move" and after[agent_id] == action.target:
                traversals.append((step, (before[agent_id], action.target)))
        # A robot serving a task that stood still waits for the next edge of its route.
        tasks_now = {task.task_id: task for task in env.tasks}
        for agent_id, task in serving.items():
            goal = tasks_now[task.task_id].current_goal
            if after[agent_id] == before[agent_id] != goal:
                waits.append((step, _next_edge(env.graph, before[agent_id], goal)))
        # The log of realised traversals agrees with what was recomputed above.
        logged = {
            e: n - edges_before.get(e, 0)
            for e, n in env.log.edge_traversals.items()
            if n != edges_before.get(e, 0)
        }
        assert logged == _counts(edge for s, edge in traversals if s == step)
    orders_released = [task.release_time for task in env.tasks]
    assert any(waits) and any(traversals)  # the run has contention, so waits are tested

    assert len(rule.calls) == 3  # epochs 10, 20, 30
    for index, (demand, traversal, waiting) in enumerate(rule.calls):
        start, end = index * delta, (index + 1) * delta  # the window [start, end)
        expected_orders = sum(1 for t in orders_released if start <= t < end)
        assert expected_orders > 0
        assert sum(demand.values()) == pytest.approx(expected_orders / delta)
        assert traversal == _per_step(_counts(e for s, e in traversals if start <= s < end), delta)
        assert waiting == _per_step(_counts(e for s, e in waits if start <= s < end), delta)


def test_the_demand_estimate_counts_the_orders_per_sku_of_the_window():
    config = _config(storage_epoch_length=10, horizon=12)
    env = _env(config)
    rule = _Calls()
    env._storage_rule = rule
    planner = PrioritisedAStarPlanner()
    for _ in range(10):
        step_with_planner(env, planner)

    demand, _, _ = rule.calls[0]
    released_in_window = [task for task in env.tasks if 0 <= task.release_time < 10]
    for sku in ("tea", "mug"):
        count = sum(1 for task in released_in_window if task.sku == sku)
        assert demand.get(sku, 0.0) == pytest.approx(count / 10)
    # An order released at the epoch itself (stage 1 of the step after it) is not in it.
    assert all(task.release_time <= 10 for task in env.tasks)


def test_estimates_are_causal_the_order_released_at_the_epoch_is_in_the_next_window():
    config = _config(storage_epoch_length=5, horizon=11, arrival_rate=3.0)
    env = _env(config)
    rule = _Calls()
    env._storage_rule = rule
    planner = PrioritisedAStarPlanner()
    for _ in range(10):
        step_with_planner(env, planner)

    first, second = rule.calls
    window_0 = sum(1 for task in env.tasks if 0 <= task.release_time < 5)
    window_1 = sum(1 for task in env.tasks if 5 <= task.release_time < 10)
    assert sum(first[0].values()) * 5 == pytest.approx(window_0)
    assert sum(second[0].values()) * 5 == pytest.approx(window_1)


def test_reset_starts_a_fresh_window():
    env = _env(_config(storage_epoch_length=10, horizon=12))
    rule = _Calls()
    env._storage_rule = rule
    planner = PrioritisedAStarPlanner()
    for _ in range(6):
        step_with_planner(env, planner)
    env.reset()
    env._storage_rule = rule
    for _ in range(10):
        step_with_planner(env, planner)

    demand, _, _ = rule.calls[0]
    released = sum(1 for task in env.tasks if task.release_time < 10)
    assert sum(demand.values()) * 10 == pytest.approx(released)  # nothing left from before


# --- which edge a wait is charged to --------------------------------------------------


def _diamond() -> WarehouseGraph:
    """Two routes of equal cost, 0 - 1 - 3 and 0 - 2 - 3, from a hub (0) to the storage
    vertex (3). The delivery vertex (4) and three parking endpoints (5, 6, 7) hang off the
    hub. Every edge costs 1 and goes both ways."""
    roles = {
        0: VertexRole(),
        1: VertexRole(),
        2: VertexRole(),
        3: VertexRole(storage=True),
        4: VertexRole(delivery=True),
        5: VertexRole(endpoint=True),
        6: VertexRole(endpoint=True),
        7: VertexRole(endpoint=True),
    }
    pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    edges = tuple(Edge(source=a, target=b, cost=1.0) for a, b in pairs) + tuple(
        Edge(source=b, target=a, cost=1.0) for a, b in pairs
    )
    vertices = {v: Vertex(id=v, role=role) for v, role in roles.items()}
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _diamond_env(
    locations: dict[int, int], tasks: list[Task], **overrides
) -> tuple[WarehouseMAPDEnv, _Calls]:
    """An env on the diamond with the robots at `locations` and only `tasks`, released and
    assigned at t = 0. No order arrives during the run, and the storage rule is a `_Calls`."""
    graph = _diamond()
    settings = dict(num_agents=len(locations), arrival_rate=1e-9, storage_epoch_length=10)
    settings.update(overrides)
    env = WarehouseMAPDEnv(
        graph,
        {"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        {"tea": {3: 5}},
        {3: 10.0},
        _config(**settings),
    )
    env.fleet = FleetState(
        agents={aid: AgentState(agent_id=aid, location=v) for aid, v in locations.items()}
    )
    env.tasks = tasks
    rule = _Calls()
    env._storage_rule = rule
    return env, rule


def _task(task_id: int, agent_id: int, picked_up: bool = False) -> Task:
    """A task for tea from the storage vertex (3) to the delivery vertex (4), assigned to
    `agent_id` at t = 0 and, if `picked_up`, picked up at t = 0 too."""
    return Task(
        task_id=task_id,
        release_time=0,
        pickup_vertex=3,
        delivery_vertex=4,
        sku="tea",
        assigned_agent=agent_id,
        assignment_time=0,
        pickup_time=0 if picked_up else None,
    )


def _move(env: WarehouseMAPDEnv, agent_id: int, target: int) -> int:
    at = env.fleet.locations()[agent_id]
    return slot_for_action(env.graph, at, Action(kind="move", target=target))


def test_the_unit_estimate_counts_units_and_the_demand_estimate_counts_lines():
    estimator = WindowEstimator()
    estimator.record_order("tea", 3)
    estimator.record_order("tea", 5)
    estimator.record_order("mugs")
    estimates = estimator.close_window(10)
    assert estimates.demand == {"tea": 0.2, "mugs": 0.1}
    assert estimates.units == {"tea": 0.8, "mugs": 0.1}
    assert estimator.close_window(10).units == {}


def test_a_chosen_wait_is_charged_to_the_next_edge_of_the_shortest_route():
    # Before pickup the goal is the storage vertex. Both 1 and 2 start a shortest route
    # from the hub, so the tie goes to the lower id.
    env, _ = _diamond_env({1: 0}, [_task(1, 1)])
    env.step({1: 0})
    assert env._estimator.waits == {(0, 1): 1}
    assert env.log.blocked_ticks_by_task == {1: 1}


def test_a_move_ends_a_run_of_blocked_ticks_and_the_longest_run_is_kept():
    # Blocked for 3 ticks, one move from the hub to 1, then blocked for 5 ticks.
    env, _ = _diamond_env({1: 0}, [_task(1, 1)])
    for _ in range(3):
        env.step({1: 0})
    env.step({1: _move(env, 1, 1)})
    for _ in range(5):
        env.step({1: 0})
    assert env.log.blocked_ticks_by_task == {1: 8}
    assert env.log.longest_blocked_run_by_task == {1: 5}


def test_after_pickup_the_wait_is_charged_towards_the_delivery_vertex():
    env, _ = _diamond_env({1: 0}, [_task(1, 1, picked_up=True)])
    env.step({1: 0})
    assert env._estimator.waits == {(0, 4): 1}


def test_a_robot_that_picks_up_while_standing_still_waits_towards_the_delivery_vertex():
    # The robot stands on the storage vertex and picks up in this step's lifecycle update,
    # so its goal is already the delivery vertex. From 3 both 1 and 2 lead there, 1 wins.
    env, _ = _diamond_env({1: 3}, [_task(1, 1)])
    env.step({1: 0})
    assert env.tasks[0].pickup_time == 1
    assert env._estimator.waits == {(3, 1): 1}


def test_a_robot_standing_at_its_goal_is_not_charged():
    # Picked up and standing on the delivery vertex, so its goal is where it is.
    env, _ = _diamond_env({1: 4}, [_task(1, 1, picked_up=True)])
    env.step({1: 0})
    assert env.tasks[0].finish_time == 1
    assert env._estimator.waits == {}


def test_an_idle_robot_is_not_charged_whether_it_chose_to_wait_or_was_overridden():
    # Neither robot serves a task. Both claim the hub, one is overridden, then both wait.
    env, _ = _diamond_env({1: 1, 2: 2}, [])
    env.step({1: _move(env, 1, 0), 2: _move(env, 2, 0)})
    assert len(env.last_overridden) == 1
    env.step({1: 0, 2: 0})
    assert env._estimator.waits == {}


def test_an_overridden_robot_is_charged_by_the_route_rule_not_on_its_intended_edge():
    # Robot 1 at the hub, heading for storage, tries the detour through 2. Robot 2 on the
    # storage vertex, heading for delivery, also claims 2 and goes first. Robot 1 is
    # overridden and waits for (0, 1), the first edge of its shortest route, not for (0, 2).
    env, _ = _diamond_env({1: 0, 2: 3}, [_task(1, 1), _task(2, 2, picked_up=True)])
    env._priority = (2, 1)
    env.step({1: _move(env, 1, 2), 2: _move(env, 2, 2)})
    assert env.last_overridden == frozenset({1})
    assert env._estimator.waits == {(0, 1): 1}


def test_a_wait_lands_in_the_window_of_the_step_it_happens_in():
    # Delta = 2. The robot moves at step 0 and waits at step 1, the last step of the first
    # window, so the wait belongs to that window. Steps 2 and 3 are moves, so the second
    # window has none.
    env, rule = _diamond_env({1: 5}, [_task(1, 1)], storage_epoch_length=2)
    env.step({1: _move(env, 1, 0)})
    env.step({1: 0})
    env.step({1: _move(env, 1, 1)})
    env.step({1: _move(env, 1, 3)})
    (_, _, first), (_, _, second) = rule.calls
    assert first == {(0, 1): 0.5}
    assert second == {}


def test_token_passing_is_never_overridden_yet_its_waits_reach_the_estimate():
    # Token Passing makes robots wait on purpose, so conflict resolution never steps in and
    # charging only overrides would leave the wait estimate at zero. Charged by the route
    # rule, its waits show in the estimate.
    params = GeneratorParams(
        max_num_aisles=6,
        max_aisle_length=6,
        max_num_cross_aisles=5,
        one_way_fraction=0.0,
        wait_cost=0.0,
        num_storage_vertices=6,
        num_delivery_vertices=3,
        num_endpoints=4,
        seed=1,
    )
    graph, report = generate_and_validate(params, fleet_size=4, require_well_formed=True)
    assert report.accepted
    storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    planner = TokenPassingPlanner()
    config = _config(
        planner="token_passing", arrival_rate=0.6, seed=1, horizon=100, storage_epoch_length=25
    )
    env = WarehouseMAPDEnv(
        graph,
        {"tea": SkuType(sku_id="tea", unit_capacity=1.0)},
        {"tea": {storage[0]: 20, storage[1]: 20}},
        {v: 50.0 for v in storage},
        config,
        assigner=planner,
    )
    rule = _Calls()
    env._storage_rule = rule
    for _ in range(100):
        step_with_planner(env, planner)
        assert not env.last_overridden
    assert sum(env.log.blocked_ticks_by_task.values()) > 0  # the robots did delay
    assert len(rule.calls) == 4
    assert any(sum(waiting.values()) > 0 for _, _, waiting in rule.calls)


def test_a_run_under_fixed_storage_runs_no_storage_update():
    config = _config(storage_mode="fixed", storage_epoch_length=None, reassignment_cap=None)
    env = _env(config)
    planner = PrioritisedAStarPlanner()
    for _ in range(15):
        step_with_planner(env, planner)
    assert env.log.relocated_units_by_update == []


def test_the_estimates_replay_bit_for_bit_under_a_fixed_seed():
    def run() -> list[tuple[DemandEstimate, TrafficEstimate, TrafficEstimate]]:
        env = _env(_config(horizon=30))
        rule = _Calls()
        env._storage_rule = rule
        planner = PrioritisedAStarPlanner()
        for _ in range(30):
            step_with_planner(env, planner)
        return rule.calls

    assert run() == run()


# --- skipping the update when the window had no orders -------------------------------


def _env_with_stock_far_from_delivery(config: ExperimentConfig) -> WarehouseMAPDEnv:
    graph = _graph()
    storage_vertices = ranked_storage_vertices(graph)  # nearest the delivery points first
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[-1]: 6}}  # all of it at the farthest vertex
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config)


def _rule_config(storage_mode: str, **overrides) -> ExperimentConfig:
    extra = {"congestion_weight": 1.0} if storage_mode == "congestion" else {}
    return _config(storage_mode=storage_mode, **extra, **overrides)


@pytest.mark.parametrize("storage_mode", ["demand", "congestion"])
def test_no_order_in_the_window_leaves_the_placement_and_logs_no_relocation(storage_mode):
    # An arrival rate this small releases no order in the whole run.
    config = _rule_config(storage_mode, arrival_rate=1e-9, horizon=25)
    env = _env_with_stock_far_from_delivery(config)
    initial = {k: dict(v) for k, v in env.storage.counts.items()}
    planner = PrioritisedAStarPlanner()
    for _ in range(25):
        step_with_planner(env, planner)

    assert env.tasks == []
    assert env.storage.counts == initial  # x_t = x_{t-Delta} at both epochs
    assert env.log.relocated_units_by_update == [0, 0]  # nothing relocated at epochs 10, 20


@pytest.mark.parametrize("storage_mode", ["demand", "congestion"])
def test_an_epoch_with_orders_still_runs_the_rule(storage_mode):
    config = _rule_config(storage_mode, arrival_rate=1.0, horizon=12)
    env = _env_with_stock_far_from_delivery(config)
    initial = {k: dict(v) for k, v in env.storage.counts.items()}
    planner = PrioritisedAStarPlanner()
    for _ in range(10):
        step_with_planner(env, planner)

    assert env.tasks  # orders were released in the window
    (moved,) = env.log.relocated_units_by_update
    assert 0 < moved <= 3  # the cap nu
    assert env.storage.counts != initial


# --- helpers -------------------------------------------------------------------------


def _next_edge(graph: WarehouseGraph, at: int, goal: int) -> tuple[int, int]:
    """The first edge of a shortest route from `at` to `goal`, ties by the lower vertex id."""
    cost = {(e.source, e.target): e.cost for e in graph.edges}
    best = min(graph.out_neighbours(at), key=lambda w: (cost[(at, w)] + graph.distance(w, goal), w))
    return (at, best)


def _counts(edges) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for edge in edges:
        counts[edge] = counts.get(edge, 0) + 1
    return counts


def _per_step(counts: dict[tuple[int, int], int], delta: int) -> dict[tuple[int, int], float]:
    return {edge: n / delta for edge, n in counts.items()}
