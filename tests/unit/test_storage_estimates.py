"""Unit tests for slap_mapd_coupling.environment.storage_estimates and its wiring into
the step loop (stage 6): the per-step estimates of the last storage period only."""

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.environment.storage_estimates import WindowEstimator
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.planners import PrioritisedAStarPlanner
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
        edges_before = dict(env.log.edge_traversals)
        step_with_planner(env, planner)
        after = env.fleet.locations()
        for agent_id, action in env.last_proposed.items():
            if action.kind != "move":
                continue
            edge = (before[agent_id], action.target)
            if after[agent_id] == action.target:
                traversals.append((step, edge))
            elif after[agent_id] == before[agent_id]:
                waits.append((step, edge))
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


def test_an_overridden_move_is_a_wait_on_that_edge_and_a_chosen_wait_is_none():
    graph = _graph()
    env = _env(_config(num_agents=2, storage_epoch_length=10, horizon=12), graph)
    target = next(v for v in graph.vertices if len(graph.out_neighbours(v)) >= 2)
    a, b = graph.out_neighbours(target)[:2]
    env.fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=a), 2: AgentState(agent_id=2, location=b)}
    )
    move_a = slot_for_action(graph, a, Action(kind="move", target=target))
    move_b = slot_for_action(graph, b, Action(kind="move", target=target))

    env.step({1: move_a, 2: move_b})  # both claim the same vertex, one is overridden

    traversed = env._estimator.traversals
    waited = env._estimator.waits
    assert sum(traversed.values()) == 1 and sum(waited.values()) == 1
    (winner_edge,) = traversed
    (loser_edge,) = waited
    assert {winner_edge, loser_edge} == {(a, target), (b, target)}
    assert len(env.last_overridden) == 1

    env.step({1: 0, 2: 0})  # both choose to wait, so they are attributed to no edge
    assert sum(env._estimator.waits.values()) == 1


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


def _counts(edges) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for edge in edges:
        counts[edge] = counts.get(edge, 0) + 1
    return counts


def _per_step(counts: dict[tuple[int, int], int], delta: int) -> dict[tuple[int, int], float]:
    return {edge: n / delta for edge, n in counts.items()}
