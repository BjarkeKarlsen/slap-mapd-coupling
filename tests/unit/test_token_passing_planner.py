"""Unit tests for slap_mapd_coupling.planners.token_passing."""

import importlib.util
from pathlib import Path

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState, is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.evaluation.trace import EpisodeTraceWriter, load_trace, summarise_trace
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_and_validate
from slap_mapd_coupling.planners import TokenPassingPlanner
from slap_mapd_coupling.planners.driver import step_with_planner

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TEA = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}


def _corridor() -> WarehouseGraph:
    """A hub (0) with three parking endpoints (1, 2, 3) and a delivery vertex (4) on it,
    and a dead-end corridor 0 - 5 - 6 that ends in the only storage vertex."""
    roles = {
        0: VertexRole(),
        1: VertexRole(endpoint=True),
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
    vertices = {v: Vertex(id=v, role=role) for v, role in roles.items()}
    return WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)


def _config(planner: str = "token_passing", **overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        planner=planner,
        congestion_sensitive=False,
        communication=False,
        num_agents=3,
        arrival_rate=0.3,
        seed=0,
        horizon=300,
        wait_cost=0.0,
        observation_depth=2,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _corridor_run(config: ExperimentConfig, **kwargs):
    return run_episode(_corridor(), TEA, {"tea": {6: 50}}, {6: 100.0}, config, **kwargs)


def _env(
    graph: WarehouseGraph, counts, capacities, config
) -> tuple[WarehouseMAPDEnv, TokenPassingPlanner]:
    planner = TokenPassingPlanner()
    return WarehouseMAPDEnv(graph, TEA, counts, capacities, config, assigner=planner), planner


def _check_run(
    env_and_planner: tuple[WarehouseMAPDEnv, TokenPassingPlanner], tmp_path: Path, horizon: int
) -> None:
    """Runs the episode and asserts what the plan promises of Token Passing."""
    env, planner = env_and_planner
    env.reset()
    path = tmp_path / "trace.jsonl"
    with EpisodeTraceWriter(path, env) as writer:
        for _ in range(horizon):
            before = env.fleet
            step_with_planner(env, planner)
            writer.record(env)
            assert not env.last_overridden, f"conflict resolution overrode a plan at t={env.t}"
            assert is_collision_free(before, env.fleet)
    assert planner.num_diverged == 0
    summary = summarise_trace(*load_trace(path))
    assert summary.stuck_agents == ()
    assert summary.completed > 0


def test_it_finishes_the_dead_end_corridor_where_prioritised_astar_stalls():
    # Three agents and one storage vertex in a dead end. Prioritised A* has agents that stand
    # still in the corridor, and completes one task in 300 steps. Token Passing keeps the
    # corridor clear, so almost every task released in that time is delivered.
    old = _corridor_run(_config("prioritised_astar"))
    new = _corridor_run(_config("token_passing"))
    assert old.num_completed_tasks <= 5
    assert new.num_completed_tasks >= 40


def test_it_never_needs_conflict_resolution_on_the_dead_end_corridor(tmp_path):
    _check_run(_env(_corridor(), {"tea": {6: 50}}, {6: 100.0}, _config()), tmp_path, 300)


def test_it_never_needs_conflict_resolution_on_the_example_04_instance(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_example_04", EXAMPLES / "04_decentralised_training.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    graph = example.build_instance()
    skus, capacities, counts = example.build_storage(graph)
    config = example.build_config("fixed_only", seed=0).model_copy(
        update={"controller": None, "planner": "token_passing"}
    )
    planner = TokenPassingPlanner()
    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config, assigner=planner)
    _check_run((env, planner), tmp_path, 500)


def test_it_never_needs_conflict_resolution_on_generated_well_formed_instances(tmp_path):
    for seed in (0, 1, 2):
        params = GeneratorParams(
            max_num_aisles=6,
            max_aisle_length=6,
            max_num_cross_aisles=5,
            one_way_fraction=0.0,
            wait_cost=0.0,
            num_storage_vertices=6,
            num_delivery_vertices=3,
            num_endpoints=4,
            seed=seed,
        )
        graph, report = generate_and_validate(params, fleet_size=4, require_well_formed=True)
        assert report.accepted
        storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
        counts = {"tea": {storage[0]: 20, storage[1]: 20}}
        capacities = {v: 50.0 for v in storage}
        config = _config(num_agents=4, seed=seed, arrival_rate=0.2, horizon=200)
        _check_run(_env(graph, counts, capacities, config), tmp_path, 200)


def test_replaying_a_seed_gives_the_same_trace_and_a_second_episode_starts_clean(tmp_path):
    config = _config(horizon=120)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _corridor_run(config, trace_path=first)
    _corridor_run(config, trace_path=second)
    assert first.read_bytes() == second.read_bytes()

    # The same environment run twice, with reset in between, gives the same episode.
    env, planner = _env(_corridor(), {"tea": {6: 50}}, {6: 100.0}, config)

    def episode() -> list[dict]:
        env.reset()
        out = []
        for _ in range(config.horizon):
            step_with_planner(env, planner)
            out.append(env.fleet.locations())
        return out

    assert episode() == episode()


def _at(graph: WarehouseGraph, **where: int) -> FleetState:
    return FleetState(
        agents={
            int(name[1:]): AgentState(agent_id=int(name[1:]), location=v)
            for name, v in where.items()
        }
    )


def _task(task_id: int, pickup: int, delivery: int) -> Task:
    return Task(
        task_id=task_id, release_time=0, pickup_vertex=pickup, delivery_vertex=delivery, sku="tea"
    )


def test_a_task_is_not_available_to_an_agent_while_another_agent_ends_on_its_pickup():
    graph, planner = _corridor(), TokenPassingPlanner()
    # Agent 2 stands on the pickup vertex, so the task is unavailable to agent 1, who asks
    # first, and agent 2 takes it. Agent 1 stays on its parking vertex.
    fleet = _at(graph, a1=1, a2=6)
    assert planner.assign(graph, fleet, [_task(1, 6, 4)], 0) == {1: 2}
    assert planner.plan(graph, fleet, [], 0)[1].kind == "wait"


def test_a_task_held_back_by_a_path_end_while_an_agent_is_idle_is_counted():
    graph, planner = _corridor(), TokenPassingPlanner()
    # Agent 2 takes task 1 and its path now ends on vertex 4. Task 2 also delivers to 4, so it
    # is unavailable to agent 1, who stays idle. Task 2 waited one step with its delivery taken.
    fleet = _at(graph, a1=1, a2=6)
    assert planner.assign(graph, fleet, [_task(1, 6, 4), _task(2, 6, 4)], 0) == {1: 2}
    assert planner.endpoint_wait == {2: (0, 1)}
    planner.reset(0)
    assert planner.endpoint_wait == {}


def test_nothing_is_counted_when_no_agent_is_idle():
    graph, planner = _corridor(), TokenPassingPlanner()
    fleet = _at(graph, a1=6)
    assert planner.assign(graph, fleet, [_task(1, 6, 4), _task(2, 6, 4)], 0) == {1: 1}
    assert planner.endpoint_wait == {}


def test_an_agent_on_a_delivery_vertex_of_a_waiting_task_moves_off_it():
    graph, planner = _corridor(), TokenPassingPlanner()
    # Agent 2 stands on the pickup, so the task is unavailable to agent 1. It delivers to the
    # vertex agent 1 stands on, so agent 1 goes to a parking endpoint (the nearest, 1, by
    # vertex id) through the hub instead of staying.
    fleet = _at(graph, a1=4, a2=6)
    assert planner.assign(graph, fleet, [_task(1, 6, 4)], 0) == {1: 2}
    assert planner.plan(graph, fleet, [], 0)[1] == Action(kind="move", target=0)


def test_an_idle_agent_on_a_vertex_no_waiting_task_needs_stays_where_it_is():
    graph, planner = _corridor(), TokenPassingPlanner()
    fleet = _at(graph, a1=1, a2=2)
    assert planner.assign(graph, fleet, [], 0) == {}
    assert all(action.kind == "wait" for action in planner.plan(graph, fleet, [], 0).values())


def test_a_planner_that_did_not_assign_waits():
    graph = _corridor()
    actions = TokenPassingPlanner().plan(graph, _at(graph, a1=1), [], 0)
    assert actions[1].kind == "wait"
