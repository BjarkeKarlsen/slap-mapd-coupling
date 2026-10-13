"""Unit tests for slap_mapd_coupling.evaluation.trace."""

import json
from pathlib import Path

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.evaluation.evaluator import _to_run_metrics
from slap_mapd_coupling.evaluation.trace import (
    EpisodeTraceWriter,
    format_summary,
    load_trace,
    main,
    summarise_trace,
)
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph
from slap_mapd_coupling.routing import get_router


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=7,
        horizon=20,
        wait_cost=0.0,
        observation_depth=2,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _generated_env(config: ExperimentConfig) -> WarehouseMAPDEnv:
    graph = generate_warehouse_graph(
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
    storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    counts = {"tea": {storage[0]: 5, storage[1]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, {v: 10.0 for v in storage}, config)


def _hub_env() -> WarehouseMAPDEnv:
    """Vertex 0 is a hub. Vertex 1 is a dead end that only leads back to it."""
    roles = {
        0: VertexRole(),
        1: VertexRole(endpoint=True),
        2: VertexRole(endpoint=True),
        3: VertexRole(storage=True),
        4: VertexRole(delivery=True),
    }
    vertices = {v: Vertex(id=v, role=role) for v, role in roles.items()}
    edges = tuple(
        Edge(source=a, target=b, cost=1.0) for v in (1, 2, 3, 4) for a, b in ((0, v), (v, 0))
    )
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    config = _config(num_agents=2, horizon=5)
    return WarehouseMAPDEnv(graph, skus, {"tea": {3: 5}}, {3: 10.0}, config)


def _run_with_trace(config: ExperimentConfig, path: Path, steps: int) -> WarehouseMAPDEnv:
    env = _generated_env(config)
    env.reset()
    with EpisodeTraceWriter(path, env) as writer:
        for _ in range(steps):
            env.step()
            writer.record(env)
    return env


def test_trace_has_a_header_then_one_line_per_step(tmp_path):
    config = _config()
    _run_with_trace(config, tmp_path / "trace.jsonl", steps=6)
    header, steps = load_trace(tmp_path / "trace.jsonl")
    assert len(steps) == 6
    assert (tmp_path / "trace.jsonl").read_text().count("\n") == 7
    assert header["config"] == config.model_dump(mode="json")
    assert set(header["initial"]) == {"1", "2", "3", "4"}
    assert {v["id"] for v in header["vertices"]} >= set(header["initial"].values())
    assert all(len(edge) == 3 for edge in header["edges"])
    assert [s["t"] for s in steps] == [1, 2, 3, 4, 5, 6]
    assert set(steps[0]["agents"]["1"]) == {
        "at",
        "intent",
        "moved",
        "overridden",
        "task",
        "phase",
        "goal",
    }


def test_head_on_pair_is_recorded_as_two_blocked_intents(tmp_path):
    env = _hub_env()
    env.reset()
    # Agent 1 stands in the dead end and agent 2 on the hub, each wants the other's vertex.
    env.fleet = FleetState(
        agents={1: AgentState(agent_id=1, location=1), 2: AgentState(agent_id=2, location=0)}
    )
    swap = {
        1: slot_for_action(env.graph, 1, Action(kind="move", target=0)),
        2: slot_for_action(env.graph, 0, Action(kind="move", target=1)),
    }
    with EpisodeTraceWriter(tmp_path / "trace.jsonl", env) as writer:
        env.step(swap)
        writer.record(env)
    _, steps = load_trace(tmp_path / "trace.jsonl")
    one, two = steps[0]["agents"]["1"], steps[0]["agents"]["2"]
    assert (one["intent"], two["intent"]) == (0, 1)
    assert (one["moved"], two["moved"]) == (False, False)
    assert (one["at"], two["at"]) == (1, 0)
    assert one["overridden"] and two["overridden"]


def test_summary_flags_stuck_agents_and_the_waiting_pair_from_a_file_alone(tmp_path):
    header = {
        "type": "header",
        "format": 1,
        "edges": [[1, 0, 1.0], [0, 1, 1.0], [0, 2, 1.0], [2, 0, 1.0]],
    }

    def agent(at, moved, intent, task, goal):
        return {
            "at": at,
            "intent": intent,
            "moved": moved,
            "overridden": intent is not None and not moved,
            "task": task,
            "phase": None if task is None else "to_pickup",
            "goal": goal,
        }

    lines = [header]
    for t in range(1, 9):
        lines.append(
            {
                "type": "step",
                "t": t,
                "completed": 1 if t >= 2 else 0,
                "released": 4,
                "backlog": 3,
                "agents": {
                    "1": agent(1, False, 0, 5, 9),
                    "2": agent(0, False, 1, 6, 9),
                    "3": agent(2, t < 3, 0 if t < 3 else None, None, None),
                },
            }
        )
    path = tmp_path / "hand.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    loaded_header, steps = load_trace(path)
    summary = summarise_trace(loaded_header, steps, window=4)
    assert summary.steps == 8
    assert summary.window == 4
    assert summary.completed == 1
    assert summary.last_completion_t == 2
    assert summary.stuck_agents == (1, 2)
    assert summary.waiting_on == ((1, 1, 0, 2), (2, 0, 1, 1))
    assert summary.exits == {1: (0,), 0: (1, 2)}
    assert summarise_trace(loaded_header, steps).window == 2
    assert "STUCK" in format_summary(summary)


def test_summary_flags_an_agent_that_moves_but_does_not_close_in_on_its_goal(tmp_path):
    # Vertices 1 - 0 - 2 on a line, goal 2. Agent 1 shuffles between 0 and 1, so half of
    # its moves go away from the goal. Agent 2 walks straight to it.
    header = {
        "type": "header",
        "format": 1,
        "initial": {"1": 0, "2": 1},
        "edges": [[1, 0, 1.0], [0, 1, 1.0], [0, 2, 1.0], [2, 0, 1.0]],
    }

    def agent(at, moved):
        return {
            "at": at,
            "intent": None,
            "moved": moved,
            "overridden": False,
            "task": 1,
            "phase": "to_pickup",
            "goal": 2,
        }

    shuffle, walk = [1, 0, 1, 0], [(0, True), (2, True), (2, False), (2, False)]
    steps = [
        {
            "type": "step",
            "t": t + 1,
            "completed": 0,
            "released": 2,
            "backlog": 2,
            "agents": {"1": agent(shuffle[t], True), "2": agent(*walk[t])},
        }
        for t in range(4)
    ]
    summary = summarise_trace(header, steps, window=4)
    shuffler, walker = summary.agents
    assert (shuffler.goal_moves_in_window, shuffler.progress_in_window) == (4, 0.5)
    assert shuffler.distinct_vertices_in_window == 2
    assert (walker.goal_moves_in_window, walker.progress_in_window) == (2, 1.0)
    assert summary.wandering_agents == ()
    strict = summarise_trace(header, steps, window=4, min_progress=0.6)
    assert strict.wandering_agents == (1,)
    assert "WANDERING" in format_summary(strict)


def test_cli_reports_the_fraction_of_episodes_with_a_stuck_agent(tmp_path, capsys):
    paths = []
    for name, moves in (("a", False), ("b", True)):
        line = {
            "type": "step",
            "t": 1,
            "completed": 0,
            "released": 1,
            "backlog": 1,
            "agents": {
                "1": {
                    "at": 1,
                    "intent": None,
                    "moved": moves,
                    "overridden": False,
                    "task": 1,
                    "phase": "to_pickup",
                    "goal": 2,
                }
            },
        }
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            json.dumps({"type": "header", "format": 1, "edges": []})
            + "\n"
            + json.dumps(line)
            + "\n"
        )
        paths.append(str(path))
    main([*paths, "--window", "1"])
    out = capsys.readouterr().out
    assert "1 of 2 episodes had a stuck agent (50%)" in out
    assert "0 of 2 episodes had a wandering agent (0%)" in out


def test_same_seed_gives_byte_identical_files(tmp_path):
    config = _config(seed=13)
    _run_with_trace(config, tmp_path / "one.jsonl", steps=15)
    _run_with_trace(config, tmp_path / "two.jsonl", steps=15)
    assert (tmp_path / "one.jsonl").read_bytes() == (tmp_path / "two.jsonl").read_bytes()


def test_writing_a_trace_does_not_change_the_run_metrics(tmp_path):
    config = _config(seed=5, horizon=25)

    def run(path: Path | None):
        get_router(config).reset(config.seed)
        env = _generated_env(config)
        env.reset()
        writer = None if path is None else EpisodeTraceWriter(path, env)
        for _ in range(config.horizon):
            env.step()
            if writer is not None:
                writer.record(env)
        if writer is not None:
            writer.close()
        # Decision runtime is wall-clock time, so it differs between any two runs.
        return _to_run_metrics(env, config).model_dump(exclude={"mean_decision_runtime_seconds"})

    assert run(None) == run(tmp_path / "trace.jsonl")
