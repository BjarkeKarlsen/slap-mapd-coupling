"""Unit tests for slap_mapd_coupling.viz.pygame_renderer.

pygame is an optional extra ([viz] in pyproject.toml) -- skip this whole
module if it isn't installed, rather than failing the suite.
"""

import os

import pytest

pygame = pytest.importorskip("pygame")

# Standard headless-testing trick for SDL/pygame: no real display needed,
# works in CI. Must be set before pygame.init() is ever called.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.graph import Edge, Vertex, VertexRole, WarehouseGraph
from slap_mapd_coupling.viz.pygame_renderer import (
    PANELS,
    VIEWS,
    TraceReplay,
    WarehouseRenderer,
    congestion,
)


def test_pygame_renderer_smoke():
    vertices = {
        1: Vertex(id=1, role=VertexRole(movable=True, storage=True)),
        2: Vertex(id=2, role=VertexRole(movable=True, delivery=True)),
    }
    edges = (Edge(source=1, target=2, cost=1.0), Edge(source=2, target=1, cost=1.0))
    graph = WarehouseGraph(vertices=vertices, edges=edges, wait_cost=0.0)
    positions = {1: (0.0, 0.0), 2: (1.0, 0.0)}
    fleet = FleetState(agents={1: AgentState(agent_id=1, location=1)})

    renderer = WarehouseRenderer(graph, positions, window_size=(200, 200))
    try:
        renderer.render(fleet)
    finally:
        renderer.close()


def test_a_trace_replays_with_every_layer_on_and_off(tmp_path):
    import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
    from slap_mapd_coupling.core.experiment_config import ExperimentConfig
    from slap_mapd_coupling.core.storage_state import SkuType
    from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
    from slap_mapd_coupling.evaluation.trace import EpisodeTraceWriter
    from slap_mapd_coupling.instances.generator import (
        GeneratorParams,
        fleet_defaults,
        generate_instance,
    )
    from slap_mapd_coupling.planners import PrioritisedAStarPlanner
    from slap_mapd_coupling.planners.driver import step_with_planner

    instance = generate_instance(GeneratorParams(**fleet_defaults(4), wait_cost=0.0))
    graph = instance.graph
    storage = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    config = ExperimentConfig(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=4,
        arrival_rate=1.0,
        seed=1,
        horizon=15,
        wait_cost=0.0,
        observation_depth=2,
    )
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    env = WarehouseMAPDEnv(
        graph, skus, {"tea": {storage[0]: 5}}, {v: 10.0 for v in storage}, config
    )
    env.reset()
    path = tmp_path / "trace.jsonl"
    planner = PrioritisedAStarPlanner()
    with EpisodeTraceWriter(path, env, positions=instance.positions, stock_every=5) as writer:
        for _ in range(15):
            step_with_planner(env, planner)
            writer.record(env)

    data = TraceReplay(path)
    renderer = WarehouseRenderer(None, header=data.header, window_size=(640, 480))
    try:
        for index in (0, 7, 14):
            step = data.steps[index]
            renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(index))
        for view in VIEWS:
            for panel in PANELS:
                renderer.view, renderer.panel = view, panel
                renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(14))
        renderer.selected = ("agent", "1")
        renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(14))
        renderer.selected = ("vertex", storage[0])
        renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(14))
        # A live run keeps its steps, so a pause can step back through them.
        assert renderer.render_env(env) is True
        assert renderer.render_env(env) is True
        assert len(renderer._live.steps) == 1  # the same step twice is a new episode
        assert renderer._step_action("back", 5, 9) == 4 and renderer.paused
        assert renderer._step_action("forward10", 5, 9) == 9
        assert renderer._step_action("seek:0.5", 0, 10) == 5
        renderer._step_action("pause", 5, 9)
        assert not renderer.paused
        # Zoom keeps the point under the mouse in place, and a reset undoes it.
        at = renderer._map_rect().center
        before = renderer._pixels[storage[0]]
        renderer._zoom_at(2.0, at)
        assert renderer.zoom == 2.0 and renderer._pixels[storage[0]] != before
        renderer.reset_view()
        assert renderer._pixels[storage[0]] == before
        for title in ("doing", "goal", "still", "agent"):
            renderer.sort, renderer.panel = title, "agents"
            renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(14))
        renderer.help = True
        renderer.draw_frame(step, data.stock_at(step["t"]), data.header, data.history(14))
    finally:
        renderer.close()


def test_congestion_counts_still_steps_and_names_the_blocker():
    def agent(at, moved, intent=None, task=1):
        return {"at": at, "moved": moved, "intent": intent, "task": task}

    history = [
        {"agents": {"1": agent(5, True), "2": agent(6, True)}},
        {"agents": {"1": agent(5, False, intent=6), "2": agent(6, False)}},
        {"agents": {"1": agent(5, False, intent=6), "2": agent(6, False)}},
    ]
    debug = congestion(history)
    assert debug["still_for"] == {"1": 2, "2": 2}
    assert debug["heat"] == {5: 2, 6: 2}
    assert debug["blocked"] == {"1"}
    assert debug["blocked_by"] == {"1": "2"}
    assert debug["fleet_still"] == 2
