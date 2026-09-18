"""Live pygame preview of a warehouse instance with agents moving.

There is no controller yet (controllers/ is still a stub -- that's a
later milestone), so this cannot show real routing behaviour. What it
CAN show, using only already-implemented and tested code, is the
renderer itself: each frame, every agent picks a uniformly random legal
action (core.graph.WarehouseGraph.legal_actions), the joint move is
retried a few times if it isn't collision-free
(core.agents.has_vertex_conflict / has_swap_conflict), and the result is
drawn with viz.pygame_renderer.WarehouseRenderer. Swap `_random_step` for
a real controller once one exists; nothing else in this script changes.

Run with a real display attached; close the window (or Ctrl+C) to stop.
"""

import random
import time

from slap_mapd_coupling.core.agents import (
    AgentState,
    FleetState,
    has_swap_conflict,
    has_vertex_conflict,
)
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance
from slap_mapd_coupling.viz.pygame_renderer import WarehouseRenderer

MAX_RETRIES = 20  # give up and let everyone wait this frame if no collision-free draw is found


def _random_step(graph: WarehouseGraph, fleet: FleetState, rng: random.Random) -> FleetState:
    for _ in range(MAX_RETRIES):
        proposal: dict[int, AgentState] = {}
        for agent_id, state in fleet.agents.items():
            action = rng.choice(graph.legal_actions(state.location))
            target = state.location if action.kind == "wait" else action.target
            proposal[agent_id] = AgentState(agent_id=agent_id, location=target)
        candidate = FleetState(agents=proposal)
        if has_vertex_conflict(fleet, candidate) is None and has_swap_conflict(fleet, candidate) is None:
            return candidate
    return fleet  # no collision-free draw found this frame; stand still rather than force one through


def main() -> None:
    num_agents = 4

    params = GeneratorParams(
        num_aisles=5,
        aisle_length=4,
        num_cross_aisles=3,
        one_way_fraction=0.05,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=8,
        num_delivery_vertices=3,
        num_endpoints=num_agents,
        seed=1,
    )
    instance = generate_instance(params)
    graph = instance.graph

    start_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.endpoint][:num_agents]
    fleet = FleetState(
        agents={i: AgentState(agent_id=i, location=v) for i, v in enumerate(start_vertices)}
    )

    rng = random.Random(0)
    renderer = WarehouseRenderer(graph, instance.positions)
    try:
        running = True
        while running:
            running = renderer.render(fleet)
            time.sleep(0.15)
            fleet = _random_step(graph, fleet, rng)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
