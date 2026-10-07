"""Run the baseline: fixed storage (F_fix) + the centralised controller, no learning.

The M2 milestone's worked example. Demonstrates the milestone's own "done
when" criteria directly -- one episode runs end-to-end through the real
six-stage loop (environment/multi_agent_env.py) with zero collisions,
feasible storage, correct task-set disjointness, and bit-for-bit
deterministic replay -- not just a RunMetrics row at the end.

Run with `--render` for a live pygame preview of the real centralised
controller routing agents (requires the `[viz]` extra and a real
display) -- unlike examples/00b_pygame_preview.py, which could only show
a uniformly-random stepper because no controller existed yet when it was
written.
"""

import sys

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import FleetState, is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType, is_feasible
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.instances.generator import (
    GeneratedInstance,
    GeneratorParams,
    generate_and_validate,
    generate_instance,
)

FLEET_SIZE = 6


def _instance_params() -> GeneratorParams:
    return GeneratorParams(
        num_aisles=5,
        aisle_length=4,
        num_cross_aisles=3,
        one_way_fraction=0.0,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=8,
        num_delivery_vertices=3,
        num_endpoints=FLEET_SIZE,
        seed=1,
    )


def build_instance() -> WarehouseGraph:
    params = _instance_params()
    graph, report = generate_and_validate(params, fleet_size=FLEET_SIZE, require_well_formed=False)
    if not report.accepted:
        raise RuntimeError(f"Instance rejected: {report.model_dump_json(indent=2)}")
    return graph


def build_instance_with_positions() -> GeneratedInstance:
    # generate_instance and generate_and_validate share the same private
    # _build (test_generate_warehouse_graph_unchanged_by_refactor
    # guarantees they never drift apart), so calling both on the same
    # params is safe -- this one is only needed when rendering, to place
    # each vertex on screen.
    build_instance()  # re-validates; raises the same way if rejected
    return generate_instance(_instance_params())


def build_storage(
    graph: WarehouseGraph,
) -> tuple[dict[SkuId, SkuType], dict[VertexId, float], dict[SkuId, dict[VertexId, int]]]:
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {
        "tea": SkuType(sku_id="tea", unit_capacity=1.0),
        "coffee": SkuType(sku_id="coffee", unit_capacity=2.0),
    }
    capacities = {v: 20.0 for v in storage_vertices}
    counts = {
        "tea": {storage_vertices[0]: 8, storage_vertices[2]: 4},
        "coffee": {storage_vertices[1]: 5, storage_vertices[3]: 3},
    }
    return skus, capacities, counts


def build_config(seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        storage_mode="fixed",
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=FLEET_SIZE,
        # A naive prioritised planner (no deadlock detection/backtracking
        # -- sec:method:controllers describes nothing beyond "each agent
        # plans around already-reserved cells") can still occasionally
        # livelock two agents facing off over a single narrow vertex,
        # even after #55's fix for delivery/endpoint dead-ends. Kept
        # modest here so the demo actually shows completed tasks rather
        # than a saturated queue; see #55 and the class docstring in
        # controllers/centralised.py for the full story.
        arrival_rate=0.1,
        seed=seed,
        horizon=150,
        wait_cost=0.5,
    )


def _run_trace(
    graph: WarehouseGraph,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
) -> list[tuple[tuple[int, int], ...]]:
    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
    env.reset()
    trace = []
    for _ in range(config.horizon):
        obs, *_ = env.step()
        trace.append(tuple(sorted(obs.items())))
    return trace


def demonstrate_done_when_criteria(
    graph: WarehouseGraph,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
) -> None:
    print("=== Demonstrating M2's 'done when' criteria directly ===")
    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
    env.reset()
    before: FleetState = env.fleet
    for t in range(config.horizon):
        env.step()
        assert is_collision_free(before, env.fleet), f"collision at t={t}"
        assert is_feasible(env.storage), f"infeasible storage at t={t}"
        for task in env.tasks:
            if task.release_time <= t + 1:
                assert task.status(t + 1) in ("waiting", "active", "completed")
        before = env.fleet
    print(f"  zero collisions across {config.horizon} timesteps: OK")
    print("  storage stayed within capacity at every step: OK")
    print("  every task had exactly one lifecycle status at every step: OK")

    first = _run_trace(graph, skus, capacities, counts, config)
    second = _run_trace(graph, skus, capacities, counts, config)
    replay_ok = first == second
    status = "OK" if replay_ok else "FAILED"
    print(f"  bit-for-bit deterministic replay (seed={config.seed}): {status}")
    if not replay_ok:
        raise RuntimeError(
            "Replay determinism check failed -- see AGENTS.md's correctness invariant."
        )


def show_run_metrics(
    graph: WarehouseGraph,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
) -> None:
    print("\n=== RunMetrics for this run ===")
    metrics = run_episode(graph, skus, counts, capacities, config)
    print(metrics.model_dump_json(indent=2))


def render_live(
    graph: WarehouseGraph,
    positions: dict[VertexId, tuple[float, float]],
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
    *,
    delay_seconds: float = 0.15,
) -> None:
    """Watch the real centralised controller route agents live via
    viz.pygame_renderer.WarehouseRenderer. Local import: pygame is an
    optional [viz] extra, not a hard dependency of running this example
    for its metrics/invariants (the default `main()` path)."""
    import time

    from slap_mapd_coupling.viz.pygame_renderer import WarehouseRenderer

    env = WarehouseMAPDEnv(graph, skus, counts, capacities, config)
    env.reset()
    renderer = WarehouseRenderer(graph, positions)
    try:
        running = renderer.render(env.fleet)
        for _ in range(config.horizon):
            if not running:
                break
            env.step()
            running = renderer.render(env.fleet)
            time.sleep(delay_seconds)
    finally:
        renderer.close()


def main() -> None:
    render = "--render" in sys.argv[1:]

    if render:
        instance = build_instance_with_positions()
        graph = instance.graph
    else:
        graph = build_instance()
    skus, capacities, counts = build_storage(graph)
    config = build_config(seed=0)

    demonstrate_done_when_criteria(graph, skus, capacities, counts, config)
    show_run_metrics(graph, skus, capacities, counts, config)

    if render:
        print("\n=== Live pygame preview (close the window or Ctrl+C to stop) ===")
        render_live(graph, instance.positions, skus, capacities, counts, config)


if __name__ == "__main__":
    main()
