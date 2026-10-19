"""Show the ExperimentConfig, StorageState and RunMetrics contracts end to end.

Numbered 01b (between 01_baseline_fixed_prioritised_astar.py and
02_storage_ladder.py) rather than renumbering anything: this script's
milestone -- "core config/storage types are real" -- sits strictly
between those two build steps.

Run with `--render` to draw the tea/coffee/mugs stock of shelves A, B and C on a generated
instance, with its delivery, receiving and parking vertices. The picture is saved to
examples/output/storage_and_config.png and shown in a window when a display is attached.
"""

import os
import sys

from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.evaluation.metrics import RunMetrics


def show_experiment_configs() -> None:
    print("=== ExperimentConfig: one cell per storage rule ===")
    # observation_depth (d_obs) is illustrative; the thesis leaves it TBD
    # by sweep (tab:rlparams), tracked in #85.
    illustrative_observation_depth = 2
    common = dict(
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=10,
        arrival_rate=1.0,
        seed=0,
        horizon=1000,
        wait_cost=0.2,
        observation_depth=illustrative_observation_depth,
    )

    fixed = ExperimentConfig(storage_mode="fixed", **common)
    demand = ExperimentConfig(
        storage_mode="demand", storage_epoch_length=50, reassignment_cap=20, **common
    )
    congestion = ExperimentConfig(
        storage_mode="congestion",
        storage_epoch_length=50,
        reassignment_cap=20,
        congestion_weight=0.3,
        **common,
    )
    for cfg in (fixed, demand, congestion):
        print(cfg.model_dump_json(indent=2))

    print("\n=== Deliberately invalid config: congestion storage without congestion_weight ===")
    try:
        ExperimentConfig(
            storage_mode="congestion", storage_epoch_length=50, reassignment_cap=20, **common
        )
    except ValidationError as exc:
        print(f"Rejected, as expected:\n{exc}")


def show_storage_state() -> StorageState:
    print("\n=== StorageState: the thesis's own tea/coffee/mugs worked example ===")
    skus = {
        "tea": SkuType(sku_id="tea", unit_capacity=1.0),
        "coffee": SkuType(sku_id="coffee", unit_capacity=2.0),
        "mugs": SkuType(sku_id="mugs", unit_capacity=1.0),
    }
    capacities = {1: 20.0, 2: 20.0, 3: 20.0}  # vertices A, B, C
    state = StorageState(
        skus=skus,
        capacities=capacities,
        counts={
            "tea": {1: 12, 3: 5},
            "coffee": {1: 4, 2: 3},
            "mugs": {2: 6, 3: 2},
        },
    )
    for vertex, cap in capacities.items():
        print(f"  used_capacity({vertex}) = {state.used_capacity(vertex)}  (cap = {cap})")

    print("\n=== Attempting an infeasible relocation (coffee at vertex 1 over capacity) ===")
    try:
        state.with_units("coffee", 1, count=17)  # 1*12 + 2*17 = 46 > 20
    except ValidationError as exc:
        print(f"Rejected, as expected:\n{exc}")

    return state


def show_task_coupling(state: StorageState) -> None:
    print("\n=== eq:coupling: a task's pickup vertex must actually hold the requested SKU ===")
    feasible = Task(task_id=1, release_time=5, pickup_vertex=1, delivery_vertex=99, sku="tea")
    infeasible = Task(task_id=2, release_time=5, pickup_vertex=1, delivery_vertex=99, sku="mugs")
    print(f"  tea at vertex 1 (x=12): is_pickup_feasible = {feasible.is_pickup_feasible(state)}")
    print(f"  mugs at vertex 1 (x=0): is_pickup_feasible = {infeasible.is_pickup_feasible(state)}")


def show_run_metrics_shape() -> None:
    print("\n=== RunMetrics: the shape of one results-table row ===")
    row = RunMetrics(
        storage_mode="fixed",
        controller="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=10,
        arrival_rate=1.0,
        seed=0,
        horizon=1000,
        observation_depth=3,
        num_completed_tasks=0,
        throughput=0.0,
        keep_up_ratio=0.0,
        num_waiting_tasks=3,
        num_active_tasks=0,
        backlog=3,
        num_storage_updates=0,
        mean_relocated_units=0.0,
        mean_relocation_distance=0.0,
        mean_decision_runtime_seconds=0.004,
        mean_crowding=0.0,
    )
    print(row.model_dump_json(indent=2))
    print(
        "Note: with num_completed_tasks=0, mean_service_time and its three parts "
        "(mean_wait_for_agent, mean_travel_time, mean_blocked_time) must stay None "
        "-- there is no finished pick to average over (eq:meanservice), and such a "
        "run fails the keep-up check (eq:throughput) anyway."
    )


def render_storage_state(state: StorageState) -> None:
    """The worked example's shelves A, B and C, placed on the first three storage vertices of
    a generated instance, drawn with their contents and capacities."""
    import matplotlib.pyplot as plt

    from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance
    from slap_mapd_coupling.viz.warehouse_plot import (
        assign_vertex_names,
        plot_graph,
        plot_node_names,
        plot_storage_capacity_list,
        plot_storage_contents,
    )

    params = GeneratorParams(
        max_num_aisles=7,
        max_aisle_length=7,
        max_num_cross_aisles=6,
        one_way_fraction=0.0,
        default_edge_cost=1.0,
        wait_cost=0.2,
        num_storage_vertices=3,
        num_delivery_vertices=4,
        # V_rcv on row 0, the inbound side, as many as the delivery vertices.
        num_receiving_vertices=4,
        num_endpoints=4,
        min_internal_role_vertices=2,
        seed=42,
    )
    instance = generate_instance(params)
    graph = instance.graph
    names = assign_vertex_names(graph)
    shelves = sorted(v for v in graph.vertices if graph.vertices[v].role.storage)
    # vertices 1, 2, 3 of the worked example are A, B, C, the shelves in vertex-id order
    on_instance = dict(zip(sorted(state.capacities), shelves))
    placed = StorageState(
        skus=state.skus,
        capacities={on_instance[v]: cap for v, cap in state.capacities.items()},
        counts={
            sku: {on_instance[v]: n for v, n in per_vertex.items()}
            for sku, per_vertex in state.counts.items()
        },
    )
    letters = {v: names[v] for v in shelves}

    ax = plot_graph(graph, instance.positions)
    plot_node_names(ax, graph, instance.positions, names)
    plot_storage_contents(ax, instance.positions, placed)
    plot_storage_capacity_list(ax, placed, letters)
    ax.set_title("The tea/coffee/mugs example on a generated instance")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "storage_and_config.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}.")
    plt.show()


def main() -> None:
    show_experiment_configs()
    state = show_storage_state()
    show_task_coupling(state)
    show_run_metrics_shape()
    if "--render" in sys.argv[1:]:
        render_storage_state(state)


if __name__ == "__main__":
    main()
