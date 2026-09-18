"""Show the ExperimentConfig, StorageState and RunMetrics contracts end to end.

Numbered 01b (between 01_baseline_fixed_centralised.py and
02_storage_ladder.py) rather than renumbering anything: this script's
milestone -- "core config/storage types are real" -- sits strictly
between those two build steps.
"""

from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.storage_state import SkuType, StorageState
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.evaluation.metrics import RunMetrics


def show_experiment_configs() -> None:
    print("=== ExperimentConfig: one cell per storage rule ===")
    common = dict(
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=10,
        arrival_rate=1.0,
        seed=0,
        horizon=1000,
        wait_cost=0.5,
    )

    fixed = ExperimentConfig(storage_mode="fixed", **common)
    demand = ExperimentConfig(storage_mode="demand", storage_epoch_length=50, reassignment_cap=20, **common)
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
        ExperimentConfig(storage_mode="congestion", storage_epoch_length=50, reassignment_cap=20, **common)
    except ValidationError as exc:
        print(f"Rejected, as expected:\n{exc}")


def show_storage_state() -> None:
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
    for vertex in capacities:
        print(f"  used_capacity({vertex}) = {state.used_capacity(vertex)}  (cap = {capacities[vertex]})")

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
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=10,
        arrival_rate=1.0,
        seed=0,
        horizon=1000,
        num_completed_tasks=0,
        throughput=0.0,
        num_waiting_tasks=3,
        num_active_tasks=0,
        backlog=3,
        num_traversed_edges=0,
        mean_decision_runtime_seconds=0.004,
    )
    print(row.model_dump_json(indent=2))
    print(
        "Note: with num_completed_tasks=0 (no simulator yet), mean_service_time, "
        "movement_cost_per_task and mean_blocked_time must stay None -- "
        "'runs with |C_T|=0 are reported separately', per eq:throughput."
    )


def main() -> None:
    show_experiment_configs()
    state = show_storage_state()
    show_task_coupling(state)
    show_run_metrics_shape()


if __name__ == "__main__":
    main()
