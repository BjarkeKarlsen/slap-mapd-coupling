"""Compare F_fix / F_dem / F_cng against the same centralised controller and instance.

The M3 milestone's own acceptance test: swapping storage_mode is the
*only* thing that changes between the three runs below -- same graph,
same initial storage, same controller, same seed. Requires zero changes
to environment/ (that's the point of the storage registry built in M2,
docs/storage_rule_integration.md).

Caveat (issue #60): environment/multi_agent_env.py's stage-6 call
currently passes permanently-empty demand/traversal/waiting estimates --
no online rho_hat_t/mu_hat_t/w_hat_t estimator exists yet anywhere in
this repo. F_dem and F_cng therefore rank every SKU as equally
undemanded when run through a real episode (unlike their own unit tests,
which feed them real estimates directly), so this script demonstrates
the *registry swap* working end to end, not yet a meaningful behavioural
difference between the three rules -- that needs #60 first.
"""

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig, StorageMode
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_and_validate

FLEET_SIZE = 6


def build_instance() -> WarehouseGraph:
    params = GeneratorParams(
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
    graph, report = generate_and_validate(params, fleet_size=FLEET_SIZE, require_well_formed=False)
    if not report.accepted:
        raise RuntimeError(f"Instance rejected: {report.model_dump_json(indent=2)}")
    return graph


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


def build_config(storage_mode: StorageMode, seed: int) -> ExperimentConfig:
    common = dict(
        controller="centralised",
        congestion_sensitive=(storage_mode == "congestion"),
        communication=False,
        num_agents=FLEET_SIZE,
        arrival_rate=0.1,
        seed=seed,
        horizon=150,
        wait_cost=0.5,
    )
    if storage_mode == "fixed":
        return ExperimentConfig(storage_mode="fixed", **common)
    if storage_mode == "demand":
        return ExperimentConfig(
            storage_mode="demand", storage_epoch_length=25, reassignment_cap=10, **common
        )
    return ExperimentConfig(
        storage_mode="congestion",
        storage_epoch_length=25,
        reassignment_cap=10,
        congestion_weight=1.0,
        **common,
    )


def main() -> None:
    graph = build_instance()
    skus, capacities, counts = build_storage(graph)

    print("=== M3 acceptance test: F_fix / F_dem / F_cng, same instance + controller ===")
    for storage_mode in ("fixed", "demand", "congestion"):
        config = build_config(storage_mode, seed=0)  # type: ignore[arg-type]
        metrics = run_episode(graph, skus, counts, capacities, config)
        print(f"\n--- storage_mode={storage_mode!r} ---")
        print(
            f"  completed={metrics.num_completed_tasks} "
            f"throughput={metrics.throughput:.3f} "
            f"backlog={metrics.backlog} "
            f"num_traversed_edges={metrics.num_traversed_edges}"
        )

    print(
        "\nNote: F_dem/F_cng ran through the real six-stage loop above with no "
        "crashes and no environment/ code changes -- the registry swap this "
        "milestone's storage ladder is meant to demonstrate. Their placement "
        "decisions are not yet meaningfully different from F_fix's here, "
        "since no online demand/traffic estimator feeds them real data yet "
        "(issue #60)."
    )


if __name__ == "__main__":
    main()
