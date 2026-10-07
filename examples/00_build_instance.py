"""Build a parametric warehouse instance and plot it. First milestone: core + instances + viz."""

import os

from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_and_validate,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_well_formedness
from slap_mapd_coupling.viz.warehouse_plot import assign_vertex_names, plot_graph, plot_node_names


def main() -> None:
    fleet_size = 6

    params = GeneratorParams(
        num_aisles=5,
        aisle_length=4,
        num_cross_aisles=3,
        one_way_fraction=0.05,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=8,
        num_delivery_vertices=3,
        num_endpoints=fleet_size,
        seed=1,
    )
    print("GeneratorParams:")
    print(params.model_dump_json(indent=2))

    # Connectivity is the check every baseline needs and this call always
    # runs. Well-formedness is only required by the baselines that need
    # it (sec:pf:env), so it is NOT part of acceptance here -- see the
    # separate check below, which reports it without gating on it.
    accepted_params = params
    graph, report = generate_and_validate(params, fleet_size=fleet_size, require_well_formed=False)
    print("\nInstanceGenerationReport (seed=1, connectivity only):")
    print(f"  accepted: {report.accepted}")
    conn = report.connectivity
    print(f"  connectivity.ok: {conn.ok} (failure_rate={conn.failure_rate:.3f})")

    if not report.accepted:
        print("\nRejected -- retrying across 5 more seeds to report a batch failure rate:")
        attempts = 0
        rejections = 0
        for seed in range(2, 7):
            attempts += 1
            retry_params = params.model_copy(update={"seed": seed})
            graph, retry_report = generate_and_validate(
                retry_params, fleet_size=fleet_size, require_well_formed=False
            )
            if not retry_report.accepted:
                rejections += 1
            else:
                accepted_params = retry_params
            report = retry_report
        print(f"  batch failure rate: {rejections}/{attempts}")

    num_storage = sum(1 for v in graph.vertices.values() if v.role.storage)
    num_delivery = sum(1 for v in graph.vertices.values() if v.role.delivery)
    num_endpoint = sum(1 for v in graph.vertices.values() if v.role.endpoint)
    num_one_way = sum(1 for e in graph.edges if e.one_way)

    print("\nAccepted instance:")
    print(f"  |V_mov| = {len(graph.vertices)}")
    print(f"  |V_str| = {num_storage}")
    print(f"  |V_del| = {num_delivery}")
    print(f"  |V_ep|  = {num_endpoint}")
    print(f"  |E|     = {len(graph.edges)} ({num_one_way} one-way)")

    well_formedness = check_well_formedness(graph, fleet_size=fleet_size)
    print("\nWell-formedness (Ma et al. 2017), reported but not required here:")
    print(
        f"  ok: {well_formedness.ok}  "
        f"endpoints checked: {well_formedness.num_endpoints}  "
        f"failure_rate: {well_formedness.failure_rate:.3f}"
    )
    print(
        "  Note: this generator's single-file aisle-stack layout is close to a "
        "tree, so with more than a couple of storage/delivery vertices it will "
        "usually fail strict well-formedness (some third endpoint sits on the "
        "only path between two others) regardless of seed -- that is an honest "
        "property of the topology, not a bug in the check. A baseline that "
        "actually requires well-formedness needs a denser, more lattice-like "
        "instance (e.g. many more cross-aisles relative to storage vertices)."
    )

    # generate_instance recomputes the same accepted graph plus the layout
    # position generate_warehouse_graph computed internally and discarded
    # -- test_generate_warehouse_graph_unchanged_by_refactor guarantees the
    # two entry points never drift apart.
    instance = generate_instance(accepted_params)
    names = assign_vertex_names(instance.graph)
    ax = plot_graph(instance.graph, instance.positions)
    plot_node_names(ax, instance.graph, instance.positions, names)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "instance.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path} (role-coloured layout via viz.warehouse_plot.plot_graph).")


if __name__ == "__main__":
    main()
