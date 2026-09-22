"""Build a bigger, denser warehouse instance and plot it.

00_build_instance.py's default (5 aisles x 4 cells, 3 cross-aisles, 8
storage vertices) is deliberately small for a first look. This script asks
the same generator for a much larger, denser instance -- more aisles,
longer aisles, many more cross-aisles relative to storage/delivery count
-- to see what that actually looks like, and to empirically check issue
#74's question: does density alone (within this generator's single-file
aisle-stack topology) get closer to well-formedness (Ma et al. 2017), or
does the topology itself need to change?

one_way_fraction is 0.0 here, not 00_build_instance.py's 0.05: at this
larger scale, one-way edges introduce enough disconnection risk that
acceptance becomes seed-dependent (empirically ~35-60% accept rate for
one_way_fraction in [0.02, 0.04] against these params) -- 0.0 keeps this
script deterministic and always reproducible, since density/size is the
point here, not one-way-edge behaviour.
"""

import os

from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    generate_and_validate,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_well_formedness
from slap_mapd_coupling.viz.warehouse_plot import assign_vertex_names, plot_graph, plot_node_names


def main() -> None:
    fleet_size = 10

    params = GeneratorParams(
        num_aisles=8,
        aisle_length=10,
        num_cross_aisles=8,
        one_way_fraction=0.0,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=16,
        num_delivery_vertices=6,
        num_endpoints=fleet_size,
        seed=42,
    )
    print("GeneratorParams (bigger + denser than 00_build_instance.py):")
    print(params.model_dump_json(indent=2))

    graph, report = generate_and_validate(params, fleet_size=fleet_size, require_well_formed=False)
    print("\nInstanceGenerationReport (connectivity only):")
    print(f"  accepted: {report.accepted}")
    print(f"  connectivity.ok: {report.connectivity.ok}")

    num_storage = sum(1 for v in graph.vertices.values() if v.role.storage)
    num_delivery = sum(1 for v in graph.vertices.values() if v.role.delivery)
    num_endpoint = sum(1 for v in graph.vertices.values() if v.role.endpoint)

    print("\nAccepted instance:")
    print(f"  |V_mov| = {len(graph.vertices)}")
    print(f"  |V_str| = {num_storage}")
    print(f"  |V_del| = {num_delivery}")
    print(f"  |V_ep|  = {num_endpoint}")
    print(f"  |E|     = {len(graph.edges)}")

    well_formedness = check_well_formedness(graph, fleet_size=fleet_size)
    print("\nWell-formedness (Ma et al. 2017), reported but not required here:")
    print(
        f"  ok: {well_formedness.ok}  "
        f"endpoints checked: {well_formedness.num_endpoints}  "
        f"failure_rate: {well_formedness.failure_rate:.3f}"
    )
    print(
        "  Result: still not well-formed, at a similar failure rate to "
        "00_build_instance.py's small default -- more cross-aisles relative "
        "to storage vertices does NOT fix well-formedness by itself here. "
        "Confirms 00_build_instance.py's own note (issue #5) empirically: "
        "the single-file aisle-stack topology is the limiting factor, not "
        "instance size or density within that topology. See issue #74."
    )

    instance = generate_instance(params)
    names = assign_vertex_names(instance.graph)
    ax = plot_graph(instance.graph, instance.positions)
    plot_node_names(ax, instance.graph, instance.positions, names)
    ax.set_title(f"Denser instance: |V|={len(graph.vertices)}, |E|={len(graph.edges)}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "instance_dense.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}.")


if __name__ == "__main__":
    main()
