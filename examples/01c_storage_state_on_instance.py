"""Visualise a StorageState on a real generated instance.

01b_storage_and_config.py demonstrates StorageState with bare vertex ids
(1, 2, 3) and no graph -- there's nothing for viz.warehouse_plot to draw
without fabricating one. This script closes that gap: a real
generate_instance() graph + positions, a StorageState assigned onto its
actual storage vertices (some holding more than one SKU at once), rendered
as one picture via viz.warehouse_plot. Each storage vertex is given a short
letter name (A, B, C, ...) -- matching 01b_storage_and_config.py's own
A/B/C worked example -- written directly on the node, with its "sku:count"
contents labelled underneath; a side list gives each letter's used/max
capacity (eq:feasiblestorage's cap(v)) -- node-level capacity reads
naturally as one list, while per-SKU contents read naturally next to the
node holding them. Delivery and endpoint vertices get their own short
handles too (D1, D2, ... / E1, E2, ...) -- no contents/capacity to show
for them yet, but a fixed per-node name is what a future path or
congestion overlay would need to refer to a specific node by.
"""

import os

from slap_mapd_coupling.core.storage_state import StorageState, SkuType
from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_instance
from slap_mapd_coupling.viz.warehouse_plot import (
    assign_delivery_endpoint_names,
    assign_storage_letters,
    plot_graph,
    plot_node_names,
    plot_storage_capacity_list,
    plot_storage_contents,
)


def build_storage_state(storage_vertices: list[VertexId]) -> StorageState:
    """The thesis's tea/coffee/mugs worked example, spread across whichever
    storage vertices this instance actually generated (same convention as
    01b_storage_and_config.py's show_storage_state), with some vertices
    deliberately holding more than one SKU at once."""
    skus = {
        "tea": SkuType(sku_id="tea", unit_capacity=1.0),
        "coffee": SkuType(sku_id="coffee", unit_capacity=2.0),
        "mugs": SkuType(sku_id="mugs", unit_capacity=1.0),
    }
    capacities = {v: 40.0 for v in storage_vertices}
    counts: dict[str, dict[VertexId, int]] = {"tea": {}, "coffee": {}, "mugs": {}}
    # Cycle through every non-empty subset of {tea, coffee, mugs} so at
    # least one vertex ends up holding two SKUs and one holds all three.
    assignments = [
        ["tea"],
        ["tea", "coffee"],
        ["coffee"],
        ["mugs"],
        ["coffee", "mugs"],
        ["tea", "coffee", "mugs"],
    ]
    for i, v in enumerate(storage_vertices):
        for sku_id in assignments[i % len(assignments)]:
            counts[sku_id][v] = 3 + (i % 3)

    return StorageState(skus=skus, capacities=capacities, counts=counts)


def main() -> None:
    params = GeneratorParams(
        num_aisles=4,
        aisle_length=4,
        num_cross_aisles=3,
        one_way_fraction=0.05,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=6,
        num_delivery_vertices=3,
        num_endpoints=4,
        seed=1,
    )
    instance = generate_instance(params)
    letters = assign_storage_letters(instance.graph)
    node_names = assign_delivery_endpoint_names(instance.graph)
    storage_vertices = sorted(letters)

    storage = build_storage_state(storage_vertices)
    print("StorageState assigned onto the generated instance:")
    for v in storage_vertices:
        used = storage.used_capacity(v)
        print(
            f"  {letters[v]} (vertex {v}): used_capacity = {used}  (cap = {storage.capacities[v]})"
        )

    ax = plot_graph(instance.graph, instance.positions)
    plot_storage_contents(ax, instance.positions, storage, letters)
    plot_storage_capacity_list(ax, storage, letters)
    plot_node_names(ax, instance.graph, instance.positions, node_names)
    ax.set_title("StorageState on the generated instance")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "storage_state.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(
        f"\nSaved {out_path} (storage nodes lettered + sku:count labelled, "
        "side list shows used/max capacity)."
    )


if __name__ == "__main__":
    main()
