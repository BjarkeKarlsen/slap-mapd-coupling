"""Build the graph v2 warehouses of the larger fleets and plot one.

00_build_instance.py looks at the smallest fleet. This script builds the
warehouse of each fleet size up to 32 agents, prints its size and roles, and
checks both conditions of app:wellformed. Graph v2 keeps 16 walkable vertices
per agent at every fleet size, with the leaves (slots, workstations, docks,
parking) on top, so the floor space per robot is the same and the slot count
grows with the rack faces. The 64-agent warehouse builds the same way but
takes about 20 seconds, so it is left out here.

The graph v1 warehouse of a fleet is still built by
``fleet_defaults_v1``, for the graph v1 results.
"""

import os

from slap_mapd_coupling.instances.generator import (
    GeneratorParams,
    fleet_defaults,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.viz.warehouse_plot import plot_graph

FLEETS = (8, 16, 32)
SHOWN_FLEET = 16


def main() -> None:
    shown = None
    for fleet_size in FLEETS:
        params = GeneratorParams(**fleet_defaults(fleet_size), wait_cost=0.2)
        instance = generate_instance(params)
        graph = instance.graph
        roles = [v.role for v in graph.vertices.values()]
        walkable = sum(not role.is_task_endpoint for role in roles)
        print(
            f"{fleet_size} agents: {params.num_aisles} aisles of {params.rack_rows} rack rows "
            f"per block, {walkable} walkable and {len(graph.vertices) - walkable} leaves, "
            f"{sum(r.storage for r in roles)} slots, {sum(r.delivery for r in roles)} "
            f"workstations, {sum(r.receiving for r in roles)} docks, "
            f"{sum(r.endpoint for r in roles)} parking, "
            f"{sum(e.one_way for e in graph.edges)} one-way edges"
        )
        print(
            f"  connected: {check_connectivity(graph).ok}, "
            f"well-formed: {check_well_formedness(graph, fleet_size).ok}"
        )
        if fleet_size == SHOWN_FLEET:
            shown = instance

    assert shown is not None
    ax = plot_graph(shown.graph, shown.positions)
    ax.set_title(f"Graph v2, {SHOWN_FLEET} agents")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "instance_v2_fleet_16.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}.")


if __name__ == "__main__":
    main()
