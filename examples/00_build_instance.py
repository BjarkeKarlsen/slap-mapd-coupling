"""Build the graph v2 warehouse of a fleet from many seeds, test them, and plot one.

Every seed must give a connected, well-formed instance (sec:impl:instances).
Graph v2 hangs every endpoint off the walkable backbone as a leaf, so it is
well-formed by construction and no seed needs a second topology. The sweep
checks every seed with the independent validators and reports the topologies
sampled, which should be one each. Then it plots the fleet's own instance,
the seed of fleet_defaults.
"""

import os
from collections import Counter

from slap_mapd_coupling.instances.generator import (
    FLEET_INSTANCE_SEED,
    GeneratedInstance,
    GeneratorParams,
    InstanceGenerationError,
    fleet_defaults,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.viz.warehouse_plot import assign_vertex_names, plot_graph, plot_node_names

# The smallest fleet of the thesis. Graph v2 needs at least 4 agents.
FLEET_SIZE = 8
NUM_SEEDS = 200
SHOWN_SEED = FLEET_INSTANCE_SEED


def make_params(seed: int) -> GeneratorParams:
    return GeneratorParams(**{**fleet_defaults(FLEET_SIZE), "seed": seed}, wait_cost=0.2)


def sweep_seeds() -> None:
    attempts: Counter[int] = Counter()
    raised = 0
    not_connected = 0
    not_well_formed = 0

    for seed in range(NUM_SEEDS):
        try:
            instance = generate_instance(make_params(seed))
        except InstanceGenerationError as error:
            raised += 1
            print(f"  seed {seed} raised: {error}")
            continue
        attempts[instance.topology_attempts] += 1
        not_connected += not check_connectivity(instance.graph).ok
        not_well_formed += not check_well_formedness(instance.graph, FLEET_SIZE).ok

    generated = sum(attempts.values())
    sampled = sum(k * n for k, n in attempts.items())
    print(f"Swept seeds 0 to {NUM_SEEDS - 1}:")
    print(f"  generated:       {generated}/{NUM_SEEDS} (raised {raised})")
    print(f"  not connected:   {not_connected}")
    print(f"  not well-formed: {not_well_formed}")
    if generated:
        print(f"  topologies sampled per instance: {sampled / generated:.2f}")
        print(f"  discard rate:    {1 - generated / sampled:.3f}")
    print(f"  most topologies for one seed: {max(attempts) if attempts else 0}")
    print(f"  seeds by topologies sampled: {sorted(attempts.items())}")


def describe(instance: GeneratedInstance) -> None:
    graph = instance.graph
    num_storage = sum(1 for v in graph.vertices.values() if v.role.storage)
    num_receiving = sum(1 for v in graph.vertices.values() if v.role.receiving)
    num_delivery = sum(1 for v in graph.vertices.values() if v.role.delivery)
    num_endpoint = sum(1 for v in graph.vertices.values() if v.role.endpoint)
    num_one_way = sum(1 for e in graph.edges if e.one_way)

    print(f"\nInstance of seed {SHOWN_SEED} ({instance.topology_attempts} topologies sampled):")
    print(
        f"  aisle lengths = {instance.aisle_lengths}, "
        f"extra link rows = {instance.cross_aisle_rows}"
    )
    walkable = sum(not v.role.is_task_endpoint for v in graph.vertices.values())
    print(f"  |V_mov| = {len(graph.vertices)} ({walkable} walkable, the rest leaves)")
    print(f"  |V_str| = {num_storage}")
    print(f"  |V_rcv| = {num_receiving}")
    print(f"  |V_del| = {num_delivery}")
    print(f"  |V_ep|  = {num_endpoint}")
    print(f"  |E|     = {len(graph.edges)} ({num_one_way} one-way)")


def plot_instance(instance: GeneratedInstance) -> None:
    names = assign_vertex_names(instance.graph)
    ax = plot_graph(instance.graph, instance.positions)
    plot_node_names(ax, instance.graph, instance.positions, names)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "instance.png")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path} (role-coloured layout via viz.warehouse_plot.plot_graph).")


def main() -> None:
    sweep_seeds()
    instance = generate_instance(make_params(SHOWN_SEED))
    describe(instance)
    plot_instance(instance)


if __name__ == "__main__":
    main()
