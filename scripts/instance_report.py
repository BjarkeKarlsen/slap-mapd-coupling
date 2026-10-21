"""Generate the warehouse of every fleet size over many seeds and report its structure.

For each config, the instance section is generated once per seed, the config's own seed first
(the instance the experiments use) and then the seeds 0 to `--seeds - 1`. Every instance is
checked for connectivity and well-formedness (app:wellformed) and measured: its size, the
vertices of each role, the degree, the layouts discarded before one was accepted, and the trip
distance d^trip(v) of its storage vertices. Each instance is one JSON row of
`OUT/instance_report.jsonl`, and `OUT/summary.md` gives the means per fleet size with the
discard rate and the share of instances that pass both checks (app:instances).
`OUT/instances_used.json` holds the instance of every config's own seed, its vertex positions,
roles and edges, for drawing.

    python scripts/instance_report.py --configs configs/fleet_{8,16,32,64}/centralised.yaml \\
        --seeds 50 --out DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from slap_mapd_coupling.instances.generator import GeneratedInstance, generate_instance
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.storage.arrival import trip_distance
from slap_mapd_coupling.training.spec import TrainingSpec, instance_params, load_spec

ROWS = "instance_report.jsonl"
SUMMARY = "summary.md"
USED = "instances_used.json"


def measure(instance: GeneratedInstance, num_agents: int) -> dict[str, Any]:
    """The structure of one instance, and whether it passes both checks."""
    graph = instance.graph
    roles = {v: vertex.role for v, vertex in graph.vertices.items()}
    storage = sorted(v for v, role in roles.items() if role.storage)
    delivery = sorted(v for v, role in roles.items() if role.delivery)
    receiving = sorted(v for v, role in roles.items() if role.receiving)
    parking = sorted(v for v, role in roles.items() if role.endpoint)
    neighbours = {v: set(graph.out_neighbours(v)) for v in graph.vertices}
    for v in graph.vertices:
        for w in list(neighbours[v]):
            neighbours[w].add(v)
    degrees = [len(neighbours[v]) for v in graph.vertices]
    segments = {(min(e.source, e.target), max(e.source, e.target)) for e in graph.edges}
    trips = [trip_distance(graph, v, receiving, delivery) for v in storage] if receiving else []
    connectivity = check_connectivity(graph)
    well_formedness = check_well_formedness(graph, num_agents)
    num_vertices = len(graph.vertices)
    num_roles = len(storage) + len(delivery) + len(receiving) + len(parking)
    return {
        "num_vertices": num_vertices,
        "num_segments": len(segments),
        "num_directed_edges": len(graph.edges),
        "num_one_way": sum(e.one_way for e in graph.edges),
        "num_aisles": instance.num_aisles,
        "aisle_length_min": min(instance.aisle_lengths),
        "aisle_length_max": max(instance.aisle_lengths),
        "num_cross_aisles": len(instance.cross_aisle_rows),
        "num_storage": len(storage),
        "num_delivery": len(delivery),
        "num_receiving": len(receiving),
        "num_parking": len(parking),
        "role_share": num_roles / num_vertices,
        "vertices_per_agent": num_vertices / num_agents,
        "mean_degree": statistics.mean(degrees),
        "num_leaves": sum(d == 1 for d in degrees),
        "roles_on_leaves": sum(
            len(neighbours[v]) == 1 for v in storage + delivery + receiving + parking
        ),
        "topology_attempts": instance.topology_attempts,
        "trip_mean": statistics.mean(trips) if trips else None,
        "trip_min": min(trips) if trips else None,
        "trip_max": max(trips) if trips else None,
        "connected": connectivity.ok,
        "well_formed": well_formedness.ok,
        "blocked_pairs": len(well_formedness.blocked_pairs),
    }


def layout(instance: GeneratedInstance) -> dict[str, Any]:
    """The positions, roles and edges of one instance, enough to draw it."""
    graph = instance.graph
    return {
        "positions": {str(v): instance.positions[v] for v in graph.vertices},
        "roles": {
            str(v): [
                name
                for name in ("storage", "delivery", "receiving", "endpoint")
                if getattr(vertex.role, name)
            ]
            for v, vertex in graph.vertices.items()
        },
        "edges": [[e.source, e.target] for e in graph.edges],
    }


def report_rows(
    spec: TrainingSpec, config: Path, seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], GeneratedInstance]:
    """One row per seed of the instance section of `spec`, and the instance of the first seed."""
    num_agents = spec.experiment.num_agents
    rows = []
    first: GeneratedInstance | None = None
    for seed in seeds:
        params = instance_params(spec).model_copy(update={"seed": seed})
        start = time.perf_counter()
        instance = generate_instance(params)
        seconds = time.perf_counter() - start
        first = first or instance
        row = {"config": str(config), "num_agents": num_agents, "seed": seed}
        row.update(measure(instance, num_agents))
        row["generation_seconds"] = seconds
        rows.append(row)
        print(
            f"m={num_agents} seed={seed}: |V|={row['num_vertices']} "
            f"attempts={row['topology_attempts']} well_formed={row['well_formed']} "
            f"{seconds:.1f}s",
            flush=True,
        )
    assert first is not None
    return rows, first


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(row[key] for row in rows if row[key] is not None)


def summary(rows: list[dict[str, Any]], instance_seeds: dict[int, int]) -> str:
    """Means per fleet size over the seeds, and the instance each config uses."""
    lines = [
        "# Warehouse instances",
        "",
        "Means over the seeds per fleet size. The discard rate is the share of sampled layouts",
        "that no role placement made well-formed. `used` is the instance of the config's seed.",
        "",
        "| m | seeds | pass | discard | V | V per agent | segments | aisles | cross-aisles | "
        "str | del | rcv | ep | role share | degree | roles on leaves | d^trip mean | "
        "seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for num_agents in sorted({row["num_agents"] for row in rows}):
        group = [row for row in rows if row["num_agents"] == num_agents]
        attempts = sum(row["topology_attempts"] for row in group)
        passed = sum(row["connected"] and row["well_formed"] for row in group)
        lines.append(
            f"| {num_agents} | {len(group)} | {passed}/{len(group)} | "
            f"{(attempts - len(group)) / attempts:.2f} | {_mean(group, 'num_vertices'):.0f} | "
            f"{_mean(group, 'vertices_per_agent'):.1f} | {_mean(group, 'num_segments'):.0f} | "
            f"{_mean(group, 'num_aisles'):.1f} | {_mean(group, 'num_cross_aisles'):.1f} | "
            f"{_mean(group, 'num_storage'):.0f} | {_mean(group, 'num_delivery'):.0f} | "
            f"{_mean(group, 'num_receiving'):.0f} | {_mean(group, 'num_parking'):.0f} | "
            f"{_mean(group, 'role_share'):.3f} | {_mean(group, 'mean_degree'):.2f} | "
            f"{_mean(group, 'roles_on_leaves'):.1f} | {_mean(group, 'trip_mean'):.1f} | "
            f"{_mean(group, 'generation_seconds'):.1f} |"
        )
    lines += ["", "## The instance each config uses", ""]
    for num_agents, seed in sorted(instance_seeds.items()):
        used = next(row for row in rows if row["num_agents"] == num_agents and row["seed"] == seed)
        lines.append(
            f"- m={num_agents}, seed {seed}: |V|={used['num_vertices']}, "
            f"{used['num_segments']} segments, {used['num_aisles']} aisles of length "
            f"{used['aisle_length_min']} to {used['aisle_length_max']}, "
            f"{used['num_cross_aisles']} cross-aisles, d^trip {used['trip_min']:.0f} to "
            f"{used['trip_max']:.0f} (mean {used['trip_mean']:.1f}), "
            f"{used['topology_attempts']} layout(s) sampled, well-formed {used['well_formed']}."
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=50, help="seeds 0 to SEEDS - 1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    instance_seeds: dict[int, int] = {}
    used: dict[str, Any] = {}
    for config in args.configs:
        spec = load_spec(config)
        own_seed = spec.instance_settings()["seed"]
        instance_seeds[spec.experiment.num_agents] = own_seed
        seeds = [own_seed] + [s for s in range(args.seeds) if s != own_seed]
        config_rows, instance = report_rows(spec, config, seeds)
        rows += config_rows
        used[str(spec.experiment.num_agents)] = {"seed": own_seed, **layout(instance)}
        (args.out / USED).write_text(json.dumps(used))
        with (args.out / ROWS).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        (args.out / SUMMARY).write_text(summary(rows, instance_seeds))


if __name__ == "__main__":
    main()
