"""% Running the script: input cmd here: python scripts/instance/visualise_digraph.py --out DIR
% Takes all generated instance folders and creates one directed-graph
% visualisation per fleet size and seed.
%
% It reads:
%   DIR/instances/fleet_*/seed_*.json
%
% and writes:
%   DIR/warehouse_graphs/fleet_*/seed_*.png
%
% Unlike visualise.py, this does not use the warehouse grid coordinates.
% It builds a networkx.DiGraph from the instance edges and uses a graph
% layout, so the figure shows the connectivity structure rather than the
% physical warehouse layout.
"""


from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Sequence


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx


from base import C


INSTANCES = "instances"
FIGURES = "warehouse_graphs"


# Receiving has no colour in base.py, since no other figure draws it apart.
RCV_F, RCV_E = "#e8d9f0", C["comm"]
ROLES = (
    ("storage", C["store_f"], C["store_e"], "o", "storage"),
    ("delivery", C["del_f"], C["del_e"], "s", "delivery"),
    ("receiving", RCV_F, RCV_E, "^", "receiving"),
    ("endpoint", C["ep_f"], C["ep_e"], "D", "parking"),
)



def _instance_graph(inst: dict) -> nx.DiGraph:
    """Build the directed graph encoded by one instance JSON file."""
    graph = nx.DiGraph()
    graph.add_nodes_from(int(v) for v in inst["positions"])
    graph.add_edges_from((int(u), int(v)) for u, v in inst["edges"])
    return graph



def _role_nodes(inst: dict) -> dict[str, list[int]]:
    roles = {int(v): r for v, r in inst["roles"].items()}
    return {
        name: [v for v, vertex_roles in roles.items() if name in vertex_roles]
        for name, *_ in ROLES
    }

def _panel(ax, inst, title, layout_seed=None):
    graph = _instance_graph(inst)

    # Use the exported coordinates directly: no layout optimisation.
    pos = {
        int(v): (float(p[0]), float(p[1]))
        for v, p in inst["positions"].items()
    }
    roles = {
        int(v): vertex_roles
        for v, vertex_roles in inst["roles"].items()
    }

    # Store each reciprocal connection once.
    two_way = [
        (u, v)
        for u, v in graph.edges()
        if u < v and graph.has_edge(v, u)
    ]
    one_way = [
        (u, v)
        for u, v in graph.edges()
        if not graph.has_edge(v, u)
    ]

    node_size = 45
    role_size = 80

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=two_way,
        ax=ax,
        edge_color=C["edge"],
        width=0.8,
        alpha=0.7,
        arrows=False,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=one_way,
        ax=ax,
        edge_color=C["accent"],
        width=1.2,
        alpha=1.0,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        node_size=role_size,
        connectionstyle="arc3,rad=0.0",
    )

    plain = [
        v for v in graph.nodes()
        if not roles.get(v, [])
    ]

    if plain:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=plain,
            node_color="white",
            edgecolors=C["node"],
            node_size=node_size,
            linewidths=0.6,
            ax=ax,
        )

    role_styles = (
        ("storage", C["store_f"], C["store_e"], "o"),
        ("delivery", C["del_f"], C["del_e"], "s"),
        ("receiving", C["rcv_f"], C["rcv_e"], "v"),
        ("endpoint", C["ep_f"], C["ep_e"], "D"),
    )

    for name, facecolor, edgecolor, marker in role_styles:
        nodes = [
            v for v in graph.nodes()
            if name in roles.get(v, [])
        ]

        if not nodes:
            continue

        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=facecolor,
            edgecolors=edgecolor,
            node_size=role_size,
            node_shape=marker,
            linewidths=0.8,
            ax=ax,
        )

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.margins(0.05)
    ax.axis("off")



def _legend_handles():
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=marker,
            markersize=8,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            label=label,
        )
        for (_, facecolor, edgecolor, marker, label) in ROLES
    ]

    handles += [
        Line2D([], [], color=C["edge"], lw=1.6, label="two-way edge"),
        Line2D([], [], color=C["accent"], lw=1.6, label="one-way edge"),
    ]

    return handles



def visualise(out: Path) -> None:
    instance_dir = out / INSTANCES
    figure_dir = out / FIGURES

    fleet_dirs = sorted(instance_dir.glob("fleet_*"))
    if not fleet_dirs:
        raise FileNotFoundError(f"No fleet_* directories found in {instance_dir}")

    handles = _legend_handles()

    for fleet_dir in fleet_dirs:
        m = fleet_dir.name.removeprefix("fleet_")

        instance_paths = sorted(
            fleet_dir.glob("seed_*.json"),
            key=lambda path: int(path.stem.removeprefix("seed_")),
        )

        if not instance_paths:
            print(f"No seed_*.json files found in {fleet_dir}")
            continue

        for instance_path in instance_paths:
            with instance_path.open() as handle:
                inst = json.load(handle)

            seed = int(instance_path.stem.removeprefix("seed_"))
            n = graph.num_nodes if False else len(inst["positions"])

            fig, ax = plt.subplots(figsize=(9, 8))
            _panel(
                ax,
                inst,
                rf"$m={m}$, seed ${seed}$, $|V|={n}$",
                layout_seed=seed,
            )

            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                frameon=False,
            )

            output_dir = figure_dir / f"fleet_{m}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"seed_{seed}.png"

            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            print(f"Saved {output_path}", flush=True)



def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw every generated warehouse instance as a directed graph, "
            "one PNG per fleet size and seed."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="report directory containing instances/fleet_*/seed_*.json",
    )
    args = parser.parse_args(argv)

    visualise(args.out)



if __name__ == "__main__":
    main()