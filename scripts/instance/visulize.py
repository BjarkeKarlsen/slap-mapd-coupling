"""% Running the script: input cmd here: python scripts/instance/visualise.py --out DIR
% Takes all generated instance folders and creates one visualisation per
% fleet size and seed.
%
% It reads:
%   DIR/instances/fleet_*/seed_*.json
%
% and writes:
%   DIR/warehouse_figures/fleet_*/seed_*.png
"""

from __future__ import annotations


import argparse
import json
import os
from pathlib import Path
from typing import Sequence


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from base import C


INSTANCES = "instances"
FIGURES = "warehouse_figures"


# Receiving has no colour in base.py, since no other figure draws it apart.
RCV_F, RCV_E = "#e8d9f0", C["comm"]
ROLES = (  # drawn in this order, so a role vertex sits on top of the plain ones
    ("storage", C["store_f"], C["store_e"], "o"),
    ("delivery", C["del_f"], C["del_e"], "s"),
    ("receiving", RCV_F, RCV_E, "^"),
    ("endpoint", C["ep_f"], C["ep_e"], "D"),
)



def _panel(ax, inst, title, size):
    pos = {int(v): p for v, p in inst["positions"].items()}
    roles = {int(v): r for v, r in inst["roles"].items()}

    seen = set()
    for u, v in inst["edges"]:
        if (v, u) in seen:
            continue
        seen.add((u, v))
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=C["edge"],
            lw=0.8,
            zorder=1,
        )

    plain = [p for v, p in pos.items() if not roles[v]]
    ax.scatter(
        *zip(*plain),
        s=size,
        facecolor="white",
        edgecolor=C["node"],
        linewidths=0.6,
        zorder=2,
    )

    for name, fc, ec, marker in ROLES:
        pts = [pos[v] for v, r in roles.items() if name in r]
        ax.scatter(
            *zip(*pts),
            s=size * 1.6,
            marker=marker,
            facecolor=fc,
            edgecolor=ec,
            linewidths=0.8,
            zorder=3,
        )

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")



def _legend_handles():
    return [
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
        for (_, facecolor, edgecolor, marker), label in zip(
            ROLES,
            ("storage", "delivery", "receiving", "parking"),
        )
    ]



def visualise(out: Path) -> None:
    instance_dir = out / INSTANCES
    figure_dir = out / FIGURES

    fleet_dirs = sorted(instance_dir.glob("fleet_*"))
    if not fleet_dirs:
        raise FileNotFoundError(f"No fleet_* directories found in {instance_dir}")

    sizes = {"8": 40, "16": 22, "32": 11, "64": 5}
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
            n = len(inst["positions"])

            fig, ax = plt.subplots(figsize=(7.5, 7.2))
            _panel(
                ax,
                inst,
                rf"$m={m}$, seed ${seed}$, $|V|={n}$",
                sizes.get(m, 12),
            )

            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=4,
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
        description="Draw every generated warehouse instance, one PNG per fleet size and seed."
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