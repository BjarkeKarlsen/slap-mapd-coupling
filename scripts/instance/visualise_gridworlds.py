"""Render every warehouse instance as a coloured cell grid.

Run:
    python scripts/instance/visualise_gridworld.py --out DIR

Reads:
    DIR/instances/fleet_*/seed_*.json

Writes:
    DIR/warehouse_gridworlds/fleet_*/seed_*.png
    DIR/warehouse_gridworlds/fleet_*/seed_*.json

Cell encoding:
    -1: no vertex
     0: ordinary traversable vertex
     1: storage
     2: delivery
     3: receiving
     4: parking

The drawing uses the exported coordinate levels, not a graph layout.
Distinct coordinate levels become distinct rows and columns.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


INSTANCES = "instances"
FIGURES = "warehouse_gridworlds"

# Self-contained palette: no import from base.py required.
CELL_STYLES = (
    (-1, "no vertex", "#43505c"),
    (0, "transit", "#ffffff"),
    (1, "storage", "#cfe3f7"),
    (2, "delivery", "#ffd9b3"),
    (3, "receiving", "#f8c8a0"),
    (4, "parking", "#d7f0d8"),
)

ROLE_CODES = {
    "storage": 1,
    "delivery": 2,
    "receiving": 3,
    "endpoint": 4,
}


def instance_to_grid(inst: dict) -> np.ndarray:
    """Convert exported coordinate levels into rows and columns.

    Reject overlapping vertices or multi-role vertices rather than
    silently losing information in the single-value cell encoding.
    """
    positions = {
        int(vertex): (float(point[0]), float(point[1]))
        for vertex, point in inst["positions"].items()
    }
    roles = {
        int(vertex): vertex_roles
        for vertex, vertex_roles in inst["roles"].items()
    }

    if not positions:
        raise ValueError("Instance has no positions")

    if any(
        not np.isfinite(point).all()
        for point in positions.values()
    ):
        raise ValueError("Positions must contain finite coordinates")

    # Largest y is rendered at the top.
    xs = sorted({point[0] for point in positions.values()})
    ys = sorted(
        {point[1] for point in positions.values()},
        reverse=True,
    )

    column_of = {x: column for column, x in enumerate(xs)}
    row_of = {y: row for row, y in enumerate(ys)}

    grid = np.full((len(ys), len(xs)), -1, dtype=int)
    occupied = set()

    for vertex, (x, y) in positions.items():
        cell = (row_of[y], column_of[x])

        if cell in occupied:
            raise ValueError(
                f"Multiple vertices occupy coordinate {(x, y)}"
            )
        occupied.add(cell)

        vertex_roles = roles.get(vertex, [])
        unknown = set(vertex_roles) - ROLE_CODES.keys()
        if unknown:
            raise ValueError(
                f"Vertex {vertex} has unknown roles: {unknown}"
            )

        codes = {
            ROLE_CODES[role]
            for role in vertex_roles
        }

        if len(codes) > 1:
            raise ValueError(
                f"Vertex {vertex} has multiple roles "
                f"{vertex_roles}; use separate role layers "
                "instead of a single-value grid."
            )

        grid[cell] = next(iter(codes), 0)

    return grid


def draw_grid(
    grid: np.ndarray,
    title: str,
    output: Path,
    show_values: bool = False,
) -> None:
    colors = [color for _, _, color in CELL_STYLES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-1.5, 5.5, 1),
        cmap.N,
    )

    rows, columns = grid.shape

    width = min(24, max(8, columns * 0.28))
    height = min(24, max(6, rows * 0.28 + 1.5))

    fig, ax = plt.subplots(figsize=(width, height))

    try:
        ax.imshow(
            grid,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            aspect="equal",
        )

        # Draw boundaries between cells.
        ax.set_xticks(
            np.arange(-0.5, columns, 1),
            minor=True,
        )
        ax.set_yticks(
            np.arange(-0.5, rows, 1),
            minor=True,
        )
        ax.grid(
            which="minor",
            color="#9aa5b1",
            linewidth=0.35,
        )

        ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        if show_values:
            for row in range(rows):
                for column in range(columns):
                    value = int(grid[row, column])
                    ax.text(
                        column,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value == -1 else "#263238",
                    )

        ax.set_title(title, fontsize=12, pad=12)

        handles = [
            Patch(
                facecolor=color,
                edgecolor="#5a6673",
                label=label,
            )
            for _, label, color in CELL_STYLES
        ]

        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
            frameon=False,
        )

        fig.savefig(
            output,
            bbox_inches="tight",
            dpi=300,
        )
    finally:
        plt.close(fig)


def visualise(out: Path, show_values: bool = False) -> None:
    instance_dir = out / INSTANCES
    figure_dir = out / FIGURES

    fleet_dirs = sorted(
        (
            path
            for path in instance_dir.glob("fleet_*")
            if path.is_dir()
        ),
        key=lambda path: int(path.name.removeprefix("fleet_")),
    )

    if not fleet_dirs:
        raise FileNotFoundError(
            f"No fleet_* directories found in {instance_dir}"
        )

    count = 0

    for fleet_dir in fleet_dirs:
        fleet = int(fleet_dir.name.removeprefix("fleet_"))

        instance_paths = sorted(
            fleet_dir.glob("seed_*.json"),
            key=lambda path: int(
                path.stem.removeprefix("seed_")
            ),
        )

        if not instance_paths:
            print(
                f"No seed_*.json files found in {fleet_dir}",
                flush=True,
            )
            continue

        output_dir = figure_dir / fleet_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for instance_path in instance_paths:
            seed = int(
                instance_path.stem.removeprefix("seed_")
            )

            with instance_path.open(encoding="utf-8") as handle:
                inst = json.load(handle)

            grid = instance_to_grid(inst)
            n = len(inst["positions"])

            image_path = output_dir / f"seed_{seed}.png"
            array_path = output_dir / f"seed_{seed}.json"

            draw_grid(
                grid,
                title=(
                    rf"$m={fleet}$, seed ${seed}$, "
                    rf"$|V|={n}$"
                ),
                output=image_path,
                show_values=show_values,
            )

            array_path.write_text(
                json.dumps(grid.tolist(), indent=2) + "\n",
                encoding="utf-8",
            )

            count += 1
            print(
                f"Saved {image_path} and {array_path} "
                f"(grid shape: {grid.shape})",
                flush=True,
            )

    print(f"Rendered {count} instances.", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render generated warehouse instances as "
            "coloured cell grids."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Report directory containing "
            "instances/fleet_*/seed_*.json"
        ),
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print the numeric cell codes inside the tiles.",
    )

    args = parser.parse_args(argv)
    visualise(args.out, show_values=args.show_values)


if __name__ == "__main__":
    main()