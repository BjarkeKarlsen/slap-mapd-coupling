#!/usr/bin/env python3
"""Write summary.md from an existing load_sweep.jsonl.

Reads OUT/load_sweep.jsonl and writes OUT/summary.md.
It does not run any new episodes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROWS = "load_sweep.jsonl"
SUMMARY = "summary.md"
LOAD_SHARE = 0.6

# Only fields that may be present in the saved rows.
_COLUMNS = (
    ("throughput", "throughput"),
    ("completed share", "completed_share"),
    ("longest stand-still", "longest_stand_still"),
    ("total service", "total_service_time"),
    ("total service finished", "total_service_time_finished"),
    ("mean pick service", "mean_service_time"),
    ("put-aways per arrival", None),
    ("put-away service", "mean_putaway_service_time"),
    ("unplaced units", "unplaced_units"),
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.fmean(values):.3f}"


def _summary(rows: list[dict[str, Any]]) -> str:
    by_condition: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        mode = row.get("storage_mode")
        rate = row.get("arrival_rate")
        if mode is None or rate is None:
            continue
        by_condition.setdefault((mode, rate), []).append(row)

    lines = [
        "# Load sweep summary",
        "",
        "## Means over seeds",
        "",
        "| rule | rate | seeds | keeps-up seeds | " + " | ".join(name for name, _ in _COLUMNS) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(_COLUMNS),
    ]

    for (mode, rate), runs in sorted(by_condition.items()):
        cells = [
            mode,
            f"{rate:g}",
            str(len(runs)),
            str(sum(1 for run in runs if run.get("keeps_up") is True)),
        ]
        for _, field in _COLUMNS:
            values = [
                run[field]
                for run in runs
                if field in run and run[field] is not None
            ]
            cells.append(_mean(values))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Capacity by keep-up",
        "",
        f"Capacity is the highest arrival rate whose runs all keep up. Main load is {LOAD_SHARE} of that rate.",
        "",
        "| rule | capacity rate | first failing rate | main load |",
        "|---|---:|---:|---:|",
    ]

    modes = sorted({mode for mode, _ in by_condition})
    for mode in modes:
        rates = sorted(rate for m, rate in by_condition if m == mode)
        passing_rates = []
        failing_rates = []
        for rate in rates:
            runs = by_condition[(mode, rate)]
            if runs and all(run.get("keeps_up", False) for run in runs):
                passing_rates.append(rate)
            else:
                failing_rates.append(rate)

        if passing_rates:
            capacity_rate = max(passing_rates)
            first_failing = min((r for r in failing_rates if r > capacity_rate), default=None)
            main_load = LOAD_SHARE * capacity_rate
            lines.append(
                f"| {mode} | {capacity_rate:g} | "
                f"{'n/a' if first_failing is None else f'{first_failing:g}'} | "
                f"{main_load:.3f} |"
            )
        else:
            first_failing = min(failing_rates, default=None)
            lines.append(
                f"| {mode} | n/a | "
                f"{'n/a' if first_failing is None else f'{first_failing:g}'} | n/a |"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory holding load_sweep.jsonl")
    args = parser.parse_args()

    rows_path = args.out / ROWS
    rows = _read_rows(rows_path)
    if not rows:
        print(f"No rows found in {rows_path}")
        return 1

    summary = _summary(rows)
    target = args.out / SUMMARY
    target.write_text(summary, encoding="utf-8")
    print(f"Saved to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())