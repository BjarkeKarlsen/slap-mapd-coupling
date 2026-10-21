#!/usr/bin/env python3
"""Collect the capacity and main load of every fleet size from its load sweep.

Reads OUT/sweep_<m>_agents/capacity.json, written by scripts/load_sweep.py, for m in 8, 16, 32
and 64, prints one table and writes OUT/fleet_parameters.json. The capacity is the highest rate
at which F_fix keeps up on every seed (eq:throughput), and the main load is 0.6 of it. Every
rule and controller of a fleet then runs at that one rate.

    python scripts/extract_sweep_parameters.py DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLEET_SIZES = (8, 16, 32, 64)


def main(argv: list[str]) -> int:
    out = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / "DIR"
    results = {}
    for agents in FLEET_SIZES:
        path = out / f"sweep_{agents}_agents" / "capacity.json"
        if not path.exists():
            print(f"{agents} agents: no capacity.json yet")
            continue
        fixed = json.loads(path.read_text(encoding="utf-8"))["by_mode"].get("fixed")
        if fixed is None:
            print(f"{agents} agents: no fixed storage rows")
            continue
        results[agents] = fixed

    if not results:
        print("No sweep results found. Run the sweeps first.")
        return 1

    def cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:g}"

    print("| agents | capacity | first failing rate | main load |")
    print("|---|---|---|---|")
    for agents, found in results.items():
        print(
            f"| {agents} | {cell(found['capacity'])} | {cell(found['first_failing_rate'])} "
            f"| {cell(found['main_load'])} |"
        )
    target = out / "fleet_parameters.json"
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
