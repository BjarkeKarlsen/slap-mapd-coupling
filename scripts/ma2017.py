"""Token Passing on the small warehouse of Ma et al. (2017), against their Table 1 (app:checks).

One run per (task frequency, fleet, seed), as in benchmarks/ma2017.py. Each run is one JSON row in
`OUT/ma2017.jsonl`, written as soon as it ends, so a run started again skips the rows already
there. `OUT/ma2017.md` gives, per frequency and fleet, the mean over seeds of the makespan and the
service time next to theirs, and the difference in percent. Runtimes are given, but they are not
comparable, since theirs come from another implementation on another machine.

    python scripts/ma2017.py --seeds 5 --out DIR [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from slap_mapd_coupling.benchmarks.ma2017 import FLEETS, FREQUENCIES, MA2017_TP, run

ROWS = "ma2017.jsonl"
REPORT = "ma2017.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, required=True, help="seeds 0 to SEEDS - 1")
    parser.add_argument("--frequencies", type=float, nargs="+", default=list(FREQUENCIES))
    parser.add_argument("--fleets", type=int, nargs="+", default=list(FLEETS))
    parser.add_argument("--horizon", type=int, default=10_000, help="a run stops here at most")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / ROWS
    rows = _read_rows(rows_path)
    done = {(row["frequency"], row["num_agents"], row["seed"]) for row in rows}
    todo = [
        (m, f, seed)
        for f in args.frequencies
        for m in args.fleets
        for seed in range(args.seeds)
        if (f, m, seed) not in done
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, m, f, seed, args.horizon) for m, f, seed in todo]
        for future in as_completed(futures):
            row = future.result().model_dump()
            with rows_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row) + "\n")
            rows.append(row)
            print(
                f"f {row['frequency']} m {row['num_agents']} seed {row['seed']}: "
                f"makespan {row['makespan']}, service {row['service_time']:.2f}"
            )
    (args.out / REPORT).write_text(report(rows), encoding="utf-8")
    return 0


def report(rows: list[dict[str, Any]]) -> str:
    """The table of means over seeds next to Ma et al.'s Token Passing column."""
    groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["frequency"], row["num_agents"]), []).append(row)
    lines = [
        "# Token Passing on the small warehouse of Ma et al. (2017)",
        "",
        "Means over seeds, with the standard deviation over seeds in brackets. The difference is",
        "ours less theirs, in percent of theirs. Runtime is ms per timestep, and theirs comes from",
        "another implementation on another machine, so it is not comparable. `unfinished` counts",
        "runs that reached the horizon.",
        "",
        "| f | agents | seeds | makespan | theirs | diff % | service time | theirs | diff % "
        "| runtime ms | theirs | unfinished | diverged |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    service_diffs = []
    for key in sorted(groups):
        group = groups[key]
        spans = [row["makespan"] for row in group if row["makespan"] is not None]
        service = [row["service_time"] for row in group]
        runtime = statistics.fmean(row["runtime_ms"] for row in group)
        their_span, their_service, their_runtime = MA2017_TP.get(key, (None, None, None))
        span = statistics.fmean(spans) if spans else None
        mean_service = statistics.fmean(service)
        service_diffs.append(_diff(mean_service, their_service))
        lines.append(
            f"| {key[0]:g} | {key[1]} | {len(group)} | {_mean_sd(spans)} | {_fmt(their_span)} "
            f"| {_fmt(_diff(span, their_span))} | {_mean_sd(service)} | {_fmt(their_service)} "
            f"| {_fmt(_diff(mean_service, their_service))} | {runtime:.2f} "
            f"| {_fmt(their_runtime)} | {sum(row['makespan'] is None for row in group)} "
            f"| {sum(row['num_diverged'] for row in group)} |"
        )
    known = [d for d in service_diffs if d is not None]
    if known:
        lines += [
            "",
            f"Service time, ours against theirs over the {len(known)} cells: mean difference "
            f"{statistics.fmean(known):+.1f} %, from {min(known):+.1f} % to {max(known):+.1f} %.",
        ]
    return "\n".join(lines) + "\n"


def _diff(ours: float | None, theirs: float | None) -> float | None:
    if ours is None or theirs is None:
        return None
    return 100.0 * (ours - theirs) / theirs


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.fmean(values):.1f} ({statistics.stdev(values):.1f})"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    sys.exit(main())
