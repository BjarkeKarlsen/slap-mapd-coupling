"""Sweep the load with Token Passing on the instance and stock model of a training spec.

One episode per (storage rule, arrival rate, test seed), Token Passing as the router, on the
spec's instance, stock model and horizon. The test seeds are the first `--seeds` of the spec's
seed split. Each episode is one JSON row of the RunMetrics fields in `OUT/load_sweep.jsonl`,
written as soon as it ends, so a sweep started again skips the rows already there.

    python scripts/load_sweep.py --config configs/fleet_8/centralised.yaml \\
        --rates-per-agent 0.025 0.05 0.075 0.1 --seeds 6 --out DIR

`--rates-per-agent` gives the rates as a share of the fleet size m of the spec, so one grid
serves every fleet. The sweep runs F_fix only unless `--storage-modes` says otherwise, since
the main load is read off fixed storage and every rule and controller of a fleet then runs at
that one rate, as the paired comparison needs.

`OUT/summary.md` is written again after every sweep. It starts with the trip distance
d^trip(v) of every storage vertex, then gives per rule and rate the means over seeds of the
measures the author reads the main load and the stall length off. The capacity of a rule is
the highest rate at which it, and every lower rate, keeps up on every seed (eq:throughput,
f_up of the spec). Throughput is not used for it, because it counts put-aways, which grow with
the rate, so it keeps rising after the fleet has fallen behind. The first rate that fails is
given next to it, so a finer grid between the two can be added to the same OUT. The rate at
0.6 of the capacity is the main load, and `OUT/capacity.json` holds the same numbers.
Nothing is chosen here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from slap_mapd_coupling.core.experiment_config import ExperimentConfig, StorageMode
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.evaluation.evaluator import keeps_up, run_episode
from slap_mapd_coupling.storage.arrival import trip_distance
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
    override_spec,
)

ROWS = "load_sweep.jsonl"
SUMMARY = "summary.md"
CAPACITY = "capacity.json"
# The share of capacity the main load is read at.
LOAD_SHARE = 0.6
# The summary columns, the RunMetrics field of each, averaged over seeds.
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="a training spec (YAML)")
    rates = parser.add_mutually_exclusive_group(required=True)
    rates.add_argument("--rates", type=float, nargs="+", help="arrival rates")
    rates.add_argument(
        "--rates-per-agent", type=float, nargs="+", help="arrival rates over the fleet size m"
    )
    parser.add_argument("--seeds", type=int, required=True, help="how many test seeds")
    parser.add_argument("--storage-modes", nargs="+", default=["fixed"], dest="modes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None, help="overrides the spec's T")
    args = parser.parse_args(argv)

    spec = load_spec(args.config)
    if args.horizon is not None:
        spec = override_spec(spec, horizon=args.horizon)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / ROWS
    rows = _read_rows(rows_path)
    done = {(row["storage_mode"], row["arrival_rate"], row["seed"]) for row in rows}
    num_agents = spec.experiment.num_agents
    sweep_rates = args.rates or [round(share * num_agents, 6) for share in args.rates_per_agent]

    for rate in sweep_rates:
        # The starting stock is sized by the load, so every rate builds its own study.
        rated = override_spec(spec, arrival_rate=rate)
        study = build_study(rated)
        seeds = split_seeds(study.seed_split).test[: args.seeds]
        for mode in args.modes:
            for seed in seeds:
                if (mode, rate, seed) in done:
                    continue
                config = _planner_config(rated, seed, mode)
                metrics = run_episode(
                    study.instance.graph, study.skus, study.counts, study.capacities, config
                )
                row = metrics.model_dump(mode="json")
                with rows_path.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(row) + "\n")
                rows.append(row)
                done.add((mode, rate, seed))
                print(f"{mode} rate {rate} seed {seed}: throughput {metrics.throughput:.3f}")

    study = build_study(spec)
    f_up = spec.experiment.keep_up_threshold
    capacities = {mode: capacity(runs, f_up) for mode, runs in _by_mode(rows).items()}
    (args.out / SUMMARY).write_text(
        _summary(spec, study.instance.graph, rows, capacities), encoding="utf-8"
    )
    (args.out / CAPACITY).write_text(
        json.dumps({"num_agents": num_agents, "f_up": f_up, "by_mode": capacities}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


def capacity(rows: list[dict[str, Any]], f_up: float) -> dict[str, float | None]:
    """The capacity of one rule from its rows: the highest rate at which it, and every lower
    rate, keeps up on every seed, the first rate that fails, and the main load, LOAD_SHARE of
    the capacity. The check is eq:throughput with f_up, worked out from the row's counts so
    a row written before the check changed is read the same way. A value is None when no
    rate passes or none fails."""
    passes_by_rate: dict[float, bool] = {}
    for row in rows:
        passed = keeps_up(row["num_completed_tasks"], row["backlog"], f_up)
        rate = row["arrival_rate"]
        passes_by_rate[rate] = passes_by_rate.get(rate, True) and bool(passed)
    highest = first_failing = None
    for rate in sorted(passes_by_rate):
        if not passes_by_rate[rate]:
            first_failing = rate
            break
        highest = rate
    return {
        "capacity": highest,
        "first_failing_rate": first_failing,
        "main_load": None if highest is None else round(LOAD_SHARE * highest, 6),
    }


def _by_mode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["storage_mode"], []).append(row)
    return by_mode


def _planner_config(spec: TrainingSpec, seed: int, mode: StorageMode) -> ExperimentConfig:
    """The spec's configuration under `mode`, routed by Token Passing."""
    config = build_experiment_config(spec, seed, mode)
    return ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing"}
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def receiving_vertices(spec: TrainingSpec, graph: WarehouseGraph) -> list[VertexId]:
    """V_rcv of the spec: the receiving role, or the delivery vertices."""
    arrival = spec.arrival
    role = "receiving" if arrival is not None and arrival.receiving == "receiving" else "delivery"
    return sorted(v for v, vertex in graph.vertices.items() if getattr(vertex.role, role))


def _summary(
    spec: TrainingSpec,
    graph: WarehouseGraph,
    rows: list[dict[str, Any]],
    capacities: dict[str, dict[str, float | None]],
) -> str:
    f_up = spec.experiment.keep_up_threshold
    receiving = receiving_vertices(spec, graph)
    delivery = sorted(v for v, vertex in graph.vertices.items() if vertex.role.delivery)
    trips = {
        v: trip_distance(graph, v, receiving, delivery)
        for v, vertex in sorted(graph.vertices.items())
        if vertex.role.storage
    }
    shortest, longest = min(trips.values()), max(trips.values())
    lines = [
        "# Token Passing load sweep",
        "",
        "## Trip distance of the storage vertices",
        "",
        f"Smallest {shortest:.1f}, largest {longest:.1f}, ratio {longest / shortest:.2f}.",
        "",
        "| vertex | d_trip |",
        "|---|---|",
        *(f"| {v} | {d:.1f} |" for v, d in trips.items()),
        "",
        "## Means over seeds",
        "",
        "| rule | rate | seeds | keep up | " + " | ".join(name for name, _ in _COLUMNS) + " |",
        "|---|---|---|---|" + "---|" * len(_COLUMNS),
    ]
    by_condition: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((row["storage_mode"], row["arrival_rate"]), []).append(row)
    for (mode, rate), runs in sorted(by_condition.items()):
        passed = sum(
            bool(keeps_up(run["num_completed_tasks"], run["backlog"], f_up)) for run in runs
        )
        cells = [mode, f"{rate:g}", str(len(runs)), f"{passed} / {len(runs)}"]
        for _, field in _COLUMNS:
            if field is None:
                values = [
                    run["num_released_putaways"] / run["num_storage_updates"]
                    for run in runs
                    if run["num_storage_updates"]
                ]
            else:
                values = [run[field] for run in runs if run[field] is not None]
            cells.append(f"{statistics.fmean(values):.3f}" if values else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Capacity",
        "",
        f"The highest rate that keeps up on every seed with f_up = {f_up:g}, every lower rate",
        f"too, the first rate that fails, and the main load at {LOAD_SHARE} of the capacity.",
        "",
        "| rule | capacity | first failing rate | main load |",
        "|---|---|---|---|",
    ]
    for mode, found in sorted(capacities.items()):
        cells = [found[key] for key in ("capacity", "first_failing_rate", "main_load")]
        lines.append(
            f"| {mode} | " + " | ".join("n/a" if c is None else f"{c:g}" for c in cells) + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
