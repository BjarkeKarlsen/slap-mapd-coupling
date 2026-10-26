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

`--vary SETTING --values ...` runs the sweep once per value of one demand, arrival or storage-rule
setting, every other setting as the spec has it. It is for the sensitivity sweeps at the main
load, so `--rates` is then usually that one rate. Each row records the setting and its value, and
the summary and `OUT/capacity.json` are given per value. One OUT holds one setting.

    python scripts/sweep/load_sweep.py --config configs/fleet_8/decentralised.yaml \\
        --rates 0.12 --seeds 50 --vary demand.skew --values 0.5 1.0 1.5 --out DIR

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
from slap_mapd_coupling.evaluation.evaluator import row_keeps_up, run_episode
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
# The settings `--vary` can change, as section.key of the spec.
VARIABLE = (
    "demand.skew",
    "demand.rotation_period",
    "demand.rotation_shift",
    "demand.rotation_share",
    "arrival.margin",
    "arrival.refill_units",
    "arrival.service_level",
    "arrival.cold_stock_top_k",
    "storage_rule.epoch_length",
    "storage_rule.congestion_weight",
    "storage.vertex_capacity",
    "storage.start_fill",
)
# The summary columns: label, RunMetrics field, and how the seeds are summed up. Each has a use
# (notes/measures_catalogue.md in thesis-progress). The two keep-up parts show which kind fails
# ([A59]), the longest gap between finished tasks is what the stall length is chosen from (D18),
# the backlog slope shows a backlog still growing at T ([A55]), and overflow and unplaced units go
# in the load and demand appendix. Service times are read from the rows and the plots, not from here.
_COLUMNS = (
    ("picks keep up", "keeps_up_picks", "count"),
    ("put-aways keep up", "keeps_up_putaways", "count"),
    ("longest gap between finishes, highest", "longest_completion_gap", "max"),
    ("backlog slope per 100 steps", "backlog_slope", "mean"),
    ("overflow units", "overflow_units", "mean"),
    ("unplaced units", "unplaced_units", "mean"),
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
    parser.add_argument(
        "--warmup", type=int, default=None, help="overrides the spec's warm-up T_warm"
    )
    parser.add_argument("--vary", choices=VARIABLE, default=None, help="a setting to sweep")
    parser.add_argument("--values", type=float, nargs="+", help="the values of --vary")
    parser.add_argument(
        "--trace-seeds",
        type=int,
        default=0,
        help="write a per-step trace (evaluation/trace.py) for the first N seeds of each run",
    )
    parser.add_argument(
        "--stop-when-overloaded",
        action="store_true",
        help="skip the rates above one at which no seed keeps up",
    )
    args = parser.parse_args(argv)
    if (args.vary is None) != (args.values is None):
        parser.error("give --vary and --values together")

    spec = load_spec(args.config)
    if args.horizon is not None or args.warmup is not None:
        spec = override_spec(spec, horizon=args.horizon, warmup=args.warmup)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / ROWS
    rows = _read_rows(rows_path)
    varied = {row.get("varied") for row in rows} - {args.vary}
    if varied:
        parser.error(f"{rows_path} holds rows of {sorted(map(str, varied))}, not {args.vary}")
    done = {(row.get("value"), row["storage_mode"], row["arrival_rate"], row["seed"]) for row in rows}
    num_agents = spec.experiment.num_agents
    sweep_rates = args.rates or [round(share * num_agents, 6) for share in args.rates_per_agent]

    f_up = spec.experiment.keep_up_threshold
    for value in args.values or [None]:
        valued = spec if value is None else with_setting(spec, args.vary, value)
        for rate in sorted(sweep_rates):
            # The starting stock is sized by the load, so every rate builds its own study. At a
            # high rate it can outgrow the shelves, and that rate and every higher one are left
            # out rather than ending the sweep, so the rows and the capacity are still written.
            rated = override_spec(valued, arrival_rate=rate)
            try:
                study = build_study(rated)
            except ValueError as error:
                print(f"rate {rate} and above skipped, the study cannot be built: {error}")
                break
            seeds = split_seeds(study.seed_split).train[: args.seeds]
            for mode in args.modes:
                for n, seed in enumerate(seeds):
                    if (value, mode, rate, seed) in done:
                        continue
                    config = _planner_config(rated, seed, mode)
                    trace = None
                    if n < args.trace_seeds:
                        at = "" if value is None else f"{args.vary}_{value:g}_"
                        trace = args.out / "traces" / f"{at}{mode}_rate{rate:g}_seed{seed}.jsonl"
                    metrics = run_episode(
                        study.instance.graph,
                        study.skus,
                        study.counts,
                        study.capacities,
                        config,
                        trace_path=trace,
                        timing_path=args.out / "timing.jsonl",
                    )
                    row = metrics.model_dump(mode="json")
                    if value is not None:
                        row.update(varied=args.vary, value=value)
                    with rows_path.open("a", encoding="utf-8") as sink:
                        sink.write(json.dumps(row) + "\n")
                    rows.append(row)
                    done.add((value, mode, rate, seed))
                    at = "" if value is None else f"{args.vary} {value:g} "
                    print(f"{at}{mode} rate {rate} seed {seed}: throughput {metrics.throughput:.3f}")
            # With --stop-when-overloaded, a rate at which no seed of any rule keeps up ends the
            # sweep, since a higher rate cannot change a capacity. The rates above capacity
            # that some seed still passes are kept, for the load and demand appendix.
            if not args.stop_when_overloaded:
                continue
            at_rate = [
                row for row in rows if row.get("value") == value and row["arrival_rate"] == rate
            ]
            if at_rate and not any(row_keeps_up(row, f_up) for row in at_rate):
                if rate != max(sweep_rates):
                    print(f"no seed keeps up at rate {rate}, the higher rates are skipped")
                break

    study = build_study(spec)
    by_value: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_value.setdefault(row.get("value"), []).append(row)
    sections, found = [], {}
    for value, runs in sorted(by_value.items(), key=lambda item: (item[0] is None, item[0])):
        capacities = {mode: capacity(of_mode, f_up) for mode, of_mode in _by_mode(runs).items()}
        found[value] = capacities
        title = None if value is None else f"{args.vary or rows[0]['varied']} = {value:g}"
        sections.append(_tables(spec, runs, capacities, title))
    (args.out / SUMMARY).write_text(
        _trips(spec, study.instance.graph) + "".join(sections), encoding="utf-8"
    )
    report: dict[str, Any] = {"num_agents": num_agents, "f_up": f_up}
    if set(found) <= {None}:  # no setting varied, or no row at all
        report["by_mode"] = found.get(None, {})
    else:
        report["varied"] = args.vary or rows[0]["varied"]
        report["by_value"] = {f"{value:g}": capacities for value, capacities in found.items()}
    (args.out / CAPACITY).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def with_setting(spec: TrainingSpec, setting: str, value: float) -> TrainingSpec:
    """`spec` with one setting of VARIABLE, given as section.key, set to `value`. The result is
    validated again, so a value the spec would reject raises here."""
    section, key = setting.split(".")
    data = spec.model_dump()
    if data.get(section) is None:
        raise ValueError(f"the spec has no {section} section to set {key} in")
    data[section][key] = value
    return TrainingSpec.model_validate(data)


def capacity(rows: list[dict[str, Any]], f_up: float) -> dict[str, float | None]:
    """The capacity of one rule from its rows: the highest rate at which it, and every lower
    rate, keeps up on every seed, the first rate that fails, and the main load, LOAD_SHARE of
    the capacity. The check is eq:throughput with f_up, for picks and put-aways separately
    ([A59]), worked out from the row's counts so a row written before the check changed is read
    the same way. A value is None when no
    rate passes or none fails."""
    passes_by_rate: dict[float, bool] = {}
    for row in rows:
        passed = row_keeps_up(row, f_up)
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


def _trips(spec: TrainingSpec, graph: WarehouseGraph) -> str:
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
        f"Smallest {shortest:.1f}, largest {longest:.1f}, ratio {longest / shortest:.2f}, over "
        f"{len(trips)} storage vertices. The ratio bounds what storage can save on a trip.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _tables(
    spec: TrainingSpec,
    rows: list[dict[str, Any]],
    capacities: dict[str, dict[str, float | None]],
    title: str | None,
) -> str:
    """The means over seeds and the capacity of `rows`, under `title` when a setting is varied."""
    f_up = spec.experiment.keep_up_threshold
    level = "##" if title is None else "###"
    lines = [] if title is None else [f"## {title}", ""]
    lines += [
        f"{level} The check per rate",
        "",
        "| rule | rate | seeds | keep up | " + " | ".join(name for name, _, _ in _COLUMNS) + " |",
        "|---|---|---|---|" + "---|" * len(_COLUMNS),
    ]
    by_condition: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((row["storage_mode"], row["arrival_rate"]), []).append(row)
    for (mode, rate), runs in sorted(by_condition.items()):
        passed = sum(bool(row_keeps_up(run, f_up)) for run in runs)
        cells = [mode, f"{rate:g}", str(len(runs)), f"{passed} / {len(runs)}"]
        for _, field, how in _COLUMNS:
            values = [run[field] for run in runs if run.get(field) is not None]
            if not values:
                cells.append("n/a")
            elif how == "count":
                cells.append(f"{sum(1 for v in values if v)} / {len(runs)}")
            elif how == "max":
                cells.append(f"{max(values):g}")
            else:
                cells.append(f"{statistics.fmean(values):.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"{level} Capacity",
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
    return "\n".join(lines) + "\n\n"


if __name__ == "__main__":
    sys.exit(main())