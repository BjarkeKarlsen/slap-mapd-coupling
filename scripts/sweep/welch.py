"""Welch's method for the warm-up ([A66]): the output of long Token Passing runs, by window of steps.

One episode per (storage rule, test seed), Token Passing as the router, on the spec's instance and
stock model, at the spec's arrival rate unless `--rate` says otherwise. Each episode is cut into
windows of `--window` steps, and each window gives the mean service time of the picks finished in
it, how many finished, and the backlog at its end. Each episode is one JSON row in
`OUT/welch.jsonl`, written as soon as it ends, so a run started again skips the rows already there.

    python scripts/sweep/welch.py --config configs/fleet_8/decentralised.yaml --seeds 10 \\
        --horizon 12000 --out DIR

`OUT/welch.md` and `OUT/welch.png` give, per rule, the mean over seeds of each window and its
moving average by Welch's rule (Welch, 1983): window i is averaged with the m windows on each side,
and the first m windows with as many on each side as they have. The warm-up is read off the plot,
where the smoothed curve stops rising, and nothing is chosen here. The slope of the smoothed curve
over the second half of the run is given too: a slope well above 0 means the warehouse is still
getting slower and never settles at this load. The demand cycle, the steps the popularity ranking
takes to turn once, is given so the measured window can be a whole number of cycles.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.spec import (
    build_experiment_config,
    build_study,
    load_spec,
    override_spec,
)

ROWS = "welch.jsonl"
REPORT = "welch.md"
PLOT = "welch.png"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="a training spec (YAML)")
    parser.add_argument("--seeds", type=int, required=True, help="how many test seeds")
    parser.add_argument("--horizon", type=int, required=True, help="T of each run")
    parser.add_argument("--rate", type=float, default=None, help="overrides the spec's rate")
    parser.add_argument("--storage-modes", nargs="+", default=["fixed"], dest="modes")
    parser.add_argument("--window", type=int, default=100, help="steps per window")
    parser.add_argument("--smoothing", type=int, default=5, help="m, windows on each side")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = override_spec(load_spec(args.config), horizon=args.horizon, arrival_rate=args.rate)
    study = build_study(spec)
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / ROWS
    rows = _read_rows(rows_path)
    done = {(row["storage_mode"], row["seed"]) for row in rows}
    seeds = split_seeds(study.seed_split).train[: args.seeds]
    for mode in args.modes:
        for seed in seeds:
            if (mode, seed) in done:
                continue
            config = build_experiment_config(spec, seed, mode)
            config = ExperimentConfig.model_validate(
                {**config.model_dump(), "controller": None, "planner": "token_passing"}
            )
            row = _episode(study, config, args.window)
            with rows_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row) + "\n")
            rows.append(row)
            print(f"{mode} seed {seed}: {sum(row['finished']):d} picks finished")

    config = build_experiment_config(spec, 0, "fixed")
    cycle = demand_cycle(config, len(study.skus))
    report = _report(rows, args.window, args.smoothing, cycle, spec.arrival_rate, len(study.skus))
    (args.out / REPORT).write_text(report, encoding="utf-8")
    _plot(rows, args.window, args.smoothing, cycle, args.out / PLOT)
    return 0


def _episode(study: Any, config: ExperimentConfig, window: int) -> dict[str, Any]:
    """One run, cut into windows: the mean service time of the picks finished in each window,
    their number, and the backlog at its end."""
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(config.horizon):
        step()
    count = math.ceil(config.horizon / window)
    service = [0] * count
    finished = [0] * count
    for task in env.tasks:
        if task.kind != "pick" or task.finish_time is None:
            continue
        # A pick finished at step t belongs to the window holding t - 1, the step it moved in.
        at = min((task.finish_time - 1) // window, count - 1)
        service[at] += task.finish_time - task.release_time
        finished[at] += 1
    backlog = env.log.backlog_by_t
    return {
        "storage_mode": config.storage_mode,
        "seed": config.seed,
        "horizon": config.horizon,
        "window": window,
        "mean_service": [s / n if n else None for s, n in zip(service, finished)],
        "finished": finished,
        "backlog": [backlog[min((i + 1) * window, len(backlog) - 1)] for i in range(count)],
    }


def demand_cycle(config: ExperimentConfig, num_skus: int) -> int | None:
    """The steps the popularity ranking takes to turn once, |K| / shift periods, rounded up to
    whole periods. None without a rotation."""
    shift = config.rotation_shift(num_skus)
    if config.demand_rotation_period is None or shift is None:
        return None
    return math.ceil(num_skus / shift) * config.demand_rotation_period


def welch_average(values: Sequence[float], m: int) -> list[float]:
    """Welch's moving average: window i averaged with the m on each side, and the first m
    windows with i on each side, so every average is centred. The last m windows have none."""
    smoothed = []
    for i in range(len(values) - m):
        half = min(i, m)
        smoothed.append(statistics.fmean(values[i - half : i + half + 1]))
    return smoothed


def mean_over_seeds(rows: list[dict[str, Any]], key: str) -> list[float]:
    """The mean over seeds of each window, windows with no value in a seed left out of it."""
    count = min(len(row[key]) for row in rows)
    means = []
    for i in range(count):
        values = [row[key][i] for row in rows if row[key][i] is not None]
        means.append(statistics.fmean(values) if values else math.nan)
    return means


def slope(values: Sequence[float], window: int) -> float | None:
    """The least-squares slope of the second half of `values`, per 1000 steps."""
    half = [v for v in values[len(values) // 2 :] if not math.isnan(v)]
    n = len(half)
    if n < 2:
        return None
    mean_x, mean_y = (n - 1) / 2, statistics.fmean(half)
    spread = sum((x - mean_x) ** 2 for x in range(n))
    per_window = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(half)) / spread
    return per_window * 1000 / window


def _by_mode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["storage_mode"], []).append(row)
    return by_mode


def _report(
    rows: list[dict[str, Any]], window: int, m: int, cycle: int | None, rate: float, skus: int
) -> str:
    lines = [
        "# Welch's method\n",
        f"Arrival rate {rate:g}, {skus} SKUs, windows of {window} steps, m = {m}. "
        + (f"The demand cycle is {cycle} steps.\n" if cycle else "No demand rotation.\n"),
    ]
    for mode, runs in sorted(_by_mode(rows).items()):
        service = mean_over_seeds(runs, "mean_service")
        backlog = mean_over_seeds(runs, "backlog")
        smoothed = welch_average(service, m)
        rise = slope(smoothed, window)
        lines.append(f"\n## {mode}, {len(runs)} seeds\n")
        lines.append(
            "Smoothed service time, slope over the second half: "
            + ("n/a" if rise is None else f"{rise:+.2f} steps per 1000 steps")
            + ".\n"
        )
        lines.append("\n| window start | service time | smoothed | backlog |\n|---|---|---|---|")
        for i, value in enumerate(service):
            smooth = f"{smoothed[i]:.2f}" if i < len(smoothed) else ""
            lines.append(f"| {i * window} | {value:.2f} | {smooth} | {backlog[i]:.1f} |")
    return "\n".join(lines) + "\n"


def _plot(
    rows: list[dict[str, Any]], window: int, m: int, cycle: int | None, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, runs in sorted(_by_mode(rows).items()):
        service = mean_over_seeds(runs, "mean_service")
        smoothed = welch_average(service, m)
        starts = [i * window for i in range(len(service))]
        (line,) = ax.plot(starts, service, alpha=0.3, linewidth=0.8)
        ax.plot(starts[: len(smoothed)], smoothed, color=line.get_color(), label=mode)
    if cycle:
        for at in range(cycle, max(len(r["finished"]) for r in rows) * window, cycle):
            ax.axvline(at, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("mean pick service time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    sys.exit(main())
