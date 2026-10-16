"""Full comparison sweep across storage rules, controllers, and load levels.

This first version sweeps controllers and load levels under fixed storage. The storage rule
axis follows once the demand and congestion rules have online estimates to run with.

It evaluates trained checkpoints (02, 03 and 04 write them) and the reference planners at
several arrival rates, on the same instance, horizon and evaluation seeds, so the arms can be
compared where the load separates them. One row per (condition, arrival rate, seed) goes to
`load_sweep.jsonl`, and `viz/study_report.py` draws it next to the training logs.

    python examples/05_full_comparison.py --out OUT \\
        --checkpoint centralised=<path> --checkpoint section=<path> \\
        --checkpoint decentralised=<path> --checkpoint decentralised+comm=<path> \\
        --planner token_passing --arrival-rates 0.15 0.25 0.3 0.4

A checkpoint is named `ARM=PATH`, with ARM one of centralised, section, decentralised. Add
`+comm` to the arm for a checkpoint trained with `--communication`, since it must be evaluated
with the same flag. PATH is the Ray checkpoint folder, as for 04b. `--horizon` and
`--eval-seeds` default to 04's, so they match a run of 02 to 04 with its defaults. A checkpoint
should be evaluated at the horizon it trained at. The arrival rate may differ, and a rate
different from the training one shows how the policy holds up outside it.

Evaluation reads the instance, storage, configuration and caps from 04, as 04b does. The
episodes run through the evaluator, so a checkpoint sees the observations it saw in training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
example_04 = importlib.import_module("04_decentralised_training")

from slap_mapd_coupling.core.experiment_config import ExperimentConfig  # noqa: E402
from slap_mapd_coupling.evaluation.evaluator import run_episode  # noqa: E402
from slap_mapd_coupling.evaluation.metrics import RunMetrics  # noqa: E402
from slap_mapd_coupling.planners.registry import PlannerName  # noqa: E402
from slap_mapd_coupling.training.policy import ModulePolicy  # noqa: E402

ARMS = ("centralised", "section", "decentralised")
PLANNERS = ("token_passing", "prioritised_astar")
ILLUSTRATIVE_ARRIVAL_RATES = (0.15, 0.25, 0.3, 0.4)  # Illustrative. The load levels are TBD (#85).
SWEEP_FILE = "load_sweep.jsonl"

# What a row keeps of RunMetrics. released is derived, since a task is completed or in the backlog.
_METRIC_FIELDS = (
    "num_completed_tasks",
    "backlog",
    "throughput",
    "keep_up_ratio",
    "keeps_up",
    "mean_service_time",
    "mean_wait_for_agent",
    "mean_travel_time",
    "mean_blocked_time",
    "mean_crowding",
    "mean_decision_runtime_seconds",
)


def parse_checkpoint(text: str) -> tuple[str, str, bool, Path]:
    """`ARM[+comm]=PATH` as (label, arm, communication, path)."""
    label, sep, path = text.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"{text!r} is not ARM=PATH")
    arm, plus, flag = label.partition("+")
    if arm not in ARMS or (plus and flag != "comm"):
        raise argparse.ArgumentTypeError(
            f"{label!r} must be one of {', '.join(ARMS)}, optionally followed by +comm"
        )
    if flag and arm != "decentralised":
        raise argparse.ArgumentTypeError("+comm only applies to the decentralised arm")
    return label, arm, bool(flag), Path(path).expanduser()


def row_of(
    label: str, kind: str, rate: float, seed: int, horizon: int, metrics: RunMetrics
) -> dict[str, Any]:
    """One line of the sweep file."""
    row: dict[str, Any] = {
        "condition": label,
        "kind": kind,
        "arrival_rate": rate,
        "seed": seed,
        "horizon": horizon,
    }
    row.update({name: getattr(metrics, name) for name in _METRIC_FIELDS})
    row["released"] = metrics.num_completed_tasks + metrics.backlog
    return row


def planner_config(seed: int, planner: PlannerName, rate: float, horizon: int) -> ExperimentConfig:
    """04's evaluation config with the controller replaced by a planner."""
    fields = example_04.evaluation_config(seed, "decentralised", rate, horizon).model_dump()
    fields["controller"] = None
    fields["planner"] = planner
    return ExperimentConfig(**fields)


def sweep(
    checkpoints: list[tuple[str, str, bool, Path]],
    planners: list[str],
    rates: list[float],
    seeds: tuple[int, ...],
    horizon: int,
    out: Path,
) -> list[dict[str, Any]]:
    """Runs every condition at every rate and seed, appending each row to `out` as it finishes."""
    instance = example_04.build_generated_instance()
    skus, capacities, counts = example_04.build_storage(instance.graph)
    rows: list[dict[str, Any]] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:

        def record(row: dict[str, Any]) -> None:
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()

        for planner in planners:
            for rate in rates:
                for seed in seeds:
                    config = planner_config(seed, planner, rate, horizon)  # type: ignore[arg-type]
                    metrics = run_episode(instance.graph, skus, counts, capacities, config)
                    record(row_of(planner, "planner", rate, seed, horizon, metrics))
                print(f"planner {planner} at {rate}: {len(seeds)} seeds", flush=True)
        for label, arm_name, communication, path in checkpoints:
            module = example_04.load_module(path)
            trained_with = bool(module.model_config["gnn_encoder_config"]["communication"])
            if trained_with != communication:
                raise RuntimeError(
                    f"{label}: the checkpoint was trained with communication={trained_with}, "
                    f"but the label says communication={communication}."
                )
            policy_config = example_04.evaluation_config(
                seeds[0], arm_name, rates[0], horizon, communication  # type: ignore[arg-type]
            )
            arm = example_04.build_arm(instance, arm_name, policy_config.observation_depth)
            # The switch is the module's own, so a checkpoint trained with it needs no label.
            conflict_signals = example_04.conflict_signals_of(module)
            if conflict_signals.enabled:
                print(f"{label}: conflict signals on ({conflict_signals.rank_encoding} rank)")
            for rate in rates:
                for seed in seeds:
                    config = example_04.evaluation_config(
                        seed, arm_name, rate, horizon, communication  # type: ignore[arg-type]
                    )
                    metrics = run_episode(
                        instance.graph,
                        skus,
                        counts,
                        capacities,
                        config,
                        policy=ModulePolicy(module),
                        encoding_config=arm.encoding,
                        zones=arm.zones,
                        pooling=arm.pooling,
                        conflict_signals=conflict_signals,
                    )
                    record(row_of(label, "controller", rate, seed, horizon, metrics))
                print(f"checkpoint {label} at {rate}: {len(seeds)} seeds", flush=True)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep controllers and load levels.")
    parser.add_argument("--out", type=Path, required=True, help="folder for load_sweep.jsonl")
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        action="append",
        default=[],
        metavar="ARM[+comm]=PATH",
        help="a trained checkpoint to evaluate, repeatable",
    )
    parser.add_argument(
        "--planner",
        action="append",
        choices=PLANNERS,
        default=None,
        help="a reference planner to run, repeatable (default token_passing)",
    )
    parser.add_argument(
        "--arrival-rates",
        type=float,
        nargs="+",
        default=list(ILLUSTRATIVE_ARRIVAL_RATES),
        help="arrival rates in tasks per step (illustrative defaults)",
    )
    parser.add_argument("--horizon", type=int, default=example_04.DEFAULT_HORIZON)
    parser.add_argument("--eval-seeds", type=int, default=6, help="number of evaluation seeds")
    args = parser.parse_args(argv)
    if args.planner is None:
        args.planner = ["token_passing"]
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.eval_seeds < 1:
        parser.error("--eval-seeds must be at least 1")
    if any(rate <= 0 for rate in args.arrival_rates):
        parser.error("--arrival-rates must be positive")
    labels = [label for label, *_ in args.checkpoint] + args.planner
    if len(set(labels)) != len(labels):
        parser.error("each checkpoint label and planner may appear once")
    for _, _, _, path in args.checkpoint:
        if not path.exists():
            parser.error(f"no checkpoint at {path}")
    return args


def main() -> None:
    args = parse_args()
    # The evaluation half of 04's seed split, so the seeds are disjoint from the training ones.
    seeds = tuple(example_04.split_seeds(example_04.build_seed_split(0, args.eval_seeds))[1])
    rows = sweep(
        args.checkpoint,
        args.planner,
        args.arrival_rates,
        seeds,
        args.horizon,
        args.out / SWEEP_FILE,
    )
    print(f"\nwrote {len(rows)} rows to {args.out / SWEEP_FILE}")
    print(f"evaluation seeds {seeds}, horizon {args.horizon}, one checkpoint per arm")


if __name__ == "__main__":
    main()
