"""Evaluate a finished training run on the instance it was trained on.

A run of `slap-mapd train` writes its effective spec (`<name>_spec.json`) next to Tune's folder
of checkpoints. This module rebuilds the instance, the storage and the arm from that spec, so a
checkpoint is played exactly as it was trained, on any warehouse.

1. Validation. Every saved checkpoint plays the validation seeds of the run's seed split. The
   checkpoint is chosen by [A60]: among those that keep up on every validation seed, the one
   with the shortest mean pick service time, or with none keeping up the highest throughput,
   the later one on a tie. The training return is not used, because its shaping term dominates
   it, and nor is the number of completed tasks, which follows how many orders arrived.
2. Test. The chosen checkpoint and the last one play the test seeds under every storage mode
   asked for, F_fix when it is among them. Under the template stock model an adaptive mode
   takes its epoch length, relocation cap and congestion weight from the spec's `storage_rule`
   section. Under the arrival stock model (the spec's `arrival` section) every mode takes the
   epoch length, the arrival period, from it, and the congestion rule its weight, and nothing
   is relocated. The choice in step 1 does not change with the modes, it is always made under
   the storage rule the run trained with.

The validation and test seeds are the same for every run (training/config.py::split_seeds), so
runs with different training seeds and different arms are scored on the same scenarios. Actions
are sampled, as in every reported result.

`summary.md` compares every adaptive mode with F_fix over the test seeds, pairing the runs seed
by seed (evaluation/comparison.py::compare_paired), once for the chosen and once for the last
checkpoint. For an arrival run it also gives the wait for stock of both arms.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, NamedTuple, Sequence

# The adaptive rules register themselves on import. Training imports them in every runner
# (training/config.py), and evaluation under --storage-modes needs them in this process.
import slap_mapd_coupling.storage.congestion  # noqa: F401
import slap_mapd_coupling.storage.demand  # noqa: F401
from slap_mapd_coupling.core.experiment_config import StorageMode
from slap_mapd_coupling.evaluation.comparison import PairedComparison, compare_paired
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.policy import ModulePolicy
from slap_mapd_coupling.training.spec import (
    Study,
    TrainingSpec,
    build_experiment_config,
    build_study,
    check_storage_rule,
    override_spec,
    training_storage_mode,
)
from slap_mapd_coupling.training.trainer import load_module


class RunEvaluation(NamedTuple):
    chosen: Path
    last: Path
    validation_rows: list[dict]
    test_rows: list[dict]


def load_run_spec(run_dir: Path) -> TrainingSpec:
    """The spec a run was trained with, the one `<name>_spec.json` in `run_dir`."""
    found = sorted(run_dir.glob("*_spec.json"))
    if len(found) != 1:
        raise ValueError(f"{run_dir} must hold one *_spec.json, found {len(found)}.")
    return TrainingSpec.model_validate_json(found[0].read_text(encoding="utf-8"))


def checkpoints_of(run_dir: Path) -> list[Path]:
    """The checkpoints of a run, in the order they were saved."""
    found = list(run_dir.glob("*/PPO_*/checkpoint_*"))
    return sorted(found, key=lambda path: int(path.name.split("_")[-1]))


def play(
    study: Study,
    spec: TrainingSpec,
    checkpoint: Path,
    seed: int,
    storage_mode: StorageMode | None = None,
    trace_path: Path | None = None,
    device: str | None = None,
) -> RunMetrics:
    """One sampled episode of `checkpoint` on the episode seed `seed`, under `storage_mode` or,
    when that is None, the storage rule the run trained with. With a warehouse per seed the
    episode plays the warehouse of `seed` (D27). With `trace_path` it writes a per-step trace
    there, with the drawing positions of the warehouse it played. `device`, such as "cuda",
    runs the policy's forward pass there, as training's learner did on a GPU."""
    instance, skus, counts, capacities, zones = (
        study.instance, study.skus, study.counts, study.capacities, study.arm.zones
    )  # fmt: skip
    if study.pool is not None:
        built = study.pool.warehouse(seed)
        instance, skus, counts, capacities, zones = (
            built.instance, built.skus, built.counts, built.capacities, built.zones
        )  # fmt: skip
    return run_episode(
        instance.graph,
        skus,
        counts,
        capacities,
        build_experiment_config(spec, seed, storage_mode),
        trace_path,
        positions=instance.positions,
        policy=ModulePolicy(load_module(checkpoint), device),
        encoding_config=study.arm.encoding,
        zones=zones,
        pooling=study.arm.pooling,
        pool=study.pool,
    )


def build_test_study(
    spec: TrainingSpec,
    study: Study,
    arrival_rate: float | None,
    horizon: int | None = None,
    warmup: int | None = None,
) -> tuple[TrainingSpec, Study]:
    """The spec and the study the test episodes play: `spec` and `study` themselves, or the
    spec with the `arrival_rate`, `horizon` and `warmup` given. The starting stock of
    "demand_sized_blocks" is sized by the arrival rate, so at another load the study is rebuilt
    and the test episodes start from stock sized at the load they play. The episode length and
    the warm-up change no part of the study, so with those alone it is the training study.
    Raises ValueError when the warm-up leaves no step to measure."""
    if arrival_rate is None and horizon is None and warmup is None:
        return spec, study
    test_spec = override_spec(spec, arrival_rate=arrival_rate, horizon=horizon, warmup=warmup)
    if test_spec.warmup >= test_spec.horizon:
        raise ValueError(
            f"The warm-up of {test_spec.warmup} steps leaves nothing of an episode of "
            f"{test_spec.horizon} steps to measure."
        )
    if arrival_rate is None:
        return test_spec, study
    return test_spec, build_study(test_spec)


def choose_checkpoint(validation: Sequence[Sequence[Any]]) -> int:
    """The index of the checkpoint to test, from its validation episodes in the order the
    checkpoints were saved ([A60]). Among the checkpoints that keep up on every validation seed,
    the one with the shortest mean pick service time. If none does, the one with the highest
    mean throughput, and the arm is reported by its throughput. A tie goes to the later
    checkpoint. The number of completed tasks is not used, since it follows how many orders
    arrived more than how well they were served."""

    def keeps_up(episodes: Sequence[Any]) -> bool:
        return all(getattr(m, "keeps_up", False) for m in episodes)

    def service(episodes: Sequence[Any]) -> float:
        times = [m.mean_service_time for m in episodes if m.mean_service_time is not None]
        return statistics.mean(times) if times else float("inf")

    passing = [i for i, episodes in enumerate(validation) if keeps_up(episodes)]
    if passing:
        return min(passing, key=lambda i: (service(validation[i]), -i))
    return max(
        range(len(validation)),
        key=lambda i: (statistics.mean(m.throughput for m in validation[i]), i),
    )


def evaluate_run(
    run_dir: Path,
    out_dir: Path,
    storage_modes: Sequence[StorageMode] | None = None,
    arrival_rate: float | None = None,
    trace_seeds: int = 0,
    horizon: int | None = None,
    warmup: int | None = None,
    test_seeds: int | None = None,
    device: str | None = None,
) -> RunEvaluation:
    """Chooses the checkpoint of the run in `run_dir` on the validation seeds
    (`choose_checkpoint`, [A60]) and plays it and
    the last checkpoint on the test seeds under each of `storage_modes`, by default only the
    storage rule the run trained with. Writes `validation.jsonl`, `test.jsonl` and `summary.md`
    to `out_dir`, replacing earlier ones. The two row files get one row per episode as it
    finishes. Raises before any episode when a mode needs settings the spec lacks.

    `arrival_rate` plays the test seeds at another load than the run trained at, for example a
    load where the fleet keeps up, so service time counts. The checkpoint is still chosen at
    the training load, so the same checkpoint is measured at every load. The test episodes
    start from the stock of that load (`build_test_study`). `horizon` and `warmup` play the test
    seeds for another episode length and warm-up, for example the 10000 steps with a 5000-step
    warm-up of the Token Passing runs, while training and the choice keep the run's own.

    The first `trace_seeds` test seeds of each checkpoint and rule also write a per-step trace
    to `out_dir/traces/`, for the viewer (viz/pygame_renderer.py).

    `test_seeds` plays that many test seeds instead of the count the run was trained with, for
    example 20 for a run that stored 6 ([A91]). The test seeds stay the same range, longer.
    The validation seeds, and so the chosen checkpoint, do not change. `device`, such as
    "cuda", runs every forward pass there."""
    spec = load_run_spec(run_dir)
    if test_seeds is not None:
        spec = spec.model_copy(update={"test_seed_count": test_seeds})
    checkpoints = checkpoints_of(run_dir)
    if not checkpoints:
        raise ValueError(f"{run_dir} holds no checkpoints.")
    modes = list(dict.fromkeys(storage_modes or [training_storage_mode(spec)]))
    for mode in modes:
        check_storage_rule(spec, mode)
    study = build_study(spec)
    test_spec, test_study = build_test_study(spec, study, arrival_rate, horizon, warmup)
    seeds = split_seeds(study.seed_split)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_path, test_path = out_dir / "validation.jsonl", out_dir / "test.jsonl"
    for path in (validation_path, test_path):
        path.write_text("", encoding="utf-8")

    validation_rows: list[dict] = []
    played_validation: list[list] = []
    for checkpoint in checkpoints:
        episodes = []
        for seed in seeds.validation:
            metrics = play(study, spec, checkpoint, seed, device=device)
            episodes.append(metrics)
            row = {"checkpoint": checkpoint.name, **metrics.model_dump(mode="json")}
            validation_rows.append(row)
            _append(validation_path, row)
        played_validation.append(episodes)
    chosen = checkpoints[choose_checkpoint(played_validation)]
    last = checkpoints[-1]

    test_rows: list[dict] = []
    played: dict[tuple[str, str], list[RunMetrics]] = {}
    for role, checkpoint in (("chosen", chosen), ("last", last)):
        for mode in modes:
            for n, seed in enumerate(seeds.test):
                trace = None
                if n < trace_seeds:
                    trace = out_dir / "traces" / f"{role}_{mode}_seed{seed}.jsonl"
                    trace.parent.mkdir(parents=True, exist_ok=True)
                metrics = play(test_study, test_spec, checkpoint, seed, mode, trace, device=device)
                played.setdefault((role, mode), []).append(metrics)
                row = {
                    "role": role,
                    "checkpoint": checkpoint.name,
                    **metrics.model_dump(mode="json"),
                    "storage_mode": mode,
                }
                test_rows.append(row)
                _append(test_path, row)
    (out_dir / "summary.md").write_text(_summary(modes, played), encoding="utf-8")
    return RunEvaluation(chosen, last, validation_rows, test_rows)


# The gain in mean pick service time and the relative gain lead. The order completion time
# (arrival runs) follows.
_SUMMARY_HEADER = (
    "| checkpoint | storage mode | seeds | all keep up | gain | CI | relative gain % [CI] "
    "| shortens | throughput F_fix | throughput mode "
    "| total service F_fix | total service mode | pick service F_fix | pick service mode |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)
# The columns an arrival run adds to every row: the wait for stock of both arms, and the order
# completion time under each rule, F_fix / mode.
_ARRIVAL_HEADER = (
    " stock wait F_fix | stock wait mode | order completion F_fix / mode |",
    "---|---|---|",
)


def _summary(modes: list[StorageMode], played: dict[tuple[str, str], list[RunMetrics]]) -> str:
    """The markdown of `summary.md`. Each adaptive mode is paired with F_fix over the same test
    seeds. The gain is F_fix minus the mode in mean pick service time, so a positive gain means the
    adaptive rule is faster, and it is only given when every run keeps up."""
    lines = ["# Adaptive storage against fixed storage on the test seeds", ""]
    adaptive = [mode for mode in modes if mode != "fixed"]
    if "fixed" not in modes or not adaptive:
        lines.append("No comparison, it needs F_fix and at least one adaptive mode.")
        return "\n".join(lines) + "\n"
    header = _SUMMARY_HEADER.rstrip("\n").split("\n")
    arrival = any(run.stock_model == "arrival" for runs in played.values() for run in runs)
    if arrival:
        header = [line + extra for line, extra in zip(header, _ARRIVAL_HEADER)]
    lines += header
    for role in ("chosen", "last"):
        for mode in adaptive:
            try:
                result = compare_paired(played[(role, "fixed")], played[(role, mode)])
            except ValueError as error:
                lines.append(f"| {role} | {mode} | {error} |" + " |" * (14 if arrival else 11))
                continue
            row = _summary_row(role, result, arrival)
            if arrival:
                fixed_time = _mean_order_completion(played[(role, "fixed")])
                mode_time = _mean_order_completion(played[(role, mode)])
                row += f" {fixed_time} / {mode_time} |"
            lines.append(row)
    return "\n".join(lines) + "\n"


def _summary_row(role: str, result: PairedComparison, arrival: bool = False) -> str:
    def mean(summary: dict, name: str) -> str:
        return f"{summary[name].mean:.3f}" if name in summary else "n/a"

    gain = ci = relative = "n/a"
    if result.mean_gain is not None:
        gain = f"{result.mean_gain:.3f}"
        ci = f"[{result.ci_low:.3f}, {result.ci_high:.3f}]"
    if result.mean_relative_gain is not None:
        relative = (
            f"{result.mean_relative_gain:.3f} "
            f"[{result.relative_ci_low:.3f}, {result.relative_ci_high:.3f}]"
        )
    cells = [
        role,
        result.adaptive_storage_mode,
        str(result.num_seeds),
        str(result.all_keep_up),
        gain,
        ci,
        relative,
        str(result.shortens_service_time),
        mean(result.fixed_summary, "throughput"),
        mean(result.adaptive_summary, "throughput"),
        mean(result.fixed_summary, "total_service_time"),
        mean(result.adaptive_summary, "total_service_time"),
        mean(result.fixed_summary, "mean_service_time"),
        mean(result.adaptive_summary, "mean_service_time"),
    ]
    if arrival:
        cells += [
            mean(result.fixed_summary, "mean_stock_wait"),
            mean(result.adaptive_summary, "mean_stock_wait"),
        ]
    return "| " + " | ".join(cells) + " |"


def _mean_order_completion(runs: list[RunMetrics]) -> str:
    """The order completion time averaged over the runs that completed an order."""
    times = [run.mean_order_completion for run in runs if run.mean_order_completion is not None]
    return f"{sum(times) / len(times):.3f}" if times else "n/a"


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")
