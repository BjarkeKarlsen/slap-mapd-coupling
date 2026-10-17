"""Evaluate a finished training run on the instance it was trained on.

A run of `slap-mapd train` writes its effective spec (`<name>_spec.json`) next to Tune's folder
of checkpoints. This module rebuilds the instance, the storage and the arm from that spec, so a
checkpoint is played exactly as it was trained, on any warehouse.

1. Validation. Every saved checkpoint plays the validation seeds of the run's seed split. The
   one that completes the most tasks on average is chosen, the later one on a tie. The training
   return is not used, because its shaping term dominates it.
2. Test. The chosen checkpoint and the last one play the test seeds under every storage mode
   asked for, F_fix when it is among them. An adaptive mode takes its epoch length, relocation
   cap and congestion weight from the spec's `storage_rule` section. The choice in step 1 does
   not change with the modes, it is always made under the storage rule the run trained with.

The validation and test seeds are the same for every run (training/config.py::split_seeds), so
runs with different training seeds and different arms are scored on the same scenarios. Actions
are sampled, as in every reported result.

`summary.md` compares every adaptive mode with F_fix over the test seeds, pairing the runs seed
by seed (evaluation/comparison.py::compare_paired), once for the chosen and once for the last
checkpoint.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import NamedTuple, Sequence

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
) -> RunMetrics:
    """One sampled episode of `checkpoint` on the episode seed `seed`, under `storage_mode` or,
    when that is None, the storage rule the run trained with."""
    return run_episode(
        study.instance.graph,
        study.skus,
        study.counts,
        study.capacities,
        build_experiment_config(spec, seed, storage_mode),
        policy=ModulePolicy(load_module(checkpoint)),
        encoding_config=study.arm.encoding,
        zones=study.arm.zones,
        pooling=study.arm.pooling,
    )


def evaluate_run(
    run_dir: Path,
    out_dir: Path,
    storage_modes: Sequence[StorageMode] | None = None,
    arrival_rate: float | None = None,
) -> RunEvaluation:
    """Chooses the checkpoint of the run in `run_dir` on the validation seeds and plays it and
    the last checkpoint on the test seeds under each of `storage_modes`, by default only the
    storage rule the run trained with. Writes `validation.jsonl`, `test.jsonl` and `summary.md`
    to `out_dir`, replacing earlier ones. The two row files get one row per episode as it
    finishes. Raises before any episode when a mode needs settings the spec lacks.

    `arrival_rate` plays the test seeds at another load than the run trained at, for example a
    load where the fleet keeps up, so service time counts. The checkpoint is still chosen at
    the training load, so the same checkpoint is measured at every load."""
    spec = load_run_spec(run_dir)
    test_spec = spec if arrival_rate is None else override_spec(spec, arrival_rate=arrival_rate)
    checkpoints = checkpoints_of(run_dir)
    if not checkpoints:
        raise ValueError(f"{run_dir} holds no checkpoints.")
    modes = list(dict.fromkeys(storage_modes or [training_storage_mode(spec)]))
    for mode in modes:
        check_storage_rule(spec, mode)
    study = build_study(spec)
    seeds = split_seeds(study.seed_split)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_path, test_path = out_dir / "validation.jsonl", out_dir / "test.jsonl"
    for path in (validation_path, test_path):
        path.write_text("", encoding="utf-8")

    validation_rows: list[dict] = []
    scores: list[tuple[float, int, Path]] = []
    for index, checkpoint in enumerate(checkpoints):
        completed = []
        for seed in seeds.validation:
            metrics = play(study, spec, checkpoint, seed)
            completed.append(metrics.num_completed_tasks)
            row = {"checkpoint": checkpoint.name, **metrics.model_dump(mode="json")}
            validation_rows.append(row)
            _append(validation_path, row)
        scores.append((statistics.mean(completed), index, checkpoint))
    chosen = max(scores, key=lambda score: (score[0], score[1]))[2]
    last = checkpoints[-1]

    test_rows: list[dict] = []
    played: dict[tuple[str, str], list[RunMetrics]] = {}
    for role, checkpoint in (("chosen", chosen), ("last", last)):
        for mode in modes:
            for seed in seeds.test:
                metrics = play(study, test_spec, checkpoint, seed, mode)
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


_SUMMARY_HEADER = (
    "| checkpoint | storage mode | seeds | all keep up | throughput F_fix | throughput mode "
    "| service time F_fix | service time mode | mean gain | CI | shortens |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _summary(modes: list[StorageMode], played: dict[tuple[str, str], list[RunMetrics]]) -> str:
    """The markdown of `summary.md`. Each adaptive mode is paired with F_fix over the same test
    seeds. The gain is F_fix minus the mode in mean service time, so a positive gain means the
    adaptive rule is faster, and it is only given when every run keeps up."""
    lines = ["# Adaptive storage against fixed storage on the test seeds", ""]
    adaptive = [mode for mode in modes if mode != "fixed"]
    if "fixed" not in modes or not adaptive:
        lines.append("No comparison, it needs F_fix and at least one adaptive mode.")
        return "\n".join(lines) + "\n"
    lines += [_SUMMARY_HEADER.rstrip("\n")]
    for role in ("chosen", "last"):
        for mode in adaptive:
            try:
                result = compare_paired(played[(role, "fixed")], played[(role, mode)])
            except ValueError as error:
                lines.append(f"| {role} | {mode} | {error} |" + " |" * 8)
                continue
            lines.append(_summary_row(role, result))
    return "\n".join(lines) + "\n"


def _summary_row(role: str, result: PairedComparison) -> str:
    def mean(summary: dict, name: str) -> str:
        return f"{summary[name].mean:.3f}" if name in summary else "n/a"

    gain = ci = "n/a"
    if result.mean_gain is not None:
        gain = f"{result.mean_gain:.3f}"
        ci = f"[{result.ci_low:.3f}, {result.ci_high:.3f}]"
    cells = [
        role,
        result.adaptive_storage_mode,
        str(result.num_seeds),
        str(result.all_keep_up),
        mean(result.fixed_summary, "throughput"),
        mean(result.adaptive_summary, "throughput"),
        mean(result.fixed_summary, "mean_service_time"),
        mean(result.adaptive_summary, "mean_service_time"),
        gain,
        ci,
        str(result.shortens_service_time),
    ]
    return "| " + " | ".join(cells) + " |"


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")
