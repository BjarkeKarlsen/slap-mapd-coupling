"""Evaluate a finished training run on the instance it was trained on.

A run of `slap-mapd train` writes its effective spec (`<name>_spec.json`) next to Tune's folder
of checkpoints. This module rebuilds the instance, the storage and the arm from that spec, so a
checkpoint is played exactly as it was trained, on any warehouse.

1. Validation. Every saved checkpoint plays the validation seeds of the run's seed split. The
   one that completes the most tasks on average is chosen, the later one on a tie. The training
   return is not used, because its shaping term dominates it.
2. Test. The chosen checkpoint and the last one play the test seeds.

The validation and test seeds are the same for every run (training/config.py::split_seeds), so
runs with different training seeds and different arms are scored on the same scenarios. Actions
are sampled, as in every reported result.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import NamedTuple

from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.policy import ModulePolicy
from slap_mapd_coupling.training.spec import (
    Study,
    TrainingSpec,
    build_experiment_config,
    build_study,
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


def play(study: Study, spec: TrainingSpec, checkpoint: Path, seed: int) -> RunMetrics:
    """One sampled episode of `checkpoint` on the episode seed `seed`."""
    return run_episode(
        study.instance.graph,
        study.skus,
        study.counts,
        study.capacities,
        build_experiment_config(spec, seed),
        policy=ModulePolicy(load_module(checkpoint)),
        encoding_config=study.arm.encoding,
        zones=study.arm.zones,
        pooling=study.arm.pooling,
    )


def evaluate_run(run_dir: Path, out_dir: Path) -> RunEvaluation:
    """Chooses the checkpoint of the run in `run_dir` on the validation seeds and plays it and
    the last checkpoint on the test seeds. Writes `validation.jsonl` and `test.jsonl` to
    `out_dir`, replacing earlier ones, one row per episode as it finishes."""
    spec = load_run_spec(run_dir)
    checkpoints = checkpoints_of(run_dir)
    if not checkpoints:
        raise ValueError(f"{run_dir} holds no checkpoints.")
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
    for role, checkpoint in (("chosen", chosen), ("last", last)):
        for seed in seeds.test:
            metrics = play(study, spec, checkpoint, seed)
            row = {"role": role, "checkpoint": checkpoint.name, **metrics.model_dump(mode="json")}
            test_rows.append(row)
            _append(test_path, row)
    return RunEvaluation(chosen, last, validation_rows, test_rows)


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")
