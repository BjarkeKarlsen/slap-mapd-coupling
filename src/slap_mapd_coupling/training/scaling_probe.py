"""Play a trained checkpoint in a bigger warehouse than the one it trained in, to see how the
decision time and the throughput grow with the fleet.

The probe starts from the run's own spec, so the arm, the model, the reward and the storage
rule stay as trained. From the target fleet's config it takes only what sets the size: the
number of agents, the load, the warehouse (`instance`) and the SKUs (`storage`). The section
arm keeps the agents per section of training, so its number of sections is given with
`--sections` (3 x fleet / 16 for a fleet 16 run). Trained runs themselves keep 3 sections.

Only the probe's own seed is prepared, not the run's whole seed split, so a big warehouse is
built once. The episode is sampled, as in every reported result.

    python -m slap_mapd_coupling.training.scaling_probe RUN_DIR CHECKPOINT \\
        --target configs/fleet_64/section.yaml --sections 12 --steps 500 --gpu --output OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.evaluate_run import load_run_spec
from slap_mapd_coupling.training.policy import ModulePolicy
from slap_mapd_coupling.training.trainer import load_module
from slap_mapd_coupling.training.spec import (
    POOLING,
    Arm,
    GNNEncoderConfig,
    PolicyValueHeadConfig,
    SeedSplitConfig,
    Study,
    TrainingSpec,
    build_experiment_config,
    load_spec,
)


def probe_spec(
    run_spec: TrainingSpec,
    target: TrainingSpec,
    sections: int | None,
    steps: int,
    warmup: int = 0,
) -> TrainingSpec:
    """`run_spec` with the size of `target`: agents, load, warehouse and SKUs. The episode is
    `steps` long, the first `warmup` of them not measured. `sections`, when given, is the
    section arm's number of zones."""
    data = run_spec.model_dump()
    data["arrival_rate"] = target.arrival_rate
    data["experiment"]["num_agents"] = target.experiment.num_agents
    data["instance"] = target.model_dump()["instance"]
    data["storage"] = target.model_dump()["storage"]
    data["horizon"] = steps
    data["warmup"] = warmup
    if sections is not None:
        data["arm"]["num_zones"] = sections
    return TrainingSpec.model_validate(data)


def probe_study(spec: TrainingSpec, seed: int, pool_seeds: list[int] | None = None) -> Study:
    """The study of `spec` with the warehouse of `seed` prepared (D27), or of every seed in
    `pool_seeds`, so probes of several seeds sharing a store write the same manifest and can
    run at once."""
    from slap_mapd_coupling.training.warehouses import WarehousePool

    store = None if spec.warehouse_store is None else Path(spec.warehouse_store).expanduser()
    pool = WarehousePool(spec, store)
    pool.prepare(sorted({seed, *(pool_seeds or [])}))
    built = pool.warehouse(seed)
    config = build_experiment_config(spec)
    arm = Arm(
        controller=spec.controller,
        pooling=POOLING[spec.controller],
        zones=built.zones,
        encoding=pool.encoding(),
    )
    return Study(
        instance=built.instance,
        skus=built.skus,
        capacities=built.capacities,
        counts=built.counts,
        config=config,
        arm=arm,
        gnn_encoder=GNNEncoderConfig(
            num_rounds=spec.model.num_rounds,
            hidden_width=spec.model.hidden_width,
            communication=config.communication,
            pooling=arm.pooling,
        ),
        head=PolicyValueHeadConfig(hidden_width=spec.model.head_hidden_width),
        seed_split=SeedSplitConfig(
            training_seed=config.seed,
            train_seed_count=spec.train_seed_count,
            validation_seed_count=spec.validation_seed_count,
            test_seed_count=spec.test_seed_count,
        ),
        pool=pool,
    )


def play_probe(study: Study, spec: TrainingSpec, checkpoint: Path, seed: int, device: str | None) -> RunMetrics:
    """One sampled episode of `checkpoint` in the probe's warehouse. The module of a run with a
    warehouse per seed reads its window tables from the pool it trained with, so its lookup is
    pointed at the probe's pool first. The weights do not depend on the warehouse."""
    module = load_module(checkpoint)
    lookup = module.encoder.lookup
    if lookup is None or not hasattr(lookup, "_pool"):
        raise ValueError(f"{checkpoint} does not read its tables from a pool, so it cannot probe")
    lookup._pool = study.pool
    lookup._tables.clear()
    lookup._joined.clear()
    built = study.pool.warehouse(seed)
    return run_episode(
        built.instance.graph,
        built.skus,
        built.counts,
        built.capacities,
        build_experiment_config(spec, seed),
        None,
        positions=built.instance.positions,
        policy=ModulePolicy(module, device),
        encoding_config=study.arm.encoding,
        zones=built.zones,
        pooling=study.arm.pooling,
        pool=study.pool,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target", type=Path, required=True, help="the bigger fleet's config")
    parser.add_argument("--sections", type=int, default=None, help="section arm only")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None, help="default: the first test seed")
    parser.add_argument("--warmup", type=int, default=0, help="unmeasured steps within --steps")
    parser.add_argument(
        "--pool-seeds",
        type=int,
        nargs="+",
        default=None,
        help="prepare these seeds too; give every concurrent probe of a store the same list",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="a .jsonl file, appended")
    args = parser.parse_args(argv)

    run_spec = load_run_spec(args.run_dir)
    if args.sections is not None and run_spec.controller != "section":
        parser.error("--sections is for the section arm only")
    spec = probe_spec(run_spec, load_spec(args.target), args.sections, args.steps, args.warmup)
    if args.seed is None:
        split = SeedSplitConfig(
            training_seed=spec.experiment.seed,
            train_seed_count=spec.train_seed_count,
            validation_seed_count=spec.validation_seed_count,
            test_seed_count=spec.test_seed_count,
        )
        args.seed = split_seeds(split).test[0]

    started = time.perf_counter()
    study = probe_study(spec, args.seed, args.pool_seeds)
    built = time.perf_counter() - started
    metrics = play_probe(study, spec, args.checkpoint, args.seed, "cuda" if args.gpu else None)
    played = time.perf_counter() - started - built

    row = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint.name,
        "target": str(args.target),
        "num_agents": spec.experiment.num_agents,
        "num_zones": spec.arm.num_zones if spec.controller == "section" else None,
        "num_vertices": study.instance.graph.num_vertices
        if hasattr(study.instance.graph, "num_vertices")
        else None,
        "device": "cuda" if args.gpu else "cpu",
        "build_seconds": built,
        "play_seconds": played,
        "seconds_per_step": played / spec.horizon,
        "seed": args.seed,
        "horizon": spec.horizon,
        "warmup": spec.warmup,
        **metrics.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(
        f"{spec.controller} comm={spec.communication} m={spec.experiment.num_agents} "
        f"zones={row['num_zones']} decision {metrics.mean_decision_runtime_seconds:.4f} s, "
        f"step {row['seconds_per_step']:.4f} s, throughput {metrics.throughput:.3f}, "
        f"built {built:.0f} s"
    )


if __name__ == "__main__":
    main()
