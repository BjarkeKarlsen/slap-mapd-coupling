"""Command-line interface for slap_mapd_coupling.

Five verbs, following the experimental design of sec:pd-aim-and-scope.

  simulate  runs one configuration, a storage rule with a controller or the
            reference planner, at one load.
  train     trains a learned controller with PPO. The reference planner is
            not learned.
  resume    continues a training run that was stopped, from its latest
            checkpoint.
  evaluate  chooses a trained run's checkpoint on the validation seeds and
            plays it on the test seeds.
  sweep     runs the experimental grid from a config file and writes one
            results table.
"""

from typing import get_args

import click

from slap_mapd_coupling.planners.registry import PlannerName

STORAGE_RULES = ["fixed", "demand", "congestion"]
CONTROLLERS = ["centralised", "section", "decentralised"]
PLANNERS = list(get_args(PlannerName))


@click.group()
@click.version_option()
def cli():
    """SLAP-MAPD coupling: warehouse storage/routing simulation and RL training."""


@cli.command()
@click.option(
    "--storage-rule",
    type=click.Choice(STORAGE_RULES),
    default="fixed",
    help="Storage update rule F.",
)
@click.option(
    "--controller",
    type=click.Choice(CONTROLLERS),
    default=None,
    help="Routing/assignment architecture. Give this or --planner, not both.",
)
@click.option(
    "--planner",
    type=click.Choice(PLANNERS),
    default=None,
    help="Non-learned reference planner, outside the three controllers.",
)
@click.option(
    "--congestion-sensitive",
    is_flag=True,
    help="Enable congestion sensitivity in storage and routing.",
)
@click.option(
    "--communication",
    is_flag=True,
    help="Enable local communication (decentralised controller only).",
)
@click.option("--num-agents", default=10, type=int, help="Fleet size.")
@click.option(
    "--arrival-rate", default=1.0, type=float, help="Expected tasks released per timestep."
)
@click.option("--num-episodes", default=10, type=int, help="Number of episodes to run.")
@click.option("--seed", default=0, type=int, help="Episode/environment seed.")
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Trained policy (required iff --controller=decentralised).",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Where to write the metrics row(s) (CSV). Prints to stdout if omitted.",
)
def simulate(
    storage_rule,
    controller,
    planner,
    congestion_sensitive,
    communication,
    num_agents,
    arrival_rate,
    num_episodes,
    seed,
    checkpoint,
    output,
):
    """Run episodes for one configuration and report the evaluation metrics."""
    if (controller is None) == (planner is None):
        raise click.UsageError("Give exactly one of --controller and --planner.")
    if controller == "decentralised" and checkpoint is None:
        raise click.UsageError("--checkpoint is required when --controller=decentralised.")
    if controller != "decentralised" and checkpoint is not None:
        raise click.UsageError("--checkpoint only applies to --controller=decentralised.")
    if controller != "decentralised" and communication:
        raise click.UsageError("--communication only applies to --controller=decentralised.")
    raise NotImplementedError("Simulation is not implemented yet.")


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Training config (YAML), for example configs/decentralised.yaml. It holds the "
    "instance, the model, the PPO hyperparameters and the checkpointing. The options below "
    "override its settings.",
)
@click.option("--controller", type=click.Choice(CONTROLLERS), default=None)
@click.option(
    "--regime",
    type=click.Choice(["fixed", "demand", "congestion"]),
    default=None,
    help="Training regime (sec:method:training), the storage rule training runs under.",
)
@click.option(
    "--communication/--no-communication",
    default=None,
    help="Local communication (RQ5, decentralised controller only).",
)
@click.option("--arrival-rate", type=float, default=None, help="Tasks released per step.")
@click.option("--horizon", type=int, default=None, help="Episode length in steps.")
@click.option(
    "--budget-env-steps", type=int, default=None, help="Training budget in environment steps."
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Training seed, RLlib's seed and the first training scenario. The validation and "
    "test seeds do not move with it.",
)
@click.option("--env-runners", default=0, type=int, help="Number of RLlib env runners.")
@click.option("--gpu", is_flag=True, help="Use one GPU for the learner.")
@click.option(
    "--batch-on-cpu",
    is_flag=True,
    help="Keep the train batch in host memory and move one minibatch at a time to the GPU, "
    "for runs whose batch does not fit on the GPU. Changes no result.",
)
@click.option(
    "--value-chunk-rows",
    type=int,
    default=None,
    help="Rows per encoder pass of the value pass (default 512). Lower it if the value pass "
    "runs out of GPU memory. Changes no result.",
)
@click.option(
    "--learner-cpus",
    type=click.IntRange(min=1),
    default=1,
    help="CPUs of the learner, the cores its update uses without a GPU (default 1). "
    "Changes no result. A resumed run keeps the value it started with.",
)
@click.option(
    "--checkpoint-every",
    type=int,
    default=None,
    help="Iterations between checkpoints, 0 keeps only the one at the end.",
)
@click.option(
    "--keep-checkpoints",
    type=int,
    default=None,
    help="Checkpoints to keep, the best by episode return and the latest.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="./checkpoints",
    help="Where Ray Tune writes the run, with its checkpoints.",
)
def train(
    config,
    controller,
    regime,
    communication,
    arrival_rate,
    horizon,
    budget_env_steps,
    seed,
    env_runners,
    gpu,
    batch_on_cpu,
    value_chunk_rows,
    learner_cpus,
    checkpoint_every,
    keep_checkpoints,
    output,
):
    """Train a learned controller (PPO). The reference planner needs no training."""
    # Imported here so the other verbs do not pay for starting Ray.
    from pathlib import Path

    from slap_mapd_coupling.training.spec import (
        build_study,
        build_study_ppo_config,
        load_spec,
        override_spec,
    )
    from slap_mapd_coupling.training.trainer import train as run_training

    try:
        spec = override_spec(
            load_spec(Path(config)),
            controller=controller,
            regime=regime,
            communication=communication,
            arrival_rate=arrival_rate,
            horizon=horizon,
            budget_env_steps=budget_env_steps,
            seed=seed,
            checkpoint_every=checkpoint_every,
            keep_checkpoints=keep_checkpoints,
        )
        study = build_study(spec)
        ppo_config = build_study_ppo_config(
            spec,
            study,
            num_env_runners=env_runners,
            use_gpu=gpu,
            batch_on_cpu=batch_on_cpu,
            value_chunk_rows=value_chunk_rows,
            learner_cpus=learner_cpus,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    out = Path(output)
    name = f"{spec.controller}_{spec.regime}" + ("_comm" if spec.communication else "")
    out.mkdir(parents=True, exist_ok=True)
    # The effective settings, so a checkpoint can be evaluated as it was trained.
    (out / f"{name}_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    run = run_training(
        ppo_config,
        spec.ppo.training_budget_env_steps,
        name,
        arm_label=spec.controller,
        settings=spec.checkpoint,
        storage_path=out,
        progress_path=out / f"{name}_training_progress.jsonl",
        validate_in=out,
    )
    _report_run(run, spec, out)


def _report_run(run, spec, out) -> None:
    """Prints where a run's checkpoints are, and fails when it stopped before its budget (Tune
    stops cleanly on Ctrl-C), so a script running it does not take it as finished."""
    click.echo(f"Checkpoint saved to {run.checkpoint}")
    click.echo(
        f"Best checkpoint by {spec.checkpoint.score_attribute} saved to {run.best_checkpoint}"
    )
    budget = spec.ppo.training_budget_env_steps
    if run.env_steps_sampled < budget:
        raise click.ClickException(
            f"Stopped at {run.env_steps_sampled} of {budget} environment steps. "
            f"Continue it with: slap-mapd resume {out}"
        )


@cli.command()
@click.argument("output", type=click.Path(exists=True, file_okay=False))
def resume(output):
    """Continue the training run in OUTPUT (the --output of train) from its latest checkpoint.
    It runs on to the budget the run started with, with the same runners and settings, and
    keeps writing the same progress file. The iterations after the checkpoint are trained
    again."""
    from pathlib import Path

    from slap_mapd_coupling.training.spec import (
        TrainingSpec,
        build_study,
        build_study_ppo_config,
    )
    from slap_mapd_coupling.training.trainer import resume as resume_training

    out = Path(output)
    specs = sorted(out.glob("*_spec.json"))
    if len(specs) != 1:
        raise click.UsageError(f"Expected one *_spec.json in {out}, found {len(specs)}.")
    spec = TrainingSpec.model_validate_json(specs[0].read_text(encoding="utf-8"))
    name = specs[0].name.removesuffix("_spec.json")
    try:
        run = resume_training(
            build_study_ppo_config(spec, build_study(spec)),
            name,
            out,
            settings=spec.checkpoint,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    _report_run(run, spec, out)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Where validation.jsonl, test.jsonl and summary.md go. Default: RUN_DIR/evaluation, "
    "or RUN_DIR/evaluation_rate<rate> with --arrival-rate.",
)
@click.option(
    "--storage-modes",
    type=click.Choice(STORAGE_RULES),
    multiple=True,
    help="Storage rules the test seeds are played under, repeated per rule, e.g. "
    "--storage-modes fixed --storage-modes demand. "
    "Default: the rule the run trained with. An adaptive rule takes its epoch length, "
    "relocation cap and congestion weight from the storage_rule section of the run's spec. "
    "With an arrival section in the spec every rule takes the epoch length, the arrival "
    "period, from it, the congestion rule its weight, and there is no relocation cap.",
)
@click.option(
    "--arrival-rate",
    type=float,
    default=None,
    help="Load the test seeds are played at, in tasks per step. Default: the training load. "
    "The checkpoint is still chosen at the training load. A starting stock sized by the load "
    "(demand_sized_blocks) is sized at this load for the test seeds.",
)
@click.option(
    "--trace-seeds",
    type=int,
    default=0,
    help="Write a per-step trace of the first N test seeds of each checkpoint and rule to "
    "OUTPUT/traces/, for the viewer (python -m slap_mapd_coupling.viz.pygame_renderer).",
)
@click.option(
    "--horizon",
    type=int,
    default=None,
    help="Episode length of the test seeds, in steps. Default: the run's. The checkpoint is "
    "still chosen with the run's own episodes.",
)
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Warm-up of the test seeds, the first steps left out of every measure. Default: the "
    "run's. With --horizon 10000 --warmup 5000 the test seeds play as the Token Passing runs "
    "of the appendices did.",
)
@click.option(
    "--test-seeds",
    type=click.IntRange(min=1),
    default=None,
    help="Number of test seeds to play. Default: the run's own count. The validation seeds, "
    "and so the chosen checkpoint, do not change.",
)
@click.option(
    "--gpu",
    is_flag=True,
    default=False,
    help="Run the policy's forward pass on the GPU (cuda). The episodes still run on the CPU.",
)
def evaluate(
    run_dir, output, storage_modes, arrival_rate, trace_seeds, horizon, warmup, test_seeds, gpu
):
    """Evaluate the training run in RUN_DIR (the --output of train). Every checkpoint plays the
    validation seeds under the rule the run trained with. Among those that keep up on every
    seed the one with the shortest pick service time is chosen ([A60]), and it and the last
    checkpoint play the test seeds under every storage rule given. summary.md compares each adaptive rule with fixed storage."""
    from pathlib import Path

    from slap_mapd_coupling.training.evaluate_run import evaluate_run

    run = Path(run_dir)
    default_name = "evaluation" + "".join(
        f"_{name}{value:g}"
        for name, value in (("rate", arrival_rate), ("horizon", horizon), ("warmup", warmup))
        if value is not None
    )
    out = Path(output) if output is not None else default_output(run, default_name)
    try:
        result = evaluate_run(
            run,
            out,
            list(storage_modes) or None,
            arrival_rate,
            trace_seeds,
            horizon,
            warmup,
            test_seeds,
            "cuda" if gpu else None,
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    for role, checkpoint in (("chosen", result.chosen), ("last", result.last)):
        for mode in dict.fromkeys(row["storage_mode"] for row in result.test_rows):
            rows = [
                row
                for row in result.test_rows
                if row["role"] == role and row["storage_mode"] == mode
            ]
            throughput = sum(row["throughput"] for row in rows) / len(rows)
            click.echo(
                f"{role} {checkpoint.name} ({mode}): throughput {throughput:.3f} "
                f"on {len(rows)} seeds"
            )
    click.echo(f"Rows written to {out}")


def default_output(run, name):
    """Where an evaluation goes when no --output is given: beside the training runs, in
    evaluation/ in place of training/ (~/thesis_runs/evaluation/fleet_16/section/fixed/<name>), so
    the results are apart from the runs. A run outside a training/ folder keeps it inside."""
    from pathlib import Path

    parts = run.resolve().parts
    if "training" in parts:
        i = len(parts) - 1 - parts[::-1].index("training")
        return Path(*parts[:i], "evaluation", *parts[i + 1 :], name)
    return run / name


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("checkpoint", type=click.Path(exists=True, file_okay=False))
def validate(run_dir, checkpoint):
    """Play CHECKPOINT of the run in RUN_DIR on the validation seeds, under the rule and episode
    length it trained with, into RUN_DIR/validation/<checkpoint>.jsonl. Training calls it for
    every checkpoint it saves. evaluate reads the rows back."""
    from pathlib import Path

    from slap_mapd_coupling.training.config import split_seeds
    from slap_mapd_coupling.training.evaluate_run import load_run_spec, validate_checkpoint
    from slap_mapd_coupling.training.spec import build_study

    run = Path(run_dir)
    spec = load_run_spec(run)
    study = build_study(spec)
    episodes = validate_checkpoint(
        run, Path(checkpoint), study, spec, split_seeds(study.seed_split).validation
    )
    click.echo(
        f"{Path(checkpoint).name}: {len(episodes)} validation seeds, keep up "
        f"{sum(bool(m.keeps_up) for m in episodes)}"
    )


@cli.command("prepare-warehouses")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Training config (YAML) with warehouse_per_seed (D27).",
)
@click.option(
    "--workers", type=int, default=1, help="Processes that build warehouses at once."
)
def prepare_warehouses(config, workers):
    """Build and store the window tables of every seed of the config's split, the training,
    validation and test warehouses, before training or evaluation starts. Seeds already stored
    are skipped. The store is the config's warehouse_store, ~/thesis_runs/warehouses, or
    SLAP_WAREHOUSE_STORE when it is set."""
    from pathlib import Path

    from slap_mapd_coupling.training.config import split_seeds
    from slap_mapd_coupling.training.spec import build_study, load_spec
    from slap_mapd_coupling.training.warehouses import WarehousePool

    spec = load_spec(Path(config))
    if not spec.warehouse_per_seed:
        raise click.UsageError(f"{config} has no warehouse_per_seed, so it has no warehouses.")
    study = build_study(spec, prepare=False)
    seeds = split_seeds(study.seed_split)
    pool: WarehousePool = study.pool
    every = [*seeds.train, *seeds.validation, *seeds.test]
    manifest = pool.prepare(every, workers=workers)
    click.echo(
        f"{len(every)} warehouses in {pool.root}, cap {manifest['max_local_nodes']}, "
        f"d_max {manifest['d_max']}."
    )


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Experiment-grid config (YAML): storage rules x controllers x congestion "
    "x communication x load x seeds.",
)
@click.option(
    "--output", type=click.Path(), default="./results.csv", help="Aggregated results table."
)
@click.option(
    "--dry-run", is_flag=True, help="Print the resulting experiment grid without running it."
)
def sweep(config, output, dry_run):
    """Run the full experimental grid, training the learned controllers as
    needed, and write one results table."""
    raise NotImplementedError("Sweep is not implemented yet.")


if __name__ == "__main__":
    cli()
