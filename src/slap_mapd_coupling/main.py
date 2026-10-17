"""Command-line interface for slap_mapd_coupling.

Four verbs, following the experimental design of sec:pd-aim-and-scope.

  simulate  runs one configuration, a storage rule with a controller or the
            reference planner, at one load.
  train     trains a learned controller with PPO. The reference planner is
            not learned.
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
    type=click.Choice(["fixed_only", "matched"]),
    default=None,
    help="Training regime (sec:method:training): F_fix only, or the evaluation storage rule.",
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
        ppo_config = build_study_ppo_config(spec, study, num_env_runners=env_runners, use_gpu=gpu)
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
    )
    click.echo(f"Checkpoint saved to {run.checkpoint}")
    click.echo(
        f"Best checkpoint by {spec.checkpoint.score_attribute} saved to {run.best_checkpoint}"
    )


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
    help="Storage rules the test seeds are played under, e.g. --storage-modes fixed demand. "
    "Default: the rule the run trained with. An adaptive rule takes its epoch length, "
    "relocation cap and congestion weight from the storage_rule section of the run's spec.",
)
@click.option(
    "--arrival-rate",
    type=float,
    default=None,
    help="Load the test seeds are played at, in tasks per step. Default: the training load. "
    "The checkpoint is still chosen at the training load.",
)
def evaluate(run_dir, output, storage_modes, arrival_rate):
    """Evaluate the training run in RUN_DIR (the --output of train). Every checkpoint plays the
    validation seeds under the rule the run trained with, the one with the most completed tasks
    is chosen, and it and the last checkpoint play the test seeds under every storage rule
    given. summary.md compares each adaptive rule with fixed storage."""
    from pathlib import Path

    from slap_mapd_coupling.training.evaluate_run import evaluate_run

    run = Path(run_dir)
    default_name = "evaluation" if arrival_rate is None else f"evaluation_rate{arrival_rate:g}"
    out = Path(output) if output is not None else run / default_name
    try:
        result = evaluate_run(run, out, list(storage_modes) or None, arrival_rate)
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
