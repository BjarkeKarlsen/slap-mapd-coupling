"""Command-line interface for slap_mapd_coupling.

Three verbs, matching the thesis's own experimental design (independent
variables: storage mode, controller architecture, congestion sensitivity,
communication, system load - sec:pd-aim-and-scope) rather than a generic
train/evaluate/checkpoint workflow that only fits the decentralised
controller:

  simulate  - run one (storage rule, controller, load) configuration,
              for any of the three controllers.
  train     - PPO training for the decentralised controller only; the
              centralised and section-based controllers are deterministic
              planners and have nothing to train.
  sweep     - run the full experimental grid from a config file and write
              one results table (this is what produces Section VII's data).
"""

import click

STORAGE_RULES = ["fixed", "demand", "congestion"]
CONTROLLERS = ["centralised", "section", "decentralised"]


@click.group()
@click.version_option()
def cli():
    """SLAP-MAPD coupling: warehouse storage/routing simulation and RL training."""


@cli.command()
@click.option("--storage-rule", type=click.Choice(STORAGE_RULES), default="fixed", help="Storage update rule F.")
@click.option("--controller", type=click.Choice(CONTROLLERS), default="centralised", help="Routing/assignment architecture.")
@click.option("--congestion-sensitive", is_flag=True, help="Enable congestion sensitivity in storage and routing.")
@click.option("--communication", is_flag=True, help="Enable local communication (decentralised controller only).")
@click.option("--num-agents", default=10, type=int, help="Fleet size.")
@click.option("--arrival-rate", default=1.0, type=float, help="Expected tasks released per timestep.")
@click.option("--num-episodes", default=10, type=int, help="Number of episodes to run.")
@click.option("--seed", default=0, type=int, help="Episode/environment seed.")
@click.option("--checkpoint", type=click.Path(exists=True), default=None, help="Trained policy (required iff --controller=decentralised).")
@click.option("--output", type=click.Path(), default=None, help="Where to write the metrics row(s) (CSV). Prints to stdout if omitted.")
def simulate(storage_rule, controller, congestion_sensitive, communication, num_agents, arrival_rate, num_episodes, seed, checkpoint, output):
    """Run episodes for one configuration and report the evaluation metrics."""
    if controller == "decentralised" and checkpoint is None:
        raise click.UsageError("--checkpoint is required when --controller=decentralised.")
    if controller != "decentralised" and checkpoint is not None:
        raise click.UsageError("--checkpoint only applies to --controller=decentralised.")
    if controller != "decentralised" and communication:
        raise click.UsageError("--communication only applies to --controller=decentralised.")
    raise NotImplementedError("Simulation is not implemented yet.")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Training config (YAML); overrides the options below.")
@click.option("--storage-rule", type=click.Choice(STORAGE_RULES), default="fixed", help="Storage rule used during training (the F_fix-only vs matched training regime).")
@click.option("--num-iterations", default=500, type=int, help="Number of PPO training iterations.")
@click.option("--num-gpus", default=0, type=int, help="Number of GPUs to use.")
@click.option("--seed", default=0, type=int, help="Training seed (disjoint from evaluation seeds).")
@click.option("--output", type=click.Path(), default="./checkpoints", help="Checkpoint output directory.")
def train(config, storage_rule, num_iterations, num_gpus, seed, output):
    """Train the decentralised controller (PPO). The other two controllers need no training."""
    raise NotImplementedError("Training is not implemented yet.")


@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Experiment-grid config (YAML): storage rules x controllers x congestion x communication x load x seeds.")
@click.option("--output", type=click.Path(), default="./results.csv", help="Aggregated results table.")
@click.option("--dry-run", is_flag=True, help="Print the resulting experiment grid without running it.")
def sweep(config, output, dry_run):
    """Run the full experimental grid, training decentralised runs as needed, and write one results table."""
    raise NotImplementedError("Sweep is not implemented yet.")


if __name__ == "__main__":
    cli()
