import click
import yaml
from hierarchical_robot_rl.training.trainer import Trainer
from hierarchical_robot_rl.environment.multi_agent_env import Environment
#from hierarchical_robot_rl.evaluation.evaluator import Evaluator

@click.group()
@click.version_option()
def cli():
    """Hierarchical Robot RL CLI"""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), 
              help='Config file (YAML)')
@click.option('--num-iterations', default=500, type=int,
              help='Number of training iterations')
@click.option('--num-gpus', default=1, type=int,
              help='Number of GPUs to use')
@click.option('--experiment-name', default='default',
              help='Experiment name for tracking')
@click.option('--visualization', is_flag=True,
              help='Enable pygame visualization')
def train(config, num_iterations, num_gpus, experiment_name, visualization):
    """Train the hierarchical RL agent"""
    # Load config if provided, otherwise use defaults
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    
    # Create trainer
    trainer = Trainer(
        config=cfg,
        #num_iterations=num_iterations,
        #num_gpus=num_gpus,
        #experiment_name=experiment_name,
        #visualization=visualization
        env=Environment,  # Assuming you have an Environment class defined
    )
    trainer.train()

@cli.command()
@click.option('--checkpoint', type=click.Path(exists=True), required=True,
              help='Path to checkpoint')
@click.option('--num-episodes', default=10, type=int,
              help='Number of episodes to evaluate')
@click.option('--render', is_flag=True,
              help='Render episodes')
def evaluate(checkpoint, num_episodes, render):
    """Evaluate a trained policy"""
    # evaluator = Evaluator(checkpoint_path=checkpoint)
    # metrics = evaluator.evaluate(num_episodes=num_episodes, render=render)
    # click.echo(f"Reward mean: {metrics['episode_return_mean']:.2f}")

@cli.command()
@click.option('--config', type=click.Path(exists=True),
              help='Config file')
@click.option('--output', type=click.Path(), default='./generated_config.yaml',
              help='Output config file')
def generate_config(config, output):
    """Generate a config file from defaults"""
    from hierarchical_robot_rl.training.config import get_default_config
    cfg = get_default_config()
    with open(output, 'w') as f:
        yaml.dump(cfg, f)
    click.echo(f"Config saved to {output}")

@cli.command()
@click.option('--checkpoint', type=click.Path(exists=True), required=True,
              help='Checkpoint to inspect')
def inspect(checkpoint):
    """Inspect checkpoint details"""
    from hierarchical_robot_rl.training.utils import load_checkpoint
    ckpt = load_checkpoint(checkpoint)
    click.echo(f"Checkpoint info: {ckpt.keys()}")

if __name__ == '__main__':
    cli()