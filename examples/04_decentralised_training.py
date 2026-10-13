"""Train a learned controller with PPO under Ray Tune and evaluate it.

The M5 milestone's worked example. It builds a small instance, trains the
GNN-based PPO policy for a training budget counted in environment steps,
saves Ray's default checkpoint, loads the module back from it and runs it for
one episode per evaluation seed through the evaluator, with a trace per
episode. This file trains the decentralised arm. 02_centralised_fixed.py and
03_section_based.py are thin variants of it for the other two arms, and
04b_evaluate_checkpoint.py evaluates a saved checkpoint without retraining.

Values that are illustrative, not the thesis's (tab:trainparams and
tab:modelparams leave them TBD, or this script has no reason to fix them):
- every PPO hyperparameter, the training budget and the seed counts,
- credit_signal="per_agent" for every arm, because the thesis leaves the
  credit signal of the centralised and section arms open,
- the number of zones of the section arm (NUM_ZONES, zones follow the aisles).

Window caps. A window larger than max_local_nodes raises, so the cap is set
from the graph to the largest window the arm uses and no window is truncated.
That is every movable vertex for the centralised arm, the largest field of view
for the decentralised arm, and the largest zone with its halo for the section
arm. Pooling is "mean" for the centralised and section arms and "none" for the
decentralised arm.

Regimes (sec:method:training). "fixed_only" trains under F_fix. "matched" trains
under the storage rule used at evaluation, EVALUATION_STORAGE_MODE. That rule is
F_fix here, so the two regimes train identically until the online storage
estimates of the demand and congestion rules exist. Evaluation samples a legal
slot with the policy function's rule. Whether it should take the most likely
slot instead is still open.

Run it with the smallest settings, for example
    python examples/04_decentralised_training.py --budget-env-steps=600
Ray writes its results, with the checkpoint, under ~/ray_results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from ray import tune

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import (
    ControllerArchitecture,
    ExperimentConfig,
)
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.observation import local_subgraph
from slap_mapd_coupling.environment.zones import halo, zones_by_aisle
from slap_mapd_coupling.evaluation.evaluator import run_episode
from slap_mapd_coupling.evaluation.trace import (
    format_summary,
    load_trace,
    pooled_progress,
    summarise_trace,
)
from slap_mapd_coupling.instances.generator import (
    GeneratedInstance,
    GeneratorParams,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig, Pooling
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.callbacks import (
    TrainingProgressCallbacks,
    format_iteration,
    iteration_record,
)
from slap_mapd_coupling.training.config import (
    DEFAULT_POLICY_ID,
    PPOHyperparameters,
    SeedSplitConfig,
    TrainingRegime,
    build_ppo_config,
    split_seeds,
)
from slap_mapd_coupling.training.env import CreditSignal, Zones
from slap_mapd_coupling.training.policy import ModulePolicy

FLEET_SIZE = 3
# Illustrative. The thesis leaves the training budget TBD (tab:trainparams).
ILLUSTRATIVE_BUDGET_ENV_STEPS = 4200
# Illustrative. The thesis leaves the credit signal of the wider arms open (#83).
CREDIT_SIGNAL: CreditSignal = "per_agent"
# Illustrative. The number of zones is a sweep parameter (tab:modelparams).
NUM_ZONES = 3
# The storage rule a policy is evaluated under. "matched" training uses it too.
EVALUATION_STORAGE_MODE = "fixed"
POOLING: Mapping[str, Pooling] = {"decentralised": "none", "centralised": "mean", "section": "mean"}
OUTPUT_DIR = Path(__file__).parent / "output"


def build_generated_instance() -> GeneratedInstance:
    """The generated instance, which keeps the aisle index the section arm needs."""
    params = GeneratorParams(
        max_num_aisles=7,
        max_aisle_length=7,
        max_num_cross_aisles=6,
        one_way_fraction=0.0,
        default_edge_cost=1.0,
        random_edge_costs=True,
        edge_cost_range=(0.5, 2.0),
        wait_cost=5.0,
        num_storage_vertices=4,
        num_delivery_vertices=4,
        num_endpoints=FLEET_SIZE,
        # Require at least two selected role nodes to have degree >= 2.
        # Use zero if leaf placements are also acceptable.
        min_internal_role_vertices=2,
        seed=42,
    )
    instance = generate_instance(params)
    connectivity = check_connectivity(instance.graph)
    well_formedness = check_well_formedness(instance.graph, FLEET_SIZE)
    if not (connectivity.ok and well_formedness.ok):
        raise RuntimeError(
            f"Instance rejected: connectivity_ok={connectivity.ok}, "
            f"well_formedness_ok={well_formedness.ok}."
        )
    return instance


def build_instance() -> WarehouseGraph:
    return build_generated_instance().graph


def build_storage(
    graph: WarehouseGraph,
) -> tuple[dict[SkuId, SkuType], dict[VertexId, float], dict[SkuId, dict[VertexId, int]]]:
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 20.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 8, storage_vertices[1]: 8}}
    return skus, capacities, counts


def build_config(
    regime: TrainingRegime, seed: int, controller: ControllerArchitecture = "decentralised"
) -> ExperimentConfig:
    """The configuration a run trains under. "fixed_only" trains under F_fix and
    "matched" under the storage rule used at evaluation (sec:method:training).
    The two differ only once EVALUATION_STORAGE_MODE is not "fixed", which needs the
    online storage estimates of the demand and congestion rules."""
    storage_mode = "fixed" if regime == "fixed_only" else EVALUATION_STORAGE_MODE
    return ExperimentConfig(
        storage_mode=storage_mode,
        controller=controller,
        congestion_sensitive=False,
        communication=False,
        num_agents=FLEET_SIZE,
        arrival_rate=1.0,
        seed=seed,
        horizon=2000,
        wait_cost=2.0,
        observation_depth=5,
        discount=0.99,
        # n_deliver raised well above 1.0: a 5000-iteration run on this instance
        # showed rising training return with zero evaluation completions, consistent
        # with the per-step shaping term outweighing a small completion bonus.
        deliver_reward=10.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )


def evaluation_config(seed: int, controller: ControllerArchitecture) -> ExperimentConfig:
    """The configuration an episode is evaluated under, the "matched" one."""
    return build_config("matched", seed, controller)


def build_seed_split(base_seed: int, eval_seed_count: int = 3) -> SeedSplitConfig:
    # Illustrative. The train and eval seed counts are TBD (tab:trainparams, #85).
    return SeedSplitConfig(base_seed=base_seed, train_seed_count=5, eval_seed_count=eval_seed_count)


def build_hyperparameters(num_env_runners: int, budget_env_steps: int) -> PPOHyperparameters:
    # Illustrative values, sized so a longer run learns some navigation on this instance.
    rollout_fragment_length = 30
    # RLlib needs train_batch_size to be about a multiple of num_env_runners *
    # rollout_fragment_length when num_env_runners > 0. This picks the multiple closest
    # to 600 samples, which is also fine for num_env_runners = 0.
    per_iteration_samples = max(num_env_runners, 1) * rollout_fragment_length
    multiple = max(1, round(600 / per_iteration_samples))
    return PPOHyperparameters(
        lr=3e-4,
        clip_param=0.2,
        gae_lambda=0.95,
        minibatch_size=128,
        num_epochs=5,
        rollout_fragment_length=rollout_fragment_length,
        entropy_coeff=0.02,
        vf_loss_coeff=1.0,
        train_batch_size=multiple * per_iteration_samples,
        training_budget_env_steps=budget_env_steps,
        credit_signal=CREDIT_SIGNAL,
    )


# NamedTuple, not a dataclass, so a test can load this file by path without registering it.
class Arm(NamedTuple):
    """What an arm needs besides the graph: its pooling, its zones (section only)
    and the padding caps."""

    controller: ControllerArchitecture
    pooling: Pooling
    zones: Zones | None
    encoding: LocalSubgraphEncodingConfig


def largest_window(
    graph: WarehouseGraph,
    controller: ControllerArchitecture,
    observation_depth: int,
    zones: Zones | None,
) -> int:
    """The vertices of the largest window the arm uses, so the cap truncates none."""
    movable = [v for v, vertex in graph.vertices.items() if vertex.role.movable]
    if controller == "centralised":
        return len(movable)
    if controller == "section":
        assert zones is not None
        return max(len(zone | halo(graph, zone, observation_depth)) for zone in zones)
    return max(len(local_subgraph(graph, v, observation_depth)) for v in movable)


def build_arm(
    instance: GeneratedInstance, controller: ControllerArchitecture, observation_depth: int
) -> Arm:
    zones = None
    if controller == "section":
        zones = zones_by_aisle(instance.graph, instance.aisle_index, NUM_ZONES)
    cap = largest_window(instance.graph, controller, observation_depth, zones)
    return Arm(
        controller=controller,
        pooling=POOLING[controller],
        zones=zones,
        encoding=LocalSubgraphEncodingConfig(max_local_nodes=cap, max_messages=2),
    )


def build_gnn_config(config: ExperimentConfig, arm: Arm) -> GNNEncoderConfig:
    return GNNEncoderConfig(
        num_rounds=2,
        hidden_width=32,
        communication=config.communication,
        pooling=arm.pooling,
    )


class ProgressLog(tune.Callback):
    """Prints one line per iteration and appends it to a JSON Lines file."""

    def __init__(self, expected_iterations: int, progress_path: Path | None) -> None:
        self._expected_iterations = expected_iterations
        self._progress_path = progress_path
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text("", encoding="utf-8")

    def on_trial_result(self, iteration, trials, trial, result: dict, **info: Any) -> None:
        i = result["training_iteration"]
        print(format_iteration(i, self._expected_iterations, result))
        if self._progress_path is not None:
            with self._progress_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(iteration_record(i, result)) + "\n")


class TrainingRun(NamedTuple):
    checkpoint: Path
    env_steps_sampled: int


def train(
    instance: GeneratedInstance,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
    arm: Arm,
    hyperparameters: PPOHyperparameters,
    seed_split: SeedSplitConfig,
    run_name: str,
    num_env_runners: int = 0,
    use_gpu: bool = False,
    progress_path: Path | None = None,
) -> TrainingRun:
    """Trains with Ray Tune, stops at the first iteration at or above the training
    budget, and returns Ray's default checkpoint of the end of the run."""
    ppo_config = build_ppo_config(
        instance.graph,
        skus,
        counts,
        capacities,
        config,
        arm.encoding,
        build_gnn_config(config, arm),
        PolicyValueHeadConfig(hidden_width=32),
        hyperparameters,
        seed_split,
        zones=arm.zones,
        num_env_runners=num_env_runners,
        use_gpu=use_gpu,
    ).callbacks(TrainingProgressCallbacks)
    budget = hyperparameters.training_budget_env_steps
    expected_iterations = math.ceil(budget / hyperparameters.train_batch_size)
    print(
        f"=== Training the {arm.controller} GNN+PPO controller "
        f"(budget {budget} environment steps) ==="
    )
    results = tune.Tuner(
        "PPO",
        param_space=ppo_config,
        run_config=tune.RunConfig(
            name=run_name,
            stop={"num_env_steps_sampled_lifetime": budget},
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=True),
            callbacks=[ProgressLog(expected_iterations, progress_path)],
        ),
    ).fit()
    result = results[0]
    if result.error is not None:
        raise result.error
    assert result.checkpoint is not None and result.metrics is not None
    return TrainingRun(
        checkpoint=Path(result.checkpoint.path),
        env_steps_sampled=int(result.metrics["num_env_steps_sampled_lifetime"]),
    )


def load_module(checkpoint: Path, policy_id: str = DEFAULT_POLICY_ID):
    """The trained RLModule of `policy_id` from an Algorithm checkpoint, with no Ray
    cluster and no registered environment. The path is built from Ray's own constants
    for the layout of a checkpoint, which a test guards."""
    from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_LEARNER_GROUP, COMPONENT_RL_MODULE
    from ray.rllib.core.rl_module.rl_module import RLModule

    path = checkpoint / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE
    return RLModule.from_checkpoint(str(path.resolve() / policy_id))


def trace_is_collision_free(steps: list[dict[str, Any]], initial: dict[str, int]) -> bool:
    """The vertex rule and the swap rule on a written trace."""
    before = dict(initial)
    for step in steps:
        after = {a: state["at"] for a, state in step["agents"].items()}
        if len(set(after.values())) != len(after):
            return False
        for a, at in after.items():
            for b, other_at in after.items():
                if a < b and before[a] == other_at and before[b] == at and at != before[a]:
                    return False
        before = after
    return True


def evaluate(
    module,
    instance: GeneratedInstance,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    arm: Arm,
    eval_seeds: tuple[int, ...],
    trace_dir: Path,
) -> None:
    """Runs `module` for one episode per evaluation seed through the evaluator, so
    the observations are the ones it saw in training. The seeds are the evaluation half
    of split_seeds, disjoint from the training seeds. Each episode writes a trace, and
    the vertex rule and the swap rule are checked on it."""
    print(f"\n=== Evaluating the trained policy over {len(eval_seeds)} evaluation seeds ===")
    policy = ModulePolicy(module)
    summaries = []
    for episode_idx, eval_seed in enumerate(eval_seeds):
        config = evaluation_config(eval_seed, arm.controller)
        trace_path = trace_dir / f"{arm.controller}_episode_{episode_idx + 1}.jsonl"
        metrics = run_episode(
            instance.graph,
            skus,
            counts,
            capacities,
            config,
            trace_path,
            policy=policy,
            encoding_config=arm.encoding,
            zones=arm.zones,
            pooling=arm.pooling,
        )
        header, steps = load_trace(trace_path)
        assert trace_is_collision_free(steps, header["initial"]), "collision during evaluation"
        print(
            f"  episode {episode_idx + 1}/{len(eval_seeds)} (seed {eval_seed}): "
            f"tasks completed = {metrics.num_completed_tasks}, collisions = none"
        )
        summary = summarise_trace(header, steps)
        summaries.append(summary)
        print(format_summary(summary))
        print(f"  trace written to {trace_path}")
    progress = pooled_progress(summaries)
    if progress is not None:
        print(f"\nPooled over the episodes, {progress:.0%} of goal moves bring a robot closer.")


def parse_args(
    argv: list[str] | None = None, controller: ControllerArchitecture = "decentralised"
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train and evaluate the {controller} controller.")
    parser.add_argument("--regime", choices=["fixed_only", "matched"], default="fixed_only")
    parser.add_argument(
        "--budget-env-steps",
        type=int,
        default=ILLUSTRATIVE_BUDGET_ENV_STEPS,
        help="training budget in environment steps (illustrative default)",
    )
    parser.add_argument("--env-runners", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=3, help="number of evaluation seeds")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_seeds < 1:
        parser.error("--eval-seeds must be at least 1")
    if args.budget_env_steps < 1:
        parser.error("--budget-env-steps must be at least 1")
    return args


def main(controller: ControllerArchitecture = "decentralised") -> None:
    args = parse_args(controller=controller)
    regime: TrainingRegime = args.regime

    instance = build_generated_instance()
    skus, capacities, counts = build_storage(instance.graph)
    config = build_config(regime, seed=0, controller=controller)
    arm = build_arm(instance, controller, config.observation_depth)
    print(
        f"{controller} arm on {len(instance.graph.vertices)} vertices, "
        f"max_local_nodes={arm.encoding.max_local_nodes}, pooling={arm.pooling}, "
        f"credit_signal={CREDIT_SIGNAL}"
    )

    seed_split = build_seed_split(config.seed, args.eval_seeds)
    run = train(
        instance,
        skus,
        capacities,
        counts,
        config,
        arm,
        build_hyperparameters(args.env_runners, args.budget_env_steps),
        seed_split,
        run_name=f"{controller}_{regime}",
        num_env_runners=args.env_runners,
        use_gpu=args.gpu,
        progress_path=OUTPUT_DIR / f"{controller}_training_progress.jsonl",
    )
    print(f"\nCheckpoint saved to {run.checkpoint}")
    overshoot = run.env_steps_sampled - args.budget_env_steps
    print(
        f"Sampled {run.env_steps_sampled} environment steps for a budget of "
        f"{args.budget_env_steps}, over the budget by {overshoot}."
    )

    _, eval_seeds = split_seeds(seed_split)
    module = load_module(run.checkpoint)
    evaluate(module, instance, skus, capacities, counts, arm, eval_seeds, OUTPUT_DIR / "traces")


if __name__ == "__main__":
    main()
