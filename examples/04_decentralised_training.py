"""Train a learned controller with PPO under Ray Tune and evaluate it.

The M5 milestone's worked example. It builds a small instance, trains the
GNN-based PPO policy for a training budget counted in environment steps,
saves Ray's default checkpoint, loads the module back from it and runs it for
one episode per evaluation seed through the evaluator, with a trace per
episode. This file trains the decentralised arm. 02_centralised_fixed.py and
03_section_based.py are thin variants of it for the other two arms, and
04b_evaluate_checkpoint.py evaluates a saved checkpoint without retraining. The
full-view setting, which is not one of the three arms, trains with
`slap-mapd train --config configs/full_view.yaml`.

Values that are illustrative, not the thesis's (tab:trainparams and
tab:modelparams leave them TBD, or this script has no reason to fix them):
- every PPO hyperparameter, the training budget and the seed counts,
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

Arrival rate. --arrival-rate sets the task arrival rate lambda_task of training
and of evaluation, so a checkpoint is evaluated at the rate it was trained at by
passing the same value to 04b_evaluate_checkpoint.py. The default of 0.25 tasks per
step is about 80% of what Token Passing completes on this instance (about 0.3 tasks per
step, measured with 3 agents, horizon 630), so the arms are loaded but can keep up. At 0.15
every arm and Token Passing complete 92 to 97% of the released tasks, which is too light to tell
the arms apart. At a rate above capacity the backlog grows without bound and the evaluator's
keep-up ratio (throughput over arrival rate) cannot reach 1, so that ratio is meaningful only at
a rate below capacity. The value is illustrative, since the thesis leaves the load levels TBD.

Episode length. --horizon sets the number of steps of an episode, the horizon T, for training
and for evaluation alike, so both use one value. The thesis leaves the evaluation horizon T
open, so the default of 630 is illustrative. A sampling call of a runner is one complete
episode, and every iteration ends one episode per runner, whatever the number of runners. The
budget counts timesteps, one per timestep however many robots choose within it, so an
iteration is num_env_runners * horizon timesteps (horizon with no env runner), and with 7
runners the default of 630 is 4410 timesteps per iteration. A checkpoint can be evaluated at
another length by passing --horizon to 04b_evaluate_checkpoint.py.

Communication. --communication switches on the thesis' local communication (an agent hears every
agent inside its own field of view, and a message is the sender's own learned embedding). It is the
thesis' conditional ablation (RQ5), decentralised arm only, so it is rejected for 02 and 03. Run it
as a separate condition next to the run without it, never in place of it, and pass the same flag to
04b_evaluate_checkpoint.py, because a checkpoint trained with messages cannot be evaluated without
them.
Messages are learned, so they need training to carry information.

Run it with the smallest settings, for example
    python examples/04_decentralised_training.py --budget-env-steps=600
Ray writes its results under ~/ray_results, with a checkpoint every few iterations, the best ones by
episode return and the one at the end. The PPO hyperparameters, the budget and the model sizes are in
configs/<arm>.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import (
    ControllerArchitecture,
    ExperimentConfig,
)
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
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
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.training.arm import Arm, build_arm as _build_arm
from slap_mapd_coupling.training.config import (
    PPOHyperparameters,
    SeedSplitConfig,
    TrainingRegime,
    build_ppo_config,
    split_seeds,
)
from slap_mapd_coupling.training.policy import ModulePolicy
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_storage as build_storage_from,
    load_spec,
    with_budget,
)
from slap_mapd_coupling.training.trainer import (
    CheckpointSettings,
    TrainingRun,
    load_module,  # noqa: F401 -- 04b and 05 take it from here
    train as run_training,
)

FLEET_SIZE = 3


# The keep-up threshold f_up of eq:throughput, 0.8 in tab:evalparams. A run keeps up if its
# throughput reaches this share of the nominal arrival rate. Arrivals are random, so a healthy run
# can finish a little under the nominal count, and a threshold of 0.9 failed healthy Token Passing
# runs (2 of 6 seeds at 0.15 and 3 of 6 at 0.25) where 0.8 passed them all.
KEEP_UP_THRESHOLD = 0.8
# Illustrative. The thesis leaves the wait cost c_wait open. One value, used by the
# generated graph and by the experiment config, because it is one study parameter.
WAIT_COST = 0.1
# Illustrative. The default arrival rate, in tasks per step. About 80% of the capacity Token
# Passing reaches on this instance (about 0.3), so the arms are loaded but can keep up. Earlier
# runs used 1.0 (far above capacity) and 0.15 (too light to tell the arms apart).
DEFAULT_ARRIVAL_RATE = 0.25
# Illustrative. The thesis leaves the evaluation horizon T open, so the episode length in steps
# is a study parameter. One value, used by training and by evaluation, and also the rollout
# length of PPO, so an iteration samples one whole episode per runner (horizon steps with no
# env runner), see PPOHyperparameters.
DEFAULT_HORIZON = 630
# Illustrative. The thesis leaves the training budget TBD (tab:trainparams).
# The PPO hyperparameters, the budget, the model sizes and the checkpointing are in the YAML file
# that `slap-mapd train` reads, so there is one place to change them.
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_SPECS: dict[str, TrainingSpec] = {}


def spec_for(controller: ControllerArchitecture) -> TrainingSpec:
    """The spec of `controller` in configs/<controller>.yaml, which 02 and 03 train with too."""
    if controller not in _SPECS:
        _SPECS[controller] = load_spec(CONFIG_DIR / f"{controller}.yaml")
    return _SPECS[controller]


_SPEC = spec_for("decentralised")
ILLUSTRATIVE_BUDGET_ENV_STEPS = _SPEC.ppo.training_budget_env_steps
# Illustrative. The number of zones is a sweep parameter (tab:modelparams).
NUM_ZONES = 3
# The storage rule a policy is evaluated under. "matched" training uses it too.
EVALUATION_STORAGE_MODE = "fixed"
# Illustrative. The cap on the messages an agent hears. With FLEET_SIZE agents an agent hears at
# most the FLEET_SIZE - 1 others, so this cap is exactly enough, and it follows the fleet size so a
# larger fleet does not silently drop senders. It only matters with --communication, and it was 2
# before.
MAX_MESSAGES = FLEET_SIZE - 1
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
        wait_cost=WAIT_COST,
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
    """The SKUs, vertex capacities and initial counts of configs/<arm>.yaml (storage section)."""
    return build_storage_from(spec_for("decentralised").storage, graph)


def build_config(
    regime: TrainingRegime,
    seed: int,
    controller: ControllerArchitecture = "decentralised",
    arrival_rate: float = DEFAULT_ARRIVAL_RATE,
    horizon: int = DEFAULT_HORIZON,
    communication: bool = False,
) -> ExperimentConfig:
    """The configuration a run trains under. "fixed_only" trains under F_fix and
    "matched" under the storage rule used at evaluation (sec:method:training).
    The two differ only once EVALUATION_STORAGE_MODE is not "fixed", which needs the
    online storage estimates of the demand and congestion rules. `communication` is the
    thesis' conditional ablation (RQ5), decentralised arm only, off unless asked for."""
    storage_mode = "fixed" if regime == "fixed_only" else EVALUATION_STORAGE_MODE
    return ExperimentConfig(
        storage_mode=storage_mode,
        controller=controller,
        congestion_sensitive=False,
        communication=communication,
        num_agents=FLEET_SIZE,
        arrival_rate=arrival_rate,
        keep_up_threshold=KEEP_UP_THRESHOLD,
        seed=seed,
        horizon=horizon,
        wait_cost=WAIT_COST,
        observation_depth=5,
        discount=0.99,
        # n_deliver raised well above 1.0: a 5000-iteration run on this instance
        # showed rising training return with zero evaluation completions, consistent
        # with the per-step shaping term outweighing a small completion bonus.
        deliver_reward=10.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )


def evaluation_config(
    seed: int,
    controller: ControllerArchitecture,
    arrival_rate: float = DEFAULT_ARRIVAL_RATE,
    horizon: int = DEFAULT_HORIZON,
    communication: bool = False,
) -> ExperimentConfig:
    """The configuration an episode is evaluated under, the "matched" one. It must carry
    the `communication` the checkpoint was trained with."""
    return build_config("matched", seed, controller, arrival_rate, horizon, communication)


def build_seed_split(training_seed: int, test_seed_count: int = 3) -> SeedSplitConfig:
    # Illustrative. The seed counts are TBD (tab:trainparams, #85).
    return SeedSplitConfig(
        training_seed=training_seed,
        train_seed_count=5,
        validation_seed_count=2,
        test_seed_count=test_seed_count,
    )


def build_hyperparameters(
    budget_env_steps: int, controller: ControllerArchitecture = "decentralised"
) -> PPOHyperparameters:
    """The PPO hyperparameters of the YAML file of `controller`, with the budget given."""
    return with_budget(spec_for(controller), budget_env_steps).ppo


def build_arm(
    instance: GeneratedInstance, controller: ControllerArchitecture, observation_depth: int
) -> Arm:
    return _build_arm(
        instance,
        controller,
        observation_depth,
        num_zones=NUM_ZONES,
        max_messages=MAX_MESSAGES,
    )


def build_gnn_config(config: ExperimentConfig, arm: Arm) -> GNNEncoderConfig:
    return GNNEncoderConfig(
        num_rounds=spec_for(arm.controller).model.num_rounds,
        hidden_width=spec_for(arm.controller).model.hidden_width,
        communication=config.communication,
        pooling=arm.pooling,
    )


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
    checkpoint_settings: CheckpointSettings | None = None,
) -> TrainingRun:
    """Trains with Ray Tune through training/trainer.py. It stops at the first iteration at or
    above the training budget and returns the checkpoint of the end of the run and the best one."""
    ppo_config = build_ppo_config(
        instance.graph,
        skus,
        counts,
        capacities,
        config,
        arm.encoding,
        build_gnn_config(config, arm),
        PolicyValueHeadConfig(hidden_width=spec_for(arm.controller).model.head_hidden_width),
        hyperparameters,
        seed_split,
        zones=arm.zones,
        num_env_runners=num_env_runners,
        use_gpu=use_gpu,
    )
    return run_training(
        ppo_config,
        hyperparameters.training_budget_env_steps,
        run_name,
        arm_label=arm.controller,
        settings=checkpoint_settings or spec_for(arm.controller).checkpoint,
        progress_path=progress_path,
    )


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
    arrival_rate: float = DEFAULT_ARRIVAL_RATE,
    horizon: int = DEFAULT_HORIZON,
    communication: bool = False,
) -> None:
    """Runs `module` for one episode per evaluation seed through the evaluator, so
    the observations are the ones it saw in training. The seeds are the test seeds of
    split_seeds, disjoint from the training seeds. Each episode writes a trace, and
    the vertex rule and the swap rule are checked on it."""
    print(f"\n=== Evaluating the trained policy over {len(eval_seeds)} evaluation seeds ===")
    policy = ModulePolicy(module)
    summaries = []
    for episode_idx, eval_seed in enumerate(eval_seeds):
        config = evaluation_config(eval_seed, arm.controller, arrival_rate, horizon, communication)
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
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=DEFAULT_ARRIVAL_RATE,
        help="task arrival rate in tasks per step, for training and evaluation "
        "(default 0.25, about 80% of the capacity of Token Passing, illustrative)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="episode length in steps, used for training and for evaluation, and the rollout "
        "length of PPO, so every iteration ends one episode per runner. An iteration samples "
        "env-runners * horizon steps (horizon with none), so with 7 runners the default of "
        "630 gives 4410 steps per iteration (default 630, illustrative)",
    )
    parser.add_argument("--env-runners", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, default=3, help="number of evaluation seeds")
    parser.add_argument("--gpu", action="store_true")
    # Illustrative. The thesis' conditional ablation (RQ5), implemented but unevaluated. Run it as a
    # separate condition next to the run without it, and pass it identically to 04b.
    parser.add_argument(
        "--communication",
        action="store_true",
        help="local communication, the thesis' conditional ablation (RQ5), decentralised arm "
        "only. Run it as a separate condition, and pass the same flag to "
        "04b_evaluate_checkpoint.py",
    )
    args = parser.parse_args(argv)
    if args.communication and controller != "decentralised":
        parser.error("--communication only applies to the decentralised controller")
    if args.eval_seeds < 1:
        parser.error("--eval-seeds must be at least 1")
    if args.budget_env_steps < 1:
        parser.error("--budget-env-steps must be at least 1")
    if args.arrival_rate <= 0:
        parser.error("--arrival-rate must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    return args


def main(controller: ControllerArchitecture = "decentralised") -> None:
    args = parse_args(controller=controller)
    regime: TrainingRegime = args.regime

    instance = build_generated_instance()
    skus, capacities, counts = build_storage(instance.graph)
    config = build_config(
        regime,
        seed=0,
        controller=controller,
        arrival_rate=args.arrival_rate,
        horizon=args.horizon,
        communication=args.communication,
    )
    arm = build_arm(instance, controller, config.observation_depth)
    print(
        f"{controller} arm on {len(instance.graph.vertices)} vertices, "
        f"max_local_nodes={arm.encoding.max_local_nodes}, pooling={arm.pooling}, "
        f"arrival_rate={config.arrival_rate}, "
        f"horizon={config.horizon}, communication={config.communication}"
    )

    seed_split = build_seed_split(config.seed, args.eval_seeds)
    run = train(
        instance,
        skus,
        capacities,
        counts,
        config,
        arm,
        build_hyperparameters(args.budget_env_steps, controller),
        seed_split,
        run_name=f"{controller}_{regime}",
        num_env_runners=args.env_runners,
        use_gpu=args.gpu,
        progress_path=OUTPUT_DIR / f"{controller}_training_progress.jsonl",
    )
    print(f"\nCheckpoint saved to {run.checkpoint}")
    print(
        f"Best checkpoint by {spec_for(controller).checkpoint.score_attribute} saved to {run.best_checkpoint}"
    )
    overshoot = run.env_steps_sampled - args.budget_env_steps
    print(
        f"Sampled {run.env_steps_sampled} environment steps for a budget of "
        f"{args.budget_env_steps}, over the budget by {overshoot}."
    )

    eval_seeds = split_seeds(seed_split).test
    module = load_module(run.checkpoint)
    evaluate(
        module,
        instance,
        skus,
        capacities,
        counts,
        arm,
        eval_seeds,
        OUTPUT_DIR / "traces",
        args.arrival_rate,
        args.horizon,
        args.communication,
    )


if __name__ == "__main__":
    main()
