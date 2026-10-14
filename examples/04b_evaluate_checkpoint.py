"""Evaluate a checkpoint trained by 04_decentralised_training.py without retraining.

Pass the checkpoint folder Ray wrote at the end of the run, for example
    python examples/04b_evaluate_checkpoint.py --checkpoint=<path> --eval-seeds=3
Pass the same --arrival-rate the checkpoint was trained at (default 1.0, as in 04).
--horizon sets the episode length in steps (default 2000, as in 04), so a checkpoint can be
evaluated at another length than it was trained at. The thesis leaves the evaluation horizon T
open, so the value is illustrative.
The module is loaded from the checkpoint's own folder, with no Ray cluster and no
registered environment, and evaluated through the evaluator as in 04. Use --arm
for a checkpoint of 02_centralised_fixed.py (centralised) or 03_section_based.py
(section). The default is the decentralised arm of 04.

The instance, storage, configuration, caps and pooling are 04's, imported from it,
because the module was built against that graph's action and observation spaces.
A checkpoint trained against another instance or another arm loads without an error
and gives nonsense actions. A checkpoint from before the per-edge policy head or the
pooled readout does not load either, because its weights have another shape.
Retrain it.

Numbered 04b like 01b and 01c, a variant of 04 and not a new build step
(docs/implementation_phases.md).

--render shows a live pygame preview of one episode (the [viz] extra and a real display)
instead of the headless summary. It steps the adapter and the policy by hand, since the
evaluator runs a whole episode at a time.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
example_04 = importlib.import_module("04_decentralised_training")

FLEET_SIZE = example_04.FLEET_SIZE
build_instance = example_04.build_instance
build_storage = example_04.build_storage
load_module = example_04.load_module


def build_config(
    seed: int,
    controller: str = "decentralised",
    arrival_rate: float = example_04.DEFAULT_ARRIVAL_RATE,
    horizon: int = example_04.DEFAULT_HORIZON,
):
    """The configuration of an evaluation episode, 04's."""
    return example_04.evaluation_config(seed, controller, arrival_rate, horizon)


def render_episode(
    module,
    instance,
    skus,
    capacities,
    counts,
    arm,
    arrival_rate: float = example_04.DEFAULT_ARRIVAL_RATE,
    delay: float = 0.15,
    horizon: int = example_04.DEFAULT_HORIZON,
) -> None:
    """Live pygame preview of one episode driven by the module's sampled slots. Close the
    window (or Ctrl+C) to stop early."""
    from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv
    from slap_mapd_coupling.training.policy import ModulePolicy
    from slap_mapd_coupling.viz.pygame_renderer import WarehouseRenderer

    config = build_config(
        seed=0, controller=arm.controller, arrival_rate=arrival_rate, horizon=horizon
    )
    adapter = WarehouseMAPDMultiAgentEnv(
        instance.graph,
        skus,
        counts,
        capacities,
        config,
        arm.encoding,
        (config.seed,),
        credit_signal=example_04.CREDIT_SIGNAL,
        pooling=arm.pooling,
        zones=arm.zones,
    )
    observations, _ = adapter.reset(seed=config.seed)
    policy = ModulePolicy(module)
    policy.reset(config.seed)

    renderer = WarehouseRenderer(instance.graph, instance.positions)
    try:
        running = renderer.render(adapter.core.fleet, storage=adapter.core.storage)
        time.sleep(delay)
        for _ in range(config.horizon):
            if not running:
                break
            observations, _, terminated, truncated, _ = adapter.step(
                policy(observations, adapter.core.t)
            )
            running = renderer.render(adapter.core.fleet, storage=adapter.core.storage)
            time.sleep(delay)
            if truncated["__all__"] or terminated["__all__"]:
                break
    finally:
        renderer.close()
    completed = sum(1 for t in adapter.core.tasks if t.finish_time is not None)
    print(f"\nRendered episode finished: tasks completed = {completed}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ray checkpoint folder")
    parser.add_argument(
        "--arm", choices=["decentralised", "centralised", "section"], default="decentralised"
    )
    parser.add_argument("--eval-seeds", type=int, default=3, help="number of evaluation seeds")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=example_04.DEFAULT_ARRIVAL_RATE,
        help="task arrival rate in tasks per step, the rate the checkpoint was trained at",
    )
    # Illustrative. The thesis leaves the evaluation horizon T open. Training ends an episode
    # in every iteration only if the horizon is at most the steps each runner samples per
    # iteration, RLlib's default batch of 4000 divided over the runners (4020 with none,
    # about 600 with 7).
    parser.add_argument(
        "--horizon",
        type=int,
        default=example_04.DEFAULT_HORIZON,
        help="episode length in steps, used for evaluation (default 2000, as in 04). In "
        "training an episode ends in every iteration only if this is at most the steps each "
        "runner samples per iteration, RLlib's default batch of 4000 divided over the "
        "runners (4020 with none, about 600 with 7)",
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_seeds < 1:
        parser.error("--eval-seeds must be at least 1")
    if args.arrival_rate <= 0:
        parser.error("--arrival-rate must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise RuntimeError(
            f"No checkpoint at {args.checkpoint}. Run 04_decentralised_training.py first, "
            "or pass --checkpoint=<path> to an existing one."
        )

    instance = example_04.build_generated_instance()
    skus, capacities, counts = build_storage(instance.graph)
    config = build_config(
        seed=0, controller=args.arm, arrival_rate=args.arrival_rate, horizon=args.horizon
    )
    arm = example_04.build_arm(instance, args.arm, config.observation_depth)

    print(f"Loading checkpoint from {args.checkpoint}")
    module = load_module(args.checkpoint)

    if args.render:
        render_episode(
            module,
            instance,
            skus,
            capacities,
            counts,
            arm,
            args.arrival_rate,
            horizon=args.horizon,
        )
    else:
        seed_split = example_04.build_seed_split(config.seed, args.eval_seeds)
        _, eval_seeds = example_04.split_seeds(seed_split)
        example_04.evaluate(
            module,
            instance,
            skus,
            capacities,
            counts,
            arm,
            eval_seeds,
            example_04.OUTPUT_DIR / "traces",
            args.arrival_rate,
            args.horizon,
        )


if __name__ == "__main__":
    main()
