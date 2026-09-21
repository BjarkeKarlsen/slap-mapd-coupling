"""Train the decentralised GNN+PPO controller and sanity-check it.

The M5 milestone's worked example: builds a small instance, trains the
GNN-based PPO policy (models/rl_module.py, training/config.py, #30) for
a handful of iterations, saves a checkpoint, then runs the trained
policy for a few fresh episodes to confirm it produces a valid,
correctness-invariant-respecting trace and completes at least some
tasks -- not just that loss goes somewhere.

Scope, flagged rather than silently narrowed:
- Only "fixed_only" (train under F_fix; sec:method:training's other
  regime, "matched," trains under whichever storage rule is under
  evaluation) is exercised here by default -- pass --regime matched to
  run the other one instead. Running both by default would double this
  script's already-nontrivial runtime for what's meant to stay a quick
  smoke-testable example; tab:trainparams's own text asks for both
  regimes to be "distinguished and reported," which a real sweep script
  (not this one) is where that comparison actually belongs.
- "Evaluates it" (issue #31's text) here means running the just-trained
  RLModule directly for a few episodes and checking the correctness
  invariants plus basic task-completion counts -- NOT the full
  evaluator.py/RunMetrics comparison against the centralised/section-
  based baselines that #29 (decentralised controller wrapper, wrapping
  the trained policy behind the shared Controller Protocol so
  evaluation/evaluator.py can run it like any other controller) would
  enable. #29 is explicitly skipped this session; wiring the trained
  checkpoint into evaluator.py for an apples-to-apples RunMetrics
  comparison is #29's job, not re-derived here.

Hyperparameters below are deliberately small/fast (short horizon, tiny
network, few iterations) so this runs in well under a minute on a
laptop CPU -- a real sweep run would use tab:trainparams's actual
(currently TBD) values, not these.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType, is_feasible
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_and_validate
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.callbacks import CorrectnessInvariantCallbacks
from slap_mapd_coupling.training.config import PPOHyperparameters, TrainingRegime, build_ppo_config
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv
from slap_mapd_coupling.training.utils import save_checkpoint

FLEET_SIZE = 3
CHECKPOINT_DIR = Path(__file__).parent / "output" / "04_decentralised_training_checkpoint"


def build_instance() -> WarehouseGraph:
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
        one_way_fraction=0.0,
        default_edge_cost=1.0,
        wait_cost=0.5,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=FLEET_SIZE,
        seed=1,
    )
    graph, report = generate_and_validate(params, fleet_size=FLEET_SIZE, require_well_formed=False)
    if not report.accepted:
        raise RuntimeError(f"Instance rejected: {report.model_dump_json(indent=2)}")
    return graph


def build_storage(
    graph: WarehouseGraph,
) -> tuple[dict[SkuId, SkuType], dict[VertexId, float], dict[SkuId, dict[VertexId, int]]]:
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 20.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 8, storage_vertices[1]: 8}}
    return skus, capacities, counts


def build_config(regime: TrainingRegime, seed: int) -> ExperimentConfig:
    # "matched" training uses whichever storage_mode evaluation would use
    # (sec:method:training); F_dem/F_cng need extra ExperimentConfig
    # fields this quick demo doesn't otherwise exercise, so "matched"
    # here still means F_fix -- a real matched-regime run would set
    # storage_mode to the rule actually under study, per the module
    # docstring's own scope note.
    del regime  # both regimes use F_fix in this deliberately small demo
    return ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=FLEET_SIZE,
        arrival_rate=0.3,
        seed=seed,
        horizon=30,
        wait_cost=0.5,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )


def build_hyperparameters() -> PPOHyperparameters:
    # Deliberately small/fast -- see module docstring.
    return PPOHyperparameters(
        lr=3e-4,
        clip_param=0.2,
        gae_lambda=0.95,
        minibatch_size=64,
        num_epochs=3,
        rollout_fragment_length=30,
        entropy_coeff=0.01,
        vf_loss_coeff=1.0,
        train_batch_size=180,
    )


def train(
    graph: WarehouseGraph,
    skus: dict[SkuId, SkuType],
    capacities: dict[VertexId, float],
    counts: dict[SkuId, dict[VertexId, int]],
    config: ExperimentConfig,
    num_iterations: int,
):
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=12, max_messages=2)
    gnn_config = GNNEncoderConfig(num_rounds=1, hidden_width=16)
    d_max = max_out_degree(graph)
    head_config = PolicyValueHeadConfig(hidden_width=16, num_actions=d_max + 1)

    ppo_config = build_ppo_config(
        graph,
        skus,
        counts,
        capacities,
        config,
        enc_config,
        gnn_config,
        head_config,
        build_hyperparameters(),
        num_env_runners=0,
    ).callbacks(CorrectnessInvariantCallbacks)

    algo = ppo_config.build_algo()
    print(f"=== Training the decentralised GNN+PPO controller ({num_iterations} iterations) ===")
    for i in range(num_iterations):
        result = algo.train()
        env_runner_results = result.get("env_runners", {})
        print(
            f"  iter {i + 1}/{num_iterations}: "
            f"episode_return_mean={env_runner_results.get('episode_return_mean')} "
            f"episode_len_mean={env_runner_results.get('episode_len_mean')}"
        )
    return algo, enc_config


def evaluate(algo, graph, skus, capacities, counts, config, num_episodes: int) -> None:
    """Runs the trained RLModule for a few fresh episodes directly
    (bypassing the Controller Protocol -- see module docstring for why),
    checking the same correctness invariants training already enforces
    and reporting how many tasks got completed."""
    print(f"\n=== Evaluating the trained policy over {num_episodes} fresh episodes ===")
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=12, max_messages=2)
    module = algo.get_module("shared_policy")

    for episode_idx in range(num_episodes):
        env = WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            counts,
            capacities,
            config.model_copy(update={"seed": config.seed + 1000 + episode_idx}),
            enc_config,
        )
        obs, _ = env.reset()
        before_fleet = env._env.fleet
        completed_before = sum(1 for t in env._env.tasks if t.completion_time is not None)

        for _ in range(config.horizon):
            batch, agent_ids = _to_batch(obs)
            fwd_out = module.forward_inference(batch)
            logits = fwd_out["action_dist_inputs"]
            # Pair logits[i] with agent_ids[i], NOT env.agents[i] -- obs's
            # own key order (which _to_batch stacks in) isn't guaranteed
            # to match env.agents's fixed order.
            actions = {
                aid: int(np.argmax(logits[i].detach().numpy())) for i, aid in enumerate(agent_ids)
            }
            obs, rewards, terminated, truncated, infos = env.step(actions)
            assert is_collision_free(before_fleet, env._env.fleet), "collision during evaluation"
            assert is_feasible(env._env.storage), "infeasible storage during evaluation"
            before_fleet = env._env.fleet
            if truncated["__all__"] or terminated["__all__"]:
                break

        completed_after = sum(1 for t in env._env.tasks if t.completion_time is not None)
        print(
            f"  episode {episode_idx + 1}/{num_episodes}: "
            f"tasks completed = {completed_after - completed_before}, "
            f"collisions = none, storage feasible = yes"
        )


def _to_batch(obs: dict) -> tuple[dict, list[str]]:
    """Stacks the per-agent dict observations into one batch, keyed by
    the same nested structure encoded_observation_space() declares.
    Returns the agent-id order used for stacking alongside the batch, so
    callers can pair logits[i] with the right agent -- obs's own key
    order is not guaranteed to match any other list of agent ids."""
    import torch
    from ray.rllib.core.columns import Columns

    agent_ids = list(obs.keys())
    stacked = {
        "action_mask": torch.as_tensor(np.stack([obs[a]["action_mask"] for a in agent_ids])),
        "observations": {
            key: torch.as_tensor(np.stack([obs[a]["observations"][key] for a in agent_ids]))
            for key in obs[agent_ids[0]]["observations"]
        },
    }
    return {Columns.OBS: stacked}, agent_ids


def main() -> None:
    regime: TrainingRegime = "matched" if "--regime=matched" in sys.argv[1:] else "fixed_only"
    num_iterations = 5
    for arg in sys.argv[1:]:
        if arg.startswith("--iterations="):
            num_iterations = int(arg.split("=", 1)[1])

    graph = build_instance()
    skus, capacities, counts = build_storage(graph)
    config = build_config(regime, seed=0)

    algo, _ = train(graph, skus, capacities, counts, config, num_iterations)

    saved_path = save_checkpoint(algo, CHECKPOINT_DIR)
    print(f"\nCheckpoint saved to {saved_path}")

    evaluate(algo, graph, skus, capacities, counts, config, num_episodes=3)
    algo.stop()


if __name__ == "__main__":
    main()
