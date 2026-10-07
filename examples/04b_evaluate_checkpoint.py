"""Evaluate an already-trained checkpoint from 04_decentralised_training.py
without retraining.

04_decentralised_training.py's own evaluate() needs a live Algorithm,
which previously meant re-running the full (now potentially
nontrivial, --iterations=500-and-up) training loop just to look at the
policy again. This script instead loads only the saved RLModule
weights (ray.rllib.core.rl_module.multi_rl_module.MultiRLModule.
from_checkpoint, on the checkpoint's own
learner_group/learner/rl_module subdirectory) and runs the identical
sampling-based evaluation loop against it, so a long training run only
has to happen once.

Deliberately NOT training/utils.py::load_checkpoint (full
Algorithm.from_checkpoint): that reconstructs env runners too, which
need "warehouse_mapd_multi_agent_env" registered via
training/config.py::build_ppo_config's own register_env call --
something this script never does, since it has no reason to spin up
an Algorithm or a Ray cluster just to run forward_inference. Loading
the RLModule directly is also the right boundary: evaluation only
ever needs the trained weights, not the training machinery around
them.

Numbered 04b rather than its own milestone number: this is a variant of
04's own M5 worked example (docs/implementation_phases.md), the same
"NN + letter" convention 01b_storage_and_config.py and
01c_storage_state_on_instance.py already use, not a new build step.

The instance, storage and ExperimentConfig built below must match
04_decentralised_training.py's own build_instance/build_storage/
build_config exactly: the checkpoint's RLModule was built against that
graph's action/observation spaces (d_max, encoded tensor shapes), and
loading it against a different instance would silently produce
nonsense actions rather than a clean error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.controllers.decentralised import _sample_legal_slot
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType, is_feasible
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_and_validate
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv

FLEET_SIZE = 3
CHECKPOINT_DIR = Path(__file__).parent / "output" / "04_decentralised_training_checkpoint"


def build_instance() -> WarehouseGraph:
    # Must match 04_decentralised_training.py::build_instance exactly --
    # see module docstring.
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
    # Must match 04_decentralised_training.py::build_storage exactly.
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 20.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 8, storage_vertices[1]: 8}}
    return skus, capacities, counts


def build_config(seed: int) -> ExperimentConfig:
    # Must match 04_decentralised_training.py::build_config exactly
    # (regardless of --regime there, since this demo always trains F_fix).
    return ExperimentConfig(
        storage_mode="fixed",
        controller="decentralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=FLEET_SIZE,
        arrival_rate=0.3,
        seed=seed,
        horizon=300,
        wait_cost=0.5,
        observation_depth=3,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )


def evaluate(module, graph, skus, capacities, counts, config, num_episodes: int) -> None:
    """Identical to 04_decentralised_training.py::evaluate -- see its
    own docstring for why sampling, not greedy argmax. `module` is the
    loaded RLModule (main()'s load_rl_module), not a live Algorithm."""
    print(f"\n=== Evaluating the checkpoint over {num_episodes} fresh episodes ===")
    enc_config = LocalSubgraphEncodingConfig(max_local_nodes=12, max_messages=2)
    rng = np.random.default_rng(config.seed + 2000)

    for episode_idx in range(num_episodes):
        episode_config = config.model_copy(update={"seed": config.seed + 1000 + episode_idx})
        env = WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            counts,
            capacities,
            episode_config,
            enc_config,
            (episode_config.seed,),
        )
        obs, _ = env.reset()
        before_fleet = env._env.fleet
        completed_before = sum(1 for t in env._env.tasks if t.finish_time is not None)

        for _ in range(config.horizon):
            batch, agent_ids = _to_batch(obs)
            fwd_out = module.forward_inference(batch)
            logits = fwd_out["action_dist_inputs"]
            actions = {}
            for i, aid in enumerate(agent_ids):
                mask = obs[aid]["action_mask"]
                actions[aid] = _sample_legal_slot(logits[i].detach().numpy(), mask, rng)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            assert is_collision_free(before_fleet, env._env.fleet), "collision during evaluation"
            assert is_feasible(env._env.storage), "infeasible storage during evaluation"
            before_fleet = env._env.fleet
            if truncated["__all__"] or terminated["__all__"]:
                break

        completed_after = sum(1 for t in env._env.tasks if t.finish_time is not None)
        print(
            f"  episode {episode_idx + 1}/{num_episodes}: "
            f"tasks completed = {completed_after - completed_before}, "
            f"collisions = none, storage feasible = yes"
        )


def _to_batch(obs: dict) -> tuple[dict, list[str]]:
    # Identical to 04_decentralised_training.py::_to_batch.
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


def load_rl_module(checkpoint_dir: Path, policy_id: str = "shared_policy"):
    """The trained RLModule for `policy_id`, from the
    learner_group/learner/rl_module subdirectory save_checkpoint's
    Algorithm.save_to_path writes -- see module docstring for why this,
    not training/utils.py::load_checkpoint's full Algorithm, is loaded."""
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

    rl_module_dir = checkpoint_dir / "learner_group" / "learner" / "rl_module"
    multi_module = MultiRLModule.from_checkpoint(str(rl_module_dir.resolve()))
    return multi_module[policy_id]


def main() -> None:
    checkpoint_dir = CHECKPOINT_DIR
    num_episodes = 3
    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint="):
            checkpoint_dir = Path(arg.split("=", 1)[1])
        elif arg.startswith("--episodes="):
            num_episodes = int(arg.split("=", 1)[1])

    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"No checkpoint at {checkpoint_dir}; run 04_decentralised_training.py first, "
            "or pass --checkpoint=<path> to an existing one."
        )

    graph = build_instance()
    skus, capacities, counts = build_storage(graph)
    config = build_config(seed=0)

    print(f"Loading checkpoint from {checkpoint_dir}")
    module = load_rl_module(checkpoint_dir)

    evaluate(module, graph, skus, capacities, counts, config, num_episodes=num_episodes)


if __name__ == "__main__":
    main()
