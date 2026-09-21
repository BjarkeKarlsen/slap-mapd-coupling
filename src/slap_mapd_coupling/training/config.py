"""PPO configuration: full parameter sharing, disjoint train/eval seeds,
storage-rule training regime (sec:method:training, tab:trainparams,
issue #30).

Every PPO core hyperparameter tab:trainparams lists (learning rate, clip
epsilon, GAE lambda, minibatch size, epochs/update, rollout length,
entropy coeff., value-loss coeff.) is TBD-by-sweep -- exposed here as
named, required PPOHyperparameters fields, never hard-coded, matching
the same "no default for a thesis-undetermined value" convention
models/gnn_encoder.py's GNNEncoderConfig already established.

`c_wait` is deliberately NOT part of PPOHyperparameters: it is already an
ExperimentConfig field (a parameter of the STUDY), and tab:trainparams's
own text says to report it "alongside training configuration," not fold
it into the PPO config table -- so it stays wherever ExperimentConfig is
threaded through, not duplicated here.

Training regime (TrainingRegime): "fixed_only" (train under F_fix,
evaluate against all three storage rules) vs "matched" (train under the
same storage rule used at evaluation) -- sec:method:training is explicit
these must be "distinguished and reported as an experimental variable,
not folded together." This module only names the regime; which
ExperimentConfig.storage_mode a given run actually trains under is the
caller's concern (build_ppo_config takes one already-built environment
instance, not a regime switch) -- TrainingRegime exists so the caller
records / reports which one a run represents, not so this module can
silently pick a storage_mode.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.spaces import max_out_degree
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.training.env import WarehouseMAPDMultiAgentEnv

TrainingRegime = Literal["fixed_only", "matched"]  # sec:method:training
DEFAULT_POLICY_ID = "shared_policy"  # full parameter sharing, sec:method:training
_ENV_NAME = "warehouse_mapd_multi_agent_env"


class PPOHyperparameters(BaseModel):
    """PPO core hyperparameters (tab:trainparams). All TBD-by-sweep --
    required, no defaults, see module docstring. `train_batch_size` isn't
    itself one of tab:trainparams's named rows, but RLlib's own API
    requires it to actually build an Algorithm; distinguished from
    rollout_fragment_length (per-env-runner fragment size) rather than
    silently derived from it, since the relationship between the two
    depends on num_env_runners (an execution detail, not a study
    parameter) -- flagged as an RLlib API requirement, not a thesis one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: PositiveFloat
    clip_param: PositiveFloat  # clip epsilon
    gae_lambda: PositiveFloat
    minibatch_size: PositiveInt
    num_epochs: PositiveInt  # epochs/update
    rollout_fragment_length: PositiveInt
    entropy_coeff: NonNegativeFloat
    vf_loss_coeff: PositiveFloat
    train_batch_size: PositiveInt  # RLlib API requirement, see docstring


class SeedSplitConfig(BaseModel):
    """train/eval seed counts (tab:trainparams: "disjoint split", TBD).
    Required, no defaults -- same convention as PPOHyperparameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_seed: int
    train_seed_count: PositiveInt
    eval_seed_count: PositiveInt


def split_seeds(config: SeedSplitConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(train_seeds, eval_seeds), disjoint by construction: two
    contiguous, non-overlapping ranges starting at base_seed. The
    thesis pins disjointness, not a specific splitting scheme -- a
    contiguous partition is the simplest one that guarantees it
    trivially (no coordination needed beyond the two counts), matching
    the same "simplest choice that satisfies the stated constraint"
    convention used for _ORDER_SEED_OFFSET elsewhere in this repo."""
    train_seeds = tuple(range(config.base_seed, config.base_seed + config.train_seed_count))
    eval_start = config.base_seed + config.train_seed_count
    eval_seeds = tuple(range(eval_start, eval_start + config.eval_seed_count))
    return train_seeds, eval_seeds


def build_ppo_config(
    graph: WarehouseGraph,
    skus: Mapping[SkuId, SkuType],
    initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
    storage_capacities: Mapping[VertexId, float],
    experiment_config: ExperimentConfig,
    encoding_config: LocalSubgraphEncodingConfig,
    gnn_encoder_config: GNNEncoderConfig,
    policy_value_head_config: PolicyValueHeadConfig,
    ppo_hyperparameters: PPOHyperparameters,
    *,
    policy_id: str = DEFAULT_POLICY_ID,
    num_env_runners: int = 0,
) -> PPOConfig:
    """Assembles a ray.rllib.algorithms.ppo.PPOConfig for the
    decentralised regime: registers training/env.py's MultiAgentEnv
    adapter, wires full parameter sharing (every agent -> policy_id,
    RLlib's standard multi-agent scaling lever), and plugs
    models/rl_module.py's GNN architecture in as the RLModule instead of
    RLlib's Catalog-built default -- see module docstring for what's
    TBD vs fixed here. `experiment_config.controller` must already be
    "decentralised" (WarehouseMAPDMultiAgentEnv enforces this itself).

    `policy_id`/`num_env_runners` are naming/execution details, not
    study parameters -- defaulted, unlike every PPOHyperparameters/
    SeedSplitConfig field above.
    """
    d_max = max_out_degree(graph)
    expected_num_actions = d_max + 1
    if policy_value_head_config.num_actions != expected_num_actions:
        raise ValueError(
            f"policy_value_head_config.num_actions={policy_value_head_config.num_actions} "
            f"!= max_out_degree(graph)+1={expected_num_actions} (eq:mask); the policy head's "
            "action count must match this instance's actual d_max, not an unrelated value."
        )

    def env_creator(_env_config: dict) -> WarehouseMAPDMultiAgentEnv:
        return WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            initial_storage_counts,
            storage_capacities,
            experiment_config,
            encoding_config,
        )

    register_env(_ENV_NAME, env_creator)

    model_config: dict[str, Any] = {
        "gnn_encoder_config": gnn_encoder_config.model_dump(),
        "policy_value_head_config": policy_value_head_config.model_dump(),
    }

    config = (
        PPOConfig()
        # disable_env_checking: RLlib's generic multi-agent env pre-check
        # samples straight from action_space.sample(), ignoring
        # action_mask entirely -- it isn't mask-aware, and eq:mask's
        # whole point is that an illegal slot IS an error (env.step()
        # legitimately raises for one, see environment/spaces.py::
        # action_for_slot). Real rollouts never hit this: the RLModule
        # masks illegal logits to -inf before sampling (#26), so the
        # policy never proposes one. Flagged, not a silent workaround.
        .environment(env=_ENV_NAME, disable_env_checking=True)
        .multi_agent(
            policies={policy_id},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: policy_id,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=GNNActionMaskingRLModule,
                model_config=model_config,
            ),
        )
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=ppo_hyperparameters.rollout_fragment_length,
        )
        .training(
            lr=ppo_hyperparameters.lr,
            clip_param=ppo_hyperparameters.clip_param,
            lambda_=ppo_hyperparameters.gae_lambda,
            minibatch_size=ppo_hyperparameters.minibatch_size,
            num_epochs=ppo_hyperparameters.num_epochs,
            entropy_coeff=ppo_hyperparameters.entropy_coeff,
            vf_loss_coeff=ppo_hyperparameters.vf_loss_coeff,
            train_batch_size=ppo_hyperparameters.train_batch_size,
        )
    )
    return config
