"""PPO configuration for the learned controllers, with full parameter
sharing and disjoint training and evaluation seeds (sec:method:training,
tab:trainparams, #30).

Every PPO hyperparameter of tab:trainparams is TBD, so PPOHyperparameters
has a required field for each and no defaults. `c_wait` is a parameter of
the study, so it stays in ExperimentConfig.

TrainingRegime names the two regimes of sec:method:training, training under
F_fix only or under the storage rule used at evaluation. The caller picks
the storage mode and records which regime a run is.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    check_communication_caps,
)
from slap_mapd_coupling.models.rl_module import GNNActionMaskingRLModule
from slap_mapd_coupling.training.env import (
    CreditSignal,
    WarehouseMAPDMultiAgentEnv,
    Zones,
    arm_windows,
    check_arm,
)

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
    # No default, because the thesis leaves it open for the centralised and
    # section-based arms (#83). The decentralised arm takes "per_agent" only.
    credit_signal: CreditSignal


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
    seed_split: SeedSplitConfig,
    *,
    zones: Optional[Zones] = None,
    policy_id: str = DEFAULT_POLICY_ID,
    num_env_runners: int = 0,
    use_gpu: bool = False,
) -> PPOConfig:
    """A PPOConfig for the learned controller `experiment_config.controller`.
    It registers the adapter of training/env.py, maps every robot to
    `policy_id` (full parameter sharing) and uses models/rl_module.py's GNN
    as the RLModule.

    The adapter gets the credit signal from `ppo_hyperparameters`, the
    pooling from `gnn_encoder_config` and `zones` (the section arm only), so
    the module and the observations always agree on pooling. Every setting
    the adapter would reject raises here, before Ray starts, including
    pooling with the decentralised arm, team_mean with the decentralised arm,
    and a window above max_local_nodes.

    `seed_split` gives the adapter its training seeds and sets RLlib's own
    seed to `seed_split.base_seed`. `policy_id`, `num_env_runners` and
    `use_gpu` (one GPU for the local learner) are execution details, not
    study parameters, so they have defaults.
    """
    if experiment_config.discount is None:
        raise ValueError(
            "experiment_config.discount (gamma, the learning objective) is required to build a "
            "PPOConfig. ExperimentConfig already requires it for every controller, so this "
            "indicates a caller bug."
        )
    if experiment_config.communication != gnn_encoder_config.communication:
        raise ValueError(
            f"experiment_config.communication={experiment_config.communication} != "
            f"gnn_encoder_config.communication={gnn_encoder_config.communication}. The "
            "environment and the encoder must agree on whether messages are used."
        )
    credit_signal = ppo_hyperparameters.credit_signal
    pooling = gnn_encoder_config.pooling
    try:
        check_arm(experiment_config.controller, credit_signal, pooling, zones)
    except ValueError as error:
        raise ValueError(
            f"{error} (ppo_hyperparameters.credit_signal={credit_signal!r}, "
            f"gnn_encoder_config.pooling={pooling!r})"
        ) from error
    assert experiment_config.controller is not None  # check_arm rejects a planner
    check_communication_caps(encoding_config, experiment_config.communication)
    arm_windows(
        graph,
        experiment_config.controller,
        experiment_config.observation_depth,
        encoding_config,
        zones,
    )

    train_seeds, _ = split_seeds(seed_split)

    def env_creator(_env_config: dict) -> WarehouseMAPDMultiAgentEnv:
        # Registers the storage rules in WHICHEVER process actually calls
        # this -- with num_env_runners > 0, that is a separate worker
        # process, not the one that imported storage/fixed.py et al. at
        # the top of the caller's own script, and each process keeps its
        # own copy of storage/registry.py's module-level registry. These
        # imports are idempotent (Python only runs a module's top-level
        # code once per process, caching it in sys.modules), so repeated
        # env_creator calls within one worker never double-register.
        import slap_mapd_coupling.storage.congestion  # noqa: F401
        import slap_mapd_coupling.storage.demand  # noqa: F401
        import slap_mapd_coupling.storage.fixed  # noqa: F401

        return WarehouseMAPDMultiAgentEnv(
            graph,
            skus,
            initial_storage_counts,
            storage_capacities,
            experiment_config,
            encoding_config,
            train_seeds,
            credit_signal=credit_signal,
            pooling=pooling,
            zones=zones,
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
        # action_mask entirely -- it isn't mask-aware, and the action
        # mask's whole point is that an illegal slot IS an error (env.step()
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
        .debugging(seed=seed_split.base_seed)
        .learners(num_gpus_per_learner=1 if use_gpu else 0)
        .training(
            gamma=experiment_config.discount,
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
