"""PPO configuration for the learned controllers, with full parameter
sharing and disjoint training, validation and test seeds (sec:method:training,
tab:trainparams, #30).

Every PPO hyperparameter of tab:trainparams is TBD, so PPOHyperparameters
has a required field for each and no defaults, the training budget included.
`c_wait` is a parameter of the study, so it stays in ExperimentConfig.

TrainingRegime names the two regimes of sec:method:training, training under
F_fix only or under the storage rule used at evaluation. The caller picks
the storage mode and records which regime a run is.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, NamedTuple, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
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
    WarehouseMAPDMultiAgentEnv,
    Zones,
    arm_window_tables,
    check_arm,
)
from slap_mapd_coupling.training.learner import BATCH_ON_CPU, GNNPPOTorchLearner

TrainingRegime = Literal["fixed_only", "matched"]  # sec:method:training
DEFAULT_POLICY_ID = "shared_policy"  # full parameter sharing, sec:method:training
_ENV_NAME = "warehouse_mapd_multi_agent_env"


class PPOHyperparameters(BaseModel):
    """PPO core hyperparameters (tab:trainparams). All TBD-by-sweep --
    required, no defaults, see module docstring.

    Two PPO settings are not fields here, because both follow from the episode
    length, the horizon of the experiment config. A sampling call of a runner is
    one complete episode (`batch_mode="complete_episodes"`), since a timestep can
    take several sub-steps and a fixed fragment length would cut an episode in the
    middle. The train batch is `max(num_env_runners, 1) * horizon`. Every timestep
    has at least one sub-step, so one round of sampling always reaches it and an
    iteration is exactly one episode per runner. That holds only if the round waits
    for every runner, so `sample_timeout_s` is None. RLlib's default of 60 seconds
    drops the episode of a runner that takes longer, which a centralised episode on
    the 8-agent warehouse does (one policy call per agent turn), and the iteration
    then trains on the episodes that arrived in time. TrainingProgressCallbacks stops
    a run whose iteration has fewer episodes than runners. This is a choice, flagged
    for the author, with no thesis edit. RLlib requires minibatch_size to be at most
    that batch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lr: PositiveFloat
    clip_param: PositiveFloat  # clip epsilon
    gae_lambda: PositiveFloat
    minibatch_size: PositiveInt
    num_epochs: PositiveInt  # epochs/update
    entropy_coeff: NonNegativeFloat
    vf_loss_coeff: PositiveFloat
    # The training budget of tab:trainparams, equal across the learned arms. One environment
    # step is one timestep, however many agents choose within it. build_ppo_config does not
    # read it. A run stops on it at the first iteration at or above it (training/trainer.py),
    # so it overshoots by up to one iteration (num_env_runners * horizon timesteps).
    training_budget_env_steps: PositiveInt


# Where the validation and test seeds start. They are the same for every run, whatever its
# training seed, so two training seeds are always scored on the same scenarios. The training
# seeds of a run must stay below VALIDATION_SEED_START, so the three sets never meet.
VALIDATION_SEED_START = 100_000
TEST_SEED_START = 200_000


class SeedSplitConfig(BaseModel):
    """The training, validation and test seeds of a run (tab:trainparams: "disjoint split", TBD).
    The counts are required with no defaults, the same convention as PPOHyperparameters.

    `training_seed` is the seed a run is started with. It seeds RLlib (the network's starting
    weights and the action sampling of training) and is the first of the run's
    `train_seed_count` training scenarios. The validation and test seeds do not depend on it,
    see VALIDATION_SEED_START."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    training_seed: NonNegativeInt
    train_seed_count: PositiveInt
    validation_seed_count: PositiveInt
    test_seed_count: PositiveInt

    @model_validator(mode="after")
    def _disjoint(self) -> "SeedSplitConfig":
        if self.training_seed + self.train_seed_count > VALIDATION_SEED_START:
            raise ValueError(
                f"The training seeds {self.training_seed} to "
                f"{self.training_seed + self.train_seed_count - 1} reach the validation seeds, "
                f"which start at {VALIDATION_SEED_START}. Use a smaller training seed or fewer "
                "training seeds."
            )
        if VALIDATION_SEED_START + self.validation_seed_count > TEST_SEED_START:
            raise ValueError(
                f"validation_seed_count={self.validation_seed_count} reaches the test seeds, "
                f"which start at {TEST_SEED_START}."
            )
        return self


class SeedSplit(NamedTuple):
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def split_seeds(config: SeedSplitConfig) -> SeedSplit:
    """The training, validation and test seeds, three contiguous ranges that are disjoint by
    construction (SeedSplitConfig checks it). Training cycles through the training seeds. The
    validation seeds choose a run's checkpoint and the test seeds score the chosen one, so the
    reported result is never measured on the seeds that chose it."""
    return SeedSplit(
        train=tuple(range(config.training_seed, config.training_seed + config.train_seed_count)),
        validation=tuple(
            range(VALIDATION_SEED_START, VALIDATION_SEED_START + config.validation_seed_count)
        ),
        test=tuple(range(TEST_SEED_START, TEST_SEED_START + config.test_seed_count)),
    )


def first_seed_index(worker_index: int, vector_index: int = 0, envs_per_runner: int = 1) -> int:
    """Where in the training cycle the env of one runner starts.

    The thesis fixes the training and evaluation seeds as a disjoint split,
    not how training draws from the training seeds. Training cycles through
    them in order (training/env.py). Every runner builds its own env, so if
    every cycle started at the first seed the runners would draw the same
    scenario at the same time. This rule starts them apart:
    worker_index * envs_per_runner + vector_index, which numbers the envs of
    all runners consecutively. The adapter takes it modulo the number of
    seeds, so more envs than seeds wrap around. Worker 0 with vector index 0,
    the single-process case, starts at 0 and draws the same sequence as
    without an offset. This is an implementation choice, not a thesis value.

    RLlib's multi-agent runner passes one EnvContext to all the envs of a
    runner, so vector_index is 0 there and the envs of one runner (RLlib's
    default is one) share a start.
    """
    return worker_index * envs_per_runner + vector_index


def key_prefix(worker_index: int, vector_index: int = 0) -> int:
    """The prefix of the row keys of one runner's env (training/env.py::row_key), so
    two runners never give the same key. The envs of one runner share a vector
    index (see first_seed_index), so this needs one env per runner, RLlib's
    default, which build_ppo_config keeps."""
    return worker_index * 16 + vector_index


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
    batch_on_cpu: bool = False,
    value_chunk_rows: Optional[int] = None,
) -> PPOConfig:
    """A PPOConfig for the learned controller `experiment_config.controller`.
    It registers the adapter of training/env.py, maps every robot to
    `policy_id` (full parameter sharing) and uses models/rl_module.py's GNN
    as the RLModule.

    The adapter gets the pooling from `gnn_encoder_config` and `zones` (the
    section arm only), so
    the module and the observations always agree on pooling. Every setting
    the adapter would reject raises here, before Ray starts, including
    pooling with the decentralised arm and a window above max_local_nodes.

    `seed_split` gives the adapter its training seeds and sets RLlib's own
    seed to `seed_split.training_seed`. `policy_id`, `num_env_runners` and
    `use_gpu` (one GPU for the local learner) are execution details, not
    study parameters, so they have defaults. So are `batch_on_cpu`, which keeps
    the train batch in host memory and moves one minibatch at a time to the GPU
    (training/learner.py), and `value_chunk_rows`, the rows per encoder pass of
    the value pass (environment/action_masking.py, 512 when None). Neither
    changes a result.
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
    pooling = gnn_encoder_config.pooling
    try:
        check_arm(experiment_config.controller, pooling, zones)
    except ValueError as error:
        raise ValueError(
            f"{error} (gnn_encoder_config.pooling={pooling!r})"
        ) from error
    assert experiment_config.controller is not None  # check_arm rejects a planner
    check_communication_caps(encoding_config, experiment_config.communication)
    window_tables = arm_window_tables(
        graph,
        experiment_config.controller,
        experiment_config.observation_depth,
        encoding_config,
        zones,
        pooling,
    )

    train_seeds = split_seeds(seed_split).train

    def env_creator(env_context: dict) -> WarehouseMAPDMultiAgentEnv:
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
            pooling=pooling,
            zones=zones,
            # RLlib passes an EnvContext. A plain dict (no runner) starts at 0.
            first_seed_index=first_seed_index(
                getattr(env_context, "worker_index", 0),
                getattr(env_context, "vector_index", 0),
            ),
            seed_from_cycle=True,
            key_prefix=key_prefix(
                getattr(env_context, "worker_index", 0),
                getattr(env_context, "vector_index", 0),
            ),
        )

    register_env(_ENV_NAME, env_creator)

    model_config: dict[str, Any] = {
        "gnn_encoder_config": gnn_encoder_config.model_dump(),
        "policy_value_head_config": policy_value_head_config.model_dump(),
        # The adapter stores compact samples (models/window_tables.py), and the module
        # rebuilds the padded arrays from these tables.
        "window_tables": window_tables.as_arrays(),
    }
    if value_chunk_rows is not None:
        if value_chunk_rows < 1:
            raise ValueError(f"value_chunk_rows must be positive, got {value_chunk_rows}.")
        model_config["value_chunk_rows"] = value_chunk_rows

    config = (
        PPOConfig()
        # RLlib's new API stack, its default, set here so the config states it.
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
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
            # One sampling call of a runner is one complete episode, see PPOHyperparameters.
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            # Wait for every runner, so an iteration is one episode per runner on every arm.
            sample_timeout_s=None,
        )
        .debugging(seed=seed_split.training_seed)
        # The learner adds the senders' rows to each minibatch and can keep the batch in
        # host memory (training/learner.py).
        .learners(
            num_gpus_per_learner=1 if use_gpu else 0,
            learner_class=GNNPPOTorchLearner,
        )
        .training(
            gamma=experiment_config.discount,
            lr=ppo_hyperparameters.lr,
            clip_param=ppo_hyperparameters.clip_param,
            lambda_=ppo_hyperparameters.gae_lambda,
            # One episode per runner, see PPOHyperparameters.
            train_batch_size_per_learner=max(num_env_runners, 1) * experiment_config.horizon,
            minibatch_size=ppo_hyperparameters.minibatch_size,
            num_epochs=ppo_hyperparameters.num_epochs,
            entropy_coeff=ppo_hyperparameters.entropy_coeff,
            vf_loss_coeff=ppo_hyperparameters.vf_loss_coeff,
            learner_config_dict={BATCH_ON_CPU: batch_on_cpu},
        )
    )
    return config
