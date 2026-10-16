"""A training run described by one YAML file, and the objects built from it.

The `slap-mapd train` command only loads this file, applies a few overrides from its options
and hands the result to training/trainer.py. The trainer holds no graph, so the instance, the
storage, the experiment config, the arm and the PPO hyperparameters are all built here from the
file. configs/decentralised.yaml is the file of the worked example,
examples/04_decentralised_training.py.

Every number in the shipped file is illustrative, because tab:trainparams and tab:modelparams
leave them TBD. A run records which values it used by keeping the file it was started from.
`wait_cost` and `num_agents` are written once and shared with the instance, because each is one
study parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from ray.rllib.algorithms.ppo import PPOConfig

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import (
    ControllerArchitecture,
    ExperimentConfig,
    StorageMode,
)
from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.instances.generator import (
    GeneratedInstance,
    GeneratorParams,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.training.arm import Arm, build_arm
from slap_mapd_coupling.training.config import (
    PPOHyperparameters,
    SeedSplitConfig,
    TrainingRegime,
    build_ppo_config,
)
from slap_mapd_coupling.training.trainer import CheckpointSettings


class StorageSettings(BaseModel):
    """The SKUs and where they start.

    There are `num_skus` SKUs named sku_00, sku_01 and so on. Their unit capacities b_k cycle
    through `unit_capacities`, so the SKUs differ in size. Every storage vertex has the capacity
    `vertex_capacity`. SKU number i starts with `initial_units` units at storage vertex number
    i mod |V_str|, with the storage vertices in increasing vertex number, so the SKUs spread over
    all storage vertices. A new task picks a uniformly random (SKU, vertex) pair that is in stock,
    so more SKUs mean more distinct pickups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_skus: PositiveInt
    unit_capacities: tuple[PositiveFloat, ...] = Field(min_length=1)
    vertex_capacity: PositiveFloat
    initial_units: PositiveInt


class ExperimentSettings(BaseModel):
    """The ExperimentConfig fields a run shares across its episodes. The controller, the storage
    mode, the arrival rate, the horizon and the communication flag are run settings of
    TrainingSpec, because the command line overrides them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_agents: PositiveInt
    wait_cost: NonNegativeFloat
    keep_up_threshold: PositiveFloat
    observation_depth: PositiveInt
    discount: PositiveFloat
    deliver_reward: NonNegativeFloat
    override_penalty: NonNegativeFloat
    congestion_reward_weight: NonNegativeFloat
    congestion_sensitive: bool = False
    seed: int = 0


class ModelSettings(BaseModel):
    """The size of the GNN encoder and of the policy and value head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_rounds: PositiveInt
    hidden_width: PositiveInt
    head_hidden_width: PositiveInt


class ArmSettings(BaseModel):
    """`num_zones` only matters to the section arm. `max_messages` is the cap on the messages an
    agent hears and defaults to the other agents of the fleet, so no sender is dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_zones: PositiveInt = 3
    max_messages: PositiveInt | None = None


class TrainingSpec(BaseModel):
    """The whole of one training run, the root of the YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller: ControllerArchitecture
    regime: TrainingRegime = "fixed_only"
    communication: bool = False
    arrival_rate: PositiveFloat
    horizon: PositiveInt
    # The storage rule a policy is evaluated under, which "matched" training uses too.
    evaluation_storage_mode: StorageMode = "fixed"
    experiment: ExperimentSettings
    # The GeneratorParams without wait_cost and num_endpoints, which follow the experiment.
    instance: dict[str, Any]
    storage: StorageSettings
    arm: ArmSettings = ArmSettings()
    model: ModelSettings
    ppo: PPOHyperparameters
    train_seed_count: PositiveInt
    eval_seed_count: PositiveInt
    checkpoint: CheckpointSettings = CheckpointSettings()


class Study(NamedTuple):
    """Everything `build_ppo_config` and an evaluation need, built from a TrainingSpec."""

    instance: GeneratedInstance
    skus: dict[SkuId, SkuType]
    capacities: dict[VertexId, float]
    counts: dict[SkuId, dict[VertexId, int]]
    config: ExperimentConfig
    arm: Arm
    gnn_encoder: GNNEncoderConfig
    head: PolicyValueHeadConfig
    seed_split: SeedSplitConfig


def load_spec(path: Path) -> TrainingSpec:
    """The spec in the YAML file at `path`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping of training settings")
    return TrainingSpec.model_validate(data)


def override_spec(
    spec: TrainingSpec,
    *,
    controller: ControllerArchitecture | None = None,
    regime: TrainingRegime | None = None,
    communication: bool | None = None,
    arrival_rate: float | None = None,
    horizon: int | None = None,
    budget_env_steps: int | None = None,
    seed: int | None = None,
    checkpoint_every: int | None = None,
    keep_checkpoints: int | None = None,
) -> TrainingSpec:
    """`spec` with the settings given, for the options of the command line. A value of None
    leaves the file's own setting alone. The result is validated again, so a combination the
    file would reject (communication on a centralised arm, say) raises here too."""
    top = {
        "controller": controller,
        "regime": regime,
        "communication": communication,
        "arrival_rate": arrival_rate,
        "horizon": horizon,
    }
    data = spec.model_dump()
    data.update({key: value for key, value in top.items() if value is not None})
    if budget_env_steps is not None:
        data["ppo"]["training_budget_env_steps"] = budget_env_steps
    if seed is not None:
        data["experiment"]["seed"] = seed
    if checkpoint_every is not None:
        data["checkpoint"]["frequency"] = checkpoint_every
    if keep_checkpoints is not None:
        data["checkpoint"]["num_to_keep"] = keep_checkpoints
    return TrainingSpec.model_validate(data)


def with_budget(spec: TrainingSpec, budget_env_steps: int | None) -> TrainingSpec:
    """`spec` with another training budget, or the same one when `budget_env_steps` is None."""
    return override_spec(spec, budget_env_steps=budget_env_steps)


def build_instance(spec: TrainingSpec) -> GeneratedInstance:
    """The generated instance, which keeps the aisle index the section arm needs."""
    params = GeneratorParams(
        **spec.instance,
        wait_cost=spec.experiment.wait_cost,
        num_endpoints=spec.experiment.num_agents,
    )
    instance = generate_instance(params)
    connectivity = check_connectivity(instance.graph)
    well_formedness = check_well_formedness(instance.graph, spec.experiment.num_agents)
    if not (connectivity.ok and well_formedness.ok):
        raise RuntimeError(
            f"Instance rejected: connectivity_ok={connectivity.ok}, "
            f"well_formedness_ok={well_formedness.ok}."
        )
    return instance


def build_storage(
    storage: StorageSettings, graph: WarehouseGraph
) -> tuple[dict[SkuId, SkuType], dict[VertexId, float], dict[SkuId, dict[VertexId, int]]]:
    """The SKUs, the capacity of every storage vertex and the initial counts, as described by
    `StorageSettings`. Raises if the initial counts do not fit the capacity (eq:feasiblestorage)."""
    storage_vertices = sorted(v for v, vertex in graph.vertices.items() if vertex.role.storage)
    skus: dict[SkuId, SkuType] = {}
    counts: dict[SkuId, dict[VertexId, int]] = {}
    used = dict.fromkeys(storage_vertices, 0.0)
    for i in range(storage.num_skus):
        sku_id = f"sku_{i:02d}"
        unit_capacity = storage.unit_capacities[i % len(storage.unit_capacities)]
        vertex = storage_vertices[i % len(storage_vertices)]
        skus[sku_id] = SkuType(sku_id=sku_id, unit_capacity=unit_capacity)
        counts[sku_id] = {vertex: storage.initial_units}
        used[vertex] += unit_capacity * storage.initial_units
    over = {v: u for v, u in used.items() if u > storage.vertex_capacity}
    if over:
        raise ValueError(
            f"The initial counts do not fit vertex_capacity={storage.vertex_capacity}: "
            f"used capacity by vertex {over}. Lower num_skus or initial_units, or raise "
            "vertex_capacity."
        )
    return skus, {v: storage.vertex_capacity for v in storage_vertices}, counts


def build_experiment_config(spec: TrainingSpec, seed: int | None = None) -> ExperimentConfig:
    """The configuration a run trains under. "fixed_only" trains under F_fix and "matched" under
    the storage rule used at evaluation (sec:method:training)."""
    experiment = spec.experiment
    storage_mode = "fixed" if spec.regime == "fixed_only" else spec.evaluation_storage_mode
    return ExperimentConfig(
        storage_mode=storage_mode,
        controller=spec.controller,
        congestion_sensitive=experiment.congestion_sensitive,
        communication=spec.communication,
        num_agents=experiment.num_agents,
        arrival_rate=spec.arrival_rate,
        keep_up_threshold=experiment.keep_up_threshold,
        seed=experiment.seed if seed is None else seed,
        horizon=spec.horizon,
        wait_cost=experiment.wait_cost,
        observation_depth=experiment.observation_depth,
        discount=experiment.discount,
        deliver_reward=experiment.deliver_reward,
        override_penalty=experiment.override_penalty,
        congestion_reward_weight=experiment.congestion_reward_weight,
    )


def build_study(spec: TrainingSpec) -> Study:
    instance = build_instance(spec)
    skus, capacities, counts = build_storage(spec.storage, instance.graph)
    config = build_experiment_config(spec)
    max_messages = spec.arm.max_messages or max(spec.experiment.num_agents - 1, 1)
    arm = build_arm(
        instance,
        spec.controller,
        config.observation_depth,
        num_zones=spec.arm.num_zones,
        max_messages=max_messages,
    )
    return Study(
        instance=instance,
        skus=skus,
        capacities=capacities,
        counts=counts,
        config=config,
        arm=arm,
        gnn_encoder=GNNEncoderConfig(
            num_rounds=spec.model.num_rounds,
            hidden_width=spec.model.hidden_width,
            communication=config.communication,
            pooling=arm.pooling,
        ),
        head=PolicyValueHeadConfig(hidden_width=spec.model.head_hidden_width),
        seed_split=SeedSplitConfig(
            base_seed=config.seed,
            train_seed_count=spec.train_seed_count,
            eval_seed_count=spec.eval_seed_count,
        ),
    )


def build_study_ppo_config(
    spec: TrainingSpec,
    study: Study,
    *,
    num_env_runners: int = 0,
    use_gpu: bool = False,
) -> PPOConfig:
    """The PPOConfig of a run, which `trainer.train` takes."""
    return build_ppo_config(
        study.instance.graph,
        study.skus,
        study.counts,
        study.capacities,
        study.config,
        study.arm.encoding,
        study.gnn_encoder,
        study.head,
        spec.ppo,
        study.seed_split,
        zones=study.arm.zones,
        num_env_runners=num_env_runners,
        use_gpu=use_gpu,
    )
