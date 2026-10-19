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

import math
import statistics
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from ray.rllib.algorithms.ppo import PPOConfig

import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import (
    ControllerArchitecture,
    ExperimentConfig,
    LinesPerOrder,
    ReceivingVertices,
    StorageMode,
)
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import popularity_ranking, zipf_weights
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

    There are `num_skus` SKUs, named by `sku_names` when it is given and sku_000, sku_001 and so
    on otherwise. The popularity ranking of `DemandSettings` starts from the order of that list,
    so the first name is the most popular at t = 0. Their unit capacities b_k cycle through `unit_capacities`, so the SKUs differ in size.
    Every storage vertex has the capacity `vertex_capacity`. Every SKU starts with
    `initial_units` units, all at one storage vertex, chosen by `initial_placement`:

    - "by_number": SKU number i at storage vertex number i mod |V_str|, with the storage
      vertices in increasing vertex number, so the SKUs spread over all storage vertices.
    - "demand_blocks": the starting stock x_0 of sec:impl:instances. The storage vertices are
      ordered by their mean d_G(v, g) over the delivery vertices g, ties by vertex number, since
      an order picks its delivery vertex uniformly. The SKUs are ranked by popularity at t = 0,
      and the SKU at rank r goes to vertex number r // ceil(|K| / |V_str|) in that order, so the
      most popular SKUs fill the vertex nearest the delivery vertices first.
    - "demand_sized_blocks": the blocks of "demand_blocks", with each SKU's units sized to its
      expected units ordered over one arrival period, for the arrival stock model. The SKU at
      rank r starts with max(initial_units, ceil((1 + epsilon) * lambda_task * Delta * q_bar *
      w_r)) units, w_r being its popularity at t = 0 and q_bar the mean of `units_per_line` (1
      without it), so every SKU holds at least `initial_units`. It needs
      the `arrival` section for epsilon and the `storage_rule` section for Delta.

    A new task picks a SKU in stock, then a vertex holding it, so more SKUs mean more distinct
    pickups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_skus: PositiveInt
    sku_names: tuple[str, ...] | None = None
    unit_capacities: tuple[PositiveFloat, ...] = Field(min_length=1)
    vertex_capacity: PositiveFloat
    initial_units: PositiveInt
    initial_placement: Literal["by_number", "demand_blocks", "demand_sized_blocks"] = "by_number"

    @model_validator(mode="after")
    def _one_name_per_sku(self) -> "StorageSettings":
        if self.sku_names is None:
            return self
        if len(self.sku_names) != self.num_skus:
            raise ValueError(
                f"sku_names has {len(self.sku_names)} names for num_skus={self.num_skus}."
            )
        if len(set(self.sku_names)) != len(self.sku_names) or not all(self.sku_names):
            raise ValueError("sku_names must be distinct and non-empty.")
        return self

    def sku_ids(self) -> list[SkuId]:
        """The SKU ids in list order, the order the popularity ranking starts from."""
        if self.sku_names is not None:
            return list(self.sku_names)
        return [f"sku_{i:03d}" for i in range(self.num_skus)]


class DemandSettings(BaseModel):
    """The order distribution P_ord (sec:pf:tasks) when SKUs differ in popularity.

    The SKU at rank r weighs 1 / (r + 1)^skew, rank 0 being the most popular. The ranking starts
    as the SKUs in the order of `StorageSettings.sku_ids` and rotates left by `rotation_shift` places every `rotation_period`
    steps, or never without a period. A spec without this section draws the SKU uniformly among
    the SKUs in stock.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skew: NonNegativeFloat
    rotation_period: PositiveInt | None = None
    rotation_shift: PositiveInt | None = None

    @model_validator(mode="after")
    def _rotation_parameters(self) -> "DemandSettings":
        """A period and a shift only mean something together."""
        if (self.rotation_period is None) != (self.rotation_shift is None):
            raise ValueError("Set both rotation_period and rotation_shift, or neither.")
        return self


class StorageRuleSettings(BaseModel):
    """The slow loop of the storage rules (sec:pf:storage).

    `epoch_length` is the storage update interval Delta in steps, and under the arrival stock
    model the arrival period of every rule, F_fix included. `reassignment_cap` is nu, the most
    units one update may relocate. The adaptive rules of the template model need it, and the
    arrival model relocates nothing and takes none. `congestion_weight` is beta, which only the
    congestion rule uses. None of them has a default, because the thesis leaves them open
    (tab:evalparams).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch_length: PositiveInt
    reassignment_cap: PositiveInt | None = None
    congestion_weight: NonNegativeFloat | None = None


class ArrivalSettings(BaseModel):
    """The arrival stock model (thesis wording pending). A spec without this section uses the
    template model, where stock never runs out.

    Every Delta steps stock arrives at the `receiving` vertices, and the storage rule decides
    which shelves it goes to. Each SKU is filled up to
    max(initial_units, ceil((1 + margin) * its orders in the last period)), so `initial_units`
    of the `storage` section is the floor. `margin` is epsilon. Neither has a default, because
    both are study parameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin: NonNegativeFloat
    receiving: ReceivingVertices
    # Order lines (tab:evalparams). None is one line of one unit per order.
    lines_per_order: LinesPerOrder | None = None
    units_per_line: tuple[PositiveInt, PositiveInt] | None = None
    # The SKUs that count as popular in the cold-stock measure. None leaves it off.
    cold_stock_top_k: PositiveInt | None = None


class StockSizing(NamedTuple):
    """What "demand_sized_blocks" sizes the starting stock by: lambda_task, Delta, epsilon and
    q_bar, the mean units per line."""

    arrival_rate: float
    epoch_length: int
    margin: float
    mean_units: float = 1.0


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
    move_cost: NonNegativeFloat
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
    # The order distribution. None is the uniform draw of the shipped configs.
    demand: DemandSettings | None = None
    # Delta, nu and beta of the storage rules. None is enough for a template run that only
    # ever plays F_fix.
    storage_rule: StorageRuleSettings | None = None
    # The arrival stock model. None is the template model of the shipped configs.
    arrival: ArrivalSettings | None = None
    arm: ArmSettings = ArmSettings()
    model: ModelSettings
    ppo: PPOHyperparameters
    # The seed split, with experiment.seed as the training seed (training/config.py::split_seeds).
    train_seed_count: PositiveInt
    validation_seed_count: PositiveInt
    test_seed_count: PositiveInt
    checkpoint: CheckpointSettings = CheckpointSettings()

    @model_validator(mode="after")
    def _arrival_settings(self) -> "TrainingSpec":
        """The arrival model needs Delta, its arrival period, under every storage rule and
        relocates nothing, so it takes no nu. The sized starting stock is the arrival model's,
        sized by its margin and Delta."""
        if self.arrival is not None:
            if self.storage_rule is None:
                raise ValueError(
                    "the arrival section needs a storage_rule section with epoch_length, the "
                    "arrival period, and there is none."
                )
            if self.storage_rule.reassignment_cap is not None:
                raise ValueError(
                    "the arrival model relocates nothing, so the storage_rule section takes no "
                    "reassignment_cap."
                )
        elif self.storage.initial_placement == "demand_sized_blocks":
            raise ValueError(
                "initial_placement 'demand_sized_blocks' needs the arrival section for its margin."
            )
        return self

    @model_validator(mode="after")
    def _matched_training_has_its_storage_rule(self) -> "TrainingSpec":
        """ "matched" trains under the evaluation storage rule, so that rule's settings must be
        there at load time and not when the first episode is built."""
        if self.regime == "matched":
            check_storage_rule(self, self.evaluation_storage_mode)
        return self


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
    storage: StorageSettings,
    graph: WarehouseGraph,
    demand: DemandSettings | None = None,
    sizing: StockSizing | None = None,
) -> tuple[dict[SkuId, SkuType], dict[VertexId, float], dict[SkuId, dict[VertexId, int]]]:
    """The SKUs, the capacity of every storage vertex and the initial counts, as described by
    `StorageSettings`. `demand` gives the popularity ranking "demand_blocks" and
    "demand_sized_blocks" place by, and None ranks the SKUs in list order, as the uniform draw
    has no ranking. "demand_sized_blocks" needs `sizing`. Raises if the initial counts break the
    capacity constraint."""
    storage_vertices = sorted(v for v, vertex in graph.vertices.items() if vertex.role.storage)
    skus: dict[SkuId, SkuType] = {}
    for i, sku_id in enumerate(storage.sku_ids()):
        unit_capacity = storage.unit_capacities[i % len(storage.unit_capacities)]
        skus[sku_id] = SkuType(sku_id=sku_id, unit_capacity=unit_capacity)
    units = dict.fromkeys(skus, storage.initial_units)
    if storage.initial_placement in ("demand_blocks", "demand_sized_blocks"):
        home = _demand_blocks(list(skus), graph, storage_vertices, demand)
    else:
        home = {
            sku_id: storage_vertices[i % len(storage_vertices)] for i, sku_id in enumerate(skus)
        }
    if storage.initial_placement == "demand_sized_blocks":
        if sizing is None:
            raise ValueError("initial_placement 'demand_sized_blocks' needs the stock sizing.")
        units = _sized_units(list(skus), storage.initial_units, demand, sizing)
    counts: dict[SkuId, dict[VertexId, int]] = {}
    used = dict.fromkeys(storage_vertices, 0.0)
    for sku_id, sku in skus.items():
        counts[sku_id] = {home[sku_id]: units[sku_id]}
        used[home[sku_id]] += sku.unit_capacity * units[sku_id]
    over = {v: u for v, u in used.items() if u > storage.vertex_capacity}
    if over:
        raise ValueError(
            f"The initial counts do not fit vertex_capacity={storage.vertex_capacity}: "
            f"used capacity by vertex {over}. Lower num_skus or initial_units, or raise "
            "vertex_capacity."
        )
    return skus, {v: storage.vertex_capacity for v in storage_vertices}, counts


def _demand_blocks(
    sku_ids: list[SkuId],
    graph: WarehouseGraph,
    storage_vertices: list[VertexId],
    demand: DemandSettings | None,
) -> dict[SkuId, VertexId]:
    """The storage vertex of every SKU under "demand_blocks" (see `StorageSettings`).

    `statistics.mean` sums the distances exactly and rounds only the result, so two vertices
    whose distances have the same mean always tie and fall back to the vertex number."""
    delivery = [v for v, vertex in graph.vertices.items() if vertex.role.delivery]
    nearest_first = sorted(
        storage_vertices,
        key=lambda v: (statistics.mean(graph.distance(v, g) for g in delivery), v),
    )
    ranked = popularity_ranking(
        sku_ids,
        0,
        None if demand is None else demand.rotation_period,
        None if demand is None else demand.rotation_shift,
    )
    block = -(-len(ranked) // len(nearest_first))  # the ceiling, in integers
    return {sku_id: nearest_first[rank // block] for rank, sku_id in enumerate(ranked)}


def _sized_units(
    sku_ids: list[SkuId], floor: int, demand: DemandSettings | None, sizing: StockSizing
) -> dict[SkuId, int]:
    """The starting units of every SKU under "demand_sized_blocks" (see `StorageSettings`).
    Without a demand section every SKU weighs the same, as the uniform draw has no ranking."""
    ranked = popularity_ranking(
        sku_ids,
        0,
        None if demand is None else demand.rotation_period,
        None if demand is None else demand.rotation_shift,
    )
    weights = zipf_weights(len(ranked), 0.0 if demand is None else demand.skew)
    return {
        sku_id: max(
            floor,
            math.ceil(
                (1 + sizing.margin)
                * sizing.arrival_rate
                * sizing.epoch_length
                * sizing.mean_units
                * weights[rank]
            ),
        )
        for rank, sku_id in enumerate(ranked)
    }


def stock_sizing(spec: TrainingSpec) -> StockSizing | None:
    """What the starting stock of `spec` is sized by, or None when it is not sized."""
    if spec.storage.initial_placement != "demand_sized_blocks":
        return None
    # _arrival_settings guarantees both sections for this placement.
    assert spec.arrival is not None and spec.storage_rule is not None
    units = spec.arrival.units_per_line
    mean_units = 1.0 if units is None else (units[0] + units[1]) / 2
    return StockSizing(
        spec.arrival_rate, spec.storage_rule.epoch_length, spec.arrival.margin, mean_units
    )


def training_storage_mode(spec: TrainingSpec) -> StorageMode:
    """The storage rule a run trains under. "fixed_only" trains under F_fix and "matched" under
    the storage rule used at evaluation (sec:method:training)."""
    return "fixed" if spec.regime == "fixed_only" else spec.evaluation_storage_mode


def check_storage_rule(spec: TrainingSpec, storage_mode: StorageMode) -> None:
    """Raises a ValueError naming what `spec` lacks to play `storage_mode`. In the template
    model F_fix needs nothing and the adaptive rules need the `storage_rule` section with nu.
    In the arrival model every rule needs the section, for Delta, and none takes nu. The
    congestion rule needs beta in it as well."""
    rule = spec.storage_rule
    if spec.arrival is None:
        if storage_mode == "fixed":
            return
        if rule is None or rule.reassignment_cap is None:
            raise ValueError(
                f"storage mode {storage_mode!r} needs a storage_rule section with epoch_length "
                "and reassignment_cap, and there is none."
            )
    elif rule is None:
        raise ValueError(
            "the arrival model needs a storage_rule section with epoch_length under every "
            "storage mode, and there is none."
        )
    if storage_mode == "congestion" and rule.congestion_weight is None:
        raise ValueError(
            "storage mode 'congestion' needs congestion_weight in the storage_rule section, "
            "and there is none."
        )


def build_experiment_config(
    spec: TrainingSpec, seed: int | None = None, storage_mode: StorageMode | None = None
) -> ExperimentConfig:
    """The configuration a run trains under, or plays under `storage_mode` when that is given.
    In the template model an adaptive mode takes Delta, nu and, for the congestion rule, beta
    from the spec's `storage_rule` section, and F_fix ignores the section. In the arrival model
    every mode takes Delta from it, and the stock settings from the `arrival` section."""
    experiment = spec.experiment
    demand = spec.demand
    arrival = spec.arrival
    if storage_mode is None:
        storage_mode = training_storage_mode(spec)
    check_storage_rule(spec, storage_mode)
    rule = None if storage_mode == "fixed" and arrival is None else spec.storage_rule
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
        move_cost=experiment.move_cost,
        congestion_reward_weight=experiment.congestion_reward_weight,
        demand_skew=None if demand is None else demand.skew,
        demand_rotation_period=None if demand is None else demand.rotation_period,
        demand_rotation_shift=None if demand is None else demand.rotation_shift,
        storage_epoch_length=None if rule is None else rule.epoch_length,
        reassignment_cap=None if rule is None else rule.reassignment_cap,
        congestion_weight=(
            rule.congestion_weight if rule is not None and storage_mode == "congestion" else None
        ),
        stock_model="template" if arrival is None else "arrival",
        stock_margin=None if arrival is None else arrival.margin,
        stock_floor=None if arrival is None else spec.storage.initial_units,
        stock_receiving=None if arrival is None else arrival.receiving,
        lines_per_order=None if arrival is None else arrival.lines_per_order,
        units_per_line=None if arrival is None else arrival.units_per_line,
        cold_stock_top_k=None if arrival is None else arrival.cold_stock_top_k,
    )


def build_study(spec: TrainingSpec) -> Study:
    instance = build_instance(spec)
    # x_0 depends on the instance, the storage settings and the demand only, so every storage
    # rule, in training and at evaluation, starts from the same stock (sec:impl:instances).
    skus, capacities, counts = build_storage(
        spec.storage, instance.graph, spec.demand, stock_sizing(spec)
    )
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
            training_seed=config.seed,
            train_seed_count=spec.train_seed_count,
            validation_seed_count=spec.validation_seed_count,
            test_seed_count=spec.test_seed_count,
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
