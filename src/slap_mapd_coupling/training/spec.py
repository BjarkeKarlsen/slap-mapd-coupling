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

import functools
import math
import random
import statistics
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
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
    StockPutaway,
    StockRule,
    StorageMode,
)
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.multi_agent_env import popularity_ranking, zipf_weights
from slap_mapd_coupling.instances.generator import (
    GeneratedInstance,
    GeneratorParams,
    fleet_defaults,
    generate_instance,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.models.base_model import PolicyValueHeadConfig
from slap_mapd_coupling.models.gnn_encoder import GNNEncoderConfig
from slap_mapd_coupling.training.arm import POOLING, Arm, build_arm
from slap_mapd_coupling.training.config import (
    PPOHyperparameters,
    SeedSplitConfig,
    TrainingRegime,
    build_ppo_config,
    split_seeds,
)
from slap_mapd_coupling.training.trainer import CheckpointSettings


class StorageSettings(BaseModel):
    """The SKUs and where they start.

    There are `num_skus` SKUs, named by `sku_names` when it is given and sku_000, sku_001 and so
    on otherwise. Under the refill rule `start_fill` may be given instead of `num_skus`: the
    number of SKUs is then derived, the fewest whose starting stock fills that share of the
    shelves' capacity, so every warehouse starts equally full whatever its number of storage
    vertices (`derived_num_skus`), rounded to the nearest multiple of 1 / `rotation_share` when
    the demand rotates by a share. `sku_names` then names as many of the first SKUs as it has. The popularity ranking of `DemandSettings` starts from the order of that list,
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
      expected units ordered over one arrival period, for the arrival stock model. Under the
      top-up rule the SKU at rank r starts with max(initial_units, ceil((1 + epsilon) *
      lambda_task * Delta * q_bar * w_r)) units, w_r being its popularity at t = 0 and q_bar the
      mean of `units_per_line` (1 without it), so every SKU holds at least `initial_units`.
      Under the refill rule it starts with S_k = Q + the service-level quantile of its units
      ordered in one period at its t = 0 popularity, the quantile taken from a simulation of the
      order process (`period_demand_quantiles`), and takes no `initial_units`. It needs
      the `arrival` section and the `storage_rule` section for Delta.
    - "random_slots": the start of D22. The units of "demand_sized_blocks", each SKU on its own
      storage vertex drawn at random from `seed`, so with more storage vertices than SKUs some
      start empty and the start carries no demand information. Needs at least as many storage
      vertices as SKUs.

    Under the refill rule `vertex_capacity` may be left out, and is then derived: the most room
    the starting stock of the SKUs of one storage vertex takes, sum of S_k * b_k, rounded up.
    A SKU's stock plus its stock on the way stays at most S_k, so F_fix, which puts refills on
    the home shelf first, has room for them there.

    A new task picks a SKU in stock, then a vertex holding it, so more SKUs mean more distinct
    pickups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_skus: PositiveInt | None = None
    # The share of the shelves' capacity the starting stock fills, refill rule only. Set it or
    # num_skus, not both.
    start_fill: float | None = Field(default=None, gt=0.0, le=1.0)
    sku_names: tuple[str, ...] | None = None
    unit_capacities: tuple[PositiveFloat, ...] = Field(min_length=1)
    # None derives it from the starting stock, refill rule only.
    vertex_capacity: PositiveFloat | None = None
    # The units every SKU starts with, and under the top-up rule its floor. The refill rule
    # sizes the starting stock by the service level and takes none.
    initial_units: PositiveInt | None = None
    initial_placement: Literal[
        "by_number", "demand_blocks", "demand_sized_blocks", "random_slots"
    ] = "by_number"
    # The draw of "random_slots".
    seed: int = 0

    @property
    def sized(self) -> bool:
        """The starting units are sized by the order process, as "demand_sized_blocks" sizes them."""
        return self.initial_placement in ("demand_sized_blocks", "random_slots")

    @model_validator(mode="after")
    def _one_name_per_sku(self) -> "StorageSettings":
        if (self.num_skus is None) == (self.start_fill is None):
            raise ValueError("Set exactly one of storage.num_skus and storage.start_fill.")
        if self.sku_names is None:
            return self
        if self.num_skus is not None and len(self.sku_names) != self.num_skus:
            raise ValueError(
                f"sku_names has {len(self.sku_names)} names for num_skus={self.num_skus}."
            )
        if len(set(self.sku_names)) != len(self.sku_names) or not all(self.sku_names):
            raise ValueError("sku_names must be distinct and non-empty.")
        return self

    def sku_ids(self, count: int | None = None) -> list[SkuId]:
        """The SKU ids in list order, the order the popularity ranking starts from. `count` is
        the number of SKUs when it is derived from `start_fill`, and the first ones take the
        names of `sku_names`."""
        count = self.num_skus if count is None else count
        assert count is not None
        names = list(self.sku_names or ())[:count]
        return names + [f"sku_{i:03d}" for i in range(len(names), count)]


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
    # The shift as a share of the SKUs instead of a number of places, rounded and at least one,
    # so the ranking turns once in the same time whatever the number of SKUs.
    rotation_share: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _rotation_parameters(self) -> "DemandSettings":
        """A period and a shift, or a share, only mean something together."""
        shifts = sum(shift is not None for shift in (self.rotation_shift, self.rotation_share))
        if shifts > 1:
            raise ValueError("Set rotation_shift or rotation_share, not both.")
        if (self.rotation_period is None) != (shifts == 0):
            raise ValueError("Set rotation_period with rotation_shift or rotation_share, or none.")
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
    which shelves it goes to. How much arrives is set by `rule`:

    - "topup": each SKU is filled up to max(initial_units, ceil((1 + margin) * its orders in the
      last period)), so `initial_units` of the `storage` section is the floor. `margin` is
      epsilon.
    - "refill": stock is replaced a refill at a time. A SKU gets one refill of `refill_units`
      (Q) units for every Q units ordered since its last refill, each refill one put-away to one
      shelf. It starts with Q plus the `service_level` quantile of a period's orders (see
      `StorageSettings`), the (R, nQ) policy of inventory control.

    None of them has a default, because all are study parameters. `receiving: receiving` needs
    `num_receiving_vertices` above 0 in the `instance` section. `putaway: robots` makes the
    robots carry the stock to its shelves (sec:pf:storage). `shipping` carries every order
    from its workstation to a dock once its last pick is in, and `dock_choice: random` draws the
    dock of every put-away and ship task at random (step 12, D21). Both need `receiving:
    receiving` and `putaway: robots`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: StockRule = "topup"
    margin: NonNegativeFloat | None = None
    refill_units: PositiveInt | None = None
    service_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    receiving: ReceivingVertices
    putaway: StockPutaway = "instant"
    shipping: bool = False
    dock_choice: Literal["nearest", "random"] = "nearest"
    # Order lines (tab:evalparams). None is one line of one unit per order.
    lines_per_order: LinesPerOrder | None = None
    units_per_line: tuple[PositiveInt, PositiveInt] | None = None
    # The SKUs that count as popular in the cold-stock measure. None leaves it off.
    cold_stock_top_k: PositiveInt | None = None
    # [A75]: the rank the safety stock is sized at, each SKU's rank at t = 0 ("start_rank") or
    # the most popular rank for every SKU ("top_rank"), since the drift takes every SKU there.
    safety_stock: Literal["start_rank", "top_rank"] = "start_rank"
    # [A76] (D29): each SKU belongs to one delivery vertex, see ExperimentConfig.sku_sections.
    sku_sections: bool = False

    @model_validator(mode="after")
    def _rule_settings(self) -> "ArrivalSettings":
        """The top-up rule takes the margin, the refill rule Q and the service level."""
        needed, unused = (
            (("margin",), ("refill_units", "service_level"))
            if self.rule == "topup"
            else (("refill_units", "service_level"), ("margin",))
        )
        for name in needed:
            if getattr(self, name) is None:
                raise ValueError(f"arrival.rule {self.rule!r} needs arrival.{name}.")
        for name in unused:
            if getattr(self, name) is not None:
                raise ValueError(f"arrival.{name} does not apply to arrival.rule {self.rule!r}.")
        return self


class StockSizing(NamedTuple):
    """What "demand_sized_blocks" sizes the starting stock by: lambda_task, Delta, epsilon and
    q_bar, the mean units per line, under the top-up rule. Under the refill rule Q, the service
    level and the order lines, which the simulated order process draws by, with margin None."""

    arrival_rate: float
    epoch_length: int
    margin: float | None
    mean_units: float = 1.0
    refill_units: int | None = None
    service_level: float | None = None
    lines_per_order: LinesPerOrder | None = None
    units_per_line: tuple[int, int] | None = None
    # [A75]: "top_rank" sizes every SKU's safety stock at the most popular rank. [A76]: the
    # number of SKU sections ([A77]), None without sections.
    safety_stock: str = "start_rank"
    sections: int | None = None


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
    # D26: an idle robot in the way gets the nearest free parking leaf as its target.
    idle_parking: bool = False
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
    # T_warm, the warm-up left out of the measures ([A66]), chosen by Welch's method. Training
    # episodes run the whole horizon either way. Whether they get the long horizon is step 10's.
    warmup: NonNegativeInt = 0
    # The storage rule a policy is evaluated under, which "matched" training uses too.
    evaluation_storage_mode: StorageMode = "fixed"
    experiment: ExperimentSettings
    # The GeneratorParams without wait_cost and num_endpoints, which follow the experiment.
    # Empty, or left out, builds the warehouse of the fleet size, `fleet_defaults(num_agents)`.
    instance: dict[str, Any] = {}
    # D27: every episode seed draws its own warehouse and start stock, the generator seed and
    # storage.seed set to the episode seed (training/warehouses.py). The window tables are
    # stored under warehouse_store, ~/thesis_runs/warehouses when it is None.
    warehouse_per_seed: bool = False
    warehouse_store: str | None = None
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

    def instance_settings(self) -> dict[str, Any]:
        """The GeneratorParams keys without wait_cost: the `instance` section as written, or
        the fleet's warehouse when it is empty."""
        if not self.instance:
            return fleet_defaults(self.experiment.num_agents)
        return {**self.instance, "num_endpoints": self.experiment.num_agents}

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
            if self.arrival.receiving == "receiving" and not self.instance_settings().get(
                "num_receiving_vertices", 0
            ):
                raise ValueError(
                    "arrival.receiving 'receiving' needs num_receiving_vertices above 0 in the "
                    "instance section."
                )
        elif self.storage.sized:
            raise ValueError(
                f"initial_placement '{self.storage.initial_placement}' needs the arrival section "
                "for its margin."
            )
        refill = self.arrival is not None and self.arrival.rule == "refill"
        if refill:
            if not self.storage.sized:
                raise ValueError(
                    "arrival.rule 'refill' sizes the starting stock by the service level, so it "
                    "needs initial_placement 'demand_sized_blocks' or 'random_slots'."
                )
            if self.storage.initial_units is not None:
                raise ValueError(
                    "arrival.rule 'refill' sizes the starting stock by the service level and "
                    "has no floor, so storage.initial_units does not apply."
                )
            if self.storage.start_fill is not None and self.storage.vertex_capacity is not None:
                raise ValueError(
                    "storage.start_fill derives the number of SKUs at the derived shelf "
                    "capacity, so it takes no storage.vertex_capacity."
                )
        else:
            if self.storage.start_fill is not None:
                raise ValueError("storage.start_fill applies to arrival.rule 'refill' only.")
            if self.storage.initial_units is None:
                raise ValueError("storage.initial_units is needed unless arrival.rule is 'refill'.")
            if self.storage.vertex_capacity is None:
                raise ValueError(
                    "storage.vertex_capacity can only be derived under arrival.rule 'refill'."
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
    """Everything `build_ppo_config` and an evaluation need, built from a TrainingSpec. With a
    warehouse per seed, `pool` holds them per seed, `arm.encoding` is the pool's cap and the
    instance, the stock and the zones are those of the first training seed."""

    instance: GeneratedInstance
    skus: dict[SkuId, SkuType]
    capacities: dict[VertexId, float]
    counts: dict[SkuId, dict[VertexId, int]]
    config: ExperimentConfig
    arm: Arm
    gnn_encoder: GNNEncoderConfig
    head: PolicyValueHeadConfig
    seed_split: SeedSplitConfig
    pool: Any = None  # training/warehouses.py::WarehousePool, with a warehouse per seed


def seed_spec(spec: TrainingSpec, seed: int) -> TrainingSpec:
    """`spec` with the warehouse and the start stock of episode seed `seed` (D27)."""
    instance = dict(spec.instance_settings())
    instance.pop("num_endpoints", None)
    instance["seed"] = seed
    storage = spec.storage.model_copy(update={"seed": seed})
    return spec.model_copy(update={"instance": instance, "storage": storage})


def build_seed_warehouse(spec: TrainingSpec, seed: int) -> Any:
    """The warehouse of episode seed `seed`: its instance, start stock and zones, with the
    padding cap of its own windows (training/warehouses.py)."""
    from slap_mapd_coupling.training.warehouses import Warehouse

    seeded = seed_spec(spec, seed)
    instance = build_instance(seeded)
    skus, capacities, counts = build_storage(
        seeded.storage, instance.graph, seeded.demand, stock_sizing(seeded)
    )
    max_messages = spec.arm.max_messages or max(spec.experiment.num_agents - 1, 1)
    arm = build_arm(
        instance,
        spec.controller,
        spec.experiment.observation_depth,
        num_zones=spec.arm.num_zones,
        max_messages=max_messages,
    )
    return Warehouse(seed, instance, skus, capacities, counts, arm.zones, arm.encoding)


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
    warmup: int | None = None,
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
        "warmup": warmup,
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


def instance_params(spec: TrainingSpec) -> GeneratorParams:
    """The generator settings, from the `instance` section or the fleet size."""
    return GeneratorParams(**spec.instance_settings(), wait_cost=spec.experiment.wait_cost)


def build_instance(spec: TrainingSpec) -> GeneratedInstance:
    """The generated instance, which keeps the aisle index the section arm needs."""
    params = instance_params(spec)
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
    capacity constraint. With `start_fill` the number of SKUs is derived first."""
    count = storage.num_skus
    if count is None:
        count = derived_num_skus(storage, graph, demand, sizing)
        if demand is not None and demand.rotation_share is not None:
            # Rounded to the nearest whole number of shifts, so the ranking turns once in exactly
            # 1 / share periods (D20). The fill then lands near start_fill, not at it.
            multiple = round(1 / demand.rotation_share)
            count = max(multiple, round(count / multiple) * multiple)
    skus, capacity, counts, _ = _stock_for(storage, graph, demand, sizing, count)
    return skus, capacity, counts


def derived_num_skus(
    storage: StorageSettings,
    graph: WarehouseGraph,
    demand: DemandSettings | None,
    sizing: StockSizing | None,
) -> int:
    """The fewest SKUs whose starting stock fills `start_fill` of the shelves' capacity, by
    bisection. The fill rises with the number of SKUs but not strictly, since the derived
    capacity follows the fullest shelf, so the count found is the first in the bisection with
    fill at least `start_fill`, and the fill it gives is reported by `starting_fill`."""
    assert storage.start_fill is not None
    slots = sum(1 for vertex in graph.vertices.values() if vertex.role.storage)
    low, high = 1, 1
    while starting_fill(storage, graph, demand, sizing, high) < storage.start_fill:
        low, high = high + 1, high * 2
        if high > 64 * slots:
            raise ValueError(f"no number of SKUs fills {storage.start_fill} of the shelves.")
    while low < high:
        middle = (low + high) // 2
        if starting_fill(storage, graph, demand, sizing, middle) >= storage.start_fill:
            high = middle
        else:
            low = middle + 1
    return low


def starting_fill(
    storage: StorageSettings,
    graph: WarehouseGraph,
    demand: DemandSettings | None,
    sizing: StockSizing | None,
    count: int,
) -> float:
    """The share of the shelves' capacity the starting stock of `count` SKUs fills."""
    _, capacity, _, used = _stock_for(storage, graph, demand, sizing, count)
    return sum(used.values()) / sum(capacity.values())


def _stock_for(
    storage: StorageSettings,
    graph: WarehouseGraph,
    demand: DemandSettings | None,
    sizing: StockSizing | None,
    count: int | None,
) -> tuple[
    dict[SkuId, SkuType],
    dict[VertexId, float],
    dict[SkuId, dict[VertexId, int]],
    dict[VertexId, float],
]:
    """`build_storage` for `count` SKUs, with the room the starting stock takes on each
    storage vertex."""
    storage_vertices = sorted(v for v, vertex in graph.vertices.items() if vertex.role.storage)
    skus: dict[SkuId, SkuType] = {}
    for i, sku_id in enumerate(storage.sku_ids(count)):
        unit_capacity = storage.unit_capacities[i % len(storage.unit_capacities)]
        skus[sku_id] = SkuType(sku_id=sku_id, unit_capacity=unit_capacity)
    units = dict.fromkeys(skus, storage.initial_units or 0)
    if storage.initial_placement in ("demand_blocks", "demand_sized_blocks"):
        home = _demand_blocks(list(skus), graph, storage_vertices, demand)
    elif storage.initial_placement == "random_slots":
        if len(skus) > len(storage_vertices):
            raise ValueError(
                f"initial_placement 'random_slots' puts {len(skus)} SKUs on their own storage "
                f"vertex, but there are {len(storage_vertices)}."
            )
        drawn = random.Random(f"{storage.seed}:start").sample(storage_vertices, len(skus))
        home = dict(zip(skus, drawn))
    else:
        home = {
            sku_id: storage_vertices[i % len(storage_vertices)] for i, sku_id in enumerate(skus)
        }
    if storage.sized:
        if sizing is None:
            raise ValueError(
                f"initial_placement '{storage.initial_placement}' needs the stock sizing."
            )
        if sizing.refill_units is None:
            assert storage.initial_units is not None  # TrainingSpec checks the top-up rule's
            units = _sized_units(list(skus), storage.initial_units, demand, sizing)
        else:
            units = refill_starting_stock(list(skus), demand, sizing)
    counts: dict[SkuId, dict[VertexId, int]] = {}
    used = dict.fromkeys(storage_vertices, 0.0)
    for sku_id, sku in skus.items():
        counts[sku_id] = {home[sku_id]: units[sku_id]}
        used[home[sku_id]] += sku.unit_capacity * units[sku_id]
    capacity = storage.vertex_capacity
    if capacity is None:
        # Derived: the room the starting stock of the fullest vertex takes, rounded up.
        capacity = float(math.ceil(max(used.values()) - 1e-9))
    over = {v: u for v, u in used.items() if u > capacity}
    if over:
        raise ValueError(
            f"The initial counts do not fit vertex_capacity={capacity}: "
            f"used capacity by vertex {over}. Lower num_skus or initial_units, or raise "
            "vertex_capacity."
        )
    return skus, dict.fromkeys(storage_vertices, capacity), counts, used


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
        None,  # at t = 0 no rotation has happened yet
        None,
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
        None,  # at t = 0 no rotation has happened yet
        None,
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


# The periods of the order process simulated for the service-level quantile, and the seed of
# that simulation. Fixed, so a spec always gives the same starting stock, and apart from every
# run's seeds, so the starting stock is the same under every run seed and storage rule.
QUANTILE_PERIODS = 20_000
QUANTILE_SEED = 7
_QUANTILE_BLOCK = 256


@functools.lru_cache(maxsize=64)
def period_demand_quantiles(
    num_skus: int,
    skew: float,
    arrival_rate: float,
    epoch_length: int,
    service_level: float,
    lines_per_order: LinesPerOrder | None = None,
    units_per_line: tuple[int, int] | None = None,
    periods: int = QUANTILE_PERIODS,
    seed: int = QUANTILE_SEED,
) -> list[int]:
    """The `service_level` quantile of the units ordered in one period of `epoch_length` steps,
    per popularity rank, rank 0 first, from a simulation of the order process of
    environment/multi_agent_env.py::_draw_order_lines with the ranking held still.

    Orders arrive as Poisson(arrival_rate / mean lines) per step, so Poisson(epoch_length *
    arrival_rate / mean lines) per period. Each has 1 + Poisson(mean - 1) lines of distinct
    ranks drawn by the Zipf weights without replacement, and each line a quantity uniform on
    `units_per_line`. The quantile is the smallest count whose share of periods at or below it
    is at least the service level, so no distribution is assumed."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(zipf_weights(num_skus, skew))
    mean_lines = 1.0 if lines_per_order is None else lines_per_order.mean
    low, high = units_per_line or (1, 1)
    orders = rng.poisson(epoch_length * arrival_rate / mean_lines, size=periods)
    period = np.repeat(np.arange(periods), orders)
    if lines_per_order is None:
        ranks = rng.choice(num_skus, size=period.size, p=weights)
        quantities = rng.integers(low, high + 1, size=period.size)
    else:
        lines = np.minimum(1 + rng.poisson(mean_lines - 1.0, size=period.size), num_skus)
        drawn = [rng.choice(num_skus, size=count, replace=False, p=weights) for count in lines]
        ranks = np.concatenate(drawn) if drawn else np.zeros(0, dtype=np.int64)
        period = np.repeat(period, lines)
        quantities = rng.integers(low, high + 1, size=ranks.size)
    # The units per period and rank, a block of ranks at a time, so a warehouse with thousands
    # of SKUs never holds the whole table.
    quantiles: list[int] = []
    for start in range(0, num_skus, _QUANTILE_BLOCK):
        stop = min(start + _QUANTILE_BLOCK, num_skus)
        inside = (ranks >= start) & (ranks < stop)
        units = np.zeros((periods, stop - start), dtype=np.int32)
        np.add.at(units, (period[inside], ranks[inside] - start), quantities[inside])
        block = np.quantile(units, service_level, axis=0, method="inverted_cdf")
        quantiles.extend(int(q) for q in block)
    return quantiles


def refill_starting_stock(
    sku_ids: list[SkuId], demand: DemandSettings | None, sizing: StockSizing
) -> dict[SkuId, int]:
    """S_k = Q + the service-level quantile of a period's units of k at its t = 0 popularity,
    the starting units of every SKU under the refill rule (see `StorageSettings`). Without a
    demand section every SKU weighs the same."""
    assert sizing.refill_units is not None and sizing.service_level is not None
    ranked = popularity_ranking(
        sku_ids,
        0,
        None,  # at t = 0 no rotation has happened yet
        None,
    )
    quantiles = period_demand_quantiles(
        len(ranked),
        0.0 if demand is None else demand.skew,
        sizing.arrival_rate,
        sizing.epoch_length,
        sizing.service_level,
        sizing.lines_per_order,
        sizing.units_per_line,
    )
    if sizing.safety_stock == "top_rank":
        # [A75]: every SKU gets the safety stock of the most popular rank. With sections
        # ([A76]) the most popular SKU of a section takes the largest share of that section's
        # lines when the rest of its section is the least popular SKUs.
        if sizing.sections:
            top = section_top_quantile(
                len(ranked), sizing, 0.0 if demand is None else demand.skew
            )
        else:
            top = quantiles[0]
        return {sku_id: sizing.refill_units + top for sku_id in ranked}
    return {sku_id: sizing.refill_units + quantiles[rank] for rank, sku_id in enumerate(ranked)}


def section_top_quantile(num_skus: int, sizing: StockSizing, skew: float) -> int:
    """The service-level quantile of one period's units of the most popular SKU with SKU
    sections ([A76]), simulated like `period_demand_quantiles`. Orders reach one section with
    Poisson(epoch_length * arrival_rate / (mean lines * sections)) per period, and draw their
    lines from its SKUs without replacement. The section is the worst case, rank 0 with the
    least popular SKUs, the largest share any SKU can take as the ranking drifts."""
    assert sizing.sections and sizing.service_level is not None
    size = max(1, num_skus // sizing.sections)
    weights = np.asarray(zipf_weights(num_skus, skew))
    own = weights[:1]
    if size > 1:
        own = np.concatenate([own, weights[num_skus - (size - 1) :]])
    own = own / own.sum()
    rng = np.random.default_rng(QUANTILE_SEED)
    mean_lines = 1.0 if sizing.lines_per_order is None else sizing.lines_per_order.mean
    low, high = sizing.units_per_line or (1, 1)
    orders = rng.poisson(
        sizing.epoch_length * sizing.arrival_rate / (mean_lines * sizing.sections),
        size=QUANTILE_PERIODS,
    )
    units = np.zeros(QUANTILE_PERIODS, dtype=np.int64)
    for period, count in enumerate(orders):
        for _ in range(count):
            lines = 1 if sizing.lines_per_order is None else 1 + rng.poisson(mean_lines - 1.0)
            picked = rng.choice(size, size=min(lines, size), replace=False, p=own)
            if 0 in picked:
                units[period] += rng.integers(low, high + 1)
    return int(np.quantile(units, sizing.service_level, method="inverted_cdf"))


def stock_sizing(spec: TrainingSpec) -> StockSizing | None:
    """What the starting stock of `spec` is sized by, or None when it is not sized."""
    if not spec.storage.sized:
        return None
    # _arrival_settings guarantees both sections for this placement.
    assert spec.arrival is not None and spec.storage_rule is not None
    units = spec.arrival.units_per_line
    mean_units = 1.0 if units is None else (units[0] + units[1]) / 2
    return StockSizing(
        spec.arrival_rate,
        spec.storage_rule.epoch_length,
        spec.arrival.margin,
        mean_units,
        spec.arrival.refill_units,
        spec.arrival.service_level,
        spec.arrival.lines_per_order,
        units,
        spec.arrival.safety_stock,
        _num_sections(spec) if spec.arrival.sku_sections else None,
    )


def _num_sections(spec: TrainingSpec) -> int:
    """The SKU sections of `spec` ([A76]), its workstations over the workstations that serve
    each section ([A77])."""
    per_section = spec.instance.get("workstations_per_section", 1)
    return spec.instance["num_delivery_vertices"] // per_section


def training_storage_mode(spec: TrainingSpec) -> StorageMode:
    """The storage rule a run trains under. "fixed_only" trains under F_fix and "matched" under
    the storage rule used at evaluation (sec:method:training)."""
    return "fixed" if spec.regime == "fixed_only" else spec.evaluation_storage_mode


def check_storage_rule(spec: TrainingSpec, storage_mode: StorageMode) -> None:
    """Raises a ValueError naming what `spec` lacks to play `storage_mode`. In the template
    model F_fix needs nothing and the adaptive rules need the `storage_rule` section with nu.
    In the arrival model every rule needs the section, for Delta, and none takes nu. The
    congestion rule needs beta in it as well. The random rule exists in the arrival model only."""
    rule = spec.storage_rule
    if spec.arrival is None:
        if storage_mode == "random":
            raise ValueError("storage mode 'random' needs the arrival model, and there is none.")
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
        idle_parking=experiment.idle_parking,
        communication=spec.communication,
        num_agents=experiment.num_agents,
        arrival_rate=spec.arrival_rate,
        keep_up_threshold=experiment.keep_up_threshold,
        seed=experiment.seed if seed is None else seed,
        horizon=spec.horizon,
        warmup=spec.warmup,
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
        demand_rotation_share=None if demand is None else demand.rotation_share,
        storage_epoch_length=None if rule is None else rule.epoch_length,
        reassignment_cap=None if rule is None else rule.reassignment_cap,
        congestion_weight=(
            rule.congestion_weight if rule is not None and storage_mode == "congestion" else None
        ),
        stock_model="template" if arrival is None else "arrival",
        stock_rule="topup" if arrival is None else arrival.rule,
        stock_margin=None if arrival is None else arrival.margin,
        stock_floor=(
            None if arrival is None or arrival.rule != "topup" else spec.storage.initial_units
        ),
        refill_units=None if arrival is None else arrival.refill_units,
        service_level=None if arrival is None else arrival.service_level,
        stock_receiving=None if arrival is None else arrival.receiving,
        stock_putaway="instant" if arrival is None else arrival.putaway,
        lines_per_order=None if arrival is None else arrival.lines_per_order,
        units_per_line=None if arrival is None else arrival.units_per_line,
        cold_stock_top_k=None if arrival is None else arrival.cold_stock_top_k,
        shipping=arrival is not None and arrival.shipping,
        sku_sections=arrival is not None and arrival.sku_sections,
        dock_choice="nearest" if arrival is None else arrival.dock_choice,
        # A random start gives F_fix class-based homes instead ([A69]).
        fixed_homes="popularity" if spec.storage.initial_placement == "random_slots" else "start",
    )


def build_study(spec: TrainingSpec, prepare: bool = True) -> Study:
    """The study of `spec`. With a warehouse per seed, `prepare` stores the window tables of
    every seed of the split first, which a learned arm needs and a planner run does not."""
    if spec.warehouse_per_seed:
        return _build_pool_study(spec, prepare)
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


def _build_pool_study(spec: TrainingSpec, prepare: bool) -> Study:
    """build_study with a warehouse per seed (D27). With `prepare` every seed of the split is
    prepared first, so the cap every warehouse is padded to is known before training or
    evaluation starts. Without it the cap is the first training seed's."""
    from slap_mapd_coupling.training.warehouses import WarehousePool

    store = None if spec.warehouse_store is None else Path(spec.warehouse_store).expanduser()
    pool = WarehousePool(spec, store)
    config = build_experiment_config(spec)
    split = SeedSplitConfig(
        training_seed=config.seed,
        train_seed_count=spec.train_seed_count,
        validation_seed_count=spec.validation_seed_count,
        test_seed_count=spec.test_seed_count,
    )
    seeds = split_seeds(split)
    if prepare:
        pool.prepare([*seeds.train, *seeds.validation, *seeds.test])
    first = pool.warehouse(seeds.train[0])
    arm = Arm(
        controller=spec.controller,
        pooling=POOLING[spec.controller],
        zones=first.zones,
        encoding=pool.encoding() if prepare else first.encoding,
    )
    return Study(
        instance=first.instance,
        skus=first.skus,
        capacities=first.capacities,
        counts=first.counts,
        config=config,
        arm=arm,
        gnn_encoder=GNNEncoderConfig(
            num_rounds=spec.model.num_rounds,
            hidden_width=spec.model.hidden_width,
            communication=config.communication,
            pooling=arm.pooling,
        ),
        head=PolicyValueHeadConfig(hidden_width=spec.model.head_hidden_width),
        seed_split=split,
        pool=pool,
    )


def build_study_ppo_config(
    spec: TrainingSpec,
    study: Study,
    *,
    num_env_runners: int = 0,
    use_gpu: bool = False,
    batch_on_cpu: bool = False,
    value_chunk_rows: int | None = None,
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
        pool=study.pool,
        num_env_runners=num_env_runners,
        use_gpu=use_gpu,
        batch_on_cpu=batch_on_cpu,
        value_chunk_rows=value_chunk_rows,
    )
