"""ExperimentConfig: the independent variables of the study (sec:pd-aim-and-scope)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from slap_mapd_coupling.planners.registry import PlannerName

StorageMode = Literal["fixed", "demand", "congestion"]  # F_fix / F_dem / F_cng
# How stock behaves. "template": stock never runs out and an adaptive rule relocates it every
# Delta steps. "arrival": an order takes a unit, stock arrives every Delta steps, and the storage
# rule decides where the arriving stock goes (thesis wording pending).
StockModel = Literal["template", "arrival"]
# Where arriving stock enters the warehouse. "delivery": the delivery vertices, one I/O side.
ReceivingVertices = Literal["delivery"]
# "full_view" is the full-view setting (sec:method:controllers), the decentralised router with
# every movable vertex in view and the pooled readout. It is not a controller in Pi.
ControllerArchitecture = Literal["centralised", "section", "decentralised", "full_view"]


class ExperimentConfig(BaseModel):
    """The five independent variables of the study, one instance per
    experimental-grid cell: storage mode, controller architecture,
    congestion sensitivity, communication, and system load (num_agents,
    arrival_rate) -- plus the seed and horizon needed to run one concrete
    episode. A future "sweep" command constructs many of these from one
    grid-spec YAML (list-valued fields product-expanded upstream of this
    model; this model is one concrete run, not a grid).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_mode: StorageMode
    # Exactly one of controller and planner is set. A planner is a
    # non-learned reference outside the three controllers in Pi.
    controller: ControllerArchitecture | None = None
    planner: PlannerName | None = None
    congestion_sensitive: bool
    communication: bool  # only meaningful if controller == "decentralised"
    num_agents: PositiveInt  # m
    arrival_rate: PositiveFloat  # lambda_task
    seed: int
    horizon: PositiveInt  # T, the evaluation horizon (tab:evalparams)

    wait_cost: NonNegativeFloat  # c_wait
    storage_epoch_length: PositiveInt | None = None  # Delta; None means F_fix (Delta = infinity)
    congestion_weight: NonNegativeFloat | None = None  # beta >= 0, congestion-aware score
    reassignment_cap: PositiveInt | None = None  # nu, max relocated units/epoch
    # d_obs, depth of the field of view. It also bounds the communication
    # graph and the congestion feature, so there are no separate radii
    # (#88). Required for every controller: the field of view, and the
    # congestion feature it bounds, are read by all three architectures
    # (sec:pf:env, sec:pf:controllers), not only the decentralised one.
    observation_depth: PositiveInt
    # f_up and chi_max in the keep-up check (tab:evalparams). chi_max is TBD, set from the
    # Token Passing load sweep. RunMetrics.keeps_up stays unset unless both are given.
    keep_up_threshold: PositiveFloat | None = Field(default=None, le=1.0)
    backlog_slope_max: float | None = None

    # Reward and learning objective (tab:rlparams, sec:method:rl). Required for
    # every controller, since all three learn from the same reward. A planner
    # is not learned, so it has no use for them and they stay optional.
    discount: PositiveFloat | None = None  # gamma, learning objective
    deliver_reward: NonNegativeFloat | None = None  # n_deliver, reward
    override_penalty: NonNegativeFloat | None = None  # n_blocked, reward
    move_cost: NonNegativeFloat | None = None  # c_move, reward
    congestion_reward_weight: NonNegativeFloat | None = None  # n_cng, reward

    # The order distribution P_ord (sec:pf:tasks). Without a skew the SKU of an order is drawn
    # uniformly among the SKUs in stock. With one it is drawn by popularity, the SKU at rank r
    # weighing 1 / (r + 1)^demand_skew, and the ranking rotates left by demand_rotation_shift
    # places every demand_rotation_period steps. These are study parameters (tab:evalparams),
    # so None, today's uniform draw, is the only default.
    demand_skew: NonNegativeFloat | None = None
    demand_rotation_period: PositiveInt | None = None  # None means no rotation
    demand_rotation_shift: PositiveInt | None = None

    # The stock model. Under "arrival" Delta is the arrival period for every storage rule,
    # F_fix included, and nothing is relocated, so there is no nu. Each SKU is filled up to
    # max(stock_floor, ceil((1 + stock_margin) * its orders in the last period)). The margin
    # epsilon, the floor and the receiving vertices are study parameters, so the arrival model
    # takes no defaults, and "template", today's model, is the default.
    stock_model: StockModel = "template"
    stock_margin: NonNegativeFloat | None = None  # epsilon
    stock_floor: PositiveInt | None = None  # the fewest units a SKU is filled up to
    stock_receiving: ReceivingVertices | None = None  # V_rcv

    @model_validator(mode="after")
    def _storage_mode_parameters(self) -> "ExperimentConfig":
        if self.stock_model == "arrival":
            return self  # _arrival_stock_parameters checks the arrival model
        if self.storage_mode == "fixed":
            if self.storage_epoch_length is not None:
                raise ValueError(
                    "storage_mode='fixed' means Delta=infinity (F_fix); "
                    "do not set storage_epoch_length."
                )
        else:
            for name in ("storage_epoch_length", "reassignment_cap"):
                if getattr(self, name) is None:
                    raise ValueError(f"storage_mode={self.storage_mode!r} requires {name}.")
            if self.storage_mode == "congestion" and self.congestion_weight is None:
                raise ValueError("storage_mode='congestion' requires congestion_weight (beta).")
        return self

    @model_validator(mode="after")
    def _arrival_stock_parameters(self) -> "ExperimentConfig":
        """The template model has no arrivals, so it takes no stock settings. The arrival model
        needs all of them, and Delta under every storage rule, and it relocates nothing, so it
        takes no nu."""
        stock = ("stock_margin", "stock_floor", "stock_receiving")
        if self.stock_model == "template":
            for name in stock:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} only applies to stock_model='arrival'.")
            return self
        for name in (*stock, "storage_epoch_length"):
            if getattr(self, name) is None:
                raise ValueError(f"stock_model='arrival' requires {name}.")
        if self.reassignment_cap is not None:
            raise ValueError(
                "stock_model='arrival' relocates nothing, so do not set reassignment_cap (nu)."
            )
        if self.storage_mode == "congestion" and self.congestion_weight is None:
            raise ValueError("storage_mode='congestion' requires congestion_weight (beta).")
        return self

    @model_validator(mode="after")
    def _demand_parameters(self) -> "ExperimentConfig":
        """A rotation reorders a popularity ranking, so it needs a skew, and a period and a
        shift only mean something together."""
        if self.demand_rotation_period is not None:
            if self.demand_skew is None:
                raise ValueError("demand_rotation_period requires demand_skew.")
            if self.demand_rotation_shift is None:
                raise ValueError("demand_rotation_period requires demand_rotation_shift.")
        elif self.demand_rotation_shift is not None:
            raise ValueError("demand_rotation_shift requires demand_rotation_period.")
        return self

    @model_validator(mode="after")
    def _exactly_one_of_controller_and_planner(self) -> "ExperimentConfig":
        if (self.controller is None) == (self.planner is None):
            raise ValueError("Set exactly one of controller and planner.")
        return self

    @property
    def router_name(self) -> ControllerArchitecture | PlannerName:
        """The controller or planner this run uses, whichever one is set."""
        name = self.controller if self.controller is not None else self.planner
        assert name is not None  # the validator above guarantees one of the two
        return name

    @model_validator(mode="after")
    def _controller_parameters(self) -> "ExperimentConfig":
        if self.controller is not None:
            for name in (
                "discount",
                "deliver_reward",
                "override_penalty",
                "move_cost",
                "congestion_reward_weight",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"controller={self.controller!r} requires {name}.")
        if self.controller != "decentralised" and self.communication:
            raise ValueError(
                "communication=True only applies to controller='decentralised' "
                "(matches main.py's CLI check)."
            )
        return self

    @model_validator(mode="after")
    def _congestion_sensitivity_parameters(self) -> "ExperimentConfig":
        if self.congestion_sensitive:
            if self.storage_mode == "congestion" and self.congestion_weight is None:
                raise ValueError(
                    "congestion_sensitive=True with storage_mode='congestion' requires "
                    "congestion_weight (beta)."
                )
        return self

    @model_validator(mode="after")
    def _keep_up_threshold_at_most_one(self) -> "ExperimentConfig":
        """The keep-up check states f_up <= 1: a fleet cannot be required to
        finish tasks faster than they arrive."""
        if self.keep_up_threshold is not None and self.keep_up_threshold > 1:
            raise ValueError(f"keep_up_threshold (f_up) = {self.keep_up_threshold} must be <= 1.")
        return self

    @model_validator(mode="after")
    def _discount_in_unit_interval(self) -> "ExperimentConfig":
        """The learning objective's gamma is a standard POSG discount factor; the
        thesis pins no explicit bound beyond that, so the conventional
        (0, 1] range is enforced here as a flagged assumption, not
        silently assumed."""
        if self.discount is not None and self.discount > 1:
            raise ValueError(f"discount (gamma) = {self.discount} must be in (0, 1].")
        return self
