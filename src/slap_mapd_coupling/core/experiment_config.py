"""ExperimentConfig: the independent variables of the study (sec:pd-aim-and-scope)."""

from __future__ import annotations

from typing import Literal

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

from slap_mapd_coupling.planners.registry import PlannerName

# F_fix / F_dem / F_cng, and "random", a check under the arrival stock model only
StorageMode = Literal["fixed", "demand", "congestion", "random"]
# How stock behaves. "template": stock never runs out and an adaptive rule relocates it every
# Delta steps. "arrival": an order takes a unit, stock arrives every Delta steps, and the storage
# rule decides where the arriving stock goes (thesis wording pending).
StockModel = Literal["template", "arrival"]
# Where arriving stock enters the warehouse. "delivery": the delivery vertices, one I/O side.
# "receiving": the vertices with the receiving role, on the inbound side (sec:impl:instances).
ReceivingVertices = Literal["delivery", "receiving"]
# How arriving stock reaches its shelf. "instant": it is on the shelf at the arrival.
# "robots": every (SKU, shelf) of an arrival is a put-away task, and the units are on their
# way, x^in_t, until it is delivered (sec:pf:storage).
StockPutaway = Literal["instant", "robots"]
# How much stock arrives, arrival model only. "topup": every SKU is filled up to its last
# period's orders, with a margin and a floor. "refill": stock is replaced a refill of Q units at a
# time, once Q units have been ordered since the last refill (the (R, nQ) policy).
StockRule = Literal["topup", "refill"]


class LinesPerOrder(BaseModel):
    """The number of lines of an order (tab:evalparams, mean 5, distribution TBD).
    "shifted_poisson" is 1 + Poisson(mean - 1), so every order has a line and the mean is exact.
    An order never has more lines than there are SKUs, since its lines are distinct SKUs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distribution: Literal["shifted_poisson"]
    mean: float = Field(ge=1.0)


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
    # T_warm, the warm-up left out of the measures ([A66]), chosen by Welch's method. The
    # measures cover the tasks released in [T_warm, T). 0 measures the whole run.
    warmup: NonNegativeInt = 0

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
    # f_up in the keep-up check (tab:evalparams). RunMetrics.keeps_up stays unset without it.
    keep_up_threshold: PositiveFloat | None = Field(default=None, le=1.0)
    # The stall length (tab:evalparams), TBD. An episode in which an agent holding a task stands
    # still longer than this is a stall. None leaves RunMetrics.stalled unset.
    stall_length: PositiveInt | None = None

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
    # The shift as a share of the SKUs instead, rounded to a whole number of places and at least
    # one, so the ranking turns once in about the same time whatever the number of SKUs.
    demand_rotation_share: float | None = Field(default=None, gt=0.0, lt=1.0)

    # The stock model. Under "arrival" Delta is the arrival period for every storage rule,
    # F_fix included, and nothing is relocated, so there is no nu. Under stock_rule "topup" each
    # SKU is filled up to max(stock_floor, ceil((1 + stock_margin) * its orders in the last
    # period)). Under "refill" a SKU gets floor(counted / refill_units) refills of refill_units
    # units at each arrival, counted being its units ordered since its last refill, and the
    # remainder carries over. The service level only sizes the starting stock (training/spec.py),
    # and is kept here so a run records it. The margin epsilon, the floor, Q, the service level
    # and the receiving vertices are study parameters, so the arrival model takes no defaults,
    # and "template", today's model, is the default.
    stock_model: StockModel = "template"
    stock_rule: StockRule = "topup"
    stock_margin: NonNegativeFloat | None = None  # epsilon
    stock_floor: PositiveInt | None = None  # the fewest units a SKU is filled up to
    refill_units: PositiveInt | None = None  # Q
    service_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    stock_receiving: ReceivingVertices | None = None  # V_rcv
    stock_putaway: StockPutaway = "instant"
    # Order lines, arrival model only. An order is lines of distinct SKUs, each asking for a
    # quantity drawn uniformly from the inclusive range units_per_line. None means one line of
    # one unit, the order before lines existed. arrival_rate stays lines per timestep.
    lines_per_order: LinesPerOrder | None = None
    units_per_line: tuple[PositiveInt, PositiveInt] | None = None
    # The cold-stock measure, arrival model only, evaluation only. A SKU is popular while it is
    # among the top cold_stock_top_k of the current popularity ranking. The prime shelves are
    # the storage vertices nearest the delivery vertices on average, as many as the starting
    # placement fills with the top cold_stock_top_k SKUs. None leaves the measure off.
    cold_stock_top_k: PositiveInt | None = None

    @model_validator(mode="after")
    def _storage_mode_parameters(self) -> "ExperimentConfig":
        if self.stock_model == "arrival":
            return self  # _arrival_stock_parameters checks the arrival model
        if self.storage_mode == "random":
            raise ValueError(
                "storage_mode='random' is a put-away rule, so needs stock_model='arrival'."
            )
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
        topup = ("stock_margin", "stock_floor")
        refill = ("refill_units", "service_level")
        if self.units_per_line is not None and self.units_per_line[0] > self.units_per_line[1]:
            raise ValueError(f"units_per_line = {self.units_per_line} must be (low, high).")
        if self.stock_model == "template":
            if self.stock_putaway != "instant":
                raise ValueError("stock_putaway only applies to stock_model='arrival'.")
            if self.stock_rule != "topup":
                raise ValueError("stock_rule only applies to stock_model='arrival'.")
            for name in ("lines_per_order", "units_per_line", "cold_stock_top_k"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} only applies to stock_model='arrival'.")
            for name in (*topup, *refill, "stock_receiving"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} only applies to stock_model='arrival'.")
            return self
        needed, unused = (topup, refill) if self.stock_rule == "topup" else (refill, topup)
        for name in (*needed, "stock_receiving", "storage_epoch_length"):
            if getattr(self, name) is None:
                raise ValueError(f"stock_model='arrival' requires {name}.")
        for name in unused:
            if getattr(self, name) is not None:
                raise ValueError(f"{name} does not apply to stock_rule={self.stock_rule!r}.")
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
        shifts = (self.demand_rotation_shift, self.demand_rotation_share)
        if self.demand_rotation_period is not None:
            if self.demand_skew is None:
                raise ValueError("demand_rotation_period requires demand_skew.")
            if sum(shift is not None for shift in shifts) != 1:
                raise ValueError(
                    "demand_rotation_period requires one of demand_rotation_shift and "
                    "demand_rotation_share."
                )
        elif any(shift is not None for shift in shifts):
            raise ValueError(
                "demand_rotation_shift and demand_rotation_share require demand_rotation_period."
            )
        if self.cold_stock_top_k is not None and self.demand_skew is None:
            raise ValueError("cold_stock_top_k needs a popularity ranking, so demand_skew.")
        return self

    @model_validator(mode="after")
    def _exactly_one_of_controller_and_planner(self) -> "ExperimentConfig":
        if (self.controller is None) == (self.planner is None):
            raise ValueError("Set exactly one of controller and planner.")
        return self

    def rotation_shift(self, num_skus: int) -> int | None:
        """The places the popularity ranking rotates by each period, for `num_skus` SKUs: the
        shift, or the share of the SKUs rounded and at least one. None without a rotation."""
        if self.demand_rotation_share is not None:
            return max(1, round(self.demand_rotation_share * num_skus))
        return self.demand_rotation_shift

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
    def _warmup_inside_the_horizon(self) -> "ExperimentConfig":
        """The measured window [T_warm, T] must hold at least one step."""
        if self.warmup >= self.horizon:
            raise ValueError(f"warmup = {self.warmup} must be below the horizon {self.horizon}.")
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
