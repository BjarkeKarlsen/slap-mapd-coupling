"""Run metrics (sec:pf:measures): keeping up, mean service time and its
three parts, crowding, relocations per storage update, and runtime.

Under the arrival stock model (thesis wording pending) the storage update is
the put-away of arriving stock, so a row also reports the wait for stock, the
orders still waiting for it, and the put-away measures of every arrival.
Nothing is relocated, so the relocation fields stay 0. Under the template
model those fields keep their defaults, so a template row is the one it was
before they existed."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from slap_mapd_coupling.core.experiment_config import (
    ControllerArchitecture,
    StockModel,
    StorageMode,
)
from slap_mapd_coupling.planners.registry import PlannerName

# The three parts of service time are integer counts averaged over the
# same tasks, so they add up to mean_service_time up to float rounding only.
_SPLIT_TOLERANCE = 1e-9

# The measures of the arrival stock model, put-away first. Every one is at its default under the
# template model.
_PUTAWAY_FIELDS = (
    "mean_putaway_distance",
    "mean_arrived_units",
    "mean_split_skus",
    "unplaced_units",
    "overflow_units",
)
_ARRIVAL_FIELDS = ("mean_stock_wait", "num_waiting_orders", *_PUTAWAY_FIELDS)


class RunMetrics(BaseModel):
    """The dependent variables of one completed run (sec:pf:measures) --
    one CSV row. Field names track the thesis's own symbols, not just
    prose, so a reviewer can grep an equation label straight to the field
    that reports it.

    The score has one result, mean service time, valid only if the run
    keeps up. The three parts of service time and crowding explain it;
    relocations and runtime are costs reported next to it. Nothing here is
    a weighted sum: each field is a count or an average.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- identifying the run, so a CSV row is self-describing ---
    storage_mode: StorageMode
    # Whichever of ExperimentConfig.controller and .planner is set.
    controller: ControllerArchitecture | PlannerName
    congestion_sensitive: bool
    communication: bool
    num_agents: NonNegativeInt
    arrival_rate: NonNegativeFloat
    seed: int
    horizon: NonNegativeInt  # T
    observation_depth: PositiveInt  # d_obs, the window the run was scored under
    # "template" (stock never runs out) or "arrival" (stock arrives every Delta steps).
    stock_model: StockModel = "template"

    # --- keeping up, eq:throughput: a check, applied first ---
    num_completed_tasks: NonNegativeInt  # |C_T|
    throughput: NonNegativeFloat  # Lambda_T = |C_T| / T
    # Lambda_T / lambda_task, throughput over the nominal rate. Reported, no longer the check.
    keep_up_ratio: NonNegativeFloat
    # |C_T| / (|C_T| + B_T), the share of the released tasks that finished. None iff nothing
    # was released.
    completed_share: NonNegativeFloat | None = None
    # chi_T, the least-squares slope of B_t over T/2 <= t <= T, in tasks per timestep.
    backlog_slope: float | None = None
    # The keep-up check: completed_share >= f_up and backlog_slope <= chi_max. None unless
    # both thresholds are configured, since tab:evalparams leaves chi_max TBD.
    keeps_up: bool | None = None
    # The longest run of consecutive blocked ticks of any task, finished or still active at T.
    longest_stand_still: NonNegativeInt = 0
    # longest_stand_still > the stall length, a jam the fleet may have caught up from. None
    # unless the stall length is configured, since tab:evalparams leaves it TBD.
    stalled: bool | None = None
    num_waiting_tasks: NonNegativeInt  # |Q_T|
    num_active_tasks: NonNegativeInt  # |B_T|
    # Arrival stock model only: orders released and still waiting for a unit of their SKU at
    # T. They are not tasks yet, and they count in the backlog. Always 0 under the template model.
    num_waiting_orders: NonNegativeInt = 0
    # B_T = |Q_T| + |B_T|, plus the orders waiting for stock under the arrival model, the last
    # entry of the B_t trace
    backlog: NonNegativeInt

    # --- the result, eq:meanservice, and its split, eq:split ---
    # All four are None iff |C_T| = 0: a run with no finished task has no
    # service time to average, and fails the keep-up check anyway.
    mean_service_time: NonNegativeFloat | None = None  # zeta_bar_T
    mean_wait_for_agent: NonNegativeFloat | None = None  # mean of t_assigned - t_released
    mean_travel_time: NonNegativeFloat | None = None  # mean of zeta_j^travel
    mean_blocked_time: NonNegativeFloat | None = None  # W_T, eq:waiting
    # Arrival stock model only: the mean over completed tasks of the steps their order waited
    # for a unit of its SKU. A task keeps its order's release time, so this wait is part of
    # mean_wait_for_agent. None under the template model, and None iff |C_T| = 0.
    mean_stock_wait: NonNegativeFloat | None = None

    # --- congestion, eq:crowding ---
    mean_crowding: NonNegativeFloat = Field(le=1.0)  # bar_delta_T in [0,1]

    # --- slow-loop cost, eq:relocation ---
    # floor(T / Delta); 0 under F_fix. Under the arrival model the number of arrivals, under
    # every rule.
    num_storage_updates: NonNegativeInt
    mean_relocated_units: NonNegativeFloat  # bar_nu_T; 0.0 when there was no update
    # The relocation measure, averaged over updates: the travel the moved units would
    # cost (units x d_G). Relocation is free in the simulator. 0.0 without an update.
    # Defaults to 0.0 so a RunMetrics built without it (no update) stays valid. The evaluator
    # always fills it.
    mean_relocation_distance: NonNegativeFloat = 0.0

    # --- put-away, arrival stock model only. All stay at 0 under the template model ---
    # The put-away measure, averaged over arrivals: the travel of the arriving units from the
    # nearest receiving vertex to their shelves (units x d_G). 0.0 without an arrival.
    mean_putaway_distance: NonNegativeFloat = 0.0
    mean_arrived_units: NonNegativeFloat = 0.0  # units arrived per arrival, placed or not
    # SKUs held at more than one storage vertex after an arrival, averaged over arrivals.
    mean_split_skus: NonNegativeFloat = 0.0
    unplaced_units: NonNegativeInt = 0  # arriving units that fit on no shelf, over the episode
    # Units F_fix placed off their SKU's home shelf, over the episode. 0 under the adaptive rules.
    overflow_units: NonNegativeInt = 0

    # --- controller cost, eq:runtime ---
    mean_decision_runtime_seconds: NonNegativeFloat  # kappa_T

    @model_validator(mode="after")
    def _zero_completion_has_no_service_time(self) -> "RunMetrics":
        names = (
            "mean_service_time",
            "mean_wait_for_agent",
            "mean_travel_time",
            "mean_blocked_time",
        )
        values = [getattr(self, name) for name in names]
        if self.num_completed_tasks == 0:
            if any(value is not None for value in values):
                raise ValueError(
                    "service-time fields must be None when num_completed_tasks == 0 "
                    "(no finished task to average over, eq:meanservice)."
                )
        elif any(value is None for value in values):
            raise ValueError(
                "with num_completed_tasks > 0, mean_service_time and all three of "
                "its parts must be set (eq:split)."
            )
        return self

    @model_validator(mode="after")
    def _split_adds_up(self) -> "RunMetrics":
        if self.mean_service_time is None:
            return self
        assert self.mean_wait_for_agent is not None  # by the validator above
        assert self.mean_travel_time is not None
        assert self.mean_blocked_time is not None
        parts = self.mean_wait_for_agent + self.mean_travel_time + self.mean_blocked_time
        if abs(parts - self.mean_service_time) > _SPLIT_TOLERANCE:
            raise ValueError(
                f"waiting + travel + blocked = {parts} must equal mean_service_time = "
                f"{self.mean_service_time} (eq:split)."
            )
        return self

    @model_validator(mode="after")
    def _backlog_is_sum(self) -> "RunMetrics":
        if self.backlog != self.num_waiting_tasks + self.num_active_tasks + self.num_waiting_orders:
            raise ValueError(
                "backlog must equal num_waiting_tasks + num_active_tasks + num_waiting_orders "
                "(B_T, with the orders waiting for stock under the arrival model)."
            )
        return self

    @model_validator(mode="after")
    def _template_has_no_arrival_fields(self) -> "RunMetrics":
        """The template model has no arrivals and no wait for stock, so every arrival field
        keeps its default."""
        if self.stock_model != "template":
            return self
        for name in _ARRIVAL_FIELDS:
            if getattr(self, name) != RunMetrics.model_fields[name].default:
                raise ValueError(f"{name} only applies to stock_model='arrival'.")
        return self

    @model_validator(mode="after")
    def _stock_wait_is_part_of_the_wait_for_agent(self) -> "RunMetrics":
        """Under the arrival model the wait for stock is set iff a task completed, and it is
        part of the wait before assignment, which counts from the order's release."""
        if self.stock_model != "arrival":
            return self
        if self.num_completed_tasks == 0:
            if self.mean_stock_wait is not None:
                raise ValueError("mean_stock_wait must be None when num_completed_tasks == 0.")
            return self
        if self.mean_stock_wait is None:
            raise ValueError(
                "under stock_model='arrival' with num_completed_tasks > 0, mean_stock_wait "
                "must be set."
            )
        assert self.mean_wait_for_agent is not None  # by _zero_completion_has_no_service_time
        if self.mean_stock_wait > self.mean_wait_for_agent + _SPLIT_TOLERANCE:
            raise ValueError(
                f"mean_stock_wait = {self.mean_stock_wait} is part of mean_wait_for_agent = "
                f"{self.mean_wait_for_agent}, so it cannot exceed it."
            )
        return self

    @model_validator(mode="after")
    def _arrival_relocates_nothing(self) -> "RunMetrics":
        """Under the arrival model nothing already on a shelf moves, so the relocation fields
        stay 0, and a run without an arrival has no put-away to report."""
        if self.stock_model != "arrival":
            return self
        if self.mean_relocated_units != 0.0 or self.mean_relocation_distance != 0.0:
            raise ValueError(
                "stock_model='arrival' relocates nothing, so mean_relocated_units and "
                "mean_relocation_distance must be 0.0."
            )
        if self.num_storage_updates == 0:
            for name in _PUTAWAY_FIELDS:
                if getattr(self, name) != 0:
                    raise ValueError(f"{name} must be 0 when there was no arrival.")
        return self

    @model_validator(mode="after")
    def _no_update_relocates_nothing(self) -> "RunMetrics":
        if self.num_storage_updates == 0 and self.mean_relocated_units != 0.0:
            raise ValueError(
                "mean_relocated_units must be 0.0 when num_storage_updates == 0 "
                "(F_fix always scores 0, eq:relocation)."
            )
        if self.num_storage_updates == 0 and self.mean_relocation_distance != 0.0:
            raise ValueError(
                "mean_relocation_distance must be 0.0 when num_storage_updates == 0 "
                "(F_fix always scores 0)."
            )
        return self
