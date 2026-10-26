"""Run metrics (sec:pf:measures): keeping up, total service time and its
three parts, mean pick service time, crowding, relocations per storage
update, and runtime.

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
    "mean_arrived_units",
    "mean_split_skus",
    "unplaced_units",
    "overflow_units",
    "num_refills",
    "num_refilled_skus",
)
_ORDER_FIELDS = (
    "mean_order_completion",
    "num_completed_orders",
    "orders_per_timestep",
    "mean_tasks_per_line",
    "num_lines_waited",
)
_COLD_STOCK_FIELDS = ("mean_cold_prime_share", "mean_unit_age_at_pick")
_SHELF_FILL_FIELDS = ("peak_shelf_fill_share", "mean_shelf_fill_share")
_ARRIVAL_FIELDS = (
    "mean_stock_wait",
    "num_waiting_orders",
    "num_released_putaways",
    "num_completed_putaways",
    "mean_putaway_service_time",
    *_PUTAWAY_FIELDS,
    *_ORDER_FIELDS,
    *_COLD_STOCK_FIELDS,
    *_SHELF_FILL_FIELDS,
)


class RunMetrics(BaseModel):
    """The dependent variables of one completed run (sec:pf:measures) --
    one CSV row. Field names track the thesis's own symbols, not just
    prose, so a reviewer can grep an equation label straight to the field
    that reports it.

    The score has one result, the mean service time of picks, valid only if
    the run keeps up (tab:measures). The three parts of service time and
    crowding explain it, and the put-away service time and runtime are costs
    reported next to it. Nothing here is a weighted sum: each field is a
    count or an average.
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
    # T_warm, the warm-up left out of the measures ([A66]). The counts, means and the keep-up
    # check below cover the tasks released in [T_warm, T), and throughput is per step of the
    # window. 0 measures the whole run. The totals of the service time, the order measures and
    # the robot busy share still cover the whole run.
    warmup: NonNegativeInt = 0
    observation_depth: PositiveInt  # d_obs, the window the run was scored under
    # "template" (stock never runs out) or "arrival" (stock arrives every Delta steps).
    stock_model: StockModel = "template"

    # --- keeping up, eq:throughput: a check, applied first ---
    num_completed_tasks: NonNegativeInt  # |C_T|
    throughput: NonNegativeFloat  # Lambda_T = |C_T| / T
    # Lambda_T / m, whether each agent finishes fewer tasks as the fleet grows ([A33], RQ4).
    throughput_per_agent: NonNegativeFloat = 0.0
    # |C_T| / (|C_T| + B_T), the share of the released tasks that finished. None iff nothing
    # was released.
    completed_share: NonNegativeFloat | None = None
    # The keep-up check, eq:throughput, applied to picks and to put-aways separately ([A59]).
    # True iff both parts pass. None unless f_up is configured.
    keeps_up: bool | None = None
    # Its two parts, |C^pick_T| >= f_up (|C^pick_T| + B^pick_T) and the same for put-aways. A
    # run with no put-away released passes the put-away part.
    keeps_up_picks: bool | None = None
    keeps_up_putaways: bool | None = None
    # The longest run of consecutive blocked ticks of any task, finished or still active at T. A
    # diagnostic, since under Token Passing standing still is often planned.
    longest_stand_still: NonNegativeInt = 0
    # The longest stretch of steps in [T_warm, T] in which no robot finished a task (D18).
    longest_completion_gap: NonNegativeInt = 0
    # longest_completion_gap > the stall length, a jam the fleet may have caught up from. None
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
    # B^put_T, the put-aways in B_T. The rest of B_T, B^pick_T, is the picks and the orders
    # waiting for stock ([A59]). Always 0 without robot put-away.
    putaway_backlog: NonNegativeInt = 0
    # The least-squares slope of B_t over the second half of the run, per 100 timesteps. Shows a
    # backlog still growing at T, which one finite horizon cannot rule out ([A55]). None with
    # fewer than two timesteps in the second half.
    backlog_slope: float | None = None

    # --- the result, the total service time zeta^tot_T, and its split ---
    # Every task released by T adds min(t^finish, T) - t^released, and every order line still
    # waiting for stock at T adds T - its release. An open task adds its wait up to its
    # assignment or T, then its moving and blocked ticks up to T. A waiting line adds all its
    # time to the wait. The four add up.
    total_service_time: NonNegativeInt = 0
    total_wait_for_agent: NonNegativeInt = 0
    total_travel_time: NonNegativeInt = 0
    total_blocked_time: NonNegativeInt = 0
    # The same sum over the completed tasks only, to show how much the open work adds.
    total_service_time_finished: NonNegativeInt = 0
    # Arrival stock model with robot put-away only: the put-aways released by T, those completed
    # by T and their mean service time, None without one. They count in num_completed_tasks.
    num_released_putaways: NonNegativeInt = 0
    num_completed_putaways: NonNegativeInt = 0
    mean_putaway_service_time: NonNegativeFloat | None = None

    # --- mean service time, eq:meanservice, and its split, eq:split ---
    # Over the completed picks only. Every task of a run without robot put-away is a pick, so
    # there they are the means over every completed task, as before put-aways existed. All
    # four are None iff no pick completed.
    mean_service_time: NonNegativeFloat | None = None  # zeta_bar_T
    mean_wait_for_agent: NonNegativeFloat | None = None  # mean of t_assigned - t_released
    mean_travel_time: NonNegativeFloat | None = None  # mean of zeta_j^travel
    mean_blocked_time: NonNegativeFloat | None = None  # W_T, eq:waiting
    # Arrival stock model only: the mean over completed picks of the steps their order waited
    # for a unit of its SKU. A task keeps its order's release time, so this wait is part of
    # mean_wait_for_agent. None under the template model, and None iff no pick completed.
    mean_stock_wait: NonNegativeFloat | None = None
    # Arrival stock model only. The order completion time, the longest service time among an
    # order's tasks, averaged over the orders whose tasks are all completed and none of whose
    # lines still waits. None without such an order.
    mean_order_completion: NonNegativeFloat | None = None
    num_completed_orders: NonNegativeInt = 0
    # Arrival stock model only. num_completed_orders / T, reported next to throughput, which
    # counts tasks ([A56]). None under the template model.
    orders_per_timestep: NonNegativeFloat | None = None
    # Arrival stock model only. Tasks per order line, over the lines with no units still
    # missing, so how often a line is split over shelves. None without such a line.
    mean_tasks_per_line: NonNegativeFloat | None = None
    # Arrival stock model only. The order lines that waited for stock at some point.
    num_lines_waited: NonNegativeInt = 0

    # --- congestion, eq:crowding ---
    mean_crowding: NonNegativeFloat = Field(le=1.0)  # bar_delta_T in [0,1]
    # The share of agent-timesteps in which an agent holds an active task, from its assignment
    # to its finish or T. A long wait for an agent while agents stand idle points at the task
    # rule rather than the fleet size. Diagnostic, not in the thesis.
    robot_busy_share: NonNegativeFloat = Field(default=0.0, le=1.0)
    # Token Passing only, None otherwise. The mean over completed picks, and over completed
    # put-aways, of the timesteps the task waited while some agent was idle and another agent's
    # path ended at the task's pickup or at its delivery vertex, so Token Passing's task rule
    # kept it from the idle agent (planners/token_passing.py). A pick's pickup is a shelf and
    # its delivery a delivery vertex. A put-away's pickup is where stock arrives (a receiving
    # vertex with arrival.receiving: receiving) and its delivery a shelf. A timestep with both
    # ends taken counts in both. Diagnostic, not in the thesis.
    mean_pick_wait_pickup_taken: NonNegativeFloat | None = None
    mean_pick_wait_delivery_taken: NonNegativeFloat | None = None
    mean_putaway_wait_pickup_taken: NonNegativeFloat | None = None
    mean_putaway_wait_delivery_taken: NonNegativeFloat | None = None

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
    mean_arrived_units: NonNegativeFloat = 0.0  # units arrived per arrival, placed or not
    # SKUs held at more than one storage vertex after an arrival, averaged over arrivals.
    mean_split_skus: NonNegativeFloat = 0.0
    unplaced_units: NonNegativeInt = 0  # arriving units that fit on no shelf, over the episode
    # Units F_fix placed off their SKU's home shelf, over the episode. 0 under the adaptive rules.
    overflow_units: NonNegativeInt = 0
    # Refill rule only, 0 under the top-up rule: the refills over the episode, placed or not,
    # which are the storage decisions of a run, and how many SKUs got at least one.
    num_refills: NonNegativeInt = 0
    num_refilled_skus: NonNegativeInt = 0

    # --- shelf fill, arrival stock model only. Diagnostic, not in the thesis ---
    # After each arrival, stock on its way counted, the used capacity of a shelf over its
    # capacity. The fullest shelf over the run, and the mean over arrivals of the mean over
    # shelves. A rule that piles stock onto a few shelves has a peak near 1 and a low mean. None
    # without an arrival.
    peak_shelf_fill_share: NonNegativeFloat | None = None
    mean_shelf_fill_share: NonNegativeFloat | None = None

    # --- cold stock, arrival stock model only, evaluation only ---
    # The share of the prime shelves' capacity that holds SKUs no longer among the top
    # cold_stock_top_k, averaged over arrivals. None unless cold_stock_top_k is set and an
    # arrival happened.
    mean_cold_prime_share: NonNegativeFloat | None = Field(default=None, le=1.0)
    # The steps a unit had been on its shelf when it was taken for a task, averaged over the
    # units taken. A shelf gives up its oldest units first. None if no unit was taken.
    mean_unit_age_at_pick: NonNegativeFloat | None = None

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
        if self.num_completed_picks == 0:
            if any(value is not None for value in values):
                raise ValueError(
                    "service-time fields must be None when no pick completed "
                    "(no finished pick to average over, eq:meanservice)."
                )
        elif any(value is None for value in values):
            raise ValueError(
                "with a completed pick, mean_service_time and all three of "
                "its parts must be set (eq:split)."
            )
        if self.num_completed_putaways > self.num_completed_tasks:
            raise ValueError("num_completed_putaways cannot exceed num_completed_tasks.")
        if self.num_completed_putaways > self.num_released_putaways:
            raise ValueError("num_completed_putaways cannot exceed num_released_putaways.")
        if (self.num_completed_putaways == 0) != (self.mean_putaway_service_time is None):
            raise ValueError(
                "mean_putaway_service_time is set iff a put-away completed."
            )
        return self

    @property
    def num_completed_picks(self) -> int:
        return self.num_completed_tasks - self.num_completed_putaways

    @model_validator(mode="after")
    def _total_split_adds_up(self) -> "RunMetrics":
        parts = self.total_wait_for_agent + self.total_travel_time + self.total_blocked_time
        if parts != self.total_service_time:
            raise ValueError(
                f"waiting + travel + blocked = {parts} must equal total_service_time = "
                f"{self.total_service_time}."
            )
        if self.total_service_time_finished > self.total_service_time:
            raise ValueError("total_service_time_finished cannot exceed total_service_time.")
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
    def _keep_up_is_both_parts(self) -> "RunMetrics":
        parts = (self.keeps_up_picks, self.keeps_up_putaways)
        if self.keeps_up is None or any(part is None for part in parts):
            return self
        if self.keeps_up != all(parts):
            raise ValueError("keeps_up must be True iff keeps_up_picks and keeps_up_putaways are.")
        return self

    @model_validator(mode="after")
    def _backlog_is_sum(self) -> "RunMetrics":
        if self.putaway_backlog > self.backlog:
            raise ValueError("putaway_backlog is part of backlog, so it cannot exceed it.")
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
        if self.num_completed_picks == 0:
            if self.mean_stock_wait is not None:
                raise ValueError("mean_stock_wait must be None when no pick completed.")
            return self
        if self.mean_stock_wait is None:
            raise ValueError(
                "under stock_model='arrival' with a completed pick, mean_stock_wait "
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
