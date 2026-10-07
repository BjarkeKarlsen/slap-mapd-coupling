"""Run metrics (sec:pf:measures): keeping up, mean service time and its
three parts, crowding, relocations per storage update, and runtime."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, model_validator

from slap_mapd_coupling.core.experiment_config import ControllerArchitecture, StorageMode

# The three parts of service time are integer counts averaged over the
# same tasks, so they add up to mean_service_time up to float rounding only.
_SPLIT_TOLERANCE = 1e-9


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
    controller: ControllerArchitecture
    congestion_sensitive: bool
    communication: bool
    num_agents: NonNegativeInt
    arrival_rate: NonNegativeFloat
    seed: int
    horizon: NonNegativeInt  # T

    # --- keeping up, eq:throughput: a check, applied first ---
    num_completed_tasks: NonNegativeInt  # |C_T|
    throughput: NonNegativeFloat  # Lambda_T = |C_T| / T
    keep_up_ratio: NonNegativeFloat  # Lambda_T / lambda_task
    # Lambda_T >= f_up * lambda_task; None iff f_up (keep_up_threshold) is
    # not configured, since tab:evalparams leaves its value TBD.
    keeps_up: bool | None = None
    num_waiting_tasks: NonNegativeInt  # |Q_T|
    num_active_tasks: NonNegativeInt  # |B_T|
    backlog: NonNegativeInt  # B_T = |Q_T| + |B_T|, the last entry of the B_t trace

    # --- the result, eq:meanservice, and its split, eq:split ---
    # All four are None iff |C_T| = 0: a run with no finished task has no
    # service time to average, and fails the keep-up check anyway.
    mean_service_time: NonNegativeFloat | None = None  # zeta_bar_T
    mean_wait_for_agent: NonNegativeFloat | None = None  # mean of t_assigned - t_released
    mean_travel_time: NonNegativeFloat | None = None  # mean of zeta_j^travel
    mean_blocked_time: NonNegativeFloat | None = None  # W_T, eq:waiting

    # --- congestion, eq:crowding ---
    # bar_delta_T in [0,1]; None iff observation_depth (d_obs) is not configured.
    mean_crowding: NonNegativeFloat | None = Field(default=None, le=1.0)

    # --- slow-loop cost, eq:relocation ---
    num_storage_updates: NonNegativeInt  # floor(T / Delta); 0 under F_fix
    mean_relocated_units: NonNegativeFloat  # bar_nu_T; 0.0 when there was no update

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
        if self.backlog != self.num_waiting_tasks + self.num_active_tasks:
            raise ValueError("backlog must equal num_waiting_tasks + num_active_tasks (B_T).")
        return self

    @model_validator(mode="after")
    def _no_update_relocates_nothing(self) -> "RunMetrics":
        if self.num_storage_updates == 0 and self.mean_relocated_units != 0.0:
            raise ValueError(
                "mean_relocated_units must be 0.0 when num_storage_updates == 0 "
                "(F_fix always scores 0, eq:relocation)."
            )
        return self
