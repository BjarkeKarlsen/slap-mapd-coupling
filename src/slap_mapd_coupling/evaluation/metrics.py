"""Run metrics: service time, throughput, movement cost, waiting/backlog,
traffic entropy and concentration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, model_validator

from slap_mapd_coupling.core.experiment_config import ControllerArchitecture, StorageMode


class RunMetrics(BaseModel):
    """The dependent variables of one completed run (sec:pf:measures) --
    one CSV row. Field names track the thesis's own symbols, not just
    prose, so a reviewer can grep an equation label straight to the field
    that reports it.
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

    # --- completed-work / cost measures, eq:throughput / eq:movementcost ---
    num_completed_tasks: NonNegativeInt  # |C_T|
    throughput: NonNegativeFloat  # Lambda_T = |C_T| / T
    mean_service_time: NonNegativeFloat | None = None  # zeta_bar_T; None iff |C_T|=0
    movement_cost_per_task: NonNegativeFloat | None = None  # c_bar_T; None iff |C_T|=0

    # --- backlog, eq:functional's B_T ---
    num_waiting_tasks: NonNegativeInt  # |Q_T|
    num_active_tasks: NonNegativeInt  # |B_T|
    backlog: NonNegativeInt  # B_T = |Q_T| + |B_T|

    # --- waiting, eq:waiting ---
    mean_blocked_time: NonNegativeFloat | None = None  # W_T; None iff |C_T|=0

    # --- traffic distribution, eq:entropy / eq:concentration ---
    num_traversed_edges: NonNegativeInt  # |E+_T|
    traffic_entropy: NonNegativeFloat | None = Field(default=None, le=1.0)  # H_T in [0,1]
    traffic_concentration: NonNegativeFloat | None = Field(default=None, le=1.0)  # C_T = 1 - H_T

    # --- controller cost, eq:runtime ---
    mean_decision_runtime_seconds: NonNegativeFloat  # kappa_T

    @model_validator(mode="after")
    def _zero_completion_reported_separately(self) -> "RunMetrics":
        if self.num_completed_tasks == 0:
            for name in ("mean_service_time", "movement_cost_per_task", "mean_blocked_time"):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} must be None when num_completed_tasks == 0 "
                        "('runs with |C_T|=0 are reported separately', eq:throughput)."
                    )
        return self

    @model_validator(mode="after")
    def _zero_or_single_traversal_conventions(self) -> "RunMetrics":
        if self.num_traversed_edges == 0:
            if self.traffic_entropy is not None or self.traffic_concentration is not None:
                raise ValueError(
                    "traffic_entropy/traffic_concentration must be None when "
                    "num_traversed_edges == 0 (reported separately, eq:entropy)."
                )
        elif self.num_traversed_edges == 1:
            if self.traffic_entropy != 0.0 or self.traffic_concentration != 1.0:
                raise ValueError(
                    "|E+_T|=1 is the by-convention maximally-concentrated case: "
                    "H_T must be 0.0 and C_T must be 1.0 (eq:entropy's convention)."
                )
        return self

    @model_validator(mode="after")
    def _entropy_concentration_are_complements(self) -> "RunMetrics":
        if self.traffic_entropy is not None and self.traffic_concentration is not None:
            if abs((self.traffic_entropy + self.traffic_concentration) - 1.0) > 1e-9:
                raise ValueError("C_T must equal 1 - H_T (eq:concentration).")
        return self

    @model_validator(mode="after")
    def _backlog_is_sum(self) -> "RunMetrics":
        if self.backlog != self.num_waiting_tasks + self.num_active_tasks:
            raise ValueError(
                "backlog must equal num_waiting_tasks + num_active_tasks (eq:functional's B_T)."
            )
        return self
