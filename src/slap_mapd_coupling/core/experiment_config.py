"""ExperimentConfig: the independent variables of the study (sec:pd-aim-and-scope)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

StorageMode = Literal["fixed", "demand", "congestion"]  # F_fix / F_dem / F_cng
ControllerArchitecture = Literal["centralised", "section", "decentralised"]


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
    controller: ControllerArchitecture
    congestion_sensitive: bool
    communication: bool  # only meaningful if controller == "decentralised"
    num_agents: PositiveInt  # m
    arrival_rate: PositiveFloat  # lambda_task
    seed: int
    horizon: PositiveInt  # T, evaluation window

    wait_cost: NonNegativeFloat  # c_wait
    storage_epoch_length: PositiveInt | None = None  # Delta; None means F_fix (Delta = infinity)
    congestion_weight: PositiveFloat | None = None  # beta, eq:storagegreedy
    reassignment_cap: PositiveInt | None = None  # nu, max relocated units/epoch
    observation_depth: PositiveInt | None = None  # d, eq:localsubgraph
    congestion_radius: PositiveInt | None = None  # r_cng, eq:congestion
    communication_radius: PositiveInt | None = None  # r_com, eq:commgraph

    @model_validator(mode="after")
    def _storage_mode_parameters(self) -> "ExperimentConfig":
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
    def _controller_parameters(self) -> "ExperimentConfig":
        if self.controller == "decentralised":
            if self.observation_depth is None:
                raise ValueError("controller='decentralised' requires observation_depth (d).")
            if self.communication and self.communication_radius is None:
                raise ValueError("communication=True requires communication_radius (r_com).")
        elif self.communication:
            raise ValueError(
                "communication=True only applies to controller='decentralised' "
                "(matches main.py's CLI check)."
            )
        return self

    @model_validator(mode="after")
    def _congestion_sensitivity_parameters(self) -> "ExperimentConfig":
        if self.congestion_sensitive:
            if self.controller == "decentralised" and self.congestion_radius is None:
                raise ValueError(
                    "congestion_sensitive=True with a decentralised controller requires "
                    "congestion_radius (r_cng)."
                )
            if self.storage_mode == "congestion" and self.congestion_weight is None:
                raise ValueError(
                    "congestion_sensitive=True with storage_mode='congestion' requires "
                    "congestion_weight (beta)."
                )
        return self
