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
    horizon: PositiveInt  # T, the evaluation horizon (tab:evalparams)

    wait_cost: NonNegativeFloat  # c_wait
    storage_epoch_length: PositiveInt | None = None  # Delta; None means F_fix (Delta = infinity)
    congestion_weight: PositiveFloat | None = None  # beta, eq:storagegreedy
    reassignment_cap: PositiveInt | None = None  # nu, max relocated units/epoch
    # d_obs, eq:localsubgraph. The one field of view: it also bounds the
    # communication graph (eq:commgraph) and the congestion feature
    # (eq:congestion), so there are no separate radii (issue #88).
    observation_depth: PositiveInt | None = None
    # f_up, keep-up threshold in eq:throughput (tab:evalparams, TBD, #85).
    # Optional until the evaluator uses it (#93).
    keep_up_threshold: PositiveFloat | None = Field(default=None, le=1.0)

    # eq:reward / eq:objective (tab:rlparams, sec:method:rl): required only
    # for controller="decentralised", same gating as observation_depth
    # above, since eq:objective explicitly "applies to whichever regime is
    # realised as a LEARNED controller" and only the decentralised regime
    # trains (AGENTS.md: --checkpoint never applies to the other two).
    discount: PositiveFloat | None = None  # gamma, eq:objective
    deliver_reward: NonNegativeFloat | None = None  # n_deliver, eq:reward
    override_penalty: NonNegativeFloat | None = None  # n_blocked, eq:reward
    congestion_reward_weight: NonNegativeFloat | None = None  # n_cng, eq:reward

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
                raise ValueError("controller='decentralised' requires observation_depth (d_obs).")
            for name in (
                "discount",
                "deliver_reward",
                "override_penalty",
                "congestion_reward_weight",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"controller='decentralised' requires {name}.")
        elif self.communication:
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
        """eq:throughput states f_up <= 1: a fleet cannot be required to
        finish tasks faster than they arrive."""
        if self.keep_up_threshold is not None and self.keep_up_threshold > 1:
            raise ValueError(f"keep_up_threshold (f_up) = {self.keep_up_threshold} must be <= 1.")
        return self

    @model_validator(mode="after")
    def _discount_in_unit_interval(self) -> "ExperimentConfig":
        """eq:objective's gamma is a standard POSG discount factor; the
        thesis pins no explicit bound beyond that, so the conventional
        (0, 1] range is enforced here as a flagged assumption, not
        silently assumed."""
        if self.discount is not None and self.discount > 1:
            raise ValueError(f"discount (gamma) = {self.discount} must be in (0, 1].")
        return self
