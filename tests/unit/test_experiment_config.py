"""Unit tests for slap_mapd_coupling.core.experiment_config."""

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig


def _base(**overrides) -> dict:
    defaults = dict(
        storage_mode="fixed",
        planner="prioritised_astar",
        congestion_sensitive=False,
        communication=False,
        num_agents=5,
        arrival_rate=1.0,
        seed=0,
        horizon=100,
        wait_cost=0.0,
        observation_depth=2,
    )
    defaults.update(overrides)
    return defaults


def test_fixed_storage_rejects_epoch_length():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base(storage_epoch_length=10))


def test_demand_storage_requires_epoch_length_and_cap():
    with pytest.raises(ValidationError, match="storage_epoch_length"):
        ExperimentConfig(**_base(storage_mode="demand"))


def test_communication_requires_decentralised():
    with pytest.raises(ValidationError):
        ExperimentConfig(**_base(communication=True))


def test_observation_depth_required_for_every_controller():
    """d_obs bounds the field of view, and every architecture reads it
    (sec:pf:env, sec:pf:controllers), not only the decentralised one."""
    base = _base()
    del base["observation_depth"]
    with pytest.raises(ValidationError, match="observation_depth"):
        ExperimentConfig(**base)


def _reward_kwargs(**overrides) -> dict:
    defaults = dict(
        controller="decentralised",
        planner=None,
        discount=0.99,
        deliver_reward=1.0,
        override_penalty=0.5,
        congestion_reward_weight=0.1,
    )
    defaults.update(overrides)
    return defaults


def test_decentralised_controller_requires_reward_parameters():
    with pytest.raises(ValidationError, match="discount"):
        ExperimentConfig(**_base(controller="decentralised", planner=None))


def test_decentralised_controller_with_reward_parameters_is_valid():
    config = ExperimentConfig(**_base(**_reward_kwargs()))
    assert config.discount == 0.99


@pytest.mark.parametrize("controller", ["centralised", "section", "decentralised"])
@pytest.mark.parametrize(
    "missing", ["discount", "deliver_reward", "override_penalty", "congestion_reward_weight"]
)
def test_every_controller_requires_every_reward_parameter(controller, missing):
    """All three architectures learn from the same reward (sec:method:rl)."""
    kwargs = _reward_kwargs(controller=controller)
    del kwargs[missing]
    with pytest.raises(ValidationError, match=missing):
        ExperimentConfig(**_base(**kwargs))


@pytest.mark.parametrize("controller", ["centralised", "section", "decentralised"])
def test_every_controller_with_reward_parameters_is_valid(controller):
    config = ExperimentConfig(**_base(**_reward_kwargs(controller=controller)))
    assert config.congestion_reward_weight == 0.1


def test_reference_planner_does_not_require_reward_parameters():
    config = ExperimentConfig(**_base())
    assert config.discount is None


def test_communication_is_rejected_for_a_centralised_controller_with_reward_parameters():
    with pytest.raises(ValidationError, match="communication"):
        ExperimentConfig(**_base(communication=True, **_reward_kwargs(controller="centralised")))


def test_exactly_one_of_controller_and_planner_is_required():
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentConfig(
            **_base(**_reward_kwargs(controller="section", planner="prioritised_astar"))
        )
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentConfig(**_base(planner=None))
    section = ExperimentConfig(**_base(**_reward_kwargs(controller="section")))
    assert section.planner is None


def test_discount_above_one_rejected():
    with pytest.raises(ValidationError, match="discount"):
        ExperimentConfig(**_base(**_reward_kwargs(discount=1.5)))


def test_discount_of_exactly_one_is_allowed():
    config = ExperimentConfig(**_base(**_reward_kwargs(discount=1.0)))
    assert config.discount == 1.0


def test_keep_up_threshold_must_not_exceed_one():
    with pytest.raises(ValueError):
        ExperimentConfig(**_base(keep_up_threshold=1.5))
    assert ExperimentConfig(**_base(keep_up_threshold=0.95)).keep_up_threshold == 0.95


def test_congestion_weight_zero_is_valid():
    """beta >= 0 (sec:method:storage): zero congestion weight is a valid,
    not just a positive, setting."""
    config = ExperimentConfig(
        **_base(
            storage_mode="congestion",
            storage_epoch_length=25,
            reassignment_cap=10,
            congestion_weight=0.0,
        )
    )
    assert config.congestion_weight == 0.0


def test_congestion_mode_still_requires_congestion_weight_set():
    with pytest.raises(ValidationError, match="congestion_weight"):
        ExperimentConfig(
            **_base(storage_mode="congestion", storage_epoch_length=25, reassignment_cap=10)
        )


def test_communication_is_rejected_for_the_full_view_setting():
    with pytest.raises(ValidationError, match="communication"):
        ExperimentConfig(**_base(communication=True, **_reward_kwargs(controller="full_view")))
