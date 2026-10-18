"""The `storage_rule` section of the training spec, and the config set of configs/medium."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.training.spec import (
    StorageRuleSettings,
    build_experiment_config,
    build_study,
    load_spec,
    override_spec,
)

ROOT = Path(__file__).resolve().parents[2]
RULE = StorageRuleSettings(epoch_length=100, reassignment_cap=48, congestion_weight=3.0)


@pytest.fixture(scope="module")
def spec():
    return load_spec(ROOT / "configs" / "decentralised.yaml")


def _with(spec, **changes):
    data = spec.model_dump()
    data.update(changes)
    return type(spec).model_validate(data)


def test_matched_training_under_an_adaptive_rule_needs_the_section(spec):
    with pytest.raises(ValidationError, match="storage_rule section"):
        _with(spec, regime="matched", evaluation_storage_mode="demand")


def test_matched_congestion_needs_the_weight(spec):
    section = RULE.model_copy(update={"congestion_weight": None})
    with pytest.raises(ValidationError, match="congestion_weight"):
        _with(spec, regime="matched", evaluation_storage_mode="congestion", storage_rule=section)
    # The demand rule does not use beta.
    _with(spec, regime="matched", evaluation_storage_mode="demand", storage_rule=section)


def test_matched_fixed_and_fixed_only_need_no_section(spec):
    assert spec.storage_rule is None
    _with(spec, regime="matched", evaluation_storage_mode="fixed")
    _with(spec, regime="fixed_only", evaluation_storage_mode="demand")


def test_the_section_has_no_defaults():
    # nu has none either. It may be left out because the arrival stock model takes no nu, and
    # an adaptive rule of the template model then refuses the section.
    assert StorageRuleSettings(epoch_length=100).reassignment_cap is None
    with pytest.raises(ValidationError):
        StorageRuleSettings(reassignment_cap=48)


def test_a_template_adaptive_rule_refuses_the_section_without_nu(spec):
    section = StorageRuleSettings(epoch_length=100)
    with pytest.raises(ValidationError, match="reassignment_cap"):
        _with(spec, regime="matched", evaluation_storage_mode="demand", storage_rule=section)
    with pytest.raises(ValueError, match="reassignment_cap"):
        build_experiment_config(_with(spec, storage_rule=section), storage_mode="demand")


def test_the_section_reaches_the_experiment_config_of_an_adaptive_mode(spec):
    with_rule = _with(spec, storage_rule=RULE)
    demand = build_experiment_config(with_rule, 5, "demand")
    assert (demand.storage_mode, demand.seed) == ("demand", 5)
    assert (demand.storage_epoch_length, demand.reassignment_cap) == (100, 48)
    assert demand.congestion_weight is None
    congestion = build_experiment_config(with_rule, 5, "congestion")
    assert (congestion.storage_epoch_length, congestion.reassignment_cap) == (100, 48)
    assert congestion.congestion_weight == 3.0


def test_the_matched_regime_builds_its_config_from_the_section(spec):
    matched = _with(spec, regime="matched", evaluation_storage_mode="congestion", storage_rule=RULE)
    config = build_experiment_config(matched)
    assert config.storage_mode == "congestion" and config.congestion_weight == 3.0


def test_fixed_ignores_the_section(spec):
    config = build_experiment_config(_with(spec, storage_rule=RULE))
    assert config.storage_mode == "fixed"
    assert (config.storage_epoch_length, config.reassignment_cap) == (None, None)
    assert config.congestion_weight is None
    explicit = build_experiment_config(_with(spec, storage_rule=RULE), storage_mode="fixed")
    assert explicit == config


def test_an_adaptive_mode_without_the_section_names_what_is_missing(spec):
    with pytest.raises(ValueError, match="storage_rule section"):
        build_experiment_config(spec, storage_mode="demand")
    section = RULE.model_copy(update={"congestion_weight": None})
    with pytest.raises(ValueError, match="congestion_weight"):
        build_experiment_config(_with(spec, storage_rule=section), storage_mode="congestion")


@pytest.mark.parametrize("arm", ["decentralised", "centralised", "section", "full_view"])
def test_every_medium_config_loads_and_builds_a_study(arm):
    spec = load_spec(ROOT / "configs" / "medium" / f"{arm}.yaml")
    assert spec.controller == arm
    assert (spec.regime, spec.evaluation_storage_mode) == ("fixed_only", "fixed")
    assert (spec.train_seed_count, spec.validation_seed_count, spec.test_seed_count) == (100, 2, 6)
    assert spec.storage_rule == RULE
    study = build_study(spec)
    assert study.config.controller == arm and study.config.storage_mode == "fixed"
    assert build_experiment_config(spec, storage_mode="congestion").congestion_weight == 3.0


def test_the_medium_configs_differ_only_in_the_controller():
    base = load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")
    for arm in ("centralised", "section", "full_view"):
        other = load_spec(ROOT / "configs" / "medium" / f"{arm}.yaml")
        assert other == override_spec(base, controller=arm)
