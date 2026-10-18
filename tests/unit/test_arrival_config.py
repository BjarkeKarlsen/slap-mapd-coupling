"""The settings of the arrival stock model.

Under the arrival model an order takes a unit, stock arrives every Delta steps, and the storage
rule decides where it goes. This file covers its settings only: the stock fields of
ExperimentConfig, the `arrival` section of the training spec, and the sized starting stock
"demand_sized_blocks". The template model, today's, stays the default.
"""

import math
import statistics
from pathlib import Path

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.training.spec import (
    ArrivalSettings,
    DemandSettings,
    StockSizing,
    StorageRuleSettings,
    build_experiment_config,
    build_storage,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
ARRIVAL = ArrivalSettings(margin=1.0, receiving="delivery")
RULE = StorageRuleSettings(epoch_length=100, congestion_weight=3.0)
STOCK = dict(stock_model="arrival", stock_margin=1.0, stock_floor=8, stock_receiving="delivery")


def _config(**overrides) -> dict:
    fields = dict(
        storage_mode="fixed",
        planner="token_passing",
        congestion_sensitive=False,
        communication=False,
        num_agents=8,
        arrival_rate=0.3,
        seed=0,
        horizon=100,
        wait_cost=0.0,
        observation_depth=2,
    )
    fields.update(overrides)
    return fields


# ExperimentConfig


def test_the_template_model_is_the_default_and_takes_no_stock_settings():
    assert ExperimentConfig(**_config()).stock_model == "template"
    for name in ("stock_margin", "stock_floor", "stock_receiving"):
        with pytest.raises(ValidationError, match=name):
            ExperimentConfig(**_config(**{name: STOCK[name]}))


@pytest.mark.parametrize(
    "missing", ["stock_margin", "stock_floor", "stock_receiving", "storage_epoch_length"]
)
@pytest.mark.parametrize("mode", ["fixed", "demand"])
def test_the_arrival_model_needs_every_stock_setting_and_delta(mode, missing):
    fields = _config(storage_mode=mode, storage_epoch_length=100, **STOCK)
    ExperimentConfig(**fields)
    fields.pop(missing)
    with pytest.raises(ValidationError, match=missing):
        ExperimentConfig(**fields)


def test_the_arrival_model_relocates_nothing_so_it_takes_no_cap():
    with pytest.raises(ValidationError, match="reassignment_cap"):
        ExperimentConfig(
            **_config(storage_mode="demand", storage_epoch_length=100, reassignment_cap=48, **STOCK)
        )


def test_the_congestion_rule_needs_beta_under_the_arrival_model_too():
    fields = _config(storage_mode="congestion", storage_epoch_length=100, **STOCK)
    with pytest.raises(ValidationError, match="congestion_weight"):
        ExperimentConfig(**fields)
    ExperimentConfig(**fields, congestion_weight=1.0)


# The training spec


def _medium():
    return load_spec(ROOT / "configs" / "medium" / "decentralised.yaml")


def _with(spec, **changes):
    data = spec.model_dump()
    data.update(changes)
    return type(spec).model_validate(data)


def _arrival_spec(spec, **changes):
    storage = spec.storage.model_copy(update={"initial_placement": "demand_sized_blocks"})
    return _with(spec, storage=storage, storage_rule=RULE, arrival=ARRIVAL, **changes)


def test_the_arrival_section_has_no_defaults():
    with pytest.raises(ValidationError):
        ArrivalSettings(margin=1.0)
    with pytest.raises(ValidationError):
        ArrivalSettings(receiving="delivery")


def test_the_arrival_section_needs_delta_and_takes_no_cap():
    spec = _medium()
    with pytest.raises(ValidationError, match="storage_rule section"):
        _with(spec, storage_rule=None, arrival=ARRIVAL)
    with pytest.raises(ValidationError, match="reassignment_cap"):
        _with(spec, arrival=ARRIVAL)  # the medium configs set nu = 48
    _with(spec, storage_rule=RULE, arrival=ARRIVAL)


def test_the_sized_stock_needs_the_arrival_section():
    spec = _medium()
    storage = spec.storage.model_copy(update={"initial_placement": "demand_sized_blocks"})
    with pytest.raises(ValidationError, match="arrival section"):
        _with(spec, storage=storage)


def test_a_template_adaptive_mode_still_needs_the_cap():
    spec = _with(_medium(), storage_rule=RULE)
    with pytest.raises(ValueError, match="reassignment_cap"):
        build_experiment_config(spec, storage_mode="demand")


@pytest.mark.parametrize("mode", ["fixed", "demand", "congestion"])
def test_every_mode_of_the_arrival_model_takes_delta_and_the_stock_settings(mode):
    config = build_experiment_config(_arrival_spec(_medium()), 7, mode)
    assert config.storage_mode == mode and config.seed == 7
    assert config.stock_model == "arrival"
    assert (config.stock_margin, config.stock_floor, config.stock_receiving) == (
        1.0,
        8,
        "delivery",
    )
    assert (config.storage_epoch_length, config.reassignment_cap) == (100, None)
    assert config.congestion_weight == (3.0 if mode == "congestion" else None)


def test_the_arrival_model_trains_under_its_own_rules_too():
    spec = _arrival_spec(_medium(), regime="matched", evaluation_storage_mode="demand")
    config = build_experiment_config(spec)
    assert (config.storage_mode, config.stock_model) == ("demand", "arrival")


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs").glob("*.yaml")) + sorted((ROOT / "configs").glob("*/*.yaml"))
)
def test_every_shipped_config_stays_on_the_template_model(path):
    spec = load_spec(path)
    assert spec.arrival is None
    config = build_experiment_config(spec)
    assert config.stock_model == "template"
    assert (config.stock_margin, config.stock_floor, config.stock_receiving) == (None, None, None)


# The sized starting stock


def _reference_sized(graph, skus, floor, skew, rate, delta, margin):
    """The sized blocks as the arrival probe computed them: blocks of SKUs in rank order over
    the storage vertices by mean distance to the delivery vertices, each SKU sized to its
    expected orders over one period with the margin, at least `floor`. Rank 0 is the first SKU
    in id order, the ranking at t = 0. Written out here so the two cannot drift apart."""
    delivery = [v for v, vertex in graph.vertices.items() if vertex.role.delivery]
    storage = sorted(
        (v for v, vertex in graph.vertices.items() if vertex.role.storage),
        key=lambda v: (statistics.mean(graph.distance(v, d) for d in delivery), v),
    )
    ranked = sorted(skus)
    raw = [1.0 / (r + 1) ** skew for r in range(len(ranked))]
    weights = [w / sum(raw) for w in raw]
    block = math.ceil(len(ranked) / len(storage))
    return {
        sku_id: {
            storage[r // block]: max(floor, math.ceil((1 + margin) * rate * delta * weights[r]))
        }
        for r, sku_id in enumerate(ranked)
    }


@pytest.mark.parametrize("rate", [0.3, 0.6])
@pytest.mark.parametrize("margin", [0.5, 1.0, 2.0])
def test_the_sized_stock_matches_the_probe_formula(rate, margin):
    spec = _arrival_spec(_medium())
    graph = build_study(spec).instance.graph
    demand = spec.demand
    skus, _, counts = build_storage(spec.storage, graph, demand, StockSizing(rate, 100, margin))
    floor = spec.storage.initial_units
    assert counts == _reference_sized(graph, skus, floor, demand.skew, rate, 100, margin)


def test_the_sized_stock_keeps_the_blocks_and_never_drops_below_the_floor():
    spec = _arrival_spec(_medium())
    graph = build_study(spec).instance.graph
    sized_storage = spec.storage
    blocks_storage = spec.storage.model_copy(update={"initial_placement": "demand_blocks"})
    _, _, sized = build_storage(sized_storage, graph, spec.demand, StockSizing(0.6, 100, 1.0))
    _, _, blocks = build_storage(blocks_storage, graph, spec.demand)
    assert {k: list(v) for k, v in sized.items()} == {k: list(v) for k, v in blocks.items()}
    floor = spec.storage.initial_units
    units = [sum(v.values()) for v in sized.values()]
    assert min(units) == floor
    # Rank 0 at load 0.6 and margin 1: ceil(2 * 0.6 * 100 * w_0), w_0 = 1 / H_100 for skew 1.
    harmonic = sum(1.0 / r for r in range(1, 101))
    assert sized["sku_000"] == {next(iter(blocks["sku_000"])): math.ceil(120 / harmonic)}


def test_without_a_demand_section_every_sku_weighs_the_same():
    spec = _arrival_spec(_medium(), demand=None)
    graph = build_study(spec).instance.graph
    _, _, counts = build_storage(spec.storage, graph, None, StockSizing(0.6, 100, 9.0))
    # 10 * 0.6 * 100 / 100 SKUs = 6 per SKU, under the floor of 8.
    assert {sum(v.values()) for v in counts.values()} == {spec.storage.initial_units}


def test_the_sized_stock_needs_its_sizing():
    spec = _arrival_spec(_medium())
    graph = build_study(spec).instance.graph
    with pytest.raises(ValueError, match="sizing"):
        build_storage(spec.storage, graph, spec.demand)


def test_the_study_sizes_its_stock_at_the_spec_load_and_shares_it_across_rules():
    spec = _arrival_spec(_medium())
    study = build_study(spec)
    _, _, expected = build_storage(
        spec.storage, study.instance.graph, spec.demand, StockSizing(spec.arrival_rate, 100, 1.0)
    )
    assert study.counts == expected
    for mode in ("demand", "congestion"):
        other = build_study(_with(spec, evaluation_storage_mode=mode))
        assert other.counts == study.counts


def test_an_overfull_sized_block_raises():
    spec = _arrival_spec(_medium())
    graph = build_study(spec).instance.graph
    with pytest.raises(ValueError, match="vertex_capacity"):
        build_storage(spec.storage, graph, DemandSettings(skew=1.0), StockSizing(0.6, 100, 30.0))
