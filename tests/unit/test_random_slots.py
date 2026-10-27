"""The start of D22 (step 12 of the model revision): every SKU on its own storage vertex drawn
at random, with the starting stock S_k of the refill rule, so the start carries no demand
information."""

import math
from pathlib import Path

import pytest

from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_study,
    load_spec,
    refill_starting_stock,
    stock_sizing,
)

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"


def _spec(num_skus=20, seed=0) -> TrainingSpec:
    data = load_spec(FLEET_8).model_dump()
    data["storage"].update(
        initial_placement="random_slots",
        num_skus=num_skus,
        start_fill=None,
        sku_names=None,
        seed=seed,
    )
    return TrainingSpec.model_validate(data)


def _homes(study) -> dict:
    homes = {}
    for sku_id, per_vertex in study.counts.items():
        assert len(per_vertex) == 1
        homes[sku_id] = next(iter(per_vertex))
    return homes


def test_every_sku_starts_alone_on_one_slot_with_its_service_level_stock():
    spec = _spec()
    study = build_study(spec)
    homes = _homes(study)
    assert len(set(homes.values())) == len(homes) == 20
    expected = refill_starting_stock(list(study.skus), spec.demand, stock_sizing(spec))
    assert {k: sum(v.values()) for k, v in study.counts.items()} == expected


def test_the_capacity_holds_the_largest_starting_stock():
    study = build_study(_spec())
    largest = max(
        units * study.skus[k].unit_capacity
        for k, per_vertex in study.counts.items()
        for units in per_vertex.values()
    )
    assert set(study.capacities.values()) == {math.ceil(largest)}


def test_the_draw_follows_the_seed():
    assert _homes(build_study(_spec(seed=0))) == _homes(build_study(_spec(seed=0)))
    assert _homes(build_study(_spec(seed=0))) != _homes(build_study(_spec(seed=1)))


def test_more_skus_than_slots_is_rejected():
    with pytest.raises(ValueError, match="own storage vertex"):
        build_study(_spec(num_skus=10_000))
