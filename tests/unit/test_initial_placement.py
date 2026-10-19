"""The starting stock x_0 (sec:impl:instances).

"by_number" spreads the SKUs over the storage vertices by number, as every shipped config does.
"demand_blocks" orders the storage vertices by their mean distance to the delivery vertices and
fills them, nearest first, with blocks of SKUs in popularity rank at t = 0. The stock is built
once per study, so every storage rule starts from it.
"""

import statistics
from pathlib import Path

import pytest

import slap_mapd_coupling.storage.congestion  # noqa: F401 -- registers "congestion"
import slap_mapd_coupling.storage.demand  # noqa: F401 -- registers "demand"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import (
    Edge,
    Vertex,
    VertexId,
    VertexRole,
    WarehouseGraph,
)
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.training.spec import (
    DemandSettings,
    StorageSettings,
    build_storage,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
_DEMAND = DemandSettings(skew=1.0, rotation_period=200, rotation_shift=5)


def _line() -> WarehouseGraph:
    """A corridor of seven plain vertices 10 to 16, with a role vertex hanging off every one but
    the middle. The role vertices are numbered from the far end, so that vertex number, nearest
    delivery vertex and mean delivery distance give three different orders.

    corridor  10 11 12 13 14 15 16
    hanging    6  5  4  .  2  1  0
    role       D  S  S  .  D  S  S

    Two role vertices are 2 apart plus their distance along the corridor. Mean distance to the
    delivery vertices 6 and 2: vertex 5 and vertex 4 are at 4, vertex 1 at 5 and vertex 0 at 6.
    Ties by number give [4, 5, 1, 0]. By number it is [0, 1, 4, 5], and by the nearest delivery
    vertex [1, 5, 0, 4].
    """
    hanging = {
        10: (6, VertexRole(delivery=True)),
        11: (5, VertexRole(storage=True)),
        12: (4, VertexRole(storage=True)),
        14: (2, VertexRole(delivery=True)),
        15: (1, VertexRole(storage=True)),
        16: (0, VertexRole(storage=True)),
    }
    vertices = {v: Vertex(id=v) for v in range(10, 17)}
    vertices.update({v: Vertex(id=v, role=role) for v, role in hanging.values()})
    pairs = [(a, a + 1) for a in range(10, 16)] + [(a, v) for a, (v, _) in hanging.items()]
    edges = []
    for a, b in pairs:
        edges += [Edge(source=a, target=b, cost=1.0), Edge(source=b, target=a, cost=1.0)]
    return WarehouseGraph(vertices=vertices, edges=tuple(edges), wait_cost=0.0)


def _settings(placement: str, **overrides) -> StorageSettings:
    fields = dict(
        num_skus=7,
        unit_capacities=(1.0,),
        vertex_capacity=100.0,
        initial_units=3,
        initial_placement=placement,
    )
    fields.update(overrides)
    return StorageSettings(**fields)


def _reference_blocks(graph: WarehouseGraph, skus, counts) -> dict[str, dict[VertexId, int]]:
    """The blocks placement as the demand pilot computed it, with rank 0 at the first SKU in id
    order. Kept here as written there, so the two cannot drift apart."""
    delivery = [v for v, vertex in graph.vertices.items() if vertex.role.delivery]
    storage = sorted(
        (v for v, vertex in graph.vertices.items() if vertex.role.storage),
        key=lambda v: (statistics.mean(graph.distance(v, d) for d in delivery), v),
    )
    ranked = sorted(skus, key=lambda name: int(name.rsplit("_", 1)[1]))
    block = -(-len(ranked) // len(storage))
    return {
        sku_id: {storage[i // block]: sum(counts[sku_id].values())}
        for i, sku_id in enumerate(ranked)
    }


def test_demand_blocks_fill_the_vertices_nearest_the_delivery_vertices_first():
    _, _, counts = build_storage(_settings("demand_blocks"), _line())
    # Seven SKUs over four vertices, so blocks of two, in the order [4, 5, 1, 0].
    assert counts == {
        "sku_000": {4: 3},
        "sku_001": {4: 3},
        "sku_002": {5: 3},
        "sku_003": {5: 3},
        "sku_004": {1: 3},
        "sku_005": {1: 3},
        "sku_006": {0: 3},
    }


_NAMES = ("tea", "coffee", "mugs", "biscuits", "sugar", "milk", "spoons")


def test_sku_names_name_the_skus_and_their_order_is_the_ranking():
    # Not alphabetical, so the blocks follow the list order rather than the sorted names.
    skus, _, counts = build_storage(
        _settings("demand_blocks", sku_names=_NAMES), _line(), _DEMAND
    )
    assert list(skus) == list(_NAMES)
    assert counts == {
        "tea": {4: 3},
        "coffee": {4: 3},
        "mugs": {5: 3},
        "biscuits": {5: 3},
        "sugar": {1: 3},
        "milk": {1: 3},
        "spoons": {0: 3},
    }


def test_sku_names_need_one_distinct_name_per_sku():
    with pytest.raises(ValueError, match="sku_names"):
        _settings("by_number", sku_names=_NAMES[:3])
    with pytest.raises(ValueError, match="distinct"):
        _settings("by_number", sku_names=("tea",) * 7)


def test_by_number_spreads_the_skus_in_vertex_number_order():
    _, _, counts = build_storage(_settings("by_number"), _line())
    storage = [0, 1, 4, 5]
    assert counts == {f"sku_{i:03d}": {storage[i % 4]: 3} for i in range(7)}


def test_by_number_is_the_default():
    fields = _settings("by_number").model_dump()
    fields.pop("initial_placement")
    assert StorageSettings(**fields).initial_placement == "by_number"


def test_the_ranking_at_the_start_is_the_id_order_with_or_without_drift():
    graph = _line()
    without = build_storage(_settings("demand_blocks"), graph)
    with_drift = build_storage(_settings("demand_blocks"), graph, _DEMAND)
    assert without == with_drift


def test_demand_blocks_match_the_pilot_placement_on_a_generated_instance():
    spec = load_spec(ROOT / "configs" / "decentralised.yaml")
    graph = build_study(spec).instance.graph
    storage = spec.storage.model_copy(
        update={"num_skus": 30, "vertex_capacity": 1000.0, "initial_placement": "demand_blocks"}
    )
    skus, _, counts = build_storage(storage, graph, _DEMAND)
    assert counts == _reference_blocks(graph, skus, counts)
    assert list(counts) == list(skus)  # id order, as by_number keeps them


def test_a_vertex_the_blocks_overfill_raises():
    # Five SKUs over four vertices: by number at most 16 units of capacity land on one vertex,
    # while the block of sku_002 (size 2) and sku_003 (size 0.5) needs 20.
    settings = dict(num_skus=5, unit_capacities=(0.5, 1.0, 2.0), initial_units=8)
    graph = _line()
    build_storage(_settings("by_number", vertex_capacity=16.0, **settings), graph)
    with pytest.raises(ValueError, match="vertex_capacity"):
        build_storage(_settings("demand_blocks", vertex_capacity=16.0, **settings), graph)


@pytest.fixture(scope="module")
def blocks_spec():
    spec = load_spec(ROOT / "configs" / "decentralised.yaml")
    storage = spec.storage.model_copy(update={"initial_placement": "demand_blocks"})
    return spec.model_copy(update={"storage": storage, "demand": _DEMAND})


def test_the_study_holds_the_blocks_placement(blocks_spec):
    study = build_study(blocks_spec)
    _, _, expected = build_storage(blocks_spec.storage, study.instance.graph, _DEMAND)
    assert study.counts == expected
    by_number = build_storage(
        blocks_spec.storage.model_copy(update={"initial_placement": "by_number"}),
        study.instance.graph,
    )[2]
    assert study.counts != by_number


def test_every_storage_rule_starts_from_the_same_stock(blocks_spec):
    studies = {
        mode: build_study(blocks_spec.model_copy(update={"evaluation_storage_mode": mode}))
        for mode in ("fixed", "demand", "congestion")
    }
    counts = studies["fixed"].counts
    assert all(study.counts == counts for study in studies.values())

    study = studies["fixed"]
    rules = {
        "fixed": {},
        "demand": dict(storage_epoch_length=100, reassignment_cap=48),
        "congestion": dict(storage_epoch_length=100, reassignment_cap=48, congestion_weight=3.0),
    }
    for mode, parameters in rules.items():
        config = ExperimentConfig(
            storage_mode=mode,
            planner="prioritised_astar",
            congestion_sensitive=False,
            communication=False,
            num_agents=blocks_spec.experiment.num_agents,
            arrival_rate=0.25,
            seed=0,
            horizon=10,
            wait_cost=blocks_spec.experiment.wait_cost,
            observation_depth=2,
            **parameters,
        )
        env = WarehouseMAPDEnv(
            study.instance.graph, study.skus, study.counts, study.capacities, config
        )
        assert env.storage.counts == counts
