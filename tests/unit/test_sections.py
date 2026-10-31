"""SKU sections served by neighbouring workstations ([A76], [A77]). Each section of SKUs has
workstations_per_section workstations on neighbouring rack faces, every workstation names its
section, an order goes to one workstation and draws its lines from that workstation's section,
and the storage rules aim a SKU at its section's workstations on average."""

from functools import cache
from pathlib import Path

import pytest

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.instances.generator import (
    GeneratedInstance,
    GeneratorParams,
    fleet_defaults,
    generate_instance,
    v2_stacked_shape,
)
from slap_mapd_coupling.instances.validation import check_connectivity, check_well_formedness
from slap_mapd_coupling.storage.arrival import _refill_scores, pick_distance
from slap_mapd_coupling.training.config import split_seeds
from slap_mapd_coupling.training.spec import (
    TrainingSpec,
    build_experiment_config,
    build_study,
    load_spec,
)

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "step12" / "fleet_8.yaml"
HORIZON = 600


@cache
def _instance(num_agents: int, per_section: int, seed: int = 42) -> GeneratedInstance:
    num_aisles, num_blocks = v2_stacked_shape(num_agents, 4)
    settings = {
        **fleet_defaults(num_agents),
        "num_aisles": num_aisles,
        "rack_rows": 4,
        "num_blocks": num_blocks,
        "lower_block": "irregular",
        "station_placement": "rack_faces",
        "dock_placement": "left_wall",
        "workstations_per_section": per_section,
        "seed": seed,
    }
    return generate_instance(GeneratorParams(**settings, wait_cost=0.0))


def _sections(instance: GeneratedInstance) -> dict:
    found: dict = {}
    for v, vertex in instance.graph.vertices.items():
        if vertex.role.delivery:
            found.setdefault(vertex.section, []).append(v)
    return found


@pytest.mark.parametrize("num_agents", (8, 16))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_each_section_has_its_workstations_side_by_side_named_by_the_first(num_agents, seed):
    instance = _instance(num_agents, 4, seed)
    sections = _sections(instance)
    assert len(sections) == num_agents // 2 // 4
    for name, workstations in sections.items():
        assert len(workstations) == 4
        assert name == min(workstations)
        graph = instance.graph
        assert max(graph.distance(a, b) for a in workstations for b in workstations) <= 12
    assert check_connectivity(instance.graph).ok
    assert check_well_formedness(instance.graph, num_agents).ok


def test_one_workstation_per_section_names_no_section():
    instance = _instance(8, 1)
    assert set(_sections(instance)) == {None}
    assert all(vertex.section is None for vertex in instance.graph.vertices.values())


def test_the_sections_must_be_whole_and_on_rack_faces():
    num_aisles, num_blocks = v2_stacked_shape(8, 4)
    settings = {
        **fleet_defaults(8),
        "num_aisles": num_aisles,
        "rack_rows": 4,
        "num_blocks": num_blocks,
        "lower_block": "irregular",
        "dock_placement": "left_wall",
        "seed": 0,
        "wait_cost": 0.0,
    }
    with pytest.raises(ValueError, match="rack_faces"):
        GeneratorParams(**settings, workstations_per_section=4)
    with pytest.raises(ValueError, match="whole sections"):
        GeneratorParams(**settings, station_placement="rack_faces", workstations_per_section=3)


def _spec(per_section: int) -> TrainingSpec:
    data = load_spec(FLEET_8).model_dump()
    data["instance"]["workstations_per_section"] = per_section
    return TrainingSpec.model_validate(data)


@cache
def _run(per_section: int, mode: str = "fixed"):
    spec = _spec(per_section)
    study = build_study(spec, prepare=False)
    seed = split_seeds(study.seed_split).test[0]
    played = study.pool.warehouse(seed)
    config = build_experiment_config(spec, seed, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing",
         "horizon": HORIZON}
    )  # fmt: skip
    env, step = _planner_run(
        played.instance.graph, played.skus, played.counts, played.capacities, config,
        "token_passing",
    )  # fmt: skip
    for _ in range(HORIZON):
        step()
    return env


def test_an_order_goes_to_one_workstation_and_takes_its_lines_from_that_section():
    env = _run(2)  # 4 workstations, 2 sections of 10 SKUs
    assert len(env._section_workstations) == 2
    assert all(len(w) == 2 for w in env._section_workstations.values())
    per_section: dict = {}
    for sku_id, section in env._sections.items():
        per_section[section] = per_section.get(section, 0) + 1
    assert sorted(per_section.values()) == [10, 10]
    picks = [task for task in env.tasks if task.kind == "pick"]
    assert picks
    by_order: dict = {}
    for task in picks:
        by_order.setdefault(task.order_id, set()).add(task.delivery_vertex)
        assert env._sections[task.sku] == env._section_of[task.delivery_vertex]
    assert all(len(workstations) == 1 for workstations in by_order.values())


def test_the_pick_leg_of_a_sku_is_the_mean_over_its_sections_workstations():
    env = _run(2)
    graph = env.graph
    sku_id, section = next(iter(env._sections.items()))
    workstations = env._section_workstations[section]
    order = _refill_scores(
        graph, None, {sku_id: 1.0}, {}, 10, sku_delivery={sku_id: workstations}
    )
    best = order(sku_id)[0]
    shelves = [v for v, x in graph.vertices.items() if x.role.storage]
    nearest = min(pick_distance(graph, v, list(workstations)) for v in shelves)
    assert pick_distance(graph, best, list(workstations)) == nearest
