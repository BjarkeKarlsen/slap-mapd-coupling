"""The YAML training spec builds what example 04 builds, so `slap-mapd train` and the
example train the same run."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slap_mapd_coupling.instances.generator import GeneratorParams
from slap_mapd_coupling.training.spec import (
    build_experiment_config,
    build_storage,
    build_study,
    instance_params,
    load_spec,
    override_spec,
    with_budget,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "decentralised.yaml"


@pytest.fixture(scope="module")
def example_04():
    spec = importlib.util.spec_from_file_location(
        "_example_04_spec", ROOT / "examples" / "04_decentralised_training.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec():
    return load_spec(CONFIG)


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_every_arm_has_a_file_that_builds_a_study(controller):
    arm_spec = load_spec(ROOT / "configs" / f"{controller}.yaml")
    study = build_study(arm_spec)
    assert arm_spec.controller == study.config.controller == controller
    assert study.arm.pooling == ("none" if controller == "decentralised" else "mean")


def test_the_files_of_the_arms_differ_only_in_the_controller(spec):
    for controller in ("centralised", "section"):
        other = load_spec(ROOT / "configs" / f"{controller}.yaml")
        assert other == override_spec(spec, controller=controller)


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_the_example_reads_its_hyperparameters_from_the_file_of_its_arm(example_04, controller):
    arm_spec = load_spec(ROOT / "configs" / f"{controller}.yaml")
    assert example_04.build_hyperparameters(900, controller) == with_budget(arm_spec, 900).ppo
    assert example_04.spec_for(controller) == arm_spec
    assert example_04.ILLUSTRATIVE_BUDGET_ENV_STEPS == (
        example_04.spec_for("decentralised").ppo.training_budget_env_steps
    )


@pytest.mark.parametrize("controller", ["decentralised", "centralised", "section"])
def test_the_file_and_the_example_build_the_same_study(example_04, spec, controller):
    # The example keeps its own study constants (the campaign script pins some of them), so
    # this is what stops the two from drifting apart.
    study = build_study(load_spec(ROOT / "configs" / f"{controller}.yaml"))
    instance = example_04.build_generated_instance()
    skus, capacities, counts = example_04.build_storage(instance.graph)
    config = example_04.build_config("fixed_only", seed=0, controller=controller)

    assert study.instance == instance
    assert (study.skus, study.capacities, study.counts) == (skus, capacities, counts)
    assert study.config == config
    assert study.arm == example_04.build_arm(instance, controller, config.observation_depth)
    assert study.gnn_encoder == example_04.build_gnn_config(config, study.arm)
    assert study.seed_split == example_04.build_seed_split(0, spec.test_seed_count)


def test_the_regime_decides_the_storage_mode_of_training(spec):
    matched = override_spec(spec, regime="matched")
    assert build_experiment_config(matched).storage_mode == spec.evaluation_storage_mode
    assert build_experiment_config(spec).storage_mode == "fixed"


def test_an_override_changes_only_what_it_names(spec):
    changed = override_spec(
        spec, horizon=200, budget_env_steps=900, seed=7, checkpoint_every=2, keep_checkpoints=5
    )
    assert changed.horizon == 200
    assert changed.ppo.training_budget_env_steps == 900
    assert changed.experiment.seed == 7
    assert (changed.checkpoint.frequency, changed.checkpoint.num_to_keep) == (2, 5)
    assert override_spec(spec) == spec


def test_a_combination_the_arm_rejects_is_rejected_on_override(spec):
    # Communication belongs to the decentralised arm only.
    with pytest.raises(ValueError):
        build_study(override_spec(spec, controller="centralised", communication=True))


def test_a_misspelt_key_in_the_file_is_rejected(tmp_path, spec):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["ppo"]["learning_rate"] = data["ppo"].pop("lr")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_spec(path)


def test_a_file_that_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_spec(path)


def test_the_storage_has_many_skus_of_different_sizes_spread_over_the_storage_vertices(spec):
    study = build_study(spec)
    storage_vertices = {
        v for v, vertex in study.instance.graph.vertices.items() if vertex.role.storage
    }
    assert len(study.skus) == spec.storage.num_skus >= 20
    assert {sku.unit_capacity for sku in study.skus.values()} == set(spec.storage.unit_capacities)
    assert set(study.capacities) == storage_vertices
    assert set(study.capacities.values()) == {100.0}
    stocked = {v for per_vertex in study.counts.values() for v in per_vertex}
    assert stocked == storage_vertices
    # eq:feasiblestorage holds at the start.
    for v in storage_vertices:
        used = sum(
            study.skus[k].unit_capacity * per_vertex.get(v, 0)
            for k, per_vertex in study.counts.items()
        )
        assert used <= study.capacities[v]


def test_initial_counts_that_do_not_fit_the_capacity_are_rejected(spec):
    crowded = spec.storage.model_copy(update={"initial_units": 50})
    with pytest.raises(ValueError, match="do not fit"):
        build_storage(crowded, build_study(spec).instance.graph)


@pytest.mark.parametrize("num_agents", [8, 16, 32, 64])
def test_the_fleet_size_alone_gives_the_warehouse(num_agents):
    """sec:impl:instances, graph v2: 16 m walkable vertices, m parking, m/2 workstations
    (delivery) and m/2 docks (receiving), the slots following from the rack faces."""
    fleet = load_spec(ROOT / "configs" / f"fleet_{num_agents}" / "centralised.yaml")
    assert fleet.instance == {}
    params = instance_params(fleet)
    assert params.layout == "v2"
    assert params.num_transit_vertices == 16 * num_agents
    assert params.num_endpoints == num_agents
    assert params.num_storage_vertices is None
    assert params.num_delivery_vertices == params.num_receiving_vertices == num_agents // 2
    assert params.seed == 42
    assert params.one_way_fraction == 0.0


def test_an_instance_section_is_built_as_written(spec):
    """A config with an instance section keeps its own warehouse, not the fleet's."""
    params = instance_params(spec)
    assert params.model_dump(exclude={"wait_cost", "num_endpoints"}) == (
        GeneratorParams(**spec.instance, wait_cost=0.0, num_endpoints=1).model_dump(
            exclude={"wait_cost", "num_endpoints"}
        )
    )
    assert params.role_backbone
