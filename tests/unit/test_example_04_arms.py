"""Example 04 keeps the generated instance and sets each arm's window cap from the graph."""

import importlib.util
from pathlib import Path

import pytest

from slap_mapd_coupling.environment.zones import check_partition
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.env import arm_windows

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_04():
    return _load("04_decentralised_training.py", "_example_04_arms")


@pytest.fixture(scope="module")
def instance(example_04):
    return example_04.build_generated_instance()


def test_the_instance_is_the_generated_one_with_its_aisle_index(example_04, instance):
    assert len(instance.graph.vertices) == 51
    assert set(instance.aisle_index) == set(instance.graph.vertices)
    assert example_04.build_instance() == instance.graph


@pytest.mark.parametrize(
    "controller, cap, pooling",
    [("decentralised", 33, "none"), ("centralised", 51, "mean"), ("section", 44, "mean")],
)
def test_the_cap_is_the_largest_window_of_the_arm(example_04, instance, controller, cap, pooling):
    config = example_04.build_config("fixed_only", seed=0, controller=controller)
    arm = example_04.build_arm(instance, controller, config.observation_depth)
    assert (arm.encoding.max_local_nodes, arm.pooling) == (cap, pooling)
    # Every window fits, and one vertex less would truncate the largest.
    arm_windows(instance.graph, controller, config.observation_depth, arm.encoding, arm.zones)
    smaller = LocalSubgraphEncodingConfig(max_local_nodes=cap - 1, max_messages=2)
    with pytest.raises(ValueError, match="max_local_nodes"):
        arm_windows(instance.graph, controller, config.observation_depth, smaller, arm.zones)


def test_only_the_section_arm_has_zones_and_they_follow_the_aisles(example_04, instance):
    config = example_04.build_config("fixed_only", seed=0)
    assert example_04.build_arm(instance, "decentralised", 5).zones is None
    assert example_04.build_arm(instance, "centralised", 5).zones is None
    zones = example_04.build_arm(instance, "section", config.observation_depth).zones
    assert zones is not None and len(zones) == example_04.NUM_ZONES
    check_partition(instance.graph, zones)


def test_the_thin_variants_run_the_same_example_for_their_arm():
    assert _load("02_centralised_fixed.py", "_example_02").example_04.build_arm
    assert _load("03_section_based.py", "_example_03").example_04.build_arm


def test_a_trace_with_two_robots_on_one_vertex_is_not_collision_free(example_04):
    step = {"agents": {"1": {"at": 5}, "2": {"at": 5}}}
    assert not example_04.trace_is_collision_free([step], {"1": 4, "2": 6})


def test_a_trace_with_a_swap_is_not_collision_free(example_04):
    step = {"agents": {"1": {"at": 6}, "2": {"at": 4}}}
    assert not example_04.trace_is_collision_free([step], {"1": 4, "2": 6})


def test_a_trace_with_robots_following_each_other_is_collision_free(example_04):
    steps = [
        {"agents": {"1": {"at": 5}, "2": {"at": 6}}},
        {"agents": {"1": {"at": 6}, "2": {"at": 7}}},
    ]
    assert example_04.trace_is_collision_free(steps, {"1": 4, "2": 5})
