"""configs/medium_putaway, the 8-agent warehouse with robots carrying the put-away."""

from pathlib import Path

import pytest

from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.evaluation.evaluator import _planner_run
from slap_mapd_coupling.instances.validation import check_well_formedness
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[2]
PUTAWAY_CONFIGS = sorted((ROOT / "configs" / "medium_putaway").glob("*.yaml"))


def test_there_is_a_putaway_config_for_every_arrival_config():
    assert [path.name for path in PUTAWAY_CONFIGS] == sorted(
        path.name for path in (ROOT / "configs" / "medium_arrival").glob("*.yaml")
    )


@pytest.mark.parametrize("path", PUTAWAY_CONFIGS, ids=lambda path: path.stem)
def test_a_putaway_config_builds_a_study_with_its_receiving_vertices_on_row_0(path):
    spec = load_spec(path)
    study = build_study(spec)
    graph = study.instance.graph
    receiving = [v for v, vertex in graph.vertices.items() if vertex.role.receiving]
    delivery = [v for v, vertex in graph.vertices.items() if vertex.role.delivery]
    assert len(receiving) == len(delivery)
    assert all(study.instance.positions[v][1] == 0.0 for v in receiving)
    assert check_well_formedness(graph, spec.experiment.num_agents).ok
    assert study.config.stock_receiving == "receiving"
    assert study.config.stock_putaway == "robots"


@pytest.mark.parametrize("path", PUTAWAY_CONFIGS, ids=lambda path: path.stem)
def test_a_putaway_config_differs_from_its_arrival_namesake_in_the_putaway_only(path):
    putaway = load_spec(path).model_dump()
    arrival = load_spec(ROOT / "configs" / "medium_arrival" / path.name).model_dump()
    assert putaway["instance"].pop("num_receiving_vertices") == arrival["instance"][
        "num_delivery_vertices"
    ]
    assert arrival["instance"].get("num_receiving_vertices") is None
    assert (putaway["arrival"].pop("receiving"), arrival["arrival"].pop("receiving")) == (
        "receiving",
        "delivery",
    )
    assert (putaway["arrival"].pop("putaway"), arrival["arrival"].pop("putaway")) == (
        "robots",
        "instant",
    )
    assert putaway == arrival


@pytest.mark.parametrize("mode", ["fixed", "demand", "congestion"])
def test_token_passing_delivers_putaways_and_keeps_the_correctness_checks(mode):
    spec = load_spec(PUTAWAY_CONFIGS[0])
    study = build_study(spec)
    config = build_experiment_config(spec, 300_000, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing", "horizon": 300}
    )

    graph = study.instance.graph

    def episode():
        env, step = _planner_run(
            graph, study.skus, study.counts, study.capacities, config, "token_passing"
        )
        for _ in range(300):
            before = env.fleet
            step()
            assert is_collision_free(before, env.fleet)
            env._shelf_and_transit()  # the shelf plus the stock on its way fits its capacity
            assert all(task.status(env.t) for task in env.tasks)  # one lifecycle set each
        return env

    first, second = episode(), episode()
    assert any(task.kind == "putaway" and task.finish_time is not None for task in first.tasks)
    assert first.tasks == second.tasks and first.storage == second.storage
