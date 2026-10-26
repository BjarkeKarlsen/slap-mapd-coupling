"""The grid loader and the Ma et al. (2017) reproduction (benchmarks/ma2017.py)."""

from __future__ import annotations

import pytest

from slap_mapd_coupling.benchmarks.ma2017 import (
    MA2017_MAP,
    TaskListEnv,
    config_for,
    instance,
    run,
    task_sequence,
)
from slap_mapd_coupling.instances.grid import (
    BLOCKED,
    START,
    TASK_ENDPOINT,
    cell_of,
    cells,
    grid_graph,
    parse_map,
)
from slap_mapd_coupling.planners.driver import step_with_planner
from slap_mapd_coupling.planners.token_passing import TokenPassingPlanner


def test_map_matches_figure_4() -> None:
    rows = parse_map(MA2017_MAP)
    assert (len(rows), len(rows[0])) == (21, 35)
    assert len(cells(rows, TASK_ENDPOINT)) == 200
    assert len(cells(rows, START)) == 152
    assert sum(row.count(BLOCKED) for row in rows) == 100


def test_grid_graph_is_four_neighbour_and_two_way() -> None:
    rows = parse_map("e.r\n.@.\n...")
    graph = grid_graph(rows, starts=[2])
    assert len(graph.vertices) == 8
    assert graph.vertices[0].role.storage and graph.vertices[0].role.delivery
    assert graph.vertices[2].role.endpoint
    assert set(graph.out_neighbours(0)) == {1, 3}
    for edge in graph.edges:
        assert edge.target in graph.out_neighbours(edge.source)
        assert edge.source in graph.out_neighbours(edge.target)
    assert graph.distance(0, 8) == 4.0


def test_unused_start_cells_are_plain_cells() -> None:
    graph = instance(num_agents=10, seed=3)
    endpoints = [v for v, vertex in graph.vertices.items() if vertex.role.endpoint]
    assert len(endpoints) == 10
    assert len(graph.vertices) == 635


def test_rejects_start_outside_r_cells() -> None:
    with pytest.raises(ValueError):
        grid_graph(parse_map("e.r"), starts=[1])


@pytest.mark.parametrize("frequency, added", [(0.2, [0, 5, 10, 15]), (2.0, [0, 0, 1, 1])])
def test_tasks_added_in_order_at_the_frequency(frequency: float, added: list[int]) -> None:
    sequence = task_sequence(frequency, seed=0, num_tasks=4)
    assert [task.added for task in sequence] == added
    assert all(task.pickup != task.delivery for task in sequence)
    width = len(parse_map(MA2017_MAP)[0])
    for task in sequence:
        r, c = cell_of(task.pickup, width)
        assert parse_map(MA2017_MAP)[r][c] == TASK_ENDPOINT


def test_env_releases_the_sequence() -> None:
    graph = instance(num_agents=5, seed=0)
    sequence = task_sequence(0.5, seed=0, num_tasks=6)
    planner = TokenPassingPlanner()
    env = TaskListEnv(graph, sequence, config_for(5, 0.5, 0, 200), planner)
    assert len(env.tasks) == 1  # task 0 at t = 0
    step_with_planner(env, planner)
    step_with_planner(env, planner)
    assert [task.release_time for task in env.tasks] == [0, 2]


def test_short_run_finishes_every_task_without_divergence() -> None:
    result = run(num_agents=10, frequency=10.0, seed=0, horizon=3000)
    assert result.finished == 500
    assert result.makespan is not None
    assert result.num_diverged == 0
