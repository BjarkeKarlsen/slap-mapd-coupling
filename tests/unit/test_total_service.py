"""The total service time zeta^tot_T and its split (sec:pf:measures)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.multi_agent_env import OrderLine
from slap_mapd_coupling.evaluation.evaluator import _planner_run, _to_run_metrics, total_service
from slap_mapd_coupling.evaluation.metrics import RunMetrics
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[2]
# fig:timeline (thesis-progress/scripts/images/timeline.py): released, assigned, finished or
# None, and the blocked timesteps of each task, T = 20.
TIMELINE = [(5, 9, 14, []), (3, 3, 11, [6, 7]), (10, 12, 18, [15]), (11, 12, None, range(14, 20))]
HORIZON = 20


def _timeline_tasks():
    tasks, blocked = [], {}
    for task_id, (released, assigned, finished, ticks) in enumerate(TIMELINE, start=1):
        task = Task(
            task_id=task_id, release_time=released, pickup_vertex=0, delivery_vertex=1, sku="tea"
        ).assign(1, assigned)
        task = task.pick_up(assigned)
        if finished is not None:
            task = task.complete(finished)
        tasks.append(task)
        blocked[task_id] = len(list(ticks))
    return tasks, blocked


def test_the_timeline_example_totals_34_and_its_parts_add_up():
    tasks, blocked = _timeline_tasks()
    totals = total_service(tasks, blocked, [], HORIZON)
    assert totals["total_service_time"] == 9 + 8 + 8 + 9
    assert totals["total_wait_for_agent"] == 4 + 0 + 2 + 1
    assert totals["total_blocked_time"] == 0 + 2 + 1 + 6
    assert totals["total_travel_time"] == 5 + 6 + 5 + 2
    assert totals["total_service_time_finished"] == 9 + 8 + 8


def test_a_line_still_waiting_for_stock_adds_its_time_to_the_wait():
    line = OrderLine(HORIZON - 7, "tea", 1, order_id=1, missing=2)
    totals = total_service([], {}, [line], HORIZON)
    assert totals["total_service_time"] == totals["total_wait_for_agent"] == 7
    assert totals["total_service_time_finished"] == 0


def test_an_unassigned_task_waits_up_to_the_horizon():
    task = Task(task_id=1, release_time=12, pickup_vertex=0, delivery_vertex=1, sku="tea")
    assert total_service([task], {}, [], HORIZON)["total_wait_for_agent"] == 8


def _episode(folder, mode="fixed", horizon=300):
    spec = load_spec(ROOT / "configs" / folder / "decentralised.yaml")
    study = build_study(spec)
    config = build_experiment_config(spec, 300_000, mode)
    config = ExperimentConfig.model_validate(
        {**config.model_dump(), "controller": None, "planner": "token_passing", "horizon": horizon}
    )
    env, step = _planner_run(
        study.instance.graph, study.skus, study.counts, study.capacities, config, "token_passing"
    )
    for _ in range(horizon):
        step()
    return env, _to_run_metrics(env, config)


def test_with_no_line_waiting_the_total_is_the_area_under_the_backlog():
    env, metrics = _episode("medium")
    # B_t of the thesis, the tasks released by t and not finished by t
    area = sum(
        sum(1 for task in env.tasks if task.release_time <= t and task.status(t) != "completed")
        for t in range(metrics.horizon)
    )
    assert metrics.total_service_time == area
    assert metrics.total_service_time > metrics.total_service_time_finished


@pytest.mark.parametrize(
    "folder, mode, mean",
    [("medium", "fixed", 37.144), ("medium_arrival", "demand", 33.29365079365079)],
)
def test_the_mean_service_time_is_the_one_before_put_aways(folder, mode, mean):
    # The values of feat/cold-stock, where every task is a pick.
    _, metrics = _episode(folder, mode)
    assert metrics.mean_service_time == pytest.approx(mean)
    assert metrics.num_completed_putaways == 0 and metrics.mean_putaway_service_time is None


def test_under_robots_the_means_are_over_picks_and_the_putaways_are_counted_apart():
    env, metrics = _episode("medium_putaway", "demand")
    picks = [t for t in env.tasks if t.kind == "pick" and t.finish_time is not None]
    putaways = [t for t in env.tasks if t.kind == "putaway" and t.finish_time is not None]
    assert putaways
    assert metrics.num_completed_putaways == len(putaways)
    assert metrics.num_released_putaways == sum(1 for t in env.tasks if t.kind == "putaway")
    assert metrics.mean_service_time == pytest.approx(
        sum(t.service_time for t in picks) / len(picks)
    )
    assert metrics.mean_putaway_service_time == pytest.approx(
        sum(t.service_time for t in putaways) / len(putaways)
    )
    assert metrics.num_completed_tasks == len(picks) + len(putaways)


def test_the_four_totals_must_add_up():
    _, metrics = _episode("medium", horizon=100)
    values = metrics.model_dump()
    values["total_travel_time"] += 1
    with pytest.raises(ValidationError, match="total_service_time"):
        RunMetrics(**values)
