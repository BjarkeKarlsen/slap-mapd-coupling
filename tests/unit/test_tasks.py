"""Unit tests for slap_mapd_coupling.core.tasks."""

import pytest

from slap_mapd_coupling.core.agents import AgentState, FleetState
from slap_mapd_coupling.core.tasks import Task, free_agents


def _task(**overrides) -> Task:
    defaults = dict(task_id=1, release_time=5, pickup_vertex=10, delivery_vertex=20, sku="tea")
    defaults.update(overrides)
    return Task(**defaults)


def test_task_status_waiting_before_assignment():
    task = _task()
    assert task.status(7) == "waiting"


def test_task_status_boundary_at_assignment_time():
    task = _task().assign(agent_id=2, t=9)
    assert task.status(8) == "waiting"
    assert task.status(9) == "active"


def test_task_status_boundary_at_completion_time():
    task = _task(release_time=5).assign(agent_id=2, t=9).complete(t=14)
    assert task.status(13) == "active"
    assert task.status(14) == "completed"


def test_task_status_before_release_raises():
    task = _task(release_time=5)
    with pytest.raises(ValueError):
        task.status(4)


def test_task_assign_sets_timestamps_and_is_permanent():
    task = _task()
    assigned = task.assign(agent_id=2, t=9)
    assert assigned.assigned_agent == 2
    assert assigned.assignment_time == 9
    with pytest.raises(ValueError):
        assigned.assign(agent_id=3, t=10)


def test_task_service_time_matches_worked_example():
    task = _task(release_time=5).assign(agent_id=2, t=9).complete(t=14)
    assert task.service_time == 9


def test_task_service_time_none_before_completion():
    task = _task().assign(agent_id=2, t=9)
    assert task.service_time is None


def test_free_agents_excludes_agents_with_active_tasks():
    fleet = FleetState(agents={aid: AgentState(agent_id=aid, location=0) for aid in (1, 2, 3)})
    task = _task().assign(agent_id=2, t=9)
    assert free_agents(fleet, [task], t=10) == {1, 3}
