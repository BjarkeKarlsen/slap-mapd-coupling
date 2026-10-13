"""Runs a Planner from outside the simulator, one step at a time."""

from __future__ import annotations

import time

from slap_mapd_coupling.core.agents import AgentId
from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.planners.base import Planner


def plan_slots(planner: Planner, env: WarehouseMAPDEnv) -> dict[AgentId, int]:
    """The planner's action for every agent at the current timestep, as one slot each.

    The planner reads the state after stages 1 and 2, which is the state the env
    holds between steps.
    """
    actions = planner.plan(env.graph, env.fleet, env.tasks, env.t)
    locations = env.fleet.locations()
    return {
        agent_id: slot_for_action(env.graph, locations[agent_id], action)
        for agent_id, action in actions.items()
    }


def step_with_planner(
    env: WarehouseMAPDEnv, planner: Planner
) -> tuple[
    dict[AgentId, VertexId],
    dict[AgentId, float],
    dict[AgentId | str, bool],
    dict[AgentId | str, bool],
    dict[AgentId | str, dict],
]:
    """Plan, record the planning time on the log, and advance the env one step.

    Returns what `env.step` returns. The time covers the planning and the
    conversion to slots.
    """
    start = time.perf_counter()
    slots = plan_slots(planner, env)
    env.log.record_routing_runtime(time.perf_counter() - start)
    return env.step(slots)
