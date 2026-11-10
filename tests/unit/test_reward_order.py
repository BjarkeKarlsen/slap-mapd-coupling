"""[A79]: the shaping potential at t+1 is the one the robot sees at t+1, after the new tasks and
parking leaves are handed out. So the shaping term of every step is gamma * Phi(t+1) - Phi(t) with
both potentials read from the state the robot sees, and a new target is charged on the step it
is handed out."""

from pathlib import Path

import pytest

from slap_mapd_coupling.core.tasks import active_tasks_by_agent
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.reward_function import current_target, potential
from slap_mapd_coupling.environment.spaces import slot_for_action
from slap_mapd_coupling.training.spec import build_experiment_config, build_study, load_spec

ROOT = Path(__file__).resolve().parents[2]
FLEET_8 = ROOT / "configs" / "fleet_8" / "decentralised.yaml"


def _potentials(env: WarehouseMAPDEnv) -> dict[int, float]:
    active = active_tasks_by_agent(env.tasks, env.t)
    locations = env.fleet.locations()
    return {
        agent_id: potential(
            env.graph, locations[agent_id], active.get(agent_id), env.parking_targets.get(agent_id)
        )
        for agent_id in locations
    }


def _towards_target(env: WarehouseMAPDEnv) -> dict[int, int]:
    """Every robot steps to the neighbour nearest its target, waiting on a tie."""
    active = active_tasks_by_agent(env.tasks, env.t)
    slots = {}
    for agent_id, here in env.fleet.locations().items():
        target = current_target(here, active.get(agent_id), env.parking_targets.get(agent_id))
        best = min(
            env.graph.legal_actions(here),
            key=lambda a: (
                env.graph.distance(here if a.kind == "wait" else a.target, target),
                a.kind != "wait",
            ),
        )
        slots[agent_id] = slot_for_action(env.graph, here, best)
    return slots


def test_the_shaping_uses_the_potential_the_robot_sees_at_the_next_step():
    spec = load_spec(FLEET_8)
    study = build_study(spec)
    config = build_experiment_config(spec, 0, "fixed")
    env = WarehouseMAPDEnv(study.instance.graph, study.skus, study.counts, study.capacities, config)
    gamma = config.discount
    new_targets = 0
    for _ in range(300):
        before = _potentials(env)
        shaping_before = dict(env.log.return_shaping_by_agent)
        env.step(_towards_target(env))
        after = _potentials(env)
        for agent_id in before:
            paid = env.log.return_shaping_by_agent.get(agent_id, 0.0) - shaping_before.get(
                agent_id, 0.0
            )
            assert paid == pytest.approx(gamma * after[agent_id] - before[agent_id], abs=1e-9)
        # Count the steps where a robot's target moved, a new task or a parking leaf, so the
        # test is known to cover the case [A79] is about.
        new_targets += sum(
            1 for agent_id in before if after[agent_id] < before[agent_id] - 1.0 - 1e-9
        )
    assert new_targets > 0
