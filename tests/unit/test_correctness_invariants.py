"""The four correctness invariants (sec:impl:validation), as a standing
test suite, not a one-off check from development.

Per AGENTS.md, these are non-negotiable for *any* change to
`environment/`, `resolution/`, `storage/`, or `controllers/`: a failure
in any of these is a correctness bug, not a metric to tune around. This
file is the acceptance gate for M2 ("one episode runs end-to-end through
the real six-stage loop with zero collisions, feasible storage, correct
task-set disjointness, and bit-for-bit deterministic replay") and is run
by CI's unconditional `pytest` step (.github/workflows/ci.yml) on every
push and pull request, same as every other test -- a regression in any
of those four directories that breaks one of these fails the build, not
just a local run.

  1. no vertex or swap conflict on the realised trace
     (eq:vertexconflict--eq:swapconflict);
  2. every realised storage configuration lies in the feasible set
     (eq:feasiblestorage);
  3. every released task is a member of exactly one of Q_t, B_t, C_t at
     every timestep (eq:lifecycle);
  4. bitwise replay determinism under fixed seeds.

Parametrised over several seeds and instance sizes rather than a single
example, since a single passing run doesn't rule out a seed-dependent
regression (e.g. a resolution edge case that only a particular priority
permutation triggers).
"""

from __future__ import annotations

import pytest

import slap_mapd_coupling.controllers.centralised  # noqa: F401 -- registers "centralised"
import slap_mapd_coupling.storage.fixed  # noqa: F401 -- registers "fixed"
from slap_mapd_coupling.core.agents import is_collision_free
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.core.storage_state import StorageState, is_feasible, SkuType
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.instances.generator import GeneratorParams, generate_warehouse_graph

SEEDS = (0, 1, 7, 42, 12345)


def _graph(num_endpoints: int, seed: int) -> WarehouseGraph:
    # one_way_fraction=0.0: connectivity/well-formedness of generated
    # instances (including one-way segments) is instances/validation.py's
    # own concern (tests/unit/test_validation.py), not this suite's --
    # this file tests the six-stage loop's own invariants given an
    # already-valid instance, matching the convention every other
    # environment-level test in this repo already uses.
    params = GeneratorParams(
        num_aisles=3,
        aisle_length=4,
        num_cross_aisles=2,
        one_way_fraction=0.0,
        wait_cost=0.5,
        num_storage_vertices=6,
        num_delivery_vertices=2,
        num_endpoints=num_endpoints,
        seed=seed,
    )
    return generate_warehouse_graph(params)


def _config(seed: int, num_agents: int = 4, horizon: int = 40) -> ExperimentConfig:
    return ExperimentConfig(
        storage_mode="fixed",
        controller="centralised",
        congestion_sensitive=False,
        communication=False,
        num_agents=num_agents,
        arrival_rate=1.5,
        seed=seed,
        horizon=horizon,
        wait_cost=0.5,
    )


def _env(graph: WarehouseGraph, config: ExperimentConfig) -> WarehouseMAPDEnv:
    storage_vertices = [v for v, vertex in graph.vertices.items() if vertex.role.storage]
    skus = {"tea": SkuType(sku_id="tea", unit_capacity=1.0)}
    capacities = {v: 10.0 for v in storage_vertices}
    counts = {"tea": {storage_vertices[0]: 5, storage_vertices[1]: 5, storage_vertices[2]: 5}}
    return WarehouseMAPDEnv(graph, skus, counts, capacities, config)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_vertex_or_swap_conflicts_on_the_realised_trace(seed: int):
    graph = _graph(num_endpoints=5, seed=seed)
    env = _env(graph, _config(seed))
    env.reset()
    before = env.fleet
    for t in range(env.config.horizon):
        env.step()
        assert is_collision_free(before, env.fleet), f"collision at t={t}, seed={seed}"
        before = env.fleet


@pytest.mark.parametrize("seed", SEEDS)
def test_storage_stays_within_capacity_at_every_epoch_boundary(seed: int):
    graph = _graph(num_endpoints=5, seed=seed)
    env = _env(graph, _config(seed))
    env.reset()
    assert is_feasible(env.storage)
    for t in range(env.config.horizon):
        env.step()
        assert is_feasible(env.storage), f"infeasible storage at t={t}, seed={seed}"
        assert isinstance(env.storage, StorageState)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_released_task_has_exactly_one_lifecycle_status_at_every_t(seed: int):
    graph = _graph(num_endpoints=5, seed=seed)
    env = _env(graph, _config(seed, horizon=30))
    env.reset()
    for t in range(env.config.horizon):
        env.step()
        now = t + 1
        for task in env.tasks:
            if task.release_time > now:
                continue  # not released yet at this point in the trace
            # status() itself enforces exactly-one-of by returning a
            # single value (not a set) and raising for an unreleased
            # task -- this is the checkable form eq:lifecycle's interval
            # partition promises.
            status = task.status(now)
            assert status in ("waiting", "active", "completed")
            # Cross-check directly against the timestamps, independent
            # of Task.status()'s own implementation.
            if task.completion_time is not None and now >= task.completion_time:
                assert status == "completed"
            elif task.assignment_time is not None and now >= task.assignment_time:
                assert status == "active"
            else:
                assert status == "waiting"


@pytest.mark.parametrize("seed", SEEDS)
def test_replay_is_bit_for_bit_deterministic_under_a_fixed_seed(seed: int):
    def run() -> list:
        graph = _graph(num_endpoints=5, seed=seed)
        env = _env(graph, _config(seed, horizon=30))
        env.reset()
        trace = []
        for _ in range(env.config.horizon):
            obs, rewards, terminated, truncated, infos = env.step()
            trace.append(
                (
                    tuple(sorted(obs.items())),
                    tuple(sorted(rewards.items())),
                    tuple(sorted((k, v) for k, v in infos.items() if k != "__all__")),
                )
            )
        task_snapshot = tuple(
            (t.task_id, t.release_time, t.assignment_time, t.pickup_time, t.completion_time)
            for t in sorted(env.tasks, key=lambda t: t.task_id)
        )
        return trace, task_snapshot, env.log.edge_traversals, env.log.total_movement_cost

    first = run()
    second = run()
    assert first == second


def test_all_four_invariants_hold_simultaneously_on_one_longer_run():
    """A single, larger, denser run exercising all four checks together
    -- the closest thing to M2's literal "done when" acceptance test."""
    seed = 2024
    graph = _graph(num_endpoints=6, seed=seed)
    env = _env(graph, _config(seed, num_agents=6, horizon=60))
    env.reset()
    before = env.fleet
    for t in range(env.config.horizon):
        env.step()
        assert is_collision_free(before, env.fleet)
        assert is_feasible(env.storage)
        now = t + 1
        for task in env.tasks:
            if task.release_time <= now:
                assert task.status(now) in ("waiting", "active", "completed")
        before = env.fleet
    assert env.tasks  # sanity: the run actually generated tasks
    assert any(task.completion_time is not None for task in env.tasks)  # and completed some
