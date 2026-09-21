"""WarehouseMAPDEnv: the six-stage step loop (sec:method:overview).

At each timestep: (1) orders drawn from P_ord become new tasks, entering
Q_t; (2) task assignment (shared across all controllers) matches free
agents to tasks in Q_t; (3) routing produces one intended action u_i(t)
per agent, using whichever information the active architecture permits;
(4) the conflict-resolution operator revises the joint intended action
into one satisfying eq:vertexconflict/eq:swapconflict, flagging
overridden agents; (5) the environment applies the resolved joint action,
advancing l_i(t), the lifecycle sets (eq:lifecycle), and the log; (6) if
t is a storage epoch, the storage rule additionally applies
eq:storageupdate. A controller swap touches only stage 3 (routing) and a
storage-rule swap touches only stage 6 -- both go through their
respective registries (docs/controller_integration.md,
docs/storage_rule_integration.md), so this loop never changes when either
is added to.

Not a subclass of ray.rllib.MultiAgentEnv or gymnasium.Env: the
dict-keyed-by-agent-id obs/reward/terminated/truncated/info shape follows
those conventions (sec:impl:simulator says so explicitly), but importing
ray here would pull a heavy RL dependency into every run, including
baseline ones that train nothing. Wiring this to RLlib's actual
MultiAgentEnv base class is M5 scope (training/), same as the
action-masking connector (environment/spaces.py).

Two design choices flagged rather than guessed, since the thesis is
silent on both:
- Initial agent positions (l_i(0)): distinct vertices drawn from V_ep,
  the non-task endpoints the well-formedness condition (sec:pf:env)
  already designates as safe places for an agent to sit without blocking
  anyone -- the natural, already-motivated choice, not an arbitrary one.
- The order generator's random stream is seeded from config.seed offset
  by a fixed constant (_ORDER_SEED_OFFSET), not a fresh ExperimentConfig
  field: sec:impl:instances requires it "seeded separately from the
  policy and the environment," and a fixed deterministic offset already
  gives an independent, reproducible stream without growing the config
  surface for a property that only needs to hold internally.

Observations returned by reset/step are a placeholder for M2: each
agent's own current vertex. The full o_i(t) feature vector
(eq:observation/eq:features) is environment/observation.py, M5 -- not
needed by the centralised controller this milestone wires up, since it
plans from full state directly, not from o_i(t).

step() also accepts an optional `actions` mapping (issue #30): stage 3
normally asks the registered Controller to compute u_i(t) for every
agent from full state (`self._controller.route(...)`), but an RLlib
rollout worker training a decentralised policy needs the OPPOSITE
direction -- it observes o_i(t) itself (training/rllib_env.py) and
supplies each agent's chosen action slot externally, every step, while
gradients are live. Passing `actions` bypasses `route()` for that step
and decodes the given slots directly (environment/spaces.py::
action_for_slot); every other stage (assignment, conflict resolution,
lifecycle, storage, reward) is untouched, so this is still the same
six-stage loop, not a second one -- only stage 3's *source* differs.
The registered-Controller path (`actions=None`) stays the only one
centralised/section-based ever use, unaffected by this. Controller
resolution is therefore lazy (looked up on first `actions=None` step,
not in `__init__`): `config.controller="decentralised"` has no
registered Controller until #29 lands, but a run driven entirely by
`actions` never needs one.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from slap_mapd_coupling.controllers.assignment import assign_tasks
from slap_mapd_coupling.controllers.base import Controller
from slap_mapd_coupling.controllers.registry import get_controller
from slap_mapd_coupling.core.agents import AgentId, AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType, StorageState
from slap_mapd_coupling.core.tasks import Task, TaskId
from slap_mapd_coupling.environment.reward_function import reward as compute_agent_reward
from slap_mapd_coupling.environment.spaces import action_for_slot
from slap_mapd_coupling.resolution.conflict_resolution import (
    priority_permutation,
    resolve_conflicts,
)
from slap_mapd_coupling.storage.base import EdgeKey
from slap_mapd_coupling.storage.registry import get_storage_rule

# sec:impl:instances: the order generator is seeded separately from the
# policy and the environment. A large odd constant, offset from
# config.seed, gives an independent-enough reproducible stream without a
# dedicated ExperimentConfig field (see module docstring).
_ORDER_SEED_OFFSET = 1_000_000_007


@dataclass
class EpisodeLog:
    """Everything sec:impl:logging/sec:impl:cost require accumulated
    online, during the step loop -- not recomputed at the horizon.
    evaluation/evaluator.py (#16) reads this off at the end of an
    episode to build one RunMetrics row; this class only accumulates,
    it does not itself compute throughput/entropy/etc.
    """

    edge_traversals: dict[EdgeKey, int] = field(default_factory=dict)  # -> mu_T(e)
    backlog_by_t: list[int] = field(default_factory=list)  # B_t, every timestep
    blocked_ticks_by_task: dict[TaskId, int] = field(default_factory=dict)  # omega_j(t) summed
    assignment_runtime_seconds: list[float] = field(default_factory=list)  # sec:impl:cost
    routing_runtime_seconds: list[float] = field(default_factory=list)
    total_movement_cost: float = 0.0  # sum_i sum_t hat_c(l_i(t), l_i(t+1)), eq:movementcost

    def record_edge(self, edge: EdgeKey) -> None:
        self.edge_traversals[edge] = self.edge_traversals.get(edge, 0) + 1

    def record_blocked_tick(self, task_id: TaskId) -> None:
        self.blocked_ticks_by_task[task_id] = self.blocked_ticks_by_task.get(task_id, 0) + 1


def _active_tasks_by_agent(tasks: list[Task], t: int) -> dict[AgentId, Task]:
    active: dict[AgentId, Task] = {}
    for task in tasks:
        if task.status(t) == "active":
            assert task.assigned_agent is not None
            active[task.assigned_agent] = task
    return active


class WarehouseMAPDEnv:
    """One episode of the coupled SLAP/MAPD simulation, per ExperimentConfig."""

    def __init__(
        self,
        graph: WarehouseGraph,
        skus: Mapping[SkuId, SkuType],
        initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
        storage_capacities: Mapping[VertexId, float],
        config: ExperimentConfig,
    ) -> None:
        self.graph = graph
        self.config = config
        self._skus = dict(skus)
        self._storage_capacities = dict(storage_capacities)
        self._initial_counts = {k: dict(v) for k, v in initial_storage_counts.items()}
        # Lazy: resolved on first actions=None step, not here -- see
        # module docstring ("Controller resolution is therefore lazy").
        self._controller: Controller | None = None
        self._storage_rule = get_storage_rule(config.storage_mode)
        # eq:onestepcost's c(v,w): graph.out_neighbours only exposes
        # targets, not per-edge cost, so cache a lookup once rather than
        # linear-scanning graph.edges every step for eq:movementcost.
        self._edge_costs: dict[EdgeKey, float] = {(e.source, e.target): e.cost for e in graph.edges}

        self.fleet: FleetState
        self.tasks: list[Task]
        self.storage: StorageState
        self.log: EpisodeLog
        self._t: int
        self._next_task_id: int
        self._priority: tuple[AgentId, ...]
        self._order_rng: np.random.Generator
        self.reset()

    @property
    def t(self) -> int:
        """Current timestep -- exposed read-only for callers building o_i(t)
        (environment/observation.py) alongside self.fleet/self.tasks
        (training/rllib_env.py, #30)."""
        return self._t

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[AgentId, VertexId], dict[AgentId, dict]]:
        episode_seed = self.config.seed if seed is None else seed
        agent_ids = tuple(range(1, self.config.num_agents + 1))

        start_vertices = self._initial_agent_vertices(agent_ids, episode_seed)
        self.fleet = FleetState(
            agents={
                aid: AgentState(agent_id=aid, location=v)
                for aid, v in zip(agent_ids, start_vertices)
            }
        )
        self.tasks = []
        self.storage = StorageState(
            skus=self._skus,
            capacities=self._storage_capacities,
            counts=self._initial_counts,
        )
        self._t = 0
        self._next_task_id = 1
        self._priority = priority_permutation(episode_seed, agent_ids)
        self._order_rng = np.random.default_rng(episode_seed + _ORDER_SEED_OFFSET)

        self.log = EpisodeLog()
        self.log.backlog_by_t.append(self._backlog(self._t))

        observations = dict(self.fleet.locations())
        infos: dict[AgentId, dict] = {aid: {} for aid in agent_ids}
        return observations, infos

    def _initial_agent_vertices(self, agent_ids: tuple[AgentId, ...], seed: int) -> list[VertexId]:
        endpoints = sorted(v for v, vertex in self.graph.vertices.items() if vertex.role.endpoint)
        if len(endpoints) < len(agent_ids):
            raise ValueError(
                f"instance has {len(endpoints)} endpoints (V_ep) but {len(agent_ids)} agents; "
                "well-formedness (sec:pf:env) requires at least as many non-task endpoints as "
                "agents to place them on."
            )
        shuffled = endpoints[:]
        random.Random(seed).shuffle(shuffled)
        return shuffled[: len(agent_ids)]

    def step(
        self,
        actions: Mapping[AgentId, int] | None = None,
    ) -> tuple[
        dict[AgentId, VertexId],
        dict[AgentId, float],
        dict[AgentId | str, bool],
        dict[AgentId | str, bool],
        dict[AgentId | str, dict],
    ]:
        """Advance one timestep through all six stages. Returns
        (observations, rewards, terminated, truncated, infos), dict-keyed
        by agent id plus an "__all__" entry (RLlib/PettingZoo convention)
        signalling whether the whole episode is done.

        `actions`, if given, is a per-agent chosen slot index (eq:mask)
        decoded via environment/spaces.py::action_for_slot and used
        directly for stage 3 instead of asking the registered Controller
        -- see module docstring ("step() also accepts an optional
        `actions` mapping").
        """
        before_fleet = self.fleet

        # Stage 1: orders -> new tasks entering Q_t.
        self.tasks.extend(self._generate_tasks())

        # Stage 2: shared greedy task assignment (pi_assign).
        assign_start = time.perf_counter()
        assignments = assign_tasks(self.graph, self.fleet, self.tasks, self._t)
        self.log.assignment_runtime_seconds.append(time.perf_counter() - assign_start)
        if assignments:
            self.tasks = [
                (
                    task.assign(assignments[task.task_id], self._t)
                    if task.task_id in assignments
                    else task
                )
                for task in self.tasks
            ]

        # eq:lifecycle: a task assigned at self._t is already active AT
        # self._t, so this has to be taken after stage 2, not before --
        # the freshly assigned agent's action this same tick still counts
        # towards omega_j(t) and this timestep's reward.
        active_tasks_by_agent = _active_tasks_by_agent(self.tasks, self._t)

        # Stage 3: routing (pi_route) -- the one stage a controller swap
        # touches, and the one stage `actions` (if given) overrides the
        # source of, per the module docstring.
        route_start = time.perf_counter()
        if actions is not None:
            locations = self.fleet.locations()
            proposed: dict[AgentId, Action] = {
                agent_id: action_for_slot(self.graph, locations[agent_id], slot)
                for agent_id, slot in actions.items()
            }
        else:
            if self._controller is None:
                self._controller = get_controller(self.config.controller)
            proposed = self._controller.route(self.graph, self.fleet, self.tasks, self._t)
        self.log.routing_runtime_seconds.append(time.perf_counter() - route_start)

        # Stage 4: conflict resolution.
        result = resolve_conflicts(self.graph, self.fleet, proposed, self._priority)

        # Stage 5: apply the resolved joint action; advance positions, log
        # traversals against the REALISED trace (not the proposal, which
        # may have been overridden), then lifecycle sets and backlog.
        realised_locations = result.fleet.locations()
        before_locations = before_fleet.locations()
        for agent_id, action in proposed.items():
            if action.kind == "move" and realised_locations[agent_id] == action.target:
                self.log.record_edge((before_locations[agent_id], action.target))

        # eq:onestepcost's hat_c(l_i(t), l_i(t+1)), summed into
        # eq:movementcost -- against the REALISED transition for every
        # agent, not the proposal, so an overridden move is correctly
        # costed as a wait, not as the move that didn't happen.
        for agent_id, before_v in before_locations.items():
            after_v = realised_locations[agent_id]
            if before_v == after_v:
                self.log.total_movement_cost += self.graph.wait_cost
            else:
                self.log.total_movement_cost += self._edge_costs[(before_v, after_v)]

        self.fleet = result.fleet
        next_t = self._t + 1
        self.tasks = self._advance_task_lifecycle(next_t)

        for agent_id, task in active_tasks_by_agent.items():
            if before_locations[agent_id] == realised_locations[agent_id]:
                self.log.record_blocked_tick(task.task_id)  # omega_j(t): realised wait

        self.log.backlog_by_t.append(self._backlog(next_t))

        # Stage 6: storage update, only at a storage epoch (Delta = None
        # means F_fix's Delta = infinity: never triggers, by construction).
        delta = self.config.storage_epoch_length
        if delta is not None and next_t % delta == 0:
            # rho_hat_t/mu_hat_t/w_hat_t (demand/traversal/waiting
            # estimates): computing these online is F_dem/F_cng's own
            # scope (M3, #19/#20), not built yet. Empty estimates are
            # exactly correct for F_fix (which ignores them outright) and
            # a placeholder for any other rule registered under this
            # config -- flagged here, not silently assumed correct for
            # rules that actually read them.
            self.storage = self._storage_rule(
                self.storage, self.graph, self.config.reassignment_cap, {}, {}, {}
            )

        rewards = self._compute_rewards(
            before_locations, active_tasks_by_agent, result.overridden, next_t
        )

        self._t = next_t
        episode_done = self._t >= self.config.horizon
        observations = dict(self.fleet.locations())
        terminated: dict[AgentId | str, bool] = {aid: False for aid in self.fleet.agents}
        terminated["__all__"] = False
        truncated: dict[AgentId | str, bool] = {aid: episode_done for aid in self.fleet.agents}
        truncated["__all__"] = episode_done
        infos: dict[AgentId | str, dict] = {
            aid: {"overridden": aid in result.overridden} for aid in self.fleet.agents
        }
        infos["__all__"] = {}
        return observations, rewards, terminated, truncated, infos

    def _generate_tasks(self) -> list[Task]:
        """Stage 1: P_ord -> new tasks (eq:task), entering Q_t.

        Flagged design choice (module docstring covers the seeding half):
        the thesis fixes only "expected tasks per timestep = lambda_task"
        and eq:coupling's stock requirement. Implemented as Poisson(lambda)
        arrivals -- the standard queueing/MAPD convention that matches
        "expected number per timestep" exactly -- with (SKU, pickup
        vertex) drawn uniformly among pairs currently in stock (which
        satisfies eq:coupling by construction) and delivery vertex drawn
        uniformly from V_del.
        """
        num_new = int(self._order_rng.poisson(self.config.arrival_rate))
        if num_new == 0:
            return []

        candidates = [
            (sku_id, vertex)
            for sku_id, per_vertex in self.storage.counts.items()
            for vertex, count in per_vertex.items()
            if count > 0
        ]
        delivery_vertices = sorted(
            v for v, vertex in self.graph.vertices.items() if vertex.role.delivery
        )
        if not candidates or not delivery_vertices:
            return []

        new_tasks: list[Task] = []
        for _ in range(num_new):
            sku_id, pickup_vertex = candidates[self._order_rng.integers(len(candidates))]
            delivery_vertex = delivery_vertices[self._order_rng.integers(len(delivery_vertices))]
            new_tasks.append(
                Task(
                    task_id=self._next_task_id,
                    release_time=self._t,
                    pickup_vertex=pickup_vertex,
                    delivery_vertex=delivery_vertex,
                    sku=sku_id,
                )
            )
            self._next_task_id += 1
        return new_tasks

    def _advance_task_lifecycle(self, next_t: int) -> list[Task]:
        """Marks pickup_time/completion_time (p_j/d_j) against the
        REALISED (post-resolution) locations at next_t. eq:lifecycle's
        Q_t/B_t/C_t themselves are untouched by p_j (core/tasks.py's own
        invariant) -- only current_goal (q_i(t)) depends on it."""
        locations = self.fleet.locations()
        updated: list[Task] = []
        for task in self.tasks:
            if task.status(next_t) != "active":
                updated.append(task)
                continue
            assert task.assigned_agent is not None
            location = locations[task.assigned_agent]
            if task.pickup_time is None and location == task.pickup_vertex:
                task = task.pick_up(next_t)
            if task.pickup_time is not None and location == task.delivery_vertex:
                task = task.complete(next_t)
            updated.append(task)
        return updated

    def _backlog(self, t: int) -> int:
        """B_t = |Q_t| + |B_t| (eq:functional's backlog)."""
        return sum(1 for task in self.tasks if task.status(t) in ("waiting", "active"))

    def _compute_rewards(
        self,
        before_locations: Mapping[AgentId, VertexId],
        active_tasks_by_agent: Mapping[AgentId, Task],
        overridden: frozenset[AgentId],
        next_t: int,
    ) -> dict[AgentId, float]:
        """R_i(t) (eq:reward) -- only meaningful for a LEARNED controller
        (GAPS.tex's fix to eq:objective's scope): the four reward weights
        are None on ExperimentConfig for every controller except
        "decentralised" (core/experiment_config.py), so this returns an
        all-zero reward for the centralised/section-based baselines this
        milestone actually runs, rather than crashing on missing weights.

        `congestion` is hardcoded to 0.0 pending eq:congestion's delta_i(t)
        (environment/observation.py, M5) -- flagged, not silently treated
        as though congestion sensitivity were already wired in.
        """
        if self.config.controller != "decentralised":
            return {aid: 0.0 for aid in self.fleet.agents}

        assert self.config.discount is not None
        assert self.config.deliver_reward is not None
        assert self.config.override_penalty is not None
        assert self.config.congestion_reward_weight is not None

        after_locations = self.fleet.locations()
        after_tasks_by_agent = _active_tasks_by_agent(self.tasks, next_t)
        rewards: dict[AgentId, float] = {}
        for agent_id in self.fleet.agents:
            task_now = active_tasks_by_agent.get(agent_id)
            task_next = after_tasks_by_agent.get(agent_id)
            completed = (
                task_now is not None
                and task_now.assigned_agent == agent_id
                and any(
                    t.task_id == task_now.task_id and t.completion_time == next_t
                    for t in self.tasks
                )
            )
            rewards[agent_id] = compute_agent_reward(
                self.graph,
                location_now=before_locations[agent_id],
                task_now=task_now,
                location_next=after_locations[agent_id],
                task_next=task_next,
                completed_task=completed,
                overridden=agent_id in overridden,
                congestion=0.0,
                discount=self.config.discount,
                deliver_reward=self.config.deliver_reward,
                override_penalty=self.config.override_penalty,
                congestion_reward_weight=self.config.congestion_reward_weight,
            )
        return rewards
