"""WarehouseMAPDEnv: the six-stage step loop (sec:method:overview).

At each timestep: (1) orders drawn from P_ord become new tasks, entering
Q_t; (2) task assignment (shared across all controllers) matches free
agents to tasks in Q_t; (3) routing produces one intended action u_i(t)
per agent, using whichever information the active architecture permits;
(4) the conflict-resolution operator revises the joint intended action
into one satisfying the vertex rule and the swap rule, flagging
overridden agents; (5) the environment applies the resolved joint action,
advancing l_i(t), the lifecycle sets (the task stages), and the log; (6) if
t is a storage epoch, the storage rule additionally applies
the storage update. A controller swap touches only stage 3 (routing) and a
storage-rule swap touches only stage 6 -- both go through their
respective registries (docs/controller_integration.md,
docs/storage_rule_integration.md), so this loop never changes when either
is added to.

Stages 1 and 2 run at the end of reset() (for t = 0) and at the end of
every step() that does not end the episode (for the new t). A robot
therefore sees the task it was assigned in the observation it acts on, and
the next step starts at stage 3. The order draws and the backlog values
are the same as when stages 1 and 2 opened the step.

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
(the decentralised observation and the vertex features) is environment/observation.py, M5 -- not
needed by the reference planner this milestone wires up, since it
plans from full state directly, not from o_i(t).

Under the arrival stock model (config.stock_model == "arrival", thesis wording pending) stage 1
works on orders, not tasks. An order is queued when it is released, and the queue is served in
release order: an order takes one unit of its SKU from the storage vertex holding it that is
nearest its delivery vertex (ties by vertex id), and only then becomes a task, with its release
time kept. An order whose SKU has no unit anywhere waits in the queue and counts in the backlog.
The SKU of an order is drawn without looking at stock, so the orders of a seed do not depend on
the storage rule or the controller. Stage 6 is then the arrival of new stock every Delta steps,
under every storage rule, F_fix included: the amount of each SKU is the same under every rule,
and the put-away rule of the storage mode decides which shelves it goes to, without moving stock
already on a shelf (storage/arrival.py). An order waiting for stock is served by stage 1 of the
next step, after the arrival.

step(actions) takes one action slot per robot (the action mask), decoded by
environment/spaces.py::action_for_slot, so the simulator has one step mode.
Whoever produces the slots, a learned policy or a planner driven from
planners/driver.py, times itself and records the time on the log.
The shared assignment rule of stage 2 can be replaced by an `Assigner`
(controllers/assignment.py), which a planner that picks its own tasks is.
"""

from __future__ import annotations

import bisect
import random
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, NamedTuple

import numpy as np

from slap_mapd_coupling.controllers.assignment import Assigner, assign_tasks
from slap_mapd_coupling.core.agents import AgentId, AgentState, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType, StorageState
from slap_mapd_coupling.core.tasks import Task, TaskId, active_tasks_by_agent
from slap_mapd_coupling.environment.observation import local_subgraph, occupancy_fraction
from slap_mapd_coupling.environment.reward_function import RewardTerms, reward_terms
from slap_mapd_coupling.environment.spaces import action_for_slot
from slap_mapd_coupling.environment.storage_estimates import WindowEstimator
from slap_mapd_coupling.resolution.conflict_resolution import (
    priority_permutation,
    resolve_conflicts,
)
from slap_mapd_coupling.storage.arrival import PutAwayRule, arrival_amounts, get_putaway_rule
from slap_mapd_coupling.storage.base import EdgeKey
from slap_mapd_coupling.storage.registry import get_storage_rule

# sec:impl:instances: the order generator is seeded separately from the
# policy and the environment. A large odd constant, offset from
# config.seed, gives an independent-enough reproducible stream without a
# dedicated ExperimentConfig field (see module docstring).
_ORDER_SEED_OFFSET = 1_000_000_007


class Order(NamedTuple):
    """An order of the arrival stock model that waits for a unit of its SKU. It becomes a task,
    with this release time, once a unit is taken for it."""

    release_time: int
    sku: SkuId
    delivery_vertex: VertexId


def popularity_ranking(
    sku_ids: list[SkuId], t: int, period: int | None, shift: int | None
) -> list[SkuId]:
    """The SKUs from most to least popular at step t, for the order distribution with a skew.

    Rank 0 is the most popular. The ranking starts as the SKUs in id order and rotates left by
    `shift` places every `period` steps, so at step t the SKU in position
    ((t // period) * shift) mod |K| of the id order is the most popular. Without a period the
    ranking never rotates.
    """
    ranked = sorted(sku_ids)
    if period is None or not ranked:
        return ranked
    assert shift is not None  # ExperimentConfig requires a shift with a period
    offset = ((t // period) * shift) % len(ranked)
    return ranked[offset:] + ranked[:offset]


def zipf_weights(count: int, skew: float) -> list[float]:
    """The popularity of ranks 0 to count - 1, the rank r weighing 1 / (r + 1)^skew, normalised
    to sum to 1. A skew of 0 gives every rank the same weight."""
    raw = [1.0 / (rank + 1) ** skew for rank in range(count)]
    total = sum(raw)
    return [w / total for w in raw]


def _running_sum(values: list[float]):
    """The running sums of `values`, the cumulative popularity over every SKU of the arrival
    stock model's order draw."""
    total = 0.0
    for value in values:
        total += value
        yield total


def popularity_cumulative(
    ranked: list[SkuId], skew: float, vertices_by_sku: Mapping[SkuId, list[VertexId]]
) -> tuple[list[SkuId], list[float]]:
    """The SKUs in stock, in rank order, and their cumulative popularity.

    Every SKU of `ranked` gets its Zipf weight from its rank. A SKU out of stock keeps its rank
    but loses its weight, as the stock condition requires, and the weights of the rest are
    renormalised. An order's SKU number u then picks the first SKU whose cumulative weight
    exceeds u.
    """
    weights = zipf_weights(len(ranked), skew)
    in_stock = [(sku_id, w) for sku_id, w in zip(ranked, weights) if vertices_by_sku[sku_id]]
    total = sum(w for _, w in in_stock)
    cumulative: list[float] = []
    running = 0.0
    for _, w in in_stock:
        running += w / total
        cumulative.append(running)
    return [sku_id for sku_id, _ in in_stock], cumulative


@dataclass
class EpisodeLog:
    """Everything sec:impl:logging/sec:impl:cost require accumulated
    online, during the step loop -- not recomputed at the horizon.
    evaluation/evaluator.py (#16) reads this off at the end of an
    episode to build one RunMetrics row; this class only accumulates,
    it does not itself compute throughput/service time/etc.

    edge_traversals is not part of the score (sec:pf:measures no longer
    reports traffic concentration or movement cost). It stays logged
    because the replay-determinism invariant compares it. The traversal
    estimate of the storage update is counted separately, over the last
    storage period only (environment/storage_estimates.py).
    """

    edge_traversals: dict[EdgeKey, int] = field(default_factory=dict)
    backlog_by_t: list[int] = field(default_factory=list)  # B_t, every timestep
    blocked_ticks_by_task: dict[TaskId, int] = field(default_factory=dict)  # omega_j(t) summed
    assignment_runtime_seconds: list[float] = field(default_factory=list)  # sec:impl:cost
    routing_runtime_seconds: list[float] = field(default_factory=list)  # set by whoever acts
    # sum_i delta_i(t) at every timestep t = 0..T-1 (the mean crowding), for
    # every controller and planner (observation_depth is required for all of them).
    crowding_sum_by_t: list[float] = field(default_factory=list)
    # nu_t, units relocated at each storage update (relocations per update).
    relocated_units_by_update: list[int] = field(default_factory=list)
    # The relocation measure of each storage update: the travel the moved units would
    # cost, in units x d_G. Relocation itself stays instant and free in the simulator.
    relocation_distance_by_update: list[float] = field(default_factory=list)
    # Arrival stock model only: the steps each task's order waited for a unit of its SKU,
    # from its release until a unit was taken for it. 0 for an order served at once.
    stock_wait_by_task: dict[TaskId, int] = field(default_factory=dict)
    # Arrival stock model only, one entry per arrival. The put-away measure: the travel of the
    # arriving units from the nearest receiving vertex to their shelves, in units x d_G.
    putaway_distance_by_update: list[float] = field(default_factory=list)
    # Arrival stock model only: the units that arrived at each arrival, placed or not.
    arrived_units_by_update: list[int] = field(default_factory=list)
    # Arrival stock model only: the arriving units that fit on no shelf at each arrival.
    unplaced_units_by_update: list[int] = field(default_factory=list)
    # Arrival stock model only: the units F_fix placed off their SKU's home shelf at each
    # arrival. Always 0 under the adaptive rules, which have no home shelf.
    overflow_units_by_update: list[int] = field(default_factory=list)
    # Arrival stock model only: the SKUs held at more than one storage vertex after each arrival.
    split_skus_by_update: list[int] = field(default_factory=list)
    # What each agent's episode return is made of, one running sum per term of the
    # reward (environment/reward_function.py::RewardTerms, each already signed, so the
    # six sums of an agent add up to its summed reward). overrides_by_agent counts the
    # steps in which the agent was overridden. All stay empty for a planner run, which
    # has no reward weights, and an agent missing from a dict has a sum of zero.
    return_shaping_by_agent: dict[AgentId, float] = field(default_factory=dict)
    return_delivery_by_agent: dict[AgentId, float] = field(default_factory=dict)
    return_override_by_agent: dict[AgentId, float] = field(default_factory=dict)
    return_move_by_agent: dict[AgentId, float] = field(default_factory=dict)
    return_wait_by_agent: dict[AgentId, float] = field(default_factory=dict)
    return_congestion_by_agent: dict[AgentId, float] = field(default_factory=dict)
    overrides_by_agent: dict[AgentId, int] = field(default_factory=dict)

    def record_reward_terms(self, agent_id: AgentId, terms: RewardTerms, overridden: bool) -> None:
        """Adds one step's reward terms of one agent to its running sums."""
        for sums, term in (
            (self.return_shaping_by_agent, terms.shaping),
            (self.return_delivery_by_agent, terms.delivery),
            (self.return_override_by_agent, terms.override),
            (self.return_move_by_agent, terms.move),
            (self.return_wait_by_agent, terms.wait),
            (self.return_congestion_by_agent, terms.congestion),
        ):
            sums[agent_id] = sums.get(agent_id, 0.0) + term
        self.overrides_by_agent[agent_id] = self.overrides_by_agent.get(agent_id, 0) + int(
            overridden
        )

    def record_edge(self, edge: EdgeKey) -> None:
        self.edge_traversals[edge] = self.edge_traversals.get(edge, 0) + 1

    def record_blocked_tick(self, task_id: TaskId) -> None:
        self.blocked_ticks_by_task[task_id] = self.blocked_ticks_by_task.get(task_id, 0) + 1

    def record_routing_runtime(self, seconds: float) -> None:
        """Whoever produces the actions times itself and records it here."""
        self.routing_runtime_seconds.append(seconds)


class WarehouseMAPDEnv:
    """One episode of the coupled SLAP/MAPD simulation, per ExperimentConfig."""

    def __init__(
        self,
        graph: WarehouseGraph,
        skus: Mapping[SkuId, SkuType],
        initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
        storage_capacities: Mapping[VertexId, float],
        config: ExperimentConfig,
        assigner: Assigner | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self._skus = dict(skus)
        self._storage_capacities = dict(storage_capacities)
        self._initial_counts = {k: dict(v) for k, v in initial_storage_counts.items()}
        self._assigner = assigner
        self._storage_rule = get_storage_rule(config.storage_mode)
        self._edge_cost = {(edge.source, edge.target): edge.cost for edge in graph.edges}
        # The order draw of the arrival model looks at every SKU, never at stock.
        self._sku_ids = sorted(self._skus)
        self._delivery_vertices = sorted(
            v for v, vertex in graph.vertices.items() if vertex.role.delivery
        )
        self._all_sku_cumulative = (
            None
            if config.demand_skew is None
            else list(_running_sum(zipf_weights(len(self._sku_ids), config.demand_skew)))
        )
        # The arrival model's put-away rule, and each SKU's home shelf, which F_fix fills.
        self._putaway_rule: PutAwayRule | None = None
        self._home: dict[SkuId, VertexId] = {}
        if config.stock_model == "arrival":
            self._putaway_rule = get_putaway_rule(config.storage_mode)
            self._home = _home_shelves(self._initial_counts)
            if config.storage_mode == "fixed" and set(self._home) != set(self._skus):
                missing = sorted(set(self._skus) - set(self._home))
                raise ValueError(
                    f"SKUs {missing} start with no units, so F_fix has no home shelf for them."
                )

        self.fleet: FleetState
        self.tasks: list[Task]
        self.storage: StorageState
        self.log: EpisodeLog
        self._t: int
        self._next_task_id: int
        self._priority: tuple[AgentId, ...]
        self._order_rng: np.random.Generator
        self._estimator: WindowEstimator
        self._congestion_by_agent: dict[AgentId, float] = {}
        # What the last step proposed and what conflict resolution overrode,
        # read by the episode trace (evaluation/trace.py). Never fed back in.
        self.last_proposed: dict[AgentId, Action] = {}
        self.last_overridden: frozenset[AgentId] = frozenset()
        self.reset()

    @property
    def congestion_by_agent(self) -> Mapping[AgentId, float]:
        """delta_i(t) of every robot at the current t, the congestion feature read over the
        field of view. Computed once per timestep. After the step that ends the episode it
        still holds the values of the last timestep acted in."""
        return MappingProxyType(self._congestion_by_agent)

    @property
    def priority(self) -> tuple[AgentId, ...]:
        """The agent order drawn at reset and held for the episode."""
        return self._priority

    @property
    def t(self) -> int:
        """Current timestep -- exposed read-only for callers building o_i(t)
        (environment/observation.py) alongside self.fleet/self.tasks
        (training/rllib_env.py, #30)."""
        return self._t

    @property
    def waiting_orders(self) -> tuple[Order, ...]:
        """The orders of the arrival stock model still waiting for a unit of their SKU, in
        release order. They are released but not yet tasks. Always empty under the template
        model."""
        return tuple(self._waiting_orders)

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
        self._waiting_orders: list[Order] = []
        self.storage = StorageState(
            skus=self._skus,
            capacities=self._storage_capacities,
            counts=self._initial_counts,
        )
        self._t = 0
        self._next_task_id = 1
        self._priority = priority_permutation(episode_seed, agent_ids)
        self._order_rng = np.random.default_rng(episode_seed + _ORDER_SEED_OFFSET)
        self._estimator = WindowEstimator()
        self.last_proposed = {}
        self.last_overridden = frozenset()
        # An assigner that keeps state (Token Passing's committed paths) must not carry
        # it into the new episode, and must be reset before the assignment of t = 0.
        if self._assigner is not None:
            self._assigner.reset(episode_seed)

        self.log = EpisodeLog()
        self.log.backlog_by_t.append(self._backlog(self._t))
        self._release_and_assign()
        self._record_crowding()

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
        actions: Mapping[AgentId, int],
    ) -> tuple[
        dict[AgentId, VertexId],
        dict[AgentId, float],
        dict[AgentId | str, bool],
        dict[AgentId | str, bool],
        dict[AgentId | str, dict],
    ]:
        """Advance one timestep: stages 3 to 6, then stages 1 and 2 for the new t
        unless the episode ends. Returns
        (observations, rewards, terminated, truncated, infos), dict-keyed
        by agent id plus an "__all__" entry (RLlib/PettingZoo convention)
        signalling whether the whole episode is done.

        `actions` maps every agent id to a chosen slot (the action mask), decoded via
        environment/spaces.py::action_for_slot.
        """
        before_fleet = self.fleet
        congestion = self._congestion_by_agent  # delta_i(t), at the step the action is chosen in

        # Stages 1 and 2 already ran for this t, at the end of reset() or of the
        # previous step. Under the task lifecycle a task assigned at self._t is
        # already active at self._t, so the freshly assigned agent's action this
        # same tick still counts towards omega_j(t) and this timestep's reward.
        active_by_agent = active_tasks_by_agent(self.tasks, self._t)

        # Stage 3: routing (pi_route), the one stage a controller swap touches. The
        # slots come from outside, so this only decodes them.
        locations = self.fleet.locations()
        proposed: dict[AgentId, Action] = {
            agent_id: action_for_slot(self.graph, locations[agent_id], slot)
            for agent_id, slot in actions.items()
        }

        # Stage 4: conflict resolution.
        result = resolve_conflicts(self.graph, self.fleet, proposed, self._priority)
        self.last_proposed = dict(proposed)
        self.last_overridden = result.overridden

        # Stage 5: apply the resolved joint action; advance positions, log
        # traversals against the REALISED trace (not the proposal, which
        # may have been overridden), then lifecycle sets and backlog.
        realised_locations = result.fleet.locations()
        before_locations = before_fleet.locations()
        for agent_id, action in proposed.items():
            if action.kind != "move":
                continue
            assert action.target is not None
            edge = (before_locations[agent_id], action.target)
            if realised_locations[agent_id] == action.target:
                self.log.record_edge(edge)
                self._estimator.record_traversal(edge)

        self.fleet = result.fleet
        next_t = self._t + 1
        self.tasks = self._advance_task_lifecycle(next_t)

        # A robot serving a task that stood still, by its own choice or overridden, is a
        # blocked tick of its task and one wait on the next edge of its shortest route to
        # its goal (storage_estimates.py). Recorded here, before stage 6 closes the window.
        tasks_by_id = {task.task_id: task for task in self.tasks}
        for agent_id, task in active_by_agent.items():
            if before_locations[agent_id] == realised_locations[agent_id]:
                self.log.record_blocked_tick(task.task_id)  # omega_j(t): realised wait
                self._record_wait(realised_locations[agent_id], tasks_by_id[task.task_id])

        self.log.backlog_by_t.append(self._backlog(next_t))

        # Stage 6: storage update, only at a storage epoch (Delta = None
        # means F_fix's Delta = infinity: never triggers, by construction).
        delta = self.config.storage_epoch_length
        if delta is not None and next_t % delta == 0:
            if self.config.stock_model == "arrival":
                self._arrive(delta)
            else:
                self._update_storage(delta)

        rewards = self._compute_rewards(
            before_locations, active_by_agent, result.overridden, next_t, congestion
        )

        self._t = next_t
        episode_done = self._t >= self.config.horizon
        if not episode_done:
            self._release_and_assign()
            self._record_crowding()
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

    def _update_storage(self, delta: int) -> None:
        """Stage 6 under the template stock model: the storage update relocates stock."""
        # rho_hat_t/mu_hat_t/w_hat_t of the window that ends now, counted per step
        # over the last Delta steps only (environment/storage_estimates.py), then
        # a fresh window. Causal by construction: the order released at next_t
        # itself is only released after this stage.
        estimates = self._estimator.close_window(delta)
        before_storage = self.storage
        self.storage = self._storage_rule(
            self.storage,
            self.graph,
            self.config.reassignment_cap,
            self.config.congestion_weight,
            estimates.demand,
            estimates.traversal,
            estimates.waiting,
        )
        self.log.relocated_units_by_update.append(_relocated_units(before_storage, self.storage))
        self.log.relocation_distance_by_update.append(
            _relocation_distance(before_storage, self.storage, self.graph)
        )

    def _arrive(self, delta: int) -> None:
        """Stage 6 under the arrival stock model (module docstring): new stock arrives and the
        put-away rule places it. The estimates are those of the window that ends now, as in the
        template model, and every storage rule closes it, since Delta is the arrival period.
        Orders still waiting are served by stage 1 of the next step, not here."""
        assert self._putaway_rule is not None
        assert self.config.stock_margin is not None
        assert self.config.stock_floor is not None
        estimates = self._estimator.close_window(delta)
        waiting_by_sku: dict[SkuId, int] = {}
        for order in self._waiting_orders:
            waiting_by_sku[order.sku] = waiting_by_sku.get(order.sku, 0) + 1
        amounts = arrival_amounts(
            self.storage,
            estimates.demand,
            delta,
            self.config.stock_margin,
            self.config.stock_floor,
            waiting_by_sku,
        )
        result = self._putaway_rule(
            self.storage,
            self.graph,
            amounts,
            self._home,
            estimates.demand,
            estimates.traversal,
            estimates.waiting,
            self.config.congestion_weight,
        )
        self.storage = result.storage
        self.log.putaway_distance_by_update.append(result.putaway_distance)
        self.log.arrived_units_by_update.append(result.arrived_units)
        self.log.unplaced_units_by_update.append(result.unplaced_units)
        self.log.overflow_units_by_update.append(result.overflow_units)
        self.log.split_skus_by_update.append(_split_skus(self.storage))

    def _release_and_assign(self) -> None:
        """Stage 1 (orders become tasks entering Q_t) and stage 2 (the shared greedy
        assignment, or the assigner's own) for the current t."""
        if self.config.stock_model == "arrival":
            new_tasks = self._release_orders()  # records each order as it is released
        else:
            new_tasks = self._generate_tasks()
            for task in new_tasks:
                self._estimator.record_order(task.sku)
        self.tasks.extend(new_tasks)

        assign_start = time.perf_counter()
        if self._assigner is not None:
            assignments = self._assigner.assign(self.graph, self.fleet, self.tasks, self._t)
        else:
            assignments = assign_tasks(self.graph, self.fleet, self.tasks, self._t, self._priority)
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

    def _release_orders(self) -> list[Task]:
        """Stage 1 under the arrival stock model (module docstring): the orders released now
        join the queue, then the queue is served in release order.

        The order draw is the one of `_generate_tasks`, the arrival count and then three uniform
        numbers per order, with the SKU mapped over every SKU instead of the SKUs in stock. The
        number for the pickup vertex is drawn and not used, because the pickup is the nearest
        holder, so the stream stays the template model's. Every order is recorded in the demand
        estimate when it is released, whether or not a unit is taken for it then.
        """
        t = self._t
        num_new = int(self._order_rng.poisson(self.config.arrival_rate))
        if num_new:
            draws = self._order_rng.random((num_new, 3))
            ranked = popularity_ranking(
                self._sku_ids,
                t,
                self.config.demand_rotation_period,
                self.config.demand_rotation_shift,
            )
            for u_sku, _u_pickup, u_delivery in draws:
                if self._all_sku_cumulative is None:
                    sku_id = ranked[int(u_sku * len(ranked))]
                else:
                    index = bisect.bisect_right(self._all_sku_cumulative, u_sku)
                    sku_id = ranked[min(index, len(ranked) - 1)]
                delivery = self._delivery_vertices[int(u_delivery * len(self._delivery_vertices))]
                self._estimator.record_order(sku_id)
                self._waiting_orders.append(Order(t, sku_id, delivery))

        counts = {sku_id: dict(per_vertex) for sku_id, per_vertex in self.storage.counts.items()}
        still_waiting: list[Order] = []
        new_tasks: list[Task] = []
        for order in self._waiting_orders:
            per_vertex = counts.get(order.sku, {})
            holders = [v for v, count in per_vertex.items() if count > 0]
            if not holders:
                still_waiting.append(order)
                continue
            pickup = min(holders, key=lambda v: (self.graph.distance(v, order.delivery_vertex), v))
            per_vertex[pickup] -= 1
            task = Task(
                task_id=self._next_task_id,
                release_time=order.release_time,
                pickup_vertex=pickup,
                delivery_vertex=order.delivery_vertex,
                sku=order.sku,
            )
            self._next_task_id += 1
            self.log.stock_wait_by_task[task.task_id] = t - order.release_time
            new_tasks.append(task)
        self._waiting_orders = still_waiting
        if new_tasks:
            self.storage = StorageState(
                skus=self._skus, capacities=self._storage_capacities, counts=counts
            )
        return new_tasks

    def _generate_tasks(self) -> list[Task]:
        """Stage 1: P_ord -> new tasks (the task definition), entering Q_t.

        Flagged design choice (module docstring covers the seeding half):
        the thesis fixes only "expected tasks per timestep = lambda_task"
        and the stock condition. Implemented as Poisson(lambda)
        arrivals -- the standard queueing/MAPD convention that matches
        "expected number per timestep" exactly -- with the SKU drawn
        uniformly among the SKUs in stock, the pickup vertex uniformly among
        the vertices holding that SKU (which satisfies the stock condition by
        construction) and the delivery vertex uniformly from V_del.

        What the generator consumes does not depend on the storage
        configuration (sec:impl:instances: the same order stream replays
        under every controller and storage rule). Every step draws the
        arrival count and then three uniform numbers in [0, 1) per task, one
        each for the SKU, the pickup vertex and the delivery vertex, whatever
        is in stock. The numbers are mapped onto the current stock
        afterwards, so the stock only decides where an order points, never
        how many numbers are drawn. Choosing the SKU first, rather than a
        (SKU, vertex) pair, also keeps the demand for a SKU the same under
        every placement. A pair draw would favour a SKU that is spread over
        more vertices.

        With config.demand_skew set, the SKU is drawn by popularity instead
        (popularity_ranking, popularity_cumulative). The same numbers are
        drawn, and only the mapping of the SKU number changes, so the order
        stream of a seed still does not depend on the storage rule. Without a
        skew the uniform mapping above runs unchanged.
        """
        num_new = int(self._order_rng.poisson(self.config.arrival_rate))
        if num_new == 0:
            return []
        draws = self._order_rng.random((num_new, 3))

        vertices_by_sku = {
            sku_id: sorted(vertex for vertex, count in per_vertex.items() if count > 0)
            for sku_id, per_vertex in sorted(self.storage.counts.items())
        }
        cumulative: list[float] | None = None
        if self.config.demand_skew is None:
            in_stock = [sku_id for sku_id, vertices in vertices_by_sku.items() if vertices]
        else:
            ranked = popularity_ranking(
                list(vertices_by_sku),
                self._t,
                self.config.demand_rotation_period,
                self.config.demand_rotation_shift,
            )
            in_stock, cumulative = popularity_cumulative(
                ranked, self.config.demand_skew, vertices_by_sku
            )
        delivery_vertices = sorted(
            v for v, vertex in self.graph.vertices.items() if vertex.role.delivery
        )
        if not in_stock or not delivery_vertices:
            return []

        new_tasks: list[Task] = []
        for u_sku, u_pickup, u_delivery in draws:
            if cumulative is None:
                sku_id = in_stock[int(u_sku * len(in_stock))]
            else:
                index = min(bisect.bisect_right(cumulative, u_sku), len(in_stock) - 1)
                sku_id = in_stock[index]
            pickup_candidates = vertices_by_sku[sku_id]
            pickup_vertex = pickup_candidates[int(u_pickup * len(pickup_candidates))]
            delivery_vertex = delivery_vertices[int(u_delivery * len(delivery_vertices))]
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

    def _record_wait(self, at: VertexId, task: Task) -> None:
        """Charges one wait of the robot standing at `at` while serving `task` to the edge
        (at, w*) it is waiting to enter, the next edge of its shortest route to its goal.

        `task` is read after the lifecycle update, so the goal is the delivery vertex once
        the task is picked up and its pickup vertex before. w* is the out-neighbour w of
        `at` with the least c(at, w) + d_G(w, goal), ties broken by the lower vertex id. A
        robot already at its goal waits for no edge, so nothing is charged.
        """
        goal = task.current_goal
        if at == goal:
            return
        next_vertex = min(
            self.graph.out_neighbours(at),
            key=lambda w: (self._edge_cost[(at, w)] + self.graph.distance(w, goal), w),
        )
        self._estimator.record_wait((at, next_vertex))

    def _advance_task_lifecycle(self, next_t: int) -> list[Task]:
        """Marks pickup_time/finish_time (t_j^pickup/t_j^finish) against
        the REALISED (post-resolution) locations at next_t. The task
        stages' Q_t/B_t/C_t themselves are untouched by t_j^pickup
        (core/tasks.py's own invariant) -- only current_goal (q_i(t))
        depends on it."""
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

    def _record_crowding(self) -> None:
        """delta_i(t) of every robot for the current t, the congestion feature: the other
        robots inside a_i's field of view over the vertices available to hold them. Read
        over d_obs (config.observation_depth), required for every controller, so this runs
        every timestep. Stores the mapping and logs its sum, the mean crowding."""
        depth = self.config.observation_depth
        self._congestion_by_agent = {
            agent_id: occupancy_fraction(
                self.fleet, local_subgraph(self.graph, state.location, depth), excl={agent_id}
            )
            for agent_id, state in self.fleet.agents.items()
        }
        self.log.crowding_sum_by_t.append(sum(self._congestion_by_agent.values()))

    def _backlog(self, t: int) -> int:
        """B_t = |Q_t| + |B_t|, the backlog that the keep-up check plots. Under the arrival
        stock model an order still waiting for stock is released and unfinished, so it counts
        too."""
        unfinished = sum(1 for task in self.tasks if task.status(t) in ("waiting", "active"))
        return unfinished + len(self._waiting_orders)

    def _compute_rewards(
        self,
        before_locations: Mapping[AgentId, VertexId],
        active_by_agent: Mapping[AgentId, Task],
        overridden: frozenset[AgentId],
        next_t: int,
        congestion: Mapping[AgentId, float],
    ) -> dict[AgentId, float]:
        """R_i(t), the reward, for every robot. `congestion` holds delta_i(t) at the
        step the action was chosen in. A planner run has no reward weights and gets zero. Every
        other run also adds each robot's terms to the episode log."""
        if self.config.controller is None:
            return {aid: 0.0 for aid in self.fleet.agents}

        assert self.config.discount is not None
        assert self.config.deliver_reward is not None
        assert self.config.override_penalty is not None
        assert self.config.move_cost is not None
        assert self.config.congestion_reward_weight is not None

        after_locations = self.fleet.locations()
        after_tasks_by_agent = active_tasks_by_agent(self.tasks, next_t)
        rewards: dict[AgentId, float] = {}
        for agent_id in self.fleet.agents:
            task_now = active_by_agent.get(agent_id)
            task_next = after_tasks_by_agent.get(agent_id)
            completed = (
                task_now is not None
                and task_now.assigned_agent == agent_id
                and any(
                    t.task_id == task_now.task_id and t.finish_time == next_t for t in self.tasks
                )
            )
            was_overridden = agent_id in overridden
            moved = before_locations[agent_id] != after_locations[agent_id]
            # omega_i(t): serving a task and standing still, chosen or overridden.
            stood_still = task_now is not None and not moved
            terms = reward_terms(
                self.graph,
                location_now=before_locations[agent_id],
                task_now=task_now,
                location_next=after_locations[agent_id],
                task_next=task_next,
                completed_task=completed,
                overridden=was_overridden,
                moved=moved,
                stood_still=stood_still,
                congestion=congestion[agent_id],
                discount=self.config.discount,
                deliver_reward=self.config.deliver_reward,
                override_penalty=self.config.override_penalty,
                move_cost=self.config.move_cost,
                wait_cost=self.config.wait_cost,
                congestion_reward_weight=self.config.congestion_reward_weight,
            )
            self.log.record_reward_terms(agent_id, terms, was_overridden)
            rewards[agent_id] = terms.total
        return rewards


def _home_shelves(counts: Mapping[SkuId, Mapping[VertexId, int]]) -> dict[SkuId, VertexId]:
    """Each SKU's home shelf under the arrival stock model: the storage vertex holding most of
    its starting units, ties by vertex id. A SKU with no starting units has none."""
    home: dict[SkuId, VertexId] = {}
    for sku_id, per_vertex in counts.items():
        held = [(-count, v) for v, count in per_vertex.items() if count > 0]
        if held:
            home[sku_id] = min(held)[1]
    return home


def _split_skus(storage: StorageState) -> int:
    """The SKUs held at more than one storage vertex."""
    return sum(
        1
        for per_vertex in storage.counts.values()
        if sum(1 for count in per_vertex.values() if count > 0) > 1
    )


def _relocated_units(before: StorageState, after: StorageState) -> int:
    """nu_t (relocations per update): units that arrive at a (SKU, vertex) cell they
    were not at before. Counted the same way storage/relocation.py's cap
    counts a relocation, so nu_t <= nu by construction."""
    moved = 0
    for sku_id, per_vertex in after.counts.items():
        previous = before.counts.get(sku_id, {})
        for vertex, count in per_vertex.items():
            moved += max(0, count - previous.get(vertex, 0))
    return moved


def _relocation_distance(before: StorageState, after: StorageState, graph: WarehouseGraph) -> float:
    """The relocation measure of one storage update: the travel it would cost.

    Per SKU, the units that left a vertex are matched to the units that arrived at
    another, cheapest pair first by d_G(from, to) and ties by (from id, to id). The
    cost is the sum over matched units of units x d_G(from, to). Under fixed storage,
    or when nothing moved, it is 0.0.
    """
    total = 0.0
    for sku_id in before.counts.keys() | after.counts.keys():
        old = before.counts.get(sku_id, {})
        new = after.counts.get(sku_id, {})
        left = {v: old[v] - new.get(v, 0) for v in old if old[v] > new.get(v, 0)}
        arrived = {v: new[v] - old.get(v, 0) for v in new if new[v] > old.get(v, 0)}
        pairs = sorted((graph.distance(v, w), v, w) for v in left for w in arrived)
        for distance, v, w in pairs:
            units = min(left[v], arrived[w])
            if units > 0:
                total += units * distance
                left[v] -= units
                arrived[w] -= units
    return total
