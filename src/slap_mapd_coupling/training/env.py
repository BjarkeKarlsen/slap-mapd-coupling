"""The RLlib MultiAgentEnv around the simulator core, for all three learned
controllers and the full-view setting (sec:method:rl, sec:method:training).

The core (environment/multi_agent_env.py, exposed as `core`) holds the state
and runs the six stages. This adapter builds observations, collects the robots'
slots and passes them to `core.step` once per timestep. Every arm has one RLlib
agent per robot, with id str(robot_id) and the action space Discrete(d_max + 1).

A timestep is split into turns. In one turn the adapter returns observations
only for the agents that choose in that turn, so RLlib asks the module only
for them. The decentralised router and the full-view setting have one turn
with every agent. The centralised router has one turn per agent, in the
priority order of the episode. The section-based router has turn k hold the
k-th agent, in priority order, of every zone that has at least k agents. The
zones are fixed at the start of the timestep by the zone of each agent's
vertex. An agent in turn k sees the vertices the earlier agents of its unit
have chosen (the claimed feature and the claim mask). The core steps once,
after the last turn, and `timestep_completed` is true on that sub-step only.
The sub-steps before it return no rewards, and RLlib adds the reward that
arrives with the last one to the action each agent took earlier.

When the episode ends, the last sub-step returns an observation for every
agent, truncated, with no claims. These observations only feed the value
bootstrap, since nobody has chosen at the cut-off.

The controller decides the window a robot's observation covers and whose
reward it is credited with.

- decentralised, the field of view.
- centralised, every movable vertex.
- full_view, every movable vertex too. It is the decentralised router with the
  whole graph in view and the pooled readout, not a controller in Pi.
- section, the zone that owns the robot's vertex together with its halo.
  Ownership follows the vertex, so the window changes on the step a robot
  enters another zone.

With pooling on, the pooled readout averages over the vertices the router
owns. These are every movable vertex for the centralised router and the
full-view setting, and the zone for the section-based router. The decentralised
readout does not pool, so pooling with that arm raises.

The credit follows the agents that choose together (the credit). Each robot is
credited with the mean reward of its group, the whole fleet for the centralised
arm and the robots in the same zone at the start of the step for the section
arm. The decentralised arm and the full-view setting credit each robot with its
own reward.

Every window is checked when the adapter is built, so a rollout never meets a
window above max_local_nodes or a move that leads outside the window. A
window is never truncated (#66).

Each observation build that feeds a decision is timed and recorded on the
core's log as routing time, one record per turn, so the mean decision runtime
of a learned arm covers producing the joint action and not only decoding slots.

Training episodes cycle through `train_seeds` in order, so every training
seed is used equally often. The thesis fixes the split, not the order.

The cycle of one env starts at `first_seed_index`, an implementation choice.
Several env runners each build their own env, and if every cycle started at
the first seed they would all draw the same scenario at the same time, so a
PPO batch would hold one scenario. The training config gives each runner a
different start (`first_seed_index` of training/config.py). The default 0 is the
single-process case, where the cycle starts at the first seed as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from slap_mapd_coupling.core.agents import AgentId
from slap_mapd_coupling.core.experiment_config import ControllerArchitecture, ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.observation import local_subgraph
from slap_mapd_coupling.environment.spaces import action_for_slot, action_space, max_out_degree
from slap_mapd_coupling.environment.zones import check_partition, halo
from slap_mapd_coupling.models.gnn_encoder import Pooling
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    check_slot_targets,
    check_windows,
    encode_all_observations,
    encoded_observation_space,
)

Zones = Sequence[frozenset[VertexId]]


@dataclass(frozen=True)
class ArmWindows:
    """For every movable vertex, what a robot standing there sees under the
    centralised or section arm. `window` is the window in encoding order,
    `pool` the vertices the router owns, and `group` the credit group."""

    window: Mapping[VertexId, tuple[VertexId, ...]]
    pool: Mapping[VertexId, frozenset[VertexId]]
    group: Mapping[VertexId, int]


def check_arm(
    controller: Optional[ControllerArchitecture],
    pooling: Pooling,
    zones: Optional[Zones],
) -> None:
    """Raises ValueError unless the controller, the pooling and the zones fit
    together. The adapter and build_ppo_config both call it,
    so the module and the observations never disagree on pooling."""
    if controller is None:
        raise ValueError("the adapter trains a learned controller, but the config names a planner.")
    if controller == "decentralised":
        if pooling != "none":
            raise ValueError(
                f"pooling={pooling!r} with the decentralised arm. Only the centralised and "
                "section-based routers use the pooled readout, so set the encoder's pooling "
                "to 'none'."
            )
    if controller == "section" and zones is None:
        raise ValueError("the section arm needs zones, a disjoint cover of the movable vertices.")
    if controller != "section" and zones is not None:
        raise ValueError(f"zones are only used by the section arm, not by {controller!r}.")


def _movable(graph: WarehouseGraph) -> frozenset[VertexId]:
    return frozenset(v for v, vertex in graph.vertices.items() if vertex.role.movable)


def _check_cover(graph: WarehouseGraph, windows: Mapping[VertexId, Sequence[VertexId]]) -> None:
    """Every movable vertex has a window, and every window holds movable
    vertices only. check_windows checks neither."""
    movable = _movable(graph)
    missing = sorted(movable - set(windows))
    if missing:
        raise ValueError(f"the movable vertices {missing} have no window.")
    for v, window in windows.items():
        outside = sorted(set(window) - movable)
        if outside:
            raise ValueError(f"the window of vertex {v} holds {outside}, which are not movable.")


def arm_windows(
    graph: WarehouseGraph,
    controller: ControllerArchitecture,
    observation_depth: int,
    encoding_config: LocalSubgraphEncodingConfig,
    zones: Optional[Zones],
) -> Optional[ArmWindows]:
    """The windows of the centralised, full_view or section arm, checked against the
    graph and the cap. None for the decentralised arm, whose window is the
    field of view, checked by check_slot_targets. The halo of each zone is
    computed once here."""
    if controller == "decentralised":
        check_slot_targets(graph, observation_depth, encoding_config)
        return None
    movable = _movable(graph)
    if controller in ("centralised", "full_view"):
        everything = tuple(sorted(movable))
        arm = ArmWindows(
            window={v: everything for v in movable},
            pool={v: movable for v in movable},
            group={v: 0 for v in movable},
        )
    else:
        if zones is None:
            raise ValueError("the section arm needs zones.")
        check_partition(graph, zones)
        window: dict[VertexId, tuple[VertexId, ...]] = {}
        pool: dict[VertexId, frozenset[VertexId]] = {}
        group: dict[VertexId, int] = {}
        for q, zone in enumerate(zones):
            zone_window = tuple(sorted(zone | halo(graph, zone, observation_depth)))
            for v in zone:
                window[v], pool[v], group[v] = zone_window, zone, q
        arm = ArmWindows(window=window, pool=pool, group=group)
    _check_cover(graph, arm.window)
    check_windows(graph, arm.window, encoding_config)
    return arm


def team_mean(
    rewards: Mapping[AgentId, float], groups: Mapping[AgentId, int]
) -> dict[AgentId, float]:
    """Each robot's reward replaced by the mean reward of the robots in its
    group."""
    members: dict[int, list[AgentId]] = {}
    for agent_id in sorted(groups):
        members.setdefault(groups[agent_id], []).append(agent_id)
    credited: dict[AgentId, float] = {}
    for agent_ids in members.values():
        mean = sum(rewards[a] for a in agent_ids) / len(agent_ids)
        credited.update({a: mean for a in agent_ids})
    return credited


class WarehouseMAPDMultiAgentEnv(MultiAgentEnv):
    """One episode of the core, exposed as an RLlib MultiAgentEnv for PPO
    training of any learned controller. See the module docstring."""

    def __init__(
        self,
        graph: WarehouseGraph,
        skus: Mapping[SkuId, SkuType],
        initial_storage_counts: Mapping[SkuId, Mapping[VertexId, int]],
        storage_capacities: Mapping[VertexId, float],
        config: ExperimentConfig,
        encoding_config: LocalSubgraphEncodingConfig,
        train_seeds: Sequence[int],
        *,
        pooling: Pooling,
        zones: Optional[Zones] = None,
        first_seed_index: int = 0,
    ) -> None:
        check_arm(config.controller, pooling, zones)
        assert config.controller is not None  # check_arm rejects a planner
        if not train_seeds:
            raise ValueError("train_seeds must hold at least one seed of the training split.")
        self._arm = arm_windows(
            graph, config.controller, config.observation_depth, encoding_config, zones
        )
        super().__init__()
        self.graph = graph
        self.controller: ControllerArchitecture = config.controller
        self._config = config
        self._encoding_config = encoding_config
        self._pooling = pooling
        self._d_max = max_out_degree(graph)
        self._train_seeds = tuple(train_seeds)
        # Where in the training cycle this env starts. Taken modulo the
        # number of seeds, so more runners than seeds wrap around.
        self._next_train_seed_index = first_seed_index % len(self._train_seeds)

        self.agents = self.possible_agents = [str(a) for a in range(1, config.num_agents + 1)]
        base_observation_space = encoded_observation_space(
            encoding_config, self._d_max, config.communication, pooling
        )
        per_agent_observation_space = masked_observation_space(
            base_observation_space, self._d_max + 1
        )
        per_agent_action_space = action_space(graph)
        self.observation_spaces = {aid: per_agent_observation_space for aid in self.agents}
        self.action_spaces = {aid: per_agent_action_space for aid in self.agents}

        self.core = WarehouseMAPDEnv(
            graph, skus, initial_storage_counts, storage_capacities, config
        )
        self.last_seed: Optional[int] = None  # the seed the most recent reset() actually used
        # True on the sub-step that ran the core, false on every other one.
        self.timestep_completed = False

    def window_of(self, agent_id: AgentId) -> tuple[VertexId, ...]:
        """The window robot `agent_id`'s observation covers at the current t."""
        location = self.core.fleet.locations()[agent_id]
        if self._arm is None:
            return local_subgraph(self.graph, location, self._config.observation_depth)
        return self._arm.window[location]

    # RLlib types reset() and step() loosely against gymnasium's single-agent
    # generics, so the overrides below ignore the signature check.
    def reset(  # type: ignore[override]
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        # RLlib calls reset(seed=None) at the start of every training
        # episode, so the seed comes from the cycle. An explicit seed is
        # honoured as it is.
        if seed is None:
            seed = self._train_seeds[self._next_train_seed_index % len(self._train_seeds)]
            self._next_train_seed_index += 1
        self.last_seed = seed
        super().reset(seed=seed, options=options)
        self.core.reset(seed=seed)
        self.timestep_completed = False
        self._start_timestep()
        acting = self._turns[0]
        return self._encode(acting), {str(aid): {} for aid in acting}

    def step(  # type: ignore[override]
        self, action_dict: Mapping[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        """One sub-step. `action_dict` must hold exactly the agents of the current turn."""
        turn = self._turns[self._turn]
        if set(action_dict) != {str(aid) for aid in turn}:
            raise ValueError(
                f"the agents of this turn are {sorted(turn)}, but actions came for "
                f"{sorted(action_dict)}."
            )
        locations = self.core.fleet.locations()
        for aid, slot in action_dict.items():
            agent_id = int(aid)
            action = action_for_slot(self.graph, locations[agent_id], int(slot))
            self._slots[agent_id] = int(slot)
            if action.kind == "wait":
                self._chosen[agent_id] = locations[agent_id]
            else:
                assert action.target is not None  # guaranteed by Action's own validator
                self._chosen[agent_id] = action.target
        self.timestep_completed = False

        if self._turn + 1 < len(self._turns):
            self._turn += 1
            acting = self._turns[self._turn]
            flags = {str(aid): False for aid in acting} | {"__all__": False}
            infos = {str(aid): {} for aid in acting} | {"__all__": {}}
            return self._encode(acting), {}, flags, dict(flags), infos

        _, rewards, terminated, truncated, infos = self.core.step(dict(sorted(self._slots.items())))
        if self.controller in ("centralised", "section"):
            rewards = team_mean(rewards, self._groups_at_start)
        self.timestep_completed = True
        str_rewards = {str(aid): value for aid, value in rewards.items()}
        if truncated["__all__"] or terminated["__all__"]:
            # Nobody has chosen at the cut-off, so the observations carry no claims.
            acting = sorted(self.core.fleet.locations())
            observations = self._encode(acting, claims=False)
            str_terminated = {str(aid): bool(terminated[aid]) for aid in acting}
            str_truncated = {str(aid): bool(truncated[aid]) for aid in acting}
            str_infos = {str(aid): infos[aid] for aid in acting}
        else:
            self._start_timestep()
            acting = self._turns[0]
            observations = self._encode(acting)
            str_terminated = {str(aid): False for aid in acting}
            str_truncated = {str(aid): False for aid in acting}
            str_infos = {str(aid): infos[aid] for aid in acting}
        str_terminated["__all__"] = bool(terminated["__all__"])
        str_truncated["__all__"] = bool(truncated["__all__"])
        str_infos["__all__"] = {}
        return observations, str_rewards, str_terminated, str_truncated, str_infos

    def _start_timestep(self) -> None:
        """Fixes the credit groups and the turns of the timestep the core is at."""
        self._groups_at_start = self._groups()
        self._turns = self._build_turns()
        self._turn = 0
        self._chosen: dict[AgentId, VertexId] = {}
        self._slots: dict[AgentId, int] = {}

    def _build_turns(self) -> list[list[AgentId]]:
        agents = sorted(self.core.fleet.locations())
        if self.controller in ("decentralised", "full_view"):
            return [agents]
        by_group: dict[int, list[AgentId]] = {}
        for agent_id in self.core.priority:
            by_group.setdefault(self._groups_at_start[agent_id], []).append(agent_id)
        queues = [by_group[group] for group in sorted(by_group)]
        return [
            [queue[k] for queue in queues if len(queue) > k]
            for k in range(max(len(queue) for queue in queues))
        ]

    def _groups(self) -> dict[AgentId, int]:
        """The credit group of every robot, the zone that owns its vertex.
        The centralised arm has one group."""
        locations = self.core.fleet.locations()
        if self._arm is None:
            return {agent_id: 0 for agent_id in locations}
        return {agent_id: self._arm.group[v] for agent_id, v in locations.items()}

    def _encode(self, acting: Sequence[AgentId], claims: bool = True) -> dict[str, Any]:
        """The observations of the agents in `acting`. With `claims`, an agent of the
        centralised or section router also sees the choices of the earlier agents of its
        unit. Those are the agents in the same group that have chosen in this timestep."""
        # One window and one pool for every robot, since the builder raises
        # if a robot is missing. The decentralised arm passes None, which is
        # the field of view.
        start = time.perf_counter()
        locations = self.core.fleet.locations()
        windows: Optional[dict[AgentId, tuple[VertexId, ...]]] = None
        pool_vertices: Optional[dict[AgentId, frozenset[VertexId]]] = None
        if self._arm is not None:
            windows = {agent_id: self._arm.window[v] for agent_id, v in locations.items()}
            if self._pooling != "none":
                pool_vertices = {agent_id: self._arm.pool[v] for agent_id, v in locations.items()}
        chosen: Optional[dict[AgentId, dict[AgentId, VertexId]]] = None
        if claims and self.controller in ("centralised", "section"):
            groups = self._groups_at_start
            chosen = {
                agent_id: {j: v for j, v in self._chosen.items() if groups[j] == groups[agent_id]}
                for agent_id in acting
            }
        encoded = encode_all_observations(
            self.graph,
            self.core.fleet,
            self.core.tasks,
            self.core.t,
            self._config,
            self._encoding_config,
            self._d_max,
            windows=windows,
            pooling=self._pooling,
            pool_vertices=pool_vertices,
            acting=acting,
            chosen=chosen,
        )
        observations = {str(agent_id): item.as_rllib_dict() for agent_id, item in encoded.items()}
        # The build is part of producing the joint action, so it counts as routing time
        # (sec:impl:cost). The observation after the last step feeds no decision.
        if self.core.t < self._config.horizon:
            self.core.log.record_routing_runtime(time.perf_counter() - start)
        return observations
