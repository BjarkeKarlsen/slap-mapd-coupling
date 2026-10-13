"""The RLlib MultiAgentEnv around the simulator core, for all three learned
controllers (sec:method:rl, sec:method:training).

The core (environment/multi_agent_env.py, exposed as `core`) holds the state
and runs the six stages. This adapter builds one observation per robot,
passes the robots' slots to `core.step`, and reshapes the result for RLlib.
Every arm has one RLlib agent per robot, with id str(robot_id) and the action
space Discrete(d_max + 1).

The controller decides only the window a robot's observation covers.

- decentralised, the field of view.
- centralised, every movable vertex.
- section, the zone that owns the robot's vertex together with its halo.
  Ownership follows the vertex, so the window changes on the step a robot
  enters another zone.

With pooling on, the pooled readout averages over the vertices the router
owns. These are every movable vertex for the centralised router and the zone
for the section-based router. The decentralised readout does not pool, so
pooling with that arm raises.

The credit signal (tab:trainparams, #83) has no default. "per_agent" keeps
each robot's own reward. "team_mean" gives each robot the mean reward of its
group, the whole fleet for the centralised arm and the robots in the same
zone at the start of the step for the section arm. The decentralised arm
learns from each robot's own reward, so "team_mean" with it raises.

Every window is checked when the adapter is built, so a rollout never meets a
window above max_local_nodes or a move that leads outside the window. A
window is never truncated (#66).

Each observation build that feeds a decision is timed and recorded on the
core's log as routing time, so the mean decision runtime of a learned arm
covers producing the joint action and not only decoding slots.

Training episodes cycle through `train_seeds` in order, so every training
seed is used equally often. The thesis fixes the split, not the order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from slap_mapd_coupling.core.agents import AgentId
from slap_mapd_coupling.core.experiment_config import ControllerArchitecture, ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.environment.action_masking import masked_observation_space
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv
from slap_mapd_coupling.environment.observation import local_subgraph
from slap_mapd_coupling.environment.spaces import action_space, max_out_degree
from slap_mapd_coupling.environment.zones import check_partition, halo
from slap_mapd_coupling.models.gnn_encoder import Pooling
from slap_mapd_coupling.models.local_subgraph_encoding import (
    LocalSubgraphEncodingConfig,
    check_slot_targets,
    check_windows,
    encode_all_observations,
    encoded_observation_space,
)

# tab:trainparams, open for the centralised and section-based arms (#83).
CreditSignal = Literal["team_mean", "per_agent"]
Zones = Sequence[frozenset[VertexId]]


@dataclass(frozen=True)
class ArmWindows:
    """For every movable vertex, what a robot standing there sees under the
    centralised or section arm. `window` is the window in encoding order,
    `pool` the vertices the router owns, and `group` the team_mean group."""

    window: Mapping[VertexId, tuple[VertexId, ...]]
    pool: Mapping[VertexId, frozenset[VertexId]]
    group: Mapping[VertexId, int]


def check_arm(
    controller: Optional[ControllerArchitecture],
    credit_signal: CreditSignal,
    pooling: Pooling,
    zones: Optional[Zones],
) -> None:
    """Raises ValueError unless the controller, the credit signal, the pooling
    and the zones fit together. The adapter and build_ppo_config both call it,
    so the module and the observations never disagree on pooling."""
    if controller is None:
        raise ValueError("the adapter trains a learned controller, but the config names a planner.")
    if controller == "decentralised":
        if credit_signal == "team_mean":
            raise ValueError(
                "credit_signal='team_mean' does not apply to the decentralised arm, which learns "
                "from each robot's own reward. Use credit_signal='per_agent'."
            )
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
    """The windows of the centralised or section arm, checked against the
    graph and the cap. None for the decentralised arm, whose window is the
    field of view, checked by check_slot_targets. The halo of each zone is
    computed once here."""
    if controller == "decentralised":
        check_slot_targets(graph, observation_depth, encoding_config)
        return None
    movable = _movable(graph)
    if controller == "centralised":
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
        credit_signal: CreditSignal,
        pooling: Pooling,
        zones: Optional[Zones] = None,
    ) -> None:
        check_arm(config.controller, credit_signal, pooling, zones)
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
        self._credit_signal = credit_signal
        self._pooling = pooling
        self._d_max = max_out_degree(graph)
        self._train_seeds = tuple(train_seeds)
        self._next_train_seed_index = 0

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
        return self._encode_all_observations(), {str(aid): {} for aid in self.agents}

    def step(  # type: ignore[override]
        self, action_dict: Mapping[str, int]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        actions: dict[AgentId, int] = {int(aid): int(slot) for aid, slot in action_dict.items()}
        groups = self._groups()  # the team_mean groups at the start of the step
        _, rewards, terminated, truncated, infos = self.core.step(actions)
        if self._credit_signal == "team_mean":
            rewards = team_mean(rewards, groups)

        observations = self._encode_all_observations()
        str_rewards = {str(aid): value for aid, value in rewards.items()}
        str_terminated = {str(aid): value for aid, value in terminated.items() if aid != "__all__"}
        str_terminated["__all__"] = bool(terminated["__all__"])
        str_truncated = {str(aid): value for aid, value in truncated.items() if aid != "__all__"}
        str_truncated["__all__"] = bool(truncated["__all__"])
        str_infos = {str(aid): value for aid, value in infos.items() if aid != "__all__"}
        str_infos["__all__"] = {}

        return observations, str_rewards, str_terminated, str_truncated, str_infos

    def _groups(self) -> dict[AgentId, int]:
        """The team_mean group of every robot, the zone that owns its vertex.
        The centralised arm has one group."""
        locations = self.core.fleet.locations()
        if self._arm is None:
            return {agent_id: 0 for agent_id in locations}
        return {agent_id: self._arm.group[v] for agent_id, v in locations.items()}

    def _encode_all_observations(self) -> dict[str, Any]:
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
        )
        observations = {str(agent_id): item.as_rllib_dict() for agent_id, item in encoded.items()}
        # The build is part of producing the joint action, so it counts as routing time
        # (sec:impl:cost). The observation after the last step feeds no decision.
        if self.core.t < self._config.horizon:
            self.core.log.record_routing_runtime(time.perf_counter() - start)
        return observations
