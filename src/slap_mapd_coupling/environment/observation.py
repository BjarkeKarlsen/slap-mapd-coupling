"""o_i(t), the decentralised observation (sec:pf:observations).

Built from the field of view G_i^(d_obs)(t), the distance label eta_i(v,t)
on every visible vertex and the congestion feature delta_i(t). The result
is a structured Observation. Messages are not part of it, since they depend
on the network's weights (sec:method:model). Encoding the observation into
tensors is the job of models/. The field of view and the communication graph
use cost distance d_G.

A robot's visible vertices are its window, the field of view by default.
The centralised and section-based routers pass a wider window
(sec:method:controllers). The congestion feature stays on the field of
view whatever the window is, so it means the same under every router.

The vertex features use a multi-hot role (storage, delivery, endpoint,
transit), because VertexRole allows a vertex several roles. Transit is
true when none of the other three apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.environment.reward_function import current_target

# How the priority rank enters the observation. Undetermined (the thesis does
# not contain this signal), so it is a named choice.
# "normalised" is rank / (m - 1) in [0, 1], and 0 when m = 1. Illustrative.
# "raw" is the rank itself, 0 for the highest priority.
PriorityRankEncoding = Literal["normalised", "raw"]


class ConflictSignalsConfig(BaseModel):
    """The switch of the conflict signals, off by default, and how the rank is
    encoded. It is stored in GNNEncoderConfig, so a checkpoint carries it and
    can be evaluated with the observations it was trained on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    rank_encoding: PriorityRankEncoding = "normalised"  # illustrative, see PriorityRankEncoding


@dataclass(frozen=True)
class ConflictSignals:
    """What the environment knows about conflict resolution at this timestep.
    `priority` is sigma, the permutation that conflict resolution draws once per
    episode, with the highest priority first. `overridden` holds the robots whose
    move conflict resolution overrode at the step that just ended, and is empty
    at the start of an episode."""

    priority: tuple[AgentId, ...]
    overridden: frozenset[AgentId]
    rank_encoding: PriorityRankEncoding = "normalised"


def encode_priority_rank(rank: int, num_agents: int, encoding: PriorityRankEncoding) -> float:
    """`rank` is the position in sigma, 0 for the highest priority. "normalised"
    maps it to [0, 1] by rank / (m - 1), and gives 0 when m = 1."""
    if not 0 <= rank < num_agents:
        raise ValueError(f"rank {rank} is outside 0..{num_agents - 1}")
    if encoding == "raw":
        return float(rank)
    return rank / (num_agents - 1) if num_agents > 1 else 0.0


@dataclass(frozen=True)
class VertexFeatures:
    """f(v,t), the vertex features: the multi-hot role, the
    distance label eta_i(v,t), and whether another agent
    occupies v."""

    storage: bool
    delivery: bool
    endpoint: bool
    transit: bool  # true iff none of the above -- see module docstring
    eta: float
    occupied: bool


@dataclass(frozen=True)
class Observation:
    """o_i(t), the decentralised observation, structured
    rather than a fixed-size vector."""

    visible_vertices: tuple[VertexId, ...]  # the window, V_i^(d_obs)(t) by default
    features: dict[VertexId, VertexFeatures]
    congestion: float  # delta_i(t); always one of o_i(t)'s three pieces


@dataclass(frozen=True)
class ObservationWithConflictSignals(Observation):
    """o_i(t) plus the two conflict signals of ConflictSignalsConfig. A subclass, so
    with the switch off build_observation still returns the plain three-piece
    Observation of the thesis, and only the switch-on path meets this type."""

    overridden_last_step: bool  # conflict resolution overrode this robot's move last step
    priority_rank: float  # this robot's rank in sigma, encoded


def local_subgraph(graph: WarehouseGraph, location: VertexId, depth: float) -> tuple[VertexId, ...]:
    """V_i^(d_obs)(t), the vertices of the field of view, the movable
    vertices with cost distance d_G(l_i(t), v) <= d_obs."""
    return tuple(
        v
        for v, vertex in graph.vertices.items()
        if vertex.role.movable and graph.distance(location, v) <= depth
    )


def role_flags(role: VertexRole) -> tuple[bool, bool, bool, bool]:
    """(storage, delivery, endpoint, transit) -- see module docstring for
    why this is multi-hot, not a strict one-hot."""
    transit = not (role.storage or role.delivery or role.endpoint)
    return role.storage, role.delivery, role.endpoint, transit


def eta(graph: WarehouseGraph, vertex: VertexId, target: VertexId) -> float:
    """eta_i(v,t) = d_G(v, q_i(t)), the distance label, for
    any visible vertex v."""
    return graph.distance(vertex, target)


def occupancy_fraction(fleet: FleetState, window: Sequence[VertexId], excl: set[AgentId]) -> float:
    """Share of `window` occupied by agents outside `excl`. With the field
    of view as window and the agent itself excluded, this is the
    congestion feature delta_i(t)."""
    window_set = set(window)
    occupants = sum(
        1
        for agent_id, state in fleet.agents.items()
        if agent_id not in excl and state.location in window_set
    )
    denominator = max(1, len(window_set) - len(excl))
    return occupants / denominator


def communication_neighbours(
    graph: WarehouseGraph, fleet: FleetState, agent_id: AgentId, observation_depth: float
) -> tuple[AgentId, ...]:
    """{a_j : (a_i,a_j) in E^A_t}, the communication graph, bounded by
    d_obs, closest first by d_G(l_i(t), l_j(t)) with ties by agent id."""
    location = fleet.locations()[agent_id]
    distances = {
        other_id: graph.distance(location, state.location)
        for other_id, state in fleet.agents.items()
        if other_id != agent_id
    }
    return tuple(
        sorted(
            (other_id for other_id, d in distances.items() if d <= observation_depth),
            key=lambda other_id: (distances[other_id], other_id),
        )
    )


def build_observation(
    graph: WarehouseGraph,
    fleet: FleetState,
    tasks: Sequence[Task],
    agent_id: AgentId,
    t: int,
    config: ExperimentConfig,
    window: Sequence[VertexId] | None = None,
    conflict_signals: ConflictSignals | None = None,
) -> Observation:
    """o_i(t) for one agent at one timestep. `window` is the set of
    visible vertices, in the order given. None means the field of view.
    The congestion feature is always taken over the field of view.
    `conflict_signals` None leaves the two conflict signals out and returns the
    thesis' Observation. Otherwise it returns an ObservationWithConflictSignals
    with the robot's own override flag and its rank in sigma, where m is the
    length of sigma."""
    if config.observation_depth is None:
        raise ValueError(
            "build_observation requires config.observation_depth (d_obs); "
            "ExperimentConfig already enforces this for controller='decentralised'."
        )

    active_by_agent = active_tasks_by_agent(tasks, t)
    locations = fleet.locations()
    location = locations[agent_id]
    target = current_target(location, active_by_agent.get(agent_id))

    field_of_view = local_subgraph(graph, location, config.observation_depth)
    visible = field_of_view if window is None else tuple(window)
    if len(set(visible)) != len(visible):
        raise ValueError(f"the window of agent {agent_id} lists a vertex more than once")
    occupied_vertices = {loc for aid, loc in locations.items() if aid != agent_id}
    features = {
        v: VertexFeatures(
            *role_flags(graph.vertices[v].role),
            eta=eta(graph, v, target),
            occupied=v in occupied_vertices,
        )
        for v in visible
    }

    congestion = occupancy_fraction(fleet, field_of_view, excl={agent_id})

    observation = Observation(visible_vertices=visible, features=features, congestion=congestion)
    if conflict_signals is None:
        return observation
    if agent_id not in conflict_signals.priority:
        raise ValueError(f"agent {agent_id} is not in the priority order")
    return ObservationWithConflictSignals(
        visible_vertices=observation.visible_vertices,
        features=observation.features,
        congestion=observation.congestion,
        overridden_last_step=agent_id in conflict_signals.overridden,
        priority_rank=encode_priority_rank(
            conflict_signals.priority.index(agent_id),
            len(conflict_signals.priority),
            conflict_signals.rank_encoding,
        ),
    )
