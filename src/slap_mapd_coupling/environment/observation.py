"""o_i(t), the decentralised observation (sec:pf:observations).

Built from the field of view G_i^(d_obs)(t), the distance label eta_i(v,t)
on every visible vertex and the congestion feature delta_i(t). The result
is a structured Observation. Messages are not part of it, since they depend
on the network's weights (sec:method:model). Encoding the observation into
tensors is the job of models/. The field of view and the communication graph
use cost distance d_G.

The vertex features use a multi-hot role (storage, delivery, endpoint,
transit), because VertexRole allows a vertex several roles. Transit is
true when none of the other three apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.environment.reward_function import current_target


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

    visible_vertices: tuple[VertexId, ...]  # V_i^(d_obs)(t)
    features: dict[VertexId, VertexFeatures]
    congestion: float  # delta_i(t); always one of o_i(t)'s three pieces


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
) -> Observation:
    """o_i(t) for one agent at one timestep."""
    if config.observation_depth is None:
        raise ValueError(
            "build_observation requires config.observation_depth (d_obs); "
            "ExperimentConfig already enforces this for controller='decentralised'."
        )

    active_by_agent = active_tasks_by_agent(tasks, t)
    locations = fleet.locations()
    location = locations[agent_id]
    target = current_target(location, active_by_agent.get(agent_id))

    visible = local_subgraph(graph, location, config.observation_depth)
    occupied_vertices = {loc for aid, loc in locations.items() if aid != agent_id}
    features = {
        v: VertexFeatures(
            *role_flags(graph.vertices[v].role),
            eta=eta(graph, v, target),
            occupied=v in occupied_vertices,
        )
        for v in visible
    }

    congestion = occupancy_fraction(fleet, visible, excl={agent_id})

    return Observation(visible_vertices=visible, features=features, congestion=congestion)
