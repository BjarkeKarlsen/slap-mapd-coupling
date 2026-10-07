"""o_i(t), the decentralised observation (eq:observation, sec:method:rl).

Built from the field of view G_i^(d_obs)(t) (eq:localsubgraph), the
distance label eta_i(v,t) on every visible vertex (eq:potential) and the
congestion feature delta_i(t) (eq:congestion). The result is a structured
Observation. Encoding it into tensors is the job of models/ (sec:method:model).

Where the code still differs from the thesis:
- Messages are carried here as hand-built (distance, eta_j) pairs. In the
  thesis they are learned by the policy and are not part of o_i(t), see
  issue #91.
- The field of view and the communication graph (eq:commgraph) use cost
  distance d_G, not hop distance, see issue #89.
- delta_i(t) is None when congestion_sensitive=False, although the thesis
  always includes it in o_i(t).

The vertex features (eq:features) use a multi-hot role (storage, delivery,
endpoint, transit), because VertexRole allows a vertex several roles.
Transit is true when none of the other three apply.
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
    """f(v,t), the vertex features (eq:features): the multi-hot role, the
    distance label eta_i(v,t) (eq:potential), and whether another agent
    occupies v."""

    storage: bool
    delivery: bool
    endpoint: bool
    transit: bool  # true iff none of the above -- see module docstring
    eta: float
    occupied: bool


@dataclass(frozen=True)
class Message:
    """One message from a_j, (d_G(l_i(t),l_j(t)), eta_j(l_j(t),t))."""

    sender: AgentId
    distance: float
    eta: float


@dataclass(frozen=True)
class Observation:
    """o_i(t), the decentralised observation (eq:observation), structured
    rather than a fixed-size vector."""

    visible_vertices: tuple[VertexId, ...]  # V_i^(d_obs)(t)
    features: dict[VertexId, VertexFeatures]
    congestion: float | None  # delta_i(t); None iff congestion_sensitive=False
    messages: tuple[Message, ...]


def local_subgraph(graph: WarehouseGraph, location: VertexId, depth: float) -> tuple[VertexId, ...]:
    """V_i^(d_obs)(t), the vertices of the field of view (eq:localsubgraph).

    Uses cost distance d_G(l_i(t), v) <= d_obs. The thesis uses hop
    distance, see issue #89."""
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
    """eta_i(v,t) = d_G(v, q_i(t)), the distance label (eq:potential), for
    any visible vertex v."""
    return graph.distance(vertex, target)


def occupancy_fraction(fleet: FleetState, window: Sequence[VertexId], excl: set[AgentId]) -> float:
    """Share of `window` occupied by agents outside `excl`. With the field
    of view as window and the agent itself excluded, this is the
    congestion feature delta_i(t) (eq:congestion)."""
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
    """{a_j : (a_i,a_j) in E^A_t}, the communication graph (eq:commgraph),
    bounded by d_obs."""
    location = fleet.locations()[agent_id]
    return tuple(
        other_id
        for other_id, state in fleet.agents.items()
        if other_id != agent_id and graph.distance(location, state.location) <= observation_depth
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

    congestion: float | None = None
    if config.congestion_sensitive:
        congestion = occupancy_fraction(fleet, visible, excl={agent_id})

    messages: tuple[Message, ...] = ()
    if config.communication:
        messages = tuple(
            Message(
                sender=other_id,
                distance=graph.distance(location, locations[other_id]),
                eta=eta(
                    graph,
                    locations[other_id],
                    current_target(locations[other_id], active_by_agent.get(other_id)),
                ),
            )
            for other_id in communication_neighbours(
                graph, fleet, agent_id, config.observation_depth
            )
        )

    return Observation(
        visible_vertices=visible, features=features, congestion=congestion, messages=messages
    )
