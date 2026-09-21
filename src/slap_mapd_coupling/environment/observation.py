"""o_i(t): the decentralised agent's observation (eq:observation, sec:method:rl).

o_i(t) = (G_i^(d)(t), eta_i(., t), delta_i(t), messages from
{a_j : (a_i,a_j) in E^A_t}) -- the local subgraph within observation depth
d (eq:localsubgraph), the progress-potential annotation on every visible
vertex (eq:potential), the agent-level congestion feature (eq:congestion,
the decentralised instance of the generic occupancy fraction
eq:occupancy), and one message per agent within communication range
(eq:commgraph), each carrying (d_G(l_i(t),l_j(t)), eta_j(l_j(t),t)).

This module returns a *structured* Observation, not a fixed-size tensor:
"how o_i(t) is encoded into a vector... [is] a modelling choice specified
in sec:method:model," i.e. models/gnn_encoder.py's job (#27/#28), not
this one's.

Two points flagged, not literally specified:
- eq:features writes role(v) as if it were a single category in
  {V_str, V_del, V_ep, transit}, but core/graph.py's VertexRole is
  deliberately non-exclusive (a vertex can be storage AND an endpoint at
  once) -- a real, pre-existing tension between the formal text and this
  repo's own graph model, not something this module introduces. Resolved
  by using a multi-hot vector (storage, delivery, endpoint, transit)
  instead of a strict one-hot, where transit is true iff none of the
  other three roles apply (the residual/default category "role(v)"
  implies, and the only one of the four that isn't an actual VertexRole
  field).
- delta_i(t) needs a congestion radius, but ExperimentConfig only
  requires congestion_radius when congestion_sensitive=True (it's
  otherwise None) -- so a decentralised agent with congestion_sensitive
  =False has no radius to evaluate eq:congestion's window at. Observation
  .congestion is therefore Optional: None when congestion_radius isn't
  configured (nothing to compute, not silently reported as zero
  congestion), a real float otherwise. This mirrors the "sensitivity
  toggle" language itself (sec:pf:controllers): whether the *policy*
  conditions on delta_i(t) is the toggle; whether it's even computed
  follows the same flag here, since there is no separate parameter for
  "compute it but let the policy ignore it."
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
    """f(v,t) (eq:features): the multi-hot role, the progress-potential
    annotation eta_i(v,t) (eq:potential), and whether another agent
    currently occupies v."""

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
    """o_i(t) (eq:observation), structured rather than a fixed-size vector
    (see module docstring)."""

    visible_vertices: tuple[VertexId, ...]  # V_i^(d)(t)
    features: dict[VertexId, VertexFeatures]
    congestion: float | None  # delta_i(t); None iff congestion_radius isn't configured
    messages: tuple[Message, ...]


def local_subgraph(graph: WarehouseGraph, location: VertexId, depth: float) -> tuple[VertexId, ...]:
    """V_i^(d)(t) = {v in V_mov : d_G(l_i(t), v) <= d} (eq:localsubgraph).
    d_G is the cost-weighted shortest-path distance (WarehouseGraph.distance),
    not a hop count -- eq:localsubgraph is explicit about using d_G."""
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
    """eta_i(v,t) = d_G(v, q_i(t)) (eq:potential), for an arbitrary
    visible vertex v -- not just the agent's own location, unlike
    reward_function.py's `potential` (which is -eta at l_i(t) alone)."""
    return graph.distance(vertex, target)


def occupancy_fraction(fleet: FleetState, window: Sequence[VertexId], excl: set[AgentId]) -> float:
    """delta(S,t;excl) (eq:occupancy)."""
    window_set = set(window)
    occupants = sum(
        1
        for agent_id, state in fleet.agents.items()
        if agent_id not in excl and state.location in window_set
    )
    denominator = max(1, len(window_set) - len(excl))
    return occupants / denominator


def communication_neighbours(
    graph: WarehouseGraph, fleet: FleetState, agent_id: AgentId, communication_radius: float
) -> tuple[AgentId, ...]:
    """{a_j : (a_i,a_j) in E^A_t} (eq:commgraph)."""
    location = fleet.locations()[agent_id]
    return tuple(
        other_id
        for other_id, state in fleet.agents.items()
        if other_id != agent_id and graph.distance(location, state.location) <= communication_radius
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
            "build_observation requires config.observation_depth (d); "
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
    if config.congestion_radius is not None:
        window = local_subgraph(graph, location, config.congestion_radius)
        congestion = occupancy_fraction(fleet, window, excl={agent_id})

    messages: tuple[Message, ...] = ()
    if config.communication and config.communication_radius is not None:
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
                graph, fleet, agent_id, config.communication_radius
            )
        )

    return Observation(
        visible_vertices=visible, features=features, congestion=congestion, messages=messages
    )
