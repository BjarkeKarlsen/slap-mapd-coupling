"""o_i(t), the decentralised observation (sec:pf:observations).

Built from the field of view G_i^(d_obs)(t), the relative distance label
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
transit, receiving), because VertexRole allows a vertex several roles.
Transit is true when none of the other four apply.

The distance feature is the relative distance label (sec:method:model):
the vertex's distance label eta_i(v,t) minus that of the robot's own
vertex, capped at plus or minus d_obs. The reward's potential still uses
eta_i itself (environment/reward_function.py).

build_observation gives one robot's observation as one object per vertex, which
reads like the definition. A step asks for every robot's observation, and for
the centralised router each window is the whole warehouse, so the encoder uses
`window_features` instead. It returns the same numbers as columns over the
window, read from the graph's distance table, with what every robot shares
worked out once per step (`step_state`). Tests check that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

import numpy as np

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, VertexRole, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.environment.reward_function import current_target

# [A82]: the distance label divided by this one number, the same in every warehouse, so the
# feature stays near 0 to 1 next to the 0/1 features (about 30 steps in fleet 8, over 50 in
# fleet 64). As in van Knippenberg et al. (2021) the robot then sees how far its target is.
DISTANCE_SCALE = 50.0


@dataclass(frozen=True)
class VertexFeatures:
    """f(v,t), the vertex features: the multi-hot role, the
    relative distance label (`eta`), whether another agent occupies v, and
    whether v is in V_{<i}(t), the vertices the earlier agents of the same
    decision have chosen to end the timestep on."""

    storage: bool
    delivery: bool
    endpoint: bool
    transit: bool  # true iff none of the above -- see module docstring
    eta: float  # the relative distance label, not eta_i(v,t) itself
    occupied: bool
    claimed: bool = False
    receiving: bool = False
    distance: float = 0.0  # [A82] eta_i(v,t) / DISTANCE_SCALE, the distance label itself


@dataclass(frozen=True)
class Observation:
    """o_i(t), the decentralised observation, structured
    rather than a fixed-size vector."""

    visible_vertices: tuple[VertexId, ...]  # the window, V_i^(d_obs)(t) by default
    features: dict[VertexId, VertexFeatures]
    congestion: float  # delta_i(t); always one of o_i(t)'s three pieces


def local_subgraph(graph: WarehouseGraph, location: VertexId, depth: float) -> tuple[VertexId, ...]:
    """V_i^(d_obs)(t), the vertices of the field of view, the movable
    vertices with cost distance d_G(l_i(t), v) <= d_obs."""
    return graph.within(location, depth)


def role_flags(role: VertexRole) -> tuple[bool, bool, bool, bool, bool]:
    """(storage, delivery, endpoint, transit, receiving), multi-hot (see
    module docstring)."""
    transit = not (role.storage or role.delivery or role.endpoint or role.receiving)
    return role.storage, role.delivery, role.endpoint, transit, role.receiving


def eta(graph: WarehouseGraph, vertex: VertexId, target: VertexId) -> float:
    """eta_i(v,t) = d_G(v, q_i(t)), the distance label, for
    any visible vertex v."""
    return graph.distance(vertex, target)


def relative_eta(own_eta: float, vertex_eta: float, observation_depth: float) -> float:
    """The relative distance label, eta_i(v,t) - eta_i(l_i(t),t) capped at
    plus or minus d_obs. 0 at the robot's own vertex, -1 one step closer to the
    target and +1 one step further on unit edge costs."""
    return min(observation_depth, max(-observation_depth, vertex_eta - own_eta))


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
    claimed: Collection[VertexId] = (),
    parking: VertexId | None = None,
) -> Observation:
    """o_i(t) for one agent at one timestep. `window` is the set of
    visible vertices, in the order given. None means the field of view.
    The congestion feature is always taken over the field of view.
    `claimed` is V_{<i}(t), empty for the decentralised router."""
    # Read once, since config is a pydantic model and the loop below visits every vertex of
    # the window, the whole warehouse for the centralised router.
    depth = config.observation_depth
    if depth is None:
        raise ValueError(
            "build_observation requires config.observation_depth (d_obs); "
            "ExperimentConfig already enforces this for controller='decentralised'."
        )

    active_by_agent = active_tasks_by_agent(tasks, t)
    locations = fleet.locations()
    location = locations[agent_id]
    target = current_target(location, active_by_agent.get(agent_id), parking)

    field_of_view = local_subgraph(graph, location, depth)
    visible = field_of_view if window is None else tuple(window)
    if len(set(visible)) != len(visible):
        raise ValueError(f"the window of agent {agent_id} lists a vertex more than once")
    occupied_vertices = {loc for aid, loc in locations.items() if aid != agent_id}
    claimed_vertices = set(claimed)
    own_eta = eta(graph, location, target)
    features = {}
    for v in visible:
        storage, delivery, endpoint, transit, receiving = role_flags(graph.vertices[v].role)
        vertex_eta = eta(graph, v, target)
        features[v] = VertexFeatures(
            storage,
            delivery,
            endpoint,
            transit,
            eta=relative_eta(own_eta, vertex_eta, depth),
            occupied=v in occupied_vertices,
            claimed=v in claimed_vertices,
            receiving=receiving,
            distance=vertex_eta / DISTANCE_SCALE,
        )

    congestion = occupancy_fraction(fleet, field_of_view, excl={agent_id})

    return Observation(visible_vertices=visible, features=features, congestion=congestion)


@dataclass(frozen=True)
class StepState:
    """What every robot's observation at one timestep shares, worked out once
    per step: where the robots stand, each robot's active task, how many
    robots stand on each vertex, and the multi-hot role of each vertex. The
    last two are indexed by the rows of graph.distance_table()."""

    locations: Mapping[AgentId, VertexId]
    active: Mapping[AgentId, Task]
    occupants: np.ndarray  # [|V|] int, robots standing on each vertex
    roles: np.ndarray  # [|V|, 5] bool, role_flags of each vertex


def _role_table(graph: WarehouseGraph) -> np.ndarray:
    """role_flags of every vertex, one row per row of graph.distance_table()."""
    table = np.array(
        [role_flags(vertex.role) for vertex in graph.vertices.values()], dtype=bool
    ).reshape(len(graph.vertices), 5)
    table.flags.writeable = False
    return table


def step_state(graph: WarehouseGraph, fleet: FleetState, tasks: Sequence[Task], t: int) -> StepState:
    """The StepState of timestep t."""
    row = graph.vertex_rows()
    locations = fleet.locations()
    occupants = np.zeros(len(row), dtype=np.int64)
    for v in locations.values():
        occupants[row[v]] += 1
    roles = graph.cached("observation.role_flags", lambda: _role_table(graph))
    return StepState(locations, active_tasks_by_agent(tasks, t), occupants, roles)


@dataclass(frozen=True)
class WindowFeatures:
    """The vertex features of build_observation as columns, one entry per
    vertex of the window in the window's order, with the congestion feature."""

    window: tuple[VertexId, ...]
    index: Mapping[VertexId, int]  # the position of each vertex in the window
    roles: np.ndarray  # [n, 5] bool: storage, delivery, endpoint, transit, receiving
    eta: np.ndarray  # [n] float, the relative distance label
    occupied: np.ndarray  # [n] bool
    claimed: np.ndarray  # [n] bool
    distance: np.ndarray  # [n] float, eta_i(v,t) / DISTANCE_SCALE
    congestion: float


def window_features(
    graph: WarehouseGraph,
    fleet: FleetState,
    state: StepState,
    agent_id: AgentId,
    depth: int,
    window: Sequence[VertexId] | None = None,
    claimed: Collection[VertexId] = (),
    parking: VertexId | None = None,
) -> WindowFeatures:
    """The features build_observation gives agent_id, the same numbers, as
    columns. `depth` is d_obs. The arguments mean what they mean there."""
    location = state.locations[agent_id]
    target = current_target(location, state.active.get(agent_id), parking)
    field_of_view = local_subgraph(graph, location, depth)
    visible = field_of_view if window is None else tuple(window)
    rows, index = graph.window_rows(visible)
    if len(index) != len(visible):
        raise ValueError(f"the window of agent {agent_id} lists a vertex more than once")

    row = graph.vertex_rows()
    table = graph.distance_table()
    own_row = row[location]
    labels = table[rows, row[target]]  # eta_i(v,t) on every vertex of the window
    own_eta = table[own_row, row[target]]
    if np.isnan(own_eta) or np.isnan(labels).any():
        raise KeyError(f"a vertex of the window of agent {agent_id} cannot reach {target}.")
    # relative_eta and the occupied set of build_observation, over the whole window at once.
    relative = np.minimum(depth, np.maximum(-depth, labels - own_eta))
    occupied = (state.occupants[rows] - (rows == own_row)) > 0
    claimed_column = np.zeros(len(visible), dtype=bool)
    for v in claimed:
        position = index.get(v)
        if position is not None:
            claimed_column[position] = True

    return WindowFeatures(
        window=visible,
        index=index,
        roles=state.roles[rows],
        eta=relative,
        occupied=occupied,
        claimed=claimed_column,
        distance=labels / DISTANCE_SCALE,
        congestion=occupancy_fraction(fleet, field_of_view, excl={agent_id}),
    )
