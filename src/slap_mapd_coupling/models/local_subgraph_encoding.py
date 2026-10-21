"""Pads and masks an Observation (environment/observation.py) into the
fixed-shape arrays GNNEncoder.forward_padded and RLlib's batched pipeline
need. An observation covers a window, the field of view by default, whose
vertex count varies, so everything is padded to max_local_nodes.

The nodes keep the order of the window. A window larger than
max_local_nodes raises ValueError for every router (issue #66), because
dropping vertices would change the window. Raise the cap instead. A
receiver keeps its max_messages closest senders by d_G, ties by agent id.

The edge list has one row per undirected pair among the window's
vertices, padded to E_max = max_local_nodes * d_max. That bound holds
because every pair is a directed edge, and no vertex has more than d_max
outgoing edges.

The move score (sec:method:model) needs, for each move slot, the vertex it
leads to and the robot standing there. Slot k is the k-th outgoing
neighbour by vertex number, so every observation carries two arrays of
length d_max: the local index of that vertex in the padded view, and a
mask for the slots that exist. Every window must hold the robot's own
vertex and all its outgoing neighbours, and fit the cap. `check_windows`
verifies this for a whole graph when the environment is built
(`check_slot_targets` for the field of view), and `encode_view` raises if
a view breaks it.

A message is the sender's own embedding (sec:method:model), so with
communication on a receiver's sample carries the padded views of its
senders, each over the sender's own window, with an extra axis M. The network
embeds them with the current weights. For each move slot the sample also
names the sender standing on that vertex, if the receiver hears it. With
communication off the sender keys are absent. `encode_all_observations`
builds the sample of every robot for one timestep.

With pooling on, the sample also carries `pool_mask`, which marks the
vertices of the robot's own view that the readout pools over. It is 0.0
for the halo and for padding. Senders need none, since a message is the
sender's own embedding. With pooling off the key is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from slap_mapd_coupling.core.agents import AgentId, FleetState
from slap_mapd_coupling.core.experiment_config import ExperimentConfig
from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.tasks import Task
from slap_mapd_coupling.environment.action_masking import to_action_mask
from slap_mapd_coupling.environment.observation import (
    Observation,
    build_observation,
    communication_neighbours,
    local_subgraph,
)
from slap_mapd_coupling.environment.spaces import Mask, claim_mask, legality_mask, max_out_degree
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM, Pooling


class LocalSubgraphEncodingConfig(BaseModel):
    """max_local_nodes/max_messages: the padding caps. Local to this module,
    like GNNEncoderConfig."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_local_nodes: PositiveInt
    max_messages: NonNegativeInt


def check_communication_caps(config: LocalSubgraphEncodingConfig, communication: bool) -> None:
    """Rejects communication with room for no sender, which would silently
    switch the messages off while the encoder still expects them."""
    if communication and config.max_messages == 0:
        raise ValueError(
            "communication is on but max_messages is 0, so no sender fits in an "
            "observation. Set max_messages to at least 1, or switch communication off."
        )


def _check_cap_fits(config: LocalSubgraphEncodingConfig, d_max: int) -> None:
    """The view keeps the robot's vertex and all its outgoing neighbours, so
    the cap must have room for d_max + 1 vertices."""
    if config.max_local_nodes < d_max + 1:
        raise ValueError(
            f"max_local_nodes={config.max_local_nodes} < d_max+1={d_max + 1}, so a vertex with "
            "d_max outgoing neighbours does not fit in its own view. Raise max_local_nodes."
        )


def check_windows(
    graph: WarehouseGraph,
    windows: Mapping[VertexId, Sequence[VertexId]],
    config: LocalSubgraphEncodingConfig,
) -> None:
    """`windows` maps each vertex a robot can stand on to the window its
    observation covers there. Raises ValueError if a window lists a vertex
    twice, leaves out its own vertex or one of that vertex's outgoing
    neighbours, or holds more than max_local_nodes vertices. The move score
    needs the embedding at the vertex each move leads to, so a missing
    neighbour would leave a slot with no input. Run it once when the
    environment is built, so a rollout never meets it."""
    for v, window in windows.items():
        members = set(window)
        if len(members) != len(window):
            raise ValueError(f"the window of vertex {v} lists a vertex more than once.")
        if v not in members:
            raise ValueError(
                f"the window of vertex {v} leaves out {v} itself, so a robot standing there "
                "has no embedding of its own."
            )
        missing = [w for w in graph.out_neighbours(v) if w not in members]
        if missing:
            raise ValueError(
                f"the outgoing neighbours {missing} of vertex {v} are outside its window, so "
                "the moves to them have no embedding."
            )
        if len(members) > config.max_local_nodes:
            raise ValueError(
                f"the window of vertex {v} has {len(members)} vertices, more than "
                f"max_local_nodes={config.max_local_nodes}. A window is never truncated, so "
                "raise max_local_nodes."
            )


def check_slot_targets(
    graph: WarehouseGraph, observation_depth: int, config: LocalSubgraphEncodingConfig
) -> None:
    """check_windows with the field of view as every movable vertex's
    window. It first names an outgoing neighbour beyond d_obs, since that
    is fixed by raising observation_depth rather than the cap."""
    _check_cap_fits(config, max_out_degree(graph))
    movable = [v for v, vertex in graph.vertices.items() if vertex.role.movable]
    for v in movable:
        for w in graph.out_neighbours(v):
            if graph.distance(v, w) > observation_depth:
                raise ValueError(
                    f"vertex {w} is an outgoing neighbour of {v} but d_G({v}, {w}) = "
                    f"{graph.distance(v, w)} > observation_depth={observation_depth}, so "
                    "the move to it has no embedding in the field of view. Raise "
                    "observation_depth or shorten the edge."
                )
    check_windows(graph, {v: local_subgraph(graph, v, observation_depth) for v in movable}, config)


@dataclass(frozen=True)
class EncodedView:
    """One robot's padded window, without a batch dimension. It is the
    same layout for the robot itself and for each of its senders."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    edge_index: np.ndarray  # [E_max, 2] int64, one row per undirected pair, pad rows are (0, 0)
    edge_mask: np.ndarray  # [E_max], 1.0 valid / 0.0 pad
    own_index: int
    location: VertexId  # l_i(t)
    move_targets: tuple[VertexId, ...]  # N+(l_i(t)) in slot order
    move_target_index: np.ndarray  # [d_max] int64, local index of the vertex slot k leads to


@dataclass(frozen=True)
class EncodedObservation:
    """The padded arrays for one agent at one timestep, without a batch
    dimension. The sender arrays are None with communication off."""

    node_features: np.ndarray  # [max_local_nodes, NODE_FEATURE_DIM]
    node_mask: np.ndarray  # [max_local_nodes], 1.0 valid / 0.0 pad
    edge_index: np.ndarray  # [E_max, 2] int64, one row per undirected pair, pad rows are (0, 0)
    edge_mask: np.ndarray  # [E_max], 1.0 valid / 0.0 pad
    own_index: int
    congestion: float
    action_mask: np.ndarray  # [num_actions], {0.0, 1.0}, to_action_mask(mask)
    move_target_index: np.ndarray  # [d_max] int64, local index of the vertex slot k leads to
    sender_node_features: Optional[np.ndarray] = None  # [max_messages, N, NODE_FEATURE_DIM]
    sender_node_mask: Optional[np.ndarray] = None  # [max_messages, N]
    sender_edge_index: Optional[np.ndarray] = None  # [max_messages, E_max, 2] int64
    sender_edge_mask: Optional[np.ndarray] = None  # [max_messages, E_max]
    sender_own_index: Optional[np.ndarray] = None  # [max_messages] int64
    sender_mask: Optional[np.ndarray] = None  # [max_messages], 1.0 real sender / 0.0 pad
    move_sender_index: Optional[np.ndarray] = None  # [d_max] int64, sender standing on slot k
    move_sender_mask: Optional[np.ndarray] = None  # [d_max], 1.0 if that sender is heard
    pool_mask: Optional[np.ndarray] = None  # [max_local_nodes], 1.0 pool / 0.0 halo or pad
    # l_i(t) and the vertices of the senders in sender order. They are not part of the dense
    # sample, and models/window_tables.py keys the compact sample by them.
    location: Optional[VertexId] = None
    sender_locations: Optional[tuple[VertexId, ...]] = None

    def as_rllib_dict(self) -> dict[str, Any]:
        """{"action_mask": ..., "observations": {...}}, the Dict shape that
        masked_observation_space() and encoded_observation_space() declare."""
        observations: dict[str, Any] = {
            "node_features": self.node_features,
            "node_mask": self.node_mask,
            "edge_index": self.edge_index,
            "edge_mask": self.edge_mask,
            "own_index": self.own_index,
            "congestion": np.float32(self.congestion),
            "move_target_index": self.move_target_index,
        }
        if self.sender_mask is not None:
            observations.update(
                sender_node_features=self.sender_node_features,
                sender_node_mask=self.sender_node_mask,
                sender_edge_index=self.sender_edge_index,
                sender_edge_mask=self.sender_edge_mask,
                sender_own_index=self.sender_own_index,
                sender_mask=self.sender_mask,
                move_sender_index=self.move_sender_index,
                move_sender_mask=self.move_sender_mask,
            )
        if self.pool_mask is not None:
            observations["pool_mask"] = self.pool_mask
        return {"action_mask": self.action_mask, "observations": observations}


def max_edges(config: LocalSubgraphEncodingConfig, d_max: int) -> int:
    """E_max, the padded length of the edge list."""
    return config.max_local_nodes * d_max


def encoded_observation_space(
    config: LocalSubgraphEncodingConfig,
    d_max: int,
    communication: bool,
    pooling: Pooling = "none",
) -> gym.spaces.Dict:
    """The "observations" half of masked_observation_space() for one agent.
    `d_max` is max_out_degree(graph), which sets the edge list's size and
    the length of the move slot arrays. The sender keys are present only
    with `communication` on, and pool_mask only with `pooling` on."""
    check_communication_caps(config, communication)
    _check_cap_fits(config, d_max)
    num_nodes = config.max_local_nodes
    num_edges = max_edges(config, d_max)
    spaces: dict[str, gym.Space] = {
        "node_features": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_nodes, NODE_FEATURE_DIM), dtype=np.float32
        ),
        "node_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(num_nodes,), dtype=np.float32),
        "edge_index": gym.spaces.Box(
            low=0, high=num_nodes - 1, shape=(num_edges, 2), dtype=np.int64
        ),
        "edge_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(num_edges,), dtype=np.float32),
        "own_index": gym.spaces.Discrete(num_nodes),
        "congestion": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        "move_target_index": gym.spaces.Box(
            low=0, high=num_nodes - 1, shape=(d_max,), dtype=np.int64
        ),
    }
    if communication:
        m = config.max_messages
        spaces.update(
            sender_node_features=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(m, num_nodes, NODE_FEATURE_DIM), dtype=np.float32
            ),
            sender_node_mask=gym.spaces.Box(
                low=0.0, high=1.0, shape=(m, num_nodes), dtype=np.float32
            ),
            sender_edge_index=gym.spaces.Box(
                low=0, high=num_nodes - 1, shape=(m, num_edges, 2), dtype=np.int64
            ),
            sender_edge_mask=gym.spaces.Box(
                low=0.0, high=1.0, shape=(m, num_edges), dtype=np.float32
            ),
            sender_own_index=gym.spaces.Box(low=0, high=num_nodes - 1, shape=(m,), dtype=np.int64),
            sender_mask=gym.spaces.Box(low=0.0, high=1.0, shape=(m,), dtype=np.float32),
            move_sender_index=gym.spaces.Box(low=0, high=m - 1, shape=(d_max,), dtype=np.int64),
            move_sender_mask=gym.spaces.Box(low=0.0, high=1.0, shape=(d_max,), dtype=np.float32),
        )
    if pooling != "none":
        spaces["pool_mask"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_nodes,), dtype=np.float32
        )
    return gym.spaces.Dict(spaces)


def window_structure(
    graph: WarehouseGraph,
    window: Sequence[VertexId],
    d_max: int,
    config: LocalSubgraphEncodingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The vertex mask, edge list and edge mask of `window`, padded, in the
    window's order. They depend on the window only, never on the time step,
    so models/window_tables.py stores them once per window."""
    index = {v: i for i, v in enumerate(window)}
    node_mask = np.zeros(config.max_local_nodes, dtype=np.float32)
    node_mask[: len(window)] = 1.0

    num_edges = max_edges(config, d_max)
    edge_index = np.zeros((num_edges, 2), dtype=np.int64)
    edge_mask = np.zeros(num_edges, dtype=np.float32)
    pairs: set[tuple[int, int]] = set()
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            a, b = index[edge.source], index[edge.target]
            pair = (min(a, b), max(a, b))
            if pair not in pairs:
                edge_index[len(pairs)] = pair
                edge_mask[len(pairs)] = 1.0
                pairs.add(pair)
    return node_mask, edge_index, edge_mask


def slot_targets(
    graph: WarehouseGraph, location: VertexId, index: Mapping[VertexId, int], d_max: int
) -> np.ndarray:
    """move_target_index, the local index in the window (`index`) of the
    vertex each move slot of `location` leads to, 0 for a missing slot."""
    move_target_index = np.zeros(d_max, dtype=np.int64)
    for k, w in enumerate(graph.out_neighbours(location)):
        move_target_index[k] = index[w]
    return move_target_index


def encode_view(
    graph: WarehouseGraph,
    location: VertexId,
    observation: Observation,
    d_max: int,
    config: LocalSubgraphEncodingConfig,
    structure: bool = True,
) -> EncodedView:
    """The window in `observation` as padded arrays, in the window's order.
    Used for the robot itself and for each sender. `d_max` is
    max_out_degree(graph). Raises ValueError if the window leaves out
    `location` or one of its outgoing neighbours, or does not fit the cap
    (see check_windows). With `structure` off the arrays that depend only on
    the window (the masks and the edge list) are left empty, for the compact
    sample of models/window_tables.py, which keeps them in tables."""
    move_targets = graph.out_neighbours(location)
    if len(move_targets) + 1 > config.max_local_nodes:
        raise ValueError(
            f"vertex {location} and its {len(move_targets)} outgoing neighbours do not fit "
            f"max_local_nodes={config.max_local_nodes}."
        )
    if location not in observation.features:
        raise ValueError(f"vertex {location} is outside its own window.")
    outside = [w for w in move_targets if w not in observation.features]
    if outside:
        raise ValueError(
            f"the outgoing neighbours {outside} of vertex {location} are outside its field of "
            "view or window, so their moves have no embedding. check_windows finds this for a "
            "whole graph before the first step."
        )
    selected = list(observation.visible_vertices)
    if len(selected) > config.max_local_nodes:
        raise ValueError(
            f"the window of vertex {location} has {len(selected)} vertices, more than "
            f"max_local_nodes={config.max_local_nodes}. A window is never truncated, so raise "
            "max_local_nodes. check_windows finds this for a whole graph before the first step."
        )
    index = {v: i for i, v in enumerate(selected)}

    node_features = np.zeros((config.max_local_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    for i, v in enumerate(selected):
        f = observation.features[v]
        node_features[i] = (
            float(f.storage),
            float(f.delivery),
            float(f.endpoint),
            float(f.transit),
            f.eta,
            float(f.occupied),
            float(f.claimed),
            float(f.receiving),
        )
    if structure:
        node_mask, edge_index, edge_mask = window_structure(graph, selected, d_max, config)
    else:
        node_mask = edge_index = edge_mask = _EMPTY
    move_target_index = slot_targets(graph, location, index, d_max)

    return EncodedView(
        node_features=node_features,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        own_index=index[location],
        location=location,
        move_targets=move_targets,
        move_target_index=move_target_index,
    )


_EMPTY = np.empty(0, dtype=np.float32)


def _pad_senders(
    sender_views: Sequence[EncodedView], max_messages: int, view: EncodedView
) -> dict[str, np.ndarray]:
    """The six sender arrays, with the views of `sender_views` first and
    zero padding after them. `view` only supplies the shapes."""
    if len(sender_views) > max_messages:
        raise ValueError(f"{len(sender_views)} sender views exceed max_messages={max_messages}")
    arrays = {
        "sender_node_features": np.zeros((max_messages, *view.node_features.shape), np.float32),
        "sender_node_mask": np.zeros((max_messages, *view.node_mask.shape), np.float32),
        "sender_edge_index": np.zeros((max_messages, *view.edge_index.shape), np.int64),
        "sender_edge_mask": np.zeros((max_messages, *view.edge_mask.shape), np.float32),
        "sender_own_index": np.zeros(max_messages, np.int64),
        "sender_mask": np.zeros(max_messages, np.float32),
    }
    for k, sender in enumerate(sender_views):
        arrays["sender_node_features"][k] = sender.node_features
        arrays["sender_node_mask"][k] = sender.node_mask
        arrays["sender_edge_index"][k] = sender.edge_index
        arrays["sender_edge_mask"][k] = sender.edge_mask
        arrays["sender_own_index"][k] = sender.own_index
        arrays["sender_mask"][k] = 1.0
    return arrays


def _move_senders(view: EncodedView, sender_views: Sequence[EncodedView]) -> dict[str, np.ndarray]:
    """For each move slot of `view`, the position among `sender_views` of the
    robot standing on the vertex that slot leads to. One robot stands on a
    vertex, so a vertex has at most one sender. The mask is 0 when nobody
    stands there or when that robot is not among `sender_views`, and the index
    is then 0."""
    position = {sender.location: j for j, sender in enumerate(sender_views)}
    index = np.zeros(len(view.move_target_index), dtype=np.int64)
    mask = np.zeros(len(view.move_target_index), dtype=np.float32)
    for k, w in enumerate(view.move_targets):
        if w in position:
            index[k] = position[w]
            mask[k] = 1.0
    return {"move_sender_index": index, "move_sender_mask": mask}


def _pool_mask(
    observation: Observation, pool: Collection[VertexId], max_local_nodes: int
) -> np.ndarray:
    """1.0 at the vertices of `pool`, in the order of the window, and 0.0
    elsewhere. Raises ValueError if `pool` is empty or reaches outside the
    window."""
    index = {v: i for i, v in enumerate(observation.visible_vertices)}
    if not pool:
        raise ValueError("a robot pools over no vertex, so the pooled mean is undefined.")
    outside = sorted(v for v in pool if v not in index)
    if outside:
        raise ValueError(f"a robot pools over the vertices {outside}, outside its window.")
    mask = np.zeros(max_local_nodes, dtype=np.float32)
    mask[[index[v] for v in pool]] = 1.0
    return mask


def _assemble(
    view: EncodedView,
    observation: Observation,
    mask: Mask,
    sender_views: Optional[Sequence[EncodedView]],
    config: LocalSubgraphEncodingConfig,
    pool_mask: Optional[np.ndarray] = None,
    padded: bool = True,
) -> EncodedObservation:
    senders: dict[str, np.ndarray] = {}
    if sender_views is not None:
        if padded:
            senders = _pad_senders(sender_views, config.max_messages, view)
        else:
            # The compact sample names its senders by row key, so it needs only which slots
            # are real, not their views.
            if len(sender_views) > config.max_messages:
                raise ValueError(
                    f"{len(sender_views)} sender views exceed max_messages={config.max_messages}"
                )
            sender_mask = np.zeros(config.max_messages, np.float32)
            sender_mask[: len(sender_views)] = 1.0
            senders = {"sender_mask": sender_mask}
        senders.update(_move_senders(view, sender_views))
    return EncodedObservation(
        node_features=view.node_features,
        node_mask=view.node_mask,
        edge_index=view.edge_index,
        edge_mask=view.edge_mask,
        own_index=view.own_index,
        congestion=observation.congestion,
        action_mask=to_action_mask(mask),
        move_target_index=view.move_target_index,
        pool_mask=pool_mask,
        location=view.location,
        sender_locations=(
            None if sender_views is None else tuple(sender.location for sender in sender_views)
        ),
        **senders,
    )


def encode_observation(
    graph: WarehouseGraph,
    agent_location: VertexId,
    observation: Observation,
    mask: Mask,
    d_max: int,
    config: LocalSubgraphEncodingConfig,
    sender_views: Optional[Sequence[EncodedView]] = None,
) -> EncodedObservation:
    """o_i(t) and its action mask -> the fixed-shape EncodedObservation.
    `mask` is the caller's own legality_mask result and `d_max` is
    max_out_degree(graph). `sender_views` holds the views of the senders
    the agent hears, closest first, at most max_messages of them, and they
    also fix which move slot each one stands on. None means communication
    is off and leaves the sender arrays out."""
    view = encode_view(graph, agent_location, observation, d_max, config)
    return _assemble(view, observation, mask, sender_views, config)


def encode_all_observations(
    graph: WarehouseGraph,
    fleet: FleetState,
    tasks: Sequence[Task],
    t: int,
    experiment_config: ExperimentConfig,
    encoding_config: LocalSubgraphEncodingConfig,
    d_max: int,
    windows: Optional[Mapping[AgentId, Sequence[VertexId]]] = None,
    pooling: Pooling = "none",
    pool_vertices: Optional[Mapping[AgentId, Collection[VertexId]]] = None,
    acting: Optional[Sequence[AgentId]] = None,
    chosen: Optional[Mapping[AgentId, Mapping[AgentId, VertexId]]] = None,
    padded: bool = True,
) -> dict[AgentId, EncodedObservation]:
    """The EncodedObservation of every acting robot at timestep t, keyed by
    agent id in ascending order. `d_max` is max_out_degree(graph). `windows`
    maps every robot to the window its observation covers. None means the
    field of view for every robot.

    `acting` lists the robots that choose in this turn. None means every
    robot. `chosen` maps each acting robot to the successor vertices its unit's
    earlier agents have chosen. A robot with an entry gets the claimed column
    from them and the claim mask added to its action mask. None means no
    claims, as for the decentralised router. Communication needs every robot
    to act.

    With `pooling` on, every sample carries pool_mask. `pool_vertices` maps
    every robot to the vertices of its window it pools over, its zone for
    the section-based router. None means the whole window. It must be None
    with pooling off.

    Each robot's view is encoded once and reused as a sender block by every
    robot that hears it. With experiment_config.communication on, a robot
    hears the max_messages closest agents of the communication graph (ties
    by agent id), closest first. A move slot's sender is the heard robot
    standing on the vertex the slot leads to, if any. The controller, the
    training env and any later env all build their observations here, so a
    message is the same wherever it is read.

    With `padded` off, only what the compact sample of models/window_tables.py
    reads is built: the vertex features, the location, the congestion, the
    action mask, which sender slots are real and the move slot senders. The
    window arrays, the pool mask and the senders' views are left empty, since
    the compact sample keeps them in tables or names the senders by row key.
    The training adapter builds its samples this way, and tests compare it
    with the padded build."""
    communication = experiment_config.communication
    check_communication_caps(encoding_config, communication)
    if experiment_config.observation_depth is None:
        raise ValueError("encode_all_observations requires experiment_config.observation_depth")

    locations = fleet.locations()
    agent_ids = sorted(locations if acting is None else acting)
    if communication and agent_ids != sorted(locations):
        raise ValueError("communication needs every robot to act in the same turn.")
    unknown = sorted(set(agent_ids) - set(locations))
    if unknown:
        raise ValueError(f"acting lists the unknown robots {unknown}.")
    if windows is not None:
        missing = [agent_id for agent_id in agent_ids if agent_id not in windows]
        if missing:
            raise ValueError(f"windows has no window for the robots {missing}.")
    if pool_vertices is not None:
        if pooling == "none":
            raise ValueError("pool_vertices were given but pooling is off.")
        missing = [agent_id for agent_id in agent_ids if agent_id not in pool_vertices]
        if missing:
            raise ValueError(f"pool_vertices has no pool for the robots {missing}.")
    observations = {
        agent_id: build_observation(
            graph,
            fleet,
            tasks,
            agent_id,
            t,
            experiment_config,
            window=None if windows is None else windows[agent_id],
            claimed=() if chosen is None or agent_id not in chosen else chosen[agent_id].values(),
        )
        for agent_id in agent_ids
    }
    views = {
        agent_id: encode_view(
            graph, locations[agent_id], observations[agent_id], d_max, encoding_config, padded
        )
        for agent_id in agent_ids
    }

    encoded: dict[AgentId, EncodedObservation] = {}
    for agent_id in agent_ids:
        sender_views: Optional[list[EncodedView]] = None
        if communication:
            heard = communication_neighbours(
                graph, fleet, agent_id, experiment_config.observation_depth
            )
            sender_views = [views[j] for j in heard[: encoding_config.max_messages]]
        pool_mask: Optional[np.ndarray] = None
        if pooling != "none" and padded:
            observation = observations[agent_id]
            pool = (
                observation.visible_vertices if pool_vertices is None else pool_vertices[agent_id]
            )
            pool_mask = _pool_mask(observation, pool, encoding_config.max_local_nodes)
        mask = legality_mask(graph, locations[agent_id], d_max=d_max)
        if chosen is not None and agent_id in chosen:
            claim = claim_mask(graph, agent_id, locations, chosen[agent_id], d_max)
            mask = tuple(a + b for a, b in zip(mask, claim))
        encoded[agent_id] = _assemble(
            views[agent_id],
            observations[agent_id],
            mask,
            sender_views,
            encoding_config,
            pool_mask,
            padded,
        )
    return encoded
