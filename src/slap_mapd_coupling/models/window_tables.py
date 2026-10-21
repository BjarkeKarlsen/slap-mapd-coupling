"""The compact training sample, and the tables that turn it back into the
padded arrays of models/local_subgraph_encoding.py.

Most of a padded sample does not depend on the time step. The vertex mask,
the edge list, the edge mask and the pool mask depend on the window only,
and a robot's window is fixed by its vertex in every arm (the field of view
of that vertex, the zone of that vertex with its halo, or every movable
vertex). The robot's own index and its move targets depend on its vertex,
and five of the eight vertex features (storage, delivery, endpoint, transit,
receiving) on the role of the vertex. Stored in every row, these parts were
over half of a row, and the whole window for the centralised and section
arms.

`WindowTables` holds them once per run. The compact sample a robot stores
holds its vertex, as a position in the table, and the three vertex features
that change with time: eta as float32, and occupied and claimed as uint8
(`compact_rllib_dict`). `WindowLookup` rebuilds the padded arrays from the
tables on the module's device. The rebuilt arrays equal the padded ones bit
for bit, so the encoder, the head and every result are unchanged.

With communication on, a message is the sender's own embedding, and the
sender's own row already holds everything that embedding needs. So a row
names each sender by the key of the sender's row (`row_key`, unique per
episode, timestep and robot), -1 for a padded slot, and the encoder looks the
sender up among the rows it embeds (GNNEncoder.encode_keyed). The robots of a
timestep choose together, so when actions are chosen every sender is in the
batch, and the value pass runs on the whole batch. A training minibatch may
cut a timestep, so the learner adds the rows of the senders outside it as
context rows (`CONTEXT_KEYS`, training/learner.py), which are embedded but
carry no loss.

The tables are built from the graph and the arm (training/env.py::
arm_window_tables), so the adapter and the module always agree on them.
They reach the module through its model config, so a checkpoint carries
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.environment.observation import role_flags
from slap_mapd_coupling.models.gnn_encoder import NODE_FEATURE_DIM, Pooling
from slap_mapd_coupling.models.local_subgraph_encoding import (
    EncodedObservation,
    LocalSubgraphEncodingConfig,
    check_communication_caps,
    slot_targets,
    window_structure,
)

# The columns of the vertex features (gnn_encoder.py): storage, delivery, endpoint, transit,
# eta, occupied, claimed, receiving. The role flags are stored once per window, in the order
# role_flags returns them.
STATIC_COLUMNS = (0, 1, 2, 3, 7)
ETA_COLUMN = 4
FLAG_COLUMNS = (5, 6)

# The compact sample's keys for the robot's own key and its senders' keys. They are present
# exactly when communication is on.
COMPACT_SENDER_KEYS = ("row_key", "sender_row_key")
# The largest row key (training/env.py::row_key), below 2^62 so that the space's range
# high - low fits in int64.
MAX_ROW_KEY = (1 << 62) - 1
# The rows the learner adds to a training minibatch for senders outside it.
CONTEXT_KEYS = ("context_row_key", "context_vertex", "context_eta", "context_flags")

_ARRAY_KEYS = (
    "window_of",
    "own_index",
    "move_target_index",
    "node_mask",
    "edge_index",
    "edge_mask",
    "static_features",
)


@dataclass(frozen=True)
class WindowTables:
    """What a padded sample holds that depends only on the robot's vertex.
    V is the number of movable vertices, W the number of distinct windows, N
    max_local_nodes and E the padded edge list length."""

    vertices: tuple[VertexId, ...]  # the movable vertices, a vertex's position is its row
    window_of: np.ndarray  # [V] int64, the window of each vertex
    own_index: np.ndarray  # [V] int64, the vertex's index in its window
    move_target_index: np.ndarray  # [V, d_max] int64
    node_mask: np.ndarray  # [W, N] float32
    edge_index: np.ndarray  # [W, E, 2] int64
    edge_mask: np.ndarray  # [W, E] float32
    static_features: np.ndarray  # [W, N, 5] float32, the role flags in STATIC_COLUMNS order
    pool_mask: Optional[np.ndarray] = None  # [W, N] float32, present exactly with pooling on

    def position(self) -> dict[VertexId, int]:
        return {v: p for p, v in enumerate(self.vertices)}

    def as_arrays(self) -> dict[str, Any]:
        """The tables as plain arrays, for the module's model config."""
        arrays: dict[str, Any] = {key: getattr(self, key) for key in _ARRAY_KEYS}
        arrays["vertices"] = list(self.vertices)
        if self.pool_mask is not None:
            arrays["pool_mask"] = self.pool_mask
        return arrays

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, Any]) -> "WindowTables":
        return cls(
            vertices=tuple(arrays["vertices"]),
            pool_mask=arrays.get("pool_mask"),
            **{key: np.asarray(arrays[key]) for key in _ARRAY_KEYS},
        )


def build_window_tables(
    graph: WarehouseGraph,
    windows: Mapping[VertexId, Sequence[VertexId]],
    d_max: int,
    config: LocalSubgraphEncodingConfig,
    pools: Optional[Mapping[VertexId, Collection[VertexId]]] = None,
) -> WindowTables:
    """The tables of every vertex in `windows`, which maps each movable vertex
    to its window in order. `pools` maps each vertex to the vertices its
    readout pools over, None with pooling off. Two vertices with the same
    window and pool share one row of the per-window tables."""
    vertices = tuple(sorted(windows))
    num_nodes = config.max_local_nodes
    keys: dict[tuple[tuple[VertexId, ...], Optional[frozenset[VertexId]]], int] = {}
    window_of = np.zeros(len(vertices), dtype=np.int64)
    own_index = np.zeros(len(vertices), dtype=np.int64)
    move_target_index = np.zeros((len(vertices), d_max), dtype=np.int64)
    node_masks: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_masks: list[np.ndarray] = []
    static: list[np.ndarray] = []
    pool_masks: list[np.ndarray] = []
    for p, v in enumerate(vertices):
        window = tuple(windows[v])
        index = {w: i for i, w in enumerate(window)}
        pool = None if pools is None else frozenset(pools[v])
        key = (window, pool)
        if key not in keys:
            keys[key] = len(keys)
            node_mask, edge_index, edge_mask = window_structure(graph, window, d_max, config)
            node_masks.append(node_mask)
            edge_indices.append(edge_index)
            edge_masks.append(edge_mask)
            flags = np.zeros((num_nodes, len(STATIC_COLUMNS)), dtype=np.float32)
            for i, w in enumerate(window):
                flags[i] = [float(flag) for flag in role_flags(graph.vertices[w].role)]
            static.append(flags)
            if pool is not None:
                pool_mask = np.zeros(num_nodes, dtype=np.float32)
                pool_mask[[index[w] for w in pool]] = 1.0
                pool_masks.append(pool_mask)
        window_of[p] = keys[key]
        own_index[p] = index[v]
        move_target_index[p] = slot_targets(graph, v, index, d_max)
    return WindowTables(
        vertices=vertices,
        window_of=window_of,
        own_index=own_index,
        move_target_index=move_target_index,
        node_mask=np.stack(node_masks),
        edge_index=np.stack(edge_indices),
        edge_mask=np.stack(edge_masks),
        static_features=np.stack(static),
        pool_mask=None if pools is None else np.stack(pool_masks),
    )


def compact_observation_space(
    config: LocalSubgraphEncodingConfig,
    num_vertices: int,
    d_max: int,
    communication: bool,
) -> gym.spaces.Dict:
    """The "observations" half of masked_observation_space() for one agent,
    in the compact layout. `num_vertices` is the number of rows of the
    tables. The pool mask is in the tables, so pooling adds no key."""
    check_communication_caps(config, communication)
    num_nodes = config.max_local_nodes
    spaces: dict[str, gym.Space] = {
        "vertex": gym.spaces.Discrete(num_vertices),
        "eta": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_nodes,), dtype=np.float32),
        "flags": gym.spaces.Box(low=0, high=1, shape=(num_nodes, 2), dtype=np.uint8),
        "congestion": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
    }
    if communication:
        m = config.max_messages
        top = MAX_ROW_KEY
        spaces.update(
            row_key=gym.spaces.Box(low=0, high=top, shape=(), dtype=np.int64),
            sender_row_key=gym.spaces.Box(low=-1, high=top, shape=(m,), dtype=np.int64),
            move_sender_index=gym.spaces.Box(low=0, high=m - 1, shape=(d_max,), dtype=np.int64),
            move_sender_mask=gym.spaces.Box(low=0.0, high=1.0, shape=(d_max,), dtype=np.float32),
        )
    return gym.spaces.Dict(spaces)


def compact_rllib_dict(
    item: EncodedObservation,
    position: Mapping[VertexId, int],
    row_key: Optional[int] = None,
    sender_row_keys: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """The compact {"action_mask", "observations"} sample of `item`.
    `position` is WindowTables.position(). The time-dependent features are
    copied out of the padded arrays, so both layouts hold the same numbers.
    With communication on, `row_key` is the key of this row and
    `sender_row_keys` the keys of the senders' rows, in sender order."""
    if item.location is None:
        raise ValueError("the encoded observation has no location, so it has no compact form.")
    observations: dict[str, Any] = {
        "vertex": position[item.location],
        "eta": np.ascontiguousarray(item.node_features[:, ETA_COLUMN]),
        "flags": item.node_features[:, FLAG_COLUMNS].astype(np.uint8),
        "congestion": np.float32(item.congestion),
    }
    if item.sender_mask is not None:
        if row_key is None or sender_row_keys is None:
            raise ValueError("communication is on, so the row key and the sender keys are needed.")
        keys = np.full(len(item.sender_mask), -1, dtype=np.int64)
        keys[: len(sender_row_keys)] = sender_row_keys
        observations.update(
            row_key=np.int64(row_key),
            sender_row_key=keys,
            move_sender_index=item.move_sender_index,
            move_sender_mask=item.move_sender_mask,
        )
    return {"action_mask": item.action_mask, "observations": observations}


class WindowLookup(nn.Module):
    """Rebuilds the padded arrays of local_subgraph_encoding.py from a batch
    of compact samples. The tables are buffers, so they follow the module to
    its device. They are not saved in the state dict, since the model config
    already carries them and the weight sync to the env runners should not."""

    # The buffers registered in __init__, declared for the type checker.
    window_of: torch.Tensor
    own_index: torch.Tensor
    move_target_index: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_mask: torch.Tensor
    static_features: torch.Tensor
    pool_mask: torch.Tensor

    def __init__(self, tables: WindowTables, pooling: Pooling, communication: bool) -> None:
        super().__init__()
        if (pooling != "none") != (tables.pool_mask is not None):
            raise ValueError("the window tables must hold a pool mask exactly when pooling is on.")
        self.pooling = pooling
        self.communication = communication
        for key in _ARRAY_KEYS:
            self.register_buffer(key, torch.as_tensor(getattr(tables, key)), persistent=False)
        if tables.pool_mask is not None:
            self.register_buffer("pool_mask", torch.as_tensor(tables.pool_mask), persistent=False)

    def views(
        self, vertex: torch.Tensor, eta: torch.Tensor, flags: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The padded view of each robot standing on `vertex` (table rows), the
        arrays GNNEncoder embeds."""
        vertex = vertex.long()
        window = self.window_of[vertex]
        static = self.static_features[window]  # [B, N, 5]
        features = torch.empty(
            (*static.shape[:-1], NODE_FEATURE_DIM), dtype=static.dtype, device=static.device
        )
        features[..., STATIC_COLUMNS] = static
        features[..., ETA_COLUMN] = eta.to(static.dtype)
        features[..., FLAG_COLUMNS] = flags.to(static.dtype)
        return {
            "node_features": features,
            "node_mask": self.node_mask[window],
            "edge_index": self.edge_index[window],
            "edge_mask": self.edge_mask[window],
            "own_index": self.own_index[vertex],
        }

    def forward(self, obs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        """The padded observation, keyed as encoded_observation_space() declares
        it, except with communication on. The senders then stay keys
        (row_key, sender_row_key and the move slot arrays are passed on), and
        the context rows of a training minibatch are rebuilt under "context"."""
        dense: dict[str, Any] = self.views(obs["vertex"], obs["eta"], obs["flags"])
        dense["congestion"] = obs["congestion"]
        dense["move_target_index"] = self.move_target_index[obs["vertex"].long()]
        if self.pooling != "none":
            dense["pool_mask"] = self.pool_mask[self.window_of[obs["vertex"].long()]]
        if self.communication:
            for key in ("row_key", "sender_row_key", "move_sender_index", "move_sender_mask"):
                dense[key] = obs[key]
            if "context_row_key" in obs:
                dense["context"] = self.views(
                    obs["context_vertex"], obs["context_eta"], obs["context_flags"]
                )
                dense["context_row_key"] = obs["context_row_key"]
        return dense


def window_tables_from_config(model_config: Mapping[str, Any]) -> Optional[WindowTables]:
    """The tables in a module's model config, or None for a module that reads
    padded samples."""
    arrays = model_config.get("window_tables")
    return None if arrays is None else WindowTables.from_arrays(arrays)


def expand_compact(
    tables: WindowTables, sample: Mapping[str, Any], pooling: Pooling, communication: bool
) -> dict[str, np.ndarray]:
    """The padded observation of one compact sample (the "observations" half),
    as numpy arrays without a batch dimension. For tests and inspection. The
    module rebuilds whole batches on its device with WindowLookup. With
    communication on, the senders stay keys."""
    lookup = WindowLookup(tables, pooling, communication)
    batch = {key: torch.as_tensor(np.asarray(value))[None] for key, value in sample.items()}
    with torch.no_grad():
        dense = lookup(batch)
    return {key: value[0].numpy() for key, value in dense.items()}
