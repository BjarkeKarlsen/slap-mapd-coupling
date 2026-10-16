"""What a learned arm needs besides the graph: its pooling, its zones (section arm only) and
the padding caps of its windows (sec:method:rl).

A window larger than `max_local_nodes` raises in the adapter, so the cap is set from the graph
to the largest window the arm uses and no window is truncated. That is every movable vertex for
the centralised arm and the full-view setting, the largest field of view for the decentralised
arm, and the largest zone with its halo for the section arm. Pooling is "mean" for every arm but
the decentralised one, which has "none".
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from slap_mapd_coupling.core.experiment_config import ControllerArchitecture
from slap_mapd_coupling.core.graph import WarehouseGraph
from slap_mapd_coupling.environment.observation import local_subgraph
from slap_mapd_coupling.environment.zones import halo, zones_by_aisle
from slap_mapd_coupling.instances.generator import GeneratedInstance
from slap_mapd_coupling.models.gnn_encoder import Pooling
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.training.env import Zones

POOLING: Mapping[str, Pooling] = {
    "decentralised": "none",
    "centralised": "mean",
    "section": "mean",
    "full_view": "mean",
}


# NamedTuple, not a dataclass, so a test can load an example by path without registering it.
class Arm(NamedTuple):
    """What an arm needs besides the graph: its pooling, its zones (section only)
    and the padding caps."""

    controller: ControllerArchitecture
    pooling: Pooling
    zones: Zones | None
    encoding: LocalSubgraphEncodingConfig


def largest_window(
    graph: WarehouseGraph,
    controller: ControllerArchitecture,
    observation_depth: int,
    zones: Zones | None,
) -> int:
    """The vertices of the largest window the arm uses, so the cap truncates none."""
    movable = [v for v, vertex in graph.vertices.items() if vertex.role.movable]
    if controller in ("centralised", "full_view"):
        return len(movable)
    if controller == "section":
        assert zones is not None
        return max(len(zone | halo(graph, zone, observation_depth)) for zone in zones)
    return max(len(local_subgraph(graph, v, observation_depth)) for v in movable)


def build_arm(
    instance: GeneratedInstance,
    controller: ControllerArchitecture,
    observation_depth: int,
    *,
    num_zones: int,
    max_messages: int,
) -> Arm:
    """The arm of `controller` on `instance`. `num_zones` only matters to the section arm and
    `max_messages` only to a run with communication."""
    zones = None
    if controller == "section":
        zones = zones_by_aisle(instance.graph, instance.aisle_index, num_zones)
    cap = largest_window(instance.graph, controller, observation_depth, zones)
    return Arm(
        controller=controller,
        pooling=POOLING[controller],
        zones=zones,
        encoding=LocalSubgraphEncodingConfig(max_local_nodes=cap, max_messages=max_messages),
    )
