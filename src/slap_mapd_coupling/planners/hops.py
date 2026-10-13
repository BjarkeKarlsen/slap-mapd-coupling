"""Hop distances shared by the reference planners."""

from __future__ import annotations

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph


def hop_distances_to(graph: WarehouseGraph, goal: VertexId) -> dict[VertexId, int]:
    """Unweighted (timestep) shortest-hop distance from every vertex to
    goal, over the graph reversed -- the admissible/consistent heuristic
    for the space-time search below. Hop count, not d_G's edge cost,
    because the search minimises timesteps."""
    reverse: dict[VertexId, list[VertexId]] = {v: [] for v in graph.vertices}
    for edge in graph.edges:
        reverse[edge.target].append(edge.source)

    dist = {goal: 0}
    frontier = [goal]
    while frontier:
        next_frontier = []
        for vertex in frontier:
            for predecessor in reverse[vertex]:
                if predecessor not in dist:
                    dist[predecessor] = dist[vertex] + 1
                    next_frontier.append(predecessor)
        frontier = next_frontier
    return dist
