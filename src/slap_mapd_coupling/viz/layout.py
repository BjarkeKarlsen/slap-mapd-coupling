"""Generic 2D layout for any WarehouseGraph, used when no better positions are supplied."""

from __future__ import annotations

import networkx as nx

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph


def to_networkx(graph: WarehouseGraph) -> nx.DiGraph:
    """WarehouseGraph -> nx.DiGraph, vertex id as node, edge cost as weight."""
    g = nx.DiGraph()
    g.add_nodes_from(graph.vertices.keys())
    for edge in graph.edges:
        g.add_edge(edge.source, edge.target, weight=edge.cost)
    return g


def spring_layout(graph: WarehouseGraph, seed: int = 0) -> dict[VertexId, tuple[float, float]]:
    """Generic fallback position for any graph.

    Works for any WarehouseGraph regardless of provenance (hand-built
    test fixtures, a future non-grid generator), just not as visually
    clean as a real grid layout -- prefer
    instances.generator.generate_instance's positions for graphs it
    produced. Deterministic given seed, matching the project-wide
    replay-determinism expectation.
    """
    raw = nx.spring_layout(to_networkx(graph), seed=seed)
    return {v: (float(x), float(y)) for v, (x, y) in raw.items()}
