"""Action space and the action mask.

This is the mask logic itself, independent of RLlib. The RLlib wiring is in
`action_masking.py`.
"""

from __future__ import annotations

from gymnasium.spaces import Discrete

from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph

Mask = tuple[float, ...]  # M(v, .): d_max+1 entries, 0.0 (legal) or -inf (illegal)


def max_out_degree(graph: WarehouseGraph) -> int:
    """d_max = max_{v in V_mov} |N+(v)|, the size of the action mask minus one."""
    movable = (v for v, vertex in graph.vertices.items() if vertex.role.movable)
    return max((len(graph.out_neighbours(v)) for v in movable), default=0)


def action_space(graph: WarehouseGraph) -> Discrete:
    """The shared per-agent action space, d_max+1 discrete slots.

    One wait slot plus d_max move slots, shared across every vertex and
    agent so the policy head has a fixed size. See legality_mask for which
    slots are legal at a vertex.
    """
    return Discrete(max_out_degree(graph) + 1)


def legality_mask(graph: WarehouseGraph, vertex: VertexId, d_max: int | None = None) -> Mask:
    """The action mask at `vertex`. Slots in U(v) get 0.0 and all others get -inf.

    Slot 0 is wait. Slot k is the k-th outgoing neighbour of `vertex` in
    increasing vertex number, so it is legal exactly when k <= |N+(v)|.
    Pass `d_max` when checking many vertices of one graph, to skip
    recomputing it.
    """
    if d_max is None:
        d_max = max_out_degree(graph)
    num_legal_moves = len(graph.legal_actions(vertex)) - 1  # exclude wait
    if num_legal_moves > d_max:
        raise ValueError(
            f"vertex {vertex} has {num_legal_moves} outgoing moves > d_max={d_max}; "
            "d_max must be max_out_degree(graph) for this graph, not an unrelated value."
        )
    return (0.0,) * (num_legal_moves + 1) + (float("-inf"),) * (d_max - num_legal_moves)


def action_for_slot(graph: WarehouseGraph, vertex: VertexId, slot: int) -> Action:
    """Decode a chosen slot into the Action it represents at `vertex`.

    Slot k follows legality_mask. Raises ValueError for a masked slot,
    since callers only sample slots the action mask leaves legal.
    """
    legal = graph.legal_actions(vertex)
    if 0 <= slot < len(legal):
        return legal[slot]
    raise ValueError(f"slot {slot} is illegal at vertex {vertex} (masked by the action mask).")


def slot_for_action(graph: WarehouseGraph, vertex: VertexId, action: Action) -> int:
    """Inverse of action_for_slot: the slot index `action` occupies at
    `vertex`. Raises ValueError if `action` isn't legal at `vertex`."""
    legal = graph.legal_actions(vertex)
    if action in legal:
        return legal.index(action)
    raise ValueError(f"{action} is not legal at vertex {vertex} (not in the action set).")
