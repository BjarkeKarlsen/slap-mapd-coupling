"""Action space and the per-vertex legality mask (sec:method:rl, eq:mask).

This is the mask itself -- shape and legality logic -- independent of
RLlib. Wiring it into RLlib's newer multi-agent API (which ships no
action masking built-in) is a separate concern, budgeted under M5's
"action masking as a custom RLlib module/connector," not here.
"""

from __future__ import annotations

from gymnasium.spaces import Discrete

from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph

Mask = tuple[float, ...]  # M(v, .): d_max+1 entries, 0.0 (legal) or -inf (illegal)


def max_out_degree(graph: WarehouseGraph) -> int:
    """d_max = max_{v in V_mov} |N+(v)| (eq:mask)."""
    movable = (v for v, vertex in graph.vertices.items() if vertex.role.movable)
    return max((len(graph.out_neighbours(v)) for v in movable), default=0)


def action_space(graph: WarehouseGraph) -> Discrete:
    """The shared per-agent action space: d_max+1 discrete slots (eq:mask) --
    one wait slot plus d_max move slots. Shared across every vertex and
    every agent, since eq:mask's whole point is a fixed-size policy head;
    see legality_mask below for which slots are actually legal at a given
    vertex, and action_for_slot for decoding a chosen slot into an Action.
    """
    return Discrete(max_out_degree(graph) + 1)


def legality_mask(graph: WarehouseGraph, vertex: VertexId, d_max: int | None = None) -> Mask:
    """M(v, .) (eq:mask): 0.0 for slots in U(v), -inf for slots in U_i \\ U(v).

    Slot 0 is always wait (legal everywhere, eq:actions). Slots 1..d_max
    are move slots; which real vertex slot k decodes to for THIS v is
    graph.legal_actions(v)'s own move order (its established, existing
    canonical enumeration -- reused here rather than inventing a second
    one), so slot k is legal at v exactly when 1 <= k <= |N+(v)|. The
    thesis does not specify a slot-assignment rule beyond "d_max slots";
    reusing legal_actions' order is the minimal choice, not an arbitrary
    new one, and action_for_slot/slot_for_action below are the single
    place that convention lives.

    Pass `d_max` when checking many vertices of the same graph, to avoid
    recomputing max_out_degree per call; it defaults to computing it
    fresh, which is always correct, just not free.
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
    """Decode a chosen slot index into the Action it represents at `vertex`,
    using legal_actions(v)'s order (see legality_mask). Raises ValueError
    for a masked (illegal) slot -- callers are expected to only ever
    sample a slot the mask marked legal, eq:mask's entire purpose."""
    legal = graph.legal_actions(vertex)
    if 0 <= slot < len(legal):
        return legal[slot]
    raise ValueError(f"slot {slot} is illegal at vertex {vertex} (masked, eq:mask).")


def slot_for_action(graph: WarehouseGraph, vertex: VertexId, action: Action) -> int:
    """Inverse of action_for_slot: the slot index `action` occupies at
    `vertex`. Raises ValueError if `action` isn't legal at `vertex`."""
    legal = graph.legal_actions(vertex)
    if action in legal:
        return legal.index(action)
    raise ValueError(f"{action} is not legal at vertex {vertex} (eq:actions).")
