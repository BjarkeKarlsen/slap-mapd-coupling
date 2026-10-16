"""Action space and the action mask.

This is the mask logic itself, independent of RLlib. The RLlib wiring is in
`action_masking.py`.
"""

from __future__ import annotations

from typing import Mapping

from gymnasium.spaces import Discrete

from slap_mapd_coupling.core.agents import AgentId
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph
from slap_mapd_coupling.resolution.conflict_resolution import move_blocked

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


def claim_mask(
    graph: WarehouseGraph,
    agent_id: AgentId,
    before_locations: Mapping[AgentId, VertexId],
    chosen: Mapping[AgentId, VertexId],
    d_max: int | None = None,
) -> Mask:
    """The claim mask of the agent inside a centralised or section decision.

    `chosen` holds the successor vertex of each earlier agent in the same
    unit, an agent that waits included. An agent missing from it counts as
    not having chosen, and so do agents in other zones. A move is switched
    off when `move_blocked` rejects it, and the wait slot is never touched.
    The result has the shape of `legality_mask`, with -inf on every slot the
    claim mask switches off and 0.0 elsewhere, so the two masks combine by
    adding.
    """
    if d_max is None:
        d_max = max_out_degree(graph)
    start = before_locations[agent_id]
    if move_blocked(agent_id, start, start, before_locations, chosen):
        raise RuntimeError(
            f"Agent {agent_id} cannot wait at {start}. A collision-free state and "
            "earlier choices that each passed the claim mask rule this out, so this "
            "is a correctness bug."
        )
    mask = [0.0] * (d_max + 1)
    for slot, action in enumerate(graph.legal_actions(start)):
        if action.kind == "move" and move_blocked(
            agent_id, start, action.target, before_locations, chosen
        ):
            mask[slot] = float("-inf")
    return tuple(mask)
