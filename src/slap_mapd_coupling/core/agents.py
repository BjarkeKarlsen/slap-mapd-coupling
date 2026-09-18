"""Agent fleet state: positions ell_i(t) and the legal per-vertex action set U(v)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph

AgentId = int


class AgentState(BaseModel):
    """ell_i(t): one agent's position at one timestep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: AgentId
    location: VertexId


class FleetState(BaseModel):
    """The agent-position portion of s_t (eq:posg).

    Task lifecycle sets and the storage state x_t live in Task /
    StorageState, not duplicated here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agents: dict[AgentId, AgentState]

    @model_validator(mode="after")
    def _agent_id_matches_key(self) -> "FleetState":
        for aid, state in self.agents.items():
            if aid != state.agent_id:
                raise ValueError(f"Fleet key {aid} != AgentState.agent_id {state.agent_id}.")
        return self

    def locations(self) -> dict[AgentId, VertexId]:
        return {aid: state.location for aid, state in self.agents.items()}


def is_legal_transition(graph: WarehouseGraph, before: VertexId, after: VertexId) -> bool:
    """eq:transition: after == before (wait), or (before, after) in E."""
    if after == before:
        return True
    return after in graph.out_neighbours(before)


def has_vertex_conflict(before: FleetState, after: FleetState) -> AgentId | None:
    """eq:vertexconflict: two agents ending the timestep at the same vertex.

    Returns one offending agent id, or None if there is no conflict.
    """
    seen: dict[VertexId, AgentId] = {}
    for agent_id, vertex in after.locations().items():
        if vertex in seen:
            return agent_id
        seen[vertex] = agent_id
    return None


def has_swap_conflict(before: FleetState, after: FleetState) -> tuple[AgentId, AgentId] | None:
    """eq:swapconflict: two agents trading places across a single edge in one timestep."""
    before_loc = before.locations()
    after_loc = after.locations()
    agent_ids = list(after_loc)
    for i, a in enumerate(agent_ids):
        for b in agent_ids[i + 1 :]:
            if before_loc[a] == after_loc[b] and before_loc[b] == after_loc[a]:
                return (a, b)
    return None


def is_collision_free(before: FleetState, after: FleetState) -> bool:
    """Conjunction of eq:vertexconflict and eq:swapconflict."""
    return has_vertex_conflict(before, after) is None and has_swap_conflict(before, after) is None
