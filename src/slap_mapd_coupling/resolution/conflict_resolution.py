"""Deterministic conflict-resolution operator (sec:method:resolution).

Turns a proposed joint action into a collision-free one. A priority
permutation sigma of {1,...,m} is drawn once per episode from the episode
seed and held fixed for that episode. A robot whose proposed move is not
accepted is overridden to `wait` (always legal in the action set) and its
override flag for t is set. That flag feeds the -n_blocked term of the
reward and defines omega_j(t) in the mean blocked time (sec:impl:logging).

[A81]: a robot is handled after the robot standing on the vertex it wants,
so it may follow a robot that leaves. Robot i depends on robot j when i
proposes to move onto l_j(t). Every robot depends on at most one other, and
a robot that waits on none. The robots are handled one at a time: of the
robots that depend on no one, or on a robot already handled, the earliest
in sigma goes next. Its move is accepted unless it breaks the vertex rule or
the swap rule against the successors already accepted, and otherwise it
waits. Robots that are never ready, because their chain of dependencies runs
into a cycle (a swap is a cycle of two), wait. So the priority order decides
between robots that compete for a vertex, and the chains decide who goes
first along an aisle.

Why it never collides:

  (a) A robot is accepted onto a vertex only if no accepted robot ends
  there (vertex rule) and, if a robot stood there at t, that robot was
  handled first (the dependency). So no accepted move ends on the vertex of
  a robot not yet handled.

  (b) A robot that waits keeps its own vertex. No robot handled before it
  moved there, by (a), and no robot handled after it starts there, since
  the robots stood apart at t. So a wait is always accepted, and the robots
  left over at the end, which all wait, keep vertices nobody moved onto.

  (c) No two robots swap. A swap needs i to move onto l_j(t) and j onto
  l_i(t), so each depends on the other, a cycle, and both wait. The swap
  check is kept as a guard.

Under the earlier rule ([A53]) a robot could not move onto the vertex of a
robot later in sigma even when that robot left, so following it cost a wait
and an override. The claim mask of the centralised and section-based
routers still uses that stricter rule (`move_blocked`), because a robot
choosing there does not yet know whether a later robot will leave. A joint
action that passes the claim mask is therefore accepted whole here.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from slap_mapd_coupling.core.agents import AgentId, AgentState, FleetState, is_legal_transition
from slap_mapd_coupling.core.graph import Action, VertexId, WarehouseGraph


def priority_permutation(seed: int, agent_ids: Sequence[AgentId]) -> tuple[AgentId, ...]:
    """sigma: a permutation of agent_ids drawn once from the episode seed.

    Callers (the environment loop, at reset) compute this once and reuse
    the same tuple for every timestep of that episode -- this function
    itself has no notion of "episode," only of seed, so it is the caller's
    responsibility not to redraw it mid-episode.
    """
    ids = list(agent_ids)
    random.Random(seed).shuffle(ids)
    return tuple(ids)


@dataclass(frozen=True)
class ResolutionResult:
    """The finalised fleet state after one timestep's conflict resolution,
    and which agents were overridden to wait at that timestep."""

    fleet: FleetState
    overridden: frozenset[AgentId]


def _successor(start: VertexId, action: Action) -> VertexId:
    if action.kind == "wait":
        return start
    assert action.target is not None  # guaranteed by Action's own validator
    return action.target


def move_blocked(
    agent_id: AgentId,
    start: VertexId,
    candidate: VertexId,
    before_loc: Mapping[AgentId, VertexId],
    finalised: Mapping[AgentId, VertexId],
) -> bool:
    """Whether `agent_id` may not go from `start` to `candidate`.

    `finalised` holds the successors of the agents that have chosen, and an
    agent missing from it counts as not having chosen. A move is blocked by
    the vertex rule, by the swap rule, or by an agent that has not chosen yet
    and stands on `candidate`. The claim mask uses it. It is stricter than
    `resolve_conflicts` (module docstring), so conflict resolution never
    overrides a centralised agent.
    """
    for other_id, other_candidate in finalised.items():
        if other_candidate == candidate:  # vertex rule
            return True
        if before_loc[other_id] == candidate and start == other_candidate:  # swap rule
            return True
    for other_id, other_start in before_loc.items():
        if other_id == agent_id or other_id in finalised:
            continue
        if other_start == candidate:
            # `other_id` currently sits at `candidate` and hasn't been
            # finalised yet -- unknown whether it will vacate, so this
            # vertex isn't safe to move into (see module docstring).
            return True
    return False


def resolve_conflicts(
    graph: WarehouseGraph,
    before: FleetState,
    proposed: Mapping[AgentId, Action],
    priority: Sequence[AgentId],
) -> ResolutionResult:
    """Apply sec:method:resolution's operator for one timestep (module docstring).

    `priority` is sigma, fixed for the whole episode (see
    priority_permutation above) -- this function does not draw it, since
    redrawing per call would violate "held fixed for the episode" and the
    determinism paired-seed comparisons across (F, pi) rely on.

    Raises ValueError if `proposed` contains an illegally-targeted action
    (masking legality is environment/spaces.py's job, upstream of this
    operator) or if `priority` isn't exactly a permutation of `before`'s
    agent ids.
    """
    before_loc = before.locations()
    if set(priority) != set(before_loc):
        raise ValueError("priority must be a permutation of exactly before's agent ids.")
    rank = {agent_id: k for k, agent_id in enumerate(priority)}
    standing_on = {v: agent_id for agent_id, v in before_loc.items()}

    candidates: dict[AgentId, VertexId] = {}
    dependants: dict[AgentId, list[AgentId]] = {}
    ready: list[tuple[int, AgentId]] = []
    for agent_id in priority:
        start = before_loc[agent_id]
        candidate = _successor(start, proposed[agent_id])
        if not is_legal_transition(graph, start, candidate):
            raise ValueError(
                f"Agent {agent_id} proposed an illegal transition {start}->{candidate} "
                "(the move rule); conflict resolution assumes already-masked proposals."
            )
        candidates[agent_id] = candidate
        ahead = standing_on.get(candidate) if candidate != start else None
        if ahead is None:
            ready.append((rank[agent_id], agent_id))
        else:
            dependants.setdefault(ahead, []).append(agent_id)
    heapq.heapify(ready)

    finalised: dict[AgentId, VertexId] = {}
    taken: set[VertexId] = set()
    overridden: set[AgentId] = set()
    while ready:
        _, agent_id = heapq.heappop(ready)
        start = before_loc[agent_id]
        candidate = candidates[agent_id]
        if candidate != start and _clashes(
            agent_id, start, candidate, before_loc, finalised, taken
        ):
            candidate = start
            overridden.add(agent_id)
        if candidate == start and start in taken:
            raise RuntimeError(
                f"Agent {agent_id} cannot wait at {start}, another robot was accepted onto it. "
                "The dependency order rules this out (module docstring), so this is a "
                "correctness bug (AGENTS.md), not something to paper over here."
            )
        finalised[agent_id] = candidate
        taken.add(candidate)
        for follower in dependants.get(agent_id, ()):
            heapq.heappush(ready, (rank[follower], follower))

    # Robots whose chain runs into a cycle were never ready. They wait, on vertices nobody
    # was accepted onto (module docstring, (b)).
    for agent_id in priority:
        if agent_id in finalised:
            continue
        start = before_loc[agent_id]
        if start in taken:
            raise RuntimeError(
                f"Agent {agent_id} is left in a cycle but its vertex {start} was taken, "
                "a correctness bug (AGENTS.md)."
            )
        finalised[agent_id] = start
        taken.add(start)
        if candidates[agent_id] != start:
            overridden.add(agent_id)

    after = FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in finalised.items()}
    )
    return ResolutionResult(fleet=after, overridden=frozenset(overridden))


def _clashes(
    agent_id: AgentId,
    start: VertexId,
    candidate: VertexId,
    before_loc: Mapping[AgentId, VertexId],
    finalised: Mapping[AgentId, VertexId],
    taken: set[VertexId],
) -> bool:
    """Whether a move breaks the vertex rule or the swap rule against the robots already
    handled. The robot on `candidate` at t, if any, is among them (the dependency)."""
    if candidate in taken:  # vertex rule
        return True
    return any(  # swap rule, a guard, since a swap is a cycle and never handled
        before_loc[other] == candidate and finalised[other] == start for other in finalised
    )
