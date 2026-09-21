"""Deterministic conflict-resolution operator (sec:method:resolution).

Turns a proposed joint action into a collision-free one. A priority
permutation sigma of {1,...,m} is drawn once per episode from the episode
seed and held fixed for that episode. At each timestep, agents are
processed in sigma order: each one's proposed successor is tentatively
accepted unless it violates eq:vertexconflict or eq:swapconflict against
the already-finalised successors of higher-priority agents, in which case
it is overridden to `wait` (always legal, eq:actions) and its override
flag for t is set. That flag feeds R_i(t)'s -r_blk term (eq:reward) and is
what defines omega_j(t) (eq:waiting); see sec:impl:logging -- log it once,
here, rather than re-deriving it at either call site.

Deviation from the thesis text (flagged, not silently patched -- see the
PR for issue #7): the text's own feasibility argument for step 3 ("waiting
never conflicts with an already-finalised successor other than by
occupying the current vertex, impossible since the joint state at t was
collision-free") conflates "no one occupies v at t" with "no one's
successor targets v at t+1." It does not, and cannot rule out a
higher-priority agent moving INTO a lower-priority agent's current
vertex before that agent is processed: the two-agent head-on swap
(v1<->v2, both propose to trade places) is the concrete counterexample --
agent 1 (higher priority) is accepted onto v2 with nothing yet to check
against, agent 2 is then correctly caught as a swap conflict and
"overridden to wait," but waiting means staying at v2, which agent 1 has
already claimed: a fresh vertex conflict the text's proof says cannot
happen. This implementation closes the gap by also reserving each other
agent's *current* vertex against being moved into until that agent has
itself been finalised as having vacated it (`_blocked`'s second check
below) -- still a single deterministic pass over sigma, still a pure
function of state and seed, just conservative about chains of moves that
depend on a not-yet-processed agent's own choice.

With that reservation in place, "waiting always terminates in a feasible
joint action" (sec:pf:agents's Figure~2.x(c), sec:method:resolution) can
actually be shown, not just asserted, by induction over the order agents
are processed in:

  Inductive hypothesis: after processing agents sigma(1),...,sigma(k-1),
  `finalised` is pairwise collision-free (no eq:vertexconflict or
  eq:swapconflict among any two of them) -- vacuously true for k=1.

  Step: when agent sigma(k) is processed, `_blocked` either accepts its
  proposed candidate (then it doesn't conflict with any already-finalised
  successor by construction, and `finalised` for k agents is still
  pairwise collision-free by the hypothesis plus this one check), or
  rejects it and candidate falls back to `start` = before_loc[sigma(k)].
  This fallback cannot itself be blocked: (a) no already-finalised
  successor can equal `start`, because any earlier agent that tried to
  move there would itself have been rejected by the current-vertex
  reservation while sigma(k) still held it (sigma(k) is only finalised
  now, at step k); (b) no not-yet-finalised agent's current vertex can
  equal `start` either, because `before` is collision-free -- at most one
  agent occupies any vertex at t, and that's sigma(k) itself. So the
  fallback is always accepted, and the hypothesis holds for k agents too.

  After all m agents: `finalised` is pairwise collision-free, i.e. `after`
  satisfies eq:vertexconflict and eq:swapconflict for every pair -- this
  is what the RuntimeError in resolve_conflicts asserts is unreachable,
  and this argument is why.
"""

from __future__ import annotations

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


def _blocked(
    agent_id: AgentId,
    start: VertexId,
    candidate: VertexId,
    before_loc: Mapping[AgentId, VertexId],
    finalised: Mapping[AgentId, VertexId],
) -> bool:
    for other_id, other_candidate in finalised.items():
        if other_candidate == candidate:  # eq:vertexconflict
            return True
        if before_loc[other_id] == candidate and start == other_candidate:  # eq:swapconflict
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
    """Apply sec:method:resolution's three-step operator for one timestep.

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

    finalised: dict[AgentId, VertexId] = {}
    overridden: set[AgentId] = set()

    for agent_id in priority:
        start = before_loc[agent_id]
        action = proposed[agent_id]
        candidate = _successor(start, action)
        if not is_legal_transition(graph, start, candidate):
            raise ValueError(
                f"Agent {agent_id} proposed an illegal transition {start}->{candidate} "
                "(eq:transition); conflict resolution assumes already-masked proposals."
            )

        if _blocked(agent_id, start, candidate, before_loc, finalised):
            candidate = start
            overridden.add(agent_id)
            if _blocked(agent_id, start, candidate, before_loc, finalised):
                raise RuntimeError(
                    f"Agent {agent_id} still conflicts after being overridden to wait at "
                    f"{start}. With the current-vertex reservation in place (see module "
                    "docstring) this should be unreachable given a collision-free `before` "
                    "state -- treat this as a correctness bug (AGENTS.md), not something to "
                    "paper over here."
                )

        finalised[agent_id] = candidate

    after = FleetState(
        agents={aid: AgentState(agent_id=aid, location=loc) for aid, loc in finalised.items()}
    )
    return ResolutionResult(fleet=after, overridden=frozenset(overridden))
