"""Per-step episode trace, for finding out what the agents actually did.

An episode ends in a handful of numbers (tasks completed, mean service
time). When one of them looks wrong, the trace is what shows why: where
every agent stood at every timestep, what it tried to do, whether conflict
resolution overrode it, and which task and goal it was serving.

The trace is a JSON Lines file. The first line is a header with the
configuration, the vertices with their roles and the edges, so the file
can be read, summarised or drawn without the instance that produced it.
Every later line is one timestep, written after the step has been applied.
A line gives the state at time t, and the intent and overridden flag of
the step that led to it.

The writer only reads the environment, so it works the same for the
reference planner, the learned controllers and a run driven by RLlib.

Summarise a file from the shell with

    python -m slap_mapd_coupling.evaluation.trace trace.jsonl [more.jsonl ...] [--window N]

With more than one file it also prints the fraction of episodes with a stuck
agent and the fraction with a wandering agent (--min-progress sets the share of
moves that must bring an agent closer to its goal).
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable

from slap_mapd_coupling.core.tasks import active_tasks_by_agent
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv

TRACE_FORMAT = 1


class EpisodeTraceWriter:
    """Writes one trace file. Call `record(env)` after every `env.step()`."""

    def __init__(self, path: Path, env: WarehouseMAPDEnv) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._previous = env.fleet.locations()
        graph = env.graph
        header = {
            "type": "header",
            "format": TRACE_FORMAT,
            "config": env.config.model_dump(mode="json"),
            "vertices": [
                {
                    "id": v,
                    "storage": vertex.role.storage,
                    "delivery": vertex.role.delivery,
                    "endpoint": vertex.role.endpoint,
                }
                for v, vertex in sorted(graph.vertices.items())
            ],
            "edges": [[e.source, e.target, e.cost] for e in graph.edges],
            "initial": {str(a): v for a, v in self._previous.items()},
        }
        self._write(header)

    def record(self, env: WarehouseMAPDEnv) -> None:
        t = env.t
        locations = env.fleet.locations()
        # The task an agent served during the step that just ended. A task
        # assigned or finished at t was active at t - 1.
        serving = active_tasks_by_agent(env.tasks, t - 1)
        agents: dict[str, dict[str, Any]] = {}
        for agent_id, at in locations.items():
            action = env.last_proposed.get(agent_id)
            task = serving.get(agent_id)
            goal: int | None = None
            phase: str | None = None
            if task is not None:
                picked_up = task.pickup_time is not None and task.pickup_time < t
                phase = "to_delivery" if picked_up else "to_pickup"
                goal = task.delivery_vertex if picked_up else task.pickup_vertex
            agents[str(agent_id)] = {
                "at": at,
                "intent": action.target if action is not None and action.kind == "move" else None,
                "moved": at != self._previous[agent_id],
                "overridden": agent_id in env.last_overridden,
                "task": None if task is None else task.task_id,
                "phase": phase,
                "goal": goal,
            }
        self._previous = locations
        self._write(
            {
                "type": "step",
                "t": t,
                "completed": sum(1 for task in env.tasks if task.finish_time is not None),
                "released": len(env.tasks),
                "backlog": env.log.backlog_by_t[-1],
                "agents": agents,
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "EpisodeTraceWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write(self, line: dict[str, Any]) -> None:
        self._file.write(json.dumps(line, separators=(",", ":")) + "\n")


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(header, steps) from a trace file."""
    with path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path} is not an episode trace (no header line).")
    if lines[0]["format"] != TRACE_FORMAT:
        raise ValueError(f"{path} has trace format {lines[0]['format']}, expected {TRACE_FORMAT}.")
    return lines[0], lines[1:]


@dataclass(frozen=True)
class AgentSummary:
    agent_id: int
    moves: int
    moves_in_window: int
    overridden: int
    longest_stand_still: int
    final_vertex: int
    final_task: int | None
    final_goal: int | None
    # Moves in the window made while holding a task, and the share of them
    # that brought the agent closer to its goal. None when there were none.
    goal_moves_in_window: int
    progress_in_window: float | None
    distinct_vertices_in_window: int


@dataclass(frozen=True)
class TraceSummary:
    steps: int
    window: int
    completed: int
    released: int
    backlog: int
    completed_at_quarters: tuple[int, ...]
    last_completion_t: int | None
    agents: tuple[AgentSummary, ...]
    # Agents holding a task that did not move once in the window.
    stuck_agents: tuple[int, ...]
    # Agents holding a task that moved in the window but mostly not towards
    # the goal, below `min_progress`.
    wandering_agents: tuple[int, ...]
    min_progress: float
    # (agent, vertex, wanted vertex, agent holding it or None) at the last step.
    waiting_on: tuple[tuple[int, int, int, int | None], ...]
    # vertex -> its out-neighbours, for the vertices the stuck agents stand on.
    exits: dict[int, tuple[int, ...]] = field(default_factory=dict)


def _distances_to(goal: int, header: dict[str, Any], cache: dict[int, dict[int, int]]):
    """Hop counts from every vertex to `goal`, along the directed edges."""
    if goal not in cache:
        into: dict[int, list[int]] = {}
        for source, target, _ in header["edges"]:
            into.setdefault(target, []).append(source)
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            current = queue.popleft()
            for source in into.get(current, ()):
                if source not in dist:
                    dist[source] = dist[current] + 1
                    queue.append(source)
        cache[goal] = dist
    return cache[goal]


def summarise_trace(
    header: dict[str, Any],
    steps: list[dict[str, Any]],
    window: int | None = None,
    min_progress: float = 0.5,
) -> TraceSummary:
    """Reduce a trace to the facts that show whether the fleet stalled.

    `window` is how many final steps count as "recently". It defaults to a
    quarter of the episode. `min_progress` is the share of goal-holding moves
    that must bring an agent closer to its goal, below which it is listed as
    wandering. Both are display settings for a diagnostic, not study
    parameters.
    """
    if not steps:
        raise ValueError("the trace has no steps.")
    n = len(steps)
    window = max(1, n // 4) if window is None else min(max(1, window), n)
    completed = [s["completed"] for s in steps]
    last_completion: int | None = None
    for i, count in enumerate(completed):
        if count > (completed[i - 1] if i else 0):
            last_completion = steps[i]["t"]

    final = steps[-1]
    agents: list[AgentSummary] = []
    distance_cache: dict[int, dict[int, int]] = {}
    for key in sorted(final["agents"], key=int):
        records = [s["agents"][key] for s in steps]
        # Closer to the goal, counted over the moves in the window that were made
        # while holding a task. A vertex with no path to the goal gives no verdict.
        closer = goal_moves = 0
        for index in range(n - window, n):
            r = records[index]
            before = records[index - 1]["at"] if index else header.get("initial", {}).get(key)
            if not r["moved"] or r["goal"] is None or before is None:
                continue
            dist = _distances_to(r["goal"], header, distance_cache)
            if before in dist and r["at"] in dist:
                goal_moves += 1
                closer += dist[r["at"]] < dist[before]
        longest = run = 0
        for r in records:
            run = 0 if r["moved"] else run + 1
            longest = max(longest, run)
        agents.append(
            AgentSummary(
                agent_id=int(key),
                moves=sum(r["moved"] for r in records),
                moves_in_window=sum(r["moved"] for r in records[-window:]),
                overridden=sum(r["overridden"] for r in records),
                longest_stand_still=longest,
                final_vertex=records[-1]["at"],
                final_task=records[-1]["task"],
                final_goal=records[-1]["goal"],
                goal_moves_in_window=goal_moves,
                progress_in_window=closer / goal_moves if goal_moves else None,
                distinct_vertices_in_window=len({r["at"] for r in records[-window:]}),
            )
        )

    stuck = tuple(a.agent_id for a in agents if a.moves_in_window == 0 and a.final_task is not None)
    wandering = tuple(
        a.agent_id
        for a in agents
        if a.final_task is not None
        and a.progress_in_window is not None
        and a.progress_in_window < min_progress
    )
    holder = {r["at"]: int(k) for k, r in final["agents"].items()}
    waiting_on = tuple(
        (int(k), r["at"], r["intent"], holder.get(r["intent"]))
        for k, r in sorted(final["agents"].items(), key=lambda kv: int(kv[0]))
        if r["intent"] is not None and not r["moved"]
    )
    out: dict[int, list[int]] = {}
    for source, target, _ in header["edges"]:
        out.setdefault(source, []).append(target)
    exits = {
        a.final_vertex: tuple(sorted(out.get(a.final_vertex, ())))
        for a in agents
        if a.agent_id in stuck
    }
    quarters = tuple(completed[max(0, (n * q) // 4 - 1)] for q in (1, 2, 3, 4))
    return TraceSummary(
        steps=n,
        window=window,
        completed=completed[-1],
        released=final["released"],
        backlog=final["backlog"],
        completed_at_quarters=quarters,
        last_completion_t=last_completion,
        agents=tuple(agents),
        stuck_agents=stuck,
        wandering_agents=wandering,
        min_progress=min_progress,
        waiting_on=waiting_on,
        exits=exits,
    )


def _progress_text(a: AgentSummary) -> str:
    if a.progress_in_window is None:
        share = "no goal moves"
    else:
        share = f"{a.progress_in_window:.0%} of {a.goal_moves_in_window} goal moves closer"
    return f"{share}, {a.distinct_vertices_in_window} distinct vertices in the window"


def format_summary(summary: TraceSummary) -> str:
    s = summary
    lines = [
        f"{s.steps} steps, {s.completed} of {s.released} released tasks completed, "
        f"backlog {s.backlog}",
        "completed at 25/50/75/100% of the episode: "
        + " / ".join(str(c) for c in s.completed_at_quarters),
        "last completion at t = "
        + ("never" if s.last_completion_t is None else str(s.last_completion_t)),
        f"per agent (window = last {s.window} steps):",
    ]
    for a in s.agents:
        task = "no task" if a.final_task is None else f"task {a.final_task} -> goal {a.final_goal}"
        lines.append(
            f"  agent {a.agent_id}: {a.moves} moves ({a.moves_in_window} in the window), "
            f"overridden {a.overridden}x, longest stand still {a.longest_stand_still}, "
            f"{_progress_text(a)}, ends at {a.final_vertex} with {task}"
        )
    if s.stuck_agents:
        lines.append(f"STUCK, holding a task and not moving in the window: {list(s.stuck_agents)}")
        for vertex, exits in s.exits.items():
            lines.append(f"  vertex {vertex} has out-neighbours {list(exits)}")
    if s.wandering_agents:
        lines.append(
            f"WANDERING, holding a task and under {s.min_progress:.0%} of moves closer to the "
            f"goal in the window: {list(s.wandering_agents)}"
        )
    for agent, vertex, wanted, other in s.waiting_on:
        who = "nobody" if other is None else f"agent {other}"
        lines.append(f"  at the last step agent {agent} at {vertex} wants {wanted}, held by {who}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise episode traces.")
    parser.add_argument("traces", type=Path, nargs="+", metavar="FILE")
    parser.add_argument("--window", type=int, default=None, help="final steps counted as recent")
    parser.add_argument(
        "--min-progress", type=float, default=0.5, help="share of moves closer to the goal"
    )
    args = parser.parse_args(None if argv is None else list(argv))
    summaries = []
    for path in args.traces:
        header, steps = load_trace(path)
        summary = summarise_trace(header, steps, args.window, args.min_progress)
        summaries.append(summary)
        print(f"== {path}")
        print(format_summary(summary))
    if len(summaries) > 1:
        stuck = sum(1 for s in summaries if s.stuck_agents)
        wandering = sum(1 for s in summaries if s.wandering_agents)
        total = len(summaries)
        print(f"\n{stuck} of {total} episodes had a stuck agent ({stuck / total:.0%})")
        share = wandering / total
        print(f"{wandering} of {total} episodes had a wandering agent ({share:.0%})")


if __name__ == "__main__":
    main()
