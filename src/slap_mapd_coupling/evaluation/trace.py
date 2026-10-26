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

Format 2 adds three things. The header also holds the drawing positions
when the writer is given them, the shelf capacities and the unit size of
every SKU. Every `stock_every` steps a stock line follows the step line,
with the units per shelf by SKU, the popularity ranking at t and the tasks
that wait on each shelf. The last line lists every task with its times,
its vertices and its SKU. A format 1 file has none of these and is still
read.

The writer only reads the environment, so it works the same for the
reference planner, the learned controllers and a run driven by RLlib.

Summarise a file from the shell with

    python -m slap_mapd_coupling.evaluation.trace trace.jsonl [more.jsonl ...] [--window N]

With more than one file it also prints the fraction of episodes with a stuck
agent and the fraction with a wandering agent (--min-progress sets the share of
moves that must bring an agent closer to its goal) and the share of goal moves,
pooled over all agents and episodes, that do bring one closer. It also prints the longest
whole-fleet still run, which is the number to choose --dead-after from. With
--dead-after N (default 50) it also prints the fraction of episodes where the fleet
was dead for N steps.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping

from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.tasks import Task, active_tasks_by_agent
from slap_mapd_coupling.environment.multi_agent_env import WarehouseMAPDEnv

TRACE_FORMAT = 2
READABLE_FORMATS = (1, 2)

# Steps between two stock lines when the writer is not told otherwise. The stock changes at
# every pick and every put-away delivery, so a line every step would be exact, but at the
# Welch horizon it would make a trace of several megabytes. One line every 10 steps is
# enough for the stock figure and for a replay. A display setting, not a study parameter.
DEFAULT_STOCK_EVERY = 10

# Finished tasks listed on every step line, newest last.
_RECENT = 5

# Steps of standing still that the command line counts as dead when --dead-after is not
# given. It is a display setting for reading a trace, not a study parameter. It is about
# 3.5 times the longest whole-fleet still run seen in a healthy episode (14 steps, example
# 04 after 7 training iterations), and that is one episode of a barely trained policy, so
# a trained policy may wait longer. The study value, if there is one, is still open.
DEFAULT_DEAD_AFTER = 50


class EpisodeTraceWriter:
    """Writes one trace file. Call `record(env)` after every `env.step()`."""

    def __init__(
        self,
        path: Path,
        env: WarehouseMAPDEnv,
        *,
        positions: Mapping[VertexId, tuple[float, float]] | None = None,
        stock_every: int = DEFAULT_STOCK_EVERY,
    ) -> None:
        if stock_every < 1:
            raise ValueError(f"stock_every must be at least 1, got {stock_every}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._previous = env.fleet.locations()
        self._stock_every = stock_every
        self._env = env
        self._write(header_line(env, positions))
        self._write(stock_line(env))

    def record(self, env: WarehouseMAPDEnv) -> None:
        t = env.t
        self._write(step_line(env, self._previous))
        self._previous = env.fleet.locations()
        if t % self._stock_every == 0:
            self._write(stock_line(env))

    def close(self) -> None:
        if not self._file.closed:
            self._write({"type": "tasks", "tasks": [_task_entry(x) for x in self._env.tasks]})
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


def header_line(
    env: WarehouseMAPDEnv,
    positions: Mapping[VertexId, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """The first line of a trace: the configuration and the warehouse, so a file can be read
    or drawn without the instance that produced it."""
    graph = env.graph
    storage = env.storage
    return {
        "type": "header",
        "format": TRACE_FORMAT,
        "config": env.config.model_dump(mode="json"),
        "vertices": [
            {
                "id": v,
                "storage": vertex.role.storage,
                "delivery": vertex.role.delivery,
                "endpoint": vertex.role.endpoint,
                "receiving": vertex.role.receiving,
            }
            for v, vertex in sorted(graph.vertices.items())
        ],
        "edges": [[e.source, e.target, e.cost] for e in graph.edges],
        "initial": {str(a): v for a, v in env.fleet.locations().items()},
        "positions": (
            None
            if positions is None
            else {str(v): [float(x), float(y)] for v, (x, y) in sorted(positions.items())}
        ),
        "capacities": {str(v): c for v, c in sorted(storage.capacities.items())},
        "skus": {k: sku.unit_capacity for k, sku in storage.skus.items()},
    }


def step_line(env: WarehouseMAPDEnv, previous: Mapping[int, VertexId]) -> dict[str, Any]:
    """The line of the step that just ended, `previous` being where the agents stood before
    it."""
    t = env.t
    # The task an agent served during the step that just ended. A task
    # assigned or finished at t was active at t - 1. A task released at t
    # (the end of the step releases the next orders) has no status at t - 1.
    serving = active_tasks_by_agent([x for x in env.tasks if x.release_time < t], t - 1)
    agents: dict[str, dict[str, Any]] = {}
    for agent_id, at in env.fleet.locations().items():
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
            "moved": at != previous.get(agent_id, at),
            "overridden": agent_id in env.last_overridden,
            "task": None if task is None else task.task_id,
            "kind": None if task is None else task.kind,
            "sku": None if task is None else task.sku,
            "quantity": None if task is None else task.quantity,
            "congestion": (
                None
                if agent_id not in env.congestion_by_agent
                else round(env.congestion_by_agent[agent_id], 3)
            ),
            "phase": phase,
            "goal": goal,
        }
    # An order is done once every pick of it has finished and none of its lines still waits
    # for stock. The last few tasks finished are listed for the replay's side panel.
    short = {line.order_id for line in env.waiting_orders}
    open_orders: set[int] = set(short)
    orders: set[int] = set(short)
    finished: list[Task] = []
    for task in env.tasks:
        if task.finish_time is not None:
            finished.append(task)
        if task.kind != "pick" or task.order_id is None:
            continue
        orders.add(task.order_id)
        if task.finish_time is None:
            open_orders.add(task.order_id)
    finished.sort(key=lambda task: (task.finish_time, task.task_id))
    return {
        "type": "step",
        "t": t,
        "completed": len(finished),
        "released": len(env.tasks),
        "backlog": env.log.backlog_by_t[-1] if env.log.backlog_by_t else 0,
        "orders_done": len(orders - open_orders),
        "orders_released": len(orders),
        "recent": [
            [
                task.finish_time,
                task.kind,
                task.sku,
                task.quantity,
                task.pickup_vertex,
                task.delivery_vertex,
                task.assigned_agent,
            ]
            for task in finished[-_RECENT:]
        ],
        "agents": agents,
    }


def stock_line(env: WarehouseMAPDEnv) -> dict[str, Any]:
    """The stock on every shelf at env.t and the tasks that wait on it. A pick waits on its
    pickup shelf until it is picked up, `queued` while no agent holds it and `coming` once
    one does. A put-away waits on its shelf until it is delivered, and `incoming` holds the
    units by SKU it brings. Stock from outside waits at its dock until an agent picks it up,
    listed under `docks` by SKU. Only shelves and docks with something to report are
    listed."""
    t = env.t
    shelves: dict[VertexId, dict[str, Any]] = {}
    docks: dict[VertexId, dict[str, int]] = {}

    def shelf(v: VertexId) -> dict[str, Any]:
        return shelves.setdefault(
            v, {"units": {}, "queued": 0, "coming": 0, "putaways": 0, "incoming": {}}
        )

    for sku_id, per_vertex in env.storage.counts.items():
        for v, count in per_vertex.items():
            if count:
                shelf(v)["units"][sku_id] = count
    for task in env.tasks:
        if task.release_time > t:
            continue
        if task.kind == "pick" and (task.pickup_time is None or task.pickup_time > t):
            held = task.assignment_time is not None and task.assignment_time <= t
            shelf(task.pickup_vertex)["coming" if held else "queued"] += 1
        elif task.kind == "putaway" and (task.finish_time is None or task.finish_time > t):
            entry = shelf(task.delivery_vertex)
            entry["putaways"] += 1
            entry["incoming"][task.sku] = entry["incoming"].get(task.sku, 0) + task.quantity
            if task.pickup_time is None or task.pickup_time > t:
                at_dock = docks.setdefault(task.pickup_vertex, {})
                at_dock[task.sku] = at_dock.get(task.sku, 0) + task.quantity
    return {
        "type": "stock",
        "t": t,
        "ranking": env.ranking,
        "orders_short": len(env.waiting_orders),
        "shelves": {str(v): entry for v, entry in sorted(shelves.items())},
        "docks": {str(v): units for v, units in sorted(docks.items())},
    }


def _task_entry(task: Task) -> dict[str, Any]:
    return {
        "id": task.task_id,
        "kind": task.kind,
        "sku": task.sku,
        "quantity": task.quantity,
        "from": task.pickup_vertex,
        "to": task.delivery_vertex,
        "released": task.release_time,
        "assigned": task.assignment_time,
        "agent": task.assigned_agent,
        "pickup_at": task.pickup_time,
        "finished": task.finish_time,
    }


def _read_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path} is not an episode trace (no header line).")
    if lines[0]["format"] not in READABLE_FORMATS:
        raise ValueError(
            f"{path} has trace format {lines[0]['format']}, expected one of {READABLE_FORMATS}."
        )
    return lines


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(header, steps) from a trace file, the step lines only."""
    lines = _read_lines(path)
    return lines[0], [line for line in lines[1:] if line.get("type") == "step"]


def load_stock(path: Path) -> list[dict[str, Any]]:
    """The stock lines of a trace, in step order. Empty for a format 1 file."""
    return [line for line in _read_lines(path)[1:] if line.get("type") == "stock"]


def load_tasks(path: Path) -> list[dict[str, Any]] | None:
    """The task list of a trace's last line, or None for a file without one."""
    for line in reversed(_read_lines(path)[1:]):
        if line.get("type") == "tasks":
            return line["tasks"]
    return None


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
    # The longest run of steps in which no agent moved while some agent held a task.
    # Read it off healthy runs to choose `dead_after`.
    longest_fleet_still_run: int
    # With `dead_after` = N, the first time t at which the whole fleet had stood still for
    # N steps in a row while a task was active, and the first (agent, t) at which one
    # agent holding a task had not moved for N steps. None when it never happened or
    # when `dead_after` was not given.
    dead_after: int | None
    fleet_dead_at: int | None
    first_agent_dead: tuple[int, int] | None
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
    dead_after: int | None = None,
) -> TraceSummary:
    """Reduce a trace to the facts that show whether the fleet stalled.

    `window` is how many final steps count as "recently". It defaults to a
    quarter of the episode. `min_progress` is the share of goal-holding moves
    that must bring an agent closer to its goal, below which it is listed as
    wandering. Both are display settings for a diagnostic, not study
    parameters. `dead_after` is off unless given, and the command line passes
    DEFAULT_DEAD_AFTER. The thesis leaves the study value open. Without it the
    summary still reports the longest run in which the whole fleet stood
    still, which is the number to choose it from.
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
    fleet_run = longest_fleet_run = 0
    fleet_dead_at: int | None = None
    agent_run = {key: 0 for key in final["agents"]}
    first_agent_dead: tuple[int, int] | None = None
    for step in steps:
        records = step["agents"]
        busy = any(r["task"] is not None for r in records.values())
        fleet_run = fleet_run + 1 if busy and not any(r["moved"] for r in records.values()) else 0
        longest_fleet_run = max(longest_fleet_run, fleet_run)
        if dead_after is not None:
            if fleet_dead_at is None and fleet_run >= dead_after:
                fleet_dead_at = step["t"]
            for key, r in records.items():
                held_still = r["task"] is not None and not r["moved"]
                agent_run[key] = agent_run[key] + 1 if held_still else 0
                if first_agent_dead is None and agent_run[key] >= dead_after:
                    first_agent_dead = (int(key), step["t"])
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
        longest_fleet_still_run=longest_fleet_run,
        dead_after=dead_after,
        fleet_dead_at=fleet_dead_at,
        first_agent_dead=first_agent_dead,
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
    lines.append(
        f"longest run with the whole fleet standing still and a task active: "
        f"{s.longest_fleet_still_run}"
    )
    if s.dead_after is not None:
        when_fleet = "never" if s.fleet_dead_at is None else f"t = {s.fleet_dead_at}"
        if s.first_agent_dead is None:
            when_agent = "never"
        else:
            when_agent = f"agent {s.first_agent_dead[0]} at t = {s.first_agent_dead[1]}"
        lines.append(
            f"dead for {s.dead_after} steps, whole fleet: {when_fleet}, "
            f"first agent holding a task: {when_agent}"
        )
    if s.wandering_agents:
        lines.append(
            f"WANDERING, holding a task and under {s.min_progress:.0%} of moves closer to the "
            f"goal in the window: {list(s.wandering_agents)}"
        )
    for agent, vertex, wanted, other in s.waiting_on:
        who = "nobody" if other is None else f"agent {other}"
        lines.append(f"  at the last step agent {agent} at {vertex} wants {wanted}, held by {who}")
    return "\n".join(lines)


def pooled_progress(summaries: Iterable[TraceSummary]) -> float | None:
    """Share of goal-holding moves that bring an agent closer, over all agents and episodes.

    Moves are counted in each summary's window. Returns None when no agent made one.
    """
    closer = goal_moves = 0
    for summary in summaries:
        for a in summary.agents:
            if a.progress_in_window is not None:
                closer += round(a.progress_in_window * a.goal_moves_in_window)
                goal_moves += a.goal_moves_in_window
    return closer / goal_moves if goal_moves else None


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise episode traces.")
    parser.add_argument("traces", type=Path, nargs="+", metavar="FILE")
    parser.add_argument("--window", type=int, default=None, help="final steps counted as recent")
    parser.add_argument(
        "--min-progress", type=float, default=0.5, help="share of moves closer to the goal"
    )
    parser.add_argument(
        "--dead-after",
        type=int,
        default=DEFAULT_DEAD_AFTER,
        help=f"steps standing still that count as dead (default {DEFAULT_DEAD_AFTER})",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if args.dead_after < 1:
        parser.error("--dead-after must be at least 1")
    summaries = []
    for path in args.traces:
        header, steps = load_trace(path)
        summary = summarise_trace(header, steps, args.window, args.min_progress, args.dead_after)
        summaries.append(summary)
        print(f"== {path}")
        print(format_summary(summary))
    if len(summaries) > 1:
        stuck = sum(1 for s in summaries if s.stuck_agents)
        wandering = sum(1 for s in summaries if s.wandering_agents)
        total = len(summaries)
        print(f"\n{stuck} of {total} episodes had a stuck agent ({stuck / total:.0%})")
        share = wandering / total
        longest = max(s.longest_fleet_still_run for s in summaries)
        print(f"longest whole-fleet still run over all episodes: {longest}")
        dead = sum(1 for s in summaries if s.fleet_dead_at is not None)
        print(f"{dead} of {total} episodes had a dead fleet (N = {args.dead_after})")
        print(f"{wandering} of {total} episodes had a wandering agent ({share:.0%})")
        progress = pooled_progress(summaries)
        print(
            "goal moves that bring an agent closer, over all episodes: "
            + (f"{progress:.0%}" if progress is not None else "no goal moves")
        )


if __name__ == "__main__":
    main()
