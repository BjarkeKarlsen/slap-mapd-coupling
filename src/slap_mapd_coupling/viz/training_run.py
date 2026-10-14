"""Plots and numbers for looking at a training run, from the files it leaves behind.

A run writes one progress log per arm (`<arm>_training_progress.jsonl`, one JSON object per
training iteration) and, optionally, episode traces (`<arm>_episode_<n>.jsonl`, see
`evaluation/trace.py`). This module turns both into figures and into a few plain numbers, so
a run can be looked at with one command:

    python -m slap_mapd_coupling.viz.training_run --run-dir DIR --out OUT

It looks for the logs in DIR and the traces in DIR or DIR/traces, writes the figures the data
allows to OUT and prints the numbers of every trace. A log from an older run misses some keys.
A panel whose keys are all absent (or all null) is left out, and the rest is drawn.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe, same as warehouse_plot

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from slap_mapd_coupling.evaluation.trace import load_trace as read_trace_file
from slap_mapd_coupling.viz import theme

Series = npt.NDArray[np.float64]
Progress = dict[str, Series]

PROGRESS_SUFFIX = "_training_progress.jsonl"
_TRACE_NAME = re.compile(r"^(?P<arm>.+)_episode_(?P<n>\d+)$")

# Ink for text and the arm-independent lines, a light grid. The muted text colour is the
# theme's, so the figures sit with the warehouse plots.
INK = "#0b0b0b"
MUTED = theme.PALETTE["text"]
GRID = "#e6e5e1"
_STYLE = {
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "axes.axisbelow": True,
    "legend.frameon": False,
}

# One fixed look per arm. Colour is never the only cue, each arm also has its own line style,
# marker and hatch. An arm not listed here gets the next free look from _SPARE_ARMS.
_ARM_ORDER = ("centralised", "section", "decentralised")
_ARM_LOOKS: dict[str, tuple[str, str, str, str]] = {  # colour, line style, marker, hatch
    "centralised": ("#2a78d6", "-", "o", ""),
    "section": ("#eb6834", "--", "s", "//"),
    "decentralised": ("#1baf7a", ":", "^", ".."),
}
_SPARE_ARMS = (
    ("#8a5cc7", "-", "D", "xx"),
    ("#c9a227", "-.", "v", "++"),
)
# One fixed colour per reward term, and the black line is the episode return.
TERM_COLOURS = {
    "shaping": "#e87ba4",
    "deliver": "#008300",
    "override": "#e34948",
    "wait": "#d9902b",
    "congestion": "#6250d6",
}
# What the raster of a trace shows, and the cell colour of each.
_STATES = ("moved", "chose to wait", "overridden")
_STATE_COLOURS = ("#9ec0ec", "#e6e5e1", "#d6403f")
_COMPLETED = "#2a78d6"
_BACKLOG = "#eb6834"

# A log with more valid points than this is drawn as a faint raw line under a rolling mean.
_MAX_RAW_POINTS = 60


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> Progress:
    """A progress log as one float array per key, with null or missing values as NaN."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys = sorted({key for row in rows for key in row})
    return {
        key: np.array(
            [np.nan if row.get(key) is None else float(row[key]) for row in rows], dtype=float
        )
        for key in keys
    }


@dataclass(frozen=True)
class Trace:
    """One episode trace, with the arm and episode number read from its file name."""

    arm: str
    episode: int | None
    header: dict[str, Any]
    steps: list[dict[str, Any]]

    @property
    def seed(self) -> int | None:
        seed = self.header.get("config", {}).get("seed")
        return None if seed is None else int(seed)

    @property
    def name(self) -> str:
        return self.arm if self.episode is None else f"{self.arm}_episode_{self.episode}"


def load_trace(path: Path) -> Trace:
    header, steps = read_trace_file(path)
    match = _TRACE_NAME.match(path.stem)
    if match:
        return Trace(match["arm"], int(match["n"]), header, steps)
    return Trace(str(header.get("config", {}).get("controller", path.stem)), None, header, steps)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentShares:
    """Shares of one agent's steps. An overridden step counts as overridden only."""

    agent_id: int
    moved: float
    waited: float
    overridden: float


@dataclass(frozen=True)
class TraceNumbers:
    steps: int
    completed: int
    released: int
    backlog: int
    last_completion_t: int | None  # None when nothing was completed
    longest_gap_start: int  # the step of the completion before it, 0 when none came before
    longest_gap_length: int  # steps from that completion to the next one, or to the last step
    agents: tuple[AgentShares, ...]
    overridden_share: float  # over all agent-steps
    steps_after_last_completion: int
    # Most frequent overridden (at, intent, count) over the steps after the last completion.
    top_overrides: tuple[tuple[int, int, int], ...]
    swap_steps: int  # steps where two agents wanted each other's vertex
    swap_steps_after_last_completion: int
    swap_pairs: tuple[tuple[int, int, int], ...]  # (agent, agent, steps) after the last completion


def _swapping_agents(agents: dict[str, dict[str, Any]]) -> list[tuple[int, int]]:
    """Agent pairs where i at a wants b while j at b wants a. An agent that stays does not want."""
    pairs = []
    for i, first in agents.items():
        for j, second in agents.items():
            if int(i) >= int(j) or first["intent"] in (None, first["at"]):
                continue
            if first["intent"] == second["at"] and second["intent"] == first["at"]:
                pairs.append((int(i), int(j)))
    return pairs


def trace_summary(trace: Trace, top: int = 5) -> TraceNumbers:
    """Plain numbers that show whether the fleet stalled, and on what."""
    steps = trace.steps
    if not steps:
        raise ValueError(f"the trace {trace.name} has no steps.")
    first_t, last_t = steps[0]["t"], steps[-1]["t"]
    marks = [first_t - 1]  # the steps at which a task was completed, after a start mark
    for before, step in zip([{"completed": 0}] + steps, steps):
        if step["completed"] > before["completed"]:
            marks.append(step["t"])
    gaps = [(start, end - start) for start, end in zip(marks, marks[1:] + [last_t])]
    longest = max(gaps, key=lambda gap: gap[1])
    last_completion = marks[-1] if len(marks) > 1 else None
    stalled = [step for step in steps if step["t"] > marks[-1]]

    ids = sorted(steps[0]["agents"], key=int)
    shares = []
    for agent in ids:
        records = [step["agents"][agent] for step in steps]
        over = sum(r["overridden"] for r in records) / len(records)
        moved = sum(r["moved"] and not r["overridden"] for r in records) / len(records)
        shares.append(AgentShares(int(agent), moved, 1.0 - moved - over, over))

    overrides = Counter(
        (r["at"], r["intent"])
        for step in stalled
        for r in step["agents"].values()
        if r["overridden"] and r["intent"] is not None
    )
    swaps = [_swapping_agents(step["agents"]) for step in steps]
    stalled_swaps = swaps[len(steps) - len(stalled) :]
    pair_counts = Counter(pair for found in stalled_swaps for pair in found)
    return TraceNumbers(
        steps=len(steps),
        completed=steps[-1]["completed"],
        released=steps[-1]["released"],
        backlog=steps[-1]["backlog"],
        last_completion_t=last_completion,
        longest_gap_start=longest[0],
        longest_gap_length=longest[1],
        agents=tuple(shares),
        overridden_share=sum(s.overridden for s in shares) / len(shares),
        steps_after_last_completion=len(stalled),
        top_overrides=tuple((at, intent, n) for (at, intent), n in overrides.most_common(top)),
        swap_steps=sum(1 for found in swaps if found),
        swap_steps_after_last_completion=sum(1 for found in stalled_swaps if found),
        swap_pairs=tuple((i, j, n) for (i, j), n in pair_counts.most_common(top)),
    )


def format_trace_numbers(trace: Trace, numbers: TraceNumbers) -> str:
    n = numbers
    last = "never" if n.last_completion_t is None else f"step {n.last_completion_t}"
    lines = [
        f"== {trace.name}"
        + ("" if trace.seed is None else f" (seed {trace.seed})")
        + f", {n.steps} steps",
        f"{n.completed} of {n.released} released tasks completed, backlog {n.backlog}",
        f"last completion at {last}, then {n.steps_after_last_completion} steps without one",
        f"longest stretch without a completion: {n.longest_gap_length} steps, "
        f"from step {n.longest_gap_start}",
        f"{n.overridden_share:.1%} of agent-steps overridden",
    ]
    for a in n.agents:
        lines.append(
            f"  agent {a.agent_id}: moved {a.moved:.1%}, chose to wait {a.waited:.1%}, "
            f"overridden {a.overridden:.1%}"
        )
    if n.top_overrides:
        pairs = ", ".join(f"{at}->{intent} x{count}" for at, intent, count in n.top_overrides)
        lines.append(f"most overridden moves after the last completion: {pairs}")
    lines.append(
        f"steps with two agents trying to swap vertices: {n.swap_steps} in all, "
        f"{n.swap_steps_after_last_completion} of the {n.steps_after_last_completion} "
        "after the last completion"
    )
    if n.swap_pairs:
        pairs = ", ".join(f"agents {i} and {j} x{count}" for i, j, count in n.swap_pairs)
        lines.append(f"  swapping pairs after the last completion: {pairs}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _arm_looks(arms: Iterable[str]) -> dict[str, tuple[str, str, str, str]]:
    ordered = [a for a in _ARM_ORDER if a in arms] + sorted(set(arms) - set(_ARM_ORDER))
    spare = iter(_SPARE_ARMS)
    return {
        arm: _ARM_LOOKS[arm] if arm in _ARM_LOOKS else next(spare, _SPARE_ARMS[-1])
        for arm in ordered
    }


def _has(series: Series | None) -> bool:
    return series is not None and bool(np.isfinite(series).any())


def _iterations(log: Progress) -> Series:
    if "iteration" in log:
        return log["iteration"]
    return np.arange(1, 1 + len(next(iter(log.values()), [])), dtype=float)


def _draw_series(ax: Axes, x: Series, y: Series, colour: str, style: Any, marker: str) -> None:
    """A line over the valid points.

    A long one is a rolling mean, with a faint raw line under it when the style is solid. The
    dashed and dotted lines are second measures, and a raw line under them only adds clutter.
    """
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) <= _MAX_RAW_POINTS:
        ax.plot(x, y, color=colour, ls=style, marker=marker, lw=1.8, ms=4)
        return
    window = max(2, len(y) // 40)
    mean = np.convolve(y, np.ones(window) / window, "valid")
    if style == "-":
        ax.plot(x, y, color=colour, lw=0.6, alpha=0.12)
    every = max(1, len(mean) // 10)  # a few markers, so the arms differ in more than colour
    ax.plot(
        x[window - 1 :], mean, color=colour, ls=style, lw=2, marker=marker, markevery=every, ms=5
    )


def _legend_below(ax: Axes, handles: Sequence[Any], columns: int) -> None:
    """A legend under the axis label, where it cannot cover a line."""
    ax.legend(
        handles=handles, ncol=columns, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22)
    )


def _grid(count: int, columns: int, height: float) -> tuple[Figure, list[Axes]]:
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, height * rows + 0.5), layout="constrained"
    )
    flat = list(np.atleast_1d(axes).ravel())
    for extra in flat[count:]:
        extra.set_visible(False)
    return fig, flat[:count]


def _arm_legend(fig: Figure, looks: dict[str, tuple[str, str, str, str]]) -> None:
    handles = [
        Line2D([], [], color=c, ls=s, marker=m, lw=1.8, ms=5, label=arm)
        for arm, (c, s, m, _) in looks.items()
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))


def plot_training_progress(logs: dict[str, Progress], out: Path) -> Path | None:
    """Training progress of one or several arms, written to `out`. None when nothing can be drawn.

    One panel per measure with a line for every arm, and one panel per arm with the episode
    return and its five terms. Lines of different measures in one panel differ in style.
    """
    looks = _arm_looks(logs)
    measures = [  # title, [(key, line style, label)]
        ("Episode return", [("episode_return_mean", "-", "episode return")]),
        (
            "Overrides and standing still",
            [
                ("override_rate", "-", "override rate"),
                ("standing_still_fraction", "--", "standing still"),
            ],
        ),
        (
            "Tasks per episode",
            [("tasks_completed", "-", "completed"), ("tasks_released", ":", "released")],
        ),
    ]
    term_keys = [f"return_{term}" for term in TERM_COLOURS]
    with_terms = [a for a in looks if any(_has(logs[a].get(k)) for k in term_keys)]
    if len(logs) == 1 and with_terms:
        measures = measures[1:]  # the return is in the panel of the arm
    measures = [m for m in measures if any(_has(logs[a].get(k)) for a in logs for k, _, _ in m[1])]
    count = len(measures) + len(with_terms)
    if count == 0:
        return None

    with plt.rc_context(_STYLE):
        fig, axes = _grid(count, min(3, count), 3.8)
        for ax, (title, parts) in zip(axes, measures):
            for arm, (colour, style, marker, _) in looks.items():
                x = _iterations(logs[arm])
                for key, part_style, _ in parts:
                    if _has(logs[arm].get(key)):
                        _draw_series(ax, x, logs[arm][key], colour, part_style, marker)
            ax.set_title(title, loc="left")
            ax.set_xlabel("training iteration")
            if len(parts) > 1:
                handles = [
                    Line2D([], [], color=MUTED, ls=s, lw=1.8, label=lab) for _, s, lab in parts
                ]
                _legend_below(ax, handles, 3)
            if title.startswith("Overrides"):
                ax.set_ylim(0, 1)
        for ax, arm in zip(axes[len(measures) :], with_terms):
            log = logs[arm]
            x = _iterations(log)
            for term, colour in TERM_COLOURS.items():
                if _has(log.get(f"return_{term}")):
                    _draw_series(ax, x, log[f"return_{term}"], colour, "-", "o")
                    ax.plot([], [], color=colour, marker="o", lw=1.8, ms=4, label=term)
            if _has(log.get("episode_return_mean")):
                _draw_series(ax, x, log["episode_return_mean"], INK, "-", "s")
                ax.plot([], [], color=INK, marker="s", lw=1.8, ms=4, label="episode return")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"Return and its terms, {arm}", loc="left")
            ax.set_xlabel("training iteration")
            _legend_below(ax, ax.get_legend_handles_labels()[0], 3)
        if len(logs) > 1:
            _arm_legend(fig, looks)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def _series(trace: Trace, key: str) -> npt.NDArray[np.float64]:
    return np.array([step[key] for step in trace.steps], dtype=float)


def plot_trace(trace: Trace, out: Path) -> Path:
    """One trace: tasks over time, and a per-agent raster of moved, chose to wait, overridden."""
    t = _series(trace, "t")
    ids = sorted(trace.steps[0]["agents"], key=int)

    def state(record: dict[str, Any]) -> int:
        return 2 if record["overridden"] else (0 if record["moved"] else 1)

    raster = np.array([[state(step["agents"][i]) for step in trace.steps] for i in ids])
    with plt.rc_context(_STYLE):
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(12, 3.2 + 0.45 * len(ids) + 1.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2, max(1.0, 0.35 * len(ids))]},
            layout="constrained",
        )
        top.plot(t, _series(trace, "released"), color=MUTED, ls="--", lw=1.6, label="released")
        top.plot(t, _series(trace, "completed"), color=_COMPLETED, lw=2, label="completed")
        top.plot(t, _series(trace, "backlog"), color=_BACKLOG, ls=":", lw=2, label="backlog")
        top.set_ylabel("tasks")
        top.legend(ncol=3, fontsize=9, loc="upper left")
        top.set_title(
            f"{trace.name}: tasks over time"
            + ("" if trace.seed is None else f" (seed {trace.seed})"),
            loc="left",
        )
        bottom.imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            extent=(t[0] - 0.5, t[-1] + 0.5, len(ids) - 0.5, -0.5),
            cmap=ListedColormap(_STATE_COLOURS),
            vmin=0,
            vmax=2,
        )
        bottom.set_yticks(range(len(ids)), [f"agent {i}" for i in ids])
        bottom.grid(False)
        bottom.set_xlabel("time step")
        handles = [
            Patch(facecolor=c, edgecolor=MUTED, label=label)
            for c, label in zip(_STATE_COLOURS, _STATES)
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def _group_label(trace: Trace) -> str:
    if trace.seed is not None:
        return f"seed {trace.seed}"
    return "episode ?" if trace.episode is None else f"episode {trace.episode}"


def plot_trace_comparison(traces: Sequence[Trace], out: Path) -> Path:
    """Several traces: completions and backlog over time per seed, and a bar panel.

    Small multiples, one column per seed (or episode when there is no seed) with a line per
    arm. On the right the completed tasks and the longest stretch without a completion of
    every trace.
    """
    looks = _arm_looks({trace.arm for trace in traces})
    groups = sorted({_group_label(trace) for trace in traces})
    by_key = {(_group_label(trace), trace.arm): trace for trace in traces}
    numbers = {key: trace_summary(trace) for key, trace in by_key.items()}
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(4 * len(groups) + 5, 7), layout="constrained")
        left, right = fig.subfigures(1, 2, width_ratios=[len(groups) * 1.0, 1.25])
        cells = left.subplots(2, len(groups), sharex=True, squeeze=False)
        for column, group in enumerate(groups):
            for arm, (colour, style, _, _) in looks.items():
                trace = by_key.get((group, arm))
                if trace is None:
                    continue
                t = _series(trace, "t")
                cells[0, column].plot(t, _series(trace, "completed"), color=colour, ls=style, lw=2)
                cells[1, column].plot(t, _series(trace, "backlog"), color=colour, ls=style, lw=2)
            cells[0, column].set_title(group, loc="left")
            cells[1, column].set_xlabel("time step")
        cells[0, 0].set_ylabel("cumulative tasks completed")
        cells[1, 0].set_ylabel("backlog")
        bars = right.subplots(2, 1)
        width = 0.8 / len(looks)
        for ax, title, pick in (
            (bars[0], "Tasks completed", lambda n: n.completed),
            (
                bars[1],
                "Longest stretch without a completion (steps)",
                lambda n: n.longest_gap_length,
            ),
        ):
            for k, (arm, (colour, _, _, hatch)) in enumerate(looks.items()):
                xs, values = [], []
                for column, group in enumerate(groups):
                    if (group, arm) in numbers:
                        xs.append(column + (k - (len(looks) - 1) / 2) * width)
                        values.append(pick(numbers[(group, arm)]))
                ax.bar(xs, values, width * 0.92, color=colour, hatch=hatch, edgecolor="white")
                for x, value in zip(xs, values):
                    ax.text(x, value, f"{value}", ha="center", va="bottom", fontsize=8, color=MUTED)
            ax.set_xticks(range(len(groups)), groups)
            ax.set_title(title, loc="left", fontsize=10)
            ax.margins(y=0.12)
        handles = [
            Line2D([], [], color=c, ls=s, lw=2, label=arm) for arm, (c, s, _, _) in looks.items()
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def find_run_files(run_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """The progress logs by arm, and the trace files, of a run directory."""
    logs = {p.name[: -len(PROGRESS_SUFFIX)]: p for p in sorted(run_dir.glob(f"*{PROGRESS_SUFFIX}"))}
    traces = sorted(p for d in (run_dir, run_dir / "traces") for p in d.glob("*_episode_*.jsonl"))
    return logs, traces


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot a training run and summarise its traces.")
    parser.add_argument("--run-dir", type=Path, required=True, help="folder with the run's files")
    parser.add_argument("--out", type=Path, required=True, help="folder for the figures")
    args = parser.parse_args(None if argv is None else list(argv))

    log_files, trace_files = find_run_files(args.run_dir)
    if not log_files and not trace_files:
        parser.error(f"no progress logs or traces in {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path | None] = []
    if log_files:
        logs = {arm: load_progress(path) for arm, path in log_files.items()}
        written.append(plot_training_progress(logs, args.out / "training_progress.png"))
    traces = [load_trace(path) for path in trace_files if path.stat().st_size > 0]
    for trace in traces:
        written.append(plot_trace(trace, args.out / f"trace_{trace.name}.png"))
        print(format_trace_numbers(trace, trace_summary(trace)))
    if len(traces) > 1:
        written.append(plot_trace_comparison(traces, args.out / "trace_comparison.png"))
    for path in written:
        if path is not None:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
