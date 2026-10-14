"""One report that puts the conditions of a study side by side, from the files runs leave behind.

`viz/training_run.py` looks at one run folder. This module takes several, so the arms (and a
variant such as the decentralised arm with communication, run into a folder of its own) sit in
the same panels, and it adds a load sweep when one exists (`examples/05_full_comparison.py`).

    python -m slap_mapd_coupling.viz.study_report \\
        --run DIR --run comm=DIR_WITH_COMMUNICATION --sweep SWEEP --out OUT

A `--run` is `[NAME=]DIR`. A folder holds `<arm>_training_progress.jsonl` logs and the traces
`<arm>_episode_<n>.jsonl` (in the folder or its `traces/`). With a NAME the condition is called
`<arm>+NAME`, so `comm=...` gives `decentralised+comm`, the label the sweep uses. `--sweep` is a
`load_sweep.jsonl` or the folder that holds it.

It writes, for what the files allow:

- `study_overview.png`, the training curves, the reward terms and the evaluation episodes side by
  side, one line, bar or dot per condition.
- `load_sweep.png`, completion share, service time and backlog against the arrival rate.
- `study_summary.md`, the numbers behind the figures with the sample sizes.

Everything in the report is read off the files. It says nothing about whether two conditions
differ, because a single checkpoint per arm gives no spread over training runs. A log from an
older run misses some keys, and a panel without data is left out.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe, same as warehouse_plot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from slap_mapd_coupling.viz import training_run as tr

SWEEP_FILE = "load_sweep.jsonl"
# A drop of the training return by more than this share of the step before counts as a collapse,
# once the first WARMUP iterations have passed (the early climb has large swings of its own).
COLLAPSE_DROP = 0.25
WARMUP = 20
# An episode that goes this many steps after its last completion without one counts as stalled.
STALL_STEPS = 50

# The arms keep the looks of training_run. Other conditions take these in order of appearance,
# and the reference planners have fixed ones.
_SPARE = (
    ("#8a5cc7", "-", "D"),
    ("#c9a227", "-.", "v"),
    ("#d6403f", "--", "P"),
    ("#20a4b5", ":", "X"),
)
_PLANNER_LOOKS = {
    "token_passing": ("#5c5c5c", "-", "x"),
    "prioritised_astar": ("#8b5a2b", "--", "+"),
}
_TERMS = ("shaping", "deliver", "override", "congestion")
_STYLE = tr._STYLE  # the look of the other training figures


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass
class Condition:
    """One arm of one run folder: its training log and its evaluation traces."""

    label: str
    log: tr.Progress | None = None
    traces: list[tr.Trace] = field(default_factory=list)


def parse_run(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    return (name, Path(path)) if sep else ("", Path(text))


def collect(runs: Iterable[tuple[str, Path]]) -> list[Condition]:
    """The conditions of the run folders, ordered as the arms are, then by label."""
    found: dict[str, Condition] = {}
    for name, folder in runs:
        logs, trace_files = tr.find_run_files(folder)

        def label_of(arm: str) -> str:
            return f"{arm}+{name}" if name else arm

        for arm, path in logs.items():
            found.setdefault(label_of(arm), Condition(label_of(arm))).log = tr.load_progress(path)
        for path in trace_files:
            if path.stat().st_size == 0:
                continue
            trace = tr.load_trace(path)
            found.setdefault(label_of(trace.arm), Condition(label_of(trace.arm))).traces.append(
                trace
            )
    order = {arm: i for i, arm in enumerate(tr._ARM_ORDER)}
    return sorted(found.values(), key=lambda c: (order.get(c.label.split("+")[0], 99), c.label))


def load_sweep(path: Path) -> list[dict[str, Any]]:
    file = path / SWEEP_FILE if path.is_dir() else path
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line]


def assign_looks(labels: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    """A colour, a line style and a marker per label, so no figure relies on colour alone."""
    spare = iter(_SPARE)
    looks: dict[str, tuple[str, str, str]] = {}
    for label in labels:
        if label in _PLANNER_LOOKS:
            looks[label] = _PLANNER_LOOKS[label]
        elif label in tr._ARM_LOOKS:
            colour, style, marker, _ = tr._ARM_LOOKS[label]
            looks[label] = (colour, style, marker)
        else:
            looks[label] = next(spare, _SPARE[-1])
    return looks


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _clean(values: Iterable[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _mean(values: Iterable[float]) -> float:
    kept = _clean(values)
    return float(np.mean(kept)) if kept else math.nan


def _late(log: tr.Progress, key: str, window: int) -> float:
    series = log.get(key)
    return math.nan if series is None else _mean(series[-window:])


def training_stats(log: tr.Progress, window: int) -> dict[str, float]:
    """What the training log says, the last `window` iterations against the whole run."""
    returns = log.get("episode_return_mean")
    stats: dict[str, float] = {"iterations": float(len(next(iter(log.values()), [])))}
    if returns is not None and np.isfinite(returns).any():
        ok = np.nan_to_num(returns, nan=0.0)
        late = _clean(returns[-window:])
        drops = [
            i
            for i in range(max(1, WARMUP), len(ok))
            if np.isfinite(returns[i - 1])
            and np.isfinite(returns[i])
            and returns[i] < returns[i - 1] - COLLAPSE_DROP * abs(returns[i - 1])
        ]
        stats.update(
            late_return=_mean(late),
            late_return_sd=float(np.std(late)) if late else math.nan,
            best_return=float(np.nanmax(returns)),
            best_iteration=float(int(np.nanargmax(returns)) + 1),
            negative_iterations=float(np.sum(returns < 0)),
            collapses=float(len(drops)),
        )
        del ok
    for term in _TERMS:
        stats[f"late_{term}"] = _late(log, f"return_{term}", window)
    done, released = log.get("episode_tasks_completed"), log.get("episode_tasks_released")
    if done is not None and released is not None:
        share = done[-window:] / np.where(released[-window:] > 0, released[-window:], np.nan)
        stats["late_completion_share"] = _mean(share)
    stats["late_override_rate"] = _late(log, "override_rate", window)
    return stats


def trace_stats(traces: Sequence[tr.Trace]) -> dict[str, float]:
    """What the evaluation episodes say, pooled over episodes (sums for tasks, means otherwise)."""
    numbers = [tr.trace_summary(trace) for trace in traces]
    completed = sum(n.completed for n in numbers)
    released = sum(n.released for n in numbers)
    return {
        "episodes": float(len(numbers)),
        "completed": float(completed),
        "released": float(released),
        "completion_share": completed / released if released else math.nan,
        "mean_backlog": _mean(n.backlog for n in numbers),
        "overridden_share": _mean(n.overridden_share for n in numbers),
        "longest_gap": float(max((n.longest_gap_length for n in numbers), default=0)),
        "swap_steps": _mean(n.swap_steps for n in numbers),
        "stalled_episodes": float(
            sum(n.steps_after_last_completion >= STALL_STEPS for n in numbers)
        ),
    }


def sweep_stats(rows: Sequence[dict[str, Any]]) -> dict[str, dict[float, dict[str, Any]]]:
    """Per condition and arrival rate: seeds, completion share (mean, min, max), service time."""
    out: dict[str, dict[float, dict[str, Any]]] = {}
    for condition in dict.fromkeys(r["condition"] for r in rows):
        by_rate: dict[float, dict[str, Any]] = {}
        for rate in sorted({r["arrival_rate"] for r in rows if r["condition"] == condition}):
            group = [r for r in rows if r["condition"] == condition and r["arrival_rate"] == rate]
            shares = [r["num_completed_tasks"] / r["released"] for r in group if r["released"]]
            service = [r["mean_service_time"] for r in group if r["mean_service_time"] is not None]
            by_rate[rate] = {
                "seeds": len(group),
                "completed": sum(r["num_completed_tasks"] for r in group),
                "released": sum(r["released"] for r in group),
                "share": _mean(shares),
                "share_min": min(shares, default=math.nan),
                "share_max": max(shares, default=math.nan),
                "service": _mean(service),
                "service_min": min(service, default=math.nan),
                "service_max": max(service, default=math.nan),
                "backlog": _mean(r["backlog"] for r in group),
                "backlog_min": min((r["backlog"] for r in group), default=math.nan),
                "backlog_max": max((r["backlog"] for r in group), default=math.nan),
            }
        out[condition] = by_rate
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _smooth(ax: Axes, x: np.ndarray, y: np.ndarray, look: tuple[str, str, str]) -> None:
    colour, style, marker = look
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) == 0:
        return
    window = max(1, min(len(y) // 20, 10))
    if window > 1:
        ax.plot(x, y, color=colour, lw=0.6, alpha=0.15)
        mean = np.convolve(y, np.ones(window) / window, "valid")
        x, y = x[window - 1 :], mean
    ax.plot(
        x, y, color=colour, ls=style, lw=1.8, marker=marker, markevery=max(1, len(y) // 8), ms=5
    )


def _curve_panel(
    ax: Axes,
    conditions: Sequence[Condition],
    looks: dict[str, tuple[str, str, str]],
    values: Any,
    title: str,
    ylabel: str,
) -> bool:
    drawn = False
    for c in conditions:
        if c.log is None:
            continue
        y = values(c.log)
        if y is None or not np.isfinite(y).any():
            continue
        _smooth(ax, tr._iterations(c.log)[: len(y)], y, looks[c.label])
        drawn = True
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    return drawn


def _share(log: tr.Progress) -> tr.Series | None:
    done, released = log.get("episode_tasks_completed"), log.get("episode_tasks_released")
    if done is None or released is None:
        return None
    return np.asarray(done / np.where(released > 0, released, np.nan), dtype=float)


def _terms_panel(
    ax: Axes, conditions: Sequence[Condition], stats: dict[str, dict[str, float]]
) -> bool:
    """Late reward terms per condition, gains stacked up, losses down, the return as a diamond."""
    labels = [
        c.label for c in conditions if c.label in stats and f"late_{_TERMS[0]}" in stats[c.label]
    ]
    if not labels:
        return False
    for i, label in enumerate(labels):
        up = down = 0.0
        for term in _TERMS:
            value = stats[label].get(f"late_{term}", math.nan)
            if not math.isfinite(value):
                continue
            bottom = up if value >= 0 else down
            ax.bar(i, value, bottom=bottom, color=tr.TERM_COLOURS[term], width=0.6)
            if value >= 0:
                up += value
            else:
                down += value
        ret = stats[label].get("late_return", math.nan)
        if math.isfinite(ret):
            ax.plot(i, ret, "D", color=tr.INK, ms=6)
    ax.axhline(0, color=tr.MUTED, lw=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Reward terms, late iterations", fontsize=10, loc="left")
    ax.set_ylabel("per episode")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", ls="", color=tr.TERM_COLOURS[t], label=t) for t in _TERMS
        ]
        + [Line2D([], [], marker="D", ls="", color=tr.INK, label="return")],
        fontsize=7,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
    )
    return True


def _dots_panel(
    ax: Axes,
    conditions: Sequence[Condition],
    looks: dict[str, tuple[str, str, str]],
    value: Any,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> bool:
    """One dot per evaluation episode of each condition, and its mean as a bar. A share is drawn
    on its full range, so a small spread between episodes does not look like a large one."""
    shown = [c for c in conditions if c.traces]
    if not shown:
        return False
    for i, c in enumerate(shown):
        ys = [value(tr.trace_summary(t)) for t in c.traces]
        colour, _, marker = looks[c.label]
        offsets = np.linspace(-0.15, 0.15, len(ys)) if len(ys) > 1 else [0.0]
        ax.plot(i + np.asarray(offsets), ys, marker, color=colour, ms=6, ls="", alpha=0.85)
        ax.hlines(np.mean(ys), i - 0.28, i + 0.28, color=tr.INK, lw=1.4)
    ax.set_xticks(range(len(shown)), [c.label for c in shown], rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return True


def plot_overview(
    conditions: Sequence[Condition], stats: dict[str, dict[str, float]], out: Path
) -> Path | None:
    """The six panels side by side, or None when the files have nothing to draw."""
    looks = assign_looks([c.label for c in conditions])
    with plt.rc_context(tr._STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), layout="constrained")
        flat = list(axes.ravel())
        drawn = [
            _curve_panel(
                flat[0],
                conditions,
                looks,
                lambda g: g.get("episode_return_mean"),
                "Training return",
                "episode return",
            ),
            _curve_panel(
                flat[1],
                conditions,
                looks,
                _share,
                "Tasks completed in training",
                "share of released",
            ),
            _curve_panel(
                flat[2],
                conditions,
                looks,
                lambda g: g.get("override_rate"),
                "Moves overridden in training",
                "share of agent-steps",
            ),
            _terms_panel(flat[3], conditions, stats),
            _dots_panel(
                flat[4],
                conditions,
                looks,
                lambda n: n.completed / n.released if n.released else 0.0,
                "Tasks completed, evaluation episodes",
                "share of released",
                ylim=(0.0, 1.02),
            ),
            _dots_panel(
                flat[5],
                conditions,
                looks,
                lambda n: float(n.longest_gap_length),
                "Longest stretch without a completion",
                "steps",
            ),
        ]
        for ax, ok in zip(flat, drawn):
            if not ok:
                ax.set_visible(False)
        if not any(drawn):
            plt.close(fig)
            return None
        handles = [
            Line2D([], [], color=c, ls=s, marker=m, lw=1.8, ms=5, label=label)
            for label, (c, s, m) in looks.items()
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


def plot_load_sweep(stats: dict[str, dict[float, dict[str, Any]]], out: Path) -> Path | None:
    """Completion share, service time and backlog against the arrival rate, a band over seeds."""
    if not stats:
        return None
    looks = assign_looks(list(stats))
    panels = (
        ("share", "Tasks completed", "share of released"),
        ("service", "Mean service time", "steps"),
        ("backlog", "Backlog at the end", "tasks"),
    )
    with plt.rc_context(tr._STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
        for ax, (key, title, ylabel) in zip(axes, panels):
            for label, by_rate in stats.items():
                colour, style, marker = looks[label]
                rates = np.array(list(by_rate))
                mean = np.array([by_rate[r][key] for r in by_rate], dtype=float)
                low = np.array([by_rate[r][f"{key}_min"] for r in by_rate], dtype=float)
                high = np.array([by_rate[r][f"{key}_max"] for r in by_rate], dtype=float)
                ax.fill_between(rates, low, high, color=colour, alpha=0.12, lw=0)
                ax.plot(rates, mean, color=colour, ls=style, marker=marker, lw=1.8, ms=6)
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_xlabel("arrival rate (tasks per step)")
            ax.set_ylabel(ylabel)
        handles = [
            Line2D([], [], color=c, ls=s, marker=m, lw=1.8, ms=5, label=label)
            for label, (c, s, m) in looks.items()
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# The written summary
# ---------------------------------------------------------------------------


def _f(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def summary_text(
    conditions: Sequence[Condition],
    training: dict[str, dict[str, float]],
    evaluation: dict[str, dict[str, float]],
    sweep: dict[str, dict[float, dict[str, Any]]] | None,
    window: int,
) -> str:
    """The numbers as markdown, with the sample sizes. Measured values only."""
    parts = [
        "# Study summary",
        "",
        "Every number is read off the run files. One checkpoint per arm gives no spread over "
        "training runs, so nothing here says whether two conditions differ.",
    ]
    if training:
        rows = [
            [
                label,
                _f(s["iterations"], 0),
                _f(s.get("late_return", math.nan), 0),
                _f(s.get("late_return_sd", math.nan), 0),
                _f(s.get("best_return", math.nan), 0),
                _f(s.get("collapses", math.nan), 0),
                _f(s.get("negative_iterations", math.nan), 0),
                _f(s.get("late_completion_share", math.nan), 2),
                _f(s.get("late_override_rate", math.nan), 3),
                " / ".join(_f(s.get(f"late_{t}", math.nan), 0) for t in _TERMS),
            ]
            for label, s in training.items()
        ]
        parts += [
            "",
            f"## Training (late = last {window} iterations)",
            "",
            _table(
                [
                    "condition",
                    "iterations",
                    "late return",
                    "sd",
                    "best",
                    "collapses",
                    "negative",
                    "completion share",
                    "override rate",
                    "shaping / deliver / override / congestion",
                ],
                rows,
            ),
            "",
            f"A collapse is a drop of more than {COLLAPSE_DROP:.0%} between two iterations after "
            f"the first {WARMUP}.",
        ]
    if evaluation:
        rows = [
            [
                label,
                _f(s["episodes"], 0),
                f"{_f(s['completed'], 0)} / {_f(s['released'], 0)}",
                _f(s["completion_share"], 2),
                _f(s["mean_backlog"], 1),
                _f(s["overridden_share"], 3),
                _f(s["longest_gap"], 0),
                _f(s["swap_steps"], 1),
                _f(s["stalled_episodes"], 0),
            ]
            for label, s in evaluation.items()
        ]
        parts += [
            "",
            "## Evaluation episodes",
            "",
            _table(
                [
                    "condition",
                    "episodes",
                    "completed / released",
                    "share",
                    "mean backlog",
                    "overridden share",
                    "longest gap",
                    "swap steps per episode",
                    "stalled episodes",
                ],
                rows,
            ),
            "",
            f"An episode is stalled if it goes {STALL_STEPS} steps or more after its last "
            "completion without one. A swap step is a step where two agents wanted each other's "
            "vertex.",
        ]
    if sweep:
        rows = [
            [
                label,
                f"{rate:g}",
                str(s["seeds"]),
                f"{s['completed']} / {s['released']}",
                f"{_f(s['share'], 2)} ({_f(s['share_min'], 2)} to {_f(s['share_max'], 2)})",
                f"{_f(s['service'], 1)} ({_f(s['service_min'], 1)} to {_f(s['service_max'], 1)})",
                _f(s["backlog"], 1),
            ]
            for label, by_rate in sweep.items()
            for rate, s in by_rate.items()
        ]
        parts += [
            "",
            "## Load sweep",
            "",
            _table(
                [
                    "condition",
                    "arrival rate",
                    "seeds",
                    "completed / released",
                    "share (min to max over seeds)",
                    "mean service time (min to max)",
                    "mean backlog",
                ],
                rows,
            ),
            "",
            "Service time averages the tasks that were completed, so it is meaningful only where "
            "the condition keeps up. A condition that completes few tasks can show a short one.",
        ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Put the conditions of a study side by side.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="[NAME=]DIR",
        help="a run folder, repeatable. With NAME its conditions are called <arm>+NAME",
    )
    parser.add_argument("--sweep", type=Path, help="a load_sweep.jsonl, or the folder holding it")
    parser.add_argument("--out", type=Path, required=True, help="folder for the report")
    parser.add_argument(
        "--window", type=int, default=20, help="last iterations that count as late (default 20)"
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if not args.run and args.sweep is None:
        parser.error("give at least one --run or a --sweep")
    if args.window < 1:
        parser.error("--window must be at least 1")

    conditions = collect(parse_run(text) for text in args.run)
    training = {
        c.label: training_stats(c.log, args.window) for c in conditions if c.log is not None
    }
    evaluation = {c.label: trace_stats(c.traces) for c in conditions if c.traces}
    sweep = sweep_stats(load_sweep(args.sweep)) if args.sweep else None

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    if conditions:
        written.append(plot_overview(conditions, training, args.out / "study_overview.png"))
    if sweep:
        written.append(plot_load_sweep(sweep, args.out / "load_sweep.png"))
    text = summary_text(conditions, training, evaluation, sweep, args.window)
    (args.out / "study_summary.md").write_text(text, encoding="utf-8")
    print(text)
    for path in [*written, args.out / "study_summary.md"]:
        if path is not None:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
