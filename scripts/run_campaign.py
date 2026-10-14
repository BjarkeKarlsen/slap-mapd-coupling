"""Run a whole training campaign unattended, then evaluate and report it.

One command trains every arm at every load level, evaluates the checkpoints and the reference
planner on the same seeds, and writes the side-by-side report. It is built to run while nobody
is watching, so it keeps going when a run fails, can be started again where it stopped, and
leaves a plain summary at the end.

    python scripts/run_campaign.py --out ~/thesis_runs/campaign --env-runners 7 --gpu

What it does, in order:

1. Checks the code is the one the thesis describes (the wait cost in the reward and the keep-up
   threshold), and that no earlier run is left in `examples/output`. A stale checkout stops the
   campaign before it spends hours on the wrong model. `--skip-preflight` turns the check off.
2. Trains, one run at a time. For each load level (`--rates`) the arms of `--arms`, and the
   decentralised arm with `--communication` at the levels of `--comm-rates`. Each run is the
   `examples/02`, `03` or `04` script with the same horizon, evaluation seeds and budget. Its
   output (`examples/output`) is moved into `OUT/runs/<condition>_rate<rate>/` the moment it
   ends, because the next run would overwrite it, and the log goes to `OUT/logs/`. A run that
   fails is tried again (`--retries`), and then left in a `_failed` folder while the campaign
   goes on.
3. Evaluates with `examples/05_full_comparison.py`. Each checkpoint is evaluated at the load it
   was trained at, and the reference planner at every load, on `--eval-seeds` seeds.
4. Writes `OUT/report/`: one overview per load level and one load sweep across levels, from
   `viz/study_report.py`, and `OUT/CAMPAIGN.md` with what ran, how long it took and what failed.

Started again with the same `--out`, a run that has a `DONE.json` is skipped, so a power cut
or a crash costs only the run that was in flight. Nothing here is imported from Ray, so the
script itself needs only the standard library. `--dry-run` prints the plan and runs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
OUTPUT = EXAMPLES / "output"
SCRIPTS = {
    "centralised": "02_centralised_fixed.py",
    "section": "03_section_based.py",
    "decentralised": "04_decentralised_training.py",
}
ARMS = tuple(SCRIPTS)
# Illustrative. The first level is the main one, so the key results exist early.
DEFAULT_RATES = (0.25, 0.15, 0.3)
DEFAULT_COMM_RATES = (0.25,)
CHECKPOINT_LINE = re.compile(r"^Checkpoint saved to (?P<path>.+)$", re.MULTILINE)
# The wait cost in the reward and the keep-up threshold, so a checkout without them is refused.
PREFLIGHT = (
    (
        "src/slap_mapd_coupling/training/callbacks.py",
        "return_wait",
        "the wait cost is not in the reward (c_wait, [A30])",
    ),
    ("examples/04_decentralised_training.py", "WAIT_COST = 0.1", "WAIT_COST is not 0.1"),
    (
        "examples/04_decentralised_training.py",
        "KEEP_UP_THRESHOLD = 0.8",
        "the keep-up threshold is not set (f_up, [A31])",
    ),
)


@dataclass(frozen=True)
class Run:
    arm: str
    rate: float
    communication: bool = False

    @property
    def condition(self) -> str:
        return f"{self.arm}+comm" if self.communication else self.arm

    @property
    def name(self) -> str:
        return f"{self.condition}_rate{self.rate:g}"


def plan(arms: Sequence[str], rates: Sequence[float], comm_rates: Sequence[float]) -> list[Run]:
    """The runs in the order they are made: level by level, the decentralised arm first."""
    runs: list[Run] = []
    ordered = sorted(arms, key=lambda a: (a != "decentralised", ARMS.index(a)))
    for rate in rates:
        runs += [Run(arm, rate) for arm in ordered]
        if rate in comm_rates:
            runs.append(Run("decentralised", rate, communication=True))
    return runs


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = args.out.expanduser().resolve()
        self.runs_dir = self.out / "runs"
        self.log_dir = self.out / "logs"
        self.python = args.python
        self.results: list[dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------------------------

    def say(self, text: str) -> None:
        line = f"[{stamp()}] {text}"
        print(line, flush=True)
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "campaign.log").open("a", encoding="utf-8") as sink:
            sink.write(line + "\n")

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        # The shared virtualenv imports the package from one checkout. Pointing it at this one
        # keeps the campaign on the code that is checked out here.
        env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def run_command(self, command: list[str], log: Path) -> int:
        with log.open("a", encoding="utf-8") as sink:
            sink.write(f"$ {' '.join(command)}\n")
            sink.flush()
            return subprocess.call(
                command, cwd=REPO, env=self.env(), stdout=sink, stderr=subprocess.STDOUT
            )

    # -- checks -------------------------------------------------------------------------------

    def preflight(self) -> None:
        problems = []
        for relative, needle, message in PREFLIGHT:
            path = REPO / relative
            if not path.exists() or needle not in path.read_text(encoding="utf-8"):
                problems.append(message)
        if problems and not self.args.skip_preflight:
            raise SystemExit(
                "Preflight failed, the checkout is not the model of the thesis:\n  - "
                + "\n  - ".join(problems)
                + "\nMerge the branch that adds it, or pass --skip-preflight."
            )
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            aside = self.out / "_stale_output" / datetime.now().strftime("%Y%m%d_%H%M%S")
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(OUTPUT), str(aside))
            self.say(f"examples/output was not empty, moved to {aside}")
        if not Path(self.python).exists() and shutil.which(self.python) is None:
            raise SystemExit(f"No Python at {self.python}, pass --python.")

    # -- training -----------------------------------------------------------------------------

    def train_command(self, run: Run) -> list[str]:
        a = self.args
        command = [
            self.python,
            str(EXAMPLES / SCRIPTS[run.arm]),
            f"--horizon={a.horizon}",
            f"--arrival-rate={run.rate:g}",
            f"--env-runners={a.env_runners}",
            f"--budget-env-steps={a.budget_env_steps}",
            f"--eval-seeds={a.eval_seeds}",
        ]
        if a.gpu:
            command.append("--gpu")
        if run.communication:
            command.append("--communication")
        return command

    def train(self, run: Run) -> dict[str, Any]:
        dest = self.runs_dir / run.name
        done = dest / "DONE.json"
        if done.exists():
            self.say(f"skip {run.name}, already done")
            return json.loads(done.read_text(encoding="utf-8"))
        if dest.exists():
            shutil.rmtree(dest)  # a run that was cut off, started again from scratch
        log = self.log_dir / f"{run.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("", encoding="utf-8")
        record: dict[str, Any] = {
            "run": run.name,
            "arm": run.arm,
            "communication": run.communication,
            "rate": run.rate,
        }
        for attempt in range(1, self.args.retries + 2):
            started = time.time()
            self.say(f"train {run.name}, attempt {attempt}")
            code = self.run_command(self.train_command(run), log)
            seconds = round(time.time() - started)
            text = log.read_text(encoding="utf-8", errors="replace")
            found = CHECKPOINT_LINE.findall(text)
            record.update(returncode=code, seconds=seconds, attempts=attempt)
            if code == 0 and found:
                record["checkpoint"] = found[-1].strip()
                dest.parent.mkdir(parents=True, exist_ok=True)
                if OUTPUT.exists():
                    shutil.move(str(OUTPUT), str(dest))
                else:
                    dest.mkdir(parents=True)
                record["log"] = str(log)
                done.write_text(json.dumps(record, indent=2), encoding="utf-8")
                self.say(f"done {run.name} in {seconds // 60} min, checkpoint {found[-1].strip()}")
                return record
            self.say(f"FAILED {run.name} (exit {code}), see {log}")
            failed = self.runs_dir / f"{run.name}_failed"
            if OUTPUT.exists():
                failed.parent.mkdir(parents=True, exist_ok=True)
                if failed.exists():
                    shutil.rmtree(failed)
                shutil.move(str(OUTPUT), str(failed))
            self.cool_down()
        record["failed"] = True
        return record

    def cool_down(self) -> None:
        """Stops stray Ray processes between runs, so the next run starts from a clean GPU."""
        ray = Path(self.python).with_name("ray")
        if ray.exists():
            subprocess.call(
                [str(ray), "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
        time.sleep(5)

    # -- evaluation and report ----------------------------------------------------------------

    def sweep(self, finished: list[dict[str, Any]]) -> list[Path]:
        """The planner at every load, then each checkpoint at the load it was trained at."""
        a = self.args
        sweep_dir = self.out / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / "sweep.log"
        files: list[Path] = []
        base = [
            self.python,
            str(EXAMPLES / "05_full_comparison.py"),
            f"--horizon={a.horizon}",
            f"--eval-seeds={a.eval_seeds}",
        ]
        rates = sorted({r["rate"] for r in finished} | set(a.rates))
        planner_dir = sweep_dir / "planners"
        planners = [part for name in a.planners for part in ("--planner", name)]
        self.say("evaluate the reference planner at every load")
        command = base + ["--out", str(planner_dir), *planners, "--arrival-rates"]
        command += [f"{rate:g}" for rate in rates]
        if self.run_command(command, log) == 0:
            files.append(planner_dir / "load_sweep.jsonl")
        for rate in sorted({r["rate"] for r in finished}):
            at_rate = [r for r in finished if r["rate"] == rate]
            checkpoints = []
            for r in at_rate:
                label = f"{r['arm']}+comm" if r["communication"] else r["arm"]
                checkpoints += ["--checkpoint", f"{label}={r['checkpoint']}"]
            rate_dir = sweep_dir / f"rate_{rate:g}"
            self.say(f"evaluate the checkpoints trained at {rate:g}")
            # The planner sits in its own sweep, so this one has a checkpoint and no planner.
            command = base + ["--out", str(rate_dir), "--arrival-rates", f"{rate:g}"]
            command += checkpoints + ["--planner", a.planners[0]]
            if self.run_command(command, log) == 0:
                files.append(rate_dir / "load_sweep.jsonl")
        return files

    def merge_sweeps(self, files: list[Path]) -> Path | None:
        """One sweep file for the load figure. The planner rows come from the planner sweep, so
        the per-level sweeps' own planner rows are left out."""
        rows: list[str] = []
        for file in files:
            lines = [ln for ln in file.read_text(encoding="utf-8").splitlines() if ln]
            if file.parent.name != "planners":
                lines = [ln for ln in lines if json.loads(ln)["kind"] != "planner"]
            rows += lines
        if not rows:
            return None
        merged = self.out / "sweep" / "load_sweep.jsonl"
        merged.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return merged

    def report(self, finished: list[dict[str, Any]], sweep_files: list[Path]) -> None:
        log = self.log_dir / "report.log"
        module = [self.python, "-m", "slap_mapd_coupling.viz.study_report"]
        merged = self.merge_sweeps(sweep_files)
        if merged is not None:
            self.say("write the load sweep report")
            self.run_command(
                module + ["--sweep", str(merged), "--out", str(self.out / "report" / "load_sweep")],
                log,
            )
        for rate in sorted({r["rate"] for r in finished}):
            runs: list[str] = []
            for r in finished:
                if r["rate"] != rate:
                    continue
                folder = self.runs_dir / r["run"]
                runs += ["--run", f"comm={folder}" if r["communication"] else str(folder)]
            command = module + runs + ["--out", str(self.out / "report" / f"rate_{rate:g}")]
            rate_sweep = self.out / "sweep" / f"rate_{rate:g}" / "load_sweep.jsonl"
            if rate_sweep.exists():
                command += ["--sweep", str(rate_sweep)]
            self.say(f"write the report of the runs trained at {rate:g}")
            self.run_command(command, log)

    def summary(self, runs: list[Run], records: list[dict[str, Any]], started: float) -> None:
        by_name = {r["run"]: r for r in records}
        lines = [
            "# Campaign summary",
            "",
            f"Written {stamp()}, {round((time.time() - started) / 60)} minutes in all.",
            "",
            "| run | result | minutes | attempts | checkpoint |",
            "|---|---|---|---|---|",
        ]
        for run in runs:
            r = by_name.get(run.name)
            if r is None:
                lines.append(f"| {run.name} | not run | | | |")
                continue
            result = "FAILED" if r.get("failed") else "done"
            minutes = round(r.get("seconds", 0) / 60)
            lines.append(
                f"| {run.name} | {result} | {minutes} | {r.get('attempts', '')} "
                f"| {r.get('checkpoint', '')} |"
            )
        lines += [
            "",
            "Reports are in `report/`, logs in `logs/`, runs in `runs/`, the sweep in `sweep/`.",
            "A failed run's output is in `runs/<name>_failed/`. Start the same command again to",
            "retry only what is missing.",
            "",
        ]
        (self.out / "CAMPAIGN.md").write_text("\n".join(lines), encoding="utf-8")
        self.say("summary written to CAMPAIGN.md")

    # -- the campaign -------------------------------------------------------------------------

    def run(self) -> None:
        started = time.time()
        runs = plan(self.args.arms, self.args.rates, self.args.comm_rates)
        self.say(f"{len(runs)} runs planned: " + ", ".join(r.name for r in runs))
        if self.args.dry_run:
            for run in runs:
                self.say("would run: " + " ".join(self.train_command(run)))
            return
        self.preflight()
        records: list[dict[str, Any]] = []
        for run in runs:
            records.append(self.train(run))
        finished = [r for r in records if not r.get("failed")]
        if not finished:
            self.say("no run finished, nothing to evaluate")
        else:
            files = self.sweep(finished)
            self.report(finished, files)
        self.summary(runs, records, started)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate and report a whole campaign unattended."
    )
    parser.add_argument("--out", type=Path, required=True, help="folder for everything")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_RATES),
        help="arrival rates to train at, one policy per arm and rate (illustrative defaults)",
    )
    parser.add_argument(
        "--comm-rates",
        type=float,
        nargs="*",
        default=list(DEFAULT_COMM_RATES),
        help="rates at which the decentralised arm is also trained with --communication",
    )
    parser.add_argument("--horizon", type=int, default=630)
    parser.add_argument("--budget-env-steps", type=int, default=500_000)
    parser.add_argument("--eval-seeds", type=int, default=6)
    parser.add_argument("--env-runners", type=int, default=7)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--planners",
        nargs="+",
        choices=["token_passing", "prioritised_astar"],
        default=["token_passing"],
        help="reference planners evaluated at every load",
    )
    parser.add_argument("--retries", type=int, default=1, help="extra attempts for a failed run")
    parser.add_argument("--python", default=str(Path.home() / ".venvs/rl/bin/python"))
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)
    unknown = [r for r in args.comm_rates if r not in args.rates]
    if unknown:
        parser.error(f"--comm-rates {unknown} are not in --rates")
    if min(args.rates, default=1) <= 0 or args.horizon < 1 or args.eval_seeds < 1:
        parser.error("rates, horizon and eval seeds must be positive")
    if args.budget_env_steps < 1 or args.retries < 0 or args.env_runners < 0:
        parser.error("budget must be positive and retries and runners not negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    Campaign(parse_args(argv)).run()


if __name__ == "__main__":
    main(sys.argv[1:])
