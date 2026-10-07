"""The paired comparison of eq:rqformal: does adaptive storage shorten
mean service time under a controller, compared with fixed storage?

Under one seed, fixed and adaptive storage face exactly the same order
stream (sec:impl:instances), so the difference in their mean service times
comes from the storage rule alone. eq:rqformal therefore pairs runs seed by
seed: gain_n = zeta_bar_T(F_fix) - zeta_bar_T(F) under seed n, and adaptive
storage wins if every run keeps up and the confidence interval of the
mean gain lies above zero -- a paired t-test over seeds.

A run that does not keep up is not compared on service time
(eq:throughput), so any such run, or any run whose keep-up check is still
unset because f_up is not configured, makes the comparison undecided
rather than counting as a loss or a win.
"""

from __future__ import annotations

import math
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from scipy import stats

from slap_mapd_coupling.evaluation.metrics import RunMetrics

# Everything that defines a matched condition besides the storage rule and
# the seed. Two runs are only a valid pair if all of these agree.
_MATCHED_FIELDS = (
    "controller",
    "congestion_sensitive",
    "communication",
    "num_agents",
    "arrival_rate",
    "horizon",
)


class PairedComparison(BaseModel):
    """Result of eq:rqformal for one controller and one matched condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adaptive_storage_mode: str
    num_seeds: PositiveInt
    confidence: float = Field(gt=0.0, lt=1.0)
    # True only if every run in both arms has keeps_up == True.
    all_keep_up: bool
    # Per-seed gains and their summary; None when all_keep_up is False,
    # since service time is then not compared (eq:throughput).
    gains: tuple[float, ...] | None = None
    mean_gain: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    # eq:rqformal: all runs keep up and ci_low > 0.
    shortens_service_time: bool


def compare_paired(
    fixed_runs: Sequence[RunMetrics],
    adaptive_runs: Sequence[RunMetrics],
    confidence: float = 0.95,
) -> PairedComparison:
    """eq:rqformal for one matched condition. `fixed_runs` are F_fix runs,
    `adaptive_runs` the runs of one adaptive rule, one run per seed in each,
    over the same seeds. The confidence level is a parameter
    (tab:evalparams fixes 95%), and at least two seeds are needed for an
    interval."""
    fixed_by_seed = _by_seed(fixed_runs, "fixed_runs")
    adaptive_by_seed = _by_seed(adaptive_runs, "adaptive_runs")
    if set(fixed_by_seed) != set(adaptive_by_seed):
        raise ValueError(
            "fixed_runs and adaptive_runs must cover the same seeds: the comparison "
            "is paired seed by seed (eq:rqformal)."
        )
    if len(fixed_by_seed) < 2:
        raise ValueError("a confidence interval over seeds needs at least two seeds.")

    if any(run.storage_mode != "fixed" for run in fixed_runs):
        raise ValueError("every run in fixed_runs must use storage_mode='fixed' (F_fix).")
    adaptive_modes = {run.storage_mode for run in adaptive_runs}
    if len(adaptive_modes) != 1 or "fixed" in adaptive_modes:
        raise ValueError("adaptive_runs must all use one adaptive storage rule, not F_fix.")

    seeds = sorted(fixed_by_seed)
    for seed in seeds:
        for name in _MATCHED_FIELDS:
            if getattr(fixed_by_seed[seed], name) != getattr(adaptive_by_seed[seed], name):
                raise ValueError(
                    f"seed {seed}: runs differ in {name}, so they are not a matched pair."
                )

    adaptive_mode = adaptive_modes.pop()
    all_keep_up = all(
        run.keeps_up is True for run in (*fixed_by_seed.values(), *adaptive_by_seed.values())
    )
    if not all_keep_up:
        return PairedComparison(
            adaptive_storage_mode=adaptive_mode,
            num_seeds=len(seeds),
            confidence=confidence,
            all_keep_up=False,
            shortens_service_time=False,
        )

    gains = []
    for seed in seeds:
        fixed_time = fixed_by_seed[seed].mean_service_time
        adaptive_time = adaptive_by_seed[seed].mean_service_time
        # A run that keeps up has finished tasks, so both are set.
        assert fixed_time is not None and adaptive_time is not None
        gains.append(fixed_time - adaptive_time)

    mean_gain, ci_low, ci_high = _mean_and_interval(gains, confidence)
    return PairedComparison(
        adaptive_storage_mode=adaptive_mode,
        num_seeds=len(seeds),
        confidence=confidence,
        all_keep_up=True,
        gains=tuple(gains),
        mean_gain=mean_gain,
        ci_low=ci_low,
        ci_high=ci_high,
        shortens_service_time=ci_low > 0.0,
    )


def _by_seed(runs: Sequence[RunMetrics], name: str) -> dict[int, RunMetrics]:
    by_seed: dict[int, RunMetrics] = {}
    for run in runs:
        if run.seed in by_seed:
            raise ValueError(f"{name} has two runs with seed {run.seed}; pairing needs one each.")
        by_seed[run.seed] = run
    return by_seed


def _mean_and_interval(values: Sequence[float], confidence: float) -> tuple[float, float, float]:
    """Mean and two-sided t confidence interval of `values`."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    half_width = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1) * math.sqrt(variance / n)
    return mean, mean - half_width, mean + half_width
