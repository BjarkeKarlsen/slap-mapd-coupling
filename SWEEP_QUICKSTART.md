# Token Passing sweeps, quick start

How to run the Token Passing sweeps for the 8, 16, 32 and 64-agent warehouses.

## What is measured

The load sweep runs fixed storage only. The capacity of a fleet is the highest arrival rate at
which fixed storage, and every lower rate, keeps up on every seed. A run keeps up when its picks
and its put-aways each finish at least `f_up` of what was released (`eq:throughput`, [A59]). The
main load is 0.6 of that capacity, and every storage rule and controller of that fleet runs at
this one rate ([A58]). The sensitivity sweeps then vary one demand, arrival or storage-rule
setting at a time at that load.

## Run everything

From the repo root:

```bash
nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
    scripts/sweep/overnight.bash > ~/Documents/thesis/thesis-progress/results/overnight.log 2>&1 &
```

`scripts/sweep/overnight.bash` runs the load sweeps, a finer grid around each capacity, and the
sensitivity sweeps at each fleet's main load. `RESULTS`, `LOAD_SEEDS` and `SEEDS` override where
the results go and the seeds per point. A sweep started again skips the rows it already has, so
move old results away before a run that should start from scratch.

## One sweep on its own

```bash
~/.venvs/rl/bin/python scripts/sweep/load_sweep.py --config configs/fleet_8/decentralised.yaml \
    --rates-per-agent 0.0125 0.025 0.0375 0.05 --seeds 20 --out DIR
scripts/sweep/parameter_sweep.bash 8   # needs DIR's capacity.json under RESULTS/arrival_rates
```

## Output

Per sweep: `load_sweep.jsonl` (one row per episode, the fields of `RunMetrics`), `capacity.json`
and `summary.md`. The summary gives the capacity and, per rate, the keep-up check by kind, the
longest stand-still, the backlog slope and the overflow and unplaced units. Service times are
read from the rows, for instance with the plots in `thesis-progress/scripts/results/`.
