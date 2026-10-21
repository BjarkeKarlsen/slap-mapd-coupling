# Fleet Load Sweep Quick Start Guide

How to run the Token Passing load sweeps for the 8, 16, 32 and 64-agent warehouses and read off
the main load of each fleet.

## What is measured

The sweep runs fixed storage only. The capacity of a fleet is the highest arrival rate at which
F_fix, and every lower rate, keeps up on every seed, meaning throughput is at least `f_up` times
the rate tasks were released (`eq:throughput`, [A49]). The main load is 0.6 of that capacity, and
every storage rule and every controller of that fleet runs at this one rate, as the paired
comparison needs.

Throughput is not used as the capacity. It counts put-aways, which grow with the rate, so it
keeps rising after the fleet has fallen behind.

## Configs and scripts

- `configs/fleet_{8,16,32,64}/` — one warehouse per fleet size, with centralised, section-based
  and decentralised configs. The sweep reads `centralised.yaml` for the instance and `f_up`.
- `scripts/run_fleet_sweeps.sh` — runs every fleet size in sequence.
- `scripts/load_sweep.py` — one fleet size.
- `scripts/extract_sweep_parameters.py` — collects the capacity of every fleet size.

## Run

```bash
./scripts/run_fleet_sweeps.sh DIR
```

or one fleet size by hand:

```bash
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_SIZE/centralised.yaml \
    --rates-per-agent 0.0125 0.025 0.0375 0.05 0.0625 0.075 0.0875 0.1 \
    --seeds 6 \
    --out DIR/sweep_SIZE_agents
```

`--rates-per-agent` is multiplied by the fleet size m, so one grid serves every fleet.
`--rates` takes absolute rates instead. Add `--storage-modes fixed demand congestion` to see
the adaptive rules at the same rates.

## Refine

Each sweep writes `summary.md` and `capacity.json` with the capacity and the first rate that
fails. Run the same command again with extra rates between the two and the same `--out`. Rows
already there are skipped, and only the new rates run.

## Collect

```bash
python scripts/extract_sweep_parameters.py DIR
```

This prints one row per fleet size (capacity, first failing rate, main load) and writes
`DIR/fleet_parameters.json`. Then set `arrival_rate` in `configs/fleet_{SIZE}/*.yaml` to the
main load.

## Output

```
DIR/
├── sweep_{8,16,32,64}_agents/
│   ├── load_sweep.jsonl   # one RunMetrics row per (rule, rate, seed)
│   ├── summary.md         # means over seeds, seeds that keep up, capacity
│   └── capacity.json      # capacity, first failing rate and main load per rule
└── fleet_parameters.json  # F_fix of every fleet size
```

## Troubleshooting

- **Sweep interrupted.** Run it again. Rows already written are skipped.
- **Warehouse generation fails (64 agents).** Raise `max_num_cross_aisles` or
  `max_aisle_length`, or lower `num_storage_vertices` or `num_delivery_vertices`, in
  `configs/fleet_64/`.
