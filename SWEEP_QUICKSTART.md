# Fleet Load Sweep Quick Start Guide

**Branch:** `feat/fleet-load-sweeps`

This guide covers how to run the Token Passing load sweeps for 8, 16, 32, and 64-agent warehouses to extract parameters for your thesis (RQ4).

## What's on this branch

✓ **New warehouse configs** for all fleet sizes:
- `configs/fleet_8/` — 8-agent baseline (already have sweep results)
- `configs/fleet_16/` — 16-agent warehouse
- `configs/fleet_32/` — 32-agent warehouse  
- `configs/fleet_64/` — 64-agent warehouse

Each has centralised, section-based, and decentralised controller configs.

✓ **Sweep scripts**:
- `scripts/run_fleet_sweeps.sh` — Run all sweeps in sequence
- `scripts/extract_sweep_parameters.py` — Extract results from completed sweeps

✓ **Documentation**:
- `SWEEP_ROADMAP.md` — Detailed roadmap
- This file — Quick start guide

## Quick Start

### 1. Checkout the branch locally

```bash
git fetch origin
git checkout feat/fleet-load-sweeps
```

### 2. Run all load sweeps

**Option A: Automatic (recommended)**
```bash
./scripts/run_fleet_sweeps.sh
```

This runs Token Passing sweeps for 8, 16, 32, and 64-agent warehouses in sequence.

**Option B: Manual (one at a time)**
```bash
# For each fleet size: 8, 16, 32, 64
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_SIZE/centralised.yaml \
    --rates 0.1 0.2 0.3 0.4 \
    --seeds 6 \
    --storage-modes fixed demand congestion \
    --out DIR/sweep_SIZE_agents
```

### 3. Extract parameters

Once all sweeps are complete:

```bash
python scripts/extract_sweep_parameters.py
```

This generates `DIR/fleet_parameters.json` with:
- `capacity` — max throughput (tasks/step) for each fleet size
- `chi_max` — max healthy backlog slope (keep-up threshold)
- `rate_at_0_6` — arrival rate at 0.6 of capacity (recommended main load)

### 4. Update your configs and code

Using the extracted parameters:

1. Update `configs/fleet_{SIZE}/*.yaml` with the `arrival_rate` from `rate_at_0_6`
2. Set `chi_max` in your keep-up check (evaluation code)
3. Proceed with long training runs (step 5 of thesis plan)

## Folder structure after sweeps

```
DIR/
├── load_sweep.jsonl              # 8-agent results (existing)
├── summary.md                    # 8-agent summary
├── sweep_8_agents/
│   ├── load_sweep.jsonl         # results for each storage mode
│   └── summary.md               # human-readable summary
├── sweep_16_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── sweep_32_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── sweep_64_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
└── fleet_parameters.json        # extracted parameters (generated)
```

## Time estimates

- **8-agent**: 20-30 minutes (already done, shows as ~0 if re-run)
- **16-agent**: 40-60 minutes
- **32-agent**: 1-2 hours
- **64-agent**: 2-4 hours

**Total: 4-8 hours** for the three new sweeps

You can run this in the background or overnight:
```bash
nohup ./scripts/run_fleet_sweeps.sh > sweep_run.log 2>&1 &
tail -f sweep_run.log  # Monitor progress
```

## What to expect in the output

Each sweep generates:

### `load_sweep.jsonl` (raw results)
One JSON object per line, one line per (storage_mode, arrival_rate, seed) combination.
Fields include: throughput, backlog_slope, mean_service_time, etc.

### `summary.md` (readable summary)
A markdown table showing per-rate aggregates (means over seeds) for:
- Throughput
- Backlog slope
- Service times
- Capacity calculation
- Rate at 0.6 of capacity

## Understanding the results

After extraction, check the summary table:

| Agents | Capacity | Chi_max | Rate at 0.6x |
|--------|----------|---------|-------------|
| 8      | 0.515    | ~0.1    | 0.309       |
| 16     | ...      | ...     | ...         |

**What each column means:**
- **Capacity**: Maximum throughput Token Passing sustains (highest mean across rates)
- **Chi_max**: Maximum healthy backlog slope (used in keep-up check)
  - Healthy runs: slope ≤ chi_max
  - Degrading runs: slope > chi_max
- **Rate at 0.6x**: Recommended arrival rate (main load) = 0.6 × capacity
  - Heavy enough to show differences between controllers
  - Light enough that all arms can keep up

## Troubleshooting

### Sweep interrupted?
The script appends to existing files, so just re-run it and it will skip completed runs.

### Parameter extraction fails?
Check that all sweeps have completed:
```bash
for SIZE in 8 16 32 64; do
  echo "=== $SIZE agents ==="
  ls -lh DIR/sweep_${SIZE}_agents/load_sweep.jsonl 2>/dev/null || echo "Not done"
done
```

### Warehouse generation fails (64-agent)?
The parameters might be infeasible for the random topology generator. If so:
- Increase `max_num_cross_aisles` or `max_aisle_length` in `configs/fleet_64/`
- Reduce `num_storage_vertices` or `num_delivery_vertices`
- Or use different `seed` values for generation

## Next steps in thesis plan

After extracting parameters:

1. **Step 5**: Long run — Train all three arms with confirmed loads
2. **Step 6**: Main result figures and tables
3. **Step 7**: RQ3 check (congestion heatmap)
4. **Step 8**: RQ4 sweep (transfer + training at larger sizes)
5. **Step 9**: RQ5 (communication, if time)

---

**Questions?** Check `SWEEP_ROADMAP.md` for detailed background.
