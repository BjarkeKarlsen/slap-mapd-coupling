# Load Sweep Roadmap for Fleet Sizes 8, 16, 32, 64

**Status: 2026-10-20**

## What needs to be done

The thesis plan (Section 4.3, RQ4) requires Token Passing load sweeps on all four fleet-size warehouses to:
1. Determine the capacity for each warehouse
2. Confirm that **0.6 of capacity** is a good main load that all arms can keep up with
3. Extract **chi_max** (the maximum healthy backlog slope) from the sweeps
4. Determine **stall_length** (agents standing still threshold) for the keep-up check

## Current state

✓ **8-agent warehouse**: Sweep completed
- Location: `DIR/load_sweep.jsonl`
- Capacity: ~0.515 tasks/timestep
- Rate at 0.6 of capacity: ~0.309 tasks/timestep
- Chi_max: ~0.102 (needs verification from data)

⏳ **16, 32, 64-agent warehouses**: Configs created, sweeps need to run
- Configs: `configs/fleet_{8,16,32,64}/{centralised,section,decentralised}.yaml`
- Visualizations: being generated in `DIR/warehouse_images/`

## Generated warehouse specifications

| Agents | Aisles | Length | Cross-aisles | Storage | Delivery | Receiving | Vertices |
|--------|--------|--------|--------------|---------|----------|-----------|----------|
| 8      | 12     | 10     | 8            | 10      | 4        | 4         | ~94      |
| 16     | 16     | 14     | 11           | 20      | 5        | 5         | ~178     |
| 32     | 24     | 20     | 16           | 40      | 8        | 8         | ~330     |
| 64     | 33     | 28     | 23           | 80      | 11       | 11        | ~650     |

## How to run the sweeps

### Option 1: Run all sweeps in background (recommended for long wait)

```bash
cd /home/bjarkek/Documents/thesis/slap-mapd-coupling
nohup bash /tmp/claude-1000/-home-bjarkek-Documents-thesis/4374a049-65b8-43ce-a697-5f43d373fe30/scratchpad/run_all_sweeps.sh > sweep_progress.log 2>&1 &
tail -f sweep_progress.log  # Monitor progress
```

### Option 2: Run sweeps manually (one at a time)

For each fleet size, run:

```bash
cd /home/bjarkek/Documents/thesis/slap-mapd-coupling

# 8-agent (already done, but can re-run)
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_8/centralised.yaml \
    --rates 0.1 0.2 0.3 0.4 \
    --seeds 6 \
    --storage-modes fixed demand congestion \
    --out DIR/sweep_8_agents

# 16-agent
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_16/centralised.yaml \
    --rates 0.1 0.2 0.3 0.4 \
    --seeds 6 \
    --storage-modes fixed demand congestion \
    --out DIR/sweep_16_agents

# 32-agent
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_32/centralised.yaml \
    --rates 0.1 0.2 0.3 0.4 \
    --seeds 6 \
    --storage-modes fixed demand congestion \
    --out DIR/sweep_32_agents

# 64-agent
~/.venvs/rl/bin/python scripts/load_sweep.py \
    --config configs/fleet_64/centralised.yaml \
    --rates 0.1 0.2 0.3 0.4 \
    --seeds 6 \
    --storage-modes fixed demand congestion \
    --out DIR/sweep_64_agents
```

## Time estimates

Each sweep takes roughly:
- **8-agent**: 20-30 minutes (4 rates × 6 seeds × 3 rules)
- **16-agent**: 40-60 minutes (larger warehouse)
- **32-agent**: 1-2 hours (significantly larger)
- **64-agent**: 2-4 hours (very large warehouse)

**Total estimated time: 4-8 hours** for all three sweeps (8-agent can skip if already done)

## After sweeps are complete

Extract parameters from all sweeps:

```bash
cd /home/bjarkek/Documents/thesis/slap-mapd-coupling
~/.venvs/rl/bin/python /tmp/claude-1000/-home-bjarkek-Documents-thesis/4374a049-65b8-43ce-a697-5f43d373fe30/scratchpad/extract_sweep_parameters.py
```

This will:
1. Read results from `DIR/sweep_{N}_agents/load_sweep.jsonl`
2. Extract capacity, chi_max, and rate_at_0_6 for each
3. Save to `DIR/fleet_parameters.json`
4. Print recommendations for thesis parameters

## Expected outputs

After each sweep completes:
- `DIR/sweep_{N}_agents/load_sweep.jsonl` — raw results (one JSON per line)
- `DIR/sweep_{N}_agents/summary.md` — human-readable summary table

Example output structure:
```
DIR/
├── sweep_8_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── sweep_16_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── sweep_32_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── sweep_64_agents/
│   ├── load_sweep.jsonl
│   └── summary.md
├── fleet_parameters.json          ← Generated after extraction
├── warehouse_images/              ← Visualizations
│   ├── warehouse_8_agents.png
│   ├── warehouse_16_agents.png
│   ├── warehouse_32_agents.png
│   └── warehouse_64_agents.png
└── ...
```

## What happens next

Once all sweeps are complete and parameters extracted:

1. **Update code with chi_max** — set the keep-up check threshold
2. **Use rate_at_0_6 for each fleet size** — update arrival rates in configs
3. **Start long training runs** — RQ1 and RQ2 with confirmed parameters
4. **Run RQ3, RQ4, RQ5** — remaining research questions

## Notes

- Token Passing needs no training, so sweeps are cheap to run (just evaluation)
- Storage modes (fixed, demand, congestion) are evaluated in parallel, no extra time
- Sweeps are resumable — if interrupted, just re-run the same command
- The `load_sweep.py` script appends to existing files, so partial runs don't repeat work
