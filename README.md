# SLAP-MAPD Coupling

Simulation and RL controllers for the coupled storage-assignment (SLAP) /
multi-agent pickup-and-delivery (MAPD) system studied in the thesis: a
warehouse graph where a periodic storage rule `F` and a routing controller
`pi` (centralised, section-based, or decentralised-learned) act on the same
graph and the same observed traffic state.

## Status

`core/`, `instances/`, and the `ExperimentConfig`/`RunMetrics` config
models are implemented and tested (41 unit tests, two `examples/`
scripts). `storage/`, `resolution/`, `controllers/`, `environment/`,
`models/`, `training/` are still docstring-only stubs — they depend on
method-level parameters the thesis hasn't fixed yet. See
`docs/implementation_phases.md` for the build order.

## Layout

- `core/` — graph, agents, tasks, storage state
- `instances/` — parametric warehouse generator + instance validation
- `storage/` — storage rules (`F_fix`, `F_dem`, `F_cng`) behind a registry
- `resolution/` — the conflict-resolution operator
- `controllers/` — shared task assignment + swappable routing
  (centralised / section-based / decentralised) behind a registry
- `environment/` — the RLlib `MultiAgentEnv`, spaces/masking, reward
- `models/` — the GNN encoder and policy/value heads
- `training/` — PPO wiring, seeding, training regimes
- `evaluation/` — run metrics and the correctness/validation checks
- `viz/` — warehouse and occupancy plotting

## Quick start

### Installation

Uses the shared `~/.venvs/rl` virtualenv, not a per-project one — see
`docs/GETTING_STARTED.md` for the full setup/editor guide. Short version:

```bash
uv pip install --python ~/.venvs/rl/bin/python -e ".[dev]"
```

### CLI

Three verbs, matching the thesis's own experimental design rather than a
generic train/evaluate/checkpoint workflow (which only fits the
decentralised controller):

```bash
# Run one (storage rule, controller, load) configuration
slap-mapd simulate --storage-rule fixed --controller centralised --num-agents 10

# Train the decentralised controller only - the other two need no training
slap-mapd train --storage-rule fixed --num-iterations 500

# Run the full experimental grid and write one results table
slap-mapd sweep --config path/to/experiment_grid.yaml
```

### Docker

```bash
docker build -t slap-mapd-coupling .
docker run --gpus all -it slap-mapd-coupling
```
