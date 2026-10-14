# SLAP-MAPD Coupling

Simulation and RL controllers for the coupled storage-assignment (SLAP) /
multi-agent pickup-and-delivery (MAPD) system studied in the thesis: a
warehouse graph where a periodic storage rule `F` and a routing controller
`pi` (centralised, section-based, or decentralised) act on the same
graph and the same observed traffic state. A non-learned reference planner
(prioritised A*) runs alongside them, outside the three controllers.

**Thesis document (read-only, live compiled PDF via Overleaf):**
https://overleaf.trit.au.dk/read/gjzcpzsxbnpr#67ffc6

## Status

`core/`, `instances/`, the `ExperimentConfig`/`RunMetrics` config models,
and `viz/` (matplotlib + pygame) are implemented and tested (47 unit
tests, three `examples/` scripts). `storage/`, `resolution/`,
`controllers/`, `environment/`, `models/`, `training/` are still
docstring-only stubs — they depend on method-level parameters the thesis
hasn't fixed yet. See `docs/implementation_phases.md` for the build
order and the repo's [closed issues](../../issues?q=is%3Aissue+is%3Aclosed)
for real bugs found and fixed along the way.

## Layout

- `core/` — graph, agents, tasks, storage state
- `instances/` — parametric warehouse generator + instance validation
- `storage/` — storage rules (`F_fix`, `F_dem`, `F_cng`) behind a registry
- `resolution/` — the conflict-resolution operator
- `controllers/` — the task-assignment rule shared by every router, and the
  `Assigner` protocol that can replace it
- `planners/` — non-learned reference planners (Token Passing, prioritised A*), outside
  the three controllers
- `environment/` — the simulator core with one step mode, spaces/masking, reward, zones
- `models/` — the GNN encoder and policy/value heads
- `training/` — the RLlib adapter for the three learned arms, PPO wiring, seeding,
  training regimes, the policy function
- `evaluation/` — run metrics and the correctness/validation checks
- `viz/` — warehouse and occupancy plotting

## Quick start

### Installation

The project has no dependency on any particular virtualenv location or
name — `pyproject.toml`/`requirements.txt` are the actual dependency
declarations, and CI (`.github/workflows/ci.yml`) installs into a fresh
one per run. On the primary dev machine there's a `~/.venvs/rl`
virtualenv already set up and shared across RL projects (see
`docs/GETTING_STARTED.md`), but that's a machine-local convenience, not
a project requirement — any virtualenv works:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### CLI

Three verbs, matching the thesis's own experimental design rather than a
generic train/evaluate/checkpoint workflow:

```bash
# Run one (storage rule, controller, load) configuration
slap-mapd simulate --storage-rule fixed --planner prioritised_astar --num-agents 10

# Train a learned controller (examples/04_decentralised_training.py shows the Tune run)
slap-mapd train --storage-rule fixed --num-iterations 500

# Run the full experimental grid and write one results table
slap-mapd sweep --config path/to/experiment_grid.yaml
```

### Docker

```bash
docker build -t slap-mapd-coupling .
docker run --gpus all -it slap-mapd-coupling
```
