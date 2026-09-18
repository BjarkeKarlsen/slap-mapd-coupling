# slap-mapd-coupling

Simulation and RL controllers for the coupled storage-assignment (SLAP) /
multi-agent pickup-and-delivery (MAPD) system studied in a Master's thesis:
a warehouse graph where a periodic storage rule `F` and a routing
controller `pi` (centralised, section-based, or decentralised-learned) act
on the same graph and the same observed traffic state.

## Status

`core/` (graph, agents, tasks, storage state, experiment config),
`instances/` (generator, validation), `evaluation/metrics.py`'s
`RunMetrics`, and `viz/` (matplotlib + pygame) are implemented and
tested. `storage/`, `resolution/`, `controllers/`, `environment/`,
`models/`, `training/` are still docstring-only stubs — they depend on
method-level parameters the thesis hasn't fixed yet. Check
`docs/implementation_phases.md` for the build order before writing code.

## Source of truth

This repo implements a mathematical model defined elsewhere; it does not
restate that model itself. Do not invent a parameter value that isn't
pinned down — if a value needed to implement something isn't obviously
determined by this repo's own docs (`README.md`, `docs/`), leave it a
named, configurable parameter rather than hard-coding a guess, and flag it
instead of picking silently.

## Build / test / run

```bash
pip install -e ".[dev]"
pytest
slap-mapd simulate --storage-rule fixed --controller centralised
```

## Layout and extensibility

Two swap points, each behind a Protocol + registry, so the environment
loop never changes when a new one is added:

- `storage/` — storage rules (`F_fix`, `F_dem`, `F_cng`) via
  `storage/registry.py`. Adding one: see `docs/storage_rule_integration.md`.
- `controllers/` — routing architectures (centralised / section-based /
  decentralised) via `controllers/registry.py`, sharing one task-assignment
  function (`controllers/assignment.py`). Adding one: see
  `docs/controller_integration.md`.

Everything else (`core/`, `instances/`, `resolution/`, `environment/`,
`models/`, `training/`, `evaluation/`, `viz/`) is described in `README.md`.

## Conventions

- "Controller" means a routing architecture (`pi`); "policy" is reserved
  for the decentralised controller's neural net (`pi_theta`) specifically.
  Don't use the two interchangeably.
- Adding a controller or storage rule needs, in this order: an
  implementation behind the matching protocol, a registry entry, a unit
  test, and an `examples/` script (see the integration docs above).
- Four correctness checks are non-negotiable for any change to
  `environment/`, `resolution/`, `storage/`, or `controllers/`: no
  vertex/swap conflicts on a realised trace, every storage configuration
  stays within capacity, every task belongs to exactly one lifecycle set
  at a time, and replay is bit-for-bit deterministic under a fixed seed.
  Treat a failure in any of these as a correctness bug, not a metric to
  tune around.
- `examples/NN_name.py` files are numbered by build milestone
  (`docs/implementation_phases.md`), not by topic. Keep new examples in
  that sequence rather than renaming or reordering existing ones.
- CLI verbs are `simulate` / `train` / `sweep` (see `main.py`), not
  `train`/`evaluate`/`generate-config`: only the decentralised controller
  trains, so `--checkpoint` never applies to the other two.
