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

Before implementing or reviewing anything against this model, validate it
against the actual thesis document, not just this repo's own docs or
memory of an earlier conversation: the Introduction chapter for scope,
terminology, and problem framing; the Method chapter for the formal
model, notation, and algorithmic definitions — whichever fits what's
being checked. This repo doesn't hold that document itself (see "Keep
this repo cloneable on its own" in the workspace root's own conventions,
if you have access to it) — ask for the relevant chapter text or a
pasted excerpt if it isn't already available. Treat a mismatch between
this repo's code and the thesis text as something to resolve explicitly
(see the correctness invariants and the `pr-description` skill's gap-
flagging convention), not something to silently reconcile either way.

## Build / test / run

The primary dev machine uses a shared `uv`-managed virtualenv at
`~/.venvs/rl` (Python 3.10) rather than a per-project one — see
`docs/GETTING_STARTED.md` for full setup. For any non-interactive /
scripted use (CI, an agent), call its binaries directly rather than
activating first, since an activation doesn't persist across separate
invocations:

```bash
uv pip install --python ~/.venvs/rl/bin/python -e ".[dev]"
~/.venvs/rl/bin/python -m pytest
~/.venvs/rl/bin/slap-mapd simulate --storage-rule fixed --controller centralised
```

Interactively (a human, one shell): `source ~/.venvs/rl/bin/activate` once,
then plain `pip install -e ".[dev]"` / `pytest` / `slap-mapd ...` as shown
in `docs/GETTING_STARTED.md`.

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
- When a bug, limitation, or feature idea needs documenting, open a
  GitHub issue (`gh issue create`) — don't write it into a standalone
  markdown file (a `bugs.md`, a `TODO.md`, etc.). A prior `bugs.md` was
  removed for exactly this reason: it duplicated content that was also
  filed as issues, and the two would only drift apart over time. If a
  bug is already fixed by the time it's documented, still file the
  issue and close it immediately with a note — that keeps a searchable
  record without the duplication.
