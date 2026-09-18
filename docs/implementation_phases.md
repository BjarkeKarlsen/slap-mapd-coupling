## Implementation Phases

The generic phases this file used to list (from the original template
scaffold) never matched this project — they're gone. This is the real
build order, milestone by milestone. Each milestone builds on the last;
don't start one before the previous is done and tested.

### M0: Scaffold — done
- [x] Renamed `hierarchical_robot_rl` -> `slap_mapd_coupling`, real module
      layout (`core`, `instances`, `storage`, `resolution`, `controllers`,
      `environment`, `models`, `training`, `evaluation`, `viz`)
- [x] `pyproject.toml`/`requirements.txt`/`environment.yml`, `.gitignore`
- [x] `AGENTS.md`/`CLAUDE.md`, `LICENSE`
- [x] CLI shape (`simulate`/`train`/`sweep` in `main.py`) — verbs only,
      bodies still `raise NotImplementedError`

### M1: Core domain model — done
- [x] `core/graph.py` — `WarehouseGraph`, `Vertex`, `VertexRole`
      (non-exclusive roles), `Edge`, `Action`
- [x] `core/agents.py` — `AgentState`, `FleetState`, collision checks
- [x] `core/tasks.py` — `Task` (timestamp-derived lifecycle, permanent
      assignment)
- [x] `core/storage_state.py` — `StorageState` (feasibility-checked),
      `SkuType`
- [x] `core/experiment_config.py` — `ExperimentConfig` (the 5 independent
      variables + TBD-parameter validation)
- [x] `instances/generator.py` — parametric warehouse generator,
      `generate_instance` (graph + layout position)
- [x] `instances/validation.py` — connectivity + well-formedness checks
- [x] `evaluation/metrics.py` — `RunMetrics`
- [x] `viz/` — `theme.py`, `layout.py`, `warehouse_plot.py` (matplotlib),
      `pygame_renderer.py` (interactive, optional `[viz]` extra)
- [x] `examples/00_build_instance.py`, `00b_pygame_preview.py`,
      `01b_storage_and_config.py`
- [x] 62 unit tests, 93% coverage, CI (`test` + `lint` jobs), black/
      flake8/mypy all clean

**Still stubs, docstring-only: `storage/`, `resolution/`, `controllers/`,
`environment/`, `models/`, `training/`.** Everything below is building
those, in order.

### M2: The baseline — `F_fix` + centralised controller, next up
This is the thesis's own "fixed-storage baseline": deterministic, no
learning. Concretely:
- [ ] `resolution/conflict_resolution.py` — the conflict-resolution
      operator (fixed priority permutation, sequential accept-or-wait)
- [ ] `controllers/assignment.py` — shared greedy task assignment
      (identical across all three controller architectures)
- [ ] `controllers/centralised.py` — prioritised planning (space-time
      A*) over the full graph
- [ ] `controllers/base.py` + `controllers/registry.py` — the Protocol +
      registry so later controllers plug in without touching the loop
- [ ] `storage/base.py` + `storage/registry.py` + `storage/fixed.py` —
      `F_fix` (identity rule) behind the same kind of registry
- [ ] `environment/spaces.py` — action space + the per-vertex legality
      mask
- [ ] `environment/reward_function.py` — `R_i(t)` (potential-based
      shaping + delivery/override/congestion terms)
- [ ] `environment/multi_agent_env.py` — `WarehouseMAPDEnv`, the
      six-stage step loop, wired to `F_fix` + the centralised controller
- [ ] `evaluation/evaluator.py` — runs episodes, produces `RunMetrics`
      rows
- [ ] The four correctness invariants as tests (no vertex/swap conflicts,
      storage feasibility, task-lifecycle disjointness, deterministic
      replay under a fixed seed) — these are the acceptance test for
      this milestone, not optional extras
- [ ] `examples/01_baseline_fixed_centralised.py` filled in

**Done when:** one episode runs end-to-end through the real six-stage
loop with zero collisions, feasible storage, correct task-set
disjointness, and bit-for-bit deterministic replay.

### M3: Storage ladder — `F_dem`, `F_cng`
- [ ] `storage/demand.py` — demand-only slotting heuristic
- [ ] `storage/congestion.py` — demand ranking + congestion penalty
      (needs `β`, `ν` as named, required `ExperimentConfig` fields —
      already modelled; don't invent values, sweep them)
- [ ] `examples/02_storage_ladder.py` filled in — should require zero
      changes to `environment/` (that's the point of the registry)

### M4: Section-based controller
- [ ] `controllers/section_based.py` — zone-local prioritised planning +
      boundary yield protocol
- [ ] `examples/03_section_based.py` filled in

### M5: Decentralised controller
- [ ] Observation builder (`o_i(t)`, `eq:observation`) — likely a new
      `environment/observation.py`
- [ ] `models/gnn_encoder.py` — multi-round message passing
- [ ] `models/base_model.py` — the policy/value head over the encoder
- [ ] `controllers/decentralised.py` — wraps the learned policy
- [ ] `training/config.py`, `training/trainer.py` — PPO wiring, parameter
      sharing, disjoint train/eval seeds, `F_fix`-only vs matched
      training regimes
- [ ] `examples/04_decentralised_training.py` filled in

### M6: Full comparison
- [ ] `examples/05_full_comparison.py` — the sweep across storage rule x
      controller x congestion x communication x load x seed
- [ ] `main.py`'s `sweep` command actually implemented (currently
      `NotImplementedError`)

### M7: Docs catch-up
- [ ] `docs/API_REFERENCE.md`, `docs/MODELS.md`, `docs/TRAINING.md`,
      `docs/EVALUATION.md`, `docs/index.md`, `docs/main.md` — still
      empty; fill in once the modules they document exist
- [ ] `docker-compose.yml` — still empty/unconfigured
