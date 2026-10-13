# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PPOHyperparameters.training_budget_env_steps`, with no default, since the thesis counts
  the training budget in environment steps and leaves its value TBD.
- `tensorboard` in the dev extras.
- A test that a saved checkpoint loads through the path example 04b uses.
- `TokenPassingPlanner` (`planners/token_passing.py`, `--planner token_passing`), the
  reference baseline of Ma et al. (2017). It picks its own tasks, so the
  environment loop asks a `SelfAssigningRouter` for the assignment in place of
  the shared `assign_tasks`, and `PlannerRouter.reset` and `WarehouseMAPDEnv.reset`
  now reset a planner that keeps state. The new
  `examples/01_baseline_fixed_centralised.py` runs it, and the prioritised A*
  example is renamed `01_baseline_fixed_prioritised_astar.py`.
- `planners/` package (`Planner` protocol, `PrioritisedAStarPlanner`). The A*
  planner moved out of `controllers/centralised.py`. It is a reference
  planner outside the three controllers, chosen with the new
  `ExperimentConfig.planner` field (exactly one of `controller` and
  `planner` is set) and run through `PlannerRouter` and `get_router`.
  `"centralised"` stays a controller name for the learned arm and raises
  until #86 lands (#98).
- `LICENSE` (MIT).
- `docs/storage_rule_integration.md` and `docs/controller_integration.md`,
  documenting the two extensibility points (mirrors the existing
  `docs/model_integration.md`).
- `AGENTS.md` and `CLAUDE.md` (repo-local, standalone: no reference to
  sibling repos outside this one).
- The first real domain code, as type-strong Pydantic v2 models matching
  the thesis's problem formulation exactly: `core/graph.py`,
  `core/agents.py`, `core/tasks.py`, `core/storage_state.py`,
  `core/experiment_config.py` (new), `instances/generator.py`,
  `instances/validation.py`, and `evaluation/metrics.py`'s `RunMetrics`.
  Fixes two real bugs found in the pre-Pydantic dataclass draft this
  replaces: vertex roles modelled as an exclusive enum when they
  genuinely overlap, and no lifecycle-timestamp model for tasks at all.
  41 unit tests and two runnable `examples/` scripts (`00_build_instance.py`
  filled in, `01b_storage_and_config.py` new) cover this code; see the
  plan this was built from for the full design rationale.
- `pydantic>=2.6` as a dependency (was missing entirely).
- `viz/theme.py` (shared colour/shape palette for both backends),
  `viz/layout.py` (generic fallback graph layout), `viz/warehouse_plot.py`
  filled in (matplotlib, layered/composable), and `viz/pygame_renderer.py`
  (new: an interactive `WarehouseRenderer`, replacing the toy
  `pygame_visualization` example's structure with a build-once/
  render-every-step class over the real `WarehouseGraph`/`FleetState`
  types). `pygame` added as the new `[viz]` optional dependency extra
  (not required, so headless training/CI environments don't need a GUI
  library). `instances/generator.py` gained `generate_instance` (additive:
  returns the graph plus the layout position it already computed
  internally and previously discarded), leaving `generate_warehouse_graph`
  unchanged and regression-tested against it.
- `examples/00b_pygame_preview.py`: a live `WarehouseRenderer` demo.
  `controllers/` doesn't exist yet, so agents move by picking a random
  legal, collision-free action each frame instead of routing anywhere on
  purpose -- swapping that for a real controller later doesn't change
  anything else in the script. `WarehouseRenderer.render` now returns
  `bool` (`False` once the window's close button is clicked) so a driving
  loop can exit cleanly instead of continuing after the window closes.
- `~/.venvs/rl` set up as this project's dev environment (shared across
  RL projects, not per-repo) and `.vscode/settings.json` pointing at it;
  see `docs/GETTING_STARTED.md`.
- `bugs.md`, documenting real bugs found and fixed while implementing
  and verifying the above (not just written from a plan and assumed
  correct).

### Changed
- Examples 02, 03, 04 and 04b run through Ray Tune with Ray's default checkpoints. 02
  (centralised) and 03 (section-based) are thin variants of 04, and 04b takes
  `--checkpoint=<path>` and `--arm`. Each example sets `max_local_nodes` from the graph
  to the largest window its arm uses, so no window is truncated, and builds its instance
  with `generate_instance` and the validation checks so the aisle index is kept.
  04 evaluates one episode per evaluation seed through `run_episode(policy=...)`, and
  `--regime matched` trains under the storage rule used at evaluation.
- `docs/controller_integration.md` describes the `Assigner`, the planner driver and the
  policy function. `docs/TRAINING.md` describes a Tune run, checkpoints and TensorBoard.
- `check_well_formedness` counts non-task endpoints, the vertices of V_ep in neither
  V_str nor V_del, against the fleet size, as the thesis says. It used to count
  V_ep, V_str and V_del together, so an instance with fewer than m vertices in V_ep
  could pass and then fail when the environment placed the agents. The pairwise path
  check still takes every endpoint. `WellFormednessReport` gains
  `num_non_task_endpoints`.
- Run scoring follows the thesis's revised `sec:pf:measures`. `RunMetrics`
  now reports a keep-up check (`keep_up_ratio`, `keeps_up` against the new
  `ExperimentConfig.keep_up_threshold` f_up, left unset because its value
  is still TBD), mean service time split into `mean_wait_for_agent`,
  `mean_travel_time` and `mean_blocked_time` (validated to add up),
  `mean_crowding` (the mean of delta_i(t), computed every timestep
  over the field of view `observation_depth`), and relocations per storage update. The new
  `evaluation/comparison.py` implements the gain test as a paired t-test
  over seeds. Traffic entropy/concentration and movement cost per task are
  no longer reported, though the episode log still records traversals and
  movement cost for the replay-determinism invariant. `scipy` is now a
  declared dependency (it was only installed transitively via ray).
- Rewrote `docs/implementation_phases.md`: it was still the generic
  template checklist (and actively wrong in one place — it listed the
  old `train`/`evaluate`/`inspect`/`generate-config` CLI). Replaced with
  the real milestone-by-milestone build order (M0-M7), marking what's
  actually done vs. still stubbed.
- Renamed the generic `hierarchical_robot_rl` template to
  `slap_mapd_coupling` and replaced its placeholder modules with the
  module layout for the SLAP-MAPD coupling thesis: `core`, `instances`,
  `storage`, `resolution`, `controllers`, `environment`, `models`,
  `training`, `evaluation`, `viz`.
- Replaced the CLI's `train`/`evaluate`/`generate-config` surface (which
  assumed every controller is trained and checkpointed) with
  `simulate`/`train`/`sweep`, matching the experimental design: only the
  decentralised controller trains, and `sweep` runs the full grid over
  storage rule x controller x congestion x communication x load x seed.
- Collapsed `utils/io.py`, `utils/logging.py`, `utils/types.py` (all
  empty) into a single `utils/utils.py` until there's enough content to
  justify separate files.
- Replaced `requirements-dev.txt` (a stale full `pip freeze`) with the
  actual dev dependencies from `pyproject.toml`'s `[dev]` extra.
- Fixed `docs/model_integration.md`'s references to files deleted in this
  restructuring.

### Removed
- `training/utils.py` (`save_checkpoint`, `load_checkpoint`) and every `CHECKPOINT_DIR`.
  Checkpoints go where Ray puts them.
- `bugs.md`. Its content was already duplicated as closed GitHub issues
  #1-6 (see the repo's issue tracker); keeping both was exactly the
  same "two copies drift apart" problem the wiki's Home page was
  deliberately kept free of. Bugs/documentation-worthy findings go
  straight to an issue from now on — see `AGENTS.md`.
- `EpisodeLog.total_movement_cost` and its per-step accumulation in
  `WarehouseMAPDEnv`, with the `_edge_costs` lookup it needed. The scoring
  (`sec:pf:measures`) no longer reports movement cost. `edge_traversals`
  stays, since the storage estimates of #60 will read it.

## [0.1.0] - 2026-08-14

### Added

### Changed

### Fixed
