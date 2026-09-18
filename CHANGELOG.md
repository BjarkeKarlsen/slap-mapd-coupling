# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

## [0.1.0] - 2026-08-14

### Added

### Changed

### Fixed
