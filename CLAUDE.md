@AGENTS.md

## Claude Code

- Domain modules under `src/slap_mapd_coupling/` are intentionally
  docstring-only stubs right now. When asked to implement something,
  check `docs/implementation_phases.md`'s build order first — don't jump
  ahead to `controllers/`, `models/`, or `training/` before `core/` and
  `instances/` are real, since every later module depends on their types.
- Prefer plan mode before writing `environment/` or `reward_function.py`
  code: the loop and reward shape are fixed, but several of their
  parameters are open. Flag any undetermined value you're about to
  hard-code rather than picking one silently.
