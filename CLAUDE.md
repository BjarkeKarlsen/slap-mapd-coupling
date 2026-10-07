@AGENTS.md

## Claude Code

This repo's guidance is split across a few files. If you're not familiar
with Claude Code's conventions, here's what each one is for and why the
same fact sometimes appears in two places:

- **`AGENTS.md`** is the authoritative, tool-agnostic doc — status, build
  order, conventions. Both Claude Code and any other AGENTS.md-reading
  tool read it directly, and it always loads in full (via the `@AGENTS.md`
  import above), every session, regardless of what you're working on.
- **`CLAUDE.md`** (this file) is Claude-Code-specific and always loads
  too. Keep it short — it's for things only Claude needs to know, not a
  second copy of `AGENTS.md`.
- **Always use `~/.venvs/rl/bin/python` / `~/.venvs/rl/bin/pytest` /
  `~/.venvs/rl/bin/slap-mapd`, not bare `python`/`pytest`/`slap-mapd`.**
  The Bash tool doesn't persist shell state between calls, so a
  `source ~/.venvs/rl/bin/activate` in one call wouldn't carry over to
  the next one anyway — see `AGENTS.md`'s "Build / test / run" and
  `docs/GETTING_STARTED.md` for the full picture.
- **`.claude/rules/*.md`** are Claude-Code-specific and *path-gated*: each
  has a `paths:` frontmatter glob and only loads into context when a
  matching file is opened or edited, instead of sitting in every session
  regardless of relevance. A few rules here restate a specific slice of
  an `AGENTS.md` convention in path-gated form, on purpose — the
  `AGENTS.md` copy stays the complete reference for humans and other
  tools, the rule is what actually surfaces at the moment it matters:
  - `stub-build-order.md` — build-order check for the still-stub modules
    (`storage/`, `resolution/`, `controllers/`, `environment/`, `models/`,
    `training/`).
  - `environment-reward.md` — plan-mode-first and flag-undetermined-values
    guidance, scoped to `environment/` (including `reward_function.py`).
  - `correctness-invariants.md` — the four non-negotiable correctness
    checks, scoped to `environment/`, `resolution/`, `storage/`,
    `controllers/`.
  - `examples-numbering.md` — the milestone-numbering convention, scoped
    to `examples/`.
- **`.claude/skills/*/SKILL.md`** are Claude-Code-specific and
  *action-gated* rather than path-gated: they're invoked for a specific
  task (typing `/name`, or Claude recognizing the task from the skill's
  `description`) instead of loading automatically. This repo has one:
  `pr-description` — the Summary/Test-plan shape for a PR here (cite the
  thesis section/equation implemented, flag any real gap in the thesis
  text instead of silently patching it, list what the tests actually
  cover). The thesis-*writing* skills (drafting prose, not PR text) stay
  in the sibling `thesis-progress` repo, not here — this repo must stay
  cloneable on its own (see the workspace-root `AGENTS.md`), so nothing
  here should assume a sibling folder exists.

When adding new guidance: if it's true regardless of which files are open
(a naming convention, a CLI contract), it belongs in `AGENTS.md`. If it
only matters when specific files are being touched, it's a new
`.claude/rules/*.md` with a `paths:` glob. If it only matters for a
specific task (opening a PR, filing a bug), it's a new
`.claude/skills/*/SKILL.md` instead of growing this file or `AGENTS.md`
indefinitely.
