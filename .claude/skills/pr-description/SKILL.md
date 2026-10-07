---
name: pr-description
description: Write the PR summary and test plan when opening a pull request in this repo. Use whenever the user asks to open, create, or draft a PR here.
---

# PR description for slap-mapd-coupling

This repo implements a math model defined elsewhere (see `AGENTS.md`'s
"Source of truth"). PR descriptions exist to let a reviewer check the code
against that model without re-deriving it themselves, so they cite the
model directly rather than just describing the code.

## Summary

- Name the file(s)/module implemented and, for each one, the section or
  equation label it implements (e.g. `sec:method:resolution`,
  `eq:reward`, `eq:waiting`) — not just a prose restatement of what the
  code does.
- Call out any new interface surface (a flag, a result field) and which
  downstream equation/consumer it exists for.
- **If implementing it surfaced a real gap, inconsistency, or ambiguity
  in the thesis text — say so explicitly, don't silently patch around
  it.** Structure that as its own bullet or paragraph:
  1. Quote the specific clause of the thesis text that's wrong or
     underspecified.
  2. Give the smallest concrete case that breaks it (e.g. a 2-agent
     example), and walk through why the text's own reasoning doesn't
     cover it.
  3. State the fix actually taken, and why it preserves the same
     properties the spec cares about (e.g. still a single deterministic
     pass, still a pure function of state and seed) rather than loosening
     them.
  4. Note it's documented in-code (module docstring) for reconciling with
     the thesis text later — a PR fixes the code; it doesn't rewrite the
     thesis.
- Do not silently invent or pick a value/behavior the thesis leaves open
  — if one came up, it belongs in the gap bullet above, not folded
  quietly into the summary.

## Test plan

- The commands actually run, with their results, not just "tests pass":
  `pytest` (pass count + coverage %), `black --check`, `flake8`, `mypy`.
- A bullet list of what the new/changed tests actually cover, in plain
  language (not test function names) — e.g. "the head-on swap gap (both
  orderings — both agents wait, no collision)", not "test_swap_gap".
  Group by what's being verified: normal-case correctness, the specific
  edge case(s) that motivated the change, determinism under a fixed seed,
  and rejection of illegal/malformed input.
- If the change touches `environment/`, `resolution/`, `storage/`, or
  `controllers/`, confirm which of the four correctness invariants (see
  `.claude/rules/correctness-invariants.md`) the new tests exercise.

## Workflow

Use `gh pr create --body "$(cat <<'EOF' ... EOF)"` (see the repo-wide git
conventions for the heredoc pattern and commit attribution). Show the
drafted Summary/Test plan to the user for approval before running it,
same as any other PR.

- Remove "Generated with Claude Code" from the PR description before
  submitting, and remove any "IGNORE" comments from the code itself.
