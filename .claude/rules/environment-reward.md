---
paths:
  - "src/slap_mapd_coupling/environment/**"
---

# Environment & reward: undetermined parameters

Prefer plan mode before writing code here: the environment loop and reward
shape are fixed by the source-of-truth math model this repo implements
(see AGENTS.md's "Source of truth"), but several of their parameters are
still open. If a value isn't obviously pinned down by this repo's own
`README.md` or `docs/`, treat it as undetermined. Flag any undetermined
value you're about to hard-code rather than picking one silently — ask or
leave it a named, configurable parameter instead.
