---
paths:
  - "src/slap_mapd_coupling/storage/**"
  - "src/slap_mapd_coupling/resolution/**"
  - "src/slap_mapd_coupling/controllers/**"
  - "src/slap_mapd_coupling/environment/**"
  - "src/slap_mapd_coupling/models/**"
  - "src/slap_mapd_coupling/training/**"
---

# Stub modules: build order

This module is (or recently was) a docstring-only stub. Before implementing
anything in it, check `docs/implementation_phases.md`'s build order —
`core/` and `instances/` must be real first, since every later module
depends on their types. Don't jump ahead to `controllers/`, `models/`, or
`training/` before that.
