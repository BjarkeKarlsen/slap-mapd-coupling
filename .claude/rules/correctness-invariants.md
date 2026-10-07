---
paths:
  - "src/slap_mapd_coupling/environment/**"
  - "src/slap_mapd_coupling/resolution/**"
  - "src/slap_mapd_coupling/storage/**"
  - "src/slap_mapd_coupling/controllers/**"
---

# Correctness invariants (non-negotiable)

Any change to `environment/`, `resolution/`, `storage/`, or `controllers/`
must keep all four of these true. A failure in any of them is a
correctness bug, not a metric to tune around:

1. No vertex/swap conflicts on a realised trace.
2. Every storage configuration stays within capacity.
3. Every task belongs to exactly one lifecycle set at a time.
4. Replay is bit-for-bit deterministic under a fixed seed.
