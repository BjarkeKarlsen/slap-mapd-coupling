# Bugs found and fixed

Real bugs caught while implementing and verifying the plan for
`core/`, `instances/`, `core/experiment_config.py` and
`evaluation/metrics.py`. Each was caught by actually running the code
(tests, examples, or ad-hoc checks) rather than by inspection — listed
here so the pattern isn't repeated in the modules still to be built
(`controllers/`, `environment/`, `resolution/`, `models/`, `training/`).

## 1. `model_copy(update=...)` silently skips validation (Pydantic v2)

**Where:** `core/storage_state.py::StorageState.with_units`,
`core/tasks.py::Task.assign`, `core/tasks.py::Task.complete`.

**What happened:** All three methods produce a new immutable snapshot
from an existing one using `self.model_copy(update={...})`. This looks
like the obvious way to do it, but **Pydantic v2's `model_copy` does not
re-run `@model_validator`s** — it's a plain, unvalidated field copy. So:

- `StorageState.with_units("coffee", vertex, 17)` could push a vertex
  over capacity and the resulting object would silently violate
  `eq:feasiblestorage` instead of raising.
- `Task.assign(agent_id, t)` could set `assignment_time` earlier than
  `release_time` (`y_j < r_j`) and the resulting `Task` would silently
  violate the ordering constraint instead of raising.
- `Task.complete(t)` had the same gap for `completion_time < assignment_time`.

**How it was caught:** `test_with_units_reraises_on_infeasible_update`
failed with `Failed: DID NOT RAISE ValidationError` when the test suite
was actually run — the plan's code sketch assumed `model_copy` behaved
like a validated constructor and nothing caught that assumption until
the test executed.

**Fix:** construct a fresh instance instead of copying, so the model's
own validators run again:
```python
# Wrong -- skips every @model_validator on StorageState:
return self.model_copy(update={"counts": new_counts})

# Right -- re-validates:
return StorageState(skus=self.skus, capacities=self.capacities, counts=new_counts)

# Same fix for Task, using model_validate + model_dump to avoid
# repeating every field name:
return Task.model_validate({**self.model_dump(), "assigned_agent": agent_id, "assignment_time": t})
```

**Rule going forward:** never use `model_copy(update=...)` for a state
transition where the update could violate a validator. Use
`model_validate({**self.model_dump(), **changes})` or an explicit
re-construction instead. This applies to every later module that
produces one immutable snapshot per timestep.

## 2. Generator: one-way bridge edges stranded whole vertices

**Where:** `instances/generator.py::generate_warehouse_graph`.

**What happened:** The `one_way_fraction` randomisation was applied to
*every* logical edge, including bridge edges — the single edge
connecting a delivery vertex, an endpoint vertex, or a corridor vertex
to its aisle stack. When a bridge edge was realised one-way in the wrong
direction, the vertex on the far end ended up with **zero edges at all**
(not "harder to reach", literally unreachable from anywhere and unable
to reach anywhere). That's a disconnected graph, not a legitimate
one-way aisle.

**How it was caught:** running `examples/00_build_instance.py` reported
`connectivity.ok: False` on every seed with a failure rate that tracked
almost exactly `one_way_fraction × 0.5` (the probability a given bridge
edge both got randomised *and* landed pointing the wrong way) — too
regular to be the generic "some seeds are just unlucky" case, which is
what prompted actually inspecting which vertices were unreachable.

**Fix:** split edges into `bridge_edges` (always bidirectional) and
`interior_edges` (within an aisle, or along a corridor/cross-aisle row —
these have an alternate route, so realising one as one-way is a
legitimate non-lattice feature, not a defect). Only `interior_edges` are
eligible for `one_way_fraction`.

## 3. Generator: endpoint/delivery cycling dropped vertices when count > aisle count

**Where:** `instances/generator.py::generate_warehouse_graph`.

**What happened:**
```python
for col in range(params.num_aisles):
    target = endpoint_ids[col % len(endpoint_ids)]
    ...attach target to that column...
```
This cycles through **columns**, indexing into `endpoint_ids` — fine
when `num_endpoints <= num_aisles`, but when `num_endpoints >
num_aisles` (e.g. 10 endpoints, 6 aisles), `col` only ever reaches
`0..5`, so `endpoint_ids[6:10]` are never indexed and never get an edge
at all. Same bug for `delivery_ids`.

**How it was caught:** re-checked connectivity with
`one_way_fraction=0.0` (i.e. no randomisation at all) after fixing bug
\#2 above, expecting a fully bidirectional graph to be trivially fully
connected. It wasn't — `check_connectivity` still reported unreachable
targets. Inspecting one of them (`out_neighbours` and in-edges both
empty) showed a genuinely edge-less vertex, which is what pointed at the
attachment loop rather than the one-way logic.

**Fix:** iterate over the endpoint/delivery vertices themselves and
cycle the *column*, not the other way around:
```python
for i, target in enumerate(endpoint_ids):
    col = i % params.num_aisles
    ...attach target to that column...
```

## 4. Generator: endpoint/delivery *position* didn't follow the column fix above

**Where:** `instances/generator.py::_build` (position assignment, added
while building `viz/`).

**What happened:** bug #3's fix made endpoint `i`'s *edge* attach to
column `i % num_aisles`, but its *position* (used only for plotting) was
still `(col=i, row=-1)` — positioned by its own sequential index, not by
the column it's actually wired to. Whenever `num_endpoints > num_aisles`,
the excess endpoints' displayed position ended up far from their real
attachment column.

**How it was caught:** not by a test — by actually looking at the
rendered `examples/output/instance.png`. It showed one long diagonal
line stretching across the entire figure, from the leftmost aisle
column up to the rightmost endpoint marker. A test asserting "every
vertex has a position" would never have caught this; only looking at
the picture did.

**Fix:** position endpoint/delivery vertices by the column they attach
to (`i % num_aisles`), stacking one row further out per repeat when
there are more of them than aisles, instead of positioning by `i`
directly:
```python
add_vertex("endpoint", -1 - i // params.num_aisles, i % params.num_aisles, ...)
```
**Rule going forward:** for any visualisation-adjacent code, actually
render it and look at the output at least once. Structural unit tests
(every vertex has a position, positions match vertex IDs) do not catch
"the picture looks wrong."

## Not a bug: strict well-formedness rarely holds on this generator's layout

Not listed as a bug because it isn't one — `check_well_formedness` is
implementing the definition correctly (proven by the unit tests built
directly from the thesis's own figure panels). It's a structural fact
about `generate_warehouse_graph`'s topology: a single-file aisle stack
with only a couple of cross-aisle rows is close to a tree, so once there
are more than a couple of storage/delivery vertices, some third
"endpoint" (recall: for this specific check, storage and delivery
vertices count as endpoints too) almost always sits on the only path
between two others. `examples/00_build_instance.py` reports this
honestly (`require_well_formed=False` for acceptance, well-formedness
checked and printed separately) rather than tuning parameters until it
happens to pass. A baseline that genuinely needs well-formedness will
need a denser, more lattice-like generator — noted here so it isn't
mistaken for a defect later.

## Environment/tooling: "Import could not be resolved" in the editor

Not a code bug, but the same class of "only visible once you actually
run it" issue. The editor's Python language server was resolving
imports against a different interpreter than the one the project is
installed into. Fixed by:
1. Editable-installing the project into `~/.venvs/rl`
   (`uv pip install --python ~/.venvs/rl/bin/python -e ".[dev]"`).
2. Adding `.vscode/settings.json` with
   `"python.defaultInterpreterPath": "~/.venvs/rl/bin/python"`.

See `docs/GETTING_STARTED.md` for the full setup guide. `.vscode/` is
gitignored (a personal setting), so this fix is local to this machine
until/unless that's changed.
