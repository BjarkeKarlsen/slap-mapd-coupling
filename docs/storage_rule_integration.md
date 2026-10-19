## New Storage Rule Integration

To add a new storage rule (a new `F` realising `eq:storageupdate`):

1. Create `src/slap_mapd_coupling/storage/<name>.py`, implementing the
   `StorageRule` protocol in `storage/base.py`:
   `x_t = F(x_{t-Delta}, rho_hat_t, mu_hat_t, w_hat_t) -> x_t`
2. Register it in `src/slap_mapd_coupling/storage/registry.py` under a
   short name (e.g. `"fixed"`, `"demand"`, `"congestion"`)
3. Add a unit test under `tests/unit/` asserting the result stays feasible
   (`eq:feasiblestorage`) and respects the reassignment cap `nu`
4. Add/point an `examples/02_storage_ladder.py` run at it

Because `environment/multi_agent_env.py` only ever calls the active rule
through the registry, adding a rule never touches the environment loop,
the controllers, or evaluation.

At an epoch the environment passes the three estimates of the estimates
equation, counted per step over the last `Delta` steps only and reset after
every epoch (`environment/storage_estimates.py`). A SKU or edge with no event
in the window has no entry, so a rule reads it with `.get(key, 0.0)`.

Usage:
```python
from slap_mapd_coupling.storage.registry import get_storage_rule

rule = get_storage_rule("congestion")
x_t = rule(x_prev, demand_estimate, traversal_estimate, waiting_estimate)
```

## Put-away rules under the arrival stock model

Under the arrival stock model (`stock_model: "arrival"`, set by the
`arrival` section of a training spec) stock runs out. An order takes one
unit of its SKU when it is released, and every `Delta` steps new stock
arrives at the receiving vertices. The storage update is then the put-away
of that stock. The rule decides which shelves the arriving units go to, and
nothing already on a shelf moves. The arrival model is not yet in the
thesis.

The put-away rules live in `src/slap_mapd_coupling/storage/arrival.py`,
with their own registry, separate from `storage/registry.py`, under the
same three names as the storage modes (`"fixed"`, `"demand"`,
`"congestion"`). Every rule has the shared signature of the `PutAwayRule`
protocol:

```python
rule(x_prev, graph, amounts, home, demand_estimate, traversal_estimate,
     waiting_estimate, congestion_weight) -> PutAwayResult
```

- `amounts` are the units of each SKU that arrive, from `arrival_amounts`.
  They are the same under every rule, so the rules differ only in where
  the units go.
- `home` maps each SKU to its home shelf, the storage vertex holding most
  of its starting units. Only `"fixed"` uses it.
- `congestion_weight` is beta. Only `"congestion"` uses it.
- `PutAwayResult` holds the new storage state, the put-away distance, and
  the units that arrived, that fit nowhere, and that `"fixed"` placed off
  the home shelf. The environment logs them per arrival, and
  `evaluation/evaluator.py` reduces them to the put-away fields of
  `RunMetrics`.

The environment picks the rule once, in its constructor: under the arrival
model it calls `get_putaway_rule(config.storage_mode)`, and at every
arrival epoch stage 6 computes the amounts and calls that rule in place of
the template model's storage rule. Under the template model the put-away
registry is never read.

Adding a put-away rule:

1. Implement it in `storage/arrival.py` (or a new module that the
   environment imports), with the `PutAwayRule` signature, and register it
   with `@register_putaway_rule("<name>")`, the name of the storage mode it
   realises under the arrival model.
2. Add a unit test under `tests/unit/` asserting that the capacity holds,
   that no unit already on a shelf moves, and that every arriving unit is
   either placed or counted unplaced (see `tests/unit/test_arrival_rules.py`).
