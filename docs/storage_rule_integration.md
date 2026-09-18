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

Usage:
```python
from slap_mapd_coupling.storage.registry import get_storage_rule

rule = get_storage_rule("congestion")
x_t = rule(x_prev, demand_estimate, traversal_estimate, waiting_estimate)
```
