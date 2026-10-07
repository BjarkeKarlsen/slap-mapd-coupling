"""F_fix: the identity storage rule, i.e. the fixed-storage baseline.

F_fix(x, .) = x, ignoring demand/traversal/waiting entirely (sec:pf:storage,
the paragraph surrounding eq:storageupdate) -- defined directly, not as the
informal limit Delta = infinity. `storage_epoch_length=None` on
ExperimentConfig already encodes Delta = infinity for storage_mode="fixed"
(core/experiment_config.py), so there is nothing epoch-scheduling-related
for this rule to do either: it is never actually invoked at a storage
epoch in the first place when Delta = infinity, and if it ever is called
regardless, returning x_prev unchanged is still exactly correct.
"""

from __future__ import annotations

from slap_mapd_coupling.core.storage_state import StorageState
from slap_mapd_coupling.storage.base import DemandEstimate, TrafficEstimate
from slap_mapd_coupling.storage.registry import register_storage_rule


@register_storage_rule("fixed")
def f_fix(
    x_prev: StorageState,
    demand_estimate: DemandEstimate,
    traversal_estimate: TrafficEstimate,
    waiting_estimate: TrafficEstimate,
) -> StorageState:
    return x_prev
