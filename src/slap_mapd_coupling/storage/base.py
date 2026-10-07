"""StorageRule protocol for F in the storage update."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from slap_mapd_coupling.core.graph import VertexId, WarehouseGraph
from slap_mapd_coupling.core.storage_state import SkuId, StorageState

EdgeKey = tuple[VertexId, VertexId]  # (source, target), keys mu_hat_t/w_hat_t below
DemandEstimate = dict[SkuId, float]  # rho_hat_t(k)
TrafficEstimate = dict[EdgeKey, float]  # mu_hat_t(e) or w_hat_t(e); same shape, different meaning


@runtime_checkable
class StorageRule(Protocol):
    """F in the storage update, x_t = F(x_{t-Delta}, rho_hat_t, mu_hat_t, w_hat_t).

    A plain callable, used as `rule = get_storage_rule(name); x_t = rule(x_prev, ...)`
    (docs/storage_rule_integration.md). The result must satisfy the
    capacity constraint, which StorageState's own validator enforces.
    Keeping the three estimates causal is the caller's job.

    The signature has three arguments that F does not have in the storage
    update. They are plumbing, not part of the model: `graph` gives d_G,
    `reassignment_cap` is nu (sec:method:storage, storage/relocation.py),
    and `congestion_weight` is beta in the congestion-aware score. Rules
    that ignore one receive None.
    """

    def __call__(
        self,
        x_prev: StorageState,
        graph: WarehouseGraph,
        reassignment_cap: int | None,
        congestion_weight: float | None,
        demand_estimate: DemandEstimate,
        traversal_estimate: TrafficEstimate,
        waiting_estimate: TrafficEstimate,
    ) -> StorageState: ...
