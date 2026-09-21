"""StorageRule protocol: x_t = F(x_{t-Delta}, rho_hat_t, mu_hat_t, w_hat_t)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, StorageState

EdgeKey = tuple[VertexId, VertexId]  # (source, target), keys mu_hat_t/w_hat_t below
DemandEstimate = dict[SkuId, float]  # rho_hat_t(k)
TrafficEstimate = dict[EdgeKey, float]  # mu_hat_t(e) or w_hat_t(e); same shape, different meaning


@runtime_checkable
class StorageRule(Protocol):
    """F in eq:storageupdate: x_t = F(x_{t-Delta}, rho_hat_t, mu_hat_t, w_hat_t).

    A plain callable, not an object with a named method -- matching
    docs/storage_rule_integration.md's usage
    (`rule = get_storage_rule(name); x_t = rule(x_prev, ...)`), and eq:F's
    own characterisation as "every admissible way of turning a prior
    configuration and the three estimates into a new feasible one." All
    three estimates are causal (computed only from data available before
    the current epoch, sec:pf:storage) -- that is the caller's
    responsibility to uphold, not something this Protocol can enforce
    structurally. Implementations must return a StorageState satisfying
    eq:feasiblestorage; StorageState's own validator already rejects an
    infeasible result at construction time, so a rule that overfills a
    vertex fails loudly rather than silently.
    """

    def __call__(
        self,
        x_prev: StorageState,
        demand_estimate: DemandEstimate,
        traversal_estimate: TrafficEstimate,
        waiting_estimate: TrafficEstimate,
    ) -> StorageState: ...
