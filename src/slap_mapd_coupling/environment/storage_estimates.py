"""The online estimates rho_hat_t, mu_hat_t and w_hat_t of the storage update
(sec:pf:storage, the estimates equation, and sec:method:storage).

Each estimate looks back over one storage period only. At an epoch t it counts
the events of the window [t - Delta, t), divides the count by the Delta steps
of the window, and the next window starts from zero:

    rho_hat_t(k) = #{orders for k released in the window} / Delta
    mu_hat_t(e)  = #{traversals of e in the window} / Delta
    w_hat_t(e)   = #{waits attributed to e in the window} / Delta

Counting per step keeps the size of the estimates, and so the weight beta of
the congestion-aware score, the same whatever Delta is.

The estimates are causal. The window ends before the epoch, so the order
released at t itself (stage 1 of the step that follows the epoch) and everything
after it belong to the next window. The environment records an event the step
it happens and calls `close_window` at stage 6, after stage 5 has recorded the
last step of the window.

A (SKU) or (edge) with no event in the window has no entry, which reads as an
estimate of zero (the storage rules use `.get(key, 0.0)`).

A wait is charged to the edge the robot is waiting to enter (sec:method:storage).
Whenever a robot serving a task stands still at u, one wait is counted on the edge
e = (u, w*) that starts its shortest route to its current goal, w* the
out-neighbour w of u with the least c(u, w) + d_G(w, goal), ties broken by the
lower vertex id. The goal is the task's delivery vertex once it is picked up and
its pickup vertex before. A wait counts the same whether the robot chose it or
conflict resolution overrode its move, so a controller that never collides but
waits on purpose, such as Token Passing, still shows where robots queue. No wait
is counted for a robot without a task or one already at its goal. These are
exactly the steps that count towards the task's blocked ticks, and the
environment charges them at stage 5 (WarehouseMAPDEnv._record_wait).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slap_mapd_coupling.core.storage_state import SkuId
from slap_mapd_coupling.storage.base import DemandEstimate, EdgeKey, TrafficEstimate


@dataclass(frozen=True)
class WindowEstimates:
    """What the storage rule reads at an epoch: the three estimates of the closed
    window, and how many orders were released in it (zero means there is no
    evidence to act on)."""

    demand: DemandEstimate  # rho_hat_t(k), orders per step
    traversal: TrafficEstimate  # mu_hat_t(e), traversals per step
    waiting: TrafficEstimate  # w_hat_t(e), waits per step
    num_orders: int  # orders released in the window, summed over SKUs


@dataclass
class WindowEstimator:
    """Counts the events of the current storage window and turns them into
    per-step estimates when the window closes."""

    orders: dict[SkuId, int] = field(default_factory=dict)
    traversals: dict[EdgeKey, int] = field(default_factory=dict)
    waits: dict[EdgeKey, int] = field(default_factory=dict)

    def record_order(self, sku: SkuId) -> None:
        """One order for `sku` was released."""
        self.orders[sku] = self.orders.get(sku, 0) + 1

    def record_traversal(self, edge: EdgeKey) -> None:
        """A robot realised a move along `edge`."""
        self.traversals[edge] = self.traversals.get(edge, 0) + 1

    def record_wait(self, edge: EdgeKey) -> None:
        """A robot serving a task stood still, waiting to enter `edge`."""
        self.waits[edge] = self.waits.get(edge, 0) + 1

    def close_window(self, window_length: int) -> WindowEstimates:
        """The estimates of the window of `window_length` steps that ends now, and a
        fresh window. The counts are divided by the number of steps."""
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}.")
        estimates = WindowEstimates(
            demand={k: n / window_length for k, n in self.orders.items()},
            traversal={e: n / window_length for e, n in self.traversals.items()},
            waiting={e: n / window_length for e, n in self.waits.items()},
            num_orders=sum(self.orders.values()),
        )
        self.orders = {}
        self.traversals = {}
        self.waits = {}
        return estimates
