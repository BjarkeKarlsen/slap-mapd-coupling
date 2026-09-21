"""Shared relocation cap R(x, x_{t-Delta}) (eq:storageobjective, sec:method:storage).

R(x, x_{t-Delta}) = 0 if the number of relocated units is <= nu, else
infinity: any finite chi > 0 in eq:storageobjective then enforces the cap
outright, so this is realised directly as a hard cap on the greedy target
computed by F_dem/F_cng, not as a penalty term those rules have to weigh.
Shared here because both rules need the identical cap-and-tie-break logic
(sec:method:storage says so explicitly) -- neither should reimplement it.

Flagged, not literally specified: the thesis says "only the nu
highest-marginal-value relocations are applied... ties broken by SKU id
then vertex id," but never defines what one unit's marginal value is, or
which specific units get left behind once the cap binds. Resolved in
discussion: each (SKU, vertex) cell that gains units relative to
x_{t-Delta} is one "relocation," ranked by whatever score the calling
rule's own eq:storagegreedy ranking already produces (that ranking IS the
marginal-value order -- no separate scoring mechanism is invented here).
Cells are accepted in that order until the nu-unit budget is exhausted;
whichever cells don't fit stay at their x_{t-Delta} count instead of the
greedy target's, and the units that would have supplied an accepted
arrival are drawn only from cells the greedy target itself wants
reduced (true surplus, per SKU) -- so per-SKU unit totals are always
conserved exactly, never invented or lost, regardless of where the cap
binds.
"""

from __future__ import annotations

from typing import Sequence

from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, StorageState


def apply_relocation_cap(
    x_prev: StorageState,
    x_target: StorageState,
    priority: Sequence[tuple[SkuId, VertexId]],
    reassignment_cap: int,
) -> StorageState:
    """Cap x_target's relocations relative to x_prev at `reassignment_cap`
    (nu) units, honouring `priority`'s order.

    `x_target` is the calling rule's uncapped greedy result (already
    feasible, eq:feasiblestorage). `priority` is every (sku, vertex) cell
    where x_target > x_prev (an arrival), in strictly descending
    marginal-value order with ties already broken (SKU id then vertex
    id) -- that ordering is the calling rule's responsibility, since only
    it knows its own eq:storagegreedy score.
    """
    counts: dict[SkuId, dict[VertexId, int]] = {k: dict(v) for k, v in x_prev.counts.items()}
    budget = reassignment_cap

    for sku_id, vertex in priority:
        if budget <= 0:
            break
        current_here = counts.get(sku_id, {}).get(vertex, 0)
        wanted = x_target.units(sku_id, vertex)
        deficit = wanted - current_here
        if deficit <= 0:
            continue  # not actually an arrival relative to the running state

        take = min(deficit, budget)
        withdrawn = 0
        sku_counts = counts.setdefault(sku_id, {})
        for source_vertex in sorted(x_prev.counts.get(sku_id, {})):
            if withdrawn >= take:
                break
            if source_vertex == vertex:
                continue
            surplus = sku_counts.get(source_vertex, 0) - x_target.units(sku_id, source_vertex)
            if surplus <= 0:
                continue  # x_target doesn't actually want this cell reduced
            grab = min(surplus, take - withdrawn)
            sku_counts[source_vertex] = sku_counts.get(source_vertex, 0) - grab
            withdrawn += grab

        if withdrawn == 0:
            continue  # nothing available to relocate here yet; leave this cell as-is

        sku_counts[vertex] = current_here + withdrawn
        budget -= withdrawn

    return StorageState(skus=x_prev.skus, capacities=x_prev.capacities, counts=counts)
