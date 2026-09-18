"""Storage state x_t: SKU x vertex unit counts and the feasible-configuration set X."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, model_validator

from slap_mapd_coupling.core.graph import VertexId

SkuId = str


class SkuType(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sku_id: SkuId
    unit_capacity: PositiveFloat  # b_k > 0


class StorageState(BaseModel):
    """x_t : K x V_str -> N_0, one immutable snapshot per storage epoch.

    Stored as nested dicts (sku -> vertex -> count) rather than a NumPy
    matrix: K and V_str are small, the natural query is a dict lookup,
    and a matrix would need a separate index<->id map kept in sync. x_t
    updates once per Delta timesteps, not once per timestep, so this is
    not expected to be a hot path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skus: dict[SkuId, SkuType]
    capacities: dict[VertexId, PositiveFloat]  # cap(v) > 0, keys = V_str
    counts: dict[SkuId, dict[VertexId, NonNegativeInt]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _counts_reference_known_sku_and_vertex(self) -> "StorageState":
        for sku_id, per_vertex in self.counts.items():
            if sku_id not in self.skus:
                raise ValueError(f"counts references unknown sku {sku_id!r}.")
            for v in per_vertex:
                if v not in self.capacities:
                    raise ValueError(f"counts references vertex {v} with no declared cap(v).")
        return self

    @model_validator(mode="after")
    def _feasible(self) -> "StorageState":
        """eq:feasiblestorage: sum_k b_k * x(k,v) <= cap(v) for every v."""
        used: dict[VertexId, float] = dict.fromkeys(self.capacities, 0.0)
        for sku_id, per_vertex in self.counts.items():
            b_k = self.skus[sku_id].unit_capacity
            for v, count in per_vertex.items():
                used[v] += b_k * count
        for v, total in used.items():
            if total > self.capacities[v]:
                raise ValueError(
                    f"Storage vertex {v} over capacity: used={total} > cap={self.capacities[v]} "
                    "(eq:feasiblestorage)."
                )
        return self

    def units(self, sku_id: SkuId, vertex: VertexId) -> int:
        """x_t(k, v)."""
        return self.counts.get(sku_id, {}).get(vertex, 0)

    def used_capacity(self, vertex: VertexId) -> float:
        """sum_k b_k * x(k, v) for one v (eq:feasiblestorage's column-sum reading)."""
        return sum(
            self.skus[sku_id].unit_capacity * per_vertex.get(vertex, 0)
            for sku_id, per_vertex in self.counts.items()
        )

    def with_units(self, sku_id: SkuId, vertex: VertexId, count: int) -> "StorageState":
        """Return a new StorageState with x(k,v) set to `count`.

        Re-validates feasibility on construction: an infeasible relocation
        raises immediately instead of silently landing in x_t. Deliberately
        NOT `self.model_copy(update=...)` -- Pydantic v2's model_copy skips
        validators entirely, so it would let an infeasible update through
        unchecked; constructing a fresh instance re-runs them.
        """
        new_counts = {k: dict(v) for k, v in self.counts.items()}
        new_counts.setdefault(sku_id, {})[vertex] = count
        return StorageState(skus=self.skus, capacities=self.capacities, counts=new_counts)


def is_feasible(state: StorageState) -> bool:
    """Free-function predicate mirroring `_feasible`.

    Lets callers (e.g. storage rules) test-fit a candidate configuration
    without relying on catching a ValidationError as control flow.
    """
    try:
        StorageState.model_validate(state.model_dump())
        return True
    except Exception:
        return False
