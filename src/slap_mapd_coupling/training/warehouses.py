"""A new warehouse every episode (step 12 of the model revision, D27).

With `warehouse_per_seed` the episode seed draws the warehouse, its aisles, links and
workstations, and the start stock, so training sees one warehouse per training seed and the
validation and test seeds play warehouses no run trained on. Fixed and demand storage still play
the same warehouse on the same seed.

The network reads compact samples, which name a robot's vertex by its row in the window tables
(models/window_tables.py). One run's tables for every warehouse are too large to hold in every
process (about 9 MB a warehouse for the decentralised arm at 64 agents), so `WarehousePool`
stores each warehouse's tables on disk, under a folder keyed by everything the tables depend
on. A sample's vertex key packs the warehouse seed and the row (`vertex_key`), and the module
loads the tables of the warehouses a batch holds (`StoredWindowLookup`). Each warehouse is built
with its own padding cap and padded at load to the largest cap of the pool, which `prepare`
writes to the manifest before any training starts.

The warehouse itself, the graph, the storage and the zones, is rebuilt from the spec and the
seed wherever it is needed, since a graph with its distances is too large to store for every
seed (about 22 MB at 32 agents). Building one takes 0.1 s at 8 agents and about 20 s at 64.

The environment variable SLAP_WAREHOUSE_STORE, when set, is the store every pool uses, whatever
the spec or a checkpoint names. So a run trained on one machine is evaluated on another, or in
the cloud, by pointing it at that machine's copy of the store. The tables follow from the code,
the spec and the seed alone, so a store can be built where it is used (`prepare`) or copied.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from slap_mapd_coupling.core.graph import VertexId
from slap_mapd_coupling.core.storage_state import SkuId, SkuType
from slap_mapd_coupling.instances.generator import GeneratedInstance
from slap_mapd_coupling.models.local_subgraph_encoding import LocalSubgraphEncodingConfig
from slap_mapd_coupling.models.window_tables import WindowTables, pad_tables

# Where the pools live unless a spec names another place, and the variable that overrides both.
DEFAULT_STORE = Path.home() / "thesis_runs" / "warehouses"
STORE_VARIABLE = "SLAP_WAREHOUSE_STORE"
# The warehouses one process keeps built, the one it plays and the one before.
_KEPT = 2


@dataclass(frozen=True)
class Warehouse:
    """One episode seed's warehouse: the instance, the start stock and, for the section arm,
    the zones. `encoding` is the padding cap of this warehouse alone."""

    seed: int
    instance: GeneratedInstance
    skus: dict[SkuId, SkuType]
    capacities: dict[VertexId, float]
    counts: dict[SkuId, dict[VertexId, int]]
    zones: Optional[Sequence[frozenset[VertexId]]]
    encoding: LocalSubgraphEncodingConfig


class WarehousePool:
    """The warehouses of one spec, one per seed, and their window tables on disk. Picklable,
    so the env runners and the module can each hold one. Only the spec and the folder travel,
    the built warehouses stay in the process that built them."""

    def __init__(self, spec: Any, root: Path | None = None) -> None:
        self.spec = spec
        store = os.environ.get(STORE_VARIABLE) or root or DEFAULT_STORE
        self.root = Path(store).expanduser() / pool_key(spec)
        self._built: OrderedDict[int, Warehouse] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        return {"spec": self.spec, "root": self.root}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.spec = state["spec"]
        self.root = state["root"]
        if os.environ.get(STORE_VARIABLE):
            self.root = Path(os.environ[STORE_VARIABLE]).expanduser() / self.root.name
        self._built = OrderedDict()

    def as_config(self) -> dict[str, Any]:
        """The pool as plain data, for the module's model config, so a checkpoint names it."""
        return {"spec": self.spec.model_dump(mode="json"), "root": str(self.root.parent)}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "WarehousePool":
        from slap_mapd_coupling.training.spec import TrainingSpec

        return cls(TrainingSpec.model_validate(config["spec"]), Path(config["root"]))

    def warehouse(self, seed: int) -> Warehouse:
        """The warehouse of `seed`, built here or kept from before."""
        if seed in self._built:
            self._built.move_to_end(seed)
            return self._built[seed]
        from slap_mapd_coupling.training.spec import build_seed_warehouse

        built = build_seed_warehouse(self.spec, seed)
        self._built[seed] = built
        while len(self._built) > _KEPT:
            self._built.popitem(last=False)
        return built

    def tables_path(self, seed: int) -> Path:
        return self.root / f"tables_{seed}.npz"

    def tables(self, seed: int, padded: bool = True) -> WindowTables:
        """The window tables of `seed`, built and stored first if they are not on disk yet.
        `padded` pads them to the pool's cap, which needs the manifest."""
        path = self.tables_path(seed)
        if not path.exists():
            self._store(seed)
        tables = load_tables(path)
        if not padded:
            return tables
        manifest = self.manifest()
        return pad_tables(tables, manifest["max_local_nodes"], manifest["d_max"])

    def _store(self, seed: int) -> int:
        """Builds the tables of `seed` at its own cap, writes them and returns the cap."""
        from slap_mapd_coupling.training.env import arm_window_tables

        built = self.warehouse(seed)
        experiment = self.spec.experiment
        tables = arm_window_tables(
            built.instance.graph,
            self.spec.controller,
            experiment.observation_depth,
            built.encoding,
            built.zones,
            _pooling(self.spec),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        # Written aside and renamed, so a process reading the pool never sees half a file.
        partial = self.root / f".tables_{seed}.{os.getpid()}.npz"
        np.savez(partial, **_flat(tables))
        partial.replace(self.tables_path(seed))
        return built.encoding.max_local_nodes

    def prepare(self, seeds: Iterable[int], workers: int = 1) -> dict[str, Any]:
        """Builds and stores the tables of every seed not stored yet, in `workers` processes at
        once, then writes the manifest: the largest cap and d_max over the seeds. Training and
        evaluation read the cap from it, so every seed a run will play must be prepared before
        it starts."""
        seeds = sorted(set(seeds))
        caps: dict[str, int] = {}
        caps_path = self.root / "caps.json"
        if caps_path.exists():
            caps = json.loads(caps_path.read_text(encoding="utf-8"))
        missing = [s for s in seeds if not (str(s) in caps and self.tables_path(s).exists())]
        if workers <= 1 or len(missing) <= 1:
            for seed in missing:
                caps[str(seed)] = self._store(seed)
                _write_json(caps_path, caps)
        else:
            # Spawned, not forked, so no worker inherits the parent's threads.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(min(workers, len(missing)), mp_context=context) as pool:
                for seed, cap in zip(missing, pool.map(_store_one, [(self, s) for s in missing])):
                    caps[str(seed)] = cap
                    _write_json(caps_path, caps)
        d_max = {int(load_tables(self.tables_path(seed)).move_target_index.shape[1]) for seed in seeds}
        if len(d_max) != 1:
            raise ValueError(f"the warehouses of the pool differ in d_max: {sorted(d_max)}.")
        manifest = {
            "max_local_nodes": max(caps[str(seed)] for seed in seeds),
            "d_max": d_max.pop(),
            "seeds": seeds,
        }
        _write_json(self.root / "manifest.json", manifest)
        return manifest

    def manifest(self) -> dict[str, Any]:
        path = self.root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Prepare the pool's seeds first (WarehousePool.prepare)."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def encoding(self) -> LocalSubgraphEncodingConfig:
        """The padding cap every warehouse of the pool is padded to."""
        built_cap = self.manifest()["max_local_nodes"]
        max_messages = self.spec.arm.max_messages or max(self.spec.experiment.num_agents - 1, 1)
        return LocalSubgraphEncodingConfig(max_local_nodes=built_cap, max_messages=max_messages)


def _store_one(job: tuple["WarehousePool", int]) -> int:
    """`WarehousePool._store` in a worker process."""
    pool, seed = job
    return pool._store(seed)


def pool_key(spec: Any) -> str:
    """A short hash of everything the window tables depend on: the warehouse settings without
    the seed, the arm and the observation depth."""
    settings = dict(spec.instance_settings())
    settings.pop("seed", None)
    content = {
        "instance": settings,
        "controller": spec.controller,
        "observation_depth": spec.experiment.observation_depth,
        "num_zones": spec.arm.num_zones,
        "pooling": _pooling(spec),
    }
    text = json.dumps(content, sort_keys=True, default=str)
    return f"fleet{spec.experiment.num_agents}_{spec.controller}_" + hashlib.sha256(
        text.encode()
    ).hexdigest()[:12]


def _pooling(spec: Any) -> str:
    from slap_mapd_coupling.training.arm import POOLING

    return POOLING[spec.controller]


def _flat(tables: WindowTables) -> dict[str, np.ndarray]:
    arrays = tables.as_arrays()
    arrays["vertices"] = np.asarray(arrays["vertices"], dtype=np.int64)
    return arrays


def load_tables(path: Path) -> WindowTables:
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays["vertices"] = [int(v) for v in arrays["vertices"]]
    return WindowTables.from_arrays(arrays)


def _write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}")
    partial.write_text(json.dumps(content), encoding="utf-8")
    partial.replace(path)
