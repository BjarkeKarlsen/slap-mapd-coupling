"""Checkpoint save/load helpers (issue #30) -- thin wrappers around
RLlib's own Algorithm.save_to_path/Algorithm.from_checkpoint, so a
training script and evaluation code use one shared entry point rather
than each re-deriving the checkpoint path convention.
"""

from __future__ import annotations

from pathlib import Path

from ray.rllib.algorithms.algorithm import Algorithm


def save_checkpoint(algo: Algorithm, checkpoint_dir: Path) -> Path:
    """Saves `algo`'s full state (RLModule weights + training progress)
    to `checkpoint_dir`, creating it if needed."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_path = algo.save_to_path(checkpoint_dir)
    return Path(saved_path)


def load_checkpoint(checkpoint_dir: Path) -> Algorithm:
    """Restores a full Algorithm (config + RLModule weights + training
    progress) from a directory save_checkpoint produced."""
    algo = Algorithm.from_checkpoint(str(checkpoint_dir))
    assert isinstance(algo, Algorithm)  # from_checkpoint's Checkpointable return is untyped
    return algo
