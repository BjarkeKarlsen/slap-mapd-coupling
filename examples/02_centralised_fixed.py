"""Train the centralised GNN+PPO controller and evaluate it.

A thin variant of 04_decentralised_training.py. The robots choose one at a time in
the episode priority order, each seeing the choices of the earlier ones, so a timestep
takes one turn per robot. The arm sees every movable vertex, so the padding cap is the
number of movable vertices, and its readout pools over all of them (pooling "mean").
Each robot is credited with the mean reward of the fleet, and conflict resolution
never overrides it. Options and output are those of 04_decentralised_training.py, for example
    python examples/02_centralised_fixed.py --budget-env-steps=600
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
example_04 = importlib.import_module("04_decentralised_training")

if __name__ == "__main__":
    example_04.main("centralised")
