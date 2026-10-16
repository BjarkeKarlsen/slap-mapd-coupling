"""Train the section-based GNN+PPO controller and evaluate it.

A thin variant of 04_decentralised_training.py. A robot sees the zone that owns
its vertex together with the zone's halo, so the padding cap is the largest zone
with its halo, and its readout pools over the zone (pooling "mean"). Zones follow
the aisles (zones_by_aisle), and the number of zones, NUM_ZONES in
04_decentralised_training.py, is illustrative because the thesis makes it a sweep
parameter. Inside a zone the robots choose one at a time in priority order, as in the
centralised arm, and the zones run in parallel, so turn k holds the k-th robot of every
zone. Each robot is credited with the mean reward of its zone. Options and output
are those of 04_decentralised_training.py, for example
    python examples/03_section_based.py --budget-env-steps=600
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
example_04 = importlib.import_module("04_decentralised_training")

if __name__ == "__main__":
    example_04.main("section")
