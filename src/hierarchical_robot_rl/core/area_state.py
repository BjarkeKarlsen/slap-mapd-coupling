from dataclasses import dataclass
import numpy as np

@dataclass
class AreaState:
    agent_position: np.ndarray
    timestep: int