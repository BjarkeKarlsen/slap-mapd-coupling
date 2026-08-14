"""
This is what you want to opimize.

Optimizing this system is the whole point of what we are doing.
"""
import numpy as np
from typing import Optional, Union

from autonomous_agent import autonomous_agent
from constants import NUM_OCEAN_FEATUTES
from area_state import AreaState

class TargetSystem:
    def __init__(self):
        self.timestep: Optional[int] = None


    def initialize_area(self) -> None:
        pass
    
    def get_area(self) -> AreaState:
        self.timestep += 1
        
        return AreaState(
            #agent_position = self.agent.get_position(),
            timestep=self.timestep,
        )
        
    def get_reward(self, agent: autonomous_agent) -> float:
        pass
    
    def step(self, action_dict):
        obs, rew, terminateds, truncateds, info = {}, {}, {}, {}, {}
        
        #******************************************************************************
        # Part 1: get actions from RLlib policies and execute in your environment
        # *****************************************************************************
        
        
        # ******************************************************************************
        # Part 2: get state of your environment after taking actions
        # ******************************************************************************
        
        # ******************************************************************************
        # Part 3: pass relevant information to RLlib
        # ******************************************************************************