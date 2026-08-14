
from typing import Dict

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env import EnvContext

class Environment(MultiAgentEnv):
    def __init__(self, config: EnvContext):
        pass

    def initialize_thing(self):
        pass
        
    def reset(self, *, seeed=None, options=None):
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
        
    def close(self):
        pass
    
    def render(self, mode='human'):
        # Render the system
        
        #self.visualization.render(self.target_system) 
        pass