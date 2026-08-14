"""
You can define custom spaces for your environment here. For example, if you want to create a custom observation space or action space, you can do so by subclassing gym.spaces.Box, gym.spaces.Discrete, or any other space type provided by OpenAI Gym.
   
"""

import numpy as np
from gymnasium import spaces
from ..core.constants import NUM_AREA_FEATURES, NUM_OBJECTIVES, NUM_DIRECTIONS

#**********************************************************
# observation space
#**********************************************************

object_area_space = spaces.Box(
    low=0, 
    high=1, 
    shape=(NUM_OBJECTIVES + NUM_AREA_FEATURES,), 
    dtype=np.float32)

object_position_space = spaces.Box(
    low=0, 
    high=1, 
    shape=(NUM_OBJECTIVES, 2), 
    dtype=np.float32)


agent_area_space = spaces.Box(
    low=0, 
    high=1,
    shape=(NUM_AREA_FEATURES,),
    dtype=np.float32)

agent_position_space = spaces.Box(
    low=0,
    high=1,
    shape=(2,),
    dtype=np.float32)

high_level_observation_space = spaces.Dict({
    "object_area": object_area_space,
    "object_positions": object_position_space,
    "agent_area": agent_area_space,
    "agent_position": agent_position_space,
})

low_level_observation_space = spaces.Dict({
    "object_positions": object_position_space,
    "agent_position": agent_position_space,
})

#************************************************************
# Action spaces
#************************************************************

# When knowing which objective to go to, the action space is simply the direction to move in.
low_level_action_space = spaces.Discrete(NUM_DIRECTIONS)

# Decide which objective to go to, the action space is simply the index of the objective.
high_level_action_space = spaces.Discrete(NUM_OBJECTIVES)