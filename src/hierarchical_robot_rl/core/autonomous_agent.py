from typing import Optional
import numpy as np

from constants import NUM_OCEAN_FEATUTES

class AutonomousAgent:
    def __init__(self):
        self.position: Optional[np.ndarray] = None
        self.timestep: Optional[int] = None
        # self.ocean: Optional[np.ndarray] = None
        
    def initialize(self) -> None:
        """
        Agent should preorm something to initialize itself, e.g. set its position at to a theseable random in the environment.
        """

    def get_position(self) -> np.ndarray:
        """ Return the current position of the agent in the environment. """
        return self.position
    
    def get_ocean(self) -> np.ndarray:
        """
        Return the current state of the ocean as a numpy array.
        The shape of the array should be (NUM_OCEAN_FEATURES,).
        """
        pass
    
    def set_choice(self, action: np.ndarray) -> None: # Guide note this as a integer or a float, depending on your action space
        """
        Set the agent's choice based on the action provided.
        """
        # Objective: Update the agent's internal state based on the action taken. This could involve moving the agent, changing its state, or any other relevant operation.
        # So for a chicken this would be passed in a chicken which will be saved as the agent's choice in it self initialized variables. 
        pass
    
    def move(self, action: np.ndarray) -> None: # Note this as a integer or a float, depending on your action space
        """
        Move the agent in the environment based on the action provided.
        """
        pass
        #else:
        #    raise ValueError("Action must be a numpy array of shape (2,) representing the movement in x and y directions.")