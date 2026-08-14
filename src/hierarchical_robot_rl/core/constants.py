# Constants for the hierarchical robot reinforcement learning environment

# Features of the area state
#NUM_OCEAN_FEATURES = 5  # Number of features representing the ocean state

# Postitions 
X=0
Y=1

# Amount of objectives in the environment
NUM_OBJECTIVES = 10 # Example: chickens, fish, etc. Adjust based on your environment's complexity.

# Amount of features representing the area state
NUM_AREA_FEATURES = 2  # Example: x and y coordinates

# 
CLOSESNESS_THRESHOLD = 0.1  # Example threshold for determining if the agent is close to an objective. Adjust based on your environment's scale.
STEP_SIZE = 0.05  # Example step size for agent movement. Adjust based on your environment's scale.


# Rewards and penalties
STEP_REWARD = -.001
ARRIVAL_REWARD = 1.0
TIMEOUT_REWARD = -1.0

# Directions
NUM_DIRECTIONS = 4
N=1
S=2
E=3
W=4