from enum import Enum

class AgentType(str, Enum):
    HIGH = "high"
    LOW = "low"
    
class PolicyID(str, Enum):
    HIGH = "high_level_policy"
    LOW = "low_level_policy"
    
    

#from dataclasses import dataclass
#
#@dataclass(frozen=True)
#class Agent:
#    HIGH: str = "high"
#    LOW: str = "low"
#
#@dataclass(frozen=True)
#class Policy:
#    HIGH: str = "high_level_policy"
#    LOW: str = "low_level_policy"