from typing import Dict

from ..policies import LowLevelPolicy, HighLevelPolicy
from ..policies.policy_types import AgentType, PolicyID

def policy_mapping_fn(agent_id: str, episode, **kwargs) -> str:
    """
    Maps agent IDs to policy IDs based on naming conventions.

    Expected agent_id formats:
        - Contains HIGH → mapped to HIGH_LEVEL_POLICY
        - Contains LOW  → mapped to LOW_LEVEL_POLICY

    Raises:
        ValueError: If agent_id does not contain either HIGH or LOW.
    """
    match agent_id:
        case _ if AgentType.HIGH in agent_id:
            return PolicyID.HIGH
        case _ if AgentType.LOW in agent_id:
            return PolicyID.LOW     
        case _:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        
        