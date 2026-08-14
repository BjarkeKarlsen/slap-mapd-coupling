from ray.rllib.algorithms.ppo import PPOTorchPolicy

from ..models.base_model import BaseModel

class HighLevelPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.model = BaseModel(
            observation_space, 
            action_space, 
            model_config=config,
            name="BaseModel")


class LowLevelPolicy(PPOTorchPolicy):
    pass