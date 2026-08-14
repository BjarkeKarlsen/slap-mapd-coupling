"""
This trainer is based off of PPO and points you in the direction of how to
customize it.

PPO is a policy gradient method called Proximal Policy Optimization.
Policy gradient means you update the parameters of the policy directly instead of using
a Q table or something like that.
"""

import copy

from ray.rllib.algorithms.ppo import PPO, PPOConfig

# Note this should not be configured from this file, but rather from the config file. This is just a placeholder to show how to configure the trainer.
config = PPOConfig()
config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
config.framework("torch")
config.rollout_fragment_length = 1000
config.batch_mode = "complete_episodes"
config.simple_optimizer = True
config.num_gpus = 1


class Trainer(PPO):
    @classmethod
    def get_default_config(cls) -> PPOConfig:
        config = super().get_default_config()
        # Customize the default config here if needed
        return copy.deepcopy(config)
