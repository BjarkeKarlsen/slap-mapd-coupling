import pytest
from hierarchical_robot_rl.environment.multi_agent_env import YourEnvironment
from hierarchical_robot_rl.core.target_system import YourTargetSystem

@pytest.fixture
def env():
    """Fixture: Create test environment"""
    return YourEnvironment({'is_use_visualization': False})

@pytest.fixture
def target_system():
    """Fixture: Create test target system"""
    return YourTargetSystem()

@pytest.fixture
def sample_config():
    """Fixture: Load sample config"""
    return {
        'num_workers': 0,
        'num_gpus': 0,
        'use_gae': False,
        'use_critic': False,
    }