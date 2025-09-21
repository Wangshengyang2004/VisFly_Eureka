"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
from pathlib import Path
import torch
from unittest.mock import Mock

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'VisFly'))


@pytest.fixture
def device():
    """Provide appropriate device for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def mock_visfly_env():
    """Create a mock VisFly environment for testing."""
    env = Mock()
    
    # Basic properties
    env.num_agent = 4
    env.num_envs = 4
    env.device = 'cpu'
    env.max_episode_steps = 256
    env._step_count = 0
    
    # State tensors
    env.position = torch.randn(4, 3)
    env.velocity = torch.randn(4, 3)
    env.orientation = torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.float32)
    env.angular_velocity = torch.randn(4, 3)
    env.target = torch.tensor([[10, 0, 1.5]] * 4, dtype=torch.float32)
    env.collision_dis = torch.ones(4) * 5.0
    
    # Sensor observations
    env.sensor_obs = {
        'depth': torch.rand(4, 1, 64, 64),
        'IMU': torch.randn(4, 13)
    }
    
    # Methods
    env.reset = Mock(return_value=env.sensor_obs)
    env.step = Mock(return_value=(env.sensor_obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {}))
    env.get_reward = Mock(return_value=torch.zeros(4))
    env.get_success = Mock(return_value=torch.zeros(4, dtype=torch.bool))
    
    # Action space
    env.action_space = Mock()
    env.action_space.sample = Mock(return_value=torch.randn(4, 4))
    
    return env


@pytest.fixture
def sample_reward_code():
    """Provide sample reward function code."""
    return """
def get_reward(self):
    # Distance to target
    distance = torch.norm(self.position - self.target, dim=1)
    distance_reward = -distance * 0.1
    
    # Velocity penalty (BPTT safe)
    velocity_penalty = -torch.norm(self.velocity - 0, dim=1) * 0.01
    
    # Stability penalty (BPTT safe)
    angular_penalty = -torch.norm(self.angular_velocity - 0, dim=1) * 0.001
    
    # Collision avoidance
    collision_penalty = -1.0 / (self.collision_dis + 0.5)
    
    # Total reward
    reward = distance_reward + velocity_penalty + angular_penalty + collision_penalty
    
    return reward
"""


@pytest.fixture
def llm_config():
    """Provide test LLM configuration."""
    return {
        'api_key': 'test-api-key',
        'model': 'gpt-4o',
        'temperature': 0.8,
        'max_tokens': 2000,
        'timeout': 30,
        'max_retries': 2
    }


@pytest.fixture
def optimization_config():
    """Provide test optimization configuration."""
    return {
        'iterations': 2,
        'samples': 4,
        'training_steps': 1000,
        'algorithm': 'bptt',
        'evaluation_episodes': 10,
        'success_threshold': 0.8
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "llm: Tests requiring LLM API")


def pytest_addoption(parser):
    """Custom CLI flags to opt into slow/gpu/llm tests."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-gpu", action="store_true", default=False, help="run GPU tests")
    parser.addoption("--run-llm", action="store_true", default=False, help="run LLM API tests")


def pytest_collection_modifyitems(config, items):
    """Skip certain categories unless explicitly enabled."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
    skip_llm = pytest.mark.skip(reason="need --run-llm option to run")

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if 'slow' in markers and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if 'gpu' in markers and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
        if 'llm' in markers and not config.getoption("--run-llm"):
            item.add_marker(skip_llm)
