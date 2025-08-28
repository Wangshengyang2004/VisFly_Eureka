"""
Unit tests for reward injection.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from quadro_llm.core.reward_injector import RewardInjector


class TestRewardInjector:
    """Test reward injection functionality."""
    
    @pytest.fixture
    def injector(self):
        """Create reward injector instance."""
        return RewardInjector()
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = Mock()
        env.num_agent = 4
        env.device = 'cpu'
        env.position = torch.randn(4, 3)
        env.velocity = torch.randn(4, 3)
        env.target = torch.randn(4, 3)
        env.collision_dis = torch.ones(4)
        env.reset = Mock(return_value=None)
        env.get_reward = Mock(return_value=torch.zeros(4))
        return env
    
    def test_inject_valid_reward(self, injector, mock_env):
        """Test injection of valid reward function."""
        reward_code = """
def get_reward(self):
    distance = torch.norm(self.position - self.target, dim=1)
    return -distance
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=False
        )
        
        assert success
        # Check that get_reward was replaced
        assert hasattr(mock_env, 'get_reward')
        
        # Test the injected function works
        reward = mock_env.get_reward()
        assert isinstance(reward, torch.Tensor)
    
    def test_inject_invalid_syntax(self, injector, mock_env):
        """Test injection fails for invalid syntax."""
        reward_code = """
def get_reward(self)  # Missing colon
    return torch.zeros(4)
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=False
        )
        
        assert not success
    
    def test_inject_missing_function(self, injector, mock_env):
        """Test injection fails when get_reward is missing."""
        reward_code = """
def other_function(self):
    return torch.zeros(4)
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=False
        )
        
        assert not success
    
    def test_validation_correct_shape(self, injector, mock_env):
        """Test validation passes for correct reward shape."""
        reward_code = """
def get_reward(self):
    return torch.zeros(self.num_agent)
        """
        
        # Mock the reset and step methods
        mock_env.reset.return_value = None
        mock_env.action_space = Mock()
        mock_env.action_space.sample.return_value = torch.randn(4, 4)
        mock_env.step = Mock(return_value=(None, None, False, {}))
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=True
        )
        
        assert success
    
    def test_validation_wrong_shape(self, injector, mock_env):
        """Test validation fails for wrong reward shape."""
        reward_code = """
def get_reward(self):
    return torch.zeros(10)  # Wrong shape
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=True
        )
        
        assert not success
    
    def test_validation_nan_values(self, injector, mock_env):
        """Test validation fails for NaN rewards."""
        reward_code = """
def get_reward(self):
    return torch.full((self.num_agent,), float('nan'))
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=True
        )
        
        assert not success
    
    def test_validation_inf_values(self, injector, mock_env):
        """Test validation fails for infinite rewards."""
        reward_code = """
def get_reward(self):
    return torch.full((self.num_agent,), float('inf'))
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=True
        )
        
        assert not success
    
    def test_safe_wrapper_creation(self, injector, mock_env):
        """Test creation of safe wrapper."""
        reward_code = """
def get_reward(self):
    return -torch.norm(self.position - self.target, dim=1)
        """
        
        wrapped = injector.create_safe_wrapper(mock_env, reward_code)
        
        assert wrapped is not None
        # Test wrapper delegates attributes
        assert wrapped.num_agent == mock_env.num_agent
        assert wrapped.device == mock_env.device
        
        # Test safe reward execution
        reward = wrapped.get_reward()
        assert isinstance(reward, torch.Tensor)
    
    def test_safe_wrapper_error_handling(self, injector, mock_env):
        """Test safe wrapper handles errors gracefully."""
        reward_code = """
def get_reward(self):
    # This will cause an error
    return 1 / 0
        """
        
        wrapped = injector.create_safe_wrapper(mock_env, reward_code)
        
        if wrapped:
            # Should return zero rewards on error
            reward = wrapped.get_reward()
            assert isinstance(reward, torch.Tensor)
            assert torch.allclose(reward, torch.zeros(4))
    
    def test_restricted_builtins(self, injector, mock_env):
        """Test that dangerous operations are restricted."""
        reward_code = """
def get_reward(self):
    # Try to access file system (should fail)
    import os
    os.listdir('/')
    return torch.zeros(self.num_agent)
        """
        
        success = injector.inject_reward_function(
            mock_env, reward_code, validate=False
        )
        
        # Should fail due to restricted builtins
        assert not success or mock_env.get_reward() is None