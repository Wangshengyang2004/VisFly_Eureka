"""
Unit tests for training components.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quadro_llm.training.visfly_training_wrapper import VisFlyTrainingWrapper
from quadro_llm.core.training_manager import TrainingManager


class TestVisFlyTrainingWrapper:
    """Test VisFly training wrapper functionality."""
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return {
            'env': {
                'num_agent_per_scene': 10,
                'max_episode_steps': 100,
                'visual': False,
                'requires_grad': True,
                'tensor_output': True,
                'device': 'cpu'
            },
            'algorithm': {
                'policy': 'MultiInputPolicy',
                'learning_rate': 0.001,
                'horizon': 32,
                'gamma': 0.99
            },
            'learn': {
                'total_timesteps': 1000
            }
        }
    
    @pytest.fixture
    def mock_env(self):
        """Create mock VisFly environment."""
        env = Mock()
        env.num_agent = 10
        env.device = 'cpu'
        env.observation_space = Mock()
        env.action_space = Mock()
        env.reset = Mock(return_value=torch.randn(10, 13))
        env.step = Mock(return_value=(
            torch.randn(10, 13),  # obs
            torch.randn(10),      # reward
            torch.zeros(10, dtype=torch.bool),  # done
            {}  # info
        ))
        env.get_reward = Mock(return_value=torch.randn(10))
        env.get_success = Mock(return_value=torch.zeros(10, dtype=torch.bool))
        return env
    
    def test_wrapper_initialization(self, training_config):
        """Test training wrapper initialization."""
        with patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'):
            wrapper = VisFlyTrainingWrapper(
                environment='navigation',
                algorithm='bptt',
                config=training_config
            )
            
            assert wrapper.environment == 'navigation'
            assert wrapper.algorithm == 'bptt'
            assert wrapper.config == training_config
    
    def test_environment_creation(self, training_config, mock_env):
        """Test environment creation and configuration."""
        with patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module') as mock_import:
            # Mock environment class
            mock_env_class = Mock(return_value=mock_env)
            mock_module = Mock()
            mock_module.NavigationEnv = mock_env_class
            mock_import.return_value = mock_module
            
            wrapper = VisFlyTrainingWrapper('navigation', 'bptt', training_config)
            env = wrapper._create_environment()
            
            assert env is not None
            mock_env_class.assert_called_once()
    
    def test_algorithm_creation(self, training_config, mock_env):
        """Test algorithm creation."""
        with patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'):
            wrapper = VisFlyTrainingWrapper('navigation', 'bptt', training_config)
            wrapper.env = mock_env
            
            with patch('quadro_llm.training.visfly_training_wrapper.BPTT') as mock_bptt:
                algorithm = wrapper._create_algorithm()
                
                mock_bptt.assert_called_once()
                assert algorithm is not None
    
    def test_reward_injection(self, training_config, mock_env):
        """Test reward function injection."""
        reward_code = """
def get_reward(self):
    return -torch.norm(self.position - self.target, dim=1)
        """
        
        with patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'):
            wrapper = VisFlyTrainingWrapper('navigation', 'bptt', training_config)
            wrapper.env = mock_env
            
            with patch('quadro_llm.training.visfly_training_wrapper.RewardInjector') as mock_injector_class:
                mock_injector = Mock()
                mock_injector.inject_reward_function.return_value = True
                mock_injector_class.return_value = mock_injector
                
                success = wrapper.inject_reward_function(reward_code)
                
                assert success
                mock_injector.inject_reward_function.assert_called_once_with(
                    mock_env, reward_code, validate=True
                )
    
    def test_training_execution(self, training_config, mock_env):
        """Test training execution."""
        with patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'):
            wrapper = VisFlyTrainingWrapper('navigation', 'bptt', training_config)
            wrapper.env = mock_env
            
            # Mock algorithm
            mock_algorithm = Mock()
            mock_algorithm.learn.return_value = None
            wrapper.algorithm = mock_algorithm
            
            # Mock evaluator
            with patch('quadro_llm.training.visfly_training_wrapper.Evaluator') as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {
                    'success_rate': 0.8,
                    'episode_length': 95.0
                }
                mock_evaluator_class.return_value = mock_evaluator
                
                result = wrapper.train(total_timesteps=1000)
                
                assert result is not None
                assert 'success_rate' in result
                assert 'episode_length' in result
                assert 'training_time' in result
                
                mock_algorithm.learn.assert_called_once_with(total_timesteps=1000)


class TestTrainingManager:
    """Test training management functionality."""
    
    @pytest.fixture
    def manager_config(self):
        """Create training manager configuration."""
        return {
            'environment': 'navigation',
            'algorithm': 'bptt',
            'training_steps': 1000,
            'evaluation_episodes': 10,
            'success_threshold': 0.8,
            'max_training_time': 300,  # 5 minutes
            'parallel_jobs': 1
        }
    
    @pytest.fixture
    def mock_training_wrapper(self):
        """Create mock training wrapper."""
        wrapper = Mock()
        wrapper.train.return_value = {
            'success_rate': 0.75,
            'episode_length': 120.0,
            'training_time': 180.0
        }
        wrapper.inject_reward_function.return_value = True
        return wrapper
    
    def test_manager_initialization(self, manager_config):
        """Test training manager initialization."""
        with patch('quadro_llm.core.training_manager.ConfigManager'):
            manager = TrainingManager(manager_config)
            
            assert manager.config == manager_config
            assert manager.environment == 'navigation'
            assert manager.algorithm == 'bptt'
    
    def test_baseline_training(self, manager_config, mock_training_wrapper):
        """Test baseline training execution."""
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper', 
                   return_value=mock_training_wrapper):
            
            manager = TrainingManager(manager_config)
            result = manager.train_baseline()
            
            assert result is not None
            assert 'success_rate' in result
            assert 'episode_length' in result
            assert 'training_time' in result
            
            mock_training_wrapper.train.assert_called_once()
    
    def test_reward_function_training(self, manager_config, mock_training_wrapper):
        """Test training with custom reward function."""
        reward_code = "def get_reward(self): return torch.zeros(self.num_agent)"
        
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper',
                   return_value=mock_training_wrapper):
            
            manager = TrainingManager(manager_config)
            result = manager.train_with_reward(reward_code)
            
            assert result is not None
            mock_training_wrapper.inject_reward_function.assert_called_once_with(reward_code)
            mock_training_wrapper.train.assert_called_once()
    
    def test_training_timeout(self, manager_config):
        """Test training timeout handling."""
        # Create wrapper that takes too long
        slow_wrapper = Mock()
        slow_wrapper.inject_reward_function.return_value = True
        
        def slow_train(*args, **kwargs):
            import time
            time.sleep(10)  # Longer than timeout
            return {'success_rate': 0.5, 'episode_length': 100, 'training_time': 10}
        
        slow_wrapper.train = slow_train
        
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper',
                   return_value=slow_wrapper):
            
            # Set short timeout for test
            config = dict(manager_config)
            config['max_training_time'] = 1  # 1 second timeout
            
            manager = TrainingManager(config)
            
            # Should raise timeout exception
            with pytest.raises(TimeoutError):
                manager.train_with_reward("def get_reward(self): return torch.zeros(4)")
    
    def test_training_failure_handling(self, manager_config):
        """Test handling of training failures."""
        failing_wrapper = Mock()
        failing_wrapper.inject_reward_function.return_value = True
        failing_wrapper.train.side_effect = Exception("Training failed")
        
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper',
                   return_value=failing_wrapper):
            
            manager = TrainingManager(manager_config)
            
            # Should re-raise training exception
            with pytest.raises(Exception, match="Training failed"):
                manager.train_with_reward("def get_reward(self): return torch.zeros(4)")
    
    def test_reward_injection_failure(self, manager_config):
        """Test handling of reward injection failures."""
        wrapper_with_bad_injection = Mock()
        wrapper_with_bad_injection.inject_reward_function.return_value = False
        
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper',
                   return_value=wrapper_with_bad_injection):
            
            manager = TrainingManager(manager_config)
            
            # Should raise exception for failed injection
            with pytest.raises(ValueError, match="Failed to inject reward function"):
                manager.train_with_reward("invalid_reward_code")
    
    def test_parallel_training(self, manager_config):
        """Test parallel training execution."""
        # Mock multiple reward functions
        reward_functions = [
            "def get_reward(self): return torch.zeros(4)",
            "def get_reward(self): return -torch.norm(self.velocity, dim=1)"
        ]
        
        mock_wrapper = Mock()
        mock_wrapper.inject_reward_function.return_value = True
        mock_wrapper.train.return_value = {
            'success_rate': 0.8,
            'episode_length': 100.0,
            'training_time': 60.0
        }
        
        with patch('quadro_llm.core.training_manager.ConfigManager'), \
             patch('quadro_llm.core.training_manager.VisFlyTrainingWrapper',
                   return_value=mock_wrapper), \
             patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            
            # Mock executor
            mock_future = Mock()
            mock_future.result.return_value = {
                'success_rate': 0.8,
                'episode_length': 100.0,
                'training_time': 60.0
            }
            
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
            
            manager = TrainingManager(manager_config)
            results = manager.train_parallel(reward_functions)
            
            assert len(results) == 2
            assert all('success_rate' in r for r in results)


class TestTrainingUtilities:
    """Test training utility functions."""
    
    def test_performance_metrics_calculation(self):
        """Test calculation of training performance metrics."""
        from quadro_llm.utils.training_utils import calculate_performance_metrics
        
        # Mock training results
        training_logs = {
            'rewards': [10.0, 15.0, 20.0, 18.0, 25.0],
            'episode_lengths': [100, 90, 85, 88, 80],
            'success_episodes': [0, 1, 1, 1, 1]
        }
        
        metrics = calculate_performance_metrics(training_logs)
        
        assert 'success_rate' in metrics
        assert 'avg_episode_length' in metrics
        assert 'avg_reward' in metrics
        
        assert metrics['success_rate'] == 0.8  # 4/5
        assert metrics['avg_episode_length'] == 88.6  # Mean of lengths
        assert metrics['avg_reward'] == 17.6  # Mean of rewards
    
    def test_early_stopping_detection(self):
        """Test early stopping detection logic."""
        from quadro_llm.utils.training_utils import should_stop_early
        
        # Converged performance (no improvement)
        converged_history = [0.5, 0.52, 0.51, 0.52, 0.51, 0.52]
        assert should_stop_early(converged_history, patience=3, min_delta=0.05)
        
        # Still improving
        improving_history = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert not should_stop_early(improving_history, patience=3, min_delta=0.05)
        
        # Not enough history
        short_history = [0.5, 0.6]
        assert not should_stop_early(short_history, patience=3, min_delta=0.05)
    
    def test_training_stability_check(self):
        """Test training stability validation."""
        from quadro_llm.utils.training_utils import check_training_stability
        
        # Stable training
        stable_rewards = [10.0, 12.0, 11.0, 13.0, 12.5, 14.0]
        assert check_training_stability(stable_rewards)
        
        # Unstable training (high variance)
        unstable_rewards = [10.0, 50.0, 2.0, 45.0, 1.0, 48.0]
        assert not check_training_stability(unstable_rewards)
        
        # NaN or infinite values
        invalid_rewards = [10.0, float('nan'), 12.0]
        assert not check_training_stability(invalid_rewards)
        
        invalid_rewards_inf = [10.0, float('inf'), 12.0]
        assert not check_training_stability(invalid_rewards_inf)