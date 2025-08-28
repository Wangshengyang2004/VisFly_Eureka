"""
Integration tests for the full quadro-llm workflow.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quadro_llm.pipeline import Pipeline
from quadro_llm.core.models import RewardFunctionResult, IterationSummary


@pytest.mark.integration
class TestFullWorkflow:
    """Test complete optimization workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create config structure
            config_dir = workspace / 'configs'
            env_dir = config_dir / 'envs'
            alg_dir = config_dir / 'algs' / 'test_env'
            
            env_dir.mkdir(parents=True)
            alg_dir.mkdir(parents=True)
            
            # Create minimal environment config
            env_config = {
                'env': {
                    'num_agent_per_scene': 4,
                    'max_episode_steps': 50,
                    'visual': False,
                    'requires_grad': True,
                    'tensor_output': True,
                    'device': 'cpu',
                    'dynamics_kwargs': {'dt': 0.03}
                },
                'eval_env': {
                    'num_agent_per_scene': 1,
                    'visual': False
                }
            }
            
            with open(env_dir / 'test_env.yaml', 'w') as f:
                yaml.dump(env_config, f)
            
            # Create minimal algorithm config
            alg_config = {
                'algorithm': {
                    'policy': 'MultiInputPolicy',
                    'learning_rate': 0.01,
                    'horizon': 16,
                    'gamma': 0.99
                },
                'env_overrides': {
                    'requires_grad': True,
                    'tensor_output': False,
                    'device': 'cpu'
                },
                'learn': {
                    'total_timesteps': 100  # Very small for testing
                }
            }
            
            with open(alg_dir / 'bptt.yaml', 'w') as f:
                yaml.dump(alg_config, f)
            
            yield workspace
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration testing."""
        return {
            'environment': 'test_env',
            'algorithm': 'bptt',
            'iterations': 2,
            'samples': 2,  # Small for testing
            'training_steps': 100,
            'evaluation_episodes': 5,
            'success_threshold': 0.6,
            'llm': {
                'api_key': 'test-key',
                'model': 'gpt-4o',
                'temperature': 0.8,
                'max_tokens': 1000
            },
            'task_description': 'Test navigation task'
        }
    
    @patch('quadro_llm.core.llm_interface.openai.OpenAI')
    @patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module')
    def test_end_to_end_optimization(self, mock_import, mock_openai, 
                                   temp_workspace, integration_config):
        """Test complete end-to-end optimization workflow."""
        
        # Mock LLM responses
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock reward function generation
        mock_response = Mock()
        mock_choice1 = Mock()
        mock_choice1.message.content = """
def get_reward(self):
    distance = torch.norm(self.position - self.target, dim=1)
    return -distance * 0.1
        """
        
        mock_choice2 = Mock()
        mock_choice2.message.content = """
def get_reward(self):
    distance = -torch.norm(self.position - self.target, dim=1)
    velocity_penalty = -torch.norm(self.velocity, dim=1) * 0.01
    return distance + velocity_penalty
        """
        
        mock_response.choices = [mock_choice1, mock_choice2]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock environment and training
        mock_env = Mock()
        mock_env.num_agent = 4
        mock_env.device = 'cpu'
        mock_env.position = torch.randn(4, 3)
        mock_env.target = torch.randn(4, 3)
        mock_env.velocity = torch.randn(4, 3)
        
        mock_env_class = Mock(return_value=mock_env)
        mock_module = Mock()
        mock_module.TestEnv = mock_env_class  # Match environment name
        mock_import.return_value = mock_module
        
        # Mock training results with varying performance
        with patch('quadro_llm.training.visfly_training_wrapper.BPTT') as mock_bptt, \
             patch('quadro_llm.training.visfly_training_wrapper.Evaluator') as mock_evaluator_class:
            
            mock_algorithm = Mock()
            mock_bptt.return_value = mock_algorithm
            
            mock_evaluator = Mock()
            # Progressive improvement across iterations
            mock_evaluator.evaluate.side_effect = [
                # Baseline
                {'success_rate': 0.2, 'episode_length': 45.0},
                # Iteration 1 samples
                {'success_rate': 0.4, 'episode_length': 40.0},
                {'success_rate': 0.5, 'episode_length': 38.0},
                # Iteration 2 samples
                {'success_rate': 0.6, 'episode_length': 35.0},
                {'success_rate': 0.7, 'episode_length': 32.0}
            ]
            mock_evaluator_class.return_value = mock_evaluator
            
            # Set config directory in integration config
            integration_config['config_dir'] = str(temp_workspace / 'configs')
            
            # Run optimization
            output_dir = temp_workspace / 'output'
            pipeline = Pipeline(
                config=integration_config,
                output_dir=str(output_dir)
            )
            
            report = pipeline.run_optimization()
        
        # Verify optimization report
        assert report is not None
        assert report.total_samples == 4  # 2 iterations * 2 samples
        assert report.successful_samples >= 0
        assert len(report.iteration_history) == 2
        
        # Check improvement over baseline
        assert report.best_performance['success_rate'] > 0.2  # Better than baseline
        
        # Verify files were created
        assert output_dir.exists()
        assert (output_dir / 'baseline').exists()
        assert (output_dir / 'iter_1').exists()
        assert (output_dir / 'iter_2').exists()
        
        # Check iteration summaries
        for i, summary in enumerate(report.iteration_history, 1):
            assert summary.iteration == i
            assert summary.num_samples == 2
            assert summary.execution_time > 0
    
    def test_workflow_with_failures(self, temp_workspace, integration_config):
        """Test workflow handling with partial failures."""
        
        with patch('quadro_llm.core.llm_interface.openai.OpenAI') as mock_openai, \
             patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'):
            
            # Mock LLM that sometimes fails
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # First call succeeds, second fails
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = """
def get_reward(self):
    return torch.zeros(self.num_agent)
            """
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.side_effect = [
                mock_response,  # First iteration succeeds
                Exception("API Error")  # Second iteration fails
            ]
            
            # Mock training that sometimes fails
            with patch('quadro_llm.training.visfly_training_wrapper.VisFlyTrainingWrapper') as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper.inject_reward_function.return_value = True
                mock_wrapper.train.side_effect = [
                    # Baseline and first sample succeed
                    {'success_rate': 0.3, 'episode_length': 40.0, 'training_time': 30.0},
                    {'success_rate': 0.5, 'episode_length': 35.0, 'training_time': 25.0},
                    # Second sample fails
                    Exception("Training crashed")
                ]
                mock_wrapper_class.return_value = mock_wrapper
                
                integration_config['config_dir'] = str(temp_workspace / 'configs')
                
                pipeline = Pipeline(
                    config=integration_config,
                    output_dir=str(temp_workspace / 'output')
                )
                
                # Should handle failures gracefully
                report = pipeline.run_optimization()
        
        # Should still produce a report with partial results
        assert report is not None
        assert len(report.iteration_history) >= 1  # At least one iteration completed
        
        # Some samples may have failed
        total_attempted = sum(s.num_samples for s in report.iteration_history)
        assert report.successful_samples <= total_attempted
    
    def test_configuration_validation_integration(self, temp_workspace):
        """Test configuration validation in full workflow."""
        
        # Invalid configuration (missing required fields)
        invalid_config = {
            'environment': 'nonexistent',
            'algorithm': 'invalid',
            'iterations': 1
            # Missing required fields
        }
        
        with pytest.raises((ValueError, FileNotFoundError)):
            Pipeline(
                config=invalid_config,
                output_dir=str(temp_workspace / 'output')
            )
    
    def test_progress_monitoring_integration(self, temp_workspace, integration_config):
        """Test progress monitoring throughout workflow."""
        
        progress_updates = []
        
        def mock_progress_callback(stage, progress, details=None):
            progress_updates.append({
                'stage': stage,
                'progress': progress,
                'details': details
            })
        
        with patch('quadro_llm.core.llm_interface.openai.OpenAI') as mock_openai, \
             patch('quadro_llm.training.visfly_training_wrapper.importlib.import_module'), \
             patch('quadro_llm.training.visfly_training_wrapper.VisFlyTrainingWrapper') as mock_wrapper_class:
            
            # Mock successful responses
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "def get_reward(self): return torch.zeros(4)"
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            mock_wrapper = Mock()
            mock_wrapper.inject_reward_function.return_value = True
            mock_wrapper.train.return_value = {
                'success_rate': 0.6,
                'episode_length': 35.0,
                'training_time': 20.0
            }
            mock_wrapper_class.return_value = mock_wrapper
            
            integration_config['config_dir'] = str(temp_workspace / 'configs')
            integration_config['progress_callback'] = mock_progress_callback
            
            pipeline = Pipeline(
                config=integration_config,
                output_dir=str(temp_workspace / 'output')
            )
            
            report = pipeline.run_optimization()
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Should have updates for major stages
        stages = [update['stage'] for update in progress_updates]
        expected_stages = ['baseline', 'iteration', 'evaluation']
        
        for stage in expected_stages:
            assert any(stage in s for s in stages)


@pytest.mark.integration
@pytest.mark.slow
class TestRealEnvironmentIntegration:
    """Test integration with actual VisFly environments (if available)."""
    
    def test_navigation_environment_integration(self):
        """Test integration with NavigationEnv (requires VisFly)."""
        pytest.skip("Requires VisFly installation and dependencies")
        
        # This test would verify:
        # 1. Actual environment can be instantiated
        # 2. Reward functions can be injected
        # 3. Training actually runs
        # 4. Evaluation produces sensible results
    
    def test_multi_algorithm_compatibility(self):
        """Test that all algorithms work with all environments."""
        pytest.skip("Requires full environment setup")
        
        # This test would verify:
        # 1. BPTT, PPO, and SHAC all work
        # 2. Different environments are compatible
        # 3. Configuration loading works correctly