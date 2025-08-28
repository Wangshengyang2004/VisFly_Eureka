"""
Unit tests for the quadro-llm pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quadro_llm.pipeline import Pipeline
from quadro_llm.core.models import RewardFunctionResult


class TestPipeline:
    """Test the main optimization pipeline."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return {
            'environment': 'navigation',
            'algorithm': 'bptt',
            'iterations': 2,
            'samples': 4,
            'training_steps': 1000,
            'evaluation_episodes': 10,
            'success_threshold': 0.8,
            'llm': {
                'api_key': 'test-key',
                'model': 'gpt-4o',
                'temperature': 0.8,
                'max_tokens': 2000
            },
            'task_description': 'Navigate to target avoiding obstacles'
        }
    
    @pytest.fixture
    def mock_pipeline(self, pipeline_config, temp_output_dir):
        """Create pipeline with mocked dependencies."""
        with patch('quadro_llm.pipeline.LLMInterface'), \
             patch('quadro_llm.pipeline.ConfigManager'), \
             patch('quadro_llm.pipeline.TrainingManager'), \
             patch('quadro_llm.pipeline.RewardInjector'):
            
            pipeline = Pipeline(
                config=pipeline_config,
                output_dir=str(temp_output_dir)
            )
            
            # Mock the components
            pipeline.llm_interface = Mock()
            pipeline.config_manager = Mock()
            pipeline.training_manager = Mock()
            pipeline.reward_injector = Mock()
            
            return pipeline
    
    def test_pipeline_initialization(self, pipeline_config, temp_output_dir):
        """Test pipeline initialization."""
        with patch('quadro_llm.pipeline.LLMInterface'), \
             patch('quadro_llm.pipeline.ConfigManager'), \
             patch('quadro_llm.pipeline.TrainingManager'), \
             patch('quadro_llm.pipeline.RewardInjector'):
            
            pipeline = Pipeline(
                config=pipeline_config,
                output_dir=str(temp_output_dir)
            )
            
            assert pipeline.config == pipeline_config
            assert pipeline.output_dir == temp_output_dir
            assert pipeline.current_iteration == 0
    
    def test_run_baseline(self, mock_pipeline):
        """Test baseline evaluation."""
        # Mock training manager baseline
        mock_pipeline.training_manager.train_baseline.return_value = {
            'success_rate': 0.5,
            'episode_length': 150.0,
            'training_time': 120.0
        }
        
        baseline = mock_pipeline._run_baseline()
        
        assert baseline is not None
        assert baseline['success_rate'] == 0.5
        mock_pipeline.training_manager.train_baseline.assert_called_once()
    
    def test_generate_reward_functions(self, mock_pipeline):
        """Test reward function generation."""
        # Mock LLM interface
        sample_rewards = [
            "def get_reward(self): return -torch.norm(self.position - self.target, dim=1)",
            "def get_reward(self): return -torch.norm(self.velocity - 0, dim=1) * 0.1"
        ]
        mock_pipeline.llm_interface.generate_reward_functions.return_value = sample_rewards
        
        rewards = mock_pipeline._generate_reward_functions(iteration=1)
        
        assert len(rewards) == 2
        assert all('def get_reward(self)' in r for r in rewards)
        mock_pipeline.llm_interface.generate_reward_functions.assert_called_once()
    
    def test_evaluate_reward_function(self, mock_pipeline):
        """Test single reward function evaluation."""
        reward_code = "def get_reward(self): return torch.zeros(self.num_agent)"
        
        # Mock training result
        mock_pipeline.training_manager.train_with_reward.return_value = {
            'success_rate': 0.8,
            'episode_length': 100.0,
            'training_time': 60.0
        }
        
        result = mock_pipeline._evaluate_reward_function(
            reward_code, iteration=1, sample_id=0
        )
        
        assert isinstance(result, RewardFunctionResult)
        assert result.success_rate == 0.8
        assert result.iteration == 1
        assert result.sample_id == 0
    
    def test_evaluate_reward_function_error(self, mock_pipeline):
        """Test reward function evaluation with error."""
        reward_code = "def get_reward(self): return invalid_code"
        
        # Mock training error
        mock_pipeline.training_manager.train_with_reward.side_effect = Exception("Training failed")
        
        result = mock_pipeline._evaluate_reward_function(
            reward_code, iteration=1, sample_id=0
        )
        
        assert isinstance(result, RewardFunctionResult)
        assert result.error is not None
        assert not result.is_successful
    
    def test_run_iteration(self, mock_pipeline):
        """Test running a single iteration."""
        # Mock reward generation
        sample_rewards = ["def get_reward(self): return torch.zeros(4)"] * 4
        mock_pipeline.llm_interface.generate_reward_functions.return_value = sample_rewards
        
        # Mock evaluation results
        mock_results = []
        for i in range(4):
            result = RewardFunctionResult(
                iteration=1,
                sample_id=i,
                code=sample_rewards[i],
                success_rate=0.7 + i * 0.1,
                episode_length=100.0,
                training_time=60.0
            )
            mock_results.append(result)
        
        with patch.object(mock_pipeline, '_evaluate_reward_function', side_effect=mock_results):
            summary = mock_pipeline._run_iteration(1)
        
        assert summary.iteration == 1
        assert summary.num_samples == 4
        assert summary.successful_samples == 4
        assert summary.best_success_rate == 1.0  # 0.7 + 3 * 0.1
    
    def test_full_optimization(self, mock_pipeline):
        """Test full optimization run."""
        # Mock baseline
        mock_pipeline.training_manager.train_baseline.return_value = {
            'success_rate': 0.3,
            'episode_length': 200.0,
            'training_time': 120.0
        }
        
        # Mock iterations
        with patch.object(mock_pipeline, '_run_iteration') as mock_run_iter:
            from quadro_llm.core.models import IterationSummary
            
            # Mock iteration summaries
            mock_summaries = []
            for i in range(1, 3):  # 2 iterations
                summary = IterationSummary(
                    iteration=i,
                    num_samples=4,
                    successful_samples=3,
                    best_success_rate=0.5 + i * 0.2,
                    best_sample_id=1,
                    execution_time=120.0
                )
                mock_summaries.append(summary)
            
            mock_run_iter.side_effect = mock_summaries
            
            report = mock_pipeline.run_optimization()
        
        assert report is not None
        assert report.total_samples == 8  # 2 iterations * 4 samples
        assert len(report.iteration_history) == 2
        assert mock_run_iter.call_count == 2
    
    def test_save_results(self, mock_pipeline, temp_output_dir):
        """Test saving optimization results."""
        # Create mock results
        result = RewardFunctionResult(
            iteration=1,
            sample_id=0,
            code="def get_reward(self): return torch.zeros(4)",
            success_rate=0.8,
            episode_length=100.0,
            training_time=60.0
        )
        
        mock_pipeline._save_reward_result(result)
        
        # Check files were created
        iteration_dir = temp_output_dir / 'iter_1'
        assert iteration_dir.exists()
        
        reward_file = iteration_dir / 'sample_0_reward.py'
        result_file = iteration_dir / 'sample_0_result.json'
        
        assert reward_file.exists()
        assert result_file.exists()
    
    def test_progress_tracking(self, mock_pipeline):
        """Test optimization progress tracking."""
        # Mock baseline and iterations
        baseline_perf = {'success_rate': 0.3}
        
        from quadro_llm.core.models import IterationSummary
        iteration_summaries = [
            IterationSummary(1, 4, 3, 0.6, 1, 120.0),
            IterationSummary(2, 4, 4, 0.8, 2, 120.0)
        ]
        
        with patch.object(mock_pipeline, '_run_baseline', return_value=baseline_perf), \
             patch.object(mock_pipeline, '_run_iteration', side_effect=iteration_summaries):
            
            report = mock_pipeline.run_optimization()
        
        # Check improvement tracking
        assert report.improvement_metrics is not None
        
        # Should show improvement from baseline
        best_performance = report.best_performance
        assert best_performance['success_rate'] > baseline_perf['success_rate']


class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery."""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            'environment': 'navigation',
            'algorithm': 'bptt',
            'iterations': 1,
            'samples': 2,
            'llm': {'api_key': 'test'},
            'task_description': 'Test task'
        }
    
    def test_llm_api_failure(self, pipeline_config):
        """Test handling of LLM API failures."""
        with patch('quadro_llm.pipeline.LLMInterface') as mock_llm_class, \
             patch('quadro_llm.pipeline.ConfigManager'), \
             patch('quadro_llm.pipeline.TrainingManager'), \
             patch('quadro_llm.pipeline.RewardInjector'):
            
            # Mock LLM to raise exception
            mock_llm = Mock()
            mock_llm.generate_reward_functions.side_effect = Exception("API Error")
            mock_llm_class.return_value = mock_llm
            
            pipeline = Pipeline(config=pipeline_config, output_dir='/tmp')
            
            # Should handle gracefully and continue with empty rewards
            rewards = pipeline._generate_reward_functions(1)
            assert rewards == []
    
    def test_training_failure_recovery(self, pipeline_config):
        """Test recovery from training failures."""
        with patch('quadro_llm.pipeline.LLMInterface'), \
             patch('quadro_llm.pipeline.ConfigManager'), \
             patch('quadro_llm.pipeline.TrainingManager') as mock_training_class, \
             patch('quadro_llm.pipeline.RewardInjector'):
            
            # Mock training to fail sometimes
            mock_training = Mock()
            mock_training.train_with_reward.side_effect = [
                Exception("Training failed"),  # First call fails
                {'success_rate': 0.8, 'episode_length': 100.0, 'training_time': 60.0}  # Second succeeds
            ]
            mock_training_class.return_value = mock_training
            
            pipeline = Pipeline(config=pipeline_config, output_dir='/tmp')
            pipeline.training_manager = mock_training
            
            # First evaluation should handle error
            result1 = pipeline._evaluate_reward_function("code1", 1, 0)
            assert not result1.is_successful
            assert result1.error is not None
            
            # Second should succeed
            result2 = pipeline._evaluate_reward_function("code2", 1, 1)
            assert result2.is_successful
            assert result2.error is None
    
    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        invalid_config = {
            'environment': 'nonexistent',
            'algorithm': 'invalid_alg'
            # Missing required fields
        }
        
        with patch('quadro_llm.pipeline.ConfigManager') as mock_config_class:
            mock_config = Mock()
            mock_config.get_merged_config.return_value = None  # Config not found
            mock_config_class.return_value = mock_config
            
            with pytest.raises(ValueError, match="Invalid configuration"):
                Pipeline(config=invalid_config, output_dir='/tmp')