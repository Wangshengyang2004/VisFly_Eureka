"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from quadro_llm.core.models import (
    RewardFunctionResult,
    IterationSummary,
    OptimizationReport
)


class TestRewardFunctionResult:
    """Test RewardFunctionResult model."""
    
    def test_initialization(self):
        """Test basic initialization."""
        result = RewardFunctionResult(
            reward_code="def get_reward(self): return torch.zeros(4)",
            identifier="test_0",
            training_successful=True,
            success_rate=0.8,
            episode_length=100.0,
            training_time=60.0,
            final_reward=10.0,
            convergence_step=500
        )
        
        assert result.identifier == "test_0"
        assert result.success_rate == 0.8
        assert result.training_successful
    
    def test_is_successful_property(self):
        """Test training_successful property."""
        # Successful case
        result = RewardFunctionResult(
            reward_code="def get_reward(self): return torch.zeros(4)",
            identifier="test_1",
            training_successful=True,
            success_rate=0.5,
            episode_length=100.0,
            training_time=60.0,
            final_reward=5.0,
            convergence_step=300
        )
        assert result.training_successful
        
        # Failed case
        result_with_error = RewardFunctionResult.failed(
            reward_code="invalid code",
            identifier="test_failed",
            error="Syntax error"
        )
        assert not result_with_error.training_successful
    
    def test_score_calculation(self):
        """Test score calculation."""
        # High success rate, short episodes (good)
        result1 = RewardFunctionResult(
            reward_code="def get_reward(self): return torch.zeros(4)",
            identifier="test_good",
            training_successful=True,
            success_rate=0.9,
            episode_length=50.0,
            training_time=60.0,
            final_reward=15.0,
            convergence_step=400
        )
        
        # Low success rate, long episodes (bad)
        result2 = RewardFunctionResult(
            reward_code="def get_reward(self): return torch.zeros(4)",
            identifier="test_bad",
            training_successful=True,
            success_rate=0.3,
            episode_length=200.0,
            training_time=60.0,
            final_reward=3.0,
            convergence_step=800
        )
        
        assert result1.score() > result2.score()
        
        # Failed function should have very negative score
        result3 = RewardFunctionResult.failed(
            reward_code="invalid code",
            identifier="test_failed",
            error="Failed"
        )
        assert result3.score() == -10000.0


class TestIterationSummary:
    """Test IterationSummary model."""
    
    def test_initialization(self):
        """Test initialization."""
        # Create sample results
        sample_results = [
            RewardFunctionResult(
                reward_code="def get_reward(self): return torch.zeros(4)",
                identifier="sample_0",
                training_successful=True,
                success_rate=0.8,
                episode_length=100.0,
                training_time=60.0,
                final_reward=8.0,
                convergence_step=400
            )
        ]
        
        summary = IterationSummary(
            iteration=1,
            samples=sample_results,
            best_sample_idx=0,
            best_success_rate=0.8,
            best_correlation=0.9,
            execution_rate=1.0,
            generation_time=30.0,
            total_training_time=120.0
        )
        
        assert summary.iteration == 1
        assert len(summary.samples) == 1
        assert summary.best_success_rate == 0.8
    
    def test_execution_rate(self):
        """Test execution rate tracking."""
        sample_results = []
        
        summary = IterationSummary(
            iteration=1,
            samples=sample_results,
            best_sample_idx=0,
            best_success_rate=0.9,
            best_correlation=0.8,
            execution_rate=0.7,
            generation_time=45.0,
            total_training_time=100.0
        )
        
        assert summary.execution_rate == 0.7


class TestOptimizationReport:
    """Test OptimizationReport model."""
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        iteration_history = [
            IterationSummary(
                iteration=1,
                samples=[],
                best_sample_idx=0,
                best_success_rate=0.8,
                best_correlation=0.9,
                execution_rate=0.8,
                generation_time=30.0,
                total_training_time=60.0
            )
        ]
        
        report = OptimizationReport(
            total_samples=5,
            successful_samples=4,
            best_performance={'success_rate': 0.8, 'episode_length': 100},
            improvement_metrics={'relative_improvement': 0.5},
            execution_time=60.0,
            output_directory="./output",
            iteration_history=iteration_history
        )
        
        report_dict = report.to_dict()
        
        assert report_dict['total_samples'] == 5
        assert report_dict['successful_samples'] == 4
        assert 'iteration_history' in report_dict
        assert len(report_dict['iteration_history']) == 1