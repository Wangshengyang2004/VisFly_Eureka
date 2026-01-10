"""
Reward function evaluator.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
import torch


class RewardEvaluator:
    """Evaluates and ranks reward functions based on training performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_performance(
        self,
        training_results: Dict[str, Any],
        evaluation_episodes: int = 20
    ) -> Dict[str, float]:
        """
        Evaluate performance metrics from training results.
        
        Args:
            training_results: Results from training
            evaluation_episodes: Number of episodes to evaluate
            
        Returns:
            Performance metrics dictionary
        """
        metrics = {}
        
        # Extract success rate
        if 'success_rates' in training_results and training_results['success_rates']:
            success_rates = training_results['success_rates']
            metrics['success_rate'] = np.mean(success_rates[-evaluation_episodes:])
            metrics['max_success_rate'] = np.max(success_rates)
            metrics['success_stability'] = 1.0 - np.std(success_rates[-evaluation_episodes:])
        else:
            metrics['success_rate'] = 0.0
            metrics['max_success_rate'] = 0.0
            metrics['success_stability'] = 0.0
        
        # Extract episode metrics
        if 'episode_lengths' in training_results and training_results['episode_lengths']:
            lengths = training_results['episode_lengths']
            metrics['avg_episode_length'] = np.mean(lengths[-evaluation_episodes:])
            metrics['min_episode_length'] = np.min(lengths[-evaluation_episodes:])
        else:
            metrics['avg_episode_length'] = float('inf')
            metrics['min_episode_length'] = float('inf')
        
        # Extract reward metrics
        if 'episode_rewards' in training_results and training_results['episode_rewards']:
            rewards = training_results['episode_rewards']
            metrics['avg_reward'] = np.mean(rewards[-evaluation_episodes:])
            metrics['max_reward'] = np.max(rewards)
            metrics['reward_improvement'] = self._calculate_improvement(rewards)
        else:
            metrics['avg_reward'] = -float('inf')
            metrics['max_reward'] = -float('inf')
            metrics['reward_improvement'] = 0.0
        
        # Training efficiency
        if 'convergence_step' in training_results:
            metrics['convergence_step'] = training_results['convergence_step'] or float('inf')
        else:
            metrics['convergence_step'] = float('inf')
        
        # Overall score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def rank_reward_functions(
        self,
        results: List[Dict[str, Any]],
        ranking_metric: str = 'overall_score'
    ) -> List[Dict[str, Any]]:
        """
        Rank reward functions based on performance.
        
        Args:
            results: List of result dictionaries
            ranking_metric: Metric to use for ranking
            
        Returns:
            Sorted list of results
        """
        # Filter out failed results
        valid_results = [r for r in results if 'error' not in r or r['error'] is None]
        
        if not valid_results:
            self.logger.warning("No valid results to rank")
            return results
        
        # Sort by ranking metric
        sorted_results = sorted(
            valid_results,
            key=lambda x: x.get('metrics', {}).get(ranking_metric, -float('inf')),
            reverse=True
        )
        
        return sorted_results
    
    def compare_to_baseline(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare current performance to baseline.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        # Success rate improvement
        if 'success_rate' in current_metrics and 'success_rate' in baseline_metrics:
            baseline_sr = baseline_metrics['success_rate']
            current_sr = current_metrics['success_rate']
            
            comparison['success_rate_improvement'] = current_sr - baseline_sr
            comparison['relative_improvement'] = (
                (current_sr - baseline_sr) / max(baseline_sr, 0.01)
            )
        
        # Episode length improvement (shorter is better)
        if 'avg_episode_length' in current_metrics and 'avg_episode_length' in baseline_metrics:
            baseline_len = baseline_metrics['avg_episode_length']
            current_len = current_metrics['avg_episode_length']
            
            comparison['episode_length_reduction'] = baseline_len - current_len
            comparison['relative_efficiency'] = (
                (baseline_len - current_len) / max(baseline_len, 1)
            )
        
        # Reward improvement
        if 'avg_reward' in current_metrics and 'avg_reward' in baseline_metrics:
            baseline_reward = baseline_metrics['avg_reward']
            current_reward = current_metrics['avg_reward']
            
            comparison['reward_improvement'] = current_reward - baseline_reward
            comparison['relative_reward_gain'] = (
                (current_reward - baseline_reward) / max(abs(baseline_reward), 0.01)
            )
        
        return comparison
    
    def _calculate_improvement(self, values: List[float], window: int = 20) -> float:
        """
        Calculate improvement rate over time.
        
        Args:
            values: Time series of values
            window: Window size for comparison
            
        Returns:
            Improvement rate
        """
        if len(values) < 2 * window:
            return 0.0
        
        early_avg = np.mean(values[:window])
        late_avg = np.mean(values[-window:])
        
        if abs(early_avg) < 1e-6:
            return 0.0
        
        return (late_avg - early_avg) / abs(early_avg)
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from multiple metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Overall score
        """
        # Weighted combination of metrics
        score = 0.0
        
        # Success rate is most important (40%)
        score += metrics.get('success_rate', 0.0) * 0.4
        
        # Episode efficiency (20%)
        max_steps = 256  # Typical max episode length
        efficiency = 1.0 - min(metrics.get('avg_episode_length', max_steps) / max_steps, 1.0)
        score += efficiency * 0.2
        
        # Stability (20%)
        score += metrics.get('success_stability', 0.0) * 0.2
        
        # Convergence speed (10%)
        max_convergence = 10000
        convergence_score = 1.0 - min(
            metrics.get('convergence_step', max_convergence) / max_convergence, 1.0
        )
        score += convergence_score * 0.1
        
        # Reward improvement (10%)
        improvement = min(max(metrics.get('reward_improvement', 0.0), -1.0), 1.0)
        score += (improvement + 1.0) / 2.0 * 0.1
        
        return score